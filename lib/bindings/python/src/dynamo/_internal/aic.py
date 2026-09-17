# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared AIC-core session helpers used by internal Dynamo integrations."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import sys
import threading

logger = logging.getLogger(__name__)

_NEXTN_ACCEPT_RATES_LEN = 5
# Dynamo's historical default when conditional acceptance rates are omitted.
_DEFAULT_NEXTN_ACCEPT_RATES = [0.85, 0.3, 0.0, 0.0, 0.0]

# Default backend versions match the AIC-core v0.11.0 perf DB.
DEFAULT_BACKEND_VERSIONS = {
    "vllm": "0.19.0",
    "sglang": "0.5.10",
    "trtllm": "1.3.0rc10",
}
_KV_CAPACITY_BACKENDS = frozenset(DEFAULT_BACKEND_VERSIONS)
DEFAULT_STATIC_STRIDE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MEM_FRACTION_STATIC = 0.88
DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9

# Perf-data provenance. SILICON reads collected grids off disk (NVIDIA parts);
# ANALYTICAL computes every kernel at query time from a vendor analytical model
# (Intel Xe), so there is nothing on disk to read.
DATABASE_MODE_SILICON = "SILICON"
DATABASE_MODE_ANALYTICAL = "ANALYTICAL"
# The mode the SDK itself runs under while the analytical backend is patched in:
# kernels the Xe backend does not implement must fall back to roofline formulas
# derived from the system YAML, never to a CSV lookup that does not exist.
_ANALYTICAL_SDK_MODE = "EMPIRICAL"
_SUPPORTED_DATABASE_MODES = (DATABASE_MODE_SILICON, DATABASE_MODE_ANALYTICAL)

# Optional shared analytical-result caches. They are deliberately opt-in
# because the package and database live outside Dynamo's normal installation.
# The legacy combined variable remains a fallback for both layers.
_KAPA_CACHE_MODE_ENV = "DYNAMO_AIC_KAPA_CACHE"
_KAPA_LOOKUP_CACHE_MODE_ENV = "DYNAMO_AIC_KAPA_LOOKUP_CACHE"
_KAPA_RUNTIME_CACHE_MODE_ENV = "DYNAMO_AIC_KAPA_RUNTIME_CACHE"
_KAPA_CONFIG_ENV = "KAPA_DATA_CONFIG"
_KAPA_VERSION_ENV = "DYNAMO_AIC_KAPA_VERSION"


def resolve_database_mode(database_mode: str | None) -> str:
    """Normalize the perf-data mode; ``None`` means the historical SILICON path."""
    if database_mode is None:
        return DATABASE_MODE_SILICON
    normalized = database_mode.strip().upper()
    if not normalized:
        return DATABASE_MODE_SILICON
    if normalized not in _SUPPORTED_DATABASE_MODES:
        supported = ", ".join(_SUPPORTED_DATABASE_MODES)
        raise ValueError(
            f"unsupported aic_database_mode {database_mode!r}; supported: {supported}"
        )
    return normalized


def _activate_analytical_backend(
    system: str,
    xe_compute_config: str | None,
    *,
    backend_name: str | None = None,
) -> None:
    """Install the vendor analytical perf backend for the process lifetime.

    The backend replaces ``PerfDatabase.query_*`` at the class level and must
    stay installed: predictions happen long after this call returns.
    """
    if not xe_compute_config:
        raise ValueError(
            f"aic_database_mode={DATABASE_MODE_ANALYTICAL} requires aic_xe_compute_config "
            "(e.g. 'xe5_96') so the analytical model knows which device to model"
        )
    try:
        from intel_xe import analytical_session
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"aic_database_mode={DATABASE_MODE_ANALYTICAL} needs the 'intel_xe' package "
            "from the Intel aiconfigurator distribution, which is not installed"
        ) from exc

    analytical_session.activate(
        system,
        xe_compute_config,
        kapa_cache_mode=os.environ.get(_KAPA_CACHE_MODE_ENV),
        kapa_lookup_cache_mode=os.environ.get(_KAPA_LOOKUP_CACHE_MODE_ENV),
        kapa_runtime_cache_mode=os.environ.get(_KAPA_RUNTIME_CACHE_MODE_ENV),
        kapa_config_path=os.environ.get(_KAPA_CONFIG_ENV),
        kapa_backend=backend_name,
        kapa_version=os.environ.get(_KAPA_VERSION_ENV),
    )
    unsupported = analytical_session.unsupported_kernels()
    if unsupported:
        # These still answer, but with the NVIDIA model — silently wrong numbers.
        logger.warning(
            "AIC analytical backend does not implement %s; those kernels keep the "
            "vendor-default cost model and will not reflect %s",
            ", ".join(sorted(unsupported)),
            system,
        )


def get_kapa_cache_stats() -> dict[str, dict[str, object]]:
    """Return KAPA counters when the optional analytical integration is active."""
    try:
        from intel_xe import analytical_session
    except ImportError:
        return {}
    return analytical_session.kapa_cache_stats()


def get_xe_perf_cache_stats() -> list[dict[str, object]]:
    """Return counters for the optional persistent Xe JSONL cache."""
    try:
        from intel_xe import xe_perf_cache
    except ImportError:
        return []
    return xe_perf_cache.all_stats()


# --- AIC per-operator latency profiling (opt-in) ----------------------------
# Set DYNAMO_AIC_PROFILE_OPS=1 to accumulate each operator's contribution to
# predicted latency across every predict_prefill/predict_decode call made
# during one dynosim run, then merge a breakdown into the --report-json file
# (under the "aic_operator_breakdown" key) when the process exits.
_OP_PROFILE_ENABLED = bool(os.environ.get("DYNAMO_AIC_PROFILE_OPS"))
_op_profile_lock = threading.Lock()
_op_profile_stats: dict[tuple[str, str], dict[str, float]] = {}


def _record_op_latency(
    phase: str, op_name: str, latency_ms: float, weight: int = 1
) -> None:
    """Accumulate one operator query's contribution to total predicted latency.

    ``weight`` lets a single query stand in for multiple repeated decode
    steps (the generation loop reuses one query result for ``repeat_count``
    consecutive output tokens).
    """
    with _op_profile_lock:
        entry = _op_profile_stats.setdefault(
            (phase, op_name or "<unnamed>"), {"total_ms": 0.0, "calls": 0}
        )
        entry["total_ms"] += latency_ms * weight
        entry["calls"] += weight


def _op_profile_report_path() -> str | None:
    """Return the --report-json path passed on this process's command line, if any."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--report-json" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--report-json="):
            return arg.split("=", 1)[1]
    return None


def _build_op_breakdown() -> dict:
    with _op_profile_lock:
        snapshot = dict(_op_profile_stats)

    grand_total_ms = sum(v["total_ms"] for v in snapshot.values())
    phases: dict[str, dict] = {}
    merged: dict[str, dict[str, float]] = {}

    for (phase, op_name), stats in snapshot.items():
        phase_bucket = phases.setdefault(phase, {"total_ms": 0.0, "ops": []})
        phase_bucket["total_ms"] += stats["total_ms"]
        merged_entry = merged.setdefault(op_name, {"total_ms": 0.0, "calls": 0})
        merged_entry["total_ms"] += stats["total_ms"]
        merged_entry["calls"] += stats["calls"]

    for phase, bucket in phases.items():
        ops = []
        for (p, op_name), stats in snapshot.items():
            if p != phase:
                continue
            calls = stats["calls"]
            total_ms = stats["total_ms"]
            ops.append(
                {
                    "op_name": op_name,
                    "total_ms": total_ms,
                    "calls": calls,
                    "avg_ms_per_call": total_ms / calls if calls else 0.0,
                    "pct_of_phase": (
                        100.0 * total_ms / bucket["total_ms"]
                        if bucket["total_ms"]
                        else 0.0
                    ),
                    "pct_of_grand_total": (
                        100.0 * total_ms / grand_total_ms if grand_total_ms else 0.0
                    ),
                }
            )
        ops.sort(key=lambda o: o["total_ms"], reverse=True)
        bucket["ops"] = ops

    op_summary_across_phases = [
        {
            "op_name": op_name,
            "total_ms": stats["total_ms"],
            "calls": stats["calls"],
            "pct_of_grand_total": (
                100.0 * stats["total_ms"] / grand_total_ms if grand_total_ms else 0.0
            ),
        }
        for op_name, stats in merged.items()
    ]
    op_summary_across_phases.sort(key=lambda o: o["total_ms"], reverse=True)

    return {
        "enabled_via": "DYNAMO_AIC_PROFILE_OPS",
        "note": (
            "Per-operator breakdown of AIC-predicted forward-pass latency, "
            "accumulated across every predict_prefill/predict_decode call made "
            "during this dynosim run. Collected via the Python op-walk path "
            "(the compiled AIC engine is bypassed while profiling is enabled "
            "so every operator query is individually timed)."
        ),
        "grand_total_ms": grand_total_ms,
        "phases": phases,
        "op_summary_across_phases": op_summary_across_phases,
    }


def _maybe_write_op_profile_report() -> None:
    if not _OP_PROFILE_ENABLED:
        return
    with _op_profile_lock:
        has_data = bool(_op_profile_stats)
    if not has_data:
        logger.warning(
            "DYNAMO_AIC_PROFILE_OPS was set but no AIC operator queries were "
            "recorded; nothing to write."
        )
        return

    breakdown = _build_op_breakdown()
    report_path = _op_profile_report_path()

    if report_path and os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not read --report-json at %s to merge the AIC op "
                "breakdown (%s); writing a standalone file instead.",
                report_path,
                exc,
            )
        else:
            report["aic_operator_breakdown"] = breakdown
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(
                "AIC operator breakdown merged into --report-json at %s "
                "(key: aic_operator_breakdown).",
                report_path,
            )
            return

    fallback_path = os.path.abspath("aic_operator_breakdown.json")
    with open(fallback_path, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2)
    logger.warning(
        "No usable --report-json path found on the command line; wrote a "
        "standalone AIC operator breakdown to %s instead.",
        fallback_path,
    )


atexit.register(_maybe_write_op_profile_report)


def _validate_kv_capacity_backend(backend_name: str) -> None:
    if backend_name not in _KV_CAPACITY_BACKENDS:
        supported = ", ".join(sorted(_KV_CAPACITY_BACKENDS))
        raise ValueError(
            "AIC KV cache capacity estimation does not support "
            f"backend {backend_name!r}; supported backends: {supported}. "
            "Set num_gpu_blocks explicitly for this backend."
        )


def resolve_backend_version(backend_name: str, backend_version: str | None) -> str:
    """Return the pinned backend version used for AIC perf lookups."""
    if backend_version is not None:
        return backend_version
    return DEFAULT_BACKEND_VERSIONS.get(backend_name, DEFAULT_BACKEND_VERSIONS["vllm"])


def _normalize_aic_quant_mode(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None
    if value == "int4":
        return "int4_wo"
    return value


def _resolve_quant_mode(field: str, value: str | None):
    """Resolve a dtype-override string to aiconfigurator's per-field quant-mode
    enum, or ``None`` to use the model default.

    The four quant fields accept *different* value sets (e.g. KV cache only
    supports ``bfloat16``/``int8``/``fp8``), so the string -> enum lookup is per
    field. On an unsupported value, raise a clear ``ValueError`` naming the
    field and its allowed values instead of letting an opaque ``KeyError``
    escape from deep inside aiconfigurator. ``field`` is one of ``gemm``,
    ``moe``, ``fmha``, ``kvcache``, ``comm``.
    """
    normalized = _normalize_aic_quant_mode(value)
    if normalized is None:
        return None
    from aiconfigurator.sdk import common

    enum_cls = {
        "gemm": common.GEMMQuantMode,
        "moe": common.MoEQuantMode,
        "fmha": common.FMHAQuantMode,
        "kvcache": common.KVCacheQuantMode,
        "comm": common.CommQuantMode,
    }[field]
    try:
        return enum_cls[normalized]
    except KeyError:
        allowed = ", ".join(member.name for member in enum_cls)
        raise ValueError(
            f"unsupported AIC {field} quant mode {value!r} "
            f"(normalized to {normalized!r}); supported values: {allowed}"
        ) from None


def _resolve_quant_mode_name(field: str, value: str | None) -> str | None:
    """Like :func:`_resolve_quant_mode` but return the canonical quant-mode
    *name* (the string aiconfigurator's string-keyed APIs expect), validated
    against the field's enum. ``None`` means "use the model default"."""
    mode = _resolve_quant_mode(field, value)
    return mode.name if mode is not None else None


def _pad_nextn_accept_rates(
    nextn_accept_rates: list[float] | str | None,
) -> list[float]:
    """Normalize accept-rates for the released ``aiconfigurator`` wheel.

    The upper AIC wheel still accepts the fixed length-5 conditional-rate
    contract. When rates are omitted entirely we preserve its CLI default;
    shorter lists are zero-padded and longer lists are truncated.
    """
    if isinstance(nextn_accept_rates, str):
        try:
            nextn_accept_rates = [
                float(x) for x in nextn_accept_rates.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "aic_nextn_accept_rates must be comma-separated floats, got "
                f"{nextn_accept_rates!r}"
            ) from exc
    if not nextn_accept_rates:
        return list(_DEFAULT_NEXTN_ACCEPT_RATES)
    rates = list(nextn_accept_rates)
    # Rates are acceptance probabilities; out-of-range or non-finite values
    # would silently skew calc_expectation rather than surface a config error.
    if any(not math.isfinite(r) or not 0.0 <= r <= 1.0 for r in rates):
        raise ValueError(
            f"aic_nextn_accept_rates must be finite floats in [0, 1], got {rates}"
        )
    if len(rates) < _NEXTN_ACCEPT_RATES_LEN:
        rates = rates + [0.0] * (_NEXTN_ACCEPT_RATES_LEN - len(rates))
    elif len(rates) > _NEXTN_ACCEPT_RATES_LEN:
        rates = rates[:_NEXTN_ACCEPT_RATES_LEN]
    return rates


def _load_aiconfigurator():
    try:
        from aiconfigurator.sdk import common, config
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.models import get_model
        from aiconfigurator.sdk.perf_database import (
            get_database,
            get_supported_databases,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "aiconfigurator":
            raise
        raise RuntimeError(
            "aiconfigurator-core is required for AIC perf modeling but is not installed"
        ) from exc

    return {
        "common": common,
        "config": config,
        "get_backend": get_backend,
        "get_model": get_model,
        "get_database": get_database,
        "get_supported_databases": get_supported_databases,
    }


class AicSession:
    """Wrap AIC-core model objects with direct prefill/decode predictors."""

    def __init__(
        self,
        backend_name: str,
        system: str,
        model_path: str,
        tp_size: int,
        backend_version: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        attention_dp_size: int | None = None,
        gemm_dtype: str | None = None,
        moe_dtype: str | None = None,
        fmha_dtype: str | None = None,
        kv_cache_dtype: str | None = None,
        comm_dtype: str | None = None,
        nextn: int | None = None,
        nextn_accept_rates: list[float] | str | None = None,
        database_mode: str | None = None,
        xe_compute_config: str | None = None,
    ):
        aic = _load_aiconfigurator()
        version = resolve_backend_version(backend_name, backend_version)
        database_mode = resolve_database_mode(database_mode)
        analytical = database_mode == DATABASE_MODE_ANALYTICAL
        if analytical:
            _activate_analytical_backend(
                system, xe_compute_config, backend_name=backend_name
            )

        database = aic["get_database"](
            system=system,
            backend=backend_name,
            version=version,
            # An analytical system has no collected grids on disk; the database
            # is a carrier for the system YAML plus the patched query methods.
            allow_missing_data=analytical,
        )
        if database is None:
            supported = (
                aic["get_supported_databases"]().get(system, {}).get(backend_name, [])
            )
            supported_versions = ", ".join(supported) if supported else "<none>"
            raise RuntimeError(
                "AIC perf database not found for "
                f"system={system!r}, backend={backend_name!r}, version={version!r}. "
                f"Supported versions for this system/backend: {supported_versions}"
            )
        if analytical:
            database.set_default_database_mode(
                aic["common"].DatabaseMode[_ANALYTICAL_SDK_MODE]
            )

        model_config_kwargs: dict = dict(
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=attention_dp_size or 1,
        )
        # Quantization overrides drive the per-op perf-DB lookups (GEMM/MoE/FMHA
        # precision) and the KV-cache element size, so predicted latency tracks
        # the quantized deployment instead of the model's default dtype. Omit
        # unset fields so ModelConfig keeps its own defaults.
        for cfg_key, field, dtype in (
            ("gemm_quant_mode", "gemm", gemm_dtype),
            ("moe_quant_mode", "moe", moe_dtype),
            ("fmha_quant_mode", "fmha", fmha_dtype),
            ("kvcache_quant_mode", "kvcache", kv_cache_dtype),
            ("comm_quant_mode", "comm", comm_dtype),
        ):
            quant_mode = _resolve_quant_mode(field, dtype)
            if quant_mode is not None:
                model_config_kwargs[cfg_key] = quant_mode
        if nextn:
            if not 1 <= nextn <= _NEXTN_ACCEPT_RATES_LEN:
                raise ValueError(
                    f"nextn must be 1..={_NEXTN_ACCEPT_RATES_LEN} when set, got {nextn}"
                )
            model_config_kwargs["nextn"] = nextn
            # aic-core models one verification iteration. Dynamo's scheduler
            # owns accepted-token progress and burst sampling above core.
            _pad_nextn_accept_rates(nextn_accept_rates)
        model_config = aic["config"].ModelConfig(**model_config_kwargs)
        model = aic["get_model"](
            model_path=model_path,
            model_config=model_config,
            backend_name=backend_name,
        )
        backend = aic["get_backend"](backend_name)
        self._backend = backend
        self._backend_name = backend_name
        self._database = database
        self._database_mode = database_mode
        self._model = model
        self._model_name = getattr(model, "model_name", None) or model_path
        logger.info(
            "AIC session initialized: backend=%s, system=%s, model=%s, tp=%d, mode=%s",
            backend_name,
            system,
            model_path,
            tp_size,
            database_mode,
        )

        # Phase 1.5: compile the model's op list to a Rust Engine ONCE, so each
        # predict call is a single Rust dispatch instead of a per-call Python
        # walk over model.context_ops / generation_ops. Falls back to the
        # Python op-walk if the compiled AIC-core engine is unavailable or fails.
        # The compiled engine reads collected parquet grids in Rust and never
        # re-enters Python, so it cannot see the analytical backend's patches.
        self._engine = None if analytical else self._build_compiled_engine()

    def _build_compiled_engine(self):
        """Build a cached aiconfigurator EngineHandle from the already-built
        model, or return None to fall back to the Python op-walk."""
        if _OP_PROFILE_ENABLED:
            logger.info(
                "AIC compiled-engine path disabled because DYNAMO_AIC_PROFILE_OPS "
                "is set; using the per-operator Python op-walk so each operator "
                "query can be timed individually."
            )
            return None
        if os.environ.get("DYNAMO_AIC_DISABLE_COMPILED_ENGINE"):
            logger.info(
                "AIC compiled-engine path disabled via env; using Python op-walk."
            )
            return None
        try:
            from aiconfigurator.sdk.rust_engine_step import _cached_engine_handle
        except Exception as exc:  # aiconfigurator-core without the compiled engine
            logger.info(
                "AIC compiled-engine path unavailable (%s); using Python op-walk.",
                exc,
            )
            return None
        try:
            handle = _cached_engine_handle(self._model, self._database)
            logger.info("AIC compiled-engine path active (Phase 1.5 Rust engine).")
            return handle
        except Exception as exc:
            logger.warning(
                "AIC compiled-engine build failed (%s); using Python op-walk.", exc
            )
            return None

    def _predict_context_latency(
        self, batch_size: int, effective_isl: int, prefix: int
    ) -> float:
        if effective_isl <= 0:
            raise ValueError(
                f"effective_isl must be positive, got effective_isl={effective_isl}"
            )

        total_latency = 0.0
        for op in self._model.context_ops:
            op_name = getattr(op, "_name", "")
            x = batch_size if "logits_gemm" in op_name else batch_size * effective_isl
            result = op.query(
                self._database,
                x=x,
                batch_size=batch_size,
                beam_width=1,
                s=effective_isl,
                prefix=prefix,
                model_name=self._model_name,
                seq_imbalance_correction_scale=1.0,
            )
            latency_ms = float(result)
            if _OP_PROFILE_ENABLED:
                _record_op_latency("context", op_name, latency_ms)
            total_latency += latency_ms

        return total_latency

    def _predict_generation_latency(self, batch_size: int, isl: int, osl: int) -> float:
        if osl <= 1:
            return 0.0

        effective_batch_size = batch_size * (self._model._nextn + 1)
        total_latency = 0.0

        for step in range(0, osl - 1, DEFAULT_STATIC_STRIDE):
            step_latency = 0.0
            repeat_count = min(DEFAULT_STATIC_STRIDE, osl - 1 - step)
            for op in self._model.generation_ops:
                op_name = getattr(op, "_name", "")
                result = op.query(
                    self._database,
                    x=effective_batch_size,
                    batch_size=effective_batch_size,
                    beam_width=1,
                    s=isl + step + 1,
                    model_name=self._model_name,
                    gen_seq_imbalance_correction_scale=1.0,
                )
                latency_ms = float(result)
                if _OP_PROFILE_ENABLED:
                    _record_op_latency("generation", op_name, latency_ms, repeat_count)
                step_latency += latency_ms

            total_latency += step_latency * repeat_count

        return total_latency

    def predict_prefill(
        self, batch_size: int, effective_isl: int, prefix: int
    ) -> float:
        """Predict prefill latency in ms from uncached tokens and cached prefix."""
        if self._engine is not None:
            # The engine's predict_prefill_latency takes the FULL isl and
            # subtracts `prefix` internally, whereas the caller already gives us
            # the post-prefix `effective_isl`. Pass effective_isl + prefix so the
            # engine recovers the same effective length (and keeps prefix for the
            # KV-cache-aware context-attention cost).
            return self._engine.predict_prefill_latency(
                batch_size, effective_isl + prefix, prefix
            )
        return self._predict_context_latency(batch_size, effective_isl, prefix)

    def predict_decode(self, batch_size: int, isl: int, osl: int) -> float:
        """Predict decode (generation) latency in ms."""
        if self._engine is not None:
            return self._engine.predict_decode_latency(batch_size, isl, osl)
        return self._predict_generation_latency(batch_size, isl, osl)


def create_session(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
    nextn: int | None = None,
    nextn_accept_rates: list[float] | str | None = None,
    database_mode: str | None = None,
    xe_compute_config: str | None = None,
) -> AicSession:
    """Factory function called from Rust via PyO3."""
    return AicSession(
        backend_name,
        system,
        model_path,
        tp_size,
        backend_version,
        moe_tp_size,
        moe_ep_size,
        attention_dp_size,
        gemm_dtype=gemm_dtype,
        moe_dtype=moe_dtype,
        fmha_dtype=fmha_dtype,
        kv_cache_dtype=kv_cache_dtype,
        comm_dtype=comm_dtype,
        nextn=nextn,
        nextn_accept_rates=nextn_accept_rates,
        database_mode=database_mode,
        xe_compute_config=xe_compute_config,
    )


def estimate_num_gpu_blocks(
    backend_name: str,
    system: str,
    model_path: str,
    tp_size: int,
    block_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    mem_fraction_static: float | None = None,
    free_gpu_memory_fraction: float | None = None,
    backend_version: str | None = None,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    attention_dp_size: int | None = None,
    gemm_dtype: str | None = None,
    moe_dtype: str | None = None,
    fmha_dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    comm_dtype: str | None = None,
    database_mode: str | None = None,
    xe_compute_config: str | None = None,
) -> int:
    """Estimate rank-local KV cache blocks for mocker/replay AIC configs.

    Delegates the budget math to aiconfigurator-core's unified
    ``sdk.memory.estimate_num_gpu_blocks`` (the single source of truth for the
    AIC memory estimator) instead of recomputing it here. The result is
    per rank (per single GPU): AIC's memory model is already sharded for the
    configured TP/DP shape, so the caller must not multiply it by TP or DP.

    The backend selects which memory-fraction knob applies, mapped onto AIC's
    ``memory_fraction_kind``/``memory_fraction_value``:

    - ``vllm``   -> ``of_total`` with ``gpu_memory_utilization`` (fraction of total HBM)
    - ``sglang`` -> ``of_total`` with ``mem_fraction_static``
    - ``trtllm`` -> ``of_free`` with ``free_gpu_memory_fraction`` (fraction of the
      HBM left after the model is loaded)
    """
    _validate_kv_capacity_backend(backend_name)
    database_mode = resolve_database_mode(database_mode)
    if database_mode == DATABASE_MODE_ANALYTICAL:
        _activate_analytical_backend(
            system, xe_compute_config, backend_name=backend_name
        )

    if backend_name == "trtllm":
        memory_fraction_kind = "of_free"
        memory_fraction_value = (
            free_gpu_memory_fraction
            if free_gpu_memory_fraction is not None
            else DEFAULT_FREE_GPU_MEMORY_FRACTION
        )
    elif backend_name == "sglang":
        memory_fraction_kind = "of_total"
        memory_fraction_value = (
            mem_fraction_static
            if mem_fraction_static is not None
            else DEFAULT_MEM_FRACTION_STATIC
        )
    else:  # vllm
        memory_fraction_kind = "of_total"
        memory_fraction_value = gpu_memory_utilization

    # Imported lazily because aiconfigurator-core is provided by the optional
    # `mocker` extra. An AIC-backed call requires that extra and fails fast when
    # it is absent.
    # TODO: account for whether specdec is enabled (pass `nextn=...`). Currently
    #   omitted due to a downstream AIC bug where `_get_memory_usage` predicts
    #   negative KV capacity with Eagle.
    try:
        from aiconfigurator.sdk.memory import (
            estimate_num_gpu_blocks as aic_estimate_num_gpu_blocks,
        )
    except ImportError as exc:
        missing = exc.name or ""
        if missing == "aiconfigurator" or missing.startswith(
            "aiconfigurator."
        ):
            raise RuntimeError(
                "aiconfigurator-core is required for AIC KV-cache estimation but is "
                "not installed; install the 'mocker' extra"
            ) from exc
        raise

    # AIC's non-KV memory is independent of batch size (activations track
    # max_num_tokens), so the fixed max_batch_size here does not affect the result.
    return int(
        aic_estimate_num_gpu_blocks(
            model_path,
            system,
            backend_name,
            backend_version=resolve_backend_version(backend_name, backend_version),
            scheduler_block_size=block_size,
            max_num_tokens=max_num_batched_tokens,
            max_batch_size=1,
            memory_fraction_kind=memory_fraction_kind,
            memory_fraction_value=memory_fraction_value,
            tp_size=tp_size,
            attention_dp_size=(
                attention_dp_size if attention_dp_size is not None else 1
            ),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=_resolve_quant_mode_name("gemm", gemm_dtype),
            moe_quant_mode=_resolve_quant_mode_name("moe", moe_dtype),
            fmha_quant_mode=_resolve_quant_mode_name("fmha", fmha_dtype),
            kvcache_quant_mode=_resolve_quant_mode_name("kvcache", kv_cache_dtype),
            comm_quant_mode=_resolve_quant_mode_name("comm", comm_dtype),
            database_mode=(
                _ANALYTICAL_SDK_MODE
                if database_mode == DATABASE_MODE_ANALYTICAL
                else database_mode
            ),
        )
    )
