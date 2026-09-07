# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline replay distribution report for `dynamo.replay` captures.

Renders input/output sequence length, per-request cache-hit rate, and predicted
latency shares by operator from one complete offline `--report-json` file.

Usage:
    python report/trace_distributions.py --report-json replay.json --output report.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PERCENTILES = (50, 75, 90, 99)
_PERCENTILE_COLORS = {50: "tab:blue", 75: "tab:green", 90: "tab:orange", 99: "tab:red"}


def _operator_shares(report: dict[str, Any]) -> list[tuple[str, float]]:
    detailed_breakdown = report.get("aic_operator_breakdown")
    if detailed_breakdown:
        shares = [
            (operator["op_name"], float(operator["pct_of_grand_total"]))
            for operator in detailed_breakdown.get("op_summary_across_phases", [])
            if operator.get("op_name") is not None
            and operator.get("pct_of_grand_total") is not None
        ]
        if shares:
            return shares

    latency_breakdown = report.get("aic_latency_breakdown")
    if latency_breakdown:
        totals: dict[str, float] = {}
        for phase in latency_breakdown.values():
            for operator, percentage in phase.items():
                totals[operator] = totals.get(operator, 0.0) + float(percentage)
        grand_total = sum(totals.values())
        if grand_total:
            return sorted(
                (
                    (operator, 100.0 * percentage / grand_total)
                    for operator, percentage in totals.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )

    return []


def _matrix_operator_shares(
    path: str | Path, model_key: str, dataset: str
) -> list[tuple[str, float]]:
    matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        breakdown = matrix[model_key]["datasets"][dataset]["breakdown_pct"]
    except KeyError as exc:
        raise ValueError(
            f"operator results do not contain {model_key}.datasets.{dataset}.breakdown_pct"
        ) from exc
    return sorted(
        (
            (operator, float(percentage))
            for operator, percentage in breakdown.items()
            if float(percentage) > 0.0
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def load_per_request_records(path: str | Path) -> list[dict[str, Any]]:
    """Load per-request records from a `--per-request-jsonl` file, a
    `--report-json` file containing a `per_request` list, or a single JSON
    record.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if "per_request" in data:
            per_request = data["per_request"]
            if per_request is None:
                raise ValueError(
                    f"{path} was captured without --per-request-jsonl / capture_per_request enabled"
                )
            return list(per_request)
        return [data]
    if isinstance(data, list):
        return list(data)

    # Not a single JSON document -- fall back to JSONL, one record per line.
    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _fmt(value: float, is_int: bool) -> str:
    return f"{value:,.0f}" if is_int else f"{value:,.3f}"


def _hist_panel(
    ax,
    values: Sequence[float],
    title: str,
    xlabel: str,
    *,
    log_x: bool,
    is_int: bool = False,
    bins: int = 60,
) -> None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        ax.set_title(f"{title}\nN = 0 -- not applicable to this trace", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    header = (
        f"N = {array.size:,}   "
        f"min = {_fmt(array.min(), is_int)}   "
        f"max = {_fmt(array.max(), is_int)}   "
        f"mean = {_fmt(array.mean(), is_int)}"
    )
    ax.set_title(f"{title}\n{header}", fontsize=10)

    if log_x:
        positive = array[array > 0]
        if positive.size == 0:
            edges = np.linspace(0, 1, bins + 1)
        else:
            lo, hi = positive.min(), positive.max()
            if lo == hi:
                lo, hi = lo * 0.5, max(hi * 1.5, hi + 1.0)
            edges = np.logspace(math.log10(lo), math.log10(hi), bins + 1)
        ax.hist(array, bins=edges, color="gray", edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
    else:
        ax.hist(array, bins=bins, range=(0.0, 1.0), color="gray", edgecolor="black", linewidth=0.3)

    for p in _PERCENTILES:
        x = float(np.percentile(array, p))
        ax.axvline(
            x,
            color=_PERCENTILE_COLORS[p],
            linestyle="--",
            linewidth=1.2,
            label=f"p{p} = {_fmt(x, is_int)}",
        )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def _operator_share_panel(ax, shares: Sequence[tuple[str, float]]) -> None:
    if not shares:
        ax.set_title(
            "Predicted latency by operator\n"
            "No operator breakdown in this offline report",
            fontsize=10,
        )
        ax.set_xlabel("operator")
        ax.set_ylabel("share of predicted latency (%)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels, percentages = zip(*shares, strict=True)
    array = np.asarray(percentages, dtype=float)
    positions = np.arange(len(labels))
    header = (
        f"N = {array.size:,}   "
        f"min = {_fmt(array.min(), False)}   "
        f"max = {_fmt(array.max(), False)}   "
        f"mean = {_fmt(array.mean(), False)}"
    )
    ax.bar(positions, array, color="gray", edgecolor="black", linewidth=0.3)
    ax.set_title(f"Predicted latency share by operator\n{header}", fontsize=10)
    ax.set_xlabel("operator")
    ax.set_ylabel("share of predicted latency (%)")
    ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(bottom=0)
    for percentile in _PERCENTILES:
        value = float(np.percentile(array, percentile))
        ax.axhline(
            value,
            color=_PERCENTILE_COLORS[percentile],
            linestyle="--",
            linewidth=1.2,
            label=f"p{percentile} = {_fmt(value, False)}",
        )
    ax.legend(fontsize=8, loc="upper right")


def _cache_hit_ratios(records: list[dict[str, Any]], block_size: int) -> list[float]:
    ratios = []
    for record in records:
        input_length = record.get("input_length")
        reused = record.get("reused_input_tokens")
        if not input_length or reused is None:
            continue
        total_blocks = math.ceil(input_length / block_size)
        if total_blocks == 0:
            continue
        hit_blocks = min(reused // block_size, total_blocks)
        ratios.append(hit_blocks / total_blocks)
    return ratios


def _inter_turn_think_times_seconds(records: list[dict[str, Any]]) -> list[float]:
    """Gap between one turn's last token and the next turn's arrival, within
    a session. Sessions are ordered by `arrival_time_ms` rather than
    `turn_index` -- agentic-trace replay always reports `turn_index=0` (each
    row becomes its own single-turn scheduling unit internally), so arrival
    order is the only reliable ordering across trace formats.
    """
    sessions: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        session_id = record.get("session_id")
        if session_id is None:
            continue
        sessions[session_id].append(record)

    think_times = []
    for turns in sessions.values():
        turns.sort(key=lambda r: r.get("arrival_time_ms") or 0.0)
        for previous, current in zip(turns, turns[1:]):
            previous_end = previous.get("last_token_ms")
            current_start = current.get("arrival_time_ms")
            if previous_end is None or current_start is None:
                continue
            gap_ms = current_start - previous_end
            if gap_ms >= 0:
                think_times.append(gap_ms / 1000.0)
    return think_times


def _main_agent_turns_per_session(records: list[dict[str, Any]]) -> list[int]:
    """Turn count per session, counting only main-agent (top-level) turns --
    i.e. records whose `parent_session_id` is unset. For non-agentic traces
    every record qualifies, since `parent_session_id` is only ever populated
    for agentic (`--trace-format agentic_mooncake`) sub-agent sessions.
    """
    turns_per_session: dict[Any, int] = defaultdict(int)
    for record in records:
        session_id = record.get("session_id")
        if session_id is None or record.get("parent_session_id") is not None:
            continue
        turns_per_session[session_id] += 1
    return list(turns_per_session.values())


def _agent_turn_depth_per_session(records: list[dict[str, Any]]) -> list[float]:
    """For each main-agent (root) session tree, the mean turn count across
    that root session and every sub-agent session it (transitively) spawned.
    Empty when the trace carries no `parent_session_id` links at all (i.e.
    any non-agentic trace, or an agentic trace replayed before this field
    existed).
    """
    turn_counts: dict[Any, int] = defaultdict(int)
    parent_of: dict[Any, Any] = {}
    for record in records:
        session_id = record.get("session_id")
        if session_id is None:
            continue
        turn_counts[session_id] += 1
        parent_session_id = record.get("parent_session_id")
        if parent_session_id is not None:
            parent_of[session_id] = parent_session_id

    if not parent_of:
        return []

    children_of: dict[Any, list[Any]] = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)

    roots = [session_id for session_id in turn_counts if session_id not in parent_of]

    depths = []
    for root in roots:
        tree, stack = [], [root]
        while stack:
            node = stack.pop()
            tree.append(node)
            stack.extend(children_of.get(node, []))
        counts = [turn_counts[node] for node in tree if node in turn_counts]
        if counts:
            depths.append(sum(counts) / len(counts))
    return depths


def build_report(
    records: list[dict[str, Any]],
    operator_shares: Sequence[tuple[str, float]],
    block_size: int = 512,
):
    isl = [r["input_length"] for r in records if r.get("input_length") is not None]
    osl = [r["output_length"] for r in records if r.get("output_length") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    fig.suptitle("DynoSim offline replay distributions", fontsize=16)

    _hist_panel(
        axes[0][0],
        isl,
        "Input sequence length per request",
        "tokens (log)",
        log_x=True,
        is_int=True,
    )
    _hist_panel(
        axes[0][1],
        osl,
        "Output sequence length per request",
        "tokens (log)",
        log_x=True,
        is_int=True,
    )
    _hist_panel(
        axes[1][0],
        _cache_hit_ratios(records, block_size),
        f"Per-request cache hit rate (block_size={block_size})",
        "hits / total blocks",
        log_x=False,
    )
    _operator_share_panel(axes[1][1], operator_shares)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "trace_distributions.png"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report/trace_distributions.py")
    parser.add_argument(
        "--report-json",
        required=True,
        help="complete offline dynamo.replay --report-json output",
    )
    parser.add_argument(
        "--operator-results-json",
        help="full dynamic matrix JSON containing datasets.<name>.breakdown_pct; "
        "defaults to the operator breakdown embedded in --report-json",
    )
    parser.add_argument(
        "--operator-model-key",
        default="qwen3_1_7b_dense",
        help="model key in --operator-results-json (default: qwen3_1_7b_dense)",
    )
    parser.add_argument(
        "--operator-dataset",
        choices=("conversation", "mooncake", "synthetic", "toolagent"),
        help="dataset key in --operator-results-json",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="tokens per cache block for the cache-hit-rate panel (default: 512, matching "
        "dynamo.replay's --trace-block-size default)",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"output PNG path (default: alongside this script, {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("per_request")
    if records is None:
        raise SystemExit(
            f"{report_path} has no per_request records; rerun offline replay with "
            "--per-request-jsonl to enable capture"
        )
    if not records:
        raise SystemExit(f"no per-request records found in {report_path}")

    if args.operator_results_json:
        if args.operator_dataset is None:
            parser.error("--operator-dataset is required with --operator-results-json")
        operator_shares = _matrix_operator_shares(
            args.operator_results_json,
            args.operator_model_key,
            args.operator_dataset,
        )
    else:
        operator_shares = _operator_shares(report)

    fig = build_report(records, operator_shares, block_size=args.block_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
