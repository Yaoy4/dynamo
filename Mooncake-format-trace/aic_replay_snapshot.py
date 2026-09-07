# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a request- and pass-level snapshot from an AIC DynoSim replay.

Usage:
    python report/aic_replay_snapshot.py \
        --report-json /tmp/dynosim-aic-qwen3-1.7b-report.json \
        --per-request-jsonl /tmp/dynosim-aic-qwen3-1.7b-request.jsonl \
        --per-pass-jsonl /tmp/dynosim-aic-qwen3-1.7b-pass.jsonl \
        --output report/aic_replay_snapshot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PERCENTILES = (50, 75, 90, 99)
_PERCENTILE_COLORS = {50: "tab:blue", 75: "tab:green", 90: "tab:orange", 99: "tab:red"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _values(records: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [record[field] for record in records if record.get(field) is not None],
        dtype=float,
    )


def _operator_shares(report: dict[str, Any]) -> list[tuple[str, float]]:
    python_breakdown = report.get("aic_operator_breakdown")
    if python_breakdown:
        operators = python_breakdown.get("op_summary_across_phases", [])
        shares = [
            (operator["op_name"], float(operator["pct_of_grand_total"]))
            for operator in operators
            if operator.get("pct_of_grand_total") is not None
        ]
        if shares:
            return shares

    native_breakdown = report.get("aic_latency_breakdown")
    if native_breakdown:
        totals: dict[str, float] = {}
        for role_breakdown in native_breakdown.values():
            for operator, percentage in role_breakdown.items():
                totals[operator] = totals.get(operator, 0.0) + float(percentage)
        total = sum(totals.values())
        if total:
            return sorted(
                ((operator, 100.0 * percentage / total) for operator, percentage in totals.items()),
                key=lambda item: item[1],
                reverse=True,
            )

    raise ValueError(
        "operator breakdown is unavailable: wait for this replay to finish, or rerun it with "
        "DYNAMO_AIC_PROFILE_OPS=1 to write aic_operator_breakdown into --report-json"
    )


def _histogram(ax: Any, values: np.ndarray, title: str, xlabel: str, *, log_x: bool = False) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.set_title(f"{title}\nN = 0 -- no records available", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        return

    ax.set_title(
        f"{title}\nN = {values.size:,}   min = {values.min():,.3f}   "
        f"max = {values.max():,.3f}   mean = {values.mean():,.3f}",
        fontsize=10,
    )
    if log_x:
        positive = values[values > 0]
        if positive.size:
            low, high = positive.min(), positive.max()
            if low == high:
                low, high = low * 0.5, high * 1.5
            bins = np.logspace(math.log10(low), math.log10(high), 41)
            ax.hist(positive, bins=bins, color="gray", edgecolor="black", linewidth=0.3)
            ax.set_xscale("log")
    else:
        ax.hist(values, bins=min(60, max(8, values.size // 3)), color="gray", edgecolor="black", linewidth=0.3)
    for percentile in _PERCENTILES:
        value = float(np.percentile(values, percentile))
        ax.axvline(
            value,
            color=_PERCENTILE_COLORS[percentile],
            linestyle="--",
            linewidth=1.2,
            label=f"p{percentile} = {value:,.3f}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8, loc="upper right")


def _operator_share_panel(ax: Any, shares: list[tuple[str, float]]) -> None:
    labels, percentages = zip(*shares, strict=True)
    positions = np.arange(len(labels))
    ax.bar(positions, percentages, color="#377eb8", edgecolor="black", linewidth=0.3)
    ax.set_title("AIC predicted latency by operator", fontsize=10)
    ax.set_xlabel("operator")
    ax.set_ylabel("share of predicted latency (%)")
    ax.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(bottom=0)
    for position, percentage in zip(positions, percentages, strict=True):
        ax.text(position, percentage, f"{percentage:.1f}%", ha="center", va="bottom", fontsize=7)


def build_snapshot(
    report: dict[str, Any], requests: list[dict[str, Any]], passes: list[dict[str, Any]]
) -> plt.Figure:
    completed = report.get("summary", {}).get("completed_requests", len(requests))
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(
        f"DynoSim AIC analytical replay distributions (interim snapshot: {completed} completed requests, "
        f"{len(passes)} passes)",
        fontsize=16,
    )

    _histogram(axes[0, 0], _values(requests, "ttft_ms"), "Time to first token", "milliseconds", log_x=True)
    _histogram(axes[0, 1], _values(requests, "e2e_latency_ms"), "End-to-end latency", "milliseconds", log_x=True)
    _histogram(axes[1, 0], _values(requests, "itl_ms"), "Inter-token latency", "milliseconds", log_x=True)
    _histogram(axes[1, 1], _values(passes, "prefill_time_ms"), "AIC predicted prefill pass time", "milliseconds", log_x=True)
    _histogram(axes[2, 0], _values(passes, "decode_time_ms"), "AIC predicted decode pass time", "milliseconds", log_x=True)
    _operator_share_panel(axes[2, 1], _operator_shares(report))

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--per-request-jsonl", type=Path, required=True)
    parser.add_argument("--per-pass-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    figure = build_snapshot(report, _load_jsonl(args.per_request_jsonl), _load_jsonl(args.per_pass_jsonl))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()