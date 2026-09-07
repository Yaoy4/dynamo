#!/usr/bin/env python3
"""Extract the REAL, replay-observed per-worker batch-size distribution from
a `--per-pass-jsonl` capture (one line per (worker, scheduler-pass)), and the
REAL system-level in-flight request count from a `--per-request-jsonl`
capture (one line per completed request, with arrival/admit/terminal
timestamps) -- both produced by re-running scenario 1 (round_robin, 8
workers) with telemetry capture enabled:

    $VENV -m dynamo.replay $TRACE \\
      --trace-format agentic_mooncake --trace-block-size 64 \\
      --replay-mode offline --router-mode round_robin --num-workers 8 \\
      --extra-engine-args '{"num_gpu_blocks": 300000}' \\
      --report-json semianalysisai/reports/scenario1b_batch_telemetry.json \\
      --per-request-jsonl semianalysisai/reports/scenario1b_per_request.jsonl \\
      --per-pass-jsonl semianalysisai/reports/scenario1b_per_pass.jsonl

This is the most direct, ground-truth answer to "what batch size does a
static AIC estimate correspond to": `decode_batch_size`/`prefill_batch_size`
in each ReplayPassRecord IS the literal number of requests the (simulated)
serving engine actually batched together in one scheduler pass on one
worker -- not a workload-only proxy.

Why this script streams instead of loading the file: the per-pass capture
covers 8 workers x tens of millions of scheduler passes over the ~10.5
simulated days and is tens of GB on disk -- far too large to load into
memory or to keep as a committed reproducible artifact. This script reduces
it to compact summary statistics and is intended to be re-run on demand;
the raw `scenario1b_per_pass.jsonl` file itself is deleted after use (see
report Sec. 3.3 / 12.4 for the regeneration command).

By default this analyzes the scenario1b (round_robin, default latency
model) capture. To analyze a different capture -- e.g. an AIC-calibrated
re-run used to check whether swapping the latency model changes the
*observed* batch-size distribution -- pass explicit paths:

    $VENV analyze_observed_batch_telemetry.py \\
      --per-pass-jsonl semianalysisai/reports/scenario5b_per_pass.jsonl \\
      --per-request-jsonl semianalysisai/reports/scenario5b_per_request.jsonl \\
      --output semianalysisai/data/observed_batch_size_stats_aic.json \\
      --source-label "scenario5b (kv_router, 8 workers, AIC-calibrated)"
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PER_PASS_PATH = ROOT / "reports" / "scenario1b_per_pass.jsonl"
DEFAULT_PER_REQUEST_PATH = ROOT / "reports" / "scenario1b_per_request.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "observed_batch_size_stats.json"
DEFAULT_SOURCE_LABEL = (
    "scenario1b (round_robin, 8 workers) with --per-pass-jsonl / --per-request-jsonl"
)


def _percentile_from_counter(counter: dict[int, int], frac: float, total: int) -> int:
    cumulative = 0
    for level in sorted(counter):
        cumulative += counter[level]
        if cumulative / total >= frac:
            return level
    return max(counter)


def _summarize_counter(counter: dict[int, int]) -> dict:
    total = sum(counter.values())
    weighted_sum = sum(level * count for level, count in counter.items())
    return {
        "n_passes": total,
        "mean": weighted_sum / total,
        "median_p50": _percentile_from_counter(counter, 0.50, total),
        "p90": _percentile_from_counter(counter, 0.90, total),
        "p99": _percentile_from_counter(counter, 0.99, total),
        "max": max(counter),
    }


def _analyze_per_pass(per_pass_path: Path) -> dict:
    """Streams a `--per-pass-jsonl` capture once, building integer histograms
    (dict[batch_size] -> count) for decode_batch_size and prefill_batch_size,
    both aggregated over all 8 workers and split per worker_idx."""
    decode_hist_all: dict[int, int] = {}
    prefill_hist_all: dict[int, int] = {}
    decode_hist_by_worker: dict[int, dict[int, int]] = {}
    prefill_hist_by_worker: dict[int, dict[int, int]] = {}
    n_lines = 0

    with per_pass_path.open() as f:
        for line in f:
            row = json.loads(line)
            n_lines += 1
            w = row["worker_idx"]
            db = row["decode_batch_size"]
            pb = row["prefill_batch_size"]

            decode_hist_all[db] = decode_hist_all.get(db, 0) + 1
            prefill_hist_all[pb] = prefill_hist_all.get(pb, 0) + 1

            decode_hist_by_worker.setdefault(w, {})
            decode_hist_by_worker[w][db] = decode_hist_by_worker[w].get(db, 0) + 1
            prefill_hist_by_worker.setdefault(w, {})
            prefill_hist_by_worker[w][pb] = prefill_hist_by_worker[w].get(pb, 0) + 1

    return {
        "n_pass_records": n_lines,
        "n_workers": len(decode_hist_by_worker),
        "decode_batch_size": {
            "all_workers": _summarize_counter(decode_hist_all),
            "per_worker": {
                str(w): _summarize_counter(h) for w, h in sorted(decode_hist_by_worker.items())
            },
        },
        "prefill_batch_size": {
            "all_workers": _summarize_counter(prefill_hist_all),
            "per_worker": {
                str(w): _summarize_counter(h) for w, h in sorted(prefill_hist_by_worker.items())
            },
        },
    }


def _sweep_line_from_intervals(intervals: list[tuple[float, float]]) -> dict:
    events = []
    for s, e in intervals:
        events.append((s, 0, +1))
        events.append((e, 1, -1))
    events.sort(key=lambda x: (x[0], x[1]))

    segments = []
    current = 0
    prev_t = events[0][0]
    for t, _kind, delta in events:
        if t > prev_t:
            segments.append((t - prev_t, current))
        current += delta
        prev_t = t

    total = sum(d for d, _ in segments)
    mean = sum(d * c for d, c in segments) / total
    by_level: dict[int, float] = {}
    peak = 0
    for d, c in segments:
        by_level[c] = by_level.get(c, 0.0) + d
        peak = max(peak, c)
    levels = sorted(by_level)
    cumulative = 0.0
    result_pct = {}
    remaining = {"p50": 0.50, "p90": 0.90, "p99": 0.99}
    for lvl in levels:
        cumulative += by_level[lvl]
        frac = cumulative / total
        for name, target in list(remaining.items()):
            if frac >= target:
                result_pct[name] = lvl
                del remaining[name]
    for name in remaining:
        result_pct[name] = peak
    return {
        "total_span_ms": total,
        "time_weighted_mean": mean,
        "median_p50": result_pct["p50"],
        "p90": result_pct["p90"],
        "p99": result_pct["p99"],
        "peak_max": peak,
    }


def _analyze_per_request(per_request_path: Path) -> dict:
    """Streams a `--per-request-jsonl` capture once: true SYSTEM-level
    in-flight count (post-admission-control) via sweep-line over
    (first_admit_ms or arrival_time_ms, terminal_time_ms)."""
    intervals_admit = []
    intervals_arrival = []
    n = 0
    with per_request_path.open() as f:
        for line in f:
            row = json.loads(line)
            n += 1
            arrival = row["arrival_time_ms"]
            admit = row.get("first_admit_ms")
            terminal = row["terminal_time_ms"]
            if terminal is None:
                continue
            intervals_arrival.append((arrival, terminal))
            if admit is not None:
                intervals_admit.append((admit, terminal))

    return {
        "n_requests": n,
        "in_flight_by_arrival_time": _sweep_line_from_intervals(intervals_arrival),
        "in_flight_by_admit_time": _sweep_line_from_intervals(intervals_admit)
        if intervals_admit
        else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-pass-jsonl", type=Path, default=DEFAULT_PER_PASS_PATH)
    parser.add_argument("--per-request-jsonl", type=Path, default=DEFAULT_PER_REQUEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = {
        "source": args.source_label,
        "per_pass": _analyze_per_pass(args.per_pass_jsonl),
        "per_request": _analyze_per_request(args.per_request_jsonl),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}\n")

    pp = result["per_pass"]
    print(f"per-pass records: {pp['n_pass_records']:,} across {pp['n_workers']} workers")
    d = pp["decode_batch_size"]["all_workers"]
    p = pp["prefill_batch_size"]["all_workers"]
    print(
        f"decode_batch_size  (全 worker 聚合): mean={d['mean']:.3f} median={d['median_p50']} "
        f"p90={d['p90']} p99={d['p99']} max={d['max']}"
    )
    print(
        f"prefill_batch_size (全 worker 聚合): mean={p['mean']:.3f} median={p['median_p50']} "
        f"p90={p['p90']} p99={p['p99']} max={p['max']}"
    )
    print("\n按 worker 拆分 decode_batch_size:")
    for w, s in pp["decode_batch_size"]["per_worker"].items():
        print(f"  worker {w}: mean={s['mean']:.3f} median={s['median_p50']} p90={s['p90']} max={s['max']}")

    pr = result["per_request"]
    ia = pr["in_flight_by_arrival_time"]
    print(f"\n系统级真实 in-flight（含 admission，按 arrival_time_ms 计）:")
    print(
        f"  mean={ia['time_weighted_mean']:.4f} median={ia['median_p50']} "
        f"p90={ia['p90']} p99={ia['p99']} peak={ia['peak_max']}"
    )
    if pr["in_flight_by_admit_time"]:
        iad = pr["in_flight_by_admit_time"]
        print(f"系统级真实 in-flight（按 first_admit_ms 计，排除排队等待）:")
        print(
            f"  mean={iad['time_weighted_mean']:.4f} median={iad['median_p50']} "
            f"p90={iad['p90']} p99={iad['p99']} peak={iad['peak_max']}"
        )


if __name__ == "__main__":
    main()
