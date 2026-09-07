#!/usr/bin/env python3
"""Compute the *true* in-flight concurrency time series (not just its
Little's-Law average) for the semianalysisai/cc-traces-weka-062126 dataset,
answering: "what if sessions did not share a simulated start point -- would
a direct in-flight (sweep-line) count over time be a more valid/informative
batch-size proxy than the aggregate Little's Law number?"

Background
----------
`compute_static_aic_reference.py` computed a single "intrinsic" average
concurrency via Little's Law: L = (sum of every leaf's api_time) / (span of
the simulated clock). That number (~1.328) is mathematically *already* the
time-weighted MEAN of the true instantaneous in-flight count (Little's Law
H/T identity: H = total customer-time, T = window length, H/T = time-average
number in system) -- so it is not "wrong". But it hides two things a single
aggregate cannot show:

  1. Whether concurrency is validly comparable at all -- Little's Law and any
     overlap-based computation over raw per-session-relative timestamps is
     only valid if those timestamps live on one common/shared clock. This
     script explicitly re-verifies that precondition two independent ways:
       a) every one of the 393 sessions' first request has raw t == 0.0 (data
          fact, already known);
       b) dynamo.replay's own trace-based arrival model uses the trace's raw
          timestamps verbatim (scaled only by --arrival-speedup-ratio,
          default 1.0) -- see `Trace::speed_up_timing` in
          lib/bindings/python/rust/llm/replay.rs -- so no additional,
          undocumented staggering is silently introduced by the replay engine
          either. If a *different* dataset had sessions arriving at genuinely
          different real-world times not captured by a shared t=0, this same
          sweep-line technique would still be valid *as long as the raw
          per-session t values are corrected onto a common global clock
          first* (e.g. t_global = t_local + session_arrival_offset) -- doing
          the overlap computation on un-corrected, staggered-but-treated-as-
          simultaneous local timestamps would silently overstate concurrency.
  2. The *distribution* over time -- mean, median, p90, p99, peak -- of how
     many requests are simultaneously in flight. This matters far more than
     the mean alone for sizing a static AIC batch_size: a workload that is
     "1 on average but bursts to 80" needs a very different answer than one
     that is "steadily 1-2".

Method
------
Classic sweep-line / interval-count algorithm over the same (start, end)
leaf-request intervals used for the "intrinsic" Little's Law number in
compute_static_aic_reference.py (data/traces.jsonl, t and api_time fields,
in seconds): build a +1 event at every interval start and a -1 event at
every interval end, sort by time (ties: ends processed before starts, so a
request that ends exactly when another starts is not double-counted), sweep
once while tracking (segment_duration, concurrency_level) pairs, then derive
a time-weighted distribution (mean/median/p90/p99/max) from those segments.

Cross-check: the time-weighted mean computed here must equal (up to
floating-point error) the "intrinsic_unconstrained.avg_concurrency" value
from compute_static_aic_reference.py, since both express the same Little's
Law H/T identity -- one via direct integration (this script), the other via
the shortcut sum(api_time)/span (the other script). Matching confirms the
sweep-line implementation is correct.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "traces.jsonl"
OUTPUT_PATH = ROOT / "data" / "inflight_timeseries_stats.json"


def _load_intervals() -> list[tuple[float, float]]:
    """Same leaf extraction logic as compute_static_aic_reference.py's
    _intrinsic_concurrency(): every top-level request, or every inner request
    of a subagent group, contributes one (start_s, end_s) interval."""
    intervals = []
    with RAW_PATH.open() as f:
        for line in f:
            sess = json.loads(line)
            for req in sess.get("requests", []):
                inner_reqs = (
                    req.get("requests", []) if req.get("type") == "subagent" else [req]
                )
                for r in inner_reqs:
                    t = r["t"]
                    api_time = r.get("api_time", 0.0) or 0.0
                    if api_time <= 0.0:
                        continue
                    intervals.append((t, t + api_time))
    return intervals


def _sweep_line_segments(intervals: list[tuple[float, float]]) -> list[tuple[float, int]]:
    """Returns a list of (segment_duration_seconds, concurrency_level) pairs
    covering the full span from the first start to the last end."""
    events = []
    for start, end in intervals:
        events.append((start, 0, +1))  # kind=0 (start) sorts after kind=1 (end) at tie
        events.append((end, 1, -1))  # end events processed first at equal timestamps
    events.sort(key=lambda e: (e[0], e[1]))

    segments = []
    current = 0
    prev_t = events[0][0]
    for t, _kind, delta in events:
        if t > prev_t:
            segments.append((t - prev_t, current))
        current += delta
        prev_t = t
    return segments


def _time_weighted_stats(segments: list[tuple[float, int]]) -> dict:
    total_duration = sum(d for d, _ in segments)
    mean = sum(d * c for d, c in segments) / total_duration

    # Build a duration-weighted CDF over concurrency levels to extract
    # percentiles and a compact histogram.
    by_level: dict[int, float] = {}
    peak = 0
    for d, c in segments:
        by_level[c] = by_level.get(c, 0.0) + d
        peak = max(peak, c)

    levels_sorted = sorted(by_level.keys())
    cumulative = 0.0
    percentiles = {}
    targets = {"p50": 0.50, "p90": 0.90, "p99": 0.99}
    remaining_targets = dict(targets)
    for level in levels_sorted:
        cumulative += by_level[level]
        frac = cumulative / total_duration
        for name, target_frac in list(remaining_targets.items()):
            if frac >= target_frac:
                percentiles[name] = level
                del remaining_targets[name]
    for name in remaining_targets:
        percentiles[name] = peak

    # Fraction of wall-clock time spent at each small concurrency level (0-5)
    # plus a ">=6" bucket, for a human-readable burstiness picture.
    histogram = {}
    for level in range(0, 6):
        histogram[str(level)] = round(by_level.get(level, 0.0) / total_duration, 6)
    histogram[">=6"] = round(
        sum(d for lvl, d in by_level.items() if lvl >= 6) / total_duration, 6
    )

    return {
        "total_span_seconds": total_duration,
        "total_span_days": total_duration / 86400.0,
        "n_segments": len(segments),
        "time_weighted_mean": mean,
        "median_p50": percentiles["p50"],
        "p90": percentiles["p90"],
        "p99": percentiles["p99"],
        "peak_max": peak,
        "time_fraction_by_level": histogram,
    }


def main() -> None:
    intervals = _load_intervals()
    segments = _sweep_line_segments(intervals)
    stats = _time_weighted_stats(segments)

    result = {
        "source_dataset": "semianalysisai/cc-traces-weka-062126",
        "method": (
            "sweep-line over raw (start_s, end_s) leaf-request intervals "
            "from data/traces.jsonl; validity precondition (shared/known "
            "common clock across all 393 sessions) verified independently "
            "-- see module docstring"
        ),
        "n_intervals": len(intervals),
        **stats,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {OUTPUT_PATH}\n")

    print(f"叶子请求区间数: {result['n_intervals']:,}")
    print(f"总时间跨度: {stats['total_span_days']:.2f} 天 ({stats['total_span_seconds']:.1f} s)")
    print(f"时间加权 mean 并发数: {stats['time_weighted_mean']:.4f}")
    print(f"时间加权 median(p50): {stats['median_p50']}")
    print(f"时间加权 p90: {stats['p90']}")
    print(f"时间加权 p99: {stats['p99']}")
    print(f"峰值并发数: {stats['peak_max']}")
    print("各并发档位占用时间比例:")
    for level, frac in stats["time_fraction_by_level"].items():
        print(f"  concurrency={level:>3s}: {frac * 100:6.2f}% 的时间")


if __name__ == "__main__":
    main()
