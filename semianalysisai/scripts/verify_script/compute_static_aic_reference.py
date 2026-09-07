#!/usr/bin/env python3
"""Compute reference ISL / OSL / batch-size numbers for handing off to a
static AIConfigurator (`aiconfigurator cli estimate --estimate-mode static`)
single-point run, derived from the semianalysisai/cc-traces-weka-062126
replay dataset.

Two independent data sources are used:

  1. data/semianalysis_cc_traces.agentic_mooncake.jsonl (the converted,
     flattened trace) for ISL/OSL distribution stats -- mean, median, p90,
     p99, max -- computed overall and split by request_kind (main-agent
     "foreground" turns vs sub-agent "agent" turns), since the two
     populations differ substantially (main turns resend the full growing
     conversation history; sub-agent turns start with a fresh, much
     shorter context).

  2. data/traces.jsonl (the raw dataset) for the batch-size/concurrency
     estimate. Every one of the 393 sessions' `requests[].t` starts at 0.0
     (verified empirically), i.e. DynoSim replays all sessions on one
     shared simulated clock starting simultaneously rather than staggering
     arrivals. That makes it valid to compute a genuine, dataset-intrinsic
     average concurrency via Little's Law:

         L = (sum of every leaf request's api_time) / (span of the
             simulated clock from the first to the last request across
             all 393 sessions)

     This is the *unconstrained* (infinite-capacity) concurrency implied
     purely by the historical arrival pattern -- i.e. "how many requests
     would naturally overlap if capacity were never a bottleneck." It is
     cross-checked against the *observed* concurrency from the completed
     full-scale 8-worker DynoSim run (scenario1), computed the same way
     (Little's Law) from that run's request_throughput_rps and
     mean_e2e_latency_ms, which additionally reflects real admission
     control / queueing.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOONCAKE_PATH = ROOT / "data" / "semianalysis_cc_traces.agentic_mooncake.jsonl"
RAW_PATH = ROOT / "data" / "traces.jsonl"
SCENARIO1_PATH = ROOT / "reports" / "scenario1_full_round_robin.json"
OUTPUT_PATH = ROOT / "data" / "static_aic_reference.json"


def _length_stats(values: list[int]) -> dict:
    vals = sorted(values)
    n = len(vals)
    return {
        "n": n,
        "mean": sum(vals) / n,
        "median": statistics.median(vals),
        "p90": vals[int(n * 0.90)],
        "p99": vals[int(n * 0.99)],
        "max": vals[-1],
    }


def _isl_osl_stats() -> dict:
    isl = {"all": [], "main": [], "child": []}
    osl = {"all": [], "main": [], "child": []}
    with MOONCAKE_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            bucket = "main" if row.get("request_kind") == "foreground" else "child"
            isl["all"].append(row["input_length"])
            osl["all"].append(row["output_length"])
            isl[bucket].append(row["input_length"])
            osl[bucket].append(row["output_length"])
    return {
        "isl": {k: _length_stats(v) for k, v in isl.items()},
        "osl": {k: _length_stats(v) for k, v in osl.items()},
    }


def _intrinsic_concurrency() -> dict:
    """Little's Law concurrency from raw historical timestamps, assuming
    all sessions replay on one shared simulated clock starting at t=0
    (verified: every session's first request has t == 0.0)."""
    total_busy_seconds = 0.0
    min_start = float("inf")
    max_end = float("-inf")
    n_leaves = 0
    with RAW_PATH.open() as f:
        for line in f:
            sess = json.loads(line)
            for req in sess.get("requests", []):
                inner_reqs = req.get("requests", []) if req.get("type") == "subagent" else [req]
                for r in inner_reqs:
                    t = r["t"]
                    api_time = r.get("api_time", 0.0) or 0.0
                    total_busy_seconds += api_time
                    min_start = min(min_start, t)
                    max_end = max(max_end, t + api_time)
                    n_leaves += 1
    span_seconds = max_end - min_start
    return {
        "n_leaves": n_leaves,
        "total_busy_seconds": total_busy_seconds,
        "span_seconds": span_seconds,
        "span_days": span_seconds / 86400.0,
        "avg_concurrency": total_busy_seconds / span_seconds,
    }


def _observed_concurrency_from_scenario1() -> dict:
    """Little's Law concurrency observed in the completed full-scale
    8-worker round_robin DynoSim run (includes real admission control and
    queueing, unlike the intrinsic/unconstrained estimate above)."""
    summary = json.loads(SCENARIO1_PATH.read_text())["summary"]
    rps = summary["request_throughput_rps"]
    mean_e2e_s = summary["mean_e2e_latency_ms"] / 1000.0
    system_wide = rps * mean_e2e_s
    num_workers = 8
    return {
        "request_throughput_rps": rps,
        "mean_e2e_latency_ms": summary["mean_e2e_latency_ms"],
        "num_workers": num_workers,
        "avg_concurrency_system_wide": system_wide,
        "avg_concurrency_per_worker": system_wide / num_workers,
    }


def main() -> None:
    result = {
        "source_dataset": "semianalysisai/cc-traces-weka-062126",
        **_isl_osl_stats(),
        "batch_size": {
            "intrinsic_unconstrained": _intrinsic_concurrency(),
            "observed_scenario1_8worker": _observed_concurrency_from_scenario1(),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {OUTPUT_PATH}\n")

    for kind in ("all", "main", "child"):
        s = result["isl"][kind]
        o = result["osl"][kind]
        print(
            f"[{kind:5s}] ISL mean={s['mean']:>10,.0f} median={s['median']:>9,.0f} "
            f"p90={s['p90']:>9,.0f} p99={s['p99']:>9,.0f} max={s['max']:>9,.0f}  |  "
            f"OSL mean={o['mean']:>7,.0f} median={o['median']:>7,.0f} "
            f"p90={o['p90']:>7,.0f} p99={o['p99']:>7,.0f} max={o['max']:>7,.0f}"
        )

    intrinsic = result["batch_size"]["intrinsic_unconstrained"]
    observed = result["batch_size"]["observed_scenario1_8worker"]
    print(f"\n批量并发（batch size）估计：")
    print(
        f"  intrinsic (无容量约束，全 393 会话共享 t=0 起点): "
        f"{intrinsic['avg_concurrency']:.3f}  (跨度 {intrinsic['span_days']:.2f} 天)"
    )
    print(
        f"  observed  (① 8-worker 全量实跑, Little's Law): "
        f"system-wide={observed['avg_concurrency_system_wide']:.3f}, "
        f"per-worker={observed['avg_concurrency_per_worker']:.3f}"
    )


if __name__ == "__main__":
    main()
