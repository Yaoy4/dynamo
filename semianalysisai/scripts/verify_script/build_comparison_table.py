#!/usr/bin/env python3
"""Build a consolidated comparison table across all DynoSim scenario reports
for the semianalysisai/cc-traces-weka-062126 replay study."""
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

SCENARIOS = [
    ("S1: round_robin, 8 workers (baseline)", "scenario1_full_round_robin.json"),
    ("S2: kv_router, 8 workers", "scenario2_full_kv_router.json"),
    ("S3: kv_router, 4 workers", "scenario4_full_kv_router_w4.json"),
    ("S4: kv_router, 16 workers", "scenario5_full_kv_router_w16.json"),
    ("S5: kv_router, 8 workers, AIC (Qwen3-32B-FP8, h200_sxm, tp2)", "scenario7_full_aic_qwen32b.json"),
    ("(subset, n=100) default kv_router, 8 workers", "scenario6b_subset100_default_kv_router.json"),
    ("(subset, n=100) AIC validation", "scenario6_subset100_aic_qwen32b.json"),
]

FIELDS = [
    "num_requests",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "prefix_cache_reused_ratio",
    "first_admission_prefix_cache_reused_ratio",
    "output_throughput_tok_s",
    "request_throughput_rps",
    "gpu_hours",
    "wall_time_ms",
]


def main():
    table = {}
    for label, filename in SCENARIOS:
        path = REPORTS_DIR / filename
        if not path.exists():
            continue
        summary = json.loads(path.read_text())["summary"]
        table[label] = {k: summary.get(k) for k in FIELDS}

    print(json.dumps(table, indent=2))
    (REPORTS_DIR / "consolidated_comparison.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
