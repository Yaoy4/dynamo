import json

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
BACKEND_VERSION = "0.19.0"


def main():
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 200000,
        "speedup_ratio": 1.0,
        "aic_backend": BACKEND,
        "aic_system": SYSTEM,
        "aic_backend_version": BACKEND_VERSION,
        "aic_tp_size": 1,
        "aic_model_path": MODEL,
    }
    args = MockEngineArgs.from_json(json.dumps(payload))

    result = run_synthetic_trace_replay(
        128,
        64,
        16,
        extra_engine_args=args,
        num_workers=1,
        replay_mode="offline",
        replay_concurrency=4,
        capture_per_request=False,
    )

    print("=== summary keys (subset) ===")
    summary = result.summary
    for k in ("mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s", "num_requests"):
        if k in summary:
            print(f"{k}: {summary[k]}")

    print("\n=== aic_latency_breakdown ===")
    breakdown = result.aic_latency_breakdown
    if breakdown is None:
        print("None (no AIC backend recorded any breakdown)")
    else:
        for role, ops in breakdown.items():
            print(f"-- role: {role} --")
            total_pct = 0.0
            for name, pct in sorted(ops.items(), key=lambda kv: -kv[1]):
                print(f"  {name:35s} {pct:6.2f}%")
                total_pct += pct
            print(f"  {'(sum)':35s} {total_pct:6.2f}%")


if __name__ == "__main__":
    main()
