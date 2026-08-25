import json
import traceback

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

# (label, model_path, extra params dict) — architectures span dense / MoE / MLA+MoE
MODELS = [
    ("Qwen3-8B (dense)",        "meta-llama/Meta-Llama-3.1-8B",         {"aic_tp_size": 1}),
    ("Qwen3-32B (dense)",       "Qwen/Qwen3-32B",                        {"aic_tp_size": 4}),
    ("Llama-3.1-70B (dense)",   "meta-llama/Llama-3.1-70B",              {"aic_tp_size": 8}),
    ("Llama-4-Scout (MoE)",     "meta-llama/Llama-4-Scout-17B-16E-Instruct", {"aic_tp_size": 8, "aic_moe_ep_size": 8}),
    ("Qwen3-235B-A22B (MoE)",   "Qwen/Qwen3-235B-A22B",                  {"aic_tp_size": 8, "aic_moe_ep_size": 8}),
    ("DeepSeek-V3 (MLA+MoE)",   "deepseek-ai/DeepSeek-V3",               {"aic_tp_size": 8, "aic_moe_ep_size": 8}),
]

BASE = {
    "engine_type": "trtllm",
    "block_size": 32,
    "num_gpu_blocks": 200000,
    "aic_backend": "trtllm",
    "aic_backend_version": "1.3.0rc10",
    "aic_system": "h200_sxm",
}

results = {}
for label, model_path, extra in MODELS:
    payload = dict(BASE)
    payload["aic_model_path"] = model_path
    payload.update(extra)
    print(f"\n{'='*70}\n### {label}  ({model_path}, {extra})\n{'='*70}", flush=True)
    try:
        args = MockEngineArgs.from_json(json.dumps(payload))
        res = run_synthetic_trace_replay(
            2048, 128, 100,
            extra_engine_args=args,
            num_workers=1,
            replay_mode="offline",
            replay_concurrency=16,
            capture_per_request=False,
        )
        bd = res.aic_latency_breakdown or {}
        agg = bd.get("aggregated", {})
        results[label] = agg
        total = sum(agg.values())
        moe_ops = sorted([n for n in agg if any(k in n for k in ("moe", "router", "dispatch"))])
        mla_ops = sorted([n for n in agg if "mla" in n.lower()])
        print(f"  ops={len(agg)}  sum={total:.2f}%")
        for name, pct in sorted(agg.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {name:36s} {pct:6.2f}%")
        if moe_ops:
            print(f"  MoE ops: {moe_ops}")
        if mla_ops:
            print(f"  MLA ops: {mla_ops}")
    except Exception as e:
        results[label] = {"__error__": str(e)}
        print(f"  FAILED: {e}")
        traceback.print_exc()

# machine-readable dump for the report
with open("/mnt/nfs02/users/tjiang/batch_models_breakdown.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n\n=== saved to /mnt/nfs02/users/tjiang/batch_models_breakdown.json ===")
