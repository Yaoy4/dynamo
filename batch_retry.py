import json, traceback
from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

MODELS = [
    # large dense, using the LOCAL FP8 config (avoids HF download)
    ("Llama-3.1-70B-FP8 (dense)", "nvidia/Llama-3.1-70B-Instruct-FP8", {"aic_tp_size": 8}),
    # try Llama-4-Scout again with tp=4/moe_ep=4 to see if it's a shape/data issue
    ("Llama-4-Scout (MoE) ep4", "meta-llama/Llama-4-Scout-17B-16E-Instruct", {"aic_tp_size": 4, "aic_moe_ep_size": 4}),
    ("Llama-4-Scout (MoE) tp8", "meta-llama/Llama-4-Scout-17B-16E-Instruct", {"aic_tp_size": 8, "aic_moe_tp_size": 8}),
]
BASE = {"engine_type":"trtllm","block_size":32,"num_gpu_blocks":200000,
        "aic_backend":"trtllm","aic_backend_version":"1.3.0rc10","aic_system":"h200_sxm"}

for label, mp, extra in MODELS:
    p = dict(BASE); p["aic_model_path"]=mp; p.update(extra)
    print(f"\n=== {label} ({mp}, {extra}) ===", flush=True)
    try:
        args = MockEngineArgs.from_json(json.dumps(p))
        res = run_synthetic_trace_replay(2048,128,100, extra_engine_args=args, num_workers=1,
                                         replay_mode="offline", replay_concurrency=16, capture_per_request=False)
        agg = (res.aic_latency_breakdown or {}).get("aggregated", {})
        print(f"  OK ops={len(agg)} sum={sum(agg.values()):.2f}%")
        for n,pct in sorted(agg.items(), key=lambda kv:-kv[1])[:8]:
            print(f"    {n:36s} {pct:6.2f}%")
    except Exception as e:
        print(f"  FAILED: {e}")
