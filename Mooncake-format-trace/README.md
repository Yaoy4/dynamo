<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `report/trace_distributions.py` — DynoSim offline replay distribution report

Renders a 2x2 summary of an offline replay: input/output sequence length,
per-request cache-hit rate, and AIC predicted latency share by operator.

The first three panels read token and cache information from the report's
embedded `per_request` records. The last panel reads `aic_operator_breakdown`,
falling back to the aggregated `aic_latency_breakdown` when detailed operator
profiling is unavailable. One complete offline report JSON produces one image.

## End-to-end pipeline

```
                    ┌────────────────────┐        ┌──────────────────────────┐
  trace file /      │   dynamo.replay     │   →    │ report/trace_             │
  synthetic config  │   --per-request-    │        │ distributions.py          │
                    │   jsonl <path>      │        │ --report-json <path>       │
                    │   --report-json     │        │                           │
                    └────────────────────┘        └──────────────────────────┘
                                                              │
                                                              ▼
                                                     trace_distributions.png
                                                       (4-panel histogram)
```

### Step 1 — build the DynoSim extension

The Rust `aic-forward-pass` feature must be compiled in:

```bash
cd lib/bindings/python
source ../../../.venv-demo/bin/activate   # or your own venv with maturin installed

export PATH=/tmp/protoc/bin:$PATH
export PROTOC=/tmp/protoc/bin/protoc
export LIBCLANG_PATH=<venv>/lib/python3.13/site-packages/clang/native
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/14/include -I/usr/include/x86_64-linux-gnu -I/usr/include"

maturin develop --release --features aic-forward-pass
```

The `LIBCLANG_PATH` / `BINDGEN_EXTRA_CLANG_ARGS` env vars work around
`nixl-sys`'s `bindgen` build step not finding a system libclang or the libc
headers on a bare Debian box. Skip them if your environment already has a
working `libclang.so` and `clang` on `PATH`.

Verify the build:

```bash
python3 -c "import dynamo._core as c; print(c.__file__)"
```

### Step 2 — capture a replay with `--per-request-jsonl`

```bash
python3 -m dynamo.replay <trace_file_or_flags...> \
    --replay-mode offline \
    --num-workers 2 \
  --per-request-jsonl report/replay.jsonl \
  --report-json report/replay_report.json
```

Works with any `--trace-format` (`mooncake`, `mooncake-delta`, `dynamo`,
`agentic_mooncake`, `applied_compute_agentic`) or the built-in synthetic
generator (`--input-tokens`/`--output-tokens`/`--request-count`/
`--turns-per-session`).

### Step 3 — render the report

```bash
python3 report/trace_distributions.py \
    --report-json report/conversation_500_report.json \
    --block-size 512
```

`--output` defaults to `trace_distributions.png` **inside this `report/`
folder** (resolved from the script's own path, not the current working
directory), so the command above works the same run from any directory.
Pass `--output` explicitly to write somewhere else, e.g. a per-run name:

```bash
OPERATOR_RESULTS=/mnt/nfs02/users/tjiang/Gitrepo/applications.aisoc.systems.aic/intel_xe/docs/dynosim_intel_aic_integration/scripts/real_sim_validation/real_sim_full_matrix_results.json

python3 report/trace_distributions.py \
    --report-json report/conversation_500_report.json \
  --operator-results-json "$OPERATOR_RESULTS" \
  --operator-dataset conversation \
    --output report/trace_distributions_conversation_500.png
```

The four checked report/image pairs are `conversation_500`, `mooncake_500`,
`synthetic_500`, and `toolagent_500`, all under `report/`.

The external matrix path reads
`qwen3_1_7b_dense.datasets.<dataset>.breakdown_pct` directly. If it is omitted,
the script falls back to `aic_operator_breakdown` or `aic_latency_breakdown`
embedded in the offline report.

## The four panels and their source fields

| # | Panel | Source field(s) | Notes |
|---|---|---|---|
| 1 | Input sequence length per request | `input_length` | Direct. |
| 2 | Output sequence length per request | `output_length` | Direct. |
| 3 | Per-request cache hit rate | `reused_input_tokens`, `input_length`, `--block-size` | Block-level: `hits = reused_input_tokens // block_size`, `total = ceil(input_length / block_size)`. |
| 4 | AIC predicted latency by operator | `datasets.<name>.breakdown_pct`, `aic_operator_breakdown.op_summary_across_phases`, or `aic_latency_breakdown` | The full dynamic matrix is preferred when supplied; embedded report fields are fallbacks. |

## Known limitations

- The cache-hit-rate panel's `--block-size` must match whatever
  `--trace-block-size` (or the engine's block size) the replay actually used;
  the script has no way to read this back out of the report automatically.
- Without `--operator-results-json`, the operator panel needs either
  `aic_operator_breakdown` or `aic_latency_breakdown` in the report JSON.
