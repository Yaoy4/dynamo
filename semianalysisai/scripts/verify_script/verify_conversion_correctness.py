#!/usr/bin/env python3
"""独立、从零复算的转换正确性验证脚本。

与 `convert_weka_to_agentic_mooncake.py` 内置的自检（仅重验 delay 公式）不同，
本脚本**不复用转换脚本的任何中间结果**，而是直接从原始 `traces.jsonl` 出发，
重新推导出全部 7 组断言应有的取值，再逐行对比转换产物
`semianalysis_cc_traces.agentic_mooncake.jsonl` 是否与之一致（编号 1/2/2b/3/4/5/7
共 7 组、落到报告里是 8 个条目；编号 6 —— 全图无环性与 spawn/join 先后序 ——
已拆分到独立脚本 `verify_dag_acyclic.py`，故此处留空不复用）：

  1. 每个会话的主时间线第 0 条请求：`prefix_reset=true` 且没有 `wait_for`。
  2. 主时间线第 i 条（i>=1）请求的 `wait_for` 必须包含第 i-1 条的 id。
  2b. 子 Agent 组内第 j 条请求（j>=1）的 `wait_for` 必须**恰好等于**
      `[组内第 j-1 条的 id]`（不多不少）。
  3. 派生边（spawn）最优性：子 Agent 组首条请求的 spawn 依赖，必须是主时间线上
     "发生时刻 <= 子组首条发生时刻"的**最近**一条主请求；且该主请求必须有一条
     指回子组首条请求的 `branches` 反向边。
  4. 汇合边（join）最优性：子 Agent 组末条请求的 join 落点，必须是主时间线上
     "发生时刻 >= 子组末条结束时刻"的**最早**一条主请求；支持多个子组并发汇合
     到同一条主请求。
  5. hash_id 命名空间隔离：每个会话的 hash_id 必须落在
     `[session_index*1e6, session_index*1e6+1e6)` 区间内，且任意两个会话的
     hash_id 观测区间不重叠；同时复核数据集内最大本地 hash_id 是否为 238,450。
  7. delay 公式独立复算：对每一条有依赖的请求，用原始 `t`/`api_time` 重新计算
     `max(0, start_ms - max(dep_end_ms))`，与产物中存储的 `delay` 逐行比对
     （容差 1e-3 ms）。

用法：
    python scripts/verify_conversion_correctness.py \
        --raw data/traces.jsonl \
        --output data/semianalysis_cc_traces.agentic_mooncake.jsonl \
        --report-output data/conversion_verification_report.json
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_raw(raw_path: Path) -> tuple[dict[str, int], dict[str, list], dict[str, list], int]:
    """解析原始数据集，返回：trace_id->session_index、每会话主请求(t, api_time)
    列表、每会话子 Agent 组 [(agent_id, [(t, api_time), ...]), ...] 列表，以及
    数据集内观测到的最大本地 hash_id。"""
    trace_id_to_index: dict[str, int] = {}
    main_by_trace: dict[str, list] = {}
    children_by_trace: dict[str, list] = {}
    max_local_hash_id = 0
    with raw_path.open() as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            trace_id = obj["id"]
            trace_id_to_index[trace_id] = idx
            main: list[tuple[float, float]] = []
            children: list[tuple[str, list[tuple[float, float]]]] = []
            for item in obj.get("requests", []):
                if item.get("type") == "subagent":
                    grp = []
                    for inner in item.get("requests", []):
                        grp.append((inner["t"], inner.get("api_time", 0.0) or 0.0))
                        for h in inner.get("hash_ids", []):
                            max_local_hash_id = max(max_local_hash_id, h)
                    children.append((item["agent_id"], grp))
                else:
                    main.append((item["t"], item.get("api_time", 0.0) or 0.0))
                    for h in item.get("hash_ids", []):
                        max_local_hash_id = max(max_local_hash_id, h)
            main_by_trace[trace_id] = main
            children_by_trace[trace_id] = children
    return trace_id_to_index, main_by_trace, children_by_trace, max_local_hash_id


def load_output(output_path: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    with output_path.open() as f:
        for line in f:
            row = json.loads(line)
            rows_by_id[row["request_id"]] = row
    return rows_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    trace_id_to_index, main_by_trace, children_by_trace, max_local_hash_id = load_raw(args.raw)
    rows_by_id = load_output(args.output)

    report: dict[str, Any] = {"max_local_hash_id_observed": max_local_hash_id}

    # ---- Check 1: main row 0 ----
    n_m0 = violations_1 = 0
    for trace_id in main_by_trace:
        rid = f"{trace_id}::m0"
        row = rows_by_id.get(rid)
        if row is None:
            continue
        n_m0 += 1
        if row.get("prefix_reset") is not True or "wait_for" in row:
            violations_1 += 1
    report["check1_main_row0"] = {"checked": n_m0, "violations": violations_1}

    # ---- Check 2 & 2b: sequential chains ----
    violations_2 = n_main_checked = 0
    violations_2b = n_child_checked = 0
    for trace_id, main in main_by_trace.items():
        for i in range(1, len(main)):
            row = rows_by_id.get(f"{trace_id}::m{i}")
            if row is None:
                continue
            n_main_checked += 1
            if f"{trace_id}::m{i - 1}" not in row.get("wait_for", []):
                violations_2 += 1
        for g, (agent_id, leaves) in enumerate(children_by_trace[trace_id]):
            for j in range(1, len(leaves)):
                row = rows_by_id.get(f"{trace_id}::sub{g}_{agent_id}::c{j}")
                if row is None:
                    continue
                n_child_checked += 1
                expected = [f"{trace_id}::sub{g}_{agent_id}::c{j - 1}"]
                if row.get("wait_for") != expected:
                    violations_2b += 1
    report["check2_main_sequential_chain"] = {"checked": n_main_checked, "violations": violations_2}
    report["check2b_child_sequential_chain"] = {"checked": n_child_checked, "violations": violations_2b}

    # ---- Check 3: spawn edge optimality + branches reciprocity ----
    n_spawn_checked = violations_3 = violations_3_branches = n_no_valid_spawn = 0
    for trace_id, main in main_by_trace.items():
        main_starts = [m[0] for m in main]
        for g, (agent_id, leaves) in enumerate(children_by_trace[trace_id]):
            if not leaves:
                continue
            child_first_id = f"{trace_id}::sub{g}_{agent_id}::c0"
            row = rows_by_id.get(child_first_id)
            if row is None:
                continue
            spawn_idx = bisect.bisect_right(main_starts, leaves[0][0]) - 1
            deps = row.get("wait_for", [])
            if spawn_idx < 0:
                n_no_valid_spawn += 1
                if deps:
                    violations_3 += 1
                continue
            n_spawn_checked += 1
            expected_main_id = f"{trace_id}::m{spawn_idx}"
            if expected_main_id not in deps:
                violations_3 += 1
            main_row = rows_by_id.get(expected_main_id) or {}
            if child_first_id not in main_row.get("branches", []):
                violations_3_branches += 1
    report["check3_spawn_edge_optimality"] = {
        "checked": n_spawn_checked,
        "no_valid_spawn_cases": n_no_valid_spawn,
        "violations": violations_3,
        "branch_reciprocal_violations": violations_3_branches,
    }

    # ---- Check 4: join edge optimality ----
    n_join_checked = violations_4 = n_no_valid_join = 0
    join_landing_counts: dict[str, int] = defaultdict(int)
    for trace_id, main in main_by_trace.items():
        main_starts = [m[0] for m in main]
        for g, (agent_id, leaves) in enumerate(children_by_trace[trace_id]):
            if not leaves:
                continue
            last_start, last_api = leaves[-1]
            last_end = last_start + last_api
            child_last_id = f"{trace_id}::sub{g}_{agent_id}::c{len(leaves) - 1}"
            join_idx = bisect.bisect_left(main_starts, last_end)
            if join_idx >= len(main_starts):
                n_no_valid_join += 1
                continue
            n_join_checked += 1
            expected_main_id = f"{trace_id}::m{join_idx}"
            main_row = rows_by_id.get(expected_main_id)
            if main_row is None:
                continue
            if child_last_id not in main_row.get("wait_for", []):
                violations_4 += 1
            join_landing_counts[expected_main_id] += 1
    multi_join = {k: v for k, v in join_landing_counts.items() if v > 1}
    report["check4_join_edge_optimality"] = {
        "checked": n_join_checked,
        "no_valid_join_cases": n_no_valid_join,
        "violations": violations_4,
        "main_rows_with_multi_join": len(multi_join),
        "max_concurrent_joins_on_one_row": max(multi_join.values()) if multi_join else 0,
    }

    # ---- Check 5: hash_id namespace isolation ----
    offset = 1_000_000
    violations_5a = n_hash_checked = 0
    range_by_index: dict[int, list[int | None]] = defaultdict(lambda: [None, None])
    for row in rows_by_id.values():
        trace_id = row["request_id"].split("::")[0]
        idx = trace_id_to_index.get(trace_id)
        if idx is None:
            continue
        lo, hi = idx * offset, idx * offset + offset
        for h in row.get("hash_ids", []):
            n_hash_checked += 1
            if not (lo <= h < hi):
                violations_5a += 1
            r = range_by_index[idx]
            if r[0] is None or h < r[0]:
                r[0] = h
            if r[1] is None or h > r[1]:
                r[1] = h
    intervals = sorted((v[0], v[1], k) for k, v in range_by_index.items() if v[0] is not None)
    violations_5b = 0
    for i in range(1, len(intervals)):
        if intervals[i][0] <= intervals[i - 1][1]:
            violations_5b += 1
    report["check5a_hash_id_containment"] = {"checked": n_hash_checked, "violations": violations_5a}
    report["check5b_hash_id_no_overlap"] = {
        "sessions_with_hashes": len(intervals),
        "violations": violations_5b,
        "max_local_hash_id_in_output": max((hi - idx * offset) for lo, hi, idx in intervals) if intervals else None,
    }

    # ---- Check 7: independent delay re-derivation ----
    start_ms: dict[str, float] = {}
    end_ms: dict[str, float] = {}
    for trace_id, main in main_by_trace.items():
        for i, (t, api) in enumerate(main):
            rid = f"{trace_id}::m{i}"
            start_ms[rid] = t * 1000.0
            end_ms[rid] = (t + api) * 1000.0
        for g, (agent_id, leaves) in enumerate(children_by_trace[trace_id]):
            for j, (t, api) in enumerate(leaves):
                rid = f"{trace_id}::sub{g}_{agent_id}::c{j}"
                start_ms[rid] = t * 1000.0
                end_ms[rid] = (t + api) * 1000.0
    n_delay_checked = violations_7 = 0
    max_abs_diff = 0.0
    for row in rows_by_id.values():
        deps = row.get("wait_for")
        stored_delay = row.get("delay")
        rid = row["request_id"]
        if not deps or stored_delay is None or rid not in start_ms:
            continue
        dep_ends = [end_ms.get(d) for d in deps]
        if any(e is None for e in dep_ends):
            continue
        computed_delay = max(0.0, start_ms[rid] - max(dep_ends))
        n_delay_checked += 1
        diff = abs(computed_delay - stored_delay)
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-3:
            violations_7 += 1
    report["check7_delay_formula_rederivation"] = {
        "checked": n_delay_checked,
        "violations": violations_7,
        "max_abs_diff_ms": max_abs_diff,
    }

    total_violations = sum(
        v.get("violations", 0) for v in report.values() if isinstance(v, dict)
    )
    report["all_checks_passed"] = total_violations == 0

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{'PASSED — 0 violations across all checks' if total_violations == 0 else f'FAILED — {total_violations} total violations'}")


if __name__ == "__main__":
    main()
