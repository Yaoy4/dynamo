#!/usr/bin/env python3
"""验证转换产物是有向无环图（DAG），并量化子 Agent 链的真实并发情况。

对应报告 §4.5「多条子链并发、且都要接回主线，为什么仍然无环？」一节：

  1. 对每个子 Agent 组统计其 spawn 落点（主线索引）与 join 落点（主线索引）
     的大小关系，验证 spawn 索引是否恒严格小于 join 索引（即子链永远被夹在
     两个不同的、先后有序的主锚点之间，不会绕回自己的出发点）。
  2. 统计剔除 day0 时间戳堆积伪影后，同一时刻真实并发的子 Agent 组峰值。
  3. 对转换产物的全部节点、全部 `wait_for` 边做一次 Kahn 拓扑排序：如果图里
     存在环，拓扑排序不可能把所有节点排出来；全部节点被排出即证明 0 个环。

用法：
    python scripts/verify_dag_acyclic.py \
        --raw data/traces.jsonl \
        --output data/semianalysis_cc_traces.agentic_mooncake.jsonl \
        --report-output data/dag_acyclic_verification_report.json
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

DAY_SECONDS = 86400.0


def check_spawn_before_join(raw_path: Path) -> dict[str, int]:
    """统计每个子 Agent 组的 spawn 主线索引是否严格小于 join 主线索引。"""
    n_lt = n_eq = n_gt = n_missing = 0
    with raw_path.open() as f:
        for line in f:
            obj = json.loads(line)
            main_starts: list[float] = []
            children: list[list[tuple[float, float]]] = []
            for item in obj.get("requests", []):
                if item.get("type") == "subagent":
                    children.append(
                        [(inner["t"], inner.get("api_time", 0.0) or 0.0) for inner in item.get("requests", [])]
                    )
                else:
                    main_starts.append(item["t"])
            main_starts.sort()
            for leaves in children:
                if not leaves:
                    continue
                first_start = leaves[0][0]
                last_start, last_api = leaves[-1]
                last_end = last_start + last_api
                spawn_idx = bisect.bisect_right(main_starts, first_start) - 1
                join_idx = bisect.bisect_left(main_starts, last_end)
                has_spawn = spawn_idx >= 0
                has_join = join_idx < len(main_starts)
                if not (has_spawn and has_join):
                    n_missing += 1
                elif spawn_idx < join_idx:
                    n_lt += 1
                elif spawn_idx == join_idx:
                    n_eq += 1
                else:
                    n_gt += 1
    return {
        "spawn_idx_lt_join_idx": n_lt,
        "spawn_idx_eq_join_idx_degenerate": n_eq,
        "spawn_idx_gt_join_idx_would_cycle": n_gt,
        "missing_spawn_or_join": n_missing,
    }


def compute_peak_concurrency(raw_path: Path) -> int:
    """剔除 day0 堆积伪影后，同一时刻并发子 Agent 组数的峰值（sweep-line）。"""
    events: list[tuple[float, int]] = []
    with raw_path.open() as f:
        for line in f:
            obj = json.loads(line)
            for item in obj.get("requests", []):
                if item.get("type") != "subagent":
                    continue
                leaves = item.get("requests", [])
                if not leaves:
                    continue
                start = leaves[0]["t"]
                if start < DAY_SECONDS:
                    continue  # day0 时间戳堆积伪影，见第 3.3 节
                last = leaves[-1]
                end = last["t"] + (last.get("api_time", 0.0) or 0.0)
                events.append((start, 1))
                events.append((end, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def check_full_graph_acyclic(output_path: Path) -> dict[str, Any]:
    """对全图做 Kahn 拓扑排序，验证 0 个环。"""
    rows = []
    with output_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    id_to_idx = {r["request_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    indeg = [0] * n
    adj: dict[int, list[int]] = defaultdict(list)
    n_edges = 0
    for i, r in enumerate(rows):
        for dep in r.get("wait_for", []):
            j = id_to_idx.get(dep)
            if j is None:
                continue
            adj[j].append(i)
            indeg[i] += 1
            n_edges += 1

    indeg_work = indeg[:]
    queue = deque(i for i in range(n) if indeg_work[i] == 0)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in adj[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                queue.append(v)

    return {
        "nodes": n,
        "edges": n_edges,
        "nodes_ordered_by_topo_sort": visited,
        "is_acyclic": visited == n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "spawn_join_ordering": check_spawn_before_join(args.raw),
        "peak_concurrent_subagent_groups_day1plus": compute_peak_concurrency(args.raw),
        "full_graph_topological_sort": check_full_graph_acyclic(args.output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
