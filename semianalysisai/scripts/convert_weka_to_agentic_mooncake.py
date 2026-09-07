#!/usr/bin/env python3
"""Convert semianalysisai/cc-traces-weka-062126 (WekaTrace) into DynoSim's
`agentic_mooncake` trace format.

Source schema (one JSON object per line in traces.jsonl == one Claude Code
session):

    {
      "id": "<trace id>",
      "models": ["claude-...", ...],
      "block_size": 64,
      "hash_id_scope": "local",
      "requests": [
        {"t": <sec>, "model": "...", "in": <int>, "out": <int>,
         "hash_ids": [...], "api_time": <sec>, "type": "s"|"n",
         "ttft": <sec>, "think_time": <sec>},
        ...
        {"type": "subagent", "agent_id": "...", "subagent_type": "Subagent",
         "duration_ms": <int>, "total_tokens": <int>, "status": "completed",
         "models": [...], "requests": [ <same leaf-request shape> ... ]},
        ...
      ]
    }

`requests` is the *main agent* timeline in chronological order (verified
strictly non-decreasing `t`). Items with `type == "subagent"` are inline
sub-agent groups: they carry their own chronologically ordered inner
`requests` list (one level of nesting only -- verified empirically, no
subagent ever spawns another subagent in this dataset).

This script lowers that two-level structure into flat
`dynamo_data_gen::AgenticMooncakeRow`-shaped JSON rows (see
`lib/data-gen/src/mooncake.rs` and the lowering reference algorithm in
`lib/data-gen/src/request_trace/agentic.rs` in the dynamo-yaoyao repo), which
`python -m dynamo.replay --trace-format agentic_mooncake` consumes directly:

  - The main-agent timeline becomes one session (`session_id = trace_id`)
    with a strict sequential `wait_for` chain (turn i waits for turn i-1).
  - Each sub-agent group becomes its own child session
    (`session_id = f"{trace_id}::{agent_id}"`, `parent_session_id =
    trace_id`) with its own sequential `wait_for` chain.
  - The child session's first row waits for the *last main-agent row whose
    recorded start time is <= the child's own start time* (the request that
    was "in flight" when the tool call spawning the sub-agent was issued),
    mirroring Dynamo's own `latest_request_starting_before` reference
    algorithm. This also records a `branches` entry on that spawning row.
  - The first main-agent row whose recorded start time is >= the child
    session's last row's end time additionally waits for that child's last
    row (`first_request_starting_after` in the reference algorithm), which
    is exactly how multiple concurrently spawned sub-agent groups all gate
    the next main-agent turn until every branch has returned.
  - `delay` for every dependent row is computed as
    `max(0, this_row_start_ms - max(end_ms of its wait_for deps))`, i.e. the
    *observed* historical gap, so replaying at the *same* engine speed as
    the original capture reproduces the original timestamps exactly, and
    replaying at a different (simulated) speed shifts downstream timing
    causally instead of naively reusing stale absolute timestamps.

Because `hash_id_scope` is "local" (hash ids are only comparable *within*
one trace/session), every hash id is remapped to a globally unique id
(`session_index * OFFSET + local_id`) before being written out, so that
replaying many sessions together in one DynoSim run cannot produce false
cross-session prefix-cache hits.

Usage:
    python3 convert_weka_to_agentic_mooncake.py \\
        --input data/traces.jsonl \\
        --output data/semianalysis_cc_traces.agentic_mooncake.jsonl \\
        [--limit-sessions 20] [--stats-output data/conversion_stats.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Local hash ids observed in the dataset top out at 238,450 (see
# exploratory analysis in the report); 1,000,000 leaves a wide, easy-to-audit
# margin between any two sessions' namespaced hash-id ranges.
HASH_ID_OFFSET_MULTIPLIER = 1_000_000


@dataclass
class Leaf:
    """One flattened model-call request (main-agent turn or sub-agent inner call)."""

    t: float
    api_time: float
    in_len: int
    out_len: int
    hash_ids: list[int]
    model: Optional[str]

    @property
    def start_ms(self) -> float:
        return self.t * 1000.0

    @property
    def end_ms(self) -> float:
        return (self.t + self.api_time) * 1000.0


@dataclass
class ChildGroup:
    agent_id: str
    subagent_type: Optional[str]
    leaves: list[Leaf] = field(default_factory=list)


def parse_leaf(raw: dict) -> Leaf:
    return Leaf(
        t=float(raw["t"]),
        api_time=float(raw.get("api_time", 0.0) or 0.0),
        in_len=int(raw["in"]),
        out_len=int(raw["out"]),
        hash_ids=[int(h) for h in raw.get("hash_ids", [])],
        model=raw.get("model"),
    )


def flatten_session(obj: dict) -> tuple[list[Leaf], list[ChildGroup]]:
    """Split one session's `requests` array into the main timeline and its
    inline sub-agent groups. Raises if a sub-agent nests another sub-agent
    (not observed in this dataset; fail loudly rather than silently drop
    data if that assumption is ever violated)."""
    main_leaves: list[Leaf] = []
    child_groups: list[ChildGroup] = []
    for item in obj.get("requests", []):
        if item.get("type") == "subagent":
            group = ChildGroup(
                agent_id=item["agent_id"],
                subagent_type=item.get("subagent_type"),
            )
            for inner in item.get("requests", []):
                if inner.get("type") == "subagent":
                    raise ValueError(
                        f"session {obj.get('id')}: nested sub-agent found "
                        "(unsupported, only one level of nesting expected)"
                    )
                group.leaves.append(parse_leaf(inner))
            child_groups.append(group)
        else:
            main_leaves.append(parse_leaf(item))
    return main_leaves, child_groups


def remap_hash_ids(hash_ids: list[int], session_index: int) -> list[int]:
    offset = session_index * HASH_ID_OFFSET_MULTIPLIER
    return [offset + h for h in hash_ids]


class RowEmitter:
    """Accumulates wait_for/branches edges for one session and emits
    AgenticMooncakeRow-shaped dicts once the DAG is fully resolved."""

    def __init__(self, trace_id: str, session_index: int):
        self.trace_id = trace_id
        self.session_index = session_index
        self.wait_for: dict[str, list[str]] = {}
        self.branches: dict[str, list[str]] = {}
        self.rows_meta: list[dict] = []  # ordered list of row-building metadata

    def _add_wait(self, request_id: str, dep_id: str) -> None:
        deps = self.wait_for.setdefault(request_id, [])
        if dep_id not in deps:
            deps.append(dep_id)

    def _add_branch(self, request_id: str, child_id: str) -> None:
        children = self.branches.setdefault(request_id, [])
        if child_id not in children:
            children.append(child_id)

    def build(self, main_leaves: list[Leaf], child_groups: list[ChildGroup]) -> list[dict]:
        trace_id = self.trace_id
        main_ids = [f"{trace_id}::m{i}" for i in range(len(main_leaves))]
        main_starts = [leaf.start_ms for leaf in main_leaves]

        # Sequential chain within the main-agent timeline.
        for i in range(1, len(main_leaves)):
            self._add_wait(main_ids[i], main_ids[i - 1])

        child_id_lists: list[list[str]] = []
        for g, group in enumerate(child_groups):
            ids = [f"{trace_id}::sub{g}_{group.agent_id}::c{j}" for j in range(len(group.leaves))]
            child_id_lists.append(ids)
            # Sequential chain within this sub-agent's own timeline.
            for j in range(1, len(group.leaves)):
                self._add_wait(ids[j], ids[j - 1])

        # Cross-session spawn (child's first row depends on the main row
        # in flight when the tool call fired) and join (the next main row
        # after the child returns waits for it) edges.
        for g, group in enumerate(child_groups):
            if not group.leaves or not main_leaves:
                continue
            ids = child_id_lists[g]
            first_start = group.leaves[0].start_ms
            spawn_idx = bisect.bisect_right(main_starts, first_start) - 1
            if spawn_idx >= 0:
                self._add_wait(ids[0], main_ids[spawn_idx])
                self._add_branch(main_ids[spawn_idx], ids[0])

            last_end = group.leaves[-1].end_ms
            join_idx = bisect.bisect_left(main_starts, last_end)
            if join_idx < len(main_leaves):
                self._add_wait(main_ids[join_idx], ids[-1])

        # Emit rows: main timeline first, then each child group.
        rows: list[dict] = []
        end_ms_by_id: dict[str, float] = {}
        for i, leaf in enumerate(main_leaves):
            end_ms_by_id[main_ids[i]] = leaf.end_ms
        for g, group in enumerate(child_groups):
            for j, leaf in enumerate(group.leaves):
                end_ms_by_id[child_id_lists[g][j]] = leaf.end_ms

        def make_row(request_id: str, session_id: str, parent_session_id: Optional[str],
                     leaf: Leaf, is_first_in_session: bool, request_kind: str) -> dict:
            deps = self.wait_for.get(request_id, [])
            delay = None
            if deps:
                dep_end = max(end_ms_by_id[d] for d in deps)
                delay = max(0.0, leaf.start_ms - dep_end)
            row = {
                "request_id": request_id,
                "session_id": session_id,
                "input_length": leaf.in_len,
                "output_length": leaf.out_len,
                "hash_ids": remap_hash_ids(leaf.hash_ids, self.session_index),
                "timestamp": leaf.start_ms,
                "prefix_reset": is_first_in_session,
                "request_kind": request_kind,
            }
            if parent_session_id is not None:
                row["parent_session_id"] = parent_session_id
            if deps:
                row["wait_for"] = deps
                row["delay"] = delay
            branches = self.branches.get(request_id)
            if branches:
                row["branches"] = branches
            return row

        for i, leaf in enumerate(main_leaves):
            rows.append(
                make_row(main_ids[i], trace_id, None, leaf, i == 0, "foreground")
            )
        for g, group in enumerate(child_groups):
            child_session_id = f"{trace_id}::{group.agent_id}"
            for j, leaf in enumerate(group.leaves):
                rows.append(
                    make_row(
                        child_id_lists[g][j],
                        child_session_id,
                        trace_id,
                        leaf,
                        j == 0,
                        "agent",
                    )
                )
        return rows


def convert(
    input_path: Path,
    output_path: Path,
    limit_sessions: Optional[int] = None,
    start_session_index: int = 0,
) -> dict:
    stats = {
        "sessions": 0,
        "main_rows": 0,
        "child_rows": 0,
        "total_rows": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "child_groups": 0,
        "rows_with_dependencies": 0,
        "rows_with_clamped_overlap": 0,
        "max_clamped_overlap_ms": 0.0,
    }

    with input_path.open("r") as fin, output_path.open("w") as fout:
        for session_index, line in enumerate(fin):
            if session_index < start_session_index:
                continue
            if limit_sessions is not None and stats["sessions"] >= limit_sessions:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            trace_id = obj["id"]
            main_leaves, child_groups = flatten_session(obj)
            emitter = RowEmitter(trace_id=trace_id, session_index=session_index)
            rows = emitter.build(main_leaves, child_groups)

            # Self-check: for every row with wait_for, confirm
            # `max(dependency end_ms) + delay == recorded start_ms` whenever
            # the observed historical gap was >= 0 (delay is defined exactly
            # this way, so this always holds by construction). When the
            # historical gap was *negative* -- the row actually started
            # before its dependency finished, which happens because a single
            # recorded "leaf" only covers one model call's own generation
            # time while the real agent harness pipelines/parallelizes
            # around it -- `delay` is clamped to 0 (`saturating_sub(...).
            # max(0)`, exactly mirroring the Rust reference algorithm at
            # lib/data-gen/src/request_trace/agentic.rs:279). That clamping
            # is expected upstream behavior, not a converter bug; we simply
            # track how often/how much it happens for the report.
            id_to_leaf: dict[str, Leaf] = {}
            for i, leaf in enumerate(main_leaves):
                id_to_leaf[f"{trace_id}::m{i}"] = leaf
            for g, group in enumerate(child_groups):
                for j, leaf in enumerate(group.leaves):
                    id_to_leaf[f"{trace_id}::sub{g}_{group.agent_id}::c{j}"] = leaf
            for row in rows:
                deps = row.get("wait_for")
                if not deps:
                    continue
                stats["rows_with_dependencies"] += 1
                dep_end = max(id_to_leaf[d].end_ms for d in deps)
                observed_gap = row["timestamp"] - dep_end
                if observed_gap < 0:
                    stats["rows_with_clamped_overlap"] += 1
                    overlap = -observed_gap
                    if overlap > stats["max_clamped_overlap_ms"]:
                        stats["max_clamped_overlap_ms"] = overlap
                else:
                    reconstructed = dep_end + row["delay"]
                    err = abs(reconstructed - row["timestamp"])
                    assert err < 1e-6, (
                        f"non-clamped reconstruction mismatch for {row['request_id']}: "
                        f"err={err}"
                    )

            for row in rows:
                fout.write(json.dumps(row) + "\n")

            stats["sessions"] += 1
            stats["main_rows"] += len(main_leaves)
            stats["child_rows"] += sum(len(g.leaves) for g in child_groups)
            stats["child_groups"] += len(child_groups)
            stats["total_input_tokens"] += sum(l.in_len for l in main_leaves)
            stats["total_input_tokens"] += sum(
                l.in_len for g in child_groups for l in g.leaves
            )
            stats["total_output_tokens"] += sum(l.out_len for l in main_leaves)
            stats["total_output_tokens"] += sum(
                l.out_len for g in child_groups for l in g.leaves
            )

    stats["total_rows"] = stats["main_rows"] + stats["child_rows"]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--start-session-index", type=int, default=0)
    parser.add_argument("--stats-output", type=Path, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = convert(
        args.input,
        args.output,
        limit_sessions=args.limit_sessions,
        start_session_index=args.start_session_index,
    )
    stats["input"] = str(args.input)
    stats["output"] = str(args.output)
    print(json.dumps(stats, indent=2))
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
