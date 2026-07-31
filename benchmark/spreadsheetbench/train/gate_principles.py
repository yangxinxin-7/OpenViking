#!/usr/bin/env python3
"""Score gate for OpenViking principles candidates, using SpreadsheetBench.

Drives the seed/pending/promote/reject cycle exposed by the OV server:

    # one-time: install the hand-written seed experience document (SkillOpt's
    # ungated initial skill equivalent) as default.md v0
    python gate_principles.py seed [--force]

    # one-time: freeze the selection set (stratified 80 tasks from the test split)
    python gate_principles.py make-selection [--size 80] [--seed 42]

    # gate the current pending candidate (no-op when none exists)
    python gate_principles.py run [--k 3] [--concurrency 20]

    # one full SkillOpt-style step, invoked by the orchestrator right after a
    # training batch finished committing: propose from that batch's trajectory
    # memories, then gate the candidate on the selection set
    python gate_principles.py cycle [--k 3] [--concurrency 20]

The gate compares candidate vs baseline PER TASK on the frozen selection set
and promotes only when net wins (rescued - regressed) >= k. Point scores are
too noisy at this sample size; paired comparison is the whole point.

The candidate is injected through VIKINGBOT_PRINCIPLES_OVERRIDE_FILE so the
server-side default.md is never touched during evaluation. The baseline
per-task vector is cached per principles version and reused across gate runs.

Environment:
    OV_API_BASE      OV server base URL (default http://127.0.0.1:1933)
    OV_ACCOUNT/OV_USER  trusted-mode identity headers (default "default")
    OPENVIKING_CONFIG_FILE  must point at the same slot the server runs on
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bot"))

GATE_DIR = REPO / "result" / "spreadsheetbench" / "principles_gate"
SELECTION_PATH = GATE_DIR / "selection_tasks.json"

API_BASE = os.environ.get("OV_API_BASE", "http://127.0.0.1:1933").rstrip("/")
HEADERS = {
    "X-OpenViking-Account": os.environ.get("OV_ACCOUNT", "default"),
    "X-OpenViking-User": os.environ.get("OV_USER", "default"),
}
# Dataset the gate rollouts run on. The selection tasks are drawn from
# GATE_SPLIT of GATE_DOMAIN (SkillOpt-aligned runs: verified_400_v1/selection).
GATE_DOMAIN = os.environ.get("SSB_GATE_DOMAIN", "all_data_912_v0.1")
GATE_SPLIT = os.environ.get("SSB_GATE_SPLIT", "test")


def _api(method: str, path: str, body: dict | None = None) -> dict:
    resp = requests.request(method, f"{API_BASE}{path}", json=body, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json().get("result") or {}


def _case_id(case) -> str:
    """Task id is the last segment of the signature (spreadsheetbench:domain:split:id)."""
    return str(case.task_signature).rsplit(":", 1)[-1]


# ── seed ───────────────────────────────────────────────────────────────────


SEED_PATH = Path(__file__).resolve().parent / "seed_experience.md"


def install_seed(force: bool) -> None:
    """Write the seed experience document as default.md (SkillOpt-style ungated
    starting baseline). Refuses to overwrite an existing document unless --force."""
    existing = _api("GET", "/api/v1/memories/principles").get("content") or ""
    if existing.strip() and not force:
        raise SystemExit("default.md already exists; use --force to overwrite")
    body = SEED_PATH.read_text(encoding="utf-8").strip()
    metadata = {
        "memory_type": "principles",
        "status": "seed",
        "proposal_source": "seed",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": 0,
        "source_trajectories": 0,
    }
    document = (
        body
        + "\n\n<!-- MEMORY_FIELDS\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n-->\n"
    )
    resp = requests.post(
        f"{API_BASE}/api/v1/content/write",
        json={
            "uri": "viking://user/default/memories/principles/default.md",
            "content": document,
            "mode": "replace" if existing.strip() else "create",
        },
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    # Reset the state watermark so a fresh seed starts at v0 with zero consumed
    # trajectories; otherwise a leftover state.json (e.g. from a prior run that
    # promoted to v2 / consumed N trajectories) makes every later propose skip
    # with "0 new trajectories" and silently disables gating.
    state_resp = requests.post(
        f"{API_BASE}/api/v1/content/write",
        json={
            "uri": "viking://user/default/memories/principles/state.json",
            "content": json.dumps({"version": 0, "consolidated_trajectories": 0}),
            "mode": "replace",
        },
        headers=HEADERS,
        timeout=60,
    )
    # state.json may not exist yet on a truly fresh space; ignore not-found.
    if state_resp.status_code not in (200, 404):
        state_resp.raise_for_status()
    print(f"seed installed as default.md v0 ({len(body)} chars) from {SEED_PATH}; state reset")


# ── selection set ──────────────────────────────────────────────────────────


def make_selection(size: int, seed: int, whole_split: bool = False) -> None:
    from benchmark.spreadsheetbench.train.case_loader import SpreadsheetBenchCaseLoader

    cases = SpreadsheetBenchCaseLoader(domain=GATE_DOMAIN, split=GATE_SPLIT).load_cases()
    if whole_split:
        # SkillOpt-aligned mode: the split IS the selection set, no sampling.
        selected_ids = sorted(_case_id(c) for c in cases)
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        SELECTION_PATH.write_text(
            json.dumps(
                {
                    "selection_id": f"ssb-sel-{GATE_DOMAIN}-{GATE_SPLIT}-full-v1",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "task_ids": selected_ids,
                },
                indent=1,
            )
        )
        print(f"selection frozen (whole split): {len(selected_ids)} tasks -> {SELECTION_PATH}")
        return
    by_type: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_type[str(case.metadata.get("instruction_type") or "unknown")].append(_case_id(case))

    rng = random.Random(seed)
    total = sum(len(ids) for ids in by_type.values())
    selected: list[str] = []
    for _itype, ids in sorted(by_type.items()):
        quota = round(size * len(ids) / total)
        selected.extend(rng.sample(sorted(ids), min(quota, len(ids))))
    # Rounding may leave us off by a task or two; top up from the largest pool.
    remaining = [i for ids in by_type.values() for i in ids if i not in set(selected)]
    rng.shuffle(remaining)
    while len(selected) < size and remaining:
        selected.append(remaining.pop())
    selected = selected[:size]

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(
        json.dumps(
            {
                "selection_id": f"ssb-sel-{size}-seed{seed}-v1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "task_ids": sorted(selected),
            },
            indent=1,
        )
    )
    print(f"selection frozen: {len(selected)} tasks -> {SELECTION_PATH}")
    print("NOTE: never regenerate this file; cached baseline vectors key off it.")


# ── rollout ────────────────────────────────────────────────────────────────


async def _run_selection(task_ids: list[str], concurrency: int) -> dict[str, int]:
    """Run the selection tasks once; returns {task_id: 0|1}."""
    from benchmark.spreadsheetbench.train.case_loader import SpreadsheetBenchCaseLoader
    from benchmark.spreadsheetbench.train.rollout_executor_vikingbot import (
        VikingBotSpreadsheetBenchRolloutExecutor,
    )
    from openviking.session.train import ExecutionContext

    wanted = set(task_ids)
    cases = [
        c
        for c in SpreadsheetBenchCaseLoader(domain=GATE_DOMAIN, split=GATE_SPLIT).load_cases()
        if _case_id(c) in wanted
    ]
    if len(cases) != len(wanted):
        missing = wanted - {_case_id(c) for c in cases}
        raise RuntimeError(f"selection tasks missing from dataset: {sorted(missing)[:5]}")

    executor = VikingBotSpreadsheetBenchRolloutExecutor(
        config_path=os.environ["OPENVIKING_CONFIG_FILE"], concurrency=concurrency
    )
    context = ExecutionContext(
        policy_snapshot_id="principles-gate", metadata={"stage": "principles-gate"}
    )
    rollouts = await executor.execute(cases, None, context)
    return {str(r.metadata["task_id"]): int(r.metadata["reward"] == 1.0) for r in rollouts}


def _experience_count() -> int:
    """Size of the experience library; baseline behavior depends on it because
    rollouts retrieve case experiences, so the cache must be keyed by it too."""
    try:
        entries = _api(
            "GET",
            "/api/v1/fs/ls?uri=viking://user/default/memories/experiences&node_limit=10000",
        )
        if isinstance(entries, dict):
            entries = entries.get("entries") or entries.get("nodes") or []
        return len(
            [e for e in entries if isinstance(e, dict) and not e.get("isDir")]
        )
    except Exception:
        return 0


def _baseline_vector(version: int, task_ids: list[str], concurrency: int) -> dict[str, int]:
    exp_count = _experience_count()
    cache = GATE_DIR / f"baseline_v{version}_e{exp_count}.json"
    if cache.exists():
        vector = json.loads(cache.read_text())
        if set(vector) == set(task_ids):
            print(
                f"baseline v{version} (exp={exp_count}): cached "
                f"({sum(vector.values())}/{len(vector)})"
            )
            return vector
        print("baseline cache selection mismatch; re-running")
    os.environ.pop("VIKINGBOT_PRINCIPLES_OVERRIDE_FILE", None)
    print(f"baseline v{version} (exp={exp_count}): running {len(task_ids)} tasks ...")
    vector = asyncio.run(_run_selection(task_ids, concurrency))
    cache.write_text(json.dumps(vector, indent=1))
    return vector


# ── gate ───────────────────────────────────────────────────────────────────


def run_gate(k: int, concurrency: int) -> None:
    if not SELECTION_PATH.exists():
        raise SystemExit("no frozen selection set; run `gate_principles.py make-selection` first")
    selection = json.loads(SELECTION_PATH.read_text())
    task_ids: list[str] = selection["task_ids"]

    pending = _api("GET", "/api/v1/memories/principles/pending")
    content = pending.get("content") or ""
    if not content.strip():
        print("no pending candidate; nothing to gate")
        return
    pending_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

    from openviking.session.memory.consolidation import _parse_memory_fields

    default_doc = _api("GET", "/api/v1/memories/principles").get("content") or ""
    version = int(_parse_memory_fields(default_doc).get("version") or 0)

    baseline = _baseline_vector(version, task_ids, concurrency)

    candidate_file = GATE_DIR / "candidate.md"
    candidate_file.write_text(content)
    os.environ["VIKINGBOT_PRINCIPLES_OVERRIDE_FILE"] = str(candidate_file)
    print(f"candidate ({pending_sha[:12]}): running {len(task_ids)} tasks ...")
    try:
        candidate = asyncio.run(_run_selection(task_ids, concurrency))
    finally:
        os.environ.pop("VIKINGBOT_PRINCIPLES_OVERRIDE_FILE", None)

    rescued = sorted(t for t in task_ids if candidate[t] and not baseline[t])
    regressed = sorted(t for t in task_ids if baseline[t] and not candidate[t])
    net = len(rescued) - len(regressed)
    score = sum(candidate.values()) / len(task_ids)
    baseline_score = sum(baseline.values()) / len(task_ids)
    passed = net >= k

    report = {
        "pending_sha256": pending_sha,
        "selection_id": selection["selection_id"],
        "net_wins": net,
        "score": round(score, 4),
        "baseline_score": round(baseline_score, 4),
        "notes": f"rescued={len(rescued)} regressed={len(regressed)} k={k}",
    }
    verdict = "promote" if passed else "reject"
    result = _api("POST", f"/api/v1/memories/principles/{verdict}", report)

    record = {
        "verdict": verdict,
        "report": report,
        "api_result": result,
        "rescued": rescued,
        "regressed": regressed,
        "baseline_vector": baseline,
        "candidate_vector": candidate,
    }
    out = GATE_DIR / f"gate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(record, indent=1))
    print(
        f"[{verdict.upper()}] candidate {score:.2%} vs baseline {baseline_score:.2%} "
        f"| net_wins={net} (rescued {len(rescued)} / regressed {len(regressed)}, k={k})"
    )
    print(f"api: {result}")
    print(f"record -> {out}")


def run_cycle(k: int, concurrency: int) -> None:
    """One SkillOpt-style step: propose from the latest batch's trajectories,
    then gate. Call after a training batch finished committing its rollouts."""
    result = _api("POST", "/api/v1/memories/principles/consolidate")
    status = result.get("status")
    print(f"propose: {result}")
    if status == "pending":
        run_gate(k, concurrency)
    elif status == "skipped" and "already in flight" in str(result.get("reason") or ""):
        # A previous cycle died between propose and verdict; finish gating it.
        run_gate(k, concurrency)
    else:
        print("no candidate to gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed", help="install the seed experience document as default.md")
    p_seed.add_argument("--force", action="store_true", help="overwrite an existing default.md")
    p_sel = sub.add_parser("make-selection", help="freeze the stratified selection set")
    p_sel.add_argument("--size", type=int, default=80)
    p_sel.add_argument("--seed", type=int, default=42)
    p_sel.add_argument(
        "--whole-split",
        action="store_true",
        help="freeze the entire gate split as the selection set (no sampling)",
    )
    p_run = sub.add_parser("run", help="gate the current pending candidate")
    p_run.add_argument("--k", type=int, default=3, help="min net wins to promote")
    p_run.add_argument("--concurrency", type=int, default=20)
    p_cycle = sub.add_parser("cycle", help="propose from the latest trajectories, then gate")
    p_cycle.add_argument("--k", type=int, default=3, help="min net wins to promote")
    p_cycle.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    if args.cmd == "seed":
        install_seed(args.force)
    elif args.cmd == "make-selection":
        make_selection(args.size, args.seed, whole_split=args.whole_split)
    elif args.cmd == "cycle":
        run_cycle(args.k, args.concurrency)
    else:
        run_gate(args.k, args.concurrency)


if __name__ == "__main__":
    main()
