#!/usr/bin/env python3
"""SkillOpt-aligned full training loop for the OV principles pipeline.

Mirrors one complete SkillOpt run on SpreadsheetBench verified_400:

    for each of 8 steps (80-task train pool, shuffled per epoch, 2 batches):
        1. rollout the 40-task train batch with the CURRENT default.md injected
        2. commit rollouts to OV -> trajectory memories extracted
        3. propose (analyst minibatches + merge) -> pending.md
        4. gate on the 40-task selection split -> promote / reject
    finally:
        test the seed document and the best (current default.md) on the
        280-task test split.

Environment (set by the caller):
    OPENVIKING_CONFIG_FILE  slot config (server the run commits to)
    OV_API_BASE             matching server URL
    SSB_GATE_DOMAIN=verified_400_v1  SSB_GATE_SPLIT=selection
    OPENVIKING_TAU2_DISABLE_EXPERIENCE_SKILL=1   (no retrieval, SkillOpt-style)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bot"))

from benchmark.spreadsheetbench.train.gate_principles import (  # noqa: E402
    _api,
    run_gate,
)

RESULT_DIR = REPO / "result" / "spreadsheetbench" / ("skillopt_aligned_codegen" if __import__("os").getenv("SSB_CODEGEN_MODE")=="1" else "skillopt_aligned")
TRAIN_DOMAIN = os.environ.get("SSB_GATE_DOMAIN", "verified_400_v1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_split_ids(split: str) -> list[str]:
    from benchmark.spreadsheetbench.train.data_paths import split_tasks_path

    return [str(i) for i in json.loads(split_tasks_path(TRAIN_DOMAIN).read_text())[split]]


def _cases_for_ids(split: str, task_ids: list[str]) -> list:
    from benchmark.spreadsheetbench.train.case_loader import SpreadsheetBenchCaseLoader

    loader = SpreadsheetBenchCaseLoader(domain=TRAIN_DOMAIN, split=split)
    by_id = {}
    for case in loader.load_cases():
        by_id[str(case.input["task_id"])] = case
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise RuntimeError(f"tasks missing from split {split}: {missing[:5]}")
    return [by_id[t] for t in task_ids]


async def _rollout(cases: list, stage: str, concurrency: int) -> list:
    from benchmark.spreadsheetbench.train.rollout_executor_vikingbot import (
        VikingBotSpreadsheetBenchRolloutExecutor,
    )
    from openviking.session.train import ExecutionContext

    executor = VikingBotSpreadsheetBenchRolloutExecutor(
        config_path=os.environ["OPENVIKING_CONFIG_FILE"], concurrency=concurrency
    )
    context = ExecutionContext(policy_snapshot_id=stage, metadata={"stage": stage})
    return list(await executor.execute(cases, None, context))


async def _commit(rollouts: list, run_id: str) -> None:
    from openviking.session.train.batch_runner import BatchTrainEvalConfig, _build_http_client
    from openviking.session.train.components.session_commit import SessionCommitPolicyTrainer
    from openviking.session.train.domain import ExperienceSet

    config = BatchTrainEvalConfig(
        dataset="spreadsheetbench",
        domain=TRAIN_DOMAIN,
        config_path=os.environ["OPENVIKING_CONFIG_FILE"],
    )
    client = _build_http_client(config)
    await client.initialize()
    try:
        trainer = SessionCommitPolicyTrainer(
            client=client,
            run_id=run_id,
            commit_concurrency=20,
            show_progress=False,
        )
        policy_set = ExperienceSet(root_uri="viking://user/memories/experiences", policies=[])
        result = await trainer.train_rollouts(rollouts, policy_set, None)
        errors = result.apply_result.errors if result.apply_result else []
        if errors:
            log(f"commit errors ({len(errors)}): {str(errors[:2])[:300]}")
    finally:
        await client.close()


def _trajectory_count() -> int:
    try:
        entries = _api(
            "GET",
            "/api/v1/fs/ls?uri=viking://user/default/memories/trajectories&node_limit=10000",
        )
        if isinstance(entries, dict):
            entries = entries.get("entries") or entries.get("nodes") or []
        return len([e for e in entries if isinstance(e, dict) and not e.get("isDir")])
    except Exception:
        return 0


def _score(vector: dict[str, int]) -> str:
    return f"{sum(vector.values())}/{len(vector)} = {sum(vector.values()) / max(len(vector), 1):.2%}"


async def _final_test(concurrency: int, seed_snapshot: Path) -> None:
    """Test split runs: seed document (via override) vs best (server default.md)."""
    test_ids = _load_split_ids("test")
    cases = _cases_for_ids("test", test_ids)
    log(f"FINAL TEST: {len(cases)} tasks")

    results: dict[str, dict[str, int]] = {}
    best_doc = _api("GET", "/api/v1/memories/principles").get("content") or ""
    best_is_seed = best_doc.strip() == seed_snapshot.read_text().strip()

    arms = [("seed", str(seed_snapshot))]
    if best_is_seed:
        log("no promotion happened: best == seed, running the single arm once")
    else:
        arms.append(("best", None))

    for arm, override in arms:
        if override:
            os.environ["VIKINGBOT_PRINCIPLES_OVERRIDE_FILE"] = override
        else:
            os.environ.pop("VIKINGBOT_PRINCIPLES_OVERRIDE_FILE", None)
        try:
            log(f"test arm [{arm}] running ...")
            rollouts = await _rollout(cases, f"finaltest-{arm}", concurrency)
        finally:
            os.environ.pop("VIKINGBOT_PRINCIPLES_OVERRIDE_FILE", None)
        vector = {str(r.metadata["task_id"]): int(r.metadata["reward"] == 1.0) for r in rollouts}
        results[arm] = vector
        (RESULT_DIR / f"finaltest_{arm}.json").write_text(json.dumps(vector, indent=1))
        log(f"test arm [{arm}]: {_score(vector)}")

    summary = {arm: _score(vector) for arm, vector in results.items()}
    if "best" in results:
        a, b = results["seed"], results["best"]
        rescued = sorted(t for t in test_ids if b[t] and not a[t])
        regressed = sorted(t for t in test_ids if a[t] and not b[t])
        summary["net_wins_best_vs_seed"] = len(rescued) - len(regressed)
        summary["rescued"] = len(rescued)
        summary["regressed"] = len(regressed)
    (RESULT_DIR / "finaltest_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"FINAL TEST SUMMARY: {summary}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=35)
    parser.add_argument("--k", type=int, default=1, help="gate net-wins threshold (SkillOpt: any improvement)")
    parser.add_argument("--skip-train", action="store_true", help="only run the final test")
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="resume from this global step (earlier steps are skipped; use after a crash)",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # Unique per-process tag so committed session ids never collide with a
    # previous run's (session_id embeds run_id; a fixed run_id makes create_session
    # fail "already exists" on re-runs, silently dropping most trajectories).
    run_tag = time.strftime("%m%d%H%M%S")

    # Snapshot the seed document (the S_0 baseline arm of the final test).
    seed_doc = _api("GET", "/api/v1/memories/principles").get("content") or ""
    if not seed_doc.strip():
        raise SystemExit("default.md is empty — run `gate_principles.py seed` first")
    seed_snapshot = RESULT_DIR / "seed_snapshot.md"
    if not seed_snapshot.exists():
        seed_snapshot.write_text(seed_doc)

    if not args.skip_train:
        train_ids = _load_split_ids("train")
        log(f"train pool: {len(train_ids)} tasks; {args.epochs} epochs x {args.batches_per_epoch} batches")
        step = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            ids = list(train_ids)
            random.Random(1000 + epoch).shuffle(ids)
            size = len(ids) // args.batches_per_epoch
            for b in range(args.batches_per_epoch):
                step += 1
                if step < args.start_step:
                    continue
                batch_ids = ids[b * size : (b + 1) * size]
                log(f"===== STEP {step} (epoch {epoch}) : rollout {len(batch_ids)} train tasks =====")
                cases = _cases_for_ids("train", batch_ids)
                rollouts = await _rollout(cases, f"train-step{step}", args.concurrency)
                vector = {
                    str(r.metadata["task_id"]): int(r.metadata["reward"] == 1.0) for r in rollouts
                }
                log(f"step {step} train score: {_score(vector)}")
                (RESULT_DIR / f"train_step{step}.json").write_text(json.dumps(vector, indent=1))

                log(f"step {step}: committing {len(rollouts)} rollouts (trajectory extraction) ...")
                await _commit(rollouts, run_id=f"skillopt-{run_tag}-s{step}")
                log(f"step {step}: trajectories on server: {_trajectory_count()}")

                log(f"step {step}: propose ...")
                propose = _api("POST", "/api/v1/memories/principles/consolidate")
                log(f"step {step}: propose -> {json.dumps(propose, ensure_ascii=False)[:400]}")
                verdict = "no_candidate"
                if propose.get("status") == "pending" or "already in flight" in str(
                    propose.get("reason") or ""
                ):
                    # run_gate drives its own asyncio.run; give it a fresh
                    # thread so it does not collide with this event loop.
                    await asyncio.to_thread(run_gate, args.k, args.concurrency)
                    verdict = "gated"
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_score": _score(vector),
                        "propose": {k: v for k, v in propose.items() if k != "edits"},
                        "edits": propose.get("edits"),
                        "verdict": verdict,
                    }
                )
                (RESULT_DIR / "loop_history.json").write_text(
                    json.dumps(history, ensure_ascii=False, indent=1)
                )

    await _final_test(args.concurrency, seed_snapshot)
    log("DONE")


if __name__ == "__main__":
    asyncio.run(main())
