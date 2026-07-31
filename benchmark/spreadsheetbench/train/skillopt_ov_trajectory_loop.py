#!/usr/bin/env python3
"""Ablation: SkillOpt-native loop but the analyst reads OV-EXTRACTED trajectory
memories instead of the raw rollout trajectories.

Single variable vs skillopt_native_loop.py: the SOURCE the analyst reads.
  - native  : raw rollout metadata (generated code + exec errors + per-case
              "expected X got Y") -> +23.58pp on 280 test
  - this run: OV commits the rollouts, the semantic-extraction pipeline
              compresses each into a trajectory-memory contract, and the analyst
              reads THOSE compressed contracts.

Everything else is identical to the native loop — failure/success minibatch
analysts, incremental add/modify/delete with an edit budget of 4, 40-task
selection net-wins gate, seed-vs-best 280-task final test. So the result
isolates the cost of OV's semantic compression as an analyst source.

Only trajectory memory is extracted on commit (OPENVIKING_TRAIN_COMMIT_MEMORY_TYPES
=trajectories) to keep commits fast — cases/experiences are skipped.

Env: OPENVIKING_CONFIG_FILE, OV_API_BASE, SSB_GATE_DOMAIN=verified_400_v1,
SSB_GATE_SPLIT=selection, OPENVIKING_TAU2_DISABLE_EXPERIENCE_SKILL=1,
VIKINGBOT_PRINCIPLES_SKILLOPT_STYLE=1, SSB_CODEGEN_MODE=1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bot"))

# reuse everything from the native loop except the trajectory source
import benchmark.spreadsheetbench.train.skillopt_native_loop as native  # noqa: E402

RESULT_DIR = REPO / "result" / "spreadsheetbench" / "skillopt_ov_trajectory"
API_BASE = os.environ.get("OV_API_BASE", "http://127.0.0.1:1933").rstrip("/")
HEADERS = {"X-OpenViking-Account": "default", "X-OpenViking-User": "default"}
TRAJ_URI = "viking://user/default/memories/trajectories"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _api_ls() -> list[dict]:
    # Returns [] when the directory doesn't exist yet (404) — before the first
    # commit the trajectories dir is absent, and ls'ing it 404s.
    u = f"{API_BASE}/api/v1/fs/ls?uri={urllib.parse.quote(TRAJ_URI, safe=':/')}&node_limit=10000"
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=HEADERS), timeout=15))["result"]
    except Exception:
        return []
    return [e for e in r if isinstance(e, dict) and not e.get("isDir")] if isinstance(r, list) else []


def _api_read(uri: str) -> str:
    u = f"{API_BASE}/api/v1/content/read?uri={urllib.parse.quote(uri, safe=':/')}"
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=HEADERS), timeout=15))["result"]
    except Exception:
        return ""
    return r if isinstance(r, str) else ""


def _wait_extraction(before: int, expected: int, timeout: float = 1800.0) -> list[dict]:
    """Block until the trajectory count stabilizes above `before` (extraction is
    an async server-side queue). Returns the entries newest-first."""
    deadline = time.time() + timeout
    last = -1
    stable = 0
    while time.time() < deadline:
        entries = _api_ls()
        n = len(entries)
        if n == last:
            stable += 1
            if stable >= 2 and n > before:
                break
        else:
            stable = 0
        last = n
        time.sleep(20)
    entries = _api_ls()
    entries.sort(key=lambda e: str(e.get("modTime") or ""), reverse=True)
    log(f"extraction: {before} -> {len(entries)} trajectories (expected ~+{expected})")
    return entries


# word-root matches (no trailing \b, so "failed"/"Successfully" both match)
_FAIL_RE = re.compile(r"(failed|failure|\bfail|mismatch|\berror|incorrect)", re.I)
_SUCC_RE = re.compile(r"(success|succeed|correct|passed)", re.I)


def _kind_from_contract(text: str) -> str:
    """Judge pass/fail from the compressed contract's Result/Evidence/outcome —
    the only signal available when the source is OV's extracted memory."""
    m = re.search(r"\"outcome\"\s*:\s*\"(\w+)\"", text)
    if m and m.group(1).lower() in ("success", "failure", "partial", "unfinished"):
        return "success" if m.group(1).lower() == "success" else "failure"
    tail = ""
    for seg in ("Result", "Evidence"):
        mm = re.search(rf"-?\s*{seg}:(.*?)(?:\n-|\n#|\Z)", text, re.DOTALL)
        if mm:
            tail += " " + mm.group(1)
    has_fail = bool(_FAIL_RE.search(tail))
    has_succ = bool(_SUCC_RE.search(tail))
    if has_succ and not has_fail:
        return "success"
    if has_fail:
        return "failure"
    return "success" if has_succ else "failure"


async def _analyze_contracts(agent, rules: list[str], contracts: list[tuple[str, str]]) -> list[dict]:
    """Same two-stage analyst as native, but digests are the OV contracts.

    contracts: list of (kind, contract_text). Builds failure/success minibatches
    and reuses native's analyst prompts + AGGREGATE/SELECT (edit budget cap).
    """
    rules_txt = "\n".join(f"R{i + 1}. {r}" for i, r in enumerate(rules)) or "(none yet)"
    fails = [c for k, c in contracts if k == "failure"]
    succs = [c for k, c in contracts if k == "success"]
    batches = [("failure", fails[i:i + native.MINIBATCH]) for i in range(0, len(fails), native.MINIBATCH)]
    if succs:
        batches.append(("success", succs[:native.MINIBATCH]))

    async def one(kind: str, batch: list[str]) -> list[dict]:
        if not batch:
            return []
        prompt = (native.ANALYST_FAILURE if kind == "failure" else native.ANALYST_SUCCESS).format(
            max_edits=native.MAX_EDITS_PER_STEP
        )
        digests = "\n\n".join(f"### Trajectory {i + 1} [{kind.upper()}]\n{c[:3000]}" for i, c in enumerate(batch))
        user = f"# CURRENT LEARNED RULES\n{rules_txt}\n\n# TRAJECTORIES\n{digests}"
        raw = ""
        async for ev in agent.provider.chat_stream(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            tools=None, model=agent.model, max_tokens=4096, temperature=0.0, session_id=f"analyst-{kind}",
        ):
            if getattr(ev, "type", None) == "response":
                raw = str(getattr(ev.response, "content", "") or "")
        return [{**e, "_kind": kind} for e in native._parse_json(raw).get("edits", []) if isinstance(e, dict)]

    results = await asyncio.gather(*(one(k, b) for k, b in batches))
    tagged = [e for r in results for e in r]
    seen: set = set()
    out: list[dict] = []
    for e in [x for x in tagged if x.get("_kind") == "failure"] + [x for x in tagged if x.get("_kind") == "success"]:
        e.pop("_kind", None)
        key = (str(e.get("op")).lower(), str(e.get("rule") or e.get("target")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= native.MAX_EDITS_PER_STEP:
            break
    return out


async def _commit(rollouts: list, run_id: str) -> None:
    from openviking.session.train.batch_runner import BatchTrainEvalConfig, _build_http_client
    from openviking.session.train.components.session_commit import SessionCommitPolicyTrainer
    from openviking.session.train.domain import ExperienceSet

    cfg = BatchTrainEvalConfig(dataset="spreadsheetbench", domain=native.TRAIN_DOMAIN,
                               config_path=os.environ["OPENVIKING_CONFIG_FILE"])
    client = _build_http_client(cfg)
    await client.initialize()
    try:
        trainer = SessionCommitPolicyTrainer(client=client, run_id=run_id, commit_concurrency=20, show_progress=False)
        await trainer.train_rollouts(rollouts, ExperienceSet(root_uri="viking://user/memories/experiences", policies=[]), None)
    finally:
        await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    # only extract trajectory memory on commit (fast)
    os.environ["OPENVIKING_TRAIN_COMMIT_MEMORY_TYPES"] = "trajectories"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    from benchmark.tau2.train.rollout_executor_vikingbot import _build_agent

    analyst_agent = await asyncio.to_thread(_build_agent, os.environ["OPENVIKING_CONFIG_FILE"], max_iterations=5)

    seed_file = RESULT_DIR / "seed.md"
    seed_file.write_text(native.initial_experience())
    current = native.initial_experience()
    cur_file = RESULT_DIR / "current.md"
    cur_file.write_text(current)

    sel_ids = native._load_split_ids("selection")
    sel_cases = native._cases_for_ids("selection", sel_ids)
    train_ids = native._load_split_ids("train")

    base_vec = native._vec(await native.rollout(sel_cases, "ovtraj-base", str(seed_file), args.concurrency))
    log(f"seed selection baseline: {native._score(base_vec)}")

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        ids = list(train_ids)
        random.Random(1000 + epoch).shuffle(ids)
        size = len(ids) // args.batches_per_epoch
        for b in range(args.batches_per_epoch):
            step += 1
            batch_ids = ids[b * size:(b + 1) * size]
            log(f"===== STEP {step} (epoch {epoch}): rollout {len(batch_ids)} train tasks =====")
            cur_file.write_text(current)
            tr = await native.rollout(native._cases_for_ids("train", batch_ids), f"ovtraj-train-s{step}", str(cur_file), args.concurrency)
            log(f"step {step} train score: {native._score(native._vec(tr))}")

            # --- the ablation: commit -> OV extracts trajectory memory -> read back ---
            before = len(_api_ls())
            log(f"step {step}: committing {len(tr)} rollouts (OV trajectory extraction only) ...")
            await _commit(tr, run_id=f"ovtraj-{time.strftime('%m%d%H%M%S')}-s{step}")
            entries = _wait_extraction(before, expected=len(tr))
            new_entries = entries[: max(len(entries) - before, 0)] if len(entries) > before else entries[:len(tr)]
            contracts: list[tuple[str, str]] = []
            for e in new_entries:
                c = _api_read(e["uri"])
                if c.strip():
                    body = re.sub(r"<!--\s*MEMORY_FIELDS.*?-->", "", c, flags=re.DOTALL).strip()
                    contracts.append((_kind_from_contract(c), body))
            log(f"step {step}: read {len(contracts)} OV trajectory contracts "
                f"({sum(1 for k,_ in contracts if k=='failure')} fail / {sum(1 for k,_ in contracts if k=='success')} success)")

            base, rules = native.split_experience(current)
            edits = await _analyze_contracts(analyst_agent, rules, contracts)
            new_rules, summaries = native.apply_edits(rules, edits)
            if not summaries:
                log(f"step {step}: analyst proposed no edits -> skip gate")
                history.append({"step": step, "train": native._score(native._vec(tr)), "edits": [], "verdict": "no_candidate"})
                (RESULT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
                continue
            candidate = native.render_experience(base, new_rules)
            cand_file = RESULT_DIR / f"candidate_s{step}.md"
            cand_file.write_text(candidate)
            log(f"step {step}: {len(summaries)} edits -> {summaries}")

            cand_vec = native._vec(await native.rollout(sel_cases, f"ovtraj-gate-s{step}", str(cand_file), args.concurrency))
            rescued = [t for t in sel_ids if cand_vec[t] and not base_vec[t]]
            regressed = [t for t in sel_ids if base_vec[t] and not cand_vec[t]]
            net = len(rescued) - len(regressed)
            promote = net >= args.k
            log(f"step {step} GATE: cand {native._score(cand_vec)} vs base {native._score(base_vec)} | "
                f"net {net} (rescued {len(rescued)} / regressed {len(regressed)}) -> {'PROMOTE' if promote else 'REJECT'}")
            if promote:
                current = candidate
                base_vec = cand_vec
                (RESULT_DIR / "current.md").write_text(current)
            history.append({"step": step, "train": native._score(native._vec(tr)), "edits": summaries,
                            "net": net, "cand": native._score(cand_vec), "verdict": "PROMOTE" if promote else "REJECT"})
            (RESULT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))

    (RESULT_DIR / "best.md").write_text(current)
    test_ids = native._load_split_ids("test")
    test_cases = native._cases_for_ids("test", test_ids)
    log(f"FINAL TEST: {len(test_cases)} tasks")
    seed_vec = native._vec(await native.rollout(test_cases, "ovtraj-final-seed", str(seed_file), args.concurrency))
    log(f"final seed: {native._score(seed_vec)}")
    best_vec = native._vec(await native.rollout(test_cases, "ovtraj-final-best", str(RESULT_DIR / 'best.md'), args.concurrency))
    log(f"final best: {native._score(best_vec)}")
    rescued = [t for t in test_ids if best_vec[t] and not seed_vec[t]]
    regressed = [t for t in test_ids if seed_vec[t] and not best_vec[t]]
    summary = {"seed": native._score(seed_vec), "best": native._score(best_vec),
               "net_wins": len(rescued) - len(regressed), "rescued": len(rescued), "regressed": len(regressed)}
    (RESULT_DIR / "finaltest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    log(f"FINAL SUMMARY: {summary}")
    log("DONE")


if __name__ == "__main__":
    asyncio.run(main())
