#!/usr/bin/env python3
"""Ablation: SkillOpt-native loop, but the analyst reads trajectory CONTRACTS
that were produced by OV's real trajectory-extraction prompt — not the raw
rollout trajectories.

Single variable vs skillopt_native_loop.py: the analyst source.
  - native      : raw rollout trajectory (code + exec errors + per-case
                  "expected X got Y")                    -> +23.58pp
  - this run    : each raw trajectory is first compressed by OV's ACTUAL
                  trajectory-extraction prompt (loaded verbatim from
                  memory/trajectories.yaml) into a generalized contract, and the
                  analyst reads THOSE contracts.

This reproduces exactly what OV's semantic-extraction pipeline would feed the
analyst (same prompt, same 11-label contract format, same "generalize away
identifiers/numbers/dates" rules), but runs the compression locally with a
direct VLM call — so it sidesteps OV's session-commit/extraction machinery
(which cannot extract trajectory-only quickly). Everything else is identical to
the native loop, so the result isolates the cost of OV's semantic compression
as an analyst source.

Env: OPENVIKING_CONFIG_FILE, SSB_GATE_DOMAIN=verified_400_v1,
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bot"))

import yaml  # noqa: E402

import benchmark.spreadsheetbench.train.skillopt_native_loop as native  # noqa: E402

RESULT_DIR = REPO / "result" / "spreadsheetbench" / "skillopt_compressed"
TRAJ_YAML = REPO / "openviking" / "prompts" / "templates" / "memory" / "trajectories.yaml"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_ov_extraction_prompt() -> str:
    """Build the compression prompt from OV's real trajectory schema so the
    contract matches what the OV pipeline would produce."""
    spec = yaml.safe_load(TRAJ_YAML.read_text(encoding="utf-8"))
    fields = {f["name"]: f.get("description", "") for f in spec.get("fields", [])}
    content_fmt = fields.get("content", "")
    outcome_fmt = fields.get("outcome", "")
    name_fmt = fields.get("trajectory_name", "")
    return (
        "You are a memory-extraction agent. Given an agent's task execution "
        "trajectory (its generated code, execution errors, and per-case results), "
        "extract ONE reusable trajectory memory — exactly as an offline extraction "
        "pipeline would. Generalize away all instance-specific values.\n\n"
        f"## trajectory_name\n{name_fmt}\n\n## outcome\n{outcome_fmt}\n\n"
        f"## content (the contract)\n{content_fmt}\n\n"
        "Respond with strict JSON only:\n"
        '{"trajectory_name": "<snake_case name>", "outcome": "<success|failure|'
        'partial|unfinished|unknown>", "content": "<the full contract in the '
        'exact format above>"}'
    )


COMPRESS_SYSTEM = _load_ov_extraction_prompt()


async def compress_trajectory(agent, rollout) -> tuple[str, str]:
    """Compress one raw rollout trajectory into an OV-style contract via a direct
    VLM call using OV's extraction prompt. Returns (outcome, contract_text)."""
    raw = native.trajectory_digest(0, rollout)  # raw: code + exec errors + per-case
    out = ""
    async for ev in agent.provider.chat_stream(
        messages=[{"role": "system", "content": COMPRESS_SYSTEM},
                  {"role": "user", "content": f"# TRAJECTORY\n{raw}"}],
        tools=None, model=agent.model, max_tokens=4096, temperature=0.0,
        session_id="compress",
    ):
        if getattr(ev, "type", None) == "response":
            out = str(getattr(ev.response, "content", "") or "")
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return "unknown", ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "unknown", ""
    return str(obj.get("outcome") or "unknown").lower(), str(obj.get("content") or "")


async def _analyze_contracts(agent, rules: list[str], contracts: list[tuple[str, str]]) -> list[dict]:
    """Two-stage analyst (failure/success minibatches + AGGREGATE/SELECT), reading
    the compressed contracts. Identical structure to native.analyze but the
    digests are contracts, and pass/fail comes from the contract's outcome."""
    rules_txt = "\n".join(f"R{i + 1}. {r}" for i, r in enumerate(rules)) or "(none yet)"
    fails = [c for k, c in contracts if k != "success" and c.strip()]
    succs = [c for k, c in contracts if k == "success" and c.strip()]
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


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    from benchmark.tau2.train.rollout_executor_vikingbot import _build_agent

    agent = await asyncio.to_thread(_build_agent, os.environ["OPENVIKING_CONFIG_FILE"], max_iterations=5)

    seed_file = RESULT_DIR / "seed.md"
    seed_file.write_text(native.initial_experience())
    current = native.initial_experience()
    cur_file = RESULT_DIR / "current.md"
    cur_file.write_text(current)

    sel_ids = native._load_split_ids("selection")
    sel_cases = native._cases_for_ids("selection", sel_ids)
    train_ids = native._load_split_ids("train")

    base_vec = native._vec(await native.rollout(sel_cases, "cmp-base", str(seed_file), args.concurrency))
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
            tr = await native.rollout(native._cases_for_ids("train", batch_ids), f"cmp-train-s{step}", str(cur_file), args.concurrency)
            log(f"step {step} train score: {native._score(native._vec(tr))}")

            # --- the ablation: compress each raw trajectory with OV's real prompt ---
            log(f"step {step}: compressing {len(tr)} trajectories with OV extraction prompt ...")
            compressed = await asyncio.gather(*(compress_trajectory(agent, r) for r in tr))
            # Split failure/success by the rollout's GROUND-TRUTH reward (same as
            # the native loop), not by the VLM-judged outcome — so the only
            # variable vs native is the compressed content, never a misjudged label.
            contracts = []
            for rollout, (_outcome, content) in zip(tr, compressed, strict=False):
                if content.strip():
                    kind = "success" if float(rollout.metadata.get("reward") or 0) >= 1.0 else "failure"
                    contracts.append((kind, content))
            nf = sum(1 for k, _ in contracts if k != "success")
            log(f"step {step}: {len(contracts)} contracts ({nf} fail / {len(contracts) - nf} success, by true reward)")

            base, rules = native.split_experience(current)
            edits = await _analyze_contracts(agent, rules, contracts)
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

            cand_vec = native._vec(await native.rollout(sel_cases, f"cmp-gate-s{step}", str(cand_file), args.concurrency))
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
    seed_vec = native._vec(await native.rollout(test_cases, "cmp-final-seed", str(seed_file), args.concurrency))
    log(f"final seed: {native._score(seed_vec)}")
    best_vec = native._vec(await native.rollout(test_cases, "cmp-final-best", str(RESULT_DIR / 'best.md'), args.concurrency))
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
