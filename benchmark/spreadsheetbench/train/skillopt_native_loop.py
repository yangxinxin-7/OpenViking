#!/usr/bin/env python3
"""SkillOpt-native training loop (no OV memory pipeline).

Same host as the OV runs — vikingbot in codegen mode (SSB_CODEGEN_MODE=1) — but
the experience document is maintained the SkillOpt way, WITHOUT going through
OpenViking's commit -> semantic-extraction -> compressed trajectory-memory ->
whole-document-rewrite pipeline:

  - The analyst reads the RAW rollout trajectories directly from rollout
    metadata (generated code, execution errors, and per-case "expected X got Y"
    from the OJ comparison) — nothing is summarized or compressed first.
  - Failures and successes are analyzed in separate minibatches (SkillOpt's
    analyst_error / analyst_success), producing add/modify/delete edits.
  - Edits are applied INCREMENTALLY to a "## Learned Rules" section appended to
    the fixed seed base (no whole-document rewrite that could perturb verified
    content).
  - The experience file is injected via VIKINGBOT_PRINCIPLES_OVERRIDE_FILE, so
    the OV server is used only to run vikingbot rollouts, never for memory.

Gate + final test mirror the OV runs: 40-task selection net-wins gate, seed vs
best on the 280-task test split.

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

RESULT_DIR = REPO / "result" / "spreadsheetbench" / "skillopt_native"
SEED_PATH = Path(__file__).resolve().parent / "seed_experience.md"
TRAIN_DOMAIN = os.environ.get("SSB_GATE_DOMAIN", "verified_400_v1")
LEARNED_HEADER = "## Learned Rules"
MINIBATCH = 8
MAX_EDITS_PER_STEP = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── experience document (fixed seed base + incremental Learned Rules) ────────


def initial_experience() -> str:
    base = SEED_PATH.read_text(encoding="utf-8").strip()
    return f"{base}\n\n{LEARNED_HEADER}\n\n(none yet)\n"


def split_experience(text: str) -> tuple[str, list[str]]:
    """Return (fixed base text, learned-rule bullets)."""
    idx = text.find(LEARNED_HEADER)
    if idx == -1:
        return text.strip(), []
    base = text[:idx].strip()
    rules = []
    for line in text[idx + len(LEARNED_HEADER):].splitlines():
        line = line.strip()
        if line.startswith("- "):
            rules.append(line[2:].strip())
    return base, rules


def render_experience(base: str, rules: list[str]) -> str:
    body = "\n".join(f"- {r}" for r in rules) if rules else "(none yet)"
    return f"{base}\n\n{LEARNED_HEADER}\n\n{body}\n"


def apply_edits(rules: list[str], edits: list[dict]) -> tuple[list[str], list[str]]:
    """Apply add/modify/delete edits incrementally. Returns (new_rules, summaries)."""
    rules = list(rules)
    summaries: list[str] = []
    # delete high-index first so earlier indices stay valid
    for edit in sorted(
        [e for e in edits if str(e.get("op")).lower() == "delete"],
        key=lambda e: -int(e.get("target", 0) or 0),
    ):
        i = int(edit.get("target", 0) or 0) - 1
        if 0 <= i < len(rules):
            summaries.append(f"delete: {rules[i][:60]}")
            rules.pop(i)
    for edit in edits:
        op = str(edit.get("op")).lower()
        if op == "modify":
            i = int(edit.get("target", 0) or 0) - 1
            text = str(edit.get("rule") or "").strip()
            if 0 <= i < len(rules) and text:
                summaries.append(f"modify R{i + 1}: {text[:60]}")
                rules[i] = text
        elif op == "add":
            text = str(edit.get("rule") or "").strip()
            if text and text not in rules:
                summaries.append(f"add: {text[:60]}")
                rules.append(text)
    return rules, summaries


# ── raw trajectory digest (the whole point: no compression) ──────────────────


def trajectory_digest(index: int, rollout, max_chars: int = 3500) -> str:
    md = rollout.metadata
    passed = float(md.get("reward") or 0.0) >= 1.0
    instruction = str(rollout.case.input.get("instruction") or "")[:400]
    itype = str(md.get("instruction_type") or "")
    parts = [
        f"### Trajectory {index} [{'PASS' if passed else 'FAIL'}] type={itype}",
        f"Instruction: {instruction}",
    ]
    # per-case expected-vs-got — the precise failure signal SkillOpt has and the
    # OV compressed memory loses.
    for case in md.get("evaluation_result", {}).get("per_case", []):
        if not case.get("passed"):
            parts.append(f"Failure detail: {str(case.get('detail'))[:300]}")
    # execution errors surfaced during the codegen turns
    for tool in md.get("tools_used") or []:
        result = str((tool or {}).get("result") or "")
        if "Error" in result or "Traceback" in result:
            parts.append(f"Exec error: {result.strip()[:400]}")
    parts.append("Final code:")
    parts.append(f"```python\n{str(md.get('solution') or '')[:1600]}\n```")
    return "\n".join(parts)[:max_chars]


ANALYST_FAILURE = """You improve an experience document injected into a spreadsheet-manipulation \
agent that writes one-shot Python scripts. You receive the CURRENT LEARNED RULES and a batch of \
FAILED task trajectories (instruction, per-case expected-vs-got, execution errors, final code).

Find the most important COMMON failure patterns across the batch and propose incremental edits to \
the learned rules that would prevent them. Ground every rule in a failure you actually see (cite the \
error or the expected-vs-got). Prefer few, high-conviction edits; an empty list is valid and often \
correct. At most {max_edits} edits.

Rules must be general (no task-specific ids/sheet names/values), procedural (HOW to act), and not \
duplicate an existing rule.

Respond with strict JSON only:
{{"edits": [
  {{"op":"add","rule":"<imperative sentence>"}},
  {{"op":"modify","target":<rule number>,"rule":"<replacement>"}},
  {{"op":"delete","target":<rule number>}}
]}}"""

ANALYST_SUCCESS = """You improve an experience document injected into a spreadsheet-manipulation \
agent that writes one-shot Python scripts. You receive the CURRENT LEARNED RULES and a batch of \
SUCCESSFUL task trajectories (instruction, final code).

Identify reliable techniques COMMON across multiple trajectories that are worth encoding and not \
already covered. Ground every rule in code actually used in the batch. Prefer few, high-conviction \
edits; an empty list is valid and often correct. At most {max_edits} edits.

Rules must be general, procedural, and not duplicate an existing rule.

Respond with strict JSON only:
{{"edits": [{{"op":"add","rule":"<imperative sentence>"}}, {{"op":"modify","target":<n>,"rule":"..."}}]}}"""


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


async def analyze(agent, rules: list[str], rollouts: list) -> list[dict]:
    """One analyst call per minibatch (failures and successes separate)."""
    rules_txt = "\n".join(f"R{i + 1}. {r}" for i, r in enumerate(rules)) or "(none yet)"
    failures = [r for r in rollouts if float(r.metadata.get("reward") or 0) < 1.0]
    successes = [r for r in rollouts if float(r.metadata.get("reward") or 0) >= 1.0]
    batches = [("failure", failures[i:i + MINIBATCH]) for i in range(0, len(failures), MINIBATCH)]
    if successes:
        batches.append(("success", successes[:MINIBATCH]))

    async def one(kind: str, batch: list) -> list[dict]:
        if not batch:
            return []
        prompt = (ANALYST_FAILURE if kind == "failure" else ANALYST_SUCCESS).format(
            max_edits=MAX_EDITS_PER_STEP
        )
        digests = "\n\n".join(trajectory_digest(i + 1, r) for i, r in enumerate(batch))
        user = f"# CURRENT LEARNED RULES\n{rules_txt}\n\n# TRAJECTORIES\n{digests}"
        raw = ""
        async for ev in agent.provider.chat_stream(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            tools=None, model=agent.model, max_tokens=4096, temperature=0.0,
            session_id=f"analyst-{kind}",
        ):
            if getattr(ev, "type", None) == "response":
                raw = str(getattr(ev.response, "content", "") or "")
        return [{**e, "_kind": kind} for e in _parse_json(raw).get("edits", []) if isinstance(e, dict)]

    results = await asyncio.gather(*(one(k, b) for k, b in batches))
    tagged = [e for r in results for e in r]
    # SkillOpt-style AGGREGATE+SELECT: dedup, failure-batch edits first, cap to
    # the per-step edit budget. Without this the analyst dumps every minibatch's
    # suggestions (8+ adds/step) — rules explode and the gate can't attribute a
    # win/loss to any single rule.
    fail = [e for e in tagged if e.get("_kind") == "failure"]
    succ = [e for e in tagged if e.get("_kind") == "success"]
    seen: set = set()
    out: list[dict] = []
    for e in fail + succ:
        e.pop("_kind", None)
        key = (str(e.get("op")).lower(), str(e.get("rule") or e.get("target")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= MAX_EDITS_PER_STEP:
            break
    return out


# ── rollout / gate (vikingbot codegen, experience via override file) ─────────


def _load_split_ids(split: str) -> list[str]:
    from benchmark.spreadsheetbench.train.data_paths import split_tasks_path

    return [str(i) for i in json.loads(split_tasks_path(TRAIN_DOMAIN).read_text())[split]]


def _cases_for_ids(split: str, task_ids: list[str]) -> list:
    from benchmark.spreadsheetbench.train.case_loader import SpreadsheetBenchCaseLoader

    by_id = {
        str(c.input["task_id"]): c
        for c in SpreadsheetBenchCaseLoader(domain=TRAIN_DOMAIN, split=split).load_cases()
    }
    return [by_id[t] for t in task_ids]


async def rollout(cases: list, stage: str, experience_file: str, concurrency: int) -> list:
    from benchmark.spreadsheetbench.train.rollout_executor_vikingbot import (
        VikingBotSpreadsheetBenchRolloutExecutor,
    )
    from openviking.session.train import ExecutionContext

    os.environ["VIKINGBOT_PRINCIPLES_OVERRIDE_FILE"] = experience_file
    try:
        executor = VikingBotSpreadsheetBenchRolloutExecutor(
            config_path=os.environ["OPENVIKING_CONFIG_FILE"], concurrency=concurrency
        )
        ctx = ExecutionContext(policy_snapshot_id=stage, metadata={"stage": stage})
        return list(await executor.execute(cases, None, ctx))
    finally:
        os.environ.pop("VIKINGBOT_PRINCIPLES_OVERRIDE_FILE", None)


def _vec(rollouts: list) -> dict[str, int]:
    return {str(r.metadata["task_id"]): int(r.metadata["reward"] == 1.0) for r in rollouts}


def _score(vec: dict) -> str:
    n = sum(vec.values())
    return f"{n}/{len(vec)} = {n / max(len(vec), 1):.2%}"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    from benchmark.tau2.train.rollout_executor_vikingbot import _build_agent

    analyst_agent = await asyncio.to_thread(_build_agent, os.environ["OPENVIKING_CONFIG_FILE"], max_iterations=5)

    seed_file = RESULT_DIR / "seed.md"
    seed_file.write_text(initial_experience())
    current = initial_experience()
    cur_file = RESULT_DIR / "current.md"
    cur_file.write_text(current)

    sel_ids = _load_split_ids("selection")
    sel_cases = _cases_for_ids("selection", sel_ids)
    train_ids = _load_split_ids("train")

    # baseline (seed) on selection, cached across steps until current changes
    base_vec = _vec(await rollout(sel_cases, "native-base", str(seed_file), args.concurrency))
    log(f"seed selection baseline: {_score(base_vec)}")

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
            tr = await rollout(_cases_for_ids("train", batch_ids), f"native-train-s{step}", str(cur_file), args.concurrency)
            log(f"step {step} train score: {_score(_vec(tr))}")

            base, rules = split_experience(current)
            edits = await analyze(analyst_agent, rules, tr)
            new_rules, summaries = apply_edits(rules, edits)
            if not summaries:
                log(f"step {step}: analyst proposed no edits -> skip gate")
                history.append({"step": step, "train": _score(_vec(tr)), "edits": [], "verdict": "no_candidate"})
                (RESULT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
                continue
            candidate = render_experience(base, new_rules)
            cand_file = RESULT_DIR / f"candidate_s{step}.md"
            cand_file.write_text(candidate)
            log(f"step {step}: {len(summaries)} edits -> {summaries}")

            cand_vec = _vec(await rollout(sel_cases, f"native-gate-s{step}", str(cand_file), args.concurrency))
            rescued = [t for t in sel_ids if cand_vec[t] and not base_vec[t]]
            regressed = [t for t in sel_ids if base_vec[t] and not cand_vec[t]]
            net = len(rescued) - len(regressed)
            promote = net >= args.k
            log(f"step {step} GATE: cand {_score(cand_vec)} vs base {_score(base_vec)} | "
                f"net {net} (rescued {len(rescued)} / regressed {len(regressed)}) -> {'PROMOTE' if promote else 'REJECT'}")
            if promote:
                current = candidate
                base_vec = cand_vec  # new incumbent baseline
                (RESULT_DIR / "current.md").write_text(current)
            history.append({
                "step": step, "train": _score(_vec(tr)), "edits": summaries,
                "net": net, "cand": _score(cand_vec), "verdict": "PROMOTE" if promote else "REJECT",
            })
            (RESULT_DIR / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))

    # ── final test: seed vs best on 280 ──
    (RESULT_DIR / "best.md").write_text(current)
    test_ids = _load_split_ids("test")
    test_cases = _cases_for_ids("test", test_ids)
    log(f"FINAL TEST: {len(test_cases)} tasks")
    seed_vec = _vec(await rollout(test_cases, "native-final-seed", str(seed_file), args.concurrency))
    log(f"final seed: {_score(seed_vec)}")
    best_vec = _vec(await rollout(test_cases, "native-final-best", str(RESULT_DIR / 'best.md'), args.concurrency))
    log(f"final best: {_score(best_vec)}")
    rescued = [t for t in test_ids if best_vec[t] and not seed_vec[t]]
    regressed = [t for t in test_ids if seed_vec[t] and not best_vec[t]]
    summary = {
        "seed": _score(seed_vec), "best": _score(best_vec),
        "net_wins": len(rescued) - len(regressed), "rescued": len(rescued), "regressed": len(regressed),
    }
    (RESULT_DIR / "finaltest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    log(f"FINAL SUMMARY: {summary}")
    log("DONE")


if __name__ == "__main__":
    asyncio.run(main())
