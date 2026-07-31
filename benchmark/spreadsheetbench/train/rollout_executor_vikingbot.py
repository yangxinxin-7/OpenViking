#!/usr/bin/env python3
"""SpreadsheetBench RolloutExecutor implementation for batch policy training.

Mirrors ``benchmark/tau2/train/rollout_executor_vikingbot.py``: a VikingBot
agent loop drives one task per rollout. Instead of a user simulator + domain
tools, the agent gets a private working directory holding test case 1's input
workbook and two tools:

- ``run_python``  — execute exploration/solution code inside the workdir.
- ``submit_solution`` — record the final self-contained script; it is validated
  on a fresh copy of test case 1 before being accepted.

Reward follows the official OJ protocol: the accepted solution is re-executed on
fresh copies of every shipped test case (3 in all_data_912, 1 in verified_400;
discovered from disk, with the upstream filename substitution) and must pass the
cell-level comparison on every one (hard restriction).

Experience memory integration (skill / selector tooling and env switches) is
reused from the tau2 executor so both benchmarks run the same memory pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.spreadsheetbench.train.data_paths import (
    answer_xlsx_name,
    input_xlsx_name,
    output_xlsx_name,
)
from benchmark.spreadsheetbench.train.oj_eval import compare_workbooks, recalc_enabled

# Shared vikingbot/experience machinery. These live in the tau2 package but are
# benchmark-agnostic; keeping one implementation means selector behaviour and
# env switches (OPENVIKING_TAU2_EXPERIENCE_SELECTOR / _DISABLE_EXPERIENCE_SKILL)
# stay identical across benchmarks.
from benchmark.tau2.train.rollout_executor_vikingbot import (
    _build_agent,
    _case_memory_context_from_tools,
    _elapsed_ms,
    _experience_selector_enabled,
    _experience_skill_disabled,
    _make_load_relevant_experience_tool,
    _make_read_experience_tool,
    _make_search_experience_tool,
    _merge_memories,
    _RolloutTiming,
    _safe_session_fragment,
    _vikingbot_imports,
)
from openviking.message import Message, TextPart, ToolPart
from openviking.session.train import (
    Case,
    CriterionResult,
    ExecutionContext,
    ExperienceSet,
    Rollout,
    RubricEvaluation,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

_PREVIEW_ROWS = int(os.getenv("SSB_PREVIEW_ROWS", "5"))
_PREVIEW_MAX_CHARS = int(os.getenv("SSB_PREVIEW_MAX_CHARS", "6000"))
_CODE_TIMEOUT_SECONDS = float(os.getenv("SSB_CODE_TIMEOUT_SECONDS", "90"))
_TOOL_OUTPUT_MAX_CHARS = int(os.getenv("SSB_TOOL_OUTPUT_MAX_CHARS", "4000"))


def _work_root() -> Path:
    root = os.getenv("SSB_WORK_ROOT")
    if root:
        path = Path(root).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.gettempdir())


def _keep_workdirs() -> bool:
    return os.getenv("SSB_KEEP_WORKDIRS") == "1"


def _codegen_enabled() -> bool:
    """SkillOpt-aligned solving mode: instead of the vikingbot tool loop
    (run_python探查 + submit验证 + 多轮自查), the model writes ONE self-contained
    script in a single turn, which is executed and — only if it raises — fed the
    error and retried (up to SSB_CODEGEN_MAX_TURNS). No workbook exploration, no
    answer feedback. This is the exact evaluation-path behaviour of SkillOpt's
    run_multi (gold_path=""), so the vikingbot baseline drops to SkillOpt's
    level and the memory/gate mechanisms are compared on an equally weak host."""
    return os.getenv("SSB_CODEGEN_MODE") == "1"


_CODEGEN_MAX_TURNS = int(os.getenv("SSB_CODEGEN_MAX_TURNS", "5"))


def _extract_code_block(text: str) -> str:
    """Pull the Python source out of a model reply.

    Prefers a ```python fenced block; falls back to any ``` fence; finally to
    the whole reply if it already looks like code.
    """
    import re

    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    stripped = text.strip()
    if stripped.startswith("import ") or stripped.startswith("from "):
        return stripped
    return ""


def _truncate(text: str, limit: int = _TOOL_OUTPUT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


def _run_code_sync(code: str, workdir: Path, script_name: str) -> tuple[bool, str]:
    """Execute one Python script inside the workdir with the service's interpreter."""
    script_path = workdir / script_name
    script_path.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=_CODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"Error: code execution timed out after {_CODE_TIMEOUT_SECONDS:.0f}s."
    output_parts = []
    if proc.stdout:
        output_parts.append(proc.stdout)
    if proc.stderr:
        output_parts.append(f"[stderr]\n{proc.stderr}")
    output = "\n".join(output_parts).strip()
    if proc.returncode != 0:
        return False, _truncate(f"Error: exit code {proc.returncode}\n{output}" if output else f"Error: exit code {proc.returncode}")
    return True, _truncate(output or "(no output)")


def _spreadsheet_preview(xlsx_path: Path, rows: int = _PREVIEW_ROWS) -> str:
    """Serialize the first rows of every sheet, upstream row_exec style."""
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(xlsx_path)
        parts: list[str] = []
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            parts.append(f"Sheet Name: {sheet_name}")
            parts.append(df.head(min(rows, df.shape[0])).to_string())
            parts.append("-" * 50)
        return _truncate("\n".join(parts), _PREVIEW_MAX_CHARS)
    except Exception as exc:
        return f"(failed to preview spreadsheet content: {exc})"


@dataclass
class _SolutionState:
    """Mutable per-rollout record of executed and submitted code."""

    last_successful_run: str | None = None
    accepted_solution: str | None = None
    run_counter: int = 0


def _make_run_python_tool(workdir: Path, state: _SolutionState):
    Tool = _vikingbot_imports()["Tool"]

    class RunPythonTool(Tool):
        @property
        def name(self) -> str:
            return "run_python"

        @property
        def description(self) -> str:
            return (
                "Execute a Python script in the task working directory and return its "
                "stdout/stderr. Use it to inspect the spreadsheet (openpyxl/pandas are "
                "available) and to develop the manipulation code. The script must be "
                "complete and self-contained; state does not persist between calls."
            )

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source code to execute.",
                    },
                },
                "required": ["code"],
            }

        async def execute(self, tool_context: Any, code: str, **kwargs: Any) -> str:
            del tool_context, kwargs
            state.run_counter += 1
            success, output = await asyncio.to_thread(
                _run_code_sync, code, workdir, f"step_{state.run_counter}.py"
            )
            if success:
                state.last_successful_run = code
            return output

    return RunPythonTool()


def _make_submit_solution_tool(
    workdir: Path,
    state: _SolutionState,
    *,
    task_id: str,
    source_input: Path,
):
    Tool = _vikingbot_imports()["Tool"]

    class SubmitSolutionTool(Tool):
        @property
        def name(self) -> str:
            return "submit_solution"

        @property
        def description(self) -> str:
            return (
                "Submit the final self-contained Python solution script. It is validated "
                "by running it against a FRESH copy of the input workbook, so it must not "
                "depend on files created by earlier runs. On success the solution is "
                "recorded and you should finish with a short final message."
            )

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python script that reads the input xlsx and writes "
                            "the output xlsx at the exact paths given in the task."
                        ),
                    },
                },
                "required": ["code"],
            }

        async def execute(self, tool_context: Any, code: str, **kwargs: Any) -> str:
            del tool_context, kwargs
            validation_dir = workdir / f"validate_{int(time.time() * 1000)}"
            validation_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_input, validation_dir / input_xlsx_name(task_id, 1))
            adjusted = code.replace(str(workdir), str(validation_dir))
            success, output = await asyncio.to_thread(
                _run_code_sync, adjusted, validation_dir, "solution.py"
            )
            produced = validation_dir / output_xlsx_name(task_id, 1)
            if not success:
                return (
                    "Solution NOT accepted: execution failed on a fresh input copy.\n"
                    f"{output}\nFix the script and submit again."
                )
            if not produced.exists():
                return (
                    "Solution NOT accepted: the script ran but did not create the output "
                    f"file {produced.name}. The script must write the output workbook at "
                    "the exact output path from the task. Fix and submit again."
                )
            state.accepted_solution = code
            return (
                "Solution accepted and recorded. Reply with a brief final summary "
                "(no more tool calls) to finish the task."
            )

    return SubmitSolutionTool()


def _configure_tools(
    agent: Any,
    *,
    workdir: Path,
    state: _SolutionState,
    task_id: str,
    source_input: Path,
) -> None:
    # Match the tau2 executor: no direct openviking_* tool access; experience
    # retrieval flows through the skill-declared tools only.
    for tool_name in list(agent.tools.tool_names):
        if str(tool_name).startswith("openviking_"):
            agent.tools.unregister(tool_name)
    if _codegen_enabled():
        # SkillOpt-style one-shot codegen: the model gets no tools at all.
        return
    if _experience_selector_enabled():
        agent.tools.register(_make_load_relevant_experience_tool(agent))
    elif not _experience_skill_disabled():
        agent.tools.register(_make_search_experience_tool())
        agent.tools.register(_make_read_experience_tool())
    agent.tools.register(_make_run_python_tool(workdir, state))
    agent.tools.register(
        _make_submit_solution_tool(workdir, state, task_id=task_id, source_input=source_input)
    )


def _formula_rule() -> str:
    return (
        "If a cell should contain a computed result, writing either the computed "
        "value or an equivalent formula is acceptable."
        if recalc_enabled()
        else "Always write computed VALUES into result cells, never formula strings: "
        "the grader reads cached cell values without recalculating, so a cell "
        "containing only a formula is read as empty and scored wrong."
    )


def _build_system_prompt() -> str:
    if _codegen_enabled():
        # SkillOpt-aligned one-shot codegen: mirrors SkillOpt's codegen_system
        # verbatim — no tools, no exploration, and deliberately NO domain hints
        # (formula-vs-value, answer-range) that would immunize the model against
        # the same pitfalls SkillOpt's baseline suffers, so the baseline is
        # comparable to SkillOpt's.
        return (
            "You are an expert Python programmer specializing in spreadsheet "
            "manipulation. You will be given a user instruction together with a preview "
            "of an input .xlsx file. Your job is to write a single self-contained Python "
            "script that reads the input file, performs the requested manipulation, and "
            "saves the result to the output path given in the task. Use only the standard "
            "library, openpyxl, and pandas. Do not print anything. Do not use input(). Do "
            "not hardcode file paths. Return ONLY the Python code inside a single "
            "```python ... ``` fenced block."
        )
    instructions = [
        "You are a spreadsheet expert who solves real-world spreadsheet manipulation "
        "tasks by writing Python code (openpyxl and pandas are available).",
        "Workflow: first inspect the input workbook with `run_python` (sheet names, "
        "headers, data shapes — the preview in the task only shows the first rows), "
        "then develop and test the manipulation code, and finally call "
        "`submit_solution` with ONE complete self-contained script that reads the "
        "input xlsx and writes the output xlsx at the exact paths given in the task.",
        "The submitted script is re-executed on fresh copies of the input workbook "
        "(including unseen test cases with different values), so it must implement the "
        "general procedure — never hardcode computed answer values into cells.",
        "Only modify or fill cells within the answer position range; preserve "
        "everything else, including all other sheets, when writing the output file.",
        _formula_rule(),
    ]
    if not _experience_skill_disabled():
        # There is no experience_loader skill file in the minimal context, so
        # the instructions reference the tools directly.
        if _experience_selector_enabled():
            instructions.append(
                "Before taking task actions, call the `load_relevant_experience` tool "
                "with a full description of the task to retrieve applicable past "
                "experiences."
            )
        else:
            instructions.append(
                "Before taking task actions, search case memories with the "
                "`search_experience` tool and read selected experiences using the "
                "`read_experience` tool."
            )
        instructions.append(
            "Loaded experiences are guidance from prior training runs. Use them only when "
            "their situation and applicability boundaries match the current task; the "
            "current task description and current tool results override prior experience."
        )
    instructions.append(
        "After `submit_solution` is accepted, finish with a brief plain-text summary and "
        "stop calling tools."
    )
    return "\n".join(instructions)


def _build_user_prompt(
    *,
    instruction: str,
    instruction_type: str,
    answer_position: str,
    answer_sheet: Any,
    input_path: Path,
    output_path: Path,
    preview: str,
) -> str:
    answer_sheet_line = (
        f"\n### answer_sheet\n{answer_sheet}\n" if answer_sheet else ""
    )
    return f"""Solve the following spreadsheet manipulation task.

### instruction
{instruction}

### spreadsheet_path (input, work on this file)
{input_path}

### spreadsheet_content (first rows of each sheet)
{preview}

### instruction_type
{instruction_type}

### answer_position
{answer_position}
{answer_sheet_line}
### output_path (write the modified workbook here)
{output_path}
"""


async def _run_codegen(
    *,
    agent: Any,
    messages: list[dict[str, Any]],
    state: _SolutionState,
    workdir: Path,
    input_path: Path,
    output_path: Path,
    session_key: Any,
):
    """SkillOpt run_multi (eval path): generate one script, execute it, and only
    on execution error feed the traceback back and retry — never any answer
    feedback. Records the last code that ran without error as the solution.

    Returns the same 5-tuple shape as ``AgentLoop._run_agent_loop``.
    """
    convo = list(messages)
    tools_used: list[dict[str, Any]] = []
    final_content = ""
    total_in = total_out = 0
    turns = 0

    for turn in range(_CODEGEN_MAX_TURNS):
        turns = turn + 1
        # Use the streaming path: the non-streaming provider.chat serializes on a
        # shared VLM client and deadlocks under concurrency, whereas chat_stream
        # (get_async_client) is concurrency-safe — same path the agent loop uses.
        raw = ""
        response = None
        async for event in agent.provider.chat_stream(
            messages=convo,
            tools=None,
            model=agent.model,
            max_tokens=16384,
            temperature=agent.temperature,
            session_id=session_key.safe_name(),
        ):
            if getattr(event, "type", None) == "response":
                response = event.response
        if response is not None:
            raw = str(getattr(response, "content", "") or "")
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_in += int(getattr(usage, "prompt_tokens", 0) or 0)
                total_out += int(getattr(usage, "completion_tokens", 0) or 0)
        final_content = raw
        code = _extract_code_block(raw)
        if not code:
            convo.append({"role": "assistant", "content": raw})
            convo.append(
                {
                    "role": "user",
                    "content": "No ```python``` code block found. Return ONE complete "
                    "script that reads the input xlsx and writes the output xlsx, inside a "
                    "single ```python ... ``` block.",
                }
            )
            continue

        success, run_output = await asyncio.to_thread(
            _run_code_sync, code, workdir, f"codegen_turn_{turns}.py"
        )
        tools_used.append(
            {
                "tool_name": "codegen",
                "args": json.dumps({"turn": turns}, ensure_ascii=False),
                "result": run_output,
                "execute_success": success,
                "input_token": 0,
                "output_token": 0,
            }
        )
        if success and output_path.exists():
            state.last_successful_run = code
            state.accepted_solution = code
            break
        # Execution error (or no output produced): feed it back, SkillOpt-style.
        problem = run_output if not success else (
            f"The script ran but did not create the output file {output_path.name}."
        )
        convo.append({"role": "assistant", "content": raw})
        convo.append(
            {
                "role": "user",
                "content": "The script failed during execution:\n\n"
                f"```\n{problem[:3000]}\n```\n\n"
                "Fix it and return one complete ```python``` script. Keep reading the input "
                "xlsx and writing the output xlsx at the exact paths given in the task.",
            }
        )

    token_usage = {"input_tokens": total_in, "output_tokens": total_out}
    return final_content, "", tools_used, token_usage, turns


async def _run_agent(
    *,
    agent: Any,
    system_prompt: str,
    user_prompt: str,
    session_key: Any,
    sender_id: str,
    state: _SolutionState,
    workdir: Path,
    input_path: Path,
    output_path: Path,
    timings: _RolloutTiming | None = None,
):
    """SkillOpt-aligned context: the agent sees ONLY the benchmark system prompt
    (+ the principles document appended as an authoritative ``## Experience``
    section) and the task user prompt — none of vikingbot's native scaffolding.
    Tool schemas still flow to the model through the normal tool registration.
    The principles read honours VIKINGBOT_PRINCIPLES_OVERRIDE_FILE, so gate
    candidate injection works unchanged."""
    stage_started_at = time.perf_counter()
    principles = ""
    try:
        workspace_id = agent.context._get_workspace_id(session_key)
        principles = await agent.context.memory.get_viking_principles_context(
            workspace_id=workspace_id,
            openviking_connection=agent.context._openviking_connection,
        )
    except Exception as exc:
        logger.debug("principles context unavailable: %s", exc)
    system_text = system_prompt
    if principles:
        system_text += "\n\n## Experience\n\n" + principles
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]
    if timings is not None:
        timings.record("build_messages", stage_started_at)

    stage_started_at = time.perf_counter()
    if _codegen_enabled():
        result = await _run_codegen(
            agent=agent,
            messages=messages,
            state=state,
            workdir=workdir,
            input_path=input_path,
            output_path=output_path,
            session_key=session_key,
        )
    else:
        result = await agent._run_agent_loop(
            messages=messages,
            session_key=session_key,
            publish_events=False,
            sender_id=sender_id,
            ov_tools_enable=False,
        )
    if timings is not None:
        timings.record("agent_loop", stage_started_at)
    final_content, final_reasoning_content, tools_used, token_usage, iteration = result
    memory_content = _merge_memories(None, _case_memory_context_from_tools(tools_used))
    return (
        final_content,
        final_reasoning_content,
        tools_used,
        token_usage,
        iteration,
        memory_content,
    )


def _discover_test_cases(spreadsheet_dir: Path, task_id: str) -> list[int]:
    """Test case numbers actually shipped for this task, from files on disk.

    Releases differ: all_data_912 ships 3 (input, answer) pairs per task while
    verified_400 ships exactly 1. Hardcoding a count would auto-fail the missing
    cases, so the pass criterion is "pass every case that exists".
    """
    cases = []
    for tc in range(1, 10):
        if (spreadsheet_dir / input_xlsx_name(task_id, tc)).exists() and (
            spreadsheet_dir / answer_xlsx_name(task_id, tc)
        ).exists():
            cases.append(tc)
    return cases


def _evaluate_solution(
    *,
    solution: str | None,
    workdir: Path,
    spreadsheet_dir: Path,
    task_id: str,
    answer_position: str,
    answer_sheet: Any,
) -> dict[str, Any]:
    """Run the accepted solution on fresh copies of every shipped test case."""
    test_cases = _discover_test_cases(spreadsheet_dir, task_id)
    if not test_cases:
        return {
            "reward": 0.0,
            "soft": 0.0,
            "per_case": [{"test_case": 1, "passed": False, "detail": "benchmark files missing"}],
        }
    per_case: list[dict[str, Any]] = []
    if not solution:
        return {
            "reward": 0.0,
            "soft": 0.0,
            "per_case": [
                {"test_case": tc, "passed": False, "detail": "no solution submitted"}
                for tc in test_cases
            ],
        }
    for tc in test_cases:
        eval_dir = workdir / f"eval_tc{tc}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        input_name = input_xlsx_name(task_id, tc)
        output_name = output_xlsx_name(task_id, tc)
        source_input = spreadsheet_dir / input_name
        answer_file = spreadsheet_dir / answer_xlsx_name(task_id, tc)
        shutil.copy2(source_input, eval_dir / input_name)
        # Upstream protocol: the recorded solution references the first test
        # case's filenames; substitute both the directory and the tc index.
        code = solution.replace(str(workdir), str(eval_dir))
        code = code.replace(input_xlsx_name(task_id, 1), input_name)
        code = code.replace(output_xlsx_name(task_id, 1), output_name)
        success, run_output = _run_code_sync(code, eval_dir, "solution.py")
        if not success:
            per_case.append(
                {"test_case": tc, "passed": False, "detail": f"execution failed: {run_output[:500]}"}
            )
            continue
        try:
            passed, detail = compare_workbooks(
                answer_file,
                eval_dir / output_name,
                answer_position,
                answer_sheet,
            )
        except Exception as exc:
            passed, detail = False, f"comparison error: {exc}"
        per_case.append({"test_case": tc, "passed": bool(passed), "detail": detail})
    passes = sum(1 for item in per_case if item["passed"])
    return {
        "reward": 1.0 if passes == len(per_case) and per_case else 0.0,
        "soft": passes / len(per_case) if per_case else 0.0,
        "per_case": per_case,
    }


def _ssb_evaluation(reward: float, evaluation_result: dict[str, Any]) -> RubricEvaluation:
    passed = reward >= 1.0
    feedback = [] if passed else [
        json.dumps(evaluation_result.get("per_case", []), ensure_ascii=False)
    ]
    return RubricEvaluation(
        passed=passed,
        score=reward,
        criterion_results=[
            CriterionResult(
                criterion_name="ssb_hard_restriction",
                passed=passed,
                score=reward,
                feedback=feedback,
                evidence=[json.dumps(evaluation_result, ensure_ascii=False)],
                metadata={"reward": reward, "soft": evaluation_result.get("soft")},
            )
        ],
        feedback=feedback,
        metadata={"source": "spreadsheetbench_executor", "reward": reward, **evaluation_result},
    )


def _text_message(message_id: str, role: str, text: str) -> Message:
    return Message(id=message_id, role=role, parts=[TextPart(text=text)])


def _build_rollout_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    tools_used: Any,
    final_content: str | None,
    evaluation_result: dict[str, Any],
    reward: float,
) -> list[Message]:
    messages = [_text_message("ssb-system", "user", f"system:\n{system_prompt}")]
    messages.append(_text_message("ssb-user", "user", user_prompt))
    if isinstance(tools_used, list):
        for idx, tool_info in enumerate(tools_used):
            if not isinstance(tool_info, dict):
                continue
            tool_name = str(tool_info.get("tool_name") or "unknown")
            args = tool_info.get("args", "")
            if isinstance(args, str):
                try:
                    tool_input = json.loads(args)
                except json.JSONDecodeError:
                    tool_input = {"arguments": args}
            else:
                tool_input = args if isinstance(args, dict) else {"arguments": args}
            result = tool_info.get("result")
            has_result = result is not None
            messages.append(
                Message(
                    id=f"ssb-tool-{idx}",
                    role="user" if has_result else "assistant",
                    parts=[
                        ToolPart(
                            tool_id=f"ssb-tool-{idx}",
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_output=str(result) if has_result else "",
                            tool_status="completed" if has_result else "running",
                        )
                    ],
                )
            )
    if final_content and str(final_content).strip():
        messages.append(_text_message("ssb-final", "assistant", str(final_content)))
    success = reward >= 1.0
    messages.append(
        _text_message(
            "ssb-reward",
            "user",
            f"task_success: {success}\ntask_reward: {reward}\n"
            f"evaluation report: {json.dumps(evaluation_result, ensure_ascii=False)}",
        )
    )
    return messages


@dataclass
class VikingBotSpreadsheetBenchRolloutExecutor:
    """Execute SpreadsheetBench cases with the VikingBot agent loop."""

    config_path: str | None = None
    concurrency: int = 20
    max_iterations: int = 30
    log_timings: bool = True

    async def execute(
        self,
        cases: list[Case],
        policy_set: ExperienceSet,
        context: ExecutionContext,
    ) -> list[Rollout]:
        del policy_set
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(case: Case) -> Rollout:
            async with semaphore:
                return await self._execute_one_async(case, context)

        return list(await asyncio.gather(*(run_one(case) for case in cases)))

    async def _execute_one_async(self, case: Case, context: ExecutionContext) -> Rollout:
        task_id = str(case.input["task_id"])
        task_no = int(case.input["task_no"])
        data_split = str(case.input["data_split"])
        instruction = str(case.input.get("instruction") or "")
        instruction_type = str(case.input.get("instruction_type") or "")
        answer_position = str(case.input.get("answer_position") or "")
        answer_sheet = case.input.get("answer_sheet")
        spreadsheet_dir = Path(str(case.input["spreadsheet_dir"]))
        trial = case.input.get("eval_trial", case.input.get("train_trial"))

        timings = _RolloutTiming(case=case.name, enabled=self.log_timings)
        total_started_at = time.perf_counter()

        workdir = Path(
            tempfile.mkdtemp(prefix=f"ssb_{_safe_session_fragment(case.name)}_", dir=_work_root())
        )
        source_input = spreadsheet_dir / input_xlsx_name(task_id, 1)
        input_path = workdir / input_xlsx_name(task_id, 1)
        output_path = workdir / output_xlsx_name(task_id, 1)
        shutil.copy2(source_input, input_path)

        stage_started_at = time.perf_counter()
        preview = await asyncio.to_thread(_spreadsheet_preview, input_path)
        timings.record("preview", stage_started_at)

        stage_started_at = time.perf_counter()
        agent = await asyncio.to_thread(
            _build_agent, self.config_path, max_iterations=self.max_iterations
        )
        timings.record("build_agent", stage_started_at)

        state = _SolutionState()
        _configure_tools(
            agent,
            workdir=workdir,
            state=state,
            task_id=task_id,
            source_input=source_input,
        )

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            instruction=instruction,
            instruction_type=instruction_type,
            answer_position=answer_position,
            answer_sheet=answer_sheet,
            input_path=input_path,
            output_path=output_path,
            preview=preview,
        )
        SessionKey = _vikingbot_imports()["SessionKey"]
        trial_suffix = "" if trial is None else f"_r{int(trial)}"
        stage = _safe_session_fragment(str(context.metadata.get("stage") or "rollout"))
        session_key = SessionKey(
            type="cli",
            channel_id="ssb",
            chat_id=f"ssb_{stage}_{data_split}_{task_no}{trial_suffix}",
        )

        (
            final_content,
            final_reasoning_content,
            tools_used,
            token_usage,
            iteration,
            memory_content,
        ) = await _run_agent(
            agent=agent,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_key=session_key,
            sender_id="ssb_user",
            state=state,
            workdir=workdir,
            input_path=input_path,
            output_path=output_path,
            timings=timings,
        )

        solution = state.accepted_solution or state.last_successful_run
        solution_source = (
            "submitted" if state.accepted_solution else ("last_successful_run" if solution else "none")
        )
        stage_started_at = time.perf_counter()
        evaluation_result = await asyncio.to_thread(
            _evaluate_solution,
            solution=solution,
            workdir=workdir,
            spreadsheet_dir=spreadsheet_dir,
            task_id=task_id,
            answer_position=answer_position,
            answer_sheet=answer_sheet,
        )
        timings.record("reward", stage_started_at)
        evaluation_result["solution_source"] = solution_source
        reward = float(evaluation_result["reward"])

        rollout = Rollout(
            case=case,
            messages=_build_rollout_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools_used=tools_used,
                final_content=final_content,
                evaluation_result=evaluation_result,
                reward=reward,
            ),
            policy_snapshot_id=context.policy_snapshot_id,
            evaluation=_ssb_evaluation(reward, evaluation_result),
            metadata={
                "domain": str(case.input.get("domain")),
                "data_split": data_split,
                "task_no": task_no,
                "task_id": task_id,
                "instruction_type": instruction_type,
                "eval_trial": case.input.get("eval_trial"),
                "eval_trial_count": case.input.get("eval_trial_count"),
                "train_trial": case.input.get("train_trial"),
                "train_trial_count": case.input.get("train_trial_count"),
                "original_case_name": case.input.get("original_case_name"),
                "reward": reward,
                "evaluation_result": evaluation_result,
                "solution": solution,
                "solution_source": solution_source,
                "tools_used": tools_used,
                "token_usage": token_usage,
                "iterations": iteration,
                "memory": memory_content,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "final_content": final_content,
                "final_reasoning_content": final_reasoning_content,
                "execution_metadata": dict(context.metadata),
                "workdir": str(workdir) if _keep_workdirs() else None,
            },
        )
        timings.log_summary(
            total_ms=_elapsed_ms(total_started_at),
            task_id=task_id,
            task_no=task_no,
            data_split=data_split,
            iterations=iteration,
            reward=reward,
            message_count=len(rollout.messages),
        )
        rollout.metadata["timing_ms"] = timings.snapshot(
            total_ms=_elapsed_ms(total_started_at),
            iterations=iteration,
        )
        if not _keep_workdirs():
            shutil.rmtree(workdir, ignore_errors=True)
        return rollout


# Convenience alias mirroring the tau2 module.
SpreadsheetBenchRolloutExecutor = VikingBotSpreadsheetBenchRolloutExecutor
