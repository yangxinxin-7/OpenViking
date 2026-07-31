# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Principles consolidation: SkillOpt-style maintenance of one always-on
experience document, learned from trajectory memories.

The training pipeline commits rollouts to OpenViking as usual, producing
trajectory memories (``memories/trajectories/``) through the normal extraction
loop. This module maintains a SEPARATE document — ``memories/principles/`` —
that plays the role of SkillOpt's skill file: a seed document describing how to
work in the environment, incrementally revised by an analyst that reads the
accumulated trajectory memories.

``memories/principles/`` is a small state machine:

- ``default.md`` — the only document ever injected into agent context. It only
  changes through :func:`promote_principles`, so every revision it holds has
  passed an external evaluation gate. The initial seed version is written by
  the external runner (see ``gate_principles.py seed``), mirroring SkillOpt's
  ungated initial skill.
- ``pending.md`` — a single-slot candidate produced by
  :meth:`PrinciplesConsolidator.propose`: the analyst VLM reads the trajectory
  memories accumulated since the last proposal and returns a fully revised
  document plus a change summary. It is never injected. While a pending
  candidate exists, no new proposal is generated.
- ``rejected.md`` — an append-only log of change batches that failed the gate,
  fed back into later proposals so the analyst does not re-propose them.
- ``history/v{N}.md`` — archived promoted versions.
- ``state.json`` — ``{"version", "consolidated_trajectories"}``; the watermark
  advances on promote AND reject, so a rejected batch is only retried once
  fresh trajectories accumulate.

Cadence mirrors SkillOpt's training step: the external orchestrator runs a
training batch (rollouts committed, trajectory memories extracted), then calls
``POST /api/v1/memories/principles/consolidate`` to propose from that batch's
trajectories, then gates the candidate on the selection set and reports the
verdict via promote/reject — one full cycle per training batch, no background
auto-trigger.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Tuple

import yaml

from openviking.prompts.manager import PromptManager
from openviking.session.memory.memory_type_registry import resolve_memory_templates_dir
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

PRINCIPLES_FILENAME = "default.md"
PENDING_FILENAME = "pending.md"
REJECTED_FILENAME = "rejected.md"
STATE_FILENAME = "state.json"


def principles_dir(user_space: str) -> str:
    return f"viking://user/{user_space}/memories/principles"


def trajectories_dir(user_space: str) -> str:
    return f"viking://user/{user_space}/memories/trajectories"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _read_text(viking_fs: Any, uri: str, ctx: Any) -> str:
    """Read a text file, returning "" when absent or unreadable."""
    try:
        return (await viking_fs.read(uri, ctx=ctx)).decode("utf-8")
    except Exception:
        return ""


def _parse_memory_fields(document: str) -> Dict[str, Any]:
    match = re.search(r"<!--\s*MEMORY_FIELDS\s*\n(.*?)\n-->", document, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _read_state(viking_fs: Any, user_space: str, ctx: Any) -> Dict[str, int]:
    raw = await _read_text(viking_fs, f"{principles_dir(user_space)}/{STATE_FILENAME}", ctx)
    try:
        state = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        state = {}
    if not isinstance(state, dict):
        state = {}
    return {
        "version": int(state.get("version") or 0),
        "consolidated_trajectories": int(state.get("consolidated_trajectories") or 0),
    }


async def _write_state(viking_fs: Any, user_space: str, state: Dict[str, int], ctx: Any) -> None:
    await viking_fs.mkdir(principles_dir(user_space), exist_ok=True, ctx=ctx)
    await viking_fs.write(
        f"{principles_dir(user_space)}/{STATE_FILENAME}",
        json.dumps(state, ensure_ascii=False),
        ctx=ctx,
    )


def _load_consolidation_spec() -> Dict[str, Any]:
    """Read the consolidation block from the principles template yaml."""
    for base in (
        resolve_memory_templates_dir(),
        PromptManager._get_bundled_templates_dir() / "memory",
    ):
        path = base / "principles.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            spec = data.get("consolidation")
            if isinstance(spec, dict):
                return spec
    raise RuntimeError("principles.yaml with a consolidation section was not found")


def _render(template: str, values: Dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
    return rendered


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _strip_memory_fields(document: str) -> str:
    return re.sub(r"\n*<!--\s*MEMORY_FIELDS.*?-->\n*", "\n", document, flags=re.DOTALL).strip()


def _with_memory_fields(body: str, metadata: Dict[str, Any]) -> str:
    return (
        body.strip()
        + "\n\n<!-- MEMORY_FIELDS\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n-->\n"
    )


def _gate_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for key in ("selection_id", "net_wins", "score", "baseline_score", "notes"):
        if report.get(key) is not None:
            summary[key] = report[key]
    return summary


async def _list_memory_markdown(viking_fs: Any, directory: str, ctx: Any) -> List[str]:
    """Sorted markdown file URIs under a memory directory.

    Uses ``ls`` rather than ``glob`` because some storage backends (RAGFS
    bindings) do not implement glob_directory. Sorted by name so the
    count-based watermark in ``state.json`` maps to a stable prefix
    (trajectory filenames embed the session timestamp).
    """
    entries = await viking_fs.ls(directory, node_limit=10000, ctx=ctx)
    uris: List[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("isDir"):
            continue
        name = str(entry.get("name") or "")
        if not name.endswith(".md") or name.startswith("."):
            continue
        uris.append(str(entry.get("uri") or f"{directory.rstrip('/')}/{name}"))
    return sorted(uris)


# ── gate verdicts ──────────────────────────────────────────────────────────


async def promote_principles(
    *, user_space: str, report: Dict[str, Any], ctx: Any = None
) -> Dict[str, Any]:
    """Promote the pending candidate to ``default.md`` after it passed the gate.

    Optimistic-concurrency checks: the caller must echo the candidate's sha256
    (proving it gated exactly this content) and the candidate must still be
    based on the current verified version.
    """
    from openviking.storage.viking_fs import get_viking_fs

    viking_fs = get_viking_fs()
    pdir = principles_dir(user_space)
    pending_uri = f"{pdir}/{PENDING_FILENAME}"
    default_uri = f"{pdir}/{PRINCIPLES_FILENAME}"

    pending = await _read_text(viking_fs, pending_uri, ctx)
    if not pending.strip():
        return {"status": "error", "reason": "no pending candidate"}
    sha = _sha256(pending)
    if str(report.get("pending_sha256") or "") != sha:
        return {"status": "conflict", "reason": "pending changed since the gate run started"}

    metadata = _parse_memory_fields(pending)
    state = await _read_state(viking_fs, user_space, ctx)
    base_version = int(metadata.get("base_version") or 0)
    if base_version != state["version"]:
        return {
            "status": "conflict",
            "reason": f"candidate based on v{base_version}, current is v{state['version']}",
        }

    old_default = await _read_text(viking_fs, default_uri, ctx)
    if old_default.strip():
        await viking_fs.mkdir(f"{pdir}/history", exist_ok=True, ctx=ctx)
        await viking_fs.write(f"{pdir}/history/v{state['version']}.md", old_default, ctx=ctx)

    new_version = state["version"] + 1
    metadata.pop("base_version", None)
    metadata.update(
        {
            "status": "verified",
            "version": new_version,
            "gate": _gate_summary(report),
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    await viking_fs.write(
        default_uri, _with_memory_fields(_strip_memory_fields(pending), metadata), ctx=ctx
    )
    await _write_state(
        viking_fs,
        user_space,
        {
            "version": new_version,
            "consolidated_trajectories": int(metadata.get("source_trajectories") or 0),
        },
        ctx,
    )
    await viking_fs.rm(pending_uri, ctx=ctx)
    logger.info("principles promoted to v%s (gate: %s)", new_version, _gate_summary(report))
    return {"status": "ok", "version": new_version, "uri": default_uri}


async def reject_principles(
    *, user_space: str, report: Dict[str, Any], ctx: Any = None
) -> Dict[str, Any]:
    """Discard the pending candidate after it failed the gate.

    The batch's change summary is appended to ``rejected.md`` (negative
    feedback for later proposals) and the trajectory watermark advances, so the
    same batch is only reconsidered once fresh trajectories accumulate.
    """
    from openviking.storage.viking_fs import get_viking_fs

    viking_fs = get_viking_fs()
    pdir = principles_dir(user_space)
    pending_uri = f"{pdir}/{PENDING_FILENAME}"

    pending = await _read_text(viking_fs, pending_uri, ctx)
    if not pending.strip():
        return {"status": "error", "reason": "no pending candidate"}
    if str(report.get("pending_sha256") or "") != _sha256(pending):
        return {"status": "conflict", "reason": "pending changed since the gate run started"}

    metadata = _parse_memory_fields(pending)
    edits = metadata.get("edits") if isinstance(metadata.get("edits"), list) else []
    summary = "; ".join(str(e) for e in edits)
    gate = _gate_summary(report)
    line = (
        f"- {time.strftime('%Y-%m-%dT%H:%M:%S%z')} base_v{metadata.get('base_version', '?')} "
        f"net_wins={gate.get('net_wins', '?')} score={gate.get('score', '?')}: {summary or '(no change summary)'}"
    )
    rejected_uri = f"{pdir}/{REJECTED_FILENAME}"
    existing = await _read_text(viking_fs, rejected_uri, ctx)
    await viking_fs.mkdir(pdir, exist_ok=True, ctx=ctx)
    await viking_fs.write(
        rejected_uri, (existing.rstrip() + "\n" if existing.strip() else "") + line + "\n", ctx=ctx
    )

    state = await _read_state(viking_fs, user_space, ctx)
    await _write_state(
        viking_fs,
        user_space,
        {
            "version": state["version"],
            "consolidated_trajectories": int(metadata.get("source_trajectories") or 0),
        },
        ctx,
    )
    await viking_fs.rm(pending_uri, ctx=ctx)
    logger.info("principles candidate rejected (gate: %s)", gate)
    return {"status": "ok", "rejected_edits": len(edits)}


# ── proposal ───────────────────────────────────────────────────────────────


class PrinciplesConsolidator:
    """SkillOpt-style analyst: revises the experience document from trajectories."""

    def __init__(self, vlm: Any):
        self.vlm = vlm
        self.spec = _load_consolidation_spec()

    async def propose(self, *, user_space: str, ctx: Any = None) -> Dict[str, Any]:
        """Produce a gated candidate: a full revised document written to
        ``pending.md`` for external validation.

        Single-slot semantics: refuses to overwrite an existing pending
        candidate. Never touches ``default.md``.
        """
        from openviking.storage.viking_fs import get_viking_fs

        viking_fs = get_viking_fs()
        pdir = principles_dir(user_space)

        pending_uri = f"{pdir}/{PENDING_FILENAME}"
        if (await _read_text(viking_fs, pending_uri, ctx)).strip():
            return {"status": "skipped", "reason": "pending candidate already in flight"}

        uris = await _list_memory_markdown(viking_fs, trajectories_dir(user_space), ctx)
        total = len(uris)
        state = await _read_state(viking_fs, user_space, ctx)
        min_trajectories = int(self.spec.get("min_trajectories", 5))
        new_uris = uris[state["consolidated_trajectories"] :]
        if len(new_uris) < min_trajectories:
            return {
                "status": "skipped",
                "reason": f"only {len(new_uris)} new trajectories (< {min_trajectories})",
                "trajectories": total,
            }
        # Consume oldest-first up to the cap; the remainder stays above the
        # watermark and is picked up by the next cycle instead of being skipped.
        max_material = int(self.spec.get("max_trajectories_per_proposal", 120))
        material = new_uris[:max_material]
        consumed_watermark = state["consolidated_trajectories"] + len(material)
        failures, successes = await self._trajectory_digests(viking_fs, material, ctx)
        if not failures and not successes:
            return {
                "status": "skipped",
                "reason": "no readable trajectories",
                "trajectories": total,
            }

        current_doc = _strip_memory_fields(
            await _read_text(viking_fs, f"{pdir}/{PRINCIPLES_FILENAME}", ctx)
        )
        rejected_log = await _read_text(viking_fs, f"{pdir}/{REJECTED_FILENAME}", ctx)

        # SkillOpt-style two stages: per-minibatch analysts collect suggestions,
        # then one merge call turns them into a single revised document.
        suggestions = await self._collect_suggestions(current_doc, failures, successes)
        if not suggestions:
            return {
                "status": "skipped",
                "reason": "analysts proposed no changes",
                "trajectories": total,
            }
        revised, edits = await self._merge_revision(current_doc, suggestions, rejected_log)
        if not revised or not edits:
            return {
                "status": "skipped",
                "reason": "merge produced no changes",
                "trajectories": total,
            }

        document = _with_memory_fields(
            revised,
            {
                "memory_type": "principles",
                "status": "pending",
                "proposal_source": "trajectories",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "base_version": state["version"],
                "source_trajectories": consumed_watermark,
                "edits": edits,
            },
        )
        await viking_fs.mkdir(pdir, exist_ok=True, ctx=ctx)
        await viking_fs.write(pending_uri, document, ctx=ctx)
        logger.info(
            "principles proposal: %s changes from %s trajectories "
            "(%s failure / %s success, base v%s) -> %s",
            len(edits),
            len(failures) + len(successes),
            len(failures),
            len(successes),
            state["version"],
            pending_uri,
        )
        return {
            "status": "pending",
            "uri": pending_uri,
            "base_version": state["version"],
            "pending_sha256": _sha256(document),
            "edits": edits,
            "trajectories": total,
        }

    async def _trajectory_digests(
        self, viking_fs: Any, uris: List[str], ctx: Any
    ) -> Tuple[List[str], List[str]]:
        """(failure_digests, success_digests), one digest per trajectory memory.

        Trajectory memories are already compact generalized contracts (the
        extraction loop did the summarizing), so the digest is the document
        itself, truncated defensively.
        """
        max_chars = int(self.spec.get("max_chars_per_trajectory", 3000))
        failures: List[str] = []
        successes: List[str] = []
        for uri in uris:
            content = await _read_text(viking_fs, uri, ctx)
            if not content.strip():
                continue
            outcome = str(_parse_memory_fields(content).get("outcome") or "unknown").lower()
            body = _strip_memory_fields(content)[:max_chars]
            name = uri.rsplit("/", 1)[-1].removesuffix(".md")
            digest = f"### [{outcome.upper()}] {name}\n{body}"
            (successes if outcome == "success" else failures).append(digest)
        return failures, successes

    async def _collect_suggestions(
        self, current_doc: str, failures: List[str], successes: List[str]
    ) -> List[str]:
        """Stage 1 — per-minibatch analyst calls, SkillOpt's REFLECT phase.

        Failure and success minibatches get separate prompts (mirroring
        SkillOpt's analyst_error.md / analyst_success.md): a batch of passing
        trajectories has no failure to ground a warning in, so demanding one
        would be a category error. Calls run concurrently; each returns a list
        of suggestion strings, possibly empty.
        """
        import asyncio

        size = max(1, int(self.spec.get("minibatch_size", 8)))
        failure_prompt = str(self.spec.get("analyze_failure_prompt", ""))
        success_prompt = str(self.spec.get("analyze_success_prompt", ""))

        async def analyze(kind: str, prompt: str, batch: List[str]) -> List[str]:
            sections = [
                "# CURRENT DOCUMENT",
                current_doc or "(empty)",
                "",
                "# TRAJECTORY MEMORIES",
                *batch,
            ]
            try:
                response = await self.vlm.get_completion_async(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "\n".join(sections)},
                    ],
                )
            except Exception as exc:
                logger.warning("principles %s analyst failed: %s", kind, exc)
                return []
            parsed = _parse_json_object(getattr(response, "content", response))
            return [
                f"[{kind}] {str(s).strip()}"
                for s in parsed.get("suggestions") or []
                if str(s).strip()
            ]

        tasks = []
        for start in range(0, len(failures), size):
            tasks.append(analyze("failure", failure_prompt, failures[start : start + size]))
        for start in range(0, len(successes), size):
            tasks.append(analyze("success", success_prompt, successes[start : start + size]))
        results = await asyncio.gather(*tasks) if tasks else []
        return [suggestion for batch in results for suggestion in batch]

    async def _merge_revision(
        self, current_doc: str, suggestions: List[str], rejected_log: str
    ) -> Tuple[str, List[str]]:
        """Stage 2 — one merge call, SkillOpt's AGGREGATE+SELECT+UPDATE phases.

        Deduplicates the minibatch suggestions, keeps at most ``max_edits`` of
        them, and returns the fully revised document plus change summaries.
        """
        values = {
            "max_edits": self.spec.get("max_edits", 4),
            "max_chars": self.spec.get("max_chars", 8000),
        }
        prompt = _render(str(self.spec.get("merge_prompt", "")), values)

        sections = ["# CURRENT DOCUMENT", current_doc or "(empty — build from scratch)", ""]
        sections.append("# PREVIOUSLY REJECTED CHANGES (do not re-propose similar changes)")
        rejected_tail = rejected_log.strip().splitlines()[-40:]
        sections.extend(rejected_tail if rejected_tail else ["(none)"])
        sections.append("")
        sections.append("# SUGGESTIONS FROM TRAJECTORY ANALYSTS")
        sections.extend(f"{i}. {s}" for i, s in enumerate(suggestions, start=1))

        response = await self.vlm.get_completion_async(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(sections)},
            ],
        )
        parsed = _parse_json_object(getattr(response, "content", response))
        edits = [str(e).strip() for e in parsed.get("edits") or [] if str(e).strip()]
        revised = str(parsed.get("document") or "").strip()
        if not revised or not edits:
            return "", []
        max_chars = int(self.spec.get("max_chars", 8000))
        if len(revised) > max_chars:
            logger.warning(
                "principles proposal dropped: revised document %s chars > max %s",
                len(revised),
                max_chars,
            )
            return "", []
        max_edits = int(self.spec.get("max_edits", 4))
        return revised, edits[:max_edits]
