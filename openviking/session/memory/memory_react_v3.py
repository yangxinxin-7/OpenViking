# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Memory Experience Extractor V3 - two-stage experience memory.

Flow:
  Stage 1 (routing): Given conversation + ls + search results, decide:
    - no_op:  nothing worth extracting
    - add:    new generalizable experience, extract title/situation/lesson
    - update: refines an existing case, identify target_uri

  Stage 2 (synthesis, update-only):
    Given existing case content + historical trajectories + current conversation,
    produce the merged title/situation/lesson.

No tool-calling loop. Stage 2 only runs when Stage 1 decides "update".
"""

import json
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple

from openviking.models.vlm.base import VLMBase
from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryOperations
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.utils import (
    detect_language_from_conversation,
    extract_json_from_markdown,
    parse_json_with_stability,
    parse_memory_file_with_fields,
)
from openviking.session.memory.utils.content import deserialize_full
from openviking.storage.viking_fs import VikingFS, get_viking_fs
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

logger = get_logger(__name__)

MAX_TRAJECTORIES = 5
MAX_STAGE1_RECENT_CASES = 6
MAX_STAGE1_SEARCH_CASES = 10


class MemoryReActV3:
    """
    Two-stage experience extractor for V3.

    Stage 1 (routing): cheap LLM call — decide no_op / add / update.
    Stage 2 (synthesis): only for "update" — merge conversation with existing
                          experience content and historical trajectories.
    """

    MEMORY_TYPE = "cases"

    def __init__(
        self,
        vlm: VLMBase,
        viking_fs: Optional[VikingFS] = None,
        max_iterations: int = 5,  # kept for API compatibility, unused
        ctx: Optional[RequestContext] = None,
        registry: Optional[MemoryTypeRegistry] = None,
        trajectory_id: str = "",
    ):
        self.vlm = vlm
        self.viking_fs = viking_fs or get_viking_fs()
        self.ctx = ctx
        self.registry = registry or MemoryTypeRegistry()
        self._trajectory_id = trajectory_id

        self._schema_gen = SchemaModelGenerator(self.registry)
        self._schema_gen.generate_all_models()

        self._last_stage1_candidates: List[Dict[str, str]] = []

        # Set by run() after Stage 1; compressor stores this alongside trajectory_id.
        self.outcome: str = ""

    # ──────────────────────────────── helpers ─────────────────────────────────

    def _agent_space(self) -> str:
        return self.ctx.user.agent_space_name() if self.ctx and self.ctx.user else "default"

    def _cases_dir(self) -> str:
        schema = self._case_schema()
        if schema and schema.directory:
            return schema.directory.format(agent_space=self._agent_space())
        return f"viking://agent/{self._agent_space()}/memories/{self.MEMORY_TYPE}"

    def _traj_dir(self) -> str:
        return f"viking://agent/{self._agent_space()}/trajectories"

    @staticmethod
    def _generate_memory_id() -> str:
        return uuid4().hex[:12]

    @staticmethod
    def _prompt_template_path(name: str) -> Path:
        return Path(__file__).resolve().parents[2] / "prompts" / "templates" / "memory_v3" / name

    @classmethod
    def _load_prompt_template(cls, name: str) -> str:
        return cls._prompt_template_path(name).read_text(encoding="utf-8")

    @staticmethod
    def _render_prompt_template(template: str, variables: Dict[str, str]) -> str:
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
        return rendered

    # ──────────────────────────────── pre-fetch ───────────────────────────────

    async def _pre_fetch(self, conversation: str) -> Tuple[str, str]:
        """Build recent/search candidate case summaries for stage 1."""
        cases_dir = self._cases_dir()
        recent_candidates: List[Dict[str, str]] = []
        search_candidates: List[Dict[str, str]] = []
        seen_uris = set()

        try:
            entries = await self.viking_fs.ls(
                cases_dir,
                output="agent",
                abs_limit=256,
                show_all_hidden=False,
                node_limit=1000,
                ctx=self.ctx,
            )
            file_entries = [
                e for e in entries
                if not e.get("isDir", False)
                and ((e.get("name", "") or e.get("uri", "")).endswith(".md"))
                and not ((e.get("name", "") or e.get("uri", "")).rsplit("/", 1)[-1].startswith("."))
            ]
            file_entries.sort(key=lambda e: e.get("modTime", ""), reverse=True)
            for entry in file_entries[:MAX_STAGE1_RECENT_CASES]:
                uri = entry.get("uri") or f"{cases_dir}/{entry.get('name', '')}"
                candidate = await self._build_case_candidate(uri, source="recent")
                if candidate:
                    recent_candidates.append(candidate)
                    seen_uris.add(uri)
        except Exception as e:
            logger.warning(f"Pre-fetch recent cases failed: {e}")

        user_msgs = [
            line[len("[user]:"):].strip()
            for line in conversation.split("\n")
            if line.startswith("[user]:")
        ]
        query = " ".join(user_msgs)
        if query and self.ctx:
            try:
                result = await self.viking_fs.search(
                    query,
                    target_uri=cases_dir,
                    limit=MAX_STAGE1_SEARCH_CASES,
                    ctx=self.ctx,
                )
                result_dict = result.to_dict() if hasattr(result, "to_dict") else result
                items = result_dict if isinstance(result_dict, list) else result_dict.get("items", []) if isinstance(result_dict, dict) else []
                for item in items[:MAX_STAGE1_SEARCH_CASES]:
                    uri = item.get("uri") or item.get("target_uri") or ""
                    if (
                        not uri
                        or uri in seen_uris
                        or not uri.endswith(".md")
                        or uri.endswith(".overview.md")
                        or uri.endswith(".abstract.md")
                        or f"/{self.MEMORY_TYPE}/" not in uri
                    ):
                        continue
                    candidate = await self._build_case_candidate(uri, source="search")
                    if candidate:
                        search_candidates.append(candidate)
                        seen_uris.add(uri)
            except Exception as e:
                logger.warning(f"Pre-fetch search failed: {e}")

        combined_candidates = recent_candidates + search_candidates
        self._last_stage1_candidates = combined_candidates
        return (
            self._format_case_candidates(recent_candidates),
            self._format_case_candidates(search_candidates),
        )

    # ────────────────────────────── trajectories ──────────────────────────────

    @staticmethod
    def _parse_traj_entries(raw: Any) -> List[Dict[str, str]]:
        """Normalize trajectory_ids to list of {id, outcome} dicts.

        Supports legacy formats: comma-separated string, list of strings,
        and the current list-of-dicts format.
        """
        if isinstance(raw, str):
            raw = [t.strip() for t in raw.split(",") if t.strip()]
        if not isinstance(raw, list):
            return []
        return [
            e if isinstance(e, dict) else {"id": e, "outcome": ""}
            for e in raw
            if e
        ]

    async def _fetch_trajectories(self, case_uri: str) -> str:
        """Read historical trajectories linked to a case file.

        Trajectories are sorted so successful ones appear first (higher signal
        for lesson extraction). Each section is prefixed with its outcome label.
        Returns formatted text, or empty string if none found.
        """
        try:
            content = await self.viking_fs.read_file(case_uri, ctx=self.ctx)
            _, metadata = deserialize_full(content or "")
            if not metadata:
                return ""

            entries = self._parse_traj_entries(metadata.get("trajectory_ids", []))
            if not entries:
                return ""

            def _ts(entry: Dict[str, str]) -> int:
                try:
                    return int(entry["id"].rsplit("_", 1)[-1])
                except (KeyError, ValueError, IndexError):
                    return 0

            # Keep the most recent N, then show successes before failures
            recent = sorted(entries, key=_ts)[-MAX_TRAJECTORIES:]
            recent.sort(key=lambda e: 0 if e.get("outcome") == "success" else 1)

            traj_dir = self._traj_dir()
            sections = []
            for entry in recent:
                tid = entry["id"]
                outcome = entry.get("outcome", "")
                label = f"[{outcome.upper()}] " if outcome else ""
                try:
                    text = await self.viking_fs.read_file(f"{traj_dir}/{tid}.md", ctx=self.ctx)
                    if text:
                        sections.append(f"### {label}Trajectory {tid}\n{text}")
                except Exception:
                    pass

            if not sections:
                return ""

            logger.info(f"Loaded {len(sections)}/{len(entries)} trajectories for {case_uri}")
            return "\n\n---\n\n".join(sections)

        except Exception as e:
            logger.warning(f"Failed to fetch trajectories for {case_uri}: {e}")
            return ""

    # ─────────────────────────────── prompts ──────────────────────────────────

    def _case_schema(self):
        return self.registry.get(self.MEMORY_TYPE)

    async def _build_case_candidate(self, uri: str, source: str) -> Optional[Dict[str, str]]:
        try:
            raw = await self.viking_fs.read_file(uri, ctx=self.ctx)
            parsed = parse_memory_file_with_fields(raw or "")
            return {
                "source": source,
                "uri": uri,
                "title": str(parsed.get("title", "") or ""),
                "situation": str(parsed.get("situation", "") or ""),
                "lesson": str(parsed.get("lesson", "") or ""),
                "pitfall": str(parsed.get("pitfall", "") or ""),
            }
        except Exception as e:
            logger.warning(f"Failed to build case candidate for {uri}: {e}")
            return None

    @staticmethod
    def _format_case_candidates(candidates: List[Dict[str, str]]) -> str:
        if not candidates:
            return "(none)"
        return "\n\n".join(
            "\n".join([
                f"- source: {candidate['source']}",
                f"  uri: {candidate['uri']}",
                f"  title: {candidate['title']}",
                f"  situation: {candidate['situation']}",
                f"  lesson: {candidate['lesson']}",
                f"  pitfall: {candidate['pitfall']}",
                "  update_hint: Choose this case for update if the current conversation is adding more rules, checks, confirmations, refund handling, or pitfalls to the same workflow.",
            ])
            for candidate in candidates
        )

    @staticmethod
    def _extract_first_candidate_uri(text: str) -> str:
        import re
        match = re.search(r"viking://[^\s]+/cases/[^\s]+\.md", text or "")
        return match.group(0) if match else ""

    def _case_schema_description(self) -> str:
        """Read experience memory description from the YAML schema."""
        schema = self._case_schema()
        return (schema.description or "").strip() if schema else ""

    def _stage1_system_prompt(self, language: str, ls_result: str, search_result: str) -> str:
        template = self._load_prompt_template("stage1_router_prompt.md")
        return self._render_prompt_template(template, {
            "recent_candidates": ls_result or "(none)",
            "search_candidates": search_result or "(none)",
            "language": language,
        })

    def _field_descriptions(self) -> Dict[str, str]:
        """Read field descriptions from the YAML schema."""
        schema = self._case_schema()
        if not schema:
            return {}
        return {f.name: (f.description or "").strip() for f in schema.fields}

    @staticmethod
    def _json_example(data: Dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False)

    def _stage1_output_examples(self, cases_dir: str) -> str:
        return ""

    def _stage2_output_example(self) -> str:
        return ""

    @staticmethod
    def _normalize_stage1_insight(stage1: Dict[str, Any]) -> Dict[str, str]:
        insight = stage1.get("insight")
        if not isinstance(insight, dict):
            insight = {}
        return {
            "lesson_delta": str(insight.get("lesson_delta", "") or ""),
            "pitfall_delta": str(insight.get("pitfall_delta", "") or ""),
        }

    def _stage2_system_prompt(self, language: str) -> str:
        template = self._load_prompt_template("stage2_synthesis_prompt.md")
        return self._render_prompt_template(template, {
            "language": language,
        })

    # ──────────────────────────────── LLM ─────────────────────────────────────

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call VLM without tools, parse JSON from response content.

        get_completion_async returns str when tools=None, VLMResponse when tools are passed.
        """
        result = await self.vlm.get_completion_async(
            messages=messages,
            tools=None,
            tool_choice=None,
            max_retries=self.vlm.max_retries,
        )
        # Normalize to string
        if hasattr(result, "content"):
            content = result.content or ""
        else:
            content = result or ""
        if not content:
            return None
        try:
            cleaned = extract_json_from_markdown(content)
            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON: {e}\nContent: {content[:500]}")
            return None

    # ───────────────────────── operations builder ─────────────────────────────

    def _build_operations(self, ops_dict: Dict[str, Any]) -> Any:
        """Parse an operations dict into the generated StructuredMemoryOperations model."""
        ops_model = self._schema_gen.create_structured_operations_model()
        ops_payload = json.dumps(ops_dict, ensure_ascii=False)
        ops, err = parse_json_with_stability(
            content=ops_payload,
            model_class=ops_model,
        )
        if err is not None:
            logger.warning(f"Failed to build operations model: {err}")
            return MemoryOperations()
        return ops

    # ──────────────────────────────── run ─────────────────────────────────────

    async def _run_stage1(self, conversation: str, language: str) -> Optional[Dict[str, Any]]:
        ls_result, search_result = await self._pre_fetch(conversation)
        stage1_system = self._stage1_system_prompt(language, ls_result, search_result)
        stage1_user = f"## Conversation\n{conversation}\n\nOutput your routing decision as JSON."
        logger.warning("[V3] Stage 1 input:\n%s", json.dumps({
            "system": stage1_system,
            "user": stage1_user,
        }, ensure_ascii=False, indent=2))
        stage1 = await self._call_llm([
            {"role": "system", "content": stage1_system},
            {"role": "user", "content": stage1_user},
        ])
        if not stage1:
            logger.warning("[V3] Stage 1 returned no output")
            return None
        decision = stage1.get("decision", "no_op")
        self.outcome = stage1.get("outcome", "") if decision != "no_op" else ""
        logger.warning("[V3] Stage 1 output:\n%s", json.dumps(stage1, ensure_ascii=False, indent=2))
        return stage1

    async def _run_stage2(
        self,
        conversation: str,
        language: str,
        stage1: Dict[str, Any],
        target_uri: str,
    ) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        try:
            existing_raw = await self.viking_fs.read_file(target_uri, ctx=self.ctx)
            existing = parse_memory_file_with_fields(existing_raw)
            existing.pop("trajectory_ids", None)
        except Exception as e:
            logger.warning(f"[V3] Failed to read existing case {target_uri}: {e}")
            return MemoryOperations(), []

        existing_title = existing.get("title") or ""
        trajectories = await self._fetch_trajectories(target_uri)
        stage1_insight = self._normalize_stage1_insight(stage1)
        stage2_user = (
            f"## Current conversation\n{conversation}\n\n"
            f"## Stage 1 insight summary\n{json.dumps(stage1_insight, ensure_ascii=False, indent=2)}\n\n"
            f"## Existing case\n{json.dumps(existing, ensure_ascii=False, indent=2)}\n\n"
        )
        if trajectories:
            stage2_user += f"## Historical trajectories\n{trajectories}\n\n"
        stage2_user += (
            "Synthesize and output the updated case as JSON. "
            "Always return title, situation, lesson, and pitfall for the updated case, even if only part of the case changes. "
            "Use the exact target URI from stage 1 as the case to update; do not switch to a different case."
        )

        logger.warning("[V3] Stage 2 input:\n%s", json.dumps({
            "target_uri": target_uri,
            "stage1_insight": stage1_insight,
            "existing_case": existing,
            "historical_trajectories": trajectories,
            "conversation": conversation,
        }, ensure_ascii=False, indent=2))
        stage2 = await self._call_llm([
            {"role": "system", "content": self._stage2_system_prompt(language)},
            {"role": "user", "content": stage2_user},
        ])
        if not stage2:
            logger.warning("[V3] Stage 2 returned no output")
            return MemoryOperations(), []

        logger.warning("[V3] Stage 2 output:\n%s", json.dumps(stage2, ensure_ascii=False, indent=2))

        new_title = stage2.get("title", existing_title) or existing_title
        situation = stage2.get("situation", existing.get("situation", "")) or existing.get("situation", "")
        lesson = stage2.get("lesson", existing.get("lesson", "")) or existing.get("lesson", "")
        pitfall = stage2.get("pitfall", existing.get("pitfall", "")) or existing.get("pitfall", "")
        rendered = "\n\n".join(p for p in [new_title, situation, lesson, pitfall] if p)

        _, old_meta = deserialize_full(existing_raw)
        memory_id = (old_meta or {}).get("memory_id", "")

        ops = self._build_operations({
            "reasoning": stage2.get("reasoning", ""),
            "write_uris": [],
            "edit_uris": [{
                "memory_type": self.MEMORY_TYPE,
                "uri": target_uri,
                "memory_id": memory_id,
                "title": new_title,
                "content": rendered,
                "situation": situation,
                "lesson": lesson,
                "pitfall": pitfall,
            }],
            "edit_overview_uris": [],
            "delete_uris": [],
        })
        return ops, []

    async def run(self, conversation: str) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """Run two-stage extraction. Returns (MemoryOperations, tools_used)."""
        config = get_openviking_config()
        fallback_lang = (config.language_fallback or "en").strip() or "en"
        language = detect_language_from_conversation(conversation, fallback_language=fallback_lang)
        logger.info(f"[V3] output language: {language}")

        stage1 = await self._run_stage1(conversation, language)
        if not stage1:
            return MemoryOperations(), []

        decision = stage1.get("decision", "no_op")
        if decision == "no_op":
            return MemoryOperations(), []

        if decision == "add":
            title = stage1.get("title", "")
            if not title:
                logger.warning("[V3] 'add' decision missing title — treating as no_op")
                return MemoryOperations(), []
            from openviking.session.memory.utils.uri import generate_uri
            user_space = self.ctx.user.user_space_name() if self.ctx and self.ctx.user else "default"
            agent_space = self._agent_space()
            case_schema = self.registry.get(self.MEMORY_TYPE)
            memory_id = self._generate_memory_id()
            uri = generate_uri(case_schema, {"memory_id": memory_id}, user_space, agent_space)
            ops = self._build_operations({
                "reasoning": stage1.get("reasoning", ""),
                "write_uris": [{
                    "memory_type": self.MEMORY_TYPE,
                    "uri": uri,
                    "memory_id": memory_id,
                    "title": title,
                    "situation": stage1.get("situation", ""),
                    "lesson": stage1.get("lesson", ""),
                    "pitfall": stage1.get("pitfall", ""),
                }],
                "edit_uris": [],
                "edit_overview_uris": [],
                "delete_uris": [],
            })
            return ops, []

        target_uri = stage1.get("target_uri", "")
        if not target_uri:
            logger.warning("[V3] 'update' decision missing target_uri — treating as no_op")
            return MemoryOperations(), []

        if not target_uri.endswith(".md"):
            logger.warning(f"[V3] Stage 1 target_uri is not a case file: {target_uri} — trying best candidate fallback")
            if self._last_stage1_candidates:
                target_uri = self._last_stage1_candidates[0].get("uri", "")
            else:
                best_candidate_uri = self._extract_first_candidate_uri(stage1.get("reasoning", ""))
                if best_candidate_uri:
                    target_uri = best_candidate_uri
                else:
                    return MemoryOperations(), []

        try:
            await self.viking_fs.read_file(target_uri, ctx=self.ctx)
        except Exception:
            logger.warning(f"[V3] Stage 1 target_uri not found: {target_uri} — treating as no_op")
            return MemoryOperations(), []

        return await self._run_stage2(conversation, language, stage1, target_uri)
