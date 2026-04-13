# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Agent Experience Context Provider - Phase 2 of agent-scope memory extraction.

Given a new trajectory summary from Phase 1, search for candidate experiences and
let the LLM decide whether to update an existing one, create a new one, or do nothing.

The LLM may call `get_source_trajectories(experience_uri)` to load historical
trajectory grounding material for the chosen experience before producing its
final output.
"""

import jinja2
from typing import Any, Dict, List

from openviking.server.identity import RequestContext, ToolContext
from openviking.session.memory.session_extract_context_provider import (
    SessionExtractContextProvider,
)
from openviking.session.memory.tools import (
    add_tool_call_pair_to_messages,
    get_tool,
)
from openviking.storage.viking_fs import VikingFS
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


EXPERIENCE_MEMORY_TYPE = "experience"
SEARCH_TOP_K = 5


class AgentExperienceContextProvider(SessionExtractContextProvider):
    """Phase 2 provider: consolidate the new trajectory into experience memories."""

    def __init__(
        self,
        messages: Any,
        trajectory_summary: str,
        trajectory_uri: str,
        latest_archive_overview: str = "",
    ):
        super().__init__(messages=messages, latest_archive_overview=latest_archive_overview)
        self.trajectory_summary = trajectory_summary
        self.trajectory_uri = trajectory_uri

    def instruction(self) -> str:
        output_language = self._output_language
        return f"""You consolidate trajectory memories into experience memories.

You are given one new trajectory and a few candidate existing experiences.
If needed, call `get_source_trajectories(experience_uri)` before editing an existing experience.

Rules:
- Prefer editing an existing experience over creating a new one when the pattern is the same.
- Output exactly one result: edit one experience, write one experience, or do nothing.
- Do not change the identity of an existing experience.
- Do not output system-managed fields such as source_trajectories.
- Output JSON only.

All memory content must be written in {output_language}.
"""

    def get_memory_schemas(self, ctx: RequestContext) -> List[Any]:
        registry = self._get_registry()
        schema = registry.get(EXPERIENCE_MEMORY_TYPE)
        if schema is None or not schema.enabled:
            return []
        return [schema]

    def get_tools(self) -> List[str]:
        return ["get_source_trajectories"]

    def _render_experience_dir(self, ctx: RequestContext) -> str:
        registry = self._get_registry()
        schema = registry.get(EXPERIENCE_MEMORY_TYPE)
        if schema is None or not schema.directory:
            return ""
        user_space = ctx.user.user_space_name() if ctx and ctx.user else "default"
        agent_space = ctx.user.agent_space_name() if ctx and ctx.user else "default"
        env = jinja2.Environment(autoescape=False)
        return env.from_string(schema.directory).render(
            user_space=user_space, agent_space=agent_space
        )

    async def prefetch(
        self,
        ctx: RequestContext,
        viking_fs: VikingFS,
        transaction_handle,
        vlm,
    ) -> List[Dict]:
        if not isinstance(self.messages, list):
            logger.warning(f"Expected List[Message], got {type(self.messages)}")
            return []

        pre_fetch_messages: List[Dict] = []

        pre_fetch_messages.append(
            {
                "role": "user",
                "content": (
                    "## New Trajectory\n"
                    f"Trajectory URI: `{self.trajectory_uri}`\n\n"
                    f"{self.trajectory_summary}\n\n"
                    "The tool call results below show candidate existing experiences. "
                    "Decide whether to edit one, write one, or do nothing."
                ),
            }
        )

        experience_dir = self._render_experience_dir(ctx)
        if not experience_dir:
            return pre_fetch_messages

        search_tool = get_tool("search")
        read_tool = get_tool("read")
        call_id_seq = 0

        candidate_uris: List[str] = []
        if search_tool and viking_fs:
            tool_ctx_search = ToolContext(
                request_ctx=ctx,
                transaction_handle=transaction_handle,
                default_search_uris=[experience_dir],
            )
            try:
                search_result = await search_tool.execute(
                    viking_fs=viking_fs,
                    ctx=tool_ctx_search,
                    query=self.trajectory_summary[:500] or "experience",
                    limit=SEARCH_TOP_K,
                )
                if isinstance(search_result, list):
                    candidate_uris = [m.get("uri", "") for m in search_result if m.get("uri")]
                elif isinstance(search_result, dict) and "memories" in search_result:
                    candidate_uris = [
                        m.get("uri", "")
                        for m in search_result.get("memories", [])
                        if m.get("uri")
                    ]
                result_value = candidate_uris if candidate_uris else search_result
                add_tool_call_pair_to_messages(
                    messages=pre_fetch_messages,
                    call_id=call_id_seq,
                    tool_name="search",
                    params={"query": "[new trajectory]", "search_uri": experience_dir},
                    result=result_value,
                )
                call_id_seq += 1
            except Exception as e:
                logger.warning(f"Failed to search experiences in {experience_dir}: {e}")

        if not read_tool or not candidate_uris:
            return pre_fetch_messages

        for exp_uri in candidate_uris:
            try:
                exp_raw = await viking_fs.read_file(exp_uri, ctx=ctx)
            except Exception as e:
                logger.warning(f"Failed to read experience {exp_uri}: {e}")
                continue

            add_tool_call_pair_to_messages(
                messages=pre_fetch_messages,
                call_id=call_id_seq,
                tool_name="read",
                params={"uri": exp_uri},
                result=exp_raw,
            )
            call_id_seq += 1

        return pre_fetch_messages
