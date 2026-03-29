# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Session Compressor V3 for OpenViking.

Uses the V3 two-stage case memory extractor.
Maintains the same interface as compressor.py for backward compatibility.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from openviking.core.context import Context
from openviking.message import Message
from openviking.server.identity import RequestContext
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config

from openviking.session.memory import MemoryTypeRegistry, MemoryUpdater
from openviking.session.memory.memory_react_v3 import MemoryReActV3
from openviking.session.memory.utils import deserialize_full, serialize_with_metadata

logger = get_logger(__name__)


@dataclass
class ExtractionStats:
    created: int = 0
    merged: int = 0
    deleted: int = 0
    skipped: int = 0


class SessionCompressorV3:
    """Session memory extractor with V3 two-stage case extraction."""

    MEMORY_TYPE = "cases"

    def __init__(self, vikingdb: VikingDBManager):
        self.vikingdb = vikingdb
        self._registry = MemoryTypeRegistry()
        schemas_dir = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "templates", "memory_v3"
        )
        self._registry.load_from_directory(schemas_dir)
        self._memory_updater: Optional[MemoryUpdater] = None

    def _get_or_create_react(
        self,
        trajectory_id: str,
        ctx: Optional[RequestContext] = None,
    ) -> MemoryReActV3:
        config = get_openviking_config()
        vlm = config.vlm.get_vlm_instance()
        viking_fs = get_viking_fs()
        return MemoryReActV3(
            vlm=vlm,
            viking_fs=viking_fs,
            ctx=ctx,
            registry=self._registry,
            trajectory_id=trajectory_id,
        )

    def _get_or_create_updater(self) -> MemoryUpdater:
        if self._memory_updater is not None:
            return self._memory_updater
        self._memory_updater = MemoryUpdater(registry=self._registry, vikingdb=self.vikingdb)
        return self._memory_updater

    def _trajectories_dir(self, ctx: RequestContext) -> str:
        agent_space = ctx.user.agent_space_name() if ctx and ctx.user else "default"
        return f"viking://agent/{agent_space}/trajectories"

    def _trajectory_uri(self, trajectory_id: str, ctx: RequestContext) -> str:
        return f"{self._trajectories_dir(ctx)}/{trajectory_id}.md"

    async def _write_trajectory(
        self,
        trajectory_id: str,
        conversation: str,
        ctx: RequestContext,
    ) -> None:
        """Persist the raw conversation trajectory for later case synthesis."""
        viking_fs = get_viking_fs()
        traj_uri = self._trajectory_uri(trajectory_id, ctx)
        await viking_fs.write_file(traj_uri, conversation, ctx=ctx)
        logger.debug(f"Wrote trajectory: {traj_uri}")

    async def _inject_trajectory_id(
        self,
        uris: List[str],
        trajectory_id: str,
        outcome: str,
        ctx: RequestContext,
    ) -> None:
        """Append a {id, outcome} entry to trajectory_ids in modified case files."""
        viking_fs = get_viking_fs()
        new_entry = {"id": trajectory_id, "outcome": outcome}

        for uri in uris:
            if not uri.endswith(".md"):
                continue
            if uri.endswith(".overview.md") or uri.endswith(".abstract.md"):
                continue
            if f"/{self.MEMORY_TYPE}/" not in uri:
                continue

            try:
                content = await viking_fs.read_file(uri, ctx=ctx) or ""
                plain_content, metadata = deserialize_full(content)
                if metadata is None:
                    metadata = {}

                existing = MemoryReActV3._parse_traj_entries(metadata.get("trajectory_ids", []))
                if not any(entry.get("id") == trajectory_id for entry in existing):
                    existing.append(new_entry)
                metadata["trajectory_ids"] = existing

                await viking_fs.write_file(
                    uri,
                    serialize_with_metadata(plain_content, metadata),
                    ctx=ctx,
                )
                logger.debug(f"Linked trajectory {trajectory_id} ({outcome}) to case: {uri}")
            except Exception as e:
                logger.warning(f"Failed to inject trajectory_id into {uri}: {e}")

    async def extract_long_term_memories(
        self,
        messages: List[Message],
        user: Optional["UserIdentifier"] = None,
        session_id: Optional[str] = None,
        ctx: Optional[RequestContext] = None,
        strict_extract_errors: bool = False,
        latest_archive_overview: str = "",
    ) -> List[Context]:
        """Extract long-term memories from messages using V3 case extraction."""
        if not messages:
            return []
        if not ctx:
            logger.warning("No RequestContext provided, skipping memory extraction")
            return []

        conversation_sections: List[str] = []
        if latest_archive_overview:
            conversation_sections.append(f"## Previous Archive Overview\n{latest_archive_overview}")
        conversation_sections.append("\n".join([f"[{msg.role}]: {msg.content}" for msg in messages]))
        conversation_str = "\n\n".join(section for section in conversation_sections if section)

        trajectory_id = f"{(session_id or 'session')[:16]}_{int(time.time() * 1000)}"
        logger.info(f"Starting v3 memory extraction from conversation, trajectory_id={trajectory_id}")

        try:
            orchestrator = self._get_or_create_react(trajectory_id=trajectory_id, ctx=ctx)
            updater = self._get_or_create_updater()

            operations, _tools_used = await orchestrator.run(conversation=conversation_str)
            if operations is None:
                logger.info("No memory operations generated")
                return []

            logger.info(
                f"Generated memory operations: write={len(operations.write_uris)}, "
                f"edit={len(operations.edit_uris)}, edit_overview={len(operations.edit_overview_uris)}, "
                f"delete={len(operations.delete_uris)}"
            )

            result = await updater.apply_operations(operations, ctx, registry=orchestrator.registry)
            await self._write_trajectory(trajectory_id, conversation_str, ctx)

            all_modified_uris = result.written_uris + result.edited_uris
            await self._inject_trajectory_id(all_modified_uris, trajectory_id, orchestrator.outcome, ctx)

            logger.info(
                f"Applied memory operations: written={len(result.written_uris)}, "
                f"edited={len(result.edited_uris)}, deleted={len(result.deleted_uris)}, "
                f"errors={len(result.errors)}"
            )

            total_changes = (
                len(result.written_uris)
                + len(result.edited_uris)
                + len(result.deleted_uris)
            )
            return [None] * total_changes

        except Exception as e:
            logger.error(f"Failed to extract memories with v3: {e}", exc_info=True)
            if strict_extract_errors:
                raise
            return []
