# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.session.memory.memory_react_v3 import MemoryReActV3
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.utils.content import serialize_with_metadata


@pytest.fixture
def registry():
    registry = MemoryTypeRegistry()
    root = Path(__file__).resolve().parents[3]
    registry.load_from_directory(root / "openviking" / "prompts" / "templates" / "memory_v3")
    return registry


@pytest.fixture
def ctx():
    user = SimpleNamespace(
        agent_space_name=lambda: "agent-space",
        user_space_name=lambda: "user-space",
    )
    return SimpleNamespace(user=user, account_id="acct")


@pytest.fixture
def vlm():
    mock = MagicMock()
    mock.max_retries = 2
    mock.get_completion_async = AsyncMock()
    return mock


@pytest.fixture
def viking_fs():
    mock = MagicMock()
    mock.read_file = AsyncMock()
    mock.write_file = AsyncMock()
    mock.rm = AsyncMock()
    mock.ls = AsyncMock(return_value=[])
    mock.search = AsyncMock()
    return mock


def make_react(vlm, viking_fs, ctx, registry):
    react = MemoryReActV3(vlm=vlm, viking_fs=viking_fs, ctx=ctx, registry=registry)
    react._schema_gen = SchemaModelGenerator(registry)
    react._schema_gen.generate_all_models()
    return react


@pytest.mark.asyncio
async def test_add_uses_stable_memory_id_uri(vlm, viking_fs, ctx, registry):
    react = make_react(vlm, viking_fs, ctx, registry)
    react._run_stage1 = AsyncMock(return_value={
        "reasoning": "new case",
        "decision": "add",
        "outcome": "success",
        "title": "duplicate flight booking resolution",
        "situation": "Situation",
        "lesson": "Lesson",
        "pitfall": "Pitfall",
    })

    ops, _ = await react.run("[user]: duplicate booking")

    assert len(ops.write_uris) == 1
    write = ops.write_uris[0]
    assert write.uri.startswith("viking://agent/agent-space/memories/cases/")
    assert write.uri.endswith(".md")
    assert write.memory_id
    assert write.title == "duplicate flight booking resolution"
    assert write.pitfall == "Pitfall"


@pytest.mark.asyncio
async def test_stage2_update_same_title_keeps_same_uri(vlm, viking_fs, ctx, registry):
    react = make_react(vlm, viking_fs, ctx, registry)
    target_uri = "viking://agent/agent-space/memories/cases/abc123def456.md"
    existing_content = serialize_with_metadata(
        "Old body",
        {
            "memory_id": "abc123def456",
            "title": "duplicate flight booking resolution",
            "situation": "Old situation",
            "lesson": "Old lesson",
            "pitfall": "Old pitfall",
            "trajectory_ids": [{"id": "traj-1", "outcome": "success"}],
        },
    )
    viking_fs.read_file.return_value = existing_content
    react._fetch_trajectories = AsyncMock(return_value="")
    react._call_llm = AsyncMock(return_value={
        "reasoning": "update existing case",
        "title": "duplicate flight booking resolution",
        "situation": "New situation",
        "lesson": "1. New lesson",
        "pitfall": "1. New pitfall",
    })

    ops, _ = await react._run_stage2(
        conversation="[user]: update",
        language="en",
        stage1={"insight": {"lesson_delta": "x", "pitfall_delta": "y"}},
        target_uri=target_uri,
    )

    assert len(ops.edit_uris) == 1
    edit = ops.edit_uris[0]
    assert edit.uri == target_uri
    assert edit.memory_id == "abc123def456"
    assert edit.title == "duplicate flight booking resolution"
    assert edit.content == "duplicate flight booking resolution\n\nNew situation\n\n1. New lesson\n\n1. New pitfall"
    assert edit.pitfall == "1. New pitfall"
    assert ops.delete_uris == []


@pytest.mark.asyncio
async def test_stage2_title_change_still_edits_same_uri(vlm, viking_fs, ctx, registry):
    react = make_react(vlm, viking_fs, ctx, registry)
    target_uri = "viking://agent/agent-space/memories/cases/abc123def456.md"
    existing_content = serialize_with_metadata(
        "Old body",
        {
            "memory_id": "abc123def456",
            "title": "duplicate flight booking resolution",
            "situation": "Old situation",
            "lesson": "Old lesson",
            "pitfall": "Old pitfall",
            "trajectory_ids": [{"id": "traj-1", "outcome": "success"}],
        },
    )
    viking_fs.read_file.return_value = existing_content
    react._fetch_trajectories = AsyncMock(return_value="")
    react._call_llm = AsyncMock(return_value={
        "reasoning": "retitle case",
        "title": "duplicate flight booking cancellation workflow",
        "situation": "New situation",
        "lesson": "1. New lesson",
        "pitfall": "1. New pitfall",
    })

    ops, _ = await react._run_stage2(
        conversation="[user]: update",
        language="en",
        stage1={"insight": {"lesson_delta": "x", "pitfall_delta": "y"}},
        target_uri=target_uri,
    )

    assert len(ops.edit_uris) == 1
    edit = ops.edit_uris[0]
    assert edit.uri == target_uri
    assert edit.memory_id == "abc123def456"
    assert edit.title == "duplicate flight booking cancellation workflow"
    assert edit.pitfall == "1. New pitfall"
    assert ops.delete_uris == []
    viking_fs.write_file.assert_not_awaited()
