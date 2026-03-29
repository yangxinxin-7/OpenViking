# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "demo_v3_data"
os.environ["OPENVIKING_DATA_DIR"] = str(DATA_DIR)
os.environ["OPENVIKING_CONFIG_FILE"] = str(ROOT / "ov.conf")

from openviking.client import LocalClient
from openviking.session.memory.utils.content import deserialize_full
from openviking_cli.exceptions import NotFoundError
from openviking_cli.utils import run_async
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config
CONVERSATION_1 = [
    (
        "user",
        "我在同一个账号下误订了两张同一天从北京飞上海的机票，帮我找出重复预订并取消多余的那一张。",
    ),
    (
        "assistant",
        "我先帮你核对账号下的全部预订记录，确认哪一张是重复预订。",
    ),
    (
        "assistant",
        "我会先调用 get_user_details 获取你的全部 reservation id，再逐个查看详情，最后和你确认要取消哪一张。",
    ),
]
CONVERSATION_2 = [
    (
        "user",
        "补充刚才那个同账号下重复机票预订取消的处理流程：如果订单里有同行乘客，也要先核对乘客信息；在取消前要明确告诉我最终保留哪一笔订单；取消成功后还要说明退款会原路退回以及大概多久到账。",
    ),
    (
        "assistant",
        "明白，我把这些补充到刚才那条重复机票预订处理经验里。",
    ),
    (
        "assistant",
        "也就是说，除了先找出重复机票订单并确认取消对象外，还要核对同行乘客、明确保留订单，并在取消后说明退款原路退回和到账时间。",
    ),
]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def wait_for_task(client: LocalClient, task_id: str, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        task = run_async(client.get_task(task_id)) or {}
        status = task.get("status") if isinstance(task, dict) else getattr(task, "status", None)
        if status in {"completed", "failed", "cancelled"}:
            if status != "completed":
                raise RuntimeError(f"记忆提取任务失败: {task}")
            return
        time.sleep(1)
    raise TimeoutError(f"等待任务超时: {task_id}")


def run_conversation(client: LocalClient, turns):
    session = run_async(client.create_session())
    session_id = session["session_id"]

    for role, content in turns:
        run_async(client.add_message(session_id=session_id, role=role, content=content))

    result = run_async(client.commit_session(session_id=session_id))
    task_id = result.get("task_id") if isinstance(result, dict) else getattr(result, "task_id", None)
    if task_id:
        wait_for_task(client, task_id)


def _read_text(client: LocalClient, uri: str) -> str:
    return run_async(client.read(uri)) or ""


def _list_entries(client: LocalClient, uri: str):
    try:
        return run_async(client.ls(uri, simple=False)) or []
    except NotFoundError:
        return []


def wait_for_cases(client: LocalClient, agent_space: str, min_cases: int, min_trajectory_ids: int, timeout_s: int = 120) -> None:
    uri = f"viking://agent/{agent_space}/memories/cases"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entries = _list_entries(client, uri)
        case_entries = []
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", "")
            if name.endswith(".md") and not name.startswith("."):
                case_entries.append(name)
        if len(case_entries) >= min_cases:
            if min_trajectory_ids <= 0:
                return
            if min_trajectory_ids == 1:
                return
            for name in case_entries:
                content = _read_text(client, f"{uri}/{name}").strip()
                _plain_content, metadata = deserialize_full(content)
                trajectory_ids = (metadata or {}).get("trajectory_ids", [])
                if len(trajectory_ids) >= min_trajectory_ids:
                    return
        time.sleep(0.5)
    raise TimeoutError(f"等待 case 可见/更新超时: {uri}")


def print_cases(client: LocalClient, agent_space: str, label: str) -> None:
    uri = f"viking://agent/{agent_space}/memories/cases"
    entries = _list_entries(client, uri)
    section(label)
    if not entries:
        print("(无 case memory)")
        return

    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", "")
        if not name.endswith(".md") or name.startswith("."):
            continue
        file_uri = f"{uri}/{name}"
        content = _read_text(client, file_uri).strip()
        _plain_content, metadata = deserialize_full(content)
        title = (metadata or {}).get("title") or name
        print(f"# {title}")
        print(f"(file: {name})\n")
        print(content)
        print()


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    os.environ["OPENVIKING_DATA_DIR"] = str(DATA_DIR)
    os.environ["OPENVIKING_CONFIG_FILE"] = str(ROOT / "ov.conf")
    OpenVikingConfigSingleton._instance = None

    print("memory.version:", get_openviking_config().memory.version)

    client = LocalClient(path=str(DATA_DIR))
    run_async(client.initialize())
    try:
        agent_space = client.service.user.agent_space_name()

        run_conversation(client, CONVERSATION_1)
        print_cases(client, agent_space, "第一轮结束后的 memory")

        run_conversation(client, CONVERSATION_2)
        print_cases(client, agent_space, "第二轮结束后的 memory")
    finally:
        run_async(client.close())


if __name__ == "__main__":
    main()
