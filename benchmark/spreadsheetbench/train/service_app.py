#!/usr/bin/env python3
"""HTTP service exposing SpreadsheetBench cases and rollout execution."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn

DEFAULT_MAX_ROLLOUT_CONCURRENCY = 100
DEFAULT_ROLLOUT_THREAD_WORKERS = 100

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.spreadsheetbench.train.case_loader import SpreadsheetBenchCaseLoader
from benchmark.spreadsheetbench.train.rollout_executor_vikingbot import (
    VikingBotSpreadsheetBenchRolloutExecutor,
)
from openviking.session.train.components.dataset_service import create_dataset_service_app


def create_app(
    *,
    data_root: str | None = None,
    config_path: str | None = None,
    max_rollout_concurrency: int | None = None,
    rollout_thread_workers: int | None = None,
):
    def make_case_loader(
        dataset: str,
        domain: str,
        split: str,
        filters: dict[str, Any],
    ) -> SpreadsheetBenchCaseLoader:
        if dataset != "spreadsheetbench":
            raise ValueError(f"Unsupported dataset: {dataset}")
        return SpreadsheetBenchCaseLoader(
            domain=domain,
            split=split,
            data_root=data_root,
            task_indices=_task_indices_from_filters(filters),
        )

    def make_rollout_executor(options: dict[str, Any]):
        return VikingBotSpreadsheetBenchRolloutExecutor(
            config_path=options.get("config_path") or config_path,
            concurrency=1,
            max_iterations=int(options.get("max_iterations") or 30),
        )

    return create_dataset_service_app(
        service_name="spreadsheetbench",
        make_case_loader=make_case_loader,
        make_rollout_executor=make_rollout_executor,
        max_rollout_concurrency=max_rollout_concurrency,
        rollout_thread_workers=rollout_thread_workers,
    )


def _task_indices_from_filters(filters: dict[str, Any]) -> list[int] | None:
    raw = filters.get("task_indices")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("task_indices filter must be a list")
    indices: list[int] = []
    for value in raw:
        index = int(value)
        if index < 0:
            raise ValueError("task index must be >= 0")
        indices.append(index)
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start SpreadsheetBench rollout HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1954)
    parser.add_argument("--data-root", default=os.getenv("SSB_DATA_ROOT"))
    parser.add_argument("--config", default=os.getenv("OPENVIKING_CONFIG_FILE"))
    parser.add_argument(
        "--max-rollout-concurrency",
        type=int,
        default=int(
            os.getenv("SSB_MAX_ROLLOUT_CONCURRENCY", str(DEFAULT_MAX_ROLLOUT_CONCURRENCY))
        ),
    )
    parser.add_argument(
        "--rollout-thread-workers",
        type=int,
        default=int(
            os.getenv("SSB_ROLLOUT_THREAD_WORKERS", str(DEFAULT_ROLLOUT_THREAD_WORKERS))
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        data_root=args.data_root,
        config_path=args.config,
        max_rollout_concurrency=args.max_rollout_concurrency,
        rollout_thread_workers=(
            None if args.rollout_thread_workers == 0 else args.rollout_thread_workers
        ),
    )
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
