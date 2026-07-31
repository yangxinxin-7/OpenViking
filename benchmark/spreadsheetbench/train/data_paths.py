#!/usr/bin/env python3
"""Path helpers shared by the SpreadsheetBench loaders, executor, and scripts."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOMAIN = "all_data_912_v0.1"


def data_root(explicit: str | None = None) -> Path:
    """Root directory containing one subdirectory per domain (dataset release)."""
    root = explicit or os.getenv("SSB_DATA_ROOT")
    if root:
        return Path(root).expanduser()
    return REPO_ROOT / "benchmark" / "spreadsheetbench" / "upstream" / "data"


def domain_dir(domain: str, root: str | None = None) -> Path:
    return data_root(root) / domain


def dataset_json_path(domain: str, root: str | None = None) -> Path:
    return domain_dir(domain, root) / "dataset.json"


def split_tasks_path(domain: str, root: str | None = None) -> Path:
    return domain_dir(domain, root) / "split_tasks.json"


def task_spreadsheet_dir(domain: str, task_id: str, root: str | None = None) -> Path:
    return domain_dir(domain, root) / "spreadsheet" / task_id


def input_xlsx_name(task_id: str, test_case: int) -> str:
    return f"{test_case}_{task_id}_input.xlsx"


def answer_xlsx_name(task_id: str, test_case: int) -> str:
    return f"{test_case}_{task_id}_answer.xlsx"


def output_xlsx_name(task_id: str, test_case: int) -> str:
    return f"{test_case}_{task_id}_output.xlsx"
