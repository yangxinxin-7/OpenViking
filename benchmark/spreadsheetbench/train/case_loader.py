#!/usr/bin/env python3
"""SpreadsheetBench task CaseLoader for OpenViking batch policy training."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from benchmark.spreadsheetbench.train.data_paths import (
    dataset_json_path,
    split_tasks_path,
    task_spreadsheet_dir,
)
from openviking.session.train import Case, Rubric, RubricCriterion


@lru_cache(maxsize=4)
def _load_dataset(domain: str, data_root: str | None = None) -> dict[str, dict[str, Any]]:
    path = dataset_json_path(domain, data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"SpreadsheetBench dataset.json not found: {path}. "
            "Extract the release tarball under benchmark/spreadsheetbench/upstream/data "
            "or point SSB_DATA_ROOT at the data directory."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"dataset.json must be a list: {path}")
    return {str(item["id"]): item for item in records if isinstance(item, dict) and "id" in item}


def _load_split_tasks(domain: str, data_root: str | None = None) -> dict[str, Any]:
    path = split_tasks_path(domain, data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"split_tasks.json not found: {path}. Generate it with "
            "benchmark/spreadsheetbench/scripts/make_split.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class SpreadsheetBenchCaseLoader:
    """Load SpreadsheetBench split tasks as train-domain Cases."""

    domain: str
    split: str
    batch_size: int | None = None
    data_root: str | None = None
    task_indices: list[int] | None = None

    async def batches(self, context: Any = None) -> AsyncIterator[list[Case]]:
        del context
        task_ids = self.load_task_ids()
        size = self.batch_size or 1
        if size <= 0:
            raise ValueError("batch_size must be > 0")
        for start in range(0, len(task_ids), size):
            yield [
                self._case_from_task(task_no, task_id)
                for task_no, task_id in task_ids[start : start + size]
            ]

    def load_cases(self) -> list[Case]:
        return [self._case_from_task(task_no, task_id) for task_no, task_id in self.load_task_ids()]

    def load_task_ids(self) -> list[tuple[int, str]]:
        data = _load_split_tasks(self.domain, self.data_root)
        values = data.get(self.split)
        if not isinstance(values, list):
            return []
        task_ids = [(task_no, str(item)) for task_no, item in enumerate(values)]
        if self.task_indices is None:
            return task_ids
        selected: list[tuple[int, str]] = []
        for index in self.task_indices:
            if index < 0:
                raise ValueError("task_indices must be >= 0")
            try:
                selected.append(task_ids[index])
            except IndexError as exc:
                raise ValueError(
                    f"task index out of range for split {self.split!r}: {index} "
                    f"(size={len(task_ids)})"
                ) from exc
        return selected

    def split_exists(self) -> bool:
        try:
            data = _load_split_tasks(self.domain, self.data_root)
        except FileNotFoundError:
            return False
        values = data.get(self.split)
        return isinstance(values, list) and bool(values)

    def _case_from_task(self, task_no: int, task_id: str) -> Case:
        record = _load_dataset(self.domain, self.data_root).get(task_id)
        if record is None:
            raise ValueError(
                f"SpreadsheetBench task not found domain={self.domain} task_id={task_id}"
            )
        spreadsheet_dir = task_spreadsheet_dir(self.domain, task_id, self.data_root)
        data_split = f"{self.domain}_{self.split}"
        return Case(
            name=f"ssb_{data_split}_{task_no}",
            task_signature=f"spreadsheetbench:{self.domain}:{self.split}:{task_id}",
            input={
                "domain": self.domain,
                "split": self.split,
                "data_split": data_split,
                "task_no": task_no,
                "task_id": task_id,
                "data_root": self.data_root,
                "instruction": str(record.get("instruction") or ""),
                "instruction_type": str(record.get("instruction_type") or ""),
                "answer_position": str(record.get("answer_position") or ""),
                "answer_sheet": record.get("answer_sheet"),
                "spreadsheet_dir": str(spreadsheet_dir),
                "user_query": str(record.get("instruction") or ""),
                "policy": "",
            },
            rubric=Rubric(
                name=f"ssb_{data_split}_{task_no}_rubric",
                description="SpreadsheetBench hard restriction: every shipped test case must pass.",
                criteria=[
                    RubricCriterion(
                        name="ssb_hard_restriction",
                        description="The final solution passes OJ comparison on all test cases.",
                        required=True,
                        weight=1.0,
                    )
                ],
            ),
            metadata={
                "source": "spreadsheetbench",
                "domain": self.domain,
                "split": self.split,
                "instruction_type": str(record.get("instruction_type") or ""),
            },
        )
