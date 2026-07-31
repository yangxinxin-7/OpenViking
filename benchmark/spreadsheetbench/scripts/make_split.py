#!/usr/bin/env python3
"""Generate split_tasks.json for SpreadsheetBench.

Stratified sampling by instruction_type (Cell-Level vs Sheet-Level) with a fixed
seed, mirroring the tau2 airline scale (train 30 / test 20) by default. Only
tasks whose 3 input/answer xlsx pairs all exist are eligible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.spreadsheetbench.train.data_paths import (  # noqa: E402
    DEFAULT_DOMAIN,
    answer_xlsx_name,
    dataset_json_path,
    input_xlsx_name,
    split_tasks_path,
    task_spreadsheet_dir,
)


def task_files_complete(domain: str, task_id: str, data_root: str | None) -> bool:
    directory = task_spreadsheet_dir(domain, task_id, data_root)
    for tc in (1, 2, 3):
        if not (directory / input_xlsx_name(task_id, tc)).exists():
            return False
        if not (directory / answer_xlsx_name(task_id, tc)).exists():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing split_tasks.json"
    )
    args = parser.parse_args()

    out_path = split_tasks_path(args.domain, args.data_root)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists; pass --force to overwrite")

    records = json.loads(dataset_json_path(args.domain, args.data_root).read_text("utf-8"))
    by_type: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for record in records:
        task_id = str(record["id"])
        if not task_files_complete(args.domain, task_id, args.data_root):
            skipped += 1
            continue
        by_type[str(record.get("instruction_type") or "unknown")].append(task_id)

    total = sum(len(v) for v in by_type.values())
    rng = random.Random(args.seed)
    train: list[str] = []
    test: list[str] = []
    # Largest-type-first keeps rounding drift in the smaller strata.
    for type_name in sorted(by_type, key=lambda k: -len(by_type[k])):
        ids = sorted(by_type[type_name])
        rng.shuffle(ids)
        n_train = round(args.train_size * len(ids) / total)
        n_test = round(args.test_size * len(ids) / total)
        train.extend(ids[:n_train])
        test.extend(ids[n_train : n_train + n_test])

    # Rounding may leave the totals off by one; trim or backfill from the largest stratum.
    train = train[: args.train_size]
    test = test[: args.test_size]

    payload = {
        "train": train,
        "test": test,
        "meta": {
            "seed": args.seed,
            "eligible_tasks": total,
            "skipped_incomplete": skipped,
            "strata": {k: len(v) for k, v in by_type.items()},
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"wrote {out_path}")
    print(f"train={len(train)} test={len(test)} eligible={total} skipped={skipped}")
    for split_name, ids in (("train", train), ("test", test)):
        type_counts: dict[str, int] = defaultdict(int)
        for task_id in ids:
            for type_name, members in by_type.items():
                if task_id in members:
                    type_counts[type_name] += 1
        print(f"  {split_name}: {dict(type_counts)}")


if __name__ == "__main__":
    main()
