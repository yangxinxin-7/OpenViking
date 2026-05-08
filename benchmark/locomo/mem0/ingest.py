"""
Ingest LoCoMo conversations into mem0 (mem0ai 2.0+).

Each sample gets an isolated mem0 namespace keyed by sample_id (e.g. "conv-26").

Usage:
    # Ingest all samples
    python ingest.py

    # Ingest a specific sample
    python ingest.py --sample conv-26

    # Ingest specific sessions
    python ingest.py --sample conv-26 --sessions 1-4

    # Force re-ingest even if already done
    python ingest.py --force-ingest

    # Set mem0 API key via env or flag
    MEM0_API_KEY=xxx python ingest.py
    python ingest.py --api-key xxx
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path.home() / ".openviking_benchmark_env")

try:
    from mem0 import MemoryClient
except ImportError:
    print("Error: mem0 package not installed. Run: pip install mem0ai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA_PATH = str(SCRIPT_DIR / ".." / "data" / "locomo10.json")
DEFAULT_RECORD_PATH = str(SCRIPT_DIR / "result" / ".ingest_record.json")
DEFAULT_LOG_PATH = str(SCRIPT_DIR / "result" / "ingest_errors.log")

CUSTOM_INSTRUCTIONS = """Extract memories from group chat conversations between two people. Each message is prefixed with the speaker's name in brackets (e.g. [Alice]: text).

Guidelines:
1. Always include the speaker's name in the memory, never use generic terms like "user"
2. Extract memories for both speakers equally
3. Each memory should be self-contained with full context: who, what, when
4. Include specific details: dates, places, names of activities, emotional states
5. Cover all meaningful topics: life events, plans, hobbies, relationships, opinions"""


# ---------------------------------------------------------------------------
# LoCoMo data loading
# ---------------------------------------------------------------------------

def load_locomo_data(path: str, sample_id: Optional[str] = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if sample_id is not None:
        try:
            idx = int(sample_id)
            if idx < 0 or idx >= len(data):
                raise ValueError(f"Sample index {idx} out of range (0-{len(data) - 1})")
            return [data[idx]]
        except ValueError:
            pass
        matched = [s for s in data if s.get("sample_id") == sample_id]
        if not matched:
            raise ValueError(f"sample_id '{sample_id}' not found")
        return matched

    return data


def parse_session_range(s: str) -> tuple[int, int]:
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


def build_session_messages(
    item: dict,
    session_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    conv = item["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        sess_num = int(sk.split("_")[1])
        if session_range:
            lo, hi = session_range
            if sess_num < lo or sess_num > hi:
                continue

        raw_messages = conv[sk]
        if not isinstance(raw_messages, list) or not raw_messages:
            continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        messages = []
        if date_time:
            messages.append({"role": "user", "content": f"[System]: This conversation took place on {date_time}."})
        for msg in raw_messages:
            speaker = msg.get("speaker", "")
            text = msg.get("text", "")
            messages.append({"role": "user", "content": f"[{speaker}]: {text}"})

        sessions.append(
            {
                "messages": messages,
                "meta": {
                    "sample_id": item["sample_id"],
                    "session_key": sk,
                    "date_time": date_time,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            }
        )

    return sessions


# ---------------------------------------------------------------------------
# Ingest record (progress tracking)
# ---------------------------------------------------------------------------

def load_ingest_record(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_ingest_record(record: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def is_already_ingested(sample_id: str, session_key: str, record: dict) -> bool:
    key = f"mem0:{sample_id}:{session_key}"
    return key in record and record[key].get("success", False)


def mark_ingested(sample_id: str, session_key: str, record: dict, meta: Optional[dict] = None) -> None:
    key = f"mem0:{sample_id}:{session_key}"
    record[key] = {
        "success": True,
        "timestamp": int(time.time()),
        "meta": meta or {},
    }


def write_error_log(path: str, sample_id: str, session_key: str, error: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] ERROR [{sample_id}/{session_key}]: {error}\n")


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------

def ingest_session(client: MemoryClient, messages: list[dict], user_id: str, meta: dict) -> None:
    """Add one session's messages to mem0 using the v2.0+ API."""
    client.add(
        messages,
        user_id=user_id,
        metadata={
            "session_key": meta.get("session_key", ""),
            "date_time": meta.get("date_time", ""),
            "speaker_a": meta.get("speaker_a", ""),
            "speaker_b": meta.get("speaker_b", ""),
        },
    )


def run_ingest(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        print("Error: mem0 API key required (--api-key or MEM0_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    client = MemoryClient(api_key=api_key)

    try:
        client.update_project(custom_instructions=CUSTOM_INSTRUCTIONS)
        print("[INFO] Updated mem0 project custom instructions", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not set custom instructions: {e}", file=sys.stderr)

    session_range = parse_session_range(args.sessions) if args.sessions else None

    if args.clear_ingest_record:
        ingest_record: dict = {}
        save_ingest_record(ingest_record, args.record)
        print("[INFO] Cleared existing ingest records", file=sys.stderr)
    else:
        ingest_record = load_ingest_record(args.record)

    samples = load_locomo_data(args.input, args.sample)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[INFO] Loaded {len(samples)} sample(s)", file=sys.stderr)

    record_lock = threading.Lock()
    counters = {"total": 0, "success": 0, "skip": 0, "error": 0}

    all_tasks: list[tuple[str, dict]] = []
    for item in samples:
        sample_id: str = item["sample_id"]
        sessions = build_session_messages(item, session_range)
        for sess in sessions:
            counters["total"] += 1
            if not args.force_ingest and is_already_ingested(sample_id, sess["meta"]["session_key"], ingest_record):
                counters["skip"] += 1
                label = f"{sess['meta']['session_key']} ({sess['meta']['date_time']})"
                print(f"  [{sample_id}/{label}] SKIP (already ingested)", file=sys.stderr)
            else:
                all_tasks.append((sample_id, sess))

    print(f"[INFO] {len(all_tasks)} sessions to ingest ({counters['skip']} skipped)", file=sys.stderr)

    def ingest_one(sample_id: str, sess: dict) -> None:
        meta = sess["meta"]
        session_key = meta["session_key"]
        label = f"{sample_id}/{session_key} ({meta['date_time']})"
        print(f"  [{label}] ingesting {len(sess['messages'])} messages ...", file=sys.stderr)
        t0 = time.time()
        try:
            ingest_session(client, sess["messages"], user_id=sample_id, meta=meta)
            elapsed = time.time() - t0
            with record_lock:
                mark_ingested(sample_id, session_key, ingest_record, meta)
                save_ingest_record(ingest_record, args.record)
                counters["success"] += 1
            print(f"  [{label}] OK  {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{label}] ERROR: {e}  {elapsed:.1f}s", file=sys.stderr)
            write_error_log(args.error_log, sample_id, session_key, str(e))
            with record_lock:
                counters["error"] += 1

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(ingest_one, sid, sess): (sid, sess) for sid, sess in all_tasks}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                sid, sess = futures[fut]
                print(f"  [FATAL] {sid}/{sess['meta']['session_key']}: {e}", file=sys.stderr)

    print(f"\n=== Ingest summary ===", file=sys.stderr)
    print(f"  Total sessions:  {counters['total']}", file=sys.stderr)
    print(f"  Succeeded:       {counters['success']}", file=sys.stderr)
    print(f"  Skipped:         {counters['skip']}", file=sys.stderr)
    print(f"  Failed:          {counters['error']}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LoCoMo conversations into mem0")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="Path to locomo10.json")
    parser.add_argument("--api-key", default=None, help="mem0 API key (or set MEM0_API_KEY env var)")
    parser.add_argument(
        "--sample",
        default=None,
        help="Sample index (0-based int) or sample_id string (e.g. conv-26). Default: all.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples to ingest. Default: all.")
    parser.add_argument("--sessions", default=None, help="Session range, e.g. '1-4' or '3'. Default: all.")
    parser.add_argument("--record", default=DEFAULT_RECORD_PATH, help="Path to ingest progress record")
    parser.add_argument("--error-log", default=DEFAULT_LOG_PATH, help="Path to error log")
    parser.add_argument("--force-ingest", action="store_true", default=False, help="Re-ingest even if already recorded")
    parser.add_argument("--clear-ingest-record", action="store_true", default=False, help="Clear existing ingest records before running")
    parser.add_argument("--threads", type=int, default=1, help="Concurrent threads (default: 1)")

    args = parser.parse_args()
    run_ingest(args)


if __name__ == "__main__":
    main()
