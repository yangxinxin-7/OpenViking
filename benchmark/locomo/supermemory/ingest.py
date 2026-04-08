"""
Ingest LoCoMo conversations into Supermemory.

Each sample gets an isolated Supermemory namespace keyed by containerTag = sample_id
(e.g. "conv-26"). Sessions are formatted as date-prefixed JSON content strings,
matching the memorybench supermemory provider convention.

Usage:
    # Ingest all samples
    python ingest.py

    # Ingest a specific sample
    python ingest.py --sample conv-26

    # Ingest specific sessions
    python ingest.py --sample conv-26 --sessions 1-4

    # Force re-ingest even if already done
    python ingest.py --force-ingest

    # Set Supermemory API key via env or flag
    SUPERMEMORY_API_KEY=xxx python ingest.py
    python ingest.py --api-key xxx
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path.home() / ".openviking_benchmark_env")

try:
    from supermemory import Supermemory
except ImportError:
    print("Error: supermemory package not installed. Run: pip install supermemory", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA_PATH = str(SCRIPT_DIR / ".." / "data" / "locomo10.json")
DEFAULT_RECORD_PATH = str(SCRIPT_DIR / "result" / ".ingest_record.json")
DEFAULT_LOG_PATH = str(SCRIPT_DIR / "result" / "ingest_errors.log")


# ---------------------------------------------------------------------------
# Tag sanitization (must match openclaw-supermemory's sanitizeTag logic)
# ---------------------------------------------------------------------------

def sanitize_tag(raw: str) -> str:
    """Sanitize a tag string to match openclaw-supermemory convention.
    Replaces non-alphanumeric/underscore chars with '_', collapses runs, strips edges.
    e.g. 'conv-26' -> 'conv_26'
    """
    tag = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    tag = re.sub(r"_+", "_", tag)
    tag = tag.strip("_")
    return tag


# ---------------------------------------------------------------------------
# LoCoMo data loading
# ---------------------------------------------------------------------------

def load_locomo_data(path: str, sample_id: Optional[str] = None) -> list[dict]:
    """Load LoCoMo JSON and optionally filter to one sample by sample_id or numeric index."""
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
    """Parse '1-4' or '3' into (lo, hi) inclusive tuple."""
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


def format_date_time(date_time: str) -> str:
    """Format a LoCoMo date_time string into a human-readable date."""
    # date_time is typically like "Tuesday, November 14, 2023"
    return date_time


def build_session_content(
    item: dict,
    session_key: str,
    date_time: str,
) -> str:
    """
    Build the content string for a session, matching memorybench supermemory format:
        "Here is the date the following session took place: {date}\n\n
         Here is the session as a stringified JSON:\n{json_string}"
    """
    conv = item["conversation"]
    raw_messages = conv[session_key]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]

    # Build messages list in the same format as memorybench UnifiedSession
    messages = []
    for msg in raw_messages:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"[{speaker}]: {text}"})

    session_str = json.dumps(messages, ensure_ascii=False).replace("<", "&lt;").replace(">", "&gt;")

    if date_time:
        return (
            f"Here is the date the following session took place: {date_time}\n\n"
            f"Here is the session as a stringified JSON:\n{session_str}"
        )
    else:
        return f"Here is the session as a stringified JSON:\n{session_str}"


def build_sessions(
    item: dict,
    session_range: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """
    Extract sessions from a LoCoMo sample.

    Returns list of dicts with keys:
        - content: formatted string for supermemory
        - meta: session metadata
    """
    conv = item["conversation"]

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

        content = build_session_content(item, sk, date_time)

        sessions.append(
            {
                "content": content,
                "meta": {
                    "sample_id": item["sample_id"],
                    "session_key": sk,
                    "date_time": date_time,
                    "speaker_a": conv["speaker_a"],
                    "speaker_b": conv["speaker_b"],
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
    key = f"supermemory:{sample_id}:{session_key}"
    return key in record and record[key].get("success", False)


def mark_ingested(
    sample_id: str,
    session_key: str,
    record: dict,
    doc_id: str,
    meta: Optional[dict] = None,
) -> None:
    key = f"supermemory:{sample_id}:{session_key}"
    record[key] = {
        "success": True,
        "timestamp": int(time.time()),
        "doc_id": doc_id,
        "meta": meta or {},
    }


def write_error_log(path: str, sample_id: str, session_key: str, error: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] ERROR [{sample_id}/{session_key}]: {error}\n")


# ---------------------------------------------------------------------------
# Supermemory indexing poll
# ---------------------------------------------------------------------------

def poll_document(client: Supermemory, doc_id: str, timeout_sec: int = 600) -> str:
    """
    Poll a supermemory document until both doc and memory are done or failed.
    Returns final status: "done", "failed", or "TIMEOUT".
    """
    backoff = 1.0
    start = time.time()

    while True:
        if time.time() - start > timeout_sec:
            return "TIMEOUT"

        try:
            doc = client.documents.get(doc_id)
            doc_status = getattr(doc, "status", None) or (doc.get("status") if isinstance(doc, dict) else None)

            if doc_status == "failed":
                return "failed"

            if doc_status == "done":
                try:
                    mem = client.memories.get(doc_id)
                    mem_status = getattr(mem, "status", None) or (mem.get("status") if isinstance(mem, dict) else None)
                    if mem_status == "done":
                        return "done"
                    elif mem_status == "failed":
                        return "failed"
                except Exception:
                    pass

        except Exception as e:
            print(f"    [poll] Error checking doc {doc_id}: {e}", file=sys.stderr)

        time.sleep(backoff)
        backoff = min(backoff * 1.2, 5.0)


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------

def ingest_session(
    client: Supermemory,
    content: str,
    container_tag: str,
    meta: dict,
    wait_for_indexing: bool = True,
) -> str:
    """
    Add one session's content to Supermemory.
    Returns doc_id.
    """
    response = client.add(
        content=content,
        container_tag=container_tag,
        metadata={
            "session_key": meta.get("session_key", ""),
            "date_time": meta.get("date_time", ""),
            "speaker_a": meta.get("speaker_a", ""),
            "speaker_b": meta.get("speaker_b", ""),
        },
    )

    doc_id = getattr(response, "id", None) or (response.get("id") if isinstance(response, dict) else None)
    if not doc_id:
        raise RuntimeError(f"Supermemory add() returned no id: {response}")

    if wait_for_indexing:
        final_status = poll_document(client, doc_id)
        if final_status != "done":
            raise RuntimeError(f"Document {doc_id} indexing ended with status: {final_status}")

    return doc_id


def run_ingest(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        print("Error: Supermemory API key required (--api-key or SUPERMEMORY_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    client = Supermemory(api_key=api_key)

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

    total_sessions = 0
    success_count = 0
    skip_count = 0
    error_count = 0

    for item in samples:
        sample_id: str = item["sample_id"]
        container_tag = sanitize_tag(sample_id)
        sessions = build_sessions(item, session_range)
        print(f"\n=== Sample {sample_id} ({len(sessions)} sessions) [containerTag={container_tag}] ===", file=sys.stderr)

        for sess in sessions:
            meta = sess["meta"]
            session_key = meta["session_key"]
            label = f"{session_key} ({meta['date_time']})"
            total_sessions += 1

            if not args.force_ingest and is_already_ingested(sample_id, session_key, ingest_record):
                print(f"  [{label}] SKIP (already ingested)", file=sys.stderr)
                skip_count += 1
                continue

            print(f"  [{label}] ingesting ...", file=sys.stderr)
            t0 = time.time()

            try:
                doc_id = ingest_session(
                    client,
                    sess["content"],
                    container_tag,
                    meta,
                    wait_for_indexing=args.wait_indexing,
                )
                elapsed = time.time() - t0
                mark_ingested(sample_id, session_key, ingest_record, doc_id, meta)
                save_ingest_record(ingest_record, args.record)
                print(f"  [{label}] OK  doc_id={doc_id}  {elapsed:.1f}s", file=sys.stderr)
                success_count += 1
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{label}] ERROR: {e}  {elapsed:.1f}s", file=sys.stderr)
                write_error_log(args.error_log, sample_id, session_key, str(e))
                error_count += 1

    print(f"\n=== Ingest summary ===", file=sys.stderr)
    print(f"  Total sessions:  {total_sessions}", file=sys.stderr)
    print(f"  Succeeded:       {success_count}", file=sys.stderr)
    print(f"  Skipped:         {skip_count}", file=sys.stderr)
    print(f"  Failed:          {error_count}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LoCoMo conversations into Supermemory")
    parser.add_argument(
        "--input",
        default=DEFAULT_DATA_PATH,
        help="Path to locomo10.json (default: ../data/locomo10.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Supermemory API key (or set SUPERMEMORY_API_KEY env var)",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Sample index (0-based int) or sample_id string (e.g. conv-26). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to ingest. Default: all.",
    )
    parser.add_argument(
        "--sessions",
        default=None,
        help="Session range, e.g. '1-4' or '3'. Default: all.",
    )
    parser.add_argument(
        "--record",
        default=DEFAULT_RECORD_PATH,
        help=f"Path to ingest progress record (default: {DEFAULT_RECORD_PATH})",
    )
    parser.add_argument(
        "--error-log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to error log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        default=False,
        help="Re-ingest even if already recorded as done",
    )
    parser.add_argument(
        "--clear-ingest-record",
        action="store_true",
        default=False,
        help="Clear all existing ingest records before running",
    )
    parser.add_argument(
        "--no-wait-indexing",
        dest="wait_indexing",
        action="store_false",
        default=True,
        help="Don't wait for Supermemory async indexing to complete (faster but no status check)",
    )

    args = parser.parse_args()
    run_ingest(args)


if __name__ == "__main__":
    main()
