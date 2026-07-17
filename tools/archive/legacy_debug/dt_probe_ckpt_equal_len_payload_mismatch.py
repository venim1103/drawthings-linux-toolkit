#!/usr/bin/env python3
"""Probe equal-length payload signature mismatches between two Draw Things .ckpt files.

This tool focuses on rows where data_len is equal between candidate and baseline,
but byte signatures differ (data head bytes and optional small-blob SHA256), which
is a high-signal branch for byte-semantic drift after metadata/length parity has
already been achieved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


def _normalize_cell(value: Any) -> str:
    return "NULL" if value is None else str(value)


def _load_keys(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(r[0]) for r in rows if r and r[0] is not None]


def _sig_query() -> str:
    return (
        "SELECT "
        "CAST(type AS TEXT), "
        "CAST(format AS TEXT), "
        "CAST(datatype AS TEXT), "
        "length(data), "
        "hex(substr(data, 1, ?)) "
        "FROM tensors WHERE name=?"
    )


def _fetch_sig(
    con: sqlite3.Connection,
    name: str,
    head_bytes: int,
    mid_bytes: int,
    tail_bytes: int,
    small_hash_limit: int,
) -> Tuple[Dict[str, Any] | None, str | None]:
    cur = con.cursor()
    try:
        row = cur.execute(
            _sig_query(),
            (head_bytes, name),
        ).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    type_text, format_text, datatype_text, data_len, head_hex = row

    data_mid_hex: str | None = None
    data_tail_hex: str | None = None

    if data_len is not None and data_len > 0 and mid_bytes > 0:
        mid_start = ((int(data_len) - int(mid_bytes)) // 2) + 1
        if mid_start < 1:
            mid_start = 1
        try:
            mid_row = cur.execute(
                "SELECT hex(substr(data, ?, ?)) FROM tensors WHERE name=?",
                (mid_start, mid_bytes, name),
            ).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"
        if mid_row is None:
            return None, "RowMissing"
        data_mid_hex = _normalize_cell(mid_row[0])

    if data_len is not None and data_len > 0 and tail_bytes > 0:
        tail_start = int(data_len) - int(tail_bytes) + 1
        if tail_start < 1:
            tail_start = 1
        try:
            tail_row = cur.execute(
                "SELECT hex(substr(data, ?, ?)) FROM tensors WHERE name=?",
                (tail_start, tail_bytes, name),
            ).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"
        if tail_row is None:
            return None, "RowMissing"
        data_tail_hex = _normalize_cell(tail_row[0])

    data_small_sha256: str | None = None
    if data_len is not None and data_len <= small_hash_limit:
        try:
            payload_row = cur.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"

        if payload_row is None:
            return None, "RowMissing"
        payload = payload_row[0]
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if payload is None:
            data_small_sha256 = "NULL"
        elif isinstance(payload, bytes):
            data_small_sha256 = hashlib.sha256(payload).hexdigest()
        else:
            data_small_sha256 = _normalize_cell(payload)

    sig = {
        "metadata": {
            "type": _normalize_cell(type_text),
            "format": _normalize_cell(format_text),
            "datatype": _normalize_cell(datatype_text),
        },
        "data_len": data_len,
        "data_head_hex": _normalize_cell(head_hex),
        "data_mid_hex": data_mid_hex,
        "data_tail_hex": data_tail_hex,
        "data_small_sha256": data_small_sha256,
    }
    return sig, None


def run(
    file_path: Path,
    baseline_path: Path,
    head_bytes: int,
    mid_bytes: int,
    tail_bytes: int,
    small_hash_limit: int,
    progress_every: int,
    prefixes: Sequence[str],
    exclude_names: Sequence[str],
    start_index: int,
    max_rows: int,
    require_metadata_equal: bool,
) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"missing file: {file_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline: {baseline_path}")

    con_file = sqlite3.connect(str(file_path))
    con_baseline = sqlite3.connect(str(baseline_path))
    try:
        file_keys = set(_load_keys(con_file))
        baseline_keys = set(_load_keys(con_baseline))
        shared = sorted(file_keys & baseline_keys)

        if prefixes:
            shared = [name for name in shared if any(name.startswith(p) for p in prefixes)]

        excluded = set(exclude_names)
        if excluded:
            shared = [name for name in shared if name not in excluded]

        total_after_filters = len(shared)
        start_pos = start_index - 1
        if start_pos >= total_after_filters:
            shared = []
        elif max_rows > 0:
            shared = shared[start_pos : start_pos + max_rows]
        else:
            shared = shared[start_pos:]

        stats: Dict[str, int] = {
            "tensor_count_file": len(file_keys),
            "tensor_count_baseline": len(baseline_keys),
            "shared_tensors_total_after_filters": total_after_filters,
            "selected_tensors": len(shared),
            "unreadable_both": 0,
            "unreadable_only_file": 0,
            "unreadable_only_baseline": 0,
            "readable_selected": 0,
            "metadata_mismatch": 0,
            "data_len_equal": 0,
            "data_len_equal_sig_match": 0,
            "data_len_equal_sig_mismatch": 0,
            "data_mid_hex_compared": 0,
            "data_mid_hex_mismatch": 0,
            "data_tail_hex_compared": 0,
            "data_tail_hex_mismatch": 0,
            "data_small_sha256_compared": 0,
            "data_small_sha256_mismatch": 0,
        }

        mismatch_names: List[str] = []
        mismatch_examples: List[Dict[str, Any]] = []

        for idx, name in enumerate(shared, start=1):
            if progress_every > 0 and idx % progress_every == 0:
                print(f"progress={idx}/{len(shared)}")

            file_sig, file_err = _fetch_sig(
                con_file,
                name,
                head_bytes,
                mid_bytes,
                tail_bytes,
                small_hash_limit,
            )
            base_sig, base_err = _fetch_sig(
                con_baseline,
                name,
                head_bytes,
                mid_bytes,
                tail_bytes,
                small_hash_limit,
            )

            if file_err and base_err:
                stats["unreadable_both"] += 1
                continue
            if file_err and not base_err:
                stats["unreadable_only_file"] += 1
                continue
            if base_err and not file_err:
                stats["unreadable_only_baseline"] += 1
                continue

            assert file_sig is not None
            assert base_sig is not None
            stats["readable_selected"] += 1

            meta_equal = all(
                file_sig["metadata"][field] == base_sig["metadata"][field]
                for field in SERIALIZATION_FIELDS
            )
            if not meta_equal:
                stats["metadata_mismatch"] += 1
                if require_metadata_equal:
                    continue

            if file_sig["data_len"] == base_sig["data_len"]:
                stats["data_len_equal"] += 1

                sig_equal = file_sig["data_head_hex"] == base_sig["data_head_hex"]

                if file_sig["data_mid_hex"] is not None and base_sig["data_mid_hex"] is not None:
                    stats["data_mid_hex_compared"] += 1
                    if file_sig["data_mid_hex"] != base_sig["data_mid_hex"]:
                        stats["data_mid_hex_mismatch"] += 1
                        sig_equal = False

                if file_sig["data_tail_hex"] is not None and base_sig["data_tail_hex"] is not None:
                    stats["data_tail_hex_compared"] += 1
                    if file_sig["data_tail_hex"] != base_sig["data_tail_hex"]:
                        stats["data_tail_hex_mismatch"] += 1
                        sig_equal = False

                if (
                    file_sig["data_small_sha256"] is not None
                    and base_sig["data_small_sha256"] is not None
                ):
                    stats["data_small_sha256_compared"] += 1
                    if file_sig["data_small_sha256"] != base_sig["data_small_sha256"]:
                        stats["data_small_sha256_mismatch"] += 1
                        sig_equal = False

                if sig_equal:
                    stats["data_len_equal_sig_match"] += 1
                else:
                    stats["data_len_equal_sig_mismatch"] += 1
                    mismatch_names.append(name)
                    if len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            {
                                "name": name,
                                "data_len": file_sig["data_len"],
                                "file_head": file_sig["data_head_hex"],
                                "base_head": base_sig["data_head_hex"],
                                "file_mid": file_sig["data_mid_hex"],
                                "base_mid": base_sig["data_mid_hex"],
                                "file_tail": file_sig["data_tail_hex"],
                                "base_tail": base_sig["data_tail_hex"],
                                "file_small_sha256": file_sig["data_small_sha256"],
                                "base_small_sha256": base_sig["data_small_sha256"],
                            }
                        )

        return {
            "inputs": {
                "file": str(file_path),
                "baseline": str(baseline_path),
                "head_bytes": head_bytes,
                "mid_bytes": mid_bytes,
                "tail_bytes": tail_bytes,
                "small_hash_limit": small_hash_limit,
                "progress_every": progress_every,
                "prefixes": list(prefixes),
                "exclude_names_count": len(excluded),
                "start_index": start_index,
                "max_rows": max_rows,
                "require_metadata_equal": require_metadata_equal,
            },
            "stats": stats,
            "mismatch_names": mismatch_names,
            "examples": mismatch_examples,
        }
    finally:
        con_file.close()
        con_baseline.close()


def _write_names(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(names)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe equal-length payload signature mismatches")
    parser.add_argument("--file", required=True, help="Candidate checkpoint")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional prefix filter (repeatable), e.g. --prefix __dit__",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Optional explicit tensor name to exclude (repeatable)",
    )
    parser.add_argument(
        "--head-bytes",
        type=int,
        default=32,
        help="Signature bytes for head comparison (default: 32)",
    )
    parser.add_argument(
        "--mid-bytes",
        type=int,
        default=0,
        help="Signature bytes for middle-window comparison (default: 0=disabled)",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=0,
        help="Signature bytes for tail-window comparison (default: 0=disabled)",
    )
    parser.add_argument(
        "--small-hash-limit",
        type=int,
        default=4096,
        help="Compare full SHA256 for payloads <= this size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N rows (default: 500; 0 disables)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based start index in filtered shared tensor list (default: 1)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Max rows to process from start index; 0 means all remaining (default: 0)",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Include rows even when type/format/datatype differs",
    )
    parser.add_argument(
        "--out-mismatch-names",
        required=True,
        help="Output file for mismatch names (one per line)",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON report output path")

    args = parser.parse_args()

    if args.head_bytes < 1:
        parser.error("--head-bytes must be >= 1")
    if args.mid_bytes < 0:
        parser.error("--mid-bytes must be >= 0")
    if args.tail_bytes < 0:
        parser.error("--tail-bytes must be >= 0")
    if args.small_hash_limit < 0:
        parser.error("--small-hash-limit must be >= 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.start_index < 1:
        parser.error("--start-index must be >= 1")
    if args.max_rows < 0:
        parser.error("--max-rows must be >= 0")

    file_path = Path(args.file).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    out_names = Path(args.out_mismatch_names).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else None

    try:
        report = run(
            file_path=file_path,
            baseline_path=baseline_path,
            head_bytes=args.head_bytes,
            mid_bytes=args.mid_bytes,
            tail_bytes=args.tail_bytes,
            small_hash_limit=args.small_hash_limit,
            progress_every=args.progress_every,
            prefixes=args.prefix,
            exclude_names=args.exclude_name,
            start_index=args.start_index,
            max_rows=args.max_rows,
            require_metadata_equal=(not args.allow_metadata_mismatch),
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    _write_names(out_names, report["mismatch_names"])

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("== Equal-Length Payload Mismatch Probe ==")
    print(f"file={report['inputs']['file']}")
    print(f"baseline={report['inputs']['baseline']}")
    print(f"head_bytes={report['inputs']['head_bytes']}")
    print(f"mid_bytes={report['inputs']['mid_bytes']}")
    print(f"tail_bytes={report['inputs']['tail_bytes']}")
    print(f"small_hash_limit={report['inputs']['small_hash_limit']}")
    print(f"progress_every={report['inputs']['progress_every']}")
    print(f"prefixes={report['inputs']['prefixes']}")
    print(f"exclude_names_count={report['inputs']['exclude_names_count']}")
    print(f"start_index={report['inputs']['start_index']}")
    print(f"max_rows={report['inputs']['max_rows']}")
    print(f"require_metadata_equal={report['inputs']['require_metadata_equal']}")

    for key, value in report["stats"].items():
        print(f"{key}={value}")

    print(f"mismatch_names_file={out_names}")
    print(f"mismatch_names_count={len(report['mismatch_names'])}")
    if out_json is not None:
        print(f"json_report={out_json}")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
