#!/usr/bin/env python3
"""Export a deterministic tensor manifest from a Draw Things .ckpt file.

The manifest is JSONL with one row per tensor name and bounded signatures that are
safe for repeatable diffing across conversion / quantization stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


def _normalize_cell(value: Any) -> str:
    return "NULL" if value is None else str(value)


def _prefix_for_name(name: str) -> str:
    if "[" in name:
        return name.split("[", 1)[0]
    return name


def _load_tensor_keys(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _decode_dim_i32_le(dim_blob: bytes | None, max_values: int) -> Tuple[List[int] | None, bool]:
    if dim_blob is None:
        return None, False
    if len(dim_blob) == 0 or len(dim_blob) % 4 != 0:
        return None, False

    total_values = len(dim_blob) // 4
    truncated = total_values > max_values
    take_values = min(total_values, max_values)

    try:
        values = list(struct.unpack("<" + "i" * take_values, dim_blob[: take_values * 4]))
    except struct.error:
        return None, False

    # Heuristic guard: reject obviously invalid decode outcomes.
    if any(v < -2_000_000_000 or v > 2_000_000_000 for v in values):
        return None, False
    return values, truncated


def _fetch_manifest_row(
    con: sqlite3.Connection,
    name: str,
    head_bytes: int,
    mid_bytes: int,
    tail_bytes: int,
    small_hash_limit: int,
    max_shape_values: int,
) -> Tuple[Dict[str, Any] | None, str | None]:
    cur = con.cursor()

    try:
        row = cur.execute(
            (
                "SELECT "
                "CAST(type AS TEXT), "
                "CAST(format AS TEXT), "
                "CAST(datatype AS TEXT), "
                "length(dim), "
                "length(data), "
                "hex(substr(dim, 1, ?)), "
                "hex(substr(data, 1, ?)) "
                "FROM tensors WHERE name=?"
            ),
            (head_bytes, head_bytes, name),
        ).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    type_text, format_text, datatype_text, dim_len, data_len, dim_head_hex, data_head_hex = row

    dim_tail_hex = None
    data_mid_hex = None
    data_tail_hex = None

    if dim_len is not None and dim_len > 0 and tail_bytes > 0:
        tail_start = int(dim_len) - int(tail_bytes) + 1
        if tail_start < 1:
            tail_start = 1
        try:
            dim_tail_row = cur.execute(
                "SELECT hex(substr(dim, ?, ?)) FROM tensors WHERE name=?",
                (tail_start, tail_bytes, name),
            ).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"
        if dim_tail_row is None:
            return None, "RowMissing"
        dim_tail_hex = _normalize_cell(dim_tail_row[0])

    if data_len is not None and data_len > 0 and mid_bytes > 0:
        mid_start = ((int(data_len) - int(mid_bytes)) // 2) + 1
        if mid_start < 1:
            mid_start = 1
        try:
            data_mid_row = cur.execute(
                "SELECT hex(substr(data, ?, ?)) FROM tensors WHERE name=?",
                (mid_start, mid_bytes, name),
            ).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"
        if data_mid_row is None:
            return None, "RowMissing"
        data_mid_hex = _normalize_cell(data_mid_row[0])

    if data_len is not None and data_len > 0 and tail_bytes > 0:
        tail_start = int(data_len) - int(tail_bytes) + 1
        if tail_start < 1:
            tail_start = 1
        try:
            data_tail_row = cur.execute(
                "SELECT hex(substr(data, ?, ?)) FROM tensors WHERE name=?",
                (tail_start, tail_bytes, name),
            ).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"
        if data_tail_row is None:
            return None, "RowMissing"
        data_tail_hex = _normalize_cell(data_tail_row[0])

    try:
        dim_row = cur.execute("SELECT dim FROM tensors WHERE name=?", (name,)).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if dim_row is None:
        return None, "RowMissing"

    dim_blob_value = dim_row[0]
    if isinstance(dim_blob_value, memoryview):
        dim_blob_value = dim_blob_value.tobytes()
    if dim_blob_value is not None and not isinstance(dim_blob_value, bytes):
        return None, "UnexpectedDimType"

    dim_sha256 = hashlib.sha256(dim_blob_value).hexdigest() if dim_blob_value is not None else "NULL"
    shape_i32_le, shape_i32_truncated = _decode_dim_i32_le(dim_blob_value, max_shape_values)

    data_sha256_small = None
    if data_len is not None and data_len <= small_hash_limit:
        try:
            data_row = cur.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"

        if data_row is None:
            return None, "RowMissing"

        data_value = data_row[0]
        if isinstance(data_value, memoryview):
            data_value = data_value.tobytes()

        if data_value is None:
            data_sha256_small = "NULL"
        elif isinstance(data_value, bytes):
            data_sha256_small = hashlib.sha256(data_value).hexdigest()
        else:
            data_sha256_small = _normalize_cell(data_value)

    metadata = {
        "type": _normalize_cell(type_text),
        "format": _normalize_cell(format_text),
        "datatype": _normalize_cell(datatype_text),
    }

    record = {
        "name": name,
        "prefix": _prefix_for_name(name),
        "metadata": metadata,
        # format is the closest stable codec hint exposed in checkpoint storage.
        "codec_key": f"type={metadata['type']},format={metadata['format']},datatype={metadata['datatype']}",
        "shape_i32_le": shape_i32_le,
        "shape_i32_le_truncated": shape_i32_truncated,
        "dim_len": dim_len,
        "data_len": data_len,
        "signatures": {
            "dim_head_hex": _normalize_cell(dim_head_hex),
            "dim_tail_hex": dim_tail_hex,
            "dim_sha256": dim_sha256,
            "data_head_hex": _normalize_cell(data_head_hex),
            "data_mid_hex": data_mid_hex,
            "data_tail_hex": data_tail_hex,
            "data_sha256_small": data_sha256_small,
        },
        "unreadable_error": None,
    }
    return record, None


def _filtered_keys(
    keys: Sequence[str],
    prefixes: Sequence[str],
    exclude_names: Sequence[str],
    start_index: int,
    max_rows: int,
) -> List[str]:
    selected = sorted(keys)

    if prefixes:
        selected = [name for name in selected if any(name.startswith(prefix) for prefix in prefixes)]

    excluded = set(exclude_names)
    if excluded:
        selected = [name for name in selected if name not in excluded]

    start_pos = start_index - 1
    if start_pos >= len(selected):
        return []

    if max_rows > 0:
        return selected[start_pos : start_pos + max_rows]
    return selected[start_pos:]


def _write_summary_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = [
        "# Checkpoint Tensor Manifest Summary",
        "",
        f"- file: {summary['inputs']['file']}",
        f"- out_jsonl: {summary['inputs']['out_jsonl']}",
        "",
        "## Settings",
        "",
    ]

    for key, value in summary["settings"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Stats", ""])
    for key, value in summary["stats"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Top Prefixes", ""])
    top_prefixes = summary.get("top_prefixes", [])
    if not top_prefixes:
        lines.append("- (none)")
    else:
        for row in top_prefixes:
            lines.append(f"- {row['prefix']}: {row['count']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    file_path: Path,
    out_jsonl: Path,
    out_summary_json: Path,
    out_summary_md: Path | None,
    prefixes: Sequence[str],
    exclude_names: Sequence[str],
    head_bytes: int,
    mid_bytes: int,
    tail_bytes: int,
    small_hash_limit: int,
    max_shape_values: int,
    start_index: int,
    max_rows: int,
    progress_every: int,
    top_prefix_count: int,
) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"missing file: {file_path}")

    con = sqlite3.connect(str(file_path))
    try:
        all_keys = _load_tensor_keys(con)
        selected = _filtered_keys(all_keys, prefixes, exclude_names, start_index, max_rows)

        stats: Dict[str, int] = {
            "tensor_count_total": len(all_keys),
            "tensor_count_selected": len(selected),
            "rows_readable": 0,
            "rows_unreadable": 0,
            "rows_with_shape_i32": 0,
            "rows_with_data_sha256_small": 0,
        }
        prefix_counter: Counter[str] = Counter()

        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f_out:
            for idx, name in enumerate(selected, start=1):
                if progress_every > 0 and idx % progress_every == 0:
                    print(f"progress={idx}/{len(selected)}")

                record, error = _fetch_manifest_row(
                    con=con,
                    name=name,
                    head_bytes=head_bytes,
                    mid_bytes=mid_bytes,
                    tail_bytes=tail_bytes,
                    small_hash_limit=small_hash_limit,
                    max_shape_values=max_shape_values,
                )

                if error is not None:
                    stats["rows_unreadable"] += 1
                    row = {
                        "name": name,
                        "prefix": _prefix_for_name(name),
                        "metadata": None,
                        "codec_key": None,
                        "shape_i32_le": None,
                        "shape_i32_le_truncated": False,
                        "dim_len": None,
                        "data_len": None,
                        "signatures": None,
                        "unreadable_error": error,
                    }
                    f_out.write(json.dumps(row, sort_keys=True) + "\n")
                    continue

                assert record is not None
                stats["rows_readable"] += 1
                prefix_counter[record["prefix"]] += 1

                if record["shape_i32_le"] is not None:
                    stats["rows_with_shape_i32"] += 1

                if record["signatures"]["data_sha256_small"] is not None:
                    stats["rows_with_data_sha256_small"] += 1

                f_out.write(json.dumps(record, sort_keys=True) + "\n")

        ranked_prefixes = [
            {"prefix": prefix, "count": count}
            for prefix, count in prefix_counter.most_common(top_prefix_count)
        ]

        summary: Dict[str, Any] = {
            "inputs": {
                "file": str(file_path),
                "out_jsonl": str(out_jsonl),
            },
            "settings": {
                "head_bytes": head_bytes,
                "mid_bytes": mid_bytes,
                "tail_bytes": tail_bytes,
                "small_hash_limit": small_hash_limit,
                "max_shape_values": max_shape_values,
                "prefixes": list(prefixes),
                "exclude_names_count": len(exclude_names),
                "start_index": start_index,
                "max_rows": max_rows,
                "progress_every": progress_every,
                "top_prefix_count": top_prefix_count,
            },
            "stats": stats,
            "top_prefixes": ranked_prefixes,
        }

        out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        out_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        if out_summary_md is not None:
            _write_summary_markdown(summary, out_summary_md)

        return summary
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export deterministic tensor manifest for Draw Things .ckpt")
    parser.add_argument("--file", required=True, help="Checkpoint file")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument(
        "--out-summary-json",
        default="",
        help="Optional summary JSON path (default: <out-jsonl>.summary.json)",
    )
    parser.add_argument(
        "--out-summary-md",
        default="",
        help="Optional summary markdown path",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional prefix filter (repeatable)",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Optional explicit tensor name to exclude (repeatable)",
    )
    parser.add_argument("--head-bytes", type=int, default=32)
    parser.add_argument("--mid-bytes", type=int, default=64)
    parser.add_argument("--tail-bytes", type=int, default=64)
    parser.add_argument(
        "--small-hash-limit",
        type=int,
        default=262144,
        help="Compute full data SHA256 only when data_len <= limit",
    )
    parser.add_argument("--max-shape-values", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--top-prefix-count", type=int, default=16)

    args = parser.parse_args()

    if args.head_bytes < 1:
        parser.error("--head-bytes must be >= 1")
    if args.mid_bytes < 0:
        parser.error("--mid-bytes must be >= 0")
    if args.tail_bytes < 0:
        parser.error("--tail-bytes must be >= 0")
    if args.small_hash_limit < 0:
        parser.error("--small-hash-limit must be >= 0")
    if args.max_shape_values < 1:
        parser.error("--max-shape-values must be >= 1")
    if args.start_index < 1:
        parser.error("--start-index must be >= 1")
    if args.max_rows < 0:
        parser.error("--max-rows must be >= 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.top_prefix_count < 1:
        parser.error("--top-prefix-count must be >= 1")

    file_path = Path(args.file).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()

    if args.out_summary_json:
        out_summary_json = Path(args.out_summary_json).expanduser().resolve()
    else:
        out_summary_json = out_jsonl.with_suffix(out_jsonl.suffix + ".summary.json")

    out_summary_md = (
        Path(args.out_summary_md).expanduser().resolve() if args.out_summary_md else None
    )

    try:
        summary = run(
            file_path=file_path,
            out_jsonl=out_jsonl,
            out_summary_json=out_summary_json,
            out_summary_md=out_summary_md,
            prefixes=args.prefix,
            exclude_names=args.exclude_name,
            head_bytes=args.head_bytes,
            mid_bytes=args.mid_bytes,
            tail_bytes=args.tail_bytes,
            small_hash_limit=args.small_hash_limit,
            max_shape_values=args.max_shape_values,
            start_index=args.start_index,
            max_rows=args.max_rows,
            progress_every=args.progress_every,
            top_prefix_count=args.top_prefix_count,
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    print("== Tensor Manifest Export ==")
    print(f"file={summary['inputs']['file']}")
    print(f"out_jsonl={summary['inputs']['out_jsonl']}")
    for key, value in summary["stats"].items():
        print(f"{key}={value}")
    print(f"summary_json={out_summary_json}")
    if out_summary_md is not None:
        print(f"summary_md={out_summary_md}")
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
