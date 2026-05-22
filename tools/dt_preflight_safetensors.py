#!/usr/bin/env python3
"""Fast preflight checks for safetensors conversion inputs.

This audits the file format and model-key layout without decoding full tensors,
so it finishes quickly even for very large checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


DTYPE_BYTES: Dict[str, int] = {
    "F16": 2,
    "FLOAT16": 2,
    "BF16": 2,
    "BFLOAT16": 2,
    "F32": 4,
    "FLOAT32": 4,
    "FLOAT": 4,
    "F64": 8,
    "FLOAT64": 8,
    "DOUBLE": 8,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

LTX23_REQUIRED_KEYS = [
    "transformer_blocks.47.attn1.to_q.weight",
    "audio_patchify_proj.weight",
    "video_embeddings_connector.learnable_registers",
    "audio_embeddings_connector.learnable_registers",
    "video_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
    "audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
]


@dataclass
class AuditResult:
    file_size: int
    header_len: int
    buffer_start: int
    data_bytes: int
    tensor_count: int
    dtype_counts: Counter
    unknown_dtype: List[Tuple[str, str]]
    out_of_range: List[Tuple[str, int, int]]
    size_mismatch: List[Tuple]
    bad_ranges: List[Tuple]
    overlaps: List[Tuple]
    missing_required: List[str]
    max_tensor_key: str
    max_tensor_range: Tuple[int, int]


def parse_header(path: str) -> Tuple[dict, int, int, int]:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Cannot read 8-byte safetensors header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len >= file_size:
            raise ValueError(f"Invalid header length: {header_len}")
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header JSON is not an object")
    buffer_start = 8 + header_len
    data_bytes = file_size - buffer_start
    return header, file_size, header_len, data_bytes


def has_key(header: dict, key: str) -> bool:
    return (
        key in header
        or f"model.diffusion_model.{key}" in header
        or f"model.{key}" in header
    )


def audit(path: str, required_keys: List[str]) -> AuditResult:
    header, file_size, header_len, data_bytes = parse_header(path)

    keys = [k for k in header.keys() if k != "__metadata__"]
    dtype_counts: Counter = Counter(str(header[k].get("dtype", "")).upper() for k in keys)

    unknown_dtype: List[Tuple[str, str]] = []
    out_of_range: List[Tuple[str, int, int]] = []
    size_mismatch: List[Tuple] = []
    bad_ranges: List[Tuple] = []
    overlaps: List[Tuple] = []
    segments: List[Tuple[int, int, str]] = []

    for key in keys:
        value = header[key]
        dtype = str(value.get("dtype", "")).upper()
        shape = value.get("shape", [])
        offsets = value.get("data_offsets", [])

        if not (isinstance(offsets, list) and len(offsets) == 2):
            bad_ranges.append((key, "bad_offsets", offsets))
            continue

        start, end = offsets
        if not (isinstance(start, int) and isinstance(end, int)):
            bad_ranges.append((key, "nonint_offsets", offsets))
            continue

        if end <= start or start < 0:
            bad_ranges.append((key, "nonpositive_range", offsets))

        # Strict safetensors check: offsets are relative to data section,
        # so they must be within data_bytes (not full file size).
        if end > data_bytes:
            out_of_range.append((key, start, end))

        segments.append((start, end, key))

        if dtype not in DTYPE_BYTES:
            unknown_dtype.append((key, dtype))
            continue

        if not isinstance(shape, list):
            size_mismatch.append((key, "shape_not_list", shape, dtype, start, end))
            continue

        nelem = 1
        bad_shape = False
        for dim in shape:
            if not isinstance(dim, int) or dim < 0:
                bad_shape = True
                break
            nelem *= dim

        if bad_shape:
            size_mismatch.append((key, "bad_shape", shape, dtype, start, end))
            continue

        expected = nelem * DTYPE_BYTES[dtype]
        got = end - start
        if expected != got:
            size_mismatch.append((key, "bytes_mismatch", expected, got, dtype, shape))

    segments.sort(key=lambda x: (x[0], x[1]))
    for i in range(1, len(segments)):
        p_start, p_end, p_key = segments[i - 1]
        start, end, key = segments[i]
        if start < p_end:
            overlaps.append((p_key, key, p_start, p_end, start, end))

    missing_required = [key for key in required_keys if not has_key(header, key)]

    max_tensor_key = max(keys, key=lambda k: header[k]["data_offsets"][1]) if keys else ""
    max_start, max_end = header[max_tensor_key]["data_offsets"] if max_tensor_key else (0, 0)

    return AuditResult(
        file_size=file_size,
        header_len=header_len,
        buffer_start=8 + header_len,
        data_bytes=data_bytes,
        tensor_count=len(keys),
        dtype_counts=dtype_counts,
        unknown_dtype=unknown_dtype,
        out_of_range=out_of_range,
        size_mismatch=size_mismatch,
        bad_ranges=bad_ranges,
        overlaps=overlaps,
        missing_required=missing_required,
        max_tensor_key=max_tensor_key,
        max_tensor_range=(max_start, max_end),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast safetensors preflight audit")
    parser.add_argument("--file", required=True, help="Path to safetensors file")
    parser.add_argument(
        "--profile",
        choices=["ltx2_3", "none"],
        default="ltx2_3",
        help="Model profile for required key checks",
    )
    args = parser.parse_args()

    required = LTX23_REQUIRED_KEYS if args.profile == "ltx2_3" else []

    try:
        result = audit(args.file, required)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FAIL: {exc}")
        return 2

    print("== Safetensors Preflight ==")
    print(f"file={args.file}")
    print(
        f"file_size={result.file_size} header_len={result.header_len} "
        f"buffer_start={result.buffer_start} data_bytes={result.data_bytes}"
    )
    print(f"tensor_count={result.tensor_count}")
    print(f"dtype_counts={dict(result.dtype_counts.most_common(12))}")

    print("== Structural Checks ==")
    print(
        "unknown_dtype={} out_of_range={} size_mismatch={} bad_ranges={} overlaps={}".format(
            len(result.unknown_dtype),
            len(result.out_of_range),
            len(result.size_mismatch),
            len(result.bad_ranges),
            len(result.overlaps),
        )
    )

    if result.out_of_range:
        print("sample_out_of_range=", result.out_of_range[:5])
    if result.size_mismatch:
        print("sample_size_mismatch=", result.size_mismatch[:5])
    if result.overlaps:
        print("sample_overlap=", result.overlaps[:5])

    if required:
        print("== Required Key Checks ==")
        print(f"missing_required={result.missing_required}")

    if result.max_tensor_key:
        s, e = result.max_tensor_range
        print(
            f"max_tensor={result.max_tensor_key} rel_offsets=({s},{e}) "
            f"abs_end={result.buffer_start + e}"
        )

    failures = (
        len(result.unknown_dtype)
        + len(result.out_of_range)
        + len(result.size_mismatch)
        + len(result.bad_ranges)
        + len(result.overlaps)
        + len(result.missing_required)
    )

    if failures:
        print("RESULT=FAIL")
        return 1

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
