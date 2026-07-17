#!/usr/bin/env python3
"""Compare two ckpt tensor manifest JSONL files and report first divergence.

The comparison is deterministic: tensor names are processed in lexical order and
field checks follow a fixed priority list.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

FIELD_ORDER: Sequence[str] = (
    "unreadable_error",
    "metadata.type",
    "metadata.format",
    "metadata.datatype",
    "codec_key",
    "dim_len",
    "data_len",
    "shape_i32_le",
    "shape_i32_le_truncated",
    "signatures.dim_head_hex",
    "signatures.dim_tail_hex",
    "signatures.dim_sha256",
    "signatures.data_head_hex",
    "signatures.data_mid_hex",
    "signatures.data_tail_hex",
    "signatures.data_sha256_small",
)


def _prefix_for_name(name: str) -> str:
    if "[" in name:
        return name.split("[", 1)[0]
    return name


def _read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest file: {path}")

    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc

            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"manifest row missing valid name at {path}:{line_no}")
            if name in rows:
                raise ValueError(f"duplicate tensor name in manifest: {name}")
            rows[name] = row

    return rows


def _extract_field(row: Dict[str, Any], field_path: str) -> Any:
    value: Any = row
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return "__MISSING__"
        if part not in value:
            return "__MISSING__"
        value = value[part]
    return value


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_markdown(report: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = [
        "# Tensor Manifest Compare",
        "",
        f"- manifest_a: {report['inputs']['manifest_a']}",
        f"- manifest_b: {report['inputs']['manifest_b']}",
        f"- label_a: {report['inputs']['label_a']}",
        f"- label_b: {report['inputs']['label_b']}",
        "",
        "## Stats",
        "",
    ]

    for key, value in report["stats"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## First Divergence", ""])
    first = report.get("first_divergence")
    if first is None:
        lines.append("- none")
    else:
        lines.append(f"- name: {first['name']}")
        lines.append(f"- prefix: {first['prefix']}")
        lines.append(f"- field: {first['field']}")
        lines.append(f"- {report['inputs']['label_a']}: {json.dumps(first['a_value'], sort_keys=True)}")
        lines.append(f"- {report['inputs']['label_b']}: {json.dumps(first['b_value'], sort_keys=True)}")

    lines.extend(["", "## Top Prefixes", ""])
    if not report["top_prefixes"]:
        lines.append("- (none)")
    else:
        for row in report["top_prefixes"]:
            lines.append(f"- {row['prefix']}: {row['mismatch_rows']}")

    lines.extend(["", "## Sample Mismatches", ""])
    if not report["samples"]:
        lines.append("- (none)")
    else:
        for sample in report["samples"]:
            lines.append(f"- {json.dumps(sample, sort_keys=True)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    manifest_a: Path,
    manifest_b: Path,
    label_a: str,
    label_b: str,
    sample_limit: int,
    top_prefix_count: int,
) -> Dict[str, Any]:
    rows_a = _read_manifest(manifest_a)
    rows_b = _read_manifest(manifest_b)

    names_a = set(rows_a.keys())
    names_b = set(rows_b.keys())

    missing_in_a = sorted(names_b - names_a)
    missing_in_b = sorted(names_a - names_b)
    shared_names = sorted(names_a & names_b)

    field_mismatch_counts: Counter[str] = Counter()
    prefix_mismatch_rows: Counter[str] = Counter()
    mismatch_names: List[str] = []
    samples: List[Dict[str, Any]] = []
    first_divergence: Dict[str, Any] | None = None

    for name in shared_names:
        row_a = rows_a[name]
        row_b = rows_b[name]
        prefix = row_a.get("prefix") or row_b.get("prefix") or _prefix_for_name(name)

        row_fields_mismatched: List[str] = []
        for field in FIELD_ORDER:
            a_value = _extract_field(row_a, field)
            b_value = _extract_field(row_b, field)
            if _json_value(a_value) == _json_value(b_value):
                continue

            field_mismatch_counts[field] += 1
            row_fields_mismatched.append(field)

            if first_divergence is None:
                first_divergence = {
                    "name": name,
                    "prefix": prefix,
                    "field": field,
                    "a_value": a_value,
                    "b_value": b_value,
                }

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "name": name,
                        "prefix": prefix,
                        "field": field,
                        label_a: a_value,
                        label_b: b_value,
                    }
                )

        if row_fields_mismatched:
            mismatch_names.append(name)
            prefix_mismatch_rows[prefix] += 1

    top_prefixes = [
        {"prefix": prefix, "mismatch_rows": count}
        for prefix, count in prefix_mismatch_rows.most_common(top_prefix_count)
    ]

    report: Dict[str, Any] = {
        "inputs": {
            "manifest_a": str(manifest_a),
            "manifest_b": str(manifest_b),
            "label_a": label_a,
            "label_b": label_b,
        },
        "stats": {
            "manifest_a_rows": len(rows_a),
            "manifest_b_rows": len(rows_b),
            "missing_in_a": len(missing_in_a),
            "missing_in_b": len(missing_in_b),
            "shared_rows": len(shared_names),
            "mismatch_rows": len(mismatch_names),
            "field_mismatch_total": sum(field_mismatch_counts.values()),
        },
        "field_mismatch_counts": dict(field_mismatch_counts),
        "first_divergence": first_divergence,
        "top_prefixes": top_prefixes,
        "samples": samples,
        "missing_names": {
            "missing_in_a": missing_in_a,
            "missing_in_b": missing_in_b,
        },
        "mismatch_names": mismatch_names,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ckpt tensor manifest JSONL files")
    parser.add_argument("--manifest-a", required=True, help="Baseline manifest JSONL")
    parser.add_argument("--manifest-b", required=True, help="Candidate manifest JSONL")
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--top-prefix-count", type=int, default=16)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-mismatch-names", default="")

    args = parser.parse_args()

    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.top_prefix_count < 1:
        parser.error("--top-prefix-count must be >= 1")

    manifest_a = Path(args.manifest_a).expanduser().resolve()
    manifest_b = Path(args.manifest_b).expanduser().resolve()

    try:
        report = run(
            manifest_a=manifest_a,
            manifest_b=manifest_b,
            label_a=args.label_a,
            label_b=args.label_b,
            sample_limit=args.sample_limit,
            top_prefix_count=args.top_prefix_count,
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    print("== Tensor Manifest Compare ==")
    print(f"manifest_a={report['inputs']['manifest_a']}")
    print(f"manifest_b={report['inputs']['manifest_b']}")
    for key, value in report["stats"].items():
        print(f"{key}={value}")

    if report["first_divergence"] is None:
        print("first_divergence=none")
    else:
        first = report["first_divergence"]
        print(f"first_divergence_name={first['name']}")
        print(f"first_divergence_field={first['field']}")

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"json_report={out_json}")

    if args.out_md:
        out_md = Path(args.out_md).expanduser().resolve()
        _write_markdown(report, out_md)
        print(f"markdown_report={out_md}")

    if args.out_mismatch_names:
        out_names = Path(args.out_mismatch_names).expanduser().resolve()
        out_names.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(report["mismatch_names"])
        if text:
            text += "\n"
        out_names.write_text(text, encoding="utf-8")
        print(f"mismatch_names_file={out_names}")

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
