#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
EXPORT_SCRIPT="$ROOT/tools/dt_export_ckpt_tensor_manifest.py"
COMPARE_SCRIPT="$ROOT/tools/dt_compare_ckpt_tensor_manifests.py"

BASELINE_F16=""
CANDIDATE_F16=""
BASELINE_Q6P=""
CANDIDATE_Q6P=""

TAG="$(date +%Y%m%d_%H%M%S)"
HEAD_BYTES=32
MID_BYTES=64
TAIL_BYTES=64
SMALL_HASH_LIMIT=262144
MAX_SHAPE_VALUES=32
START_INDEX=1
MAX_ROWS=0
PROGRESS_EVERY=500
SAMPLE_LIMIT=20
TOP_PREFIX_COUNT=16
FAIL_ON_DIVERGENCE=0

PREFIXES=()
EXCLUDE_NAMES=()

usage() {
  cat <<'EOF'
Usage:
  tools/run_ltx23_first_divergence_stage.sh [options]

Purpose:
  Compare baseline vs candidate checkpoints at f16 and q6p stages using
  deterministic per-tensor manifests, then report the earliest divergent stage.

Required:
  --baseline-f16 <path>
  --candidate-f16 <path>
  --baseline-q6p <path>
  --candidate-q6p <path>

Options:
  --tag <value>                 Output tag (default: timestamp).

  --head-bytes <n>              Head signature bytes (default: 32).
  --mid-bytes <n>               Mid-window signature bytes (default: 64).
  --tail-bytes <n>              Tail signature bytes (default: 64).
  --small-hash-limit <n>        Full data SHA256 limit (default: 262144).
  --max-shape-values <n>        Max decoded shape ints in manifest (default: 32).

  --start-index <n>             1-based row start for manifest export (default: 1).
  --max-rows <n>                Max rows to export; 0 means all (default: 0).
  --progress-every <n>          Progress logging interval (default: 500).

  --sample-limit <n>            Comparator sample limit (default: 20).
  --top-prefix-count <n>        Comparator prefix count (default: 16).
  --prefix <value>              Optional prefix filter (repeatable).
  --exclude-name <value>        Optional exact name exclusion (repeatable).

  --fail-on-divergence          Exit non-zero when earliest divergent stage is not none.
  -h, --help                    Show this help.

Outputs:
  output/first_divergence_<tag>/
    - manifests/*.jsonl
    - compare_f16.json / .md
    - compare_q6p.json / .md
    - summary.json / summary.md
EOF
}

abs_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    echo "$value"
  else
    echo "$ROOT/$value"
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-f16)
      BASELINE_F16="${2:-}"
      shift 2
      ;;
    --candidate-f16)
      CANDIDATE_F16="${2:-}"
      shift 2
      ;;
    --baseline-q6p)
      BASELINE_Q6P="${2:-}"
      shift 2
      ;;
    --candidate-q6p)
      CANDIDATE_Q6P="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --head-bytes)
      HEAD_BYTES="${2:-}"
      shift 2
      ;;
    --mid-bytes)
      MID_BYTES="${2:-}"
      shift 2
      ;;
    --tail-bytes)
      TAIL_BYTES="${2:-}"
      shift 2
      ;;
    --small-hash-limit)
      SMALL_HASH_LIMIT="${2:-}"
      shift 2
      ;;
    --max-shape-values)
      MAX_SHAPE_VALUES="${2:-}"
      shift 2
      ;;
    --start-index)
      START_INDEX="${2:-}"
      shift 2
      ;;
    --max-rows)
      MAX_ROWS="${2:-}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:-}"
      shift 2
      ;;
    --sample-limit)
      SAMPLE_LIMIT="${2:-}"
      shift 2
      ;;
    --top-prefix-count)
      TOP_PREFIX_COUNT="${2:-}"
      shift 2
      ;;
    --prefix)
      PREFIXES+=("${2:-}")
      shift 2
      ;;
    --exclude-name)
      EXCLUDE_NAMES+=("${2:-}")
      shift 2
      ;;
    --fail-on-divergence)
      FAIL_ON_DIVERGENCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASELINE_F16" || -z "$CANDIDATE_F16" || -z "$BASELINE_Q6P" || -z "$CANDIDATE_Q6P" ]]; then
  echo "error: all baseline/candidate stage paths are required" >&2
  usage
  exit 1
fi

BASELINE_F16="$(abs_path "$BASELINE_F16")"
CANDIDATE_F16="$(abs_path "$CANDIDATE_F16")"
BASELINE_Q6P="$(abs_path "$BASELINE_Q6P")"
CANDIDATE_Q6P="$(abs_path "$CANDIDATE_Q6P")"

for p in "$PYTHON_BIN" "$EXPORT_SCRIPT" "$COMPARE_SCRIPT" "$BASELINE_F16" "$CANDIDATE_F16" "$BASELINE_Q6P" "$CANDIDATE_Q6P"; do
  if [[ ! -e "$p" ]]; then
    echo "error: missing required path: $p" >&2
    exit 1
  fi
done

for n in "$HEAD_BYTES" "$MID_BYTES" "$TAIL_BYTES" "$SMALL_HASH_LIMIT" "$MAX_SHAPE_VALUES" "$START_INDEX" "$MAX_ROWS" "$PROGRESS_EVERY" "$SAMPLE_LIMIT" "$TOP_PREFIX_COUNT"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "error: numeric option contains non-integer value: $n" >&2
    exit 1
  fi
done

WORK_DIR="$ROOT/output/first_divergence_${TAG}"
MANIFEST_DIR="$WORK_DIR/manifests"
mkdir -p "$MANIFEST_DIR"

build_manifest_cmd() {
  local ckpt_path="$1"
  local out_jsonl="$2"
  local out_summary_json="$3"
  local out_summary_md="$4"

  local cmd=(
    "$PYTHON_BIN" "$EXPORT_SCRIPT"
    --file "$ckpt_path"
    --out-jsonl "$out_jsonl"
    --out-summary-json "$out_summary_json"
    --out-summary-md "$out_summary_md"
    --head-bytes "$HEAD_BYTES"
    --mid-bytes "$MID_BYTES"
    --tail-bytes "$TAIL_BYTES"
    --small-hash-limit "$SMALL_HASH_LIMIT"
    --max-shape-values "$MAX_SHAPE_VALUES"
    --start-index "$START_INDEX"
    --max-rows "$MAX_ROWS"
    --progress-every "$PROGRESS_EVERY"
    --top-prefix-count "$TOP_PREFIX_COUNT"
  )

  for prefix in "${PREFIXES[@]}"; do
    cmd+=(--prefix "$prefix")
  done
  for name in "${EXCLUDE_NAMES[@]}"; do
    cmd+=(--exclude-name "$name")
  done

  "${cmd[@]}"
}

run_manifest_export() {
  local label="$1"
  local ckpt_path="$2"

  local out_jsonl="$MANIFEST_DIR/${label}.jsonl"
  local out_summary_json="$MANIFEST_DIR/${label}.summary.json"
  local out_summary_md="$MANIFEST_DIR/${label}.summary.md"

  echo "== Export manifest: $label =="
  build_manifest_cmd "$ckpt_path" "$out_jsonl" "$out_summary_json" "$out_summary_md"
}

run_compare() {
  local label="$1"
  local manifest_a="$2"
  local manifest_b="$3"
  local out_json="$4"
  local out_md="$5"
  local out_names="$6"

  echo "== Compare manifests: $label =="
  "$PYTHON_BIN" "$COMPARE_SCRIPT" \
    --manifest-a "$manifest_a" \
    --manifest-b "$manifest_b" \
    --label-a baseline \
    --label-b candidate \
    --sample-limit "$SAMPLE_LIMIT" \
    --top-prefix-count "$TOP_PREFIX_COUNT" \
    --out-json "$out_json" \
    --out-md "$out_md" \
    --out-mismatch-names "$out_names"
}

run_manifest_export "baseline_f16" "$BASELINE_F16"
run_manifest_export "candidate_f16" "$CANDIDATE_F16"
run_manifest_export "baseline_q6p" "$BASELINE_Q6P"
run_manifest_export "candidate_q6p" "$CANDIDATE_Q6P"

run_compare \
  "f16" \
  "$MANIFEST_DIR/baseline_f16.jsonl" \
  "$MANIFEST_DIR/candidate_f16.jsonl" \
  "$WORK_DIR/compare_f16.json" \
  "$WORK_DIR/compare_f16.md" \
  "$WORK_DIR/compare_f16_mismatch_names.txt"

run_compare \
  "q6p" \
  "$MANIFEST_DIR/baseline_q6p.jsonl" \
  "$MANIFEST_DIR/candidate_q6p.jsonl" \
  "$WORK_DIR/compare_q6p.json" \
  "$WORK_DIR/compare_q6p.md" \
  "$WORK_DIR/compare_q6p_mismatch_names.txt"

FIRST_STAGE="$("$PYTHON_BIN" - "$WORK_DIR/compare_f16.json" "$WORK_DIR/compare_q6p.json" <<'PY'
import json
import sys

f16 = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
q6p = json.load(open(sys.argv[2], 'r', encoding='utf-8'))

f16_diverged = (
    f16['stats']['mismatch_rows'] > 0
    or f16['stats']['missing_in_a'] > 0
    or f16['stats']['missing_in_b'] > 0
)
q6p_diverged = (
    q6p['stats']['mismatch_rows'] > 0
    or q6p['stats']['missing_in_a'] > 0
    or q6p['stats']['missing_in_b'] > 0
)

if f16_diverged:
    print('f16')
elif q6p_diverged:
    print('q6p')
else:
    print('none')
PY
)"

SUMMARY_JSON="$WORK_DIR/summary.json"
SUMMARY_MD="$WORK_DIR/summary.md"

"$PYTHON_BIN" - "$WORK_DIR" "$FIRST_STAGE" "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
first_stage = sys.argv[2]
out_json = Path(sys.argv[3])

f16 = json.load(open(work_dir / 'compare_f16.json', 'r', encoding='utf-8'))
q6p = json.load(open(work_dir / 'compare_q6p.json', 'r', encoding='utf-8'))

summary = {
    'work_dir': str(work_dir),
    'first_divergent_stage': first_stage,
    'f16': {
        'mismatch_rows': f16['stats']['mismatch_rows'],
        'missing_in_a': f16['stats']['missing_in_a'],
        'missing_in_b': f16['stats']['missing_in_b'],
        'first_divergence': f16['first_divergence'],
    },
    'q6p': {
        'mismatch_rows': q6p['stats']['mismatch_rows'],
        'missing_in_a': q6p['stats']['missing_in_a'],
        'missing_in_b': q6p['stats']['missing_in_b'],
        'first_divergence': q6p['first_divergence'],
    },
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(summary, sort_keys=True))
PY

cat > "$SUMMARY_MD" <<EOF
# LTX2.3 First Divergence Stage

- work_dir: $WORK_DIR
- first_divergent_stage: $FIRST_STAGE

## Compare Outputs

- f16 compare json: $WORK_DIR/compare_f16.json
- f16 compare md: $WORK_DIR/compare_f16.md
- q6p compare json: $WORK_DIR/compare_q6p.json
- q6p compare md: $WORK_DIR/compare_q6p.md

## Manifest Outputs

- baseline_f16: $MANIFEST_DIR/baseline_f16.jsonl
- candidate_f16: $MANIFEST_DIR/candidate_f16.jsonl
- baseline_q6p: $MANIFEST_DIR/baseline_q6p.jsonl
- candidate_q6p: $MANIFEST_DIR/candidate_q6p.jsonl
EOF

echo "summary_json=$SUMMARY_JSON"
echo "summary_md=$SUMMARY_MD"
echo "first_divergent_stage=$FIRST_STAGE"

if [[ "$FAIL_ON_DIVERGENCE" == "1" && "$FIRST_STAGE" != "none" ]]; then
  echo "RESULT=FAIL"
  exit 1
fi

echo "RESULT=PASS"
