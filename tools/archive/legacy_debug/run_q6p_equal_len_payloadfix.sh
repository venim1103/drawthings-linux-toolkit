#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
PROBE_SCRIPT="$ROOT/tools/dt_probe_ckpt_equal_len_payload_mismatch.py"
RECURSIVE_SCRIPT="$ROOT/tools/run_q6p_payloadfix_recursive.sh"
MATRIX_SCRIPT="$ROOT/tools/run_10e_v1_model_validity_matrix.sh"

TARGET_Q6P="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
BASELINE_F16="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
SOURCE_SAFETENSORS="$ROOT/dt-models/10_e_v1_bf16.safetensors"

TAG="$(date +%Y%m%d_%H%M%S)_q6p_equal_len_payloadfix"
HOST="127.0.0.1:7861"

HEAD_BYTES=32
PROBE_SMALL_HASH_LIMIT=4096
PROBE_PROGRESS_EVERY=400
PROBE_CHUNK_ROWS=0
ALLOW_METADATA_MISMATCH=0

INITIAL_SPLIT_LINES=200
CHUNK_SIZE=4
JOURNAL_MODE=delete
MIN_FREE_GB=50
CANARY_TIMEOUT_SEC=120
CANARY_TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
FINAL_MODE=0
CANARY_NO_TIMEOUT=0

RUN_MATRIX=1
MATRIX_SKIP_F16_CANARY=1
MATRIX_STRICT_ALL=0

declare -a PREFIXES=()
declare -a EXCLUDE_NAMES=()

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_equal_len_payloadfix.sh [options]

Purpose:
  1) Probe equal-length payload signature mismatches (q6p vs f16 baseline).
  2) Apply baseline payload copy recursively for the mismatch names.
  3) Optionally run post-fix matrix gate and emit summary artifacts.

Options:
  --target-q6p <path>              Target q6p checkpoint.
  --baseline-f16 <path>            Baseline f16 checkpoint.
  --source-safetensors <path>      Source safetensors for matrix validation.
  --tag <value>                    Output tag (default: timestamped).
  --host <host:port>               Runtime host (default: 127.0.0.1:7861).

  --head-bytes <n>                 Probe signature bytes (default: 32).
  --probe-small-hash-limit <n>     Full SHA compare threshold in bytes (default: 4096).
  --probe-progress-every <n>       Probe progress interval (default: 400).
  --probe-chunk-rows <n>           Process probe in fixed row chunks; 0 means single run (default: 0).
  --prefix <value>                 Optional probe prefix filter (repeatable).
  --exclude-name <value>           Optional explicit probe exclusion (repeatable).
  --allow-metadata-mismatch        Include rows with metadata mismatch.

  --initial-split-lines <n>        Recursive apply initial split size (default: 200).
  --chunk-size <n>                 Recursive apply write chunk size (default: 4).
  --journal-mode <delete|wal|preserve>
                                   SQLite journal mode for apply (default: delete).
  --min-free-gb <n>                Disk floor for apply (default: 50).
  --canary-timeout-sec <n>         Canary timeout (default: 120).
  --final-mode                     Use longer final validation timeout (900s)
                                   unless --canary-timeout-sec is explicitly set.
  --canary-no-timeout              Disable canary timeout wrapper.
  --max-responses <n>              Canary max responses (default: 10).

  --skip-matrix                    Skip post-fix matrix gate.
  --matrix-run-f16-canary          Enable f16 canary in matrix (default skipped).
  --matrix-strict-all              Make matrix fail on q6p canary failure.

  -h, --help                       Show help.

Outputs:
  output/<tag>/
    - equal_len_payload_probe.log
    - equal_len_payload_probe.json
    - equal_len_payload_mismatch_names.txt
    - recursive_payloadfix.log
    - matrix.log (if enabled)
    - summary.md
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

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target-q6p)
        TARGET_Q6P="${2:-}"
        shift 2
        ;;
      --baseline-f16)
        BASELINE_F16="${2:-}"
        shift 2
        ;;
      --source-safetensors)
        SOURCE_SAFETENSORS="${2:-}"
        shift 2
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
        ;;
      --host)
        HOST="${2:-}"
        shift 2
        ;;
      --head-bytes)
        HEAD_BYTES="${2:-}"
        shift 2
        ;;
      --probe-small-hash-limit)
        PROBE_SMALL_HASH_LIMIT="${2:-}"
        shift 2
        ;;
      --probe-progress-every)
        PROBE_PROGRESS_EVERY="${2:-}"
        shift 2
        ;;
      --probe-chunk-rows)
        PROBE_CHUNK_ROWS="${2:-}"
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
      --allow-metadata-mismatch)
        ALLOW_METADATA_MISMATCH=1
        shift
        ;;
      --initial-split-lines)
        INITIAL_SPLIT_LINES="${2:-}"
        shift 2
        ;;
      --chunk-size)
        CHUNK_SIZE="${2:-}"
        shift 2
        ;;
      --journal-mode)
        JOURNAL_MODE="${2:-}"
        shift 2
        ;;
      --min-free-gb)
        MIN_FREE_GB="${2:-}"
        shift 2
        ;;
      --canary-timeout-sec)
        CANARY_TIMEOUT_SEC="${2:-}"
        CANARY_TIMEOUT_EXPLICIT=1
        shift 2
        ;;
      --final-mode)
        FINAL_MODE=1
        shift
        ;;
      --canary-no-timeout)
        CANARY_NO_TIMEOUT=1
        shift
        ;;
      --max-responses)
        MAX_RESPONSES="${2:-}"
        shift 2
        ;;
      --skip-matrix)
        RUN_MATRIX=0
        shift
        ;;
      --matrix-run-f16-canary)
        MATRIX_SKIP_F16_CANARY=0
        shift
        ;;
      --matrix-strict-all)
        MATRIX_STRICT_ALL=1
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
fi

if [[ "$FINAL_MODE" == "1" && "$CANARY_TIMEOUT_EXPLICIT" != "1" && "$CANARY_NO_TIMEOUT" != "1" ]]; then
  CANARY_TIMEOUT_SEC=900
fi

TARGET_Q6P="$(abs_path "$TARGET_Q6P")"
BASELINE_F16="$(abs_path "$BASELINE_F16")"
SOURCE_SAFETENSORS="$(abs_path "$SOURCE_SAFETENSORS")"

for p in "$PYTHON_BIN" "$PROBE_SCRIPT" "$RECURSIVE_SCRIPT" "$MATRIX_SCRIPT"; do
  if [[ ! -e "$p" ]]; then
    echo "error: required helper missing: $p" >&2
    exit 1
  fi
done

for p in "$TARGET_Q6P" "$BASELINE_F16"; do
  if [[ ! -f "$p" ]]; then
    echo "error: required checkpoint missing: $p" >&2
    exit 1
  fi
done

if [[ "$RUN_MATRIX" == "1" && ! -f "$SOURCE_SAFETENSORS" ]]; then
  echo "error: source safetensors missing for matrix: $SOURCE_SAFETENSORS" >&2
  exit 1
fi

if [[ "$HEAD_BYTES" -lt 1 ]]; then
  echo "error: --head-bytes must be >= 1" >&2
  exit 1
fi
if [[ "$PROBE_PROGRESS_EVERY" -lt 0 ]]; then
  echo "error: --probe-progress-every must be >= 0" >&2
  exit 1
fi
if [[ "$PROBE_SMALL_HASH_LIMIT" -lt 0 ]]; then
  echo "error: --probe-small-hash-limit must be >= 0" >&2
  exit 1
fi
if [[ "$PROBE_CHUNK_ROWS" -lt 0 ]]; then
  echo "error: --probe-chunk-rows must be >= 0" >&2
  exit 1
fi
if [[ "$INITIAL_SPLIT_LINES" -lt 1 ]]; then
  echo "error: --initial-split-lines must be >= 1" >&2
  exit 1
fi
if [[ "$CHUNK_SIZE" -lt 1 ]]; then
  echo "error: --chunk-size must be >= 1" >&2
  exit 1
fi
if [[ "$CANARY_NO_TIMEOUT" != "1" && "$CANARY_TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --canary-timeout-sec must be >= 1 unless --canary-no-timeout is used" >&2
  exit 1
fi
if [[ "$MAX_RESPONSES" -lt 1 ]]; then
  echo "error: --max-responses must be >= 1" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/$TAG"
PROBE_LOG="$WORK_DIR/equal_len_payload_probe.log"
PROBE_JSON="$WORK_DIR/equal_len_payload_probe.json"
MISMATCH_NAMES="$WORK_DIR/equal_len_payload_mismatch_names.txt"
RECURSIVE_LOG="$WORK_DIR/recursive_payloadfix.log"
MATRIX_LOG="$WORK_DIR/matrix.log"
SUMMARY_MD="$WORK_DIR/summary.md"

mkdir -p "$WORK_DIR"

echo "== q6p equal-length payloadfix =="
echo "tag=$TAG"
echo "work_dir=$WORK_DIR"
echo "target_q6p=$TARGET_Q6P"
echo "baseline_f16=$BASELINE_F16"
echo "probe_small_hash_limit=$PROBE_SMALL_HASH_LIMIT"
echo "probe_chunk_rows=$PROBE_CHUNK_ROWS"
echo "final_mode=$FINAL_MODE"
echo "canary_no_timeout=$CANARY_NO_TIMEOUT"

declare -a probe_base_cmd=(
  "$PYTHON_BIN" "$PROBE_SCRIPT"
  --file "$TARGET_Q6P"
  --baseline "$BASELINE_F16"
  --head-bytes "$HEAD_BYTES"
  --small-hash-limit "$PROBE_SMALL_HASH_LIMIT"
  --progress-every "$PROBE_PROGRESS_EVERY"
)

if [[ "$ALLOW_METADATA_MISMATCH" == "1" ]]; then
  probe_base_cmd+=(--allow-metadata-mismatch)
fi

if [[ "${#PREFIXES[@]}" -gt 0 ]]; then
  for p in "${PREFIXES[@]}"; do
    probe_base_cmd+=(--prefix "$p")
  done
fi

if [[ "${#EXCLUDE_NAMES[@]}" -gt 0 ]]; then
  for n in "${EXCLUDE_NAMES[@]}"; do
    probe_base_cmd+=(--exclude-name "$n")
  done
fi

echo "== probe equal-length payload mismatches =="
PROBE_RC=0

if [[ "$PROBE_CHUNK_ROWS" -gt 0 ]]; then
  PROBE_CHUNK_DIR="$WORK_DIR/probe_chunks"
  mkdir -p "$PROBE_CHUNK_DIR"
  : > "$PROBE_LOG"

  echo "probe_mode=chunked" | tee -a "$PROBE_LOG"
  echo "probe_chunk_dir=$PROBE_CHUNK_DIR" | tee -a "$PROBE_LOG"

  META_JSON="$PROBE_CHUNK_DIR/meta_probe.json"
  META_NAMES="$PROBE_CHUNK_DIR/meta_probe_names.txt"
  set +e
  env PYTHONUNBUFFERED=1 "${probe_base_cmd[@]}" \
    --start-index 1 \
    --max-rows 1 \
    --out-mismatch-names "$META_NAMES" \
    --out-json "$META_JSON" | tee -a "$PROBE_LOG"
  PROBE_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$PROBE_RC" != "0" ]]; then
    echo "RESULT=FAIL probe_meta_rc=$PROBE_RC" | tee -a "$PROBE_LOG"
    exit 1
  fi

  TOTAL_ROWS="$(env PYTHONUNBUFFERED=1 "$PYTHON_BIN" - <<'PY' "$META_JSON"
import json, sys
payload = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
stats = payload.get('stats', {})
print(int(stats.get('shared_tensors_total_after_filters', 0)))
PY
)"
  TOTAL_ROWS="$(echo "$TOTAL_ROWS" | tr -d '[:space:]')"
  if [[ -z "$TOTAL_ROWS" || "$TOTAL_ROWS" -lt 0 ]]; then
    echo "RESULT=FAIL invalid total rows from meta probe: $TOTAL_ROWS" | tee -a "$PROBE_LOG"
    exit 1
  fi

  echo "probe_total_rows=$TOTAL_ROWS" | tee -a "$PROBE_LOG"

  if [[ "$TOTAL_ROWS" -eq 0 ]]; then
    : > "$MISMATCH_NAMES"
    echo '{"mode":"chunked","total_rows":0,"chunks":[],"stats":{}}' > "$PROBE_JSON"
  else
    start=1
    while [[ "$start" -le "$TOTAL_ROWS" ]]; do
      remaining=$((TOTAL_ROWS - start + 1))
      rows="$PROBE_CHUNK_ROWS"
      if [[ "$rows" -gt "$remaining" ]]; then
        rows="$remaining"
      fi
      end=$((start + rows - 1))

      chunk_json="$PROBE_CHUNK_DIR/report_${start}_${end}.json"
      chunk_names="$PROBE_CHUNK_DIR/mismatch_${start}_${end}.txt"
      echo "probe_chunk=${start}-${end}" | tee -a "$PROBE_LOG"

      set +e
      env PYTHONUNBUFFERED=1 "${probe_base_cmd[@]}" \
        --start-index "$start" \
        --max-rows "$rows" \
        --out-mismatch-names "$chunk_names" \
        --out-json "$chunk_json" | tee -a "$PROBE_LOG"
      chunk_rc=${PIPESTATUS[0]}
      set -e
      if [[ "$chunk_rc" != "0" ]]; then
        echo "RESULT=FAIL probe_chunk_rc=$chunk_rc range=${start}-${end}" | tee -a "$PROBE_LOG"
        exit 1
      fi

      start=$((end + 1))
    done

    cat "$PROBE_CHUNK_DIR"/mismatch_*.txt 2>/dev/null | sed '/^$/d' | sort -u > "$MISMATCH_NAMES"

    env PYTHONUNBUFFERED=1 "$PYTHON_BIN" - <<'PY' "$PROBE_CHUNK_DIR" "$PROBE_JSON" "$TOTAL_ROWS"
import glob
import json
import os
import sys

chunk_dir = sys.argv[1]
out_json = sys.argv[2]
total_rows = int(sys.argv[3])

reports = sorted(glob.glob(os.path.join(chunk_dir, 'report_*.json')))
stats_sum = {}
chunks = []
for path in reports:
    payload = json.load(open(path, 'r', encoding='utf-8'))
    stats = payload.get('stats', {})
    for k, v in stats.items():
        if isinstance(v, int):
            stats_sum[k] = int(stats_sum.get(k, 0)) + int(v)
    chunks.append({'path': path, 'stats': stats})

summary = {
    'mode': 'chunked',
    'total_rows': total_rows,
    'chunk_count': len(reports),
    'stats': stats_sum,
    'chunks': chunks,
}

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY
  fi
else
  set +e
  env PYTHONUNBUFFERED=1 "${probe_base_cmd[@]}" \
    --out-mismatch-names "$MISMATCH_NAMES" \
    --out-json "$PROBE_JSON" | tee "$PROBE_LOG"
  PROBE_RC=${PIPESTATUS[0]}
  set -e
fi

if [[ "$PROBE_RC" != "0" ]]; then
  echo "RESULT=FAIL probe_rc=$PROBE_RC"
  exit 1
fi

MISMATCH_COUNT="$(wc -l < "$MISMATCH_NAMES" | tr -d '[:space:]')"
if [[ -z "$MISMATCH_COUNT" ]]; then
  MISMATCH_COUNT=0
fi

echo "mismatch_names_file=$MISMATCH_NAMES"
echo "mismatch_names_count=$MISMATCH_COUNT"

RECURSIVE_RC=0
if [[ "$MISMATCH_COUNT" -gt 0 ]]; then
  echo "== recursive payload apply =="
  set +e
  declare -a recursive_cmd=(
    bash "$RECURSIVE_SCRIPT"
    --target-q6p "$TARGET_Q6P"
    --baseline-f16 "$BASELINE_F16"
    --names-file "$MISMATCH_NAMES"
    --tag "${TAG}_recursive"
    --initial-split-lines "$INITIAL_SPLIT_LINES"
    --chunk-size "$CHUNK_SIZE"
    --journal-mode "$JOURNAL_MODE"
    --min-free-gb "$MIN_FREE_GB"
    --head-bytes "$HEAD_BYTES"
    --host "$HOST"
    --max-responses "$MAX_RESPONSES"
    --canary-tag "${TAG}_equal_len_canary"
  )
  if [[ "$CANARY_NO_TIMEOUT" == "1" ]]; then
    recursive_cmd+=(--canary-no-timeout)
  else
    recursive_cmd+=(--canary-timeout-sec "$CANARY_TIMEOUT_SEC")
  fi
  if [[ "$FINAL_MODE" == "1" ]]; then
    recursive_cmd+=(--final-mode)
  fi
  "${recursive_cmd[@]}" | tee "$RECURSIVE_LOG"
  RECURSIVE_RC=${PIPESTATUS[0]}
  set -e
else
  echo "skip_recursive=true (mismatch set is empty)"
  echo "skip_recursive=true" > "$RECURSIVE_LOG"
fi

MATRIX_RC=0
if [[ "$RUN_MATRIX" == "1" ]]; then
  echo "== post-fix matrix gate =="
  declare -a matrix_cmd=(
    bash "$MATRIX_SCRIPT"
    --f16-ckpt "$BASELINE_F16"
    --q6p-ckpt "$TARGET_Q6P"
    --source-safetensors "$SOURCE_SAFETENSORS"
    --host "$HOST"
    --tag "${TAG}_matrix"
    --q6p-timeout-sec "$CANARY_TIMEOUT_SEC"
    --q6p-max-responses "$MAX_RESPONSES"
  )
  if [[ "$MATRIX_SKIP_F16_CANARY" == "1" ]]; then
    matrix_cmd+=(--skip-f16-canary)
  fi
  if [[ "$MATRIX_STRICT_ALL" == "1" ]]; then
    matrix_cmd+=(--strict-all)
  fi

  set +e
  "${matrix_cmd[@]}" | tee "$MATRIX_LOG"
  MATRIX_RC=${PIPESTATUS[0]}
  set -e
else
  echo "skip_matrix=true"
  echo "skip_matrix=true" > "$MATRIX_LOG"
fi

FINAL_RC=0
if [[ "$RECURSIVE_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$MATRIX_RC" != "0" ]]; then
  FINAL_RC=1
fi

{
  echo "# Q6P Equal-Length Payloadfix Summary"
  echo
  echo "- tag: $TAG"
  echo "- work_dir: $WORK_DIR"
  echo "- target_q6p: $TARGET_Q6P"
  echo "- baseline_f16: $BASELINE_F16"
  echo "- mismatch_names_count: $MISMATCH_COUNT"
  echo "- probe_rc: $PROBE_RC"
  echo "- recursive_rc: $RECURSIVE_RC"
  echo "- matrix_rc: $MATRIX_RC"
  echo "- final_rc: $FINAL_RC"
  echo
  echo "## Artifacts"
  echo
  echo "- probe_log: $PROBE_LOG"
  echo "- probe_json: $PROBE_JSON"
  echo "- mismatch_names: $MISMATCH_NAMES"
  echo "- recursive_log: $RECURSIVE_LOG"
  echo "- matrix_log: $MATRIX_LOG"
} > "$SUMMARY_MD"

echo "summary=$SUMMARY_MD"
if [[ "$FINAL_RC" == "0" ]]; then
  echo "RESULT=PASS"
else
  echo "RESULT=FAIL"
fi

exit "$FINAL_RC"
