#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
PROBE_SCRIPT="$ROOT/tools/dt_probe_ckpt_equal_len_payload_mismatch.py"
RECURSIVE_SCRIPT="$ROOT/tools/run_q6p_payloadfix_recursive.sh"

TARGET_Q6P="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
BASELINE_F16="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"

TAG="$(date +%Y%m%d_%H%M%S)_q6p_window_sig_payloadfix"
HOST="127.0.0.1:7861"

HEAD_BYTES=32
MID_BYTES=64
TAIL_BYTES=64
PROBE_SMALL_HASH_LIMIT=0
PROBE_PROGRESS_EVERY=200
PROBE_CHUNK_ROWS=500
ALLOW_METADATA_MISMATCH=0

BATCH_SIZE=24
MAX_BATCHES=1
CONTINUE_ON_BATCH_FAILURE=1
NO_STOP_RUNTIME_AFTER_FIRST=1

INITIAL_SPLIT_LINES=24
CHUNK_SIZE=1
JOURNAL_MODE=delete
MIN_FREE_GB=50
CANARY_TIMEOUT_SEC=120
CANARY_TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
FINAL_MODE=0
CANARY_NO_TIMEOUT=0

RUN_FINAL_ROWWISE_PROBE=0
IGNORE_CANARY_FAILURE=1

declare -a PREFIXES=()
declare -a EXCLUDE_NAMES=()

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_window_sig_payloadfix.sh [options]

Purpose:
  1) Probe equal-length payload mismatches using head + optional mid/tail windows.
  2) Build deterministic mismatch set.
  3) Apply fixes in bounded micro-batches and run canary after each batch.

Options:
  --target-q6p <path>              Target q6p checkpoint.
  --baseline-f16 <path>            Baseline f16 checkpoint.
  --tag <value>                    Output tag (default: timestamped).
  --host <host:port>               Runtime host (default: 127.0.0.1:7861).

  --head-bytes <n>                 Head signature bytes (default: 32).
  --mid-bytes <n>                  Mid-window signature bytes (default: 64).
  --tail-bytes <n>                 Tail-window signature bytes (default: 64).
  --probe-small-hash-limit <n>     Full SHA compare threshold (default: 0).
  --probe-progress-every <n>       Probe progress interval (default: 200).
  --probe-chunk-rows <n>           Probe rows per chunk; 0 means single run (default: 500).
  --prefix <value>                 Probe prefix filter (repeatable; default: __dit__).
  --exclude-name <value>           Explicit probe exclusion (repeatable).
  --allow-metadata-mismatch        Include metadata-mismatch rows in signature compare.

  --batch-size <n>                 Names per remediation batch (default: 24).
  --max-batches <n>                Max batches to execute; 0 means all (default: 1).
  --stop-on-batch-failure          Stop immediately if a batch fails.
  --keep-stop-runtime              Do not set --no-stop-runtime on batches after first.

  --initial-split-lines <n>        Recursive apply initial split size (default: 24).
  --chunk-size <n>                 Recursive apply write chunk size (default: 1).
  --journal-mode <delete|wal|preserve>
                                   SQLite journal mode for apply (default: delete).
  --min-free-gb <n>                Disk floor for apply (default: 50).
  --canary-timeout-sec <n>         Canary timeout (default: 120).
  --final-mode                     Use longer final validation timeout (900s)
                                   unless --canary-timeout-sec is explicitly set.
  --canary-no-timeout              Disable canary timeout wrapper.
  --max-responses <n>              Canary max responses (default: 10).

  --strict-canary                  Do not pass --ignore-canary-failure to recursive runs.
  --run-final-rowwise-probe        Run row-wise probe on final batch only.

  -h, --help                       Show help.

Outputs:
  output/<tag>/
    - window_sig_probe.log
    - window_sig_probe.json
    - window_sig_mismatch_names.txt
    - batches/names_batch_*.txt
    - batch_XXX_recursive.log
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
      --mid-bytes)
        MID_BYTES="${2:-}"
        shift 2
        ;;
      --tail-bytes)
        TAIL_BYTES="${2:-}"
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
      --batch-size)
        BATCH_SIZE="${2:-}"
        shift 2
        ;;
      --max-batches)
        MAX_BATCHES="${2:-}"
        shift 2
        ;;
      --stop-on-batch-failure)
        CONTINUE_ON_BATCH_FAILURE=0
        shift
        ;;
      --keep-stop-runtime)
        NO_STOP_RUNTIME_AFTER_FIRST=0
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
      --strict-canary)
        IGNORE_CANARY_FAILURE=0
        shift
        ;;
      --run-final-rowwise-probe)
        RUN_FINAL_ROWWISE_PROBE=1
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

for p in "$PYTHON_BIN" "$PROBE_SCRIPT" "$RECURSIVE_SCRIPT"; do
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

if [[ "$HEAD_BYTES" -lt 1 ]]; then
  echo "error: --head-bytes must be >= 1" >&2
  exit 1
fi
if [[ "$MID_BYTES" -lt 0 ]]; then
  echo "error: --mid-bytes must be >= 0" >&2
  exit 1
fi
if [[ "$TAIL_BYTES" -lt 0 ]]; then
  echo "error: --tail-bytes must be >= 0" >&2
  exit 1
fi
if [[ "$PROBE_SMALL_HASH_LIMIT" -lt 0 ]]; then
  echo "error: --probe-small-hash-limit must be >= 0" >&2
  exit 1
fi
if [[ "$PROBE_PROGRESS_EVERY" -lt 0 ]]; then
  echo "error: --probe-progress-every must be >= 0" >&2
  exit 1
fi
if [[ "$PROBE_CHUNK_ROWS" -lt 0 ]]; then
  echo "error: --probe-chunk-rows must be >= 0" >&2
  exit 1
fi
if [[ "$BATCH_SIZE" -lt 1 ]]; then
  echo "error: --batch-size must be >= 1" >&2
  exit 1
fi
if [[ "$MAX_BATCHES" -lt 0 ]]; then
  echo "error: --max-batches must be >= 0" >&2
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
PROBE_LOG="$WORK_DIR/window_sig_probe.log"
PROBE_JSON="$WORK_DIR/window_sig_probe.json"
MISMATCH_NAMES="$WORK_DIR/window_sig_mismatch_names.txt"
BATCH_DIR="$WORK_DIR/batches"
SUMMARY_MD="$WORK_DIR/summary.md"

mkdir -p "$WORK_DIR" "$BATCH_DIR"

echo "== q6p window-signature payloadfix =="
echo "tag=$TAG"
echo "work_dir=$WORK_DIR"
echo "target_q6p=$TARGET_Q6P"
echo "baseline_f16=$BASELINE_F16"
echo "head_bytes=$HEAD_BYTES"
echo "mid_bytes=$MID_BYTES"
echo "tail_bytes=$TAIL_BYTES"
echo "probe_small_hash_limit=$PROBE_SMALL_HASH_LIMIT"
echo "probe_chunk_rows=$PROBE_CHUNK_ROWS"
echo "batch_size=$BATCH_SIZE"
echo "max_batches=$MAX_BATCHES"
echo "final_mode=$FINAL_MODE"
echo "canary_no_timeout=$CANARY_NO_TIMEOUT"

declare -a probe_base_cmd=(
  "$PYTHON_BIN" "$PROBE_SCRIPT"
  --file "$TARGET_Q6P"
  --baseline "$BASELINE_F16"
  --head-bytes "$HEAD_BYTES"
  --mid-bytes "$MID_BYTES"
  --tail-bytes "$TAIL_BYTES"
  --small-hash-limit "$PROBE_SMALL_HASH_LIMIT"
  --progress-every "$PROBE_PROGRESS_EVERY"
)

if [[ "$ALLOW_METADATA_MISMATCH" == "1" ]]; then
  probe_base_cmd+=(--allow-metadata-mismatch)
fi
if [[ "${#PREFIXES[@]}" -eq 0 ]]; then
  PREFIXES=("__dit__")
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

PROBE_RC=0

echo "== probe window-signature payload mismatches =="
if [[ "$PROBE_CHUNK_ROWS" -gt 0 ]]; then
  PROBE_CHUNK_DIR="$WORK_DIR/probe_chunks"
  mkdir -p "$PROBE_CHUNK_DIR"
  : > "$PROBE_LOG"

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
import json
import sys
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

if [[ "$MISMATCH_COUNT" -eq 0 ]]; then
  {
    echo "# Q6P Window-Signature Payloadfix Summary"
    echo
    echo "- tag: $TAG"
    echo "- work_dir: $WORK_DIR"
    echo "- mismatch_names_count: 0"
    echo "- probe_rc: 0"
    echo "- batches_executed: 0"
    echo "- final_rc: 0"
  } > "$SUMMARY_MD"
  echo "summary=$SUMMARY_MD"
  echo "RESULT=PASS"
  exit 0
fi

split -d -a 4 -l "$BATCH_SIZE" "$MISMATCH_NAMES" "$BATCH_DIR/names_batch_"
mapfile -t batch_files < <(find "$BATCH_DIR" -maxdepth 1 -type f -name 'names_batch_*' | sort)

if [[ "${#batch_files[@]}" -eq 0 ]]; then
  echo "RESULT=FAIL no batch files created" >&2
  exit 1
fi

BATCH_TOTAL="${#batch_files[@]}"
BATCH_LIMIT="$BATCH_TOTAL"
if [[ "$MAX_BATCHES" -gt 0 && "$MAX_BATCHES" -lt "$BATCH_TOTAL" ]]; then
  BATCH_LIMIT="$MAX_BATCHES"
fi

FINAL_RC=0
BATCH_EXECUTED=0
BATCH_STATUS_TSV="$WORK_DIR/batch_status.tsv"
: > "$BATCH_STATUS_TSV"

echo -e "batch_index\tbatch_file\tnames_count\trc\tlog" >> "$BATCH_STATUS_TSV"

for ((i=1; i<=BATCH_LIMIT; i++)); do
  batch_file="${batch_files[$((i-1))]}"
  batch_names_count="$(wc -l < "$batch_file" | tr -d '[:space:]')"
  if [[ -z "$batch_names_count" || "$batch_names_count" -eq 0 ]]; then
    continue
  fi

  batch_tag="${TAG}_batch_$(printf '%03d' "$i")"
  batch_log="$WORK_DIR/batch_$(printf '%03d' "$i")_recursive.log"

  declare -a batch_cmd=(
    bash "$RECURSIVE_SCRIPT"
    --target-q6p "$TARGET_Q6P"
    --baseline-f16 "$BASELINE_F16"
    --names-file "$batch_file"
    --tag "$batch_tag"
    --initial-split-lines "$INITIAL_SPLIT_LINES"
    --chunk-size "$CHUNK_SIZE"
    --journal-mode "$JOURNAL_MODE"
    --min-free-gb "$MIN_FREE_GB"
    --head-bytes "$HEAD_BYTES"
    --host "$HOST"
    --canary-timeout-sec "$CANARY_TIMEOUT_SEC"
    --max-responses "$MAX_RESPONSES"
    --canary-tag "${batch_tag}_canary"
    --skip-probe
  )

  if [[ "$IGNORE_CANARY_FAILURE" == "1" ]]; then
    batch_cmd+=(--ignore-canary-failure)
  fi

  if [[ "$FINAL_MODE" == "1" ]]; then
    batch_cmd+=(--final-mode)
  fi

  if [[ "$CANARY_NO_TIMEOUT" == "1" ]]; then
    batch_cmd+=(--canary-no-timeout)
  fi

  if [[ "$NO_STOP_RUNTIME_AFTER_FIRST" == "1" && "$i" -gt 1 ]]; then
    batch_cmd+=(--no-stop-runtime)
  fi

  if [[ "$RUN_FINAL_ROWWISE_PROBE" == "1" && "$i" -eq "$BATCH_LIMIT" ]]; then
    # Keep probe on the last batch if requested.
    for j in "${!batch_cmd[@]}"; do
      if [[ "${batch_cmd[$j]}" == "--skip-probe" ]]; then
        unset 'batch_cmd[j]'
        break
      fi
    done
    batch_cmd=("${batch_cmd[@]}")
  fi

  set +e
  "${batch_cmd[@]}" | tee "$batch_log"
  batch_rc=${PIPESTATUS[0]}
  set -e

  echo -e "${i}\t${batch_file}\t${batch_names_count}\t${batch_rc}\t${batch_log}" >> "$BATCH_STATUS_TSV"

  BATCH_EXECUTED=$((BATCH_EXECUTED + 1))

  if [[ "$batch_rc" != "0" ]]; then
    FINAL_RC=1
    if [[ "$CONTINUE_ON_BATCH_FAILURE" == "0" ]]; then
      break
    fi
  fi
done

{
  echo "# Q6P Window-Signature Payloadfix Summary"
  echo
  echo "- tag: $TAG"
  echo "- work_dir: $WORK_DIR"
  echo "- target_q6p: $TARGET_Q6P"
  echo "- baseline_f16: $BASELINE_F16"
  echo "- mismatch_names_count: $MISMATCH_COUNT"
  echo "- batch_total_available: $BATCH_TOTAL"
  echo "- batch_executed: $BATCH_EXECUTED"
  echo "- batch_limit: $BATCH_LIMIT"
  echo "- final_rc: $FINAL_RC"
  echo
  echo "## Artifacts"
  echo
  echo "- probe_log: $PROBE_LOG"
  echo "- probe_json: $PROBE_JSON"
  echo "- mismatch_names: $MISMATCH_NAMES"
  echo "- batch_status: $BATCH_STATUS_TSV"
} > "$SUMMARY_MD"

echo "summary=$SUMMARY_MD"
if [[ "$FINAL_RC" == "0" ]]; then
  echo "RESULT=PASS"
else
  echo "RESULT=FAIL"
fi

exit "$FINAL_RC"
