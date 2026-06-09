#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"

ALIGN_SUBSET="$ROOT/tools/dt_align_ckpt_content_subset.py"
ROWWISE_PROBE="$ROOT/tools/dt_probe_ckpt_meta_len_rowwise.py"
CANARY_ONCE="$ROOT/tools/run_q6p_canary_once.sh"

TARGET_Q6P="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
BASELINE_F16="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
NAMES_FILE="$ROOT/tools/patch_sets/10_e_v1_q6p_post_metachunkfix_mismatch_meta_len_all5746_20260608.txt"

TAG="$(date +%Y%m%d_%H%M%S)"
INITIAL_SPLIT_LINES=200
CHUNK_SIZE=4
JOURNAL_MODE=delete
MIN_FREE_GB=50
HEAD_BYTES=32

STOP_RUNTIME=1
RETRY_FAILED_LEAVES=1
RUN_PROBE=1
RUN_CANARY=1
IGNORE_CANARY_FAILURE=0

HOST="127.0.0.1:7861"
CANARY_TIMEOUT_SEC=120
CANARY_TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
CANARY_TAG=""
PROBE_PROGRESS_EVERY=400
FINAL_MODE=0
CANARY_NO_TIMEOUT=0

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_payloadfix_recursive.sh [options]

This script automates the Run-006 style payload-fix workflow:
  1) Split names into micro-batches.
  2) Recursively split-on-failure until leaf windows.
  3) Optionally retry failed leaves in one-name mode.
  4) Run row-wise meta/len verification.
  5) Run bounded canary.

Options:
  --target-q6p <path>              Target q6p checkpoint to patch.
  --baseline-f16 <path>            Baseline f16 checkpoint used as copy source.
  --names-file <path>              Tensor names file (one name per line).
  --tag <value>                    Output tag (default: timestamp).
  --initial-split-lines <n>        Initial split size (default: 200).
  --chunk-size <n>                 dt_align apply chunk size (default: 4).
  --journal-mode <delete|wal|preserve>
                                   SQLite journal mode for dt_align (default: delete).
  --min-free-gb <n>                Disk floor for dt_align (default: 50).
  --head-bytes <n>                 Head bytes used by dt_align checks (default: 32).
  --no-stop-runtime                Do not stop Draw Things/runtime processes.
  --no-retry-failed-leaves         Skip one-name retry pass for failed leaves.
  --skip-probe                     Skip row-wise probe after remediation.
  --skip-canary                    Skip bounded canary after remediation.
  --ignore-canary-failure          Do not fail overall script on canary failure.
  --probe-progress-every <n>       Probe progress interval (default: 400).
  --host <host:port>               Canary host (default: 127.0.0.1:7861).
  --canary-timeout-sec <n>         Canary timeout seconds (default: 120).
  --final-mode                     Use longer final validation timeout (900s)
                                   unless --canary-timeout-sec is explicitly set.
  --canary-no-timeout              Disable canary timeout wrapper.
  --max-responses <n>              Canary max streamed responses (default: 10).
  --canary-tag <value>             Canary output tag (default: <tag>_post_leafretry).
  -h, --help                       Show help.

Examples:
  tools/run_q6p_payloadfix_recursive.sh \
    --names-file tools/patch_sets/10_e_v1_q6p_post_metachunkfix_mismatch_meta_len_all5746_20260608.txt \
    --tag 20260608_run006_payloadfix

  tools/run_q6p_payloadfix_recursive.sh \
    --skip-canary \
    --skip-probe
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
      --names-file)
        NAMES_FILE="${2:-}"
        shift 2
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
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
      --head-bytes)
        HEAD_BYTES="${2:-}"
        shift 2
        ;;
      --no-stop-runtime)
        STOP_RUNTIME=0
        shift
        ;;
      --no-retry-failed-leaves)
        RETRY_FAILED_LEAVES=0
        shift
        ;;
      --skip-probe)
        RUN_PROBE=0
        shift
        ;;
      --skip-canary)
        RUN_CANARY=0
        shift
        ;;
      --ignore-canary-failure)
        IGNORE_CANARY_FAILURE=1
        shift
        ;;
      --probe-progress-every)
        PROBE_PROGRESS_EVERY="${2:-}"
        shift 2
        ;;
      --host)
        HOST="${2:-}"
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
      --canary-tag)
        CANARY_TAG="${2:-}"
        shift 2
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
NAMES_FILE="$(abs_path "$NAMES_FILE")"

if [[ -z "$CANARY_TAG" ]]; then
  CANARY_TAG="${TAG}_post_leafretry"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: missing python env at $PYTHON_BIN" >&2
  exit 1
fi

for p in "$ALIGN_SUBSET" "$ROWWISE_PROBE" "$CANARY_ONCE"; do
  if [[ ! -f "$p" ]]; then
    echo "error: missing helper script: $p" >&2
    exit 1
  fi
done

for p in "$TARGET_Q6P" "$BASELINE_F16" "$NAMES_FILE"; do
  if [[ ! -f "$p" ]]; then
    echo "error: required file missing: $p" >&2
    exit 1
  fi
done

if [[ "$INITIAL_SPLIT_LINES" -lt 1 ]]; then
  echo "error: --initial-split-lines must be >= 1" >&2
  exit 1
fi
if [[ "$CHUNK_SIZE" -lt 1 ]]; then
  echo "error: --chunk-size must be >= 1" >&2
  exit 1
fi
if [[ "$HEAD_BYTES" -lt 1 ]]; then
  echo "error: --head-bytes must be >= 1" >&2
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

NAMES_COUNT="$(wc -l < "$NAMES_FILE" | tr -d '[:space:]')"
if [[ -z "$NAMES_COUNT" || "$NAMES_COUNT" == "0" ]]; then
  echo "error: names file has zero entries: $NAMES_FILE" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/${TAG}"
CHUNK_DIR="$WORK_DIR/chunks"
FAILED_FILE="$WORK_DIR/failed_leaf_names.txt"
PROBE_JSON="$WORK_DIR/probe_meta_len_rowwise_after_leafretry.json"
PROBE_TSV="$WORK_DIR/probe_meta_len_rowwise_after_leafretry.tsv"
PROBE_MISMATCH_NAMES="$WORK_DIR/probe_mismatch_names_after_leafretry.txt"

mkdir -p "$CHUNK_DIR"
: > "$FAILED_FILE"

echo "== recursive payload fix =="
echo "target_q6p=$TARGET_Q6P"
echo "baseline_f16=$BASELINE_F16"
echo "names_file=$NAMES_FILE"
echo "names_count=$NAMES_COUNT"
echo "work_dir=$WORK_DIR"
echo "initial_split_lines=$INITIAL_SPLIT_LINES"
echo "chunk_size=$CHUNK_SIZE"
echo "journal_mode=$JOURNAL_MODE"
echo "min_free_gb=$MIN_FREE_GB"
echo "head_bytes=$HEAD_BYTES"
echo "run_probe=$RUN_PROBE"
echo "run_canary=$RUN_CANARY"
echo "final_mode=$FINAL_MODE"
echo "canary_no_timeout=$CANARY_NO_TIMEOUT"

if [[ "$STOP_RUNTIME" == "1" ]]; then
  echo "== stop runtime processes =="
  pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
  pkill -9 -f 'tools/dt_api_client.py .*generate-raw' || true
fi

split -d -l "$INITIAL_SPLIT_LINES" "$NAMES_FILE" "$CHUNK_DIR/init_"

apply_file() {
  local file="$1"
  local n
  n="$(wc -l < "$file" | tr -d '[:space:]')"
  if [[ -z "$n" || "$n" == "0" ]]; then
    return 0
  fi

  echo "apply_file file=$file n=$n"
  set +e
  "$PYTHON_BIN" "$ALIGN_SUBSET" \
    --file "$TARGET_Q6P" \
    --baseline "$BASELINE_F16" \
    --data-names-file "$file" \
    --mode apply \
    --chunk-size "$CHUNK_SIZE" \
    --journal-mode "$JOURNAL_MODE" \
    --min-free-gb "$MIN_FREE_GB" \
    --head-bytes "$HEAD_BYTES"
  local rc=$?
  set -e

  if [[ "$rc" == "0" ]]; then
    echo "apply_pass file=$file n=$n"
    return 0
  fi

  echo "apply_fail rc=$rc file=$file n=$n"
  if [[ "$n" -le 1 ]]; then
    cat "$file" >> "$FAILED_FILE"
    echo "leaf_fail file=$file"
    return 0
  fi

  local half
  half=$(( (n + 1) / 2 ))
  local split_prefix
  split_prefix="${file}.split."
  split -d -l "$half" "$file" "$split_prefix"

  local sub
  for sub in "${split_prefix}"*; do
    apply_file "$sub"
  done
}

for f in "$CHUNK_DIR"/init_*; do
  [[ -e "$f" ]] || continue
  apply_file "$f"
done

sort -u "$FAILED_FILE" -o "$FAILED_FILE"
FAILED_COUNT_BEFORE_RETRY="$(wc -l < "$FAILED_FILE" | tr -d '[:space:]')"
echo "failed_leaf_count_before_retry=$FAILED_COUNT_BEFORE_RETRY"
echo "failed_leaf_file=$FAILED_FILE"

RETRY_RC=0
if [[ "$RETRY_FAILED_LEAVES" == "1" && "$FAILED_COUNT_BEFORE_RETRY" -gt 0 ]]; then
  echo "== retry failed leaves in one-name mode =="
  set +e
  "$PYTHON_BIN" "$ALIGN_SUBSET" \
    --file "$TARGET_Q6P" \
    --baseline "$BASELINE_F16" \
    --data-names-file "$FAILED_FILE" \
    --mode apply \
    --chunk-size 1 \
    --journal-mode "$JOURNAL_MODE" \
    --min-free-gb "$MIN_FREE_GB" \
    --head-bytes "$HEAD_BYTES"
  RETRY_RC=$?
  set -e
  echo "failed_leaf_retry_rc=$RETRY_RC"

  if [[ "$RETRY_RC" == "0" ]]; then
    echo "== reconcile failed leaf file with dry-run =="
    set +e
    DRY_RUN_OUTPUT="$("$PYTHON_BIN" "$ALIGN_SUBSET" \
      --file "$TARGET_Q6P" \
      --baseline "$BASELINE_F16" \
      --data-names-file "$FAILED_FILE" \
      --mode dry-run \
      --head-bytes "$HEAD_BYTES" 2>&1)"
    DRY_RUN_RC=$?
    set -e
    printf "%s\n" "$DRY_RUN_OUTPUT"
    if [[ "$DRY_RUN_RC" == "0" ]]; then
      PRE_DATA_MISMATCH="$(printf "%s\n" "$DRY_RUN_OUTPUT" | awk -F= '/^pre_data_head_mismatch=/{print $2}' | tail -n 1 | tr -d '[:space:]')"
      if [[ "$PRE_DATA_MISMATCH" == "0" ]]; then
        : > "$FAILED_FILE"
      fi
    fi
  fi
fi

FAILED_COUNT_AFTER_RETRY="$(wc -l < "$FAILED_FILE" | tr -d '[:space:]')"
echo "failed_leaf_count_after_retry=$FAILED_COUNT_AFTER_RETRY"

PROBE_RC=0
if [[ "$RUN_PROBE" == "1" ]]; then
  echo "== row-wise probe =="
  set +e
  "$PYTHON_BIN" "$ROWWISE_PROBE" \
    --file "$TARGET_Q6P" \
    --baseline "$BASELINE_F16" \
    --progress-every "$PROBE_PROGRESS_EVERY" \
    --out-json "$PROBE_JSON" \
    --out-tsv "$PROBE_TSV" \
    --out-mismatch-names "$PROBE_MISMATCH_NAMES"
  PROBE_RC=$?
  set -e
  echo "probe_rc=$PROBE_RC"
  echo "probe_json=$PROBE_JSON"
  echo "probe_tsv=$PROBE_TSV"
  echo "probe_mismatch_names=$PROBE_MISMATCH_NAMES"
fi

CANARY_RC=0
if [[ "$RUN_CANARY" == "1" ]]; then
  echo "== bounded canary =="
  declare -a canary_cmd=(
    bash "$CANARY_ONCE"
    --model "$(basename "$TARGET_Q6P")"
    --host "$HOST"
    --max-responses "$MAX_RESPONSES"
    --tag "$CANARY_TAG"
  )
  if [[ "$CANARY_NO_TIMEOUT" == "1" ]]; then
    canary_cmd+=(--no-timeout)
  else
    canary_cmd+=(--timeout-sec "$CANARY_TIMEOUT_SEC")
  fi
  if [[ "$FINAL_MODE" == "1" ]]; then
    canary_cmd+=(--final-mode)
  fi
  set +e
  "${canary_cmd[@]}"
  CANARY_RC=$?
  set -e
  echo "canary_rc=$CANARY_RC"
  echo "canary_tag=$CANARY_TAG"
fi

FINAL_RC=0
if [[ "$FAILED_COUNT_AFTER_RETRY" -gt 0 ]]; then
  FINAL_RC=1
fi
if [[ "$RETRY_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$PROBE_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$CANARY_RC" != "0" && "$IGNORE_CANARY_FAILURE" != "1" ]]; then
  FINAL_RC=1
fi

if [[ "$FINAL_RC" == "0" ]]; then
  echo "RESULT=PASS"
else
  echo "RESULT=FAIL"
fi

exit "$FINAL_RC"
