#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

TARGET="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
BASELINE="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
HIGHLIMIT_SQLITE="$ROOT/output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3"
ROWS_FILE="$ROOT/output/run013_text_feature_family_names.txt"
TAG="$(date +%Y%m%d_%H%M%S)_highlimit_rows_patch"

STOP_RUNTIME=1
RUN_CANARY=1
IGNORE_CANARY_FAILURE=0
FINAL_MODE=0
CANARY_NO_TIMEOUT=0
CANARY_TIMEOUT_SEC=120
CANARY_TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
HOST="127.0.0.1:7861"

declare -a ROW_NAMES=()

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_highlimit_rows_patch_canary.sh [options]

Purpose:
  1) Patch a set of tensor rows in target ckpt using high-limit sqlite.
  2) Verify quick_check before/after patch.
  3) Optionally run bounded canary.

Options:
  --target <path>                Target checkpoint.
  --baseline <path>              Baseline checkpoint used as source.
  --highlimit-sqlite <path>      sqlite3 binary with higher blob limits.
  --rows-file <path>             Rows file (one tensor name per line).
  --row-name <name>              Additional row name (repeatable).
  --tag <value>                  Output tag.
  --host <host:port>             Canary host (default: 127.0.0.1:7861).
  --canary-timeout-sec <n>       Canary timeout seconds (default: 120).
  --final-mode                   Use longer final validation timeout (900s)
                                 unless --canary-timeout-sec is explicitly set.
  --canary-no-timeout            Disable canary timeout wrapper.
  --max-responses <n>            Canary max responses.

  --no-stop-runtime              Do not stop runtime processes before patching.
  --skip-canary                  Skip canary run.
  --ignore-canary-failure        Keep script exit status successful on canary failure.
  -h, --help                     Show help.
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

sql_escape_single_quotes() {
  printf "%s" "$1" | sed "s/'/''/g"
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target)
        TARGET="${2:-}"
        shift 2
        ;;
      --baseline)
        BASELINE="${2:-}"
        shift 2
        ;;
      --highlimit-sqlite)
        HIGHLIMIT_SQLITE="${2:-}"
        shift 2
        ;;
      --rows-file)
        ROWS_FILE="${2:-}"
        shift 2
        ;;
      --row-name)
        ROW_NAMES+=("${2:-}")
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
      --no-stop-runtime)
        STOP_RUNTIME=0
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

TARGET="$(abs_path "$TARGET")"
BASELINE="$(abs_path "$BASELINE")"
HIGHLIMIT_SQLITE="$(abs_path "$HIGHLIMIT_SQLITE")"
ROWS_FILE="$(abs_path "$ROWS_FILE")"

if [[ ! -f "$TARGET" ]]; then
  echo "error: target missing: $TARGET" >&2
  exit 1
fi
if [[ ! -f "$BASELINE" ]]; then
  echo "error: baseline missing: $BASELINE" >&2
  exit 1
fi
if [[ ! -x "$HIGHLIMIT_SQLITE" ]]; then
  echo "error: high-limit sqlite not executable: $HIGHLIMIT_SQLITE" >&2
  exit 1
fi
if [[ ! -x "$CANARY_SCRIPT" ]]; then
  echo "error: canary script missing or not executable: $CANARY_SCRIPT" >&2
  exit 1
fi
if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "error: sqlite3 command not found" >&2
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
PATCH_SQL="$WORK_DIR/patch_rows.sql"
PATCH_LOG="$WORK_DIR/highlimit_patch.log"
NAMES_NORM="$WORK_DIR/rows_norm.txt"
mkdir -p "$WORK_DIR"

{
  if [[ -f "$ROWS_FILE" ]]; then
    cat "$ROWS_FILE"
  fi
  if [[ "${#ROW_NAMES[@]}" -gt 0 ]]; then
    for n in "${ROW_NAMES[@]}"; do
      printf "%s\n" "$n"
    done
  fi
} | sed '/^\s*#/d; /^\s*$/d' | sort -u > "$NAMES_NORM"

NAMES_COUNT="$(wc -l < "$NAMES_NORM" | tr -d '[:space:]')"
if [[ -z "$NAMES_COUNT" || "$NAMES_COUNT" == "0" ]]; then
  echo "error: no row names provided or found" >&2
  exit 1
fi

echo "== q6p high-limit rows patch canary =="
echo "target=$TARGET"
echo "baseline=$BASELINE"
echo "highlimit_sqlite=$HIGHLIMIT_SQLITE"
echo "rows_file=$ROWS_FILE"
echo "rows_count=$NAMES_COUNT"
echo "tag=$TAG"
echo "work_dir=$WORK_DIR"
echo "final_mode=$FINAL_MODE"
echo "canary_no_timeout=$CANARY_NO_TIMEOUT"

if [[ "$STOP_RUNTIME" == "1" ]]; then
  echo "== stop runtime processes =="
  pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
  pkill -9 -f 'tools/dt_generate_video.sh|tools/dt_api_client.py .*generate-raw|./test_run.sh' || true
fi

echo "== remove sqlite sidecars =="
rm -f "$TARGET-wal" "$TARGET-shm"

echo "== pre quick_check =="
PRE_QC="$(sqlite3 "$TARGET" 'PRAGMA quick_check;' | tail -n 1 | tr -d '\r')"
echo "pre_quick_check=$PRE_QC"
if [[ "$PRE_QC" != "ok" ]]; then
  echo "RESULT=FAIL pre quick_check not ok"
  exit 1
fi

BASELINE_SQL="$(sql_escape_single_quotes "$BASELINE")"
{
  echo ".timeout 15000"
  echo "PRAGMA journal_mode=WAL;"
  echo "ATTACH DATABASE '$BASELINE_SQL' AS base;"
  echo "BEGIN IMMEDIATE;"
  while IFS= read -r row_name; do
    esc_name="$(sql_escape_single_quotes "$row_name")"
    echo "UPDATE tensors"
    echo "SET dim=(SELECT dim FROM base.tensors WHERE name='$esc_name'),"
    echo "    data=(SELECT data FROM base.tensors WHERE name='$esc_name'),"
    echo "    type=(SELECT type FROM base.tensors WHERE name='$esc_name'),"
    echo "    format=(SELECT format FROM base.tensors WHERE name='$esc_name'),"
    echo "    datatype=(SELECT datatype FROM base.tensors WHERE name='$esc_name')"
    echo "WHERE name='$esc_name';"
    echo "SELECT 'row_patch_changes|$esc_name|' || changes();"
  done < "$NAMES_NORM"
  echo "COMMIT;"
  echo "DETACH DATABASE base;"
  echo "PRAGMA wal_checkpoint(TRUNCATE);"
} > "$PATCH_SQL"

echo "== apply high-limit rows patch =="
"$HIGHLIMIT_SQLITE" "$TARGET" < "$PATCH_SQL" | tee "$PATCH_LOG"

echo "== remove sqlite sidecars after patch =="
rm -f "$TARGET-wal" "$TARGET-shm"

echo "== post quick_check =="
POST_QC="$(sqlite3 "$TARGET" 'PRAGMA quick_check;' | tail -n 1 | tr -d '\r')"
echo "post_quick_check=$POST_QC"
if [[ "$POST_QC" != "ok" ]]; then
  echo "RESULT=FAIL post quick_check not ok"
  exit 1
fi

CANARY_RC=0
CANARY_TAG="${TAG}_canary"
if [[ "$RUN_CANARY" == "1" ]]; then
  echo "== run canary =="
  declare -a canary_cmd=(
    bash "$CANARY_SCRIPT"
    --model 10_e_v1_bf16_regen_0_q6p.ckpt
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
fi

SERVER_LOG="$ROOT/output/q6p_canary_${CANARY_TAG}/server.log"
if [[ -f "$SERVER_LOG" ]] && rg -q "Program crashed|Signal 11|ccv_nnc_tensor_read" "$SERVER_LOG"; then
  echo "crash_detected=1"
else
  echo "crash_detected=0"
fi

echo "rows_norm=$NAMES_NORM"
echo "patch_sql=$PATCH_SQL"
echo "patch_log=$PATCH_LOG"

if [[ "$RUN_CANARY" == "1" && "$CANARY_RC" != "0" && "$IGNORE_CANARY_FAILURE" != "1" ]]; then
  echo "RESULT=FAIL"
  exit 1
fi

echo "RESULT=PASS"
exit 0
