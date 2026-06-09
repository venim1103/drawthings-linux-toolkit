#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

TARGET="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
BASELINE="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
HIGHLIMIT_SQLITE="$ROOT/output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3"
ROW_NAME="__text_feature_extractor__[t-video_aggregate_embed-0-0]"
TAG="$(date +%Y%m%d_%H%M%S)_highlimit_rowpatch"

STOP_RUNTIME=1
RUN_CANARY=1
IGNORE_CANARY_FAILURE=0

HOST="127.0.0.1:7861"
CANARY_TIMEOUT_SEC=120
CANARY_TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
FINAL_MODE=0
CANARY_NO_TIMEOUT=0

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_highlimit_row_patch_canary.sh [options]

Purpose:
  1) Patch one oversized/unreadable tensor row in target ckpt using high-limit sqlite.
  2) Verify quick_check.
  3) Optionally run bounded q6p canary.

Options:
  --target <path>                Target checkpoint (default: 10_e_v1 q6p).
  --baseline <path>              Baseline checkpoint copy source (default: 10_e_v1 f16).
  --highlimit-sqlite <path>      sqlite3 binary with higher blob limits.
  --row-name <name>              Tensor row name to patch.
  --tag <value>                  Output tag (default: timestamped).
  --host <host:port>             Canary host (default: 127.0.0.1:7861).
  --canary-timeout-sec <n>       Canary timeout seconds (default: 120).
  --final-mode                   Use longer final validation timeout (900s)
                                 unless --canary-timeout-sec is explicitly set.
  --canary-no-timeout            Disable canary timeout wrapper.
  --max-responses <n>            Canary max responses (default: 10).

  --no-stop-runtime              Do not kill runtime processes before patching.
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
      --row-name)
        ROW_NAME="${2:-}"
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

echo "== q6p high-limit row patch canary =="
echo "target=$TARGET"
echo "baseline=$BASELINE"
echo "highlimit_sqlite=$HIGHLIMIT_SQLITE"
echo "row_name=$ROW_NAME"
echo "tag=$TAG"
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
ROW_SQL="$(sql_escape_single_quotes "$ROW_NAME")"
TMP_SQL="$(mktemp)"
trap 'rm -f "$TMP_SQL"' EXIT

cat > "$TMP_SQL" <<SQL
.timeout 15000
PRAGMA journal_mode=WAL;
ATTACH DATABASE '$BASELINE_SQL' AS base;
BEGIN IMMEDIATE;
UPDATE tensors
SET dim=(SELECT dim FROM base.tensors WHERE name='$ROW_SQL'),
    data=(SELECT data FROM base.tensors WHERE name='$ROW_SQL'),
    type=(SELECT type FROM base.tensors WHERE name='$ROW_SQL'),
    format=(SELECT format FROM base.tensors WHERE name='$ROW_SQL'),
    datatype=(SELECT datatype FROM base.tensors WHERE name='$ROW_SQL')
WHERE name='$ROW_SQL';
SELECT 'row_patch_changes=' || changes();
COMMIT;
DETACH DATABASE base;
PRAGMA wal_checkpoint(TRUNCATE);
SQL

echo "== apply high-limit row patch =="
"$HIGHLIMIT_SQLITE" "$TARGET" < "$TMP_SQL"

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
  echo "== run bounded canary =="
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

if [[ "$RUN_CANARY" == "1" && "$CANARY_RC" != "0" && "$IGNORE_CANARY_FAILURE" != "1" ]]; then
  echo "RESULT=FAIL"
  exit 1
fi

echo "RESULT=PASS"
exit 0
