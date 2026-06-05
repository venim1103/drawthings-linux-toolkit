#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTPATCH_SCRIPT="$ROOT/tools/run_clipfix2_postpatch_q6p.sh"

DEFAULT_TARGET="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
DEFAULT_RESTORE_FROM="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.pre_clipfix2_postpatch_20260604_postpatch1.ckpt"
DEFAULT_BASELINE="$ROOT/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt"
DEFAULT_HIGHLIMIT_SQLITE="$ROOT/output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3"
DEFAULT_ROW_NAME="__text_feature_extractor__[t-video_aggregate_embed-0-0]"

TARGET="$DEFAULT_TARGET"
RESTORE_FROM="$DEFAULT_RESTORE_FROM"
BASELINE="$DEFAULT_BASELINE"
HIGHLIMIT_SQLITE="$DEFAULT_HIGHLIMIT_SQLITE"
ROW_NAME="$DEFAULT_ROW_NAME"
TAG="$(date +%Y%m%d_%H%M%S)"

STOP_RUNTIME=1
RUN_POSTPATCH=1
POSTPATCH_SKIP_BACKUP=1
POSTPATCH_SKIP_PROBE=1

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_restore_postpatch_highlimit.sh [options]

This automates the recovery flow used in this repo:
  1) Restore target ckpt from backup/source file.
  2) Run clipfix2 postpatch (expected to fail on one oversized row).
  3) Apply high-limit sqlite one-row patch for the oversized clip tensor.
  4) Run PRAGMA quick_check.

Defaults are prefilled for 10_e_v1 in this workspace.

Options:
      --target <path>            Target ckpt to mutate in place.
                                 Default: dt-models/10_e_v1_bf16_regen_0_q6p.ckpt
      --restore-from <path>      Source file copied into target before patching.
                                 Default: dt-models/10_e_v1_bf16_regen_0_q6p.pre_clipfix2_postpatch_20260604_postpatch1.ckpt
      --baseline <path>          Baseline ckpt used as source for copied rows.
                                 Default: dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt
      --highlimit-sqlite <path>  sqlite3 binary built with higher blob limits.
                                 Default: output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3
      --row-name <name>          Oversized row name for high-limit fallback.
                                 Default: __text_feature_extractor__[t-video_aggregate_embed-0-0]
      --tag <value>              Tag passed to postpatch script.
                                 Default: YYYYMMDD_HHMMSS
      --no-stop-runtime          Do not kill gRPC/test processes before patching.
      --skip-postpatch           Skip clipfix2 postpatch and only do restore + high-limit row patch.
      --postpatch-backup         Allow postpatch script to create backup copy.
                                 Default behavior is --skip-backup for speed.
      --postpatch-probe          Enable postpatch probes.
                                 Default behavior is --skip-probe.
  -h, --help                     Show help.

Examples:
  tools/run_q6p_restore_postpatch_highlimit.sh

  tools/run_q6p_restore_postpatch_highlimit.sh \
    --target dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
    --restore-from dt-models/10_e_v1_bf16_regen_0_q6p.pre_clipfix2_postpatch_20260604_postpatch1.ckpt \
    --baseline dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt \
    --tag repatch_from_restored_manual
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
      --restore-from)
        RESTORE_FROM="${2:-}"
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
      --no-stop-runtime)
        STOP_RUNTIME=0
        shift
        ;;
      --skip-postpatch)
        RUN_POSTPATCH=0
        shift
        ;;
      --postpatch-backup)
        POSTPATCH_SKIP_BACKUP=0
        shift
        ;;
      --postpatch-probe)
        POSTPATCH_SKIP_PROBE=0
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

TARGET="$(abs_path "$TARGET")"
RESTORE_FROM="$(abs_path "$RESTORE_FROM")"
BASELINE="$(abs_path "$BASELINE")"
HIGHLIMIT_SQLITE="$(abs_path "$HIGHLIMIT_SQLITE")"

if [[ ! -f "$RESTORE_FROM" ]]; then
  echo "error: --restore-from file not found: $RESTORE_FROM" >&2
  exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: --baseline file not found: $BASELINE" >&2
  exit 1
fi

if [[ ! -x "$HIGHLIMIT_SQLITE" ]]; then
  echo "error: --highlimit-sqlite is not executable: $HIGHLIMIT_SQLITE" >&2
  exit 1
fi

if [[ "$TARGET" == "$RESTORE_FROM" ]]; then
  echo "error: target and restore source must be different paths" >&2
  exit 1
fi

if [[ "$RUN_POSTPATCH" == "1" ]] && [[ ! -x "$POSTPATCH_SCRIPT" ]]; then
  echo "error: postpatch script not executable: $POSTPATCH_SCRIPT" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "error: sqlite3 command not found on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET")"

if [[ "$STOP_RUNTIME" == "1" ]]; then
  echo "==> Stopping Draw Things/runtime processes"
  pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
  pkill -9 -f 'tools/dt_generate_video.sh|tools/dt_api_client.py .*generate-raw|./test_run.sh' || true
fi

echo "==> Restoring target from source"
if [[ -L "$TARGET" ]]; then
  rm -f "$TARGET"
fi
cp -f "$RESTORE_FROM" "$TARGET"

if [[ -L "$TARGET" ]]; then
  echo "error: target is still a symlink after restore: $TARGET" >&2
  exit 1
fi

echo "==> Removing stale sqlite sidecars"
rm -f "$TARGET-wal" "$TARGET-shm"

echo "==> Pre-check integrity"
PRE_QC="$(sqlite3 "$TARGET" 'PRAGMA quick_check;' | tail -n 1 | tr -d '\r')"
echo "pre_quick_check=$PRE_QC"
if [[ "$PRE_QC" != "ok" ]]; then
  echo "error: pre-quick_check is not ok" >&2
  exit 1
fi

POSTPATCH_EXIT=0
if [[ "$RUN_POSTPATCH" == "1" ]]; then
  echo "==> Running clipfix2 postpatch (may fail on one oversized row; fallback follows)"
  postpatch_cmd=(
    "$POSTPATCH_SCRIPT"
    -f "$TARGET"
    --baseline "$BASELINE"
    --tag "$TAG"
  )

  if [[ "$POSTPATCH_SKIP_BACKUP" == "1" ]]; then
    postpatch_cmd+=(--skip-backup)
  fi
  if [[ "$POSTPATCH_SKIP_PROBE" == "1" ]]; then
    postpatch_cmd+=(--skip-probe)
  fi

  set +e
  "${postpatch_cmd[@]}"
  POSTPATCH_EXIT=$?
  set -e

  if [[ "$POSTPATCH_EXIT" != "0" ]]; then
    echo "postpatch_exit=$POSTPATCH_EXIT (continuing to high-limit fallback row patch)"
  else
    echo "postpatch_exit=0"
  fi
fi

echo "==> Applying high-limit one-row patch"
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

"$HIGHLIMIT_SQLITE" "$TARGET" < "$TMP_SQL"

echo "==> Removing sqlite sidecars after patch"
rm -f "$TARGET-wal" "$TARGET-shm"

echo "==> Final integrity check"
FINAL_QC="$(sqlite3 "$TARGET" 'PRAGMA quick_check;' | tail -n 1 | tr -d '\r')"
echo "final_quick_check=$FINAL_QC"
if [[ "$FINAL_QC" != "ok" ]]; then
  echo "error: final quick_check is not ok" >&2
  exit 1
fi

echo "==> Done"
echo "    target         : $TARGET"
echo "    restore_from   : $RESTORE_FROM"
echo "    baseline       : $BASELINE"
echo "    highlimit_sql  : $HIGHLIMIT_SQLITE"
echo "    oversized_row  : $ROW_NAME"
echo "    postpatch_exit : $POSTPATCH_EXIT"
echo "    quick_check    : $FINAL_QC"
echo "    next: start drawthings-grpc and run ./test_run.sh"
