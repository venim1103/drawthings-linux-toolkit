#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
DEFAULT_BASELINE="$ROOT/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt"

TARGET=""
BASELINE="$DEFAULT_BASELINE"
BACKUP=""
TAG="$(date +%Y%m%d_%H%M%S)"
SKIP_BACKUP=0
SKIP_PROBE=0
DRY_RUN=0
ALLOW_SYMLINK_TARGET=0
CHUNK_SIZE=8
MIN_FREE_GB=5
SAMPLE_LIMIT=12
PROGRESS_EVERY=50

PREFIX_1="__text_feature_extractor__"
PREFIX_2="__text_video_connector__"
PREFIX_3="__text_audio_connector__"
PREFIX_4="text_video_connector_learnable_registers"
PREFIX_5="text_audio_connector_learnable_registers"

usage() {
  cat <<'EOF'
Usage:
  tools/run_clipfix2_postpatch_q6p.sh -f <existing_q6p_ckpt> [options]

Required:
  -f, --file <path>              Existing q6p checkpoint to patch in place.

Options:
      --baseline <path>          Baseline q6p used as source for clip-path rows.
                                 Default: dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt
      --backup <path>            Backup path for original file before patching.
                                 Default: <file>.pre_clipfix2_postpatch_<tag>.ckpt
      --tag <value>              Suffix tag for backup/probe file naming.
                                 Default: YYYYMMDD_HHMMSS
      --skip-backup              Skip backup step.
      --skip-probe               Skip pre/post targeted probes.
      --dry-run                  Selection + dry-run only, no file mutation.
      --allow-symlink-target     Allow mutating a symlink path (disabled by default).
      --chunk-size <n>           Chunk size for row updates (default: 8).
      --min-free-gb <n>          Minimum free disk threshold (default: 5).
      --sample-limit <n>         Probe sample limit (default: 12).
      --progress-every <n>       Probe progress interval (default: 50).
  -h, --help                     Show help.

What this script patches:
  - __text_feature_extractor__
  - __text_video_connector__
  - __text_audio_connector__
  - text_video_connector_learnable_registers
  - text_audio_connector_learnable_registers

It copies dim/data + metadata(type/format/datatype) for those rows from baseline
into your existing q6p file, then optionally runs targeted before/after probes.
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

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      TARGET="${2:-}"
      shift 2
      ;;
    --baseline)
      BASELINE="${2:-}"
      shift 2
      ;;
    --backup)
      BACKUP="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --skip-backup)
      SKIP_BACKUP=1
      shift
      ;;
    --skip-probe)
      SKIP_PROBE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-symlink-target)
      ALLOW_SYMLINK_TARGET=1
      shift
      ;;
    --chunk-size)
      CHUNK_SIZE="${2:-}"
      shift 2
      ;;
    --min-free-gb)
      MIN_FREE_GB="${2:-}"
      shift 2
      ;;
    --sample-limit)
      SAMPLE_LIMIT="${2:-}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:-}"
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

if [[ -z "$TARGET" ]]; then
  echo "error: --file is required" >&2
  usage
  exit 1
fi

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || ((CHUNK_SIZE < 1)); then
  echo "error: --chunk-size must be a positive integer" >&2
  exit 1
fi

if ! [[ "$SAMPLE_LIMIT" =~ ^[0-9]+$ ]] || ((SAMPLE_LIMIT < 1)); then
  echo "error: --sample-limit must be a positive integer" >&2
  exit 1
fi

if ! [[ "$PROGRESS_EVERY" =~ ^[0-9]+$ ]] || ((PROGRESS_EVERY < 1)); then
  echo "error: --progress-every must be a positive integer" >&2
  exit 1
fi

TARGET="$(abs_path "$TARGET")"
BASELINE="$(abs_path "$BASELINE")"

if [[ ! -f "$TARGET" ]]; then
  echo "error: target q6p file not found: $TARGET" >&2
  exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: baseline file not found: $BASELINE" >&2
  exit 1
fi

TARGET_REAL="$(readlink -f "$TARGET")"
BASELINE_REAL="$(readlink -f "$BASELINE")"

if [[ "$ALLOW_SYMLINK_TARGET" == "0" ]] && [[ -L "$TARGET" ]] && [[ "$DRY_RUN" == "0" ]]; then
  echo "error: target is a symlink; refusing in-place mutation by default" >&2
  echo "       target      : $TARGET" >&2
  echo "       resolves_to : $TARGET_REAL" >&2
  echo "       use --allow-symlink-target to override intentionally" >&2
  exit 1
fi

if [[ "$TARGET_REAL" == "$BASELINE_REAL" ]] && [[ "$DRY_RUN" == "0" ]]; then
  echo "error: target and baseline resolve to the same file; refusing no-op/self patch" >&2
  echo "       target_real  : $TARGET_REAL" >&2
  echo "       baseline_real: $BASELINE_REAL" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python not found at: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$BACKUP" ]]; then
  if [[ "$TARGET" == *.ckpt ]]; then
    BACKUP="${TARGET%.ckpt}.pre_clipfix2_postpatch_${TAG}.ckpt"
  else
    BACKUP="${TARGET}.pre_clipfix2_postpatch_${TAG}"
  fi
else
  BACKUP="$(abs_path "$BACKUP")"
fi

target_stem="$(basename "$TARGET")"
target_stem="${target_stem%.ckpt}"
target_slug="${target_stem//[^a-zA-Z0-9_]/_}"
WORK_DIR="$ROOT/output/clipfix2_postpatch_${target_slug}_${TAG}"
NAMES_FILE="$WORK_DIR/clipfix2_prefix_tensor_names.txt"
PRE_JSON="$WORK_DIR/probe_pre_vs_baseline.json"
PRE_MD="$WORK_DIR/probe_pre_vs_baseline.md"
POST_JSON="$WORK_DIR/probe_post_vs_baseline.json"
POST_MD="$WORK_DIR/probe_post_vs_baseline.md"

mkdir -p "$WORK_DIR"
if [[ "$SKIP_BACKUP" == "0" ]]; then
  mkdir -p "$(dirname "$BACKUP")"
fi

echo "==> clipfix2 postpatch on existing q6p"
echo "    target  : $TARGET"
echo "    target_real: $TARGET_REAL"
echo "    baseline: $BASELINE"
echo "    baseline_real: $BASELINE_REAL"
echo "    workdir : $WORK_DIR"
echo "    dry_run : $DRY_RUN"

echo "==> Selecting clip-path tensor names from target"
"$PYTHON_BIN" - "$TARGET" "$NAMES_FILE" \
  "$PREFIX_1" "$PREFIX_2" "$PREFIX_3" "$PREFIX_4" "$PREFIX_5" <<'PY'
import sqlite3
import sys
from pathlib import Path

target = Path(sys.argv[1])
out = Path(sys.argv[2])
prefixes = sys.argv[3:]

con = sqlite3.connect(str(target))
try:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors ORDER BY name").fetchall()
finally:
    con.close()

names = []
for row in rows:
    if not row or row[0] is None:
        continue
    name = str(row[0])
    if any(name.startswith(prefix) for prefix in prefixes):
        names.append(name)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")

print(f"selected_names={len(names)}")
for prefix in prefixes:
    count = sum(1 for n in names if n.startswith(prefix))
    print(f"selected_by_prefix[{prefix}]={count}")

if not names:
    print("RESULT=FAIL zero selected names")
    raise SystemExit(2)

print(f"names_file={out}")
PY

if [[ "$SKIP_PROBE" == "0" ]]; then
  echo "==> Pre-patch targeted probe"
  "$PYTHON_BIN" "$ROOT/tools/dt_probe_ckpt_targeted_content.py" \
    --file "$TARGET" \
    --baseline "$BASELINE" \
    --prefix "$PREFIX_1" \
    --prefix "$PREFIX_2" \
    --prefix "$PREFIX_3" \
    --prefix "$PREFIX_4" \
    --prefix "$PREFIX_5" \
    --sample-limit "$SAMPLE_LIMIT" \
    --progress-every "$PROGRESS_EVERY" \
    --out-json "$PRE_JSON" \
    --out-md "$PRE_MD"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==> Dry-run content alignment preview"
  "$PYTHON_BIN" "$ROOT/tools/dt_align_ckpt_content_subset.py" \
    --file "$TARGET" \
    --baseline "$BASELINE" \
    --dim-names-file "$NAMES_FILE" \
    --data-names-file "$NAMES_FILE" \
    --mode dry-run \
    --chunk-size "$CHUNK_SIZE" \
    --journal-mode preserve \
    --min-free-gb "$MIN_FREE_GB"

  echo "==> Done (dry-run, no file mutation)"
  exit 0
fi

if [[ "$SKIP_BACKUP" == "0" ]]; then
  echo "==> Backup target file"
  echo "    backup: $BACKUP"
  cp -f "$TARGET" "$BACKUP"
fi

echo "==> Apply dim/data patch for selected clip-path rows"
"$PYTHON_BIN" "$ROOT/tools/dt_align_ckpt_content_subset.py" \
  --file "$TARGET" \
  --baseline "$BASELINE" \
  --dim-names-file "$NAMES_FILE" \
  --data-names-file "$NAMES_FILE" \
  --mode apply \
  --chunk-size "$CHUNK_SIZE" \
  --journal-mode wal \
  --min-free-gb "$MIN_FREE_GB"

echo "==> Apply metadata patch (type/format/datatype) for selected rows"
"$PYTHON_BIN" - "$TARGET" "$BASELINE" "$NAMES_FILE" "$CHUNK_SIZE" <<'PY'
import sqlite3
import sys
from pathlib import Path

target = Path(sys.argv[1])
baseline = Path(sys.argv[2])
names_file = Path(sys.argv[3])
chunk_size = int(sys.argv[4])

names = [
    line.strip()
    for line in names_file.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

if not names:
    print("metadata_patch_result=FAIL no names")
    raise SystemExit(2)

def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]

tcon = sqlite3.connect(str(target))
try:
    tcur = tcon.cursor()
    tcur.execute("PRAGMA busy_timeout=15000")
    row = tcur.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = str(row[0]).lower() if row and row[0] is not None else "unknown"
    print(f"metadata_mutation_journal_mode={mode}")
    tcur.execute("ATTACH DATABASE ? AS baseline_db", (str(baseline),))

    updated = 0
    missing_baseline = 0
    missing_target = 0
    skipped_dataerror = 0
    total_chunks = (len(names) + chunk_size - 1) // chunk_size

    for idx, chunk in enumerate(chunks(names, chunk_size), start=1):
        tcon.execute("BEGIN IMMEDIATE")
        try:
            tcur.execute("DROP TABLE IF EXISTS temp._selected_names")
            tcur.execute("CREATE TEMP TABLE _selected_names(name TEXT PRIMARY KEY)")
            tcur.executemany(
                "INSERT INTO _selected_names(name) VALUES(?)",
                [(name,) for name in chunk],
            )

            row = tcur.execute(
                "SELECT COUNT(*) "
                "FROM _selected_names s "
                "LEFT JOIN tensors t ON t.name=s.name "
                "WHERE t.name IS NULL"
            ).fetchone()
            missing_target += int(row[0] if row else 0)

            row = tcur.execute(
                "SELECT COUNT(*) "
                "FROM _selected_names s "
                "LEFT JOIN baseline_db.tensors b ON b.name=s.name "
                "WHERE b.name IS NULL"
            ).fetchone()
            missing_baseline += int(row[0] if row else 0)

            row = tcur.execute(
                "SELECT COUNT(*) "
                "FROM _selected_names s "
                "JOIN tensors t ON t.name=s.name "
                "JOIN baseline_db.tensors b ON b.name=s.name"
            ).fetchone()
            updated += int(row[0] if row else 0)

            tcur.execute(
                "UPDATE tensors "
                "SET type=(SELECT b.type FROM baseline_db.tensors b WHERE b.name=tensors.name), "
                "format=(SELECT b.format FROM baseline_db.tensors b WHERE b.name=tensors.name), "
                "datatype=(SELECT b.datatype FROM baseline_db.tensors b WHERE b.name=tensors.name) "
                "WHERE name IN (SELECT name FROM _selected_names) "
                "AND EXISTS(SELECT 1 FROM baseline_db.tensors b WHERE b.name=tensors.name)"
            )

            tcon.commit()
            tcur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"metadata_chunk={idx}/{total_chunks} rows={len(chunk)}")
        except Exception:
            tcon.rollback()
            raise

    print(f"metadata_rows_updated={updated}")
    print(f"metadata_rows_missing_baseline={missing_baseline}")
    print(f"metadata_rows_missing_target={missing_target}")
    print(f"metadata_rows_skipped_dataerror={skipped_dataerror}")
finally:
    try:
        tcur.execute("DROP TABLE IF EXISTS temp._selected_names")
        tcur.execute("DETACH DATABASE baseline_db")
    except sqlite3.DatabaseError:
        pass
    tcon.close()
PY

echo "==> Structural validation"
"$PYTHON_BIN" "$ROOT/tools/dt_validate_converted_ckpt.py" --file "$TARGET" --profile ltx2_3

if [[ "$SKIP_PROBE" == "0" ]]; then
  echo "==> Post-patch targeted probe"
  "$PYTHON_BIN" "$ROOT/tools/dt_probe_ckpt_targeted_content.py" \
    --file "$TARGET" \
    --baseline "$BASELINE" \
    --prefix "$PREFIX_1" \
    --prefix "$PREFIX_2" \
    --prefix "$PREFIX_3" \
    --prefix "$PREFIX_4" \
    --prefix "$PREFIX_5" \
    --sample-limit "$SAMPLE_LIMIT" \
    --progress-every "$PROGRESS_EVERY" \
    --out-json "$POST_JSON" \
    --out-md "$POST_MD"
fi

echo "==> Done"
echo "    patched_file: $TARGET"
if [[ "$SKIP_BACKUP" == "0" ]]; then
  echo "    backup_file : $BACKUP"
fi
if [[ "$SKIP_PROBE" == "0" ]]; then
  echo "    pre_probe   : $PRE_MD"
  echo "    post_probe  : $POST_MD"
fi
