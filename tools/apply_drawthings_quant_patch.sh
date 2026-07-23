#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$DEFAULT_ROOT"
PATCH_ROOT=""
REPO_ROOT=""
PATCHES_DIR=""
MANIFEST_FILE=""

STRICT_MODE=0
VERIFY_MODE=0
JSON_REPORT_PATH=""

EXIT_PATCH_FAIL=2
EXIT_PATCH_MISSING=3
EXIT_CHECKOUT_PREP=4
EXIT_IDEMPOTENCE_FAIL=5
EXIT_VERIFY_INTEGRITY=6

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [workspace_root] [--strict] [--verify] [--json-report <path>]

Modes:
  default         Apply unified patches first; fallback to snapshot copy if needed.
  --strict        Unified patch apply is required. Fallback copy is disabled.
  --verify        Implies --strict and enforces idempotence:
                  apply strict once, then apply strict again and ensure no changes.

Options:
  --json-report <path>
                  Write a JSON summary report.
  -h, --help      Show this help text.

Exit codes:
  0 success
  2 unified patch apply failed in strict/verify mode
  3 required patch file missing/empty
  4 checkout preparation failure (swift package resolve / missing checkout dirs)
  5 idempotence failure in --verify mode
  6 verify integrity failure
EOF
}

escape_json() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

write_json_report() {
  local status="$1"
  local message="$2"
  local root_applied="$3"
  local s4nnc_applied="$4"
  local ccv_applied="$5"
  local root_fallback="$6"
  local s4nnc_fallback="$7"
  local ccv_fallback="$8"
  local verify_second_changed="$9"

  [[ -n "$JSON_REPORT_PATH" ]] || return 0
  mkdir -p "$(dirname "$JSON_REPORT_PATH")"

  cat > "$JSON_REPORT_PATH" <<EOF
{
  "status": "$(escape_json "$status")",
  "message": "$(escape_json "$message")",
  "strict_mode": $STRICT_MODE,
  "verify_mode": $VERIFY_MODE,
  "patches": {
    "draw_things_community": {
      "unified_applied": $root_applied,
      "fallback_used": $root_fallback
    },
    "s4nnc": {
      "unified_applied": $s4nnc_applied,
      "fallback_used": $s4nnc_fallback
    },
    "ccv": {
      "unified_applied": $ccv_applied,
      "fallback_used": $ccv_fallback
    }
  },
  "verify": {
    "second_apply_changed_files": $verify_second_changed
  }
}
EOF
}

fail_with_code() {
  local code="$1"
  local message="$2"
  local root_applied="${3:-false}"
  local s4nnc_applied="${4:-false}"
  local ccv_applied="${5:-false}"
  local root_fallback="${6:-false}"
  local s4nnc_fallback="${7:-false}"
  local ccv_fallback="${8:-false}"
  local verify_second_changed="${9:-false}"

  echo "error: $message" >&2
  write_json_report \
    "error" "$message" \
    "$root_applied" "$s4nnc_applied" "$ccv_applied" \
    "$root_fallback" "$s4nnc_fallback" "$ccv_fallback" \
    "$verify_second_changed"
  exit "$code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --strict)
      STRICT_MODE=1
      ;;
    --verify)
      VERIFY_MODE=1
      STRICT_MODE=1
      ;;
    --json-report)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --json-report requires a path argument" >&2
        exit 1
      fi
      JSON_REPORT_PATH="$1"
      ;;
    --json-report=*)
      JSON_REPORT_PATH="${1#--json-report=}"
      ;;
    *)
      if [[ "$ROOT" != "$DEFAULT_ROOT" ]]; then
        echo "error: multiple workspace roots provided: '$ROOT' and '$1'" >&2
        exit 1
      fi
      ROOT="$1"
      ;;
  esac
  shift
done

PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
REPO_ROOT="$ROOT/draw-things-community"
PATCHES_DIR="$PATCH_ROOT/patches"
MANIFEST_FILE="$PATCHES_DIR/manifest.sh"

if [[ ! -d "$PATCH_ROOT" ]]; then
  fail_with_code "$EXIT_VERIFY_INTEGRITY" "patch root not found: $PATCH_ROOT"
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  fail_with_code "$EXIT_VERIFY_INTEGRITY" "draw-things-community repo not found: $REPO_ROOT"
fi

if [[ ! -f "$MANIFEST_FILE" ]]; then
  fail_with_code "$EXIT_VERIFY_INTEGRITY" "patch manifest not found: $MANIFEST_FILE"
fi

# shellcheck disable=SC1090
source "$MANIFEST_FILE"

ensure_checkout_layout() {
  echo "==> Resolving Swift package dependencies to materialize checkouts..."
  if ! (
    cd "$REPO_ROOT"
    swift package resolve
  ); then
    fail_with_code "$EXIT_CHECKOUT_PREP" "swift package resolve failed"
  fi

  if [[ ! -d "$REPO_ROOT/.build/checkouts/s4nnc" ]]; then
    fail_with_code "$EXIT_CHECKOUT_PREP" "missing checkout after resolve: $REPO_ROOT/.build/checkouts/s4nnc"
  fi
  if [[ ! -d "$REPO_ROOT/.build/checkouts/ccv" ]]; then
    fail_with_code "$EXIT_CHECKOUT_PREP" "missing checkout after resolve: $REPO_ROOT/.build/checkouts/ccv"
  fi
}

require_nonempty_patch() {
  local patch_file="$1"
  local label="$2"
  if [[ ! -s "$patch_file" ]]; then
    fail_with_code "$EXIT_PATCH_MISSING" "patch file not found or empty for $label: $patch_file"
  fi
}

copy_snapshot_patch() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    fail_with_code "$EXIT_VERIFY_INTEGRITY" "required snapshot file missing: $src"
  fi

  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "patched: ${dst#$REPO_ROOT/}"
}

apply_unified_patch() {
  local repo="$1"
  local patch_file="$2"
  local label="$3"

  if git -C "$repo" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repo" apply --whitespace=nowarn "$patch_file"
    echo "applied patch: $label"
    return 0
  fi

  if git -C "$repo" apply --3way --whitespace=nowarn "$patch_file" >/dev/null 2>&1; then
    echo "applied patch with 3-way merge: $label"
    return 0
  fi

  if git -C "$repo" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "patch already applied: $label"
    return 0
  fi

  return 1
}

apply_one_patchset() {
  local repo="$1"
  local patch_file="$2"
  local label="$3"
  local snapshot_root="$4"
  shift 4
  local paths=("$@")

  require_nonempty_patch "$patch_file" "$label"

  if apply_unified_patch "$repo" "$patch_file" "$label"; then
    return 0
  fi

  if [[ "$STRICT_MODE" == "1" ]]; then
    fail_with_code "$EXIT_PATCH_FAIL" "could not apply unified patch for $label in strict mode"
  fi

  echo "warning: could not apply unified patch for $label; falling back to snapshot copy"
  local rel_path
  for rel_path in "${paths[@]}"; do
    copy_snapshot_patch "$snapshot_root/$rel_path" "$repo/$rel_path"
  done
  return 10
}

run_apply_once() {
  local result_file="$1"
  local root_applied="false"
  local s4nnc_applied="false"
  local ccv_applied="false"
  local root_fallback="false"
  local s4nnc_fallback="false"
  local ccv_fallback="false"

  if apply_one_patchset \
    "$REPO_ROOT" \
    "$PATCHES_DIR/draw-things-community.patch" \
    "draw-things-community" \
    "$PATCH_ROOT" \
    "${PATCH_ROOT_FILES[@]}"; then
    root_applied="true"
  else
    root_fallback="true"
  fi

  if apply_one_patchset \
    "$REPO_ROOT/.build/checkouts/s4nnc" \
    "$PATCHES_DIR/s4nnc.patch" \
    "s4nnc" \
    "$PATCH_ROOT/checkouts/s4nnc" \
    "${PATCH_S4NNC_FILES[@]}"; then
    s4nnc_applied="true"
  else
    s4nnc_fallback="true"
  fi

  if apply_one_patchset \
    "$REPO_ROOT/.build/checkouts/ccv" \
    "$PATCHES_DIR/ccv.patch" \
    "ccv" \
    "$PATCH_ROOT/checkouts/ccv" \
    "${PATCH_CCV_FILES[@]}"; then
    ccv_applied="true"
  else
    ccv_fallback="true"
  fi

  cat > "$result_file" <<EOF
root_applied=$root_applied
s4nnc_applied=$s4nnc_applied
ccv_applied=$ccv_applied
root_fallback=$root_fallback
s4nnc_fallback=$s4nnc_fallback
ccv_fallback=$ccv_fallback
EOF
}

run_verify_idempotence() {
  local first_result_file="$1"
  local root_after_first s4nnc_after_first ccv_after_first
  local root_after_second s4nnc_after_second ccv_after_second
  root_after_first="$(mktemp)"
  s4nnc_after_first="$(mktemp)"
  ccv_after_first="$(mktemp)"
  root_after_second="$(mktemp)"
  s4nnc_after_second="$(mktemp)"
  ccv_after_second="$(mktemp)"

  run_apply_once "$first_result_file"

  git -C "$REPO_ROOT" status --porcelain > "$root_after_first"
  git -C "$REPO_ROOT/.build/checkouts/s4nnc" status --porcelain > "$s4nnc_after_first"
  git -C "$REPO_ROOT/.build/checkouts/ccv" status --porcelain > "$ccv_after_first"

  local second_result_file
  second_result_file="$(mktemp)"
  run_apply_once "$second_result_file"
  rm -f "$second_result_file"

  git -C "$REPO_ROOT" status --porcelain > "$root_after_second"
  git -C "$REPO_ROOT/.build/checkouts/s4nnc" status --porcelain > "$s4nnc_after_second"
  git -C "$REPO_ROOT/.build/checkouts/ccv" status --porcelain > "$ccv_after_second"

  if ! cmp -s "$root_after_first" "$root_after_second" \
    || ! cmp -s "$s4nnc_after_first" "$s4nnc_after_second" \
    || ! cmp -s "$ccv_after_first" "$ccv_after_second"; then
    rm -f "$root_after_first" "$s4nnc_after_first" "$ccv_after_first" "$root_after_second" "$s4nnc_after_second" "$ccv_after_second"
    fail_with_code \
      "$EXIT_IDEMPOTENCE_FAIL" \
      "verify failed: second strict apply changed repository state" \
      "false" "false" "false" "false" "false" "false" "true"
  fi

  rm -f "$root_after_first" "$s4nnc_after_first" "$ccv_after_first" "$root_after_second" "$s4nnc_after_second" "$ccv_after_second"
}

echo "==> Applying Draw Things quantization patch set..."
ensure_checkout_layout

apply_result_file="$(mktemp)"

if [[ "$VERIFY_MODE" == "1" ]]; then
  run_verify_idempotence "$apply_result_file"
else
  run_apply_once "$apply_result_file"
fi

# shellcheck disable=SC1090
source "$apply_result_file"
rm -f "$apply_result_file"

if [[ "$STRICT_MODE" == "1" ]]; then
  if [[ "$root_fallback" == "true" || "$s4nnc_fallback" == "true" || "$ccv_fallback" == "true" ]]; then
    fail_with_code \
      "$EXIT_PATCH_FAIL" \
      "strict mode violation: fallback snapshot copy was used" \
      "$root_applied" "$s4nnc_applied" "$ccv_applied" \
      "$root_fallback" "$s4nnc_fallback" "$ccv_fallback"
  fi
fi

write_json_report \
  "ok" "patch apply complete" \
  "$root_applied" "$s4nnc_applied" "$ccv_applied" \
  "$root_fallback" "$s4nnc_fallback" "$ccv_fallback" \
  "false"

echo "==> Patch apply complete."
