#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VERIFY_SCRIPT="$SCRIPT_DIR/verify_drawthings_patch_bundle.sh"
SAFETY_SCRIPT="$SCRIPT_DIR/strict_drawthings_safety_check.sh"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") before [workspace_root]
  $(basename "$0") after [workspace_root] [strict_safety_check options]

Commands:
  before   Run strict patch verify and save an update baseline snapshot.
  after    Run full strict safety gate after upstream update/rebase.

Examples:
  # Before updating draw-things-community
  bash tools/drawthings_update_guard.sh before

  # After upstream update/rebase
  bash tools/drawthings_update_guard.sh after --allow-dirty

  # Faster after-check (skip smoke build)
  bash tools/drawthings_update_guard.sh after --allow-dirty --skip-build
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

ROOT="$DEFAULT_ROOT"

if [[ $# -gt 0 && "$1" != --* ]]; then
  ROOT="$1"
  shift
fi

PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
PATCHES_DIR="$PATCH_ROOT/patches"
REPO_ROOT="$ROOT/draw-things-community"
BASELINE_FILE="$PATCHES_DIR/update_baseline_latest.env"

if [[ ! -d "$ROOT" ]]; then
  echo "error: workspace root not found: $ROOT" >&2
  exit 1
fi
if [[ ! -d "$PATCH_ROOT" ]]; then
  echo "error: patch root not found: $PATCH_ROOT" >&2
  exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$PATCHES_DIR"

save_baseline() {
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  local toolkit_head dt_head s4nnc_head ccv_head dt_branch
  toolkit_head="$(git -C "$ROOT" rev-parse HEAD)"
  dt_head="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dt_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

  s4nnc_head="unknown"
  if [[ -d "$REPO_ROOT/.build/checkouts/s4nnc/.git" ]]; then
    s4nnc_head="$(git -C "$REPO_ROOT/.build/checkouts/s4nnc" rev-parse HEAD)"
  fi

  ccv_head="unknown"
  if [[ -d "$REPO_ROOT/.build/checkouts/ccv/.git" ]]; then
    ccv_head="$(git -C "$REPO_ROOT/.build/checkouts/ccv" rev-parse HEAD)"
  fi

  cat > "$BASELINE_FILE" <<EOF
timestamp="$timestamp"
toolkit_head="$toolkit_head"
draw_things_branch="$dt_branch"
draw_things_head="$dt_head"
s4nnc_head="$s4nnc_head"
ccv_head="$ccv_head"
EOF

  echo "==> Saved baseline: $BASELINE_FILE"
  echo "    timestamp:          $timestamp"
  echo "    draw-things branch: $dt_branch"
  echo "    draw-things head:   $dt_head"
}

show_baseline() {
  if [[ -f "$BASELINE_FILE" ]]; then
    echo "==> Existing baseline"
    # shellcheck disable=SC1090
    source "$BASELINE_FILE"
    echo "    timestamp:          ${timestamp:-unknown}"
    echo "    draw-things branch: ${draw_things_branch:-unknown}"
    echo "    draw-things head:   ${draw_things_head:-unknown}"
    echo "    s4nnc head:         ${s4nnc_head:-unknown}"
    echo "    ccv head:           ${ccv_head:-unknown}"
  else
    echo "==> No prior baseline found at $BASELINE_FILE"
  fi
}

case "$COMMAND" in
  before)
    echo "==> BEFORE update guard"
    bash "$VERIFY_SCRIPT" "$ROOT"
    save_baseline
    echo "==> Ready for upstream update/rebase."
    ;;
  after)
    echo "==> AFTER update guard"
    show_baseline
    bash "$SAFETY_SCRIPT" "$ROOT" "$@"
    echo "==> After-update safety checks passed."
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command '$COMMAND'" >&2
    usage
    exit 1
    ;;
esac
