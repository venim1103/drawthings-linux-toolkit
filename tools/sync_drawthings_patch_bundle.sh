#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$DEFAULT_ROOT"
RUN_REGENERATE=1

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [workspace_root] [--no-regenerate]

Options:
  --no-regenerate   Skip unified patch regeneration step.
  -h, --help        Show this help message.

Behavior:
  1) Copy snapshot files from DRAW_THINGS_PATCH into draw-things-community.
  2) Resolve Swift packages to materialize .build/checkouts.
  3) Copy checkout snapshot files.
  4) Regenerate patch files (unless --no-regenerate is used).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --no-regenerate)
      RUN_REGENERATE=0
      ;;
    *)
      ROOT="$1"
      ;;
  esac
  shift
done

PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
REPO_ROOT="$ROOT/draw-things-community"
MANIFEST_FILE="$PATCH_ROOT/patches/manifest.sh"

if [[ ! -d "$PATCH_ROOT" ]]; then
  echo "error: patch root not found: $PATCH_ROOT" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST_FILE" ]]; then
  echo "error: patch manifest not found: $MANIFEST_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$MANIFEST_FILE"

copy_required() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    echo "error: required snapshot file missing: $src" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "synced: ${dst#$REPO_ROOT/}"
}

echo "==> Syncing DRAW_THINGS_PATCH snapshots into draw-things-community..."
for rel_path in "${PATCH_ROOT_FILES[@]}"; do
  copy_required "$PATCH_ROOT/$rel_path" "$REPO_ROOT/$rel_path"
done

echo "==> Resolving Swift packages to ensure checkouts are present..."
(
  cd "$REPO_ROOT"
  swift package resolve
)

echo "==> Syncing checkout snapshots..."
for rel_path in "${PATCH_S4NNC_FILES[@]}"; do
  copy_required "$PATCH_ROOT/checkouts/s4nnc/$rel_path" "$REPO_ROOT/.build/checkouts/s4nnc/$rel_path"
done

for rel_path in "${PATCH_CCV_FILES[@]}"; do
  copy_required "$PATCH_ROOT/checkouts/ccv/$rel_path" "$REPO_ROOT/.build/checkouts/ccv/$rel_path"
done

if [[ "$RUN_REGENERATE" == "1" ]]; then
  echo "==> Regenerating unified patch files..."
  bash "$SCRIPT_DIR/generate_drawthings_quant_patches.sh" "$ROOT"
else
  echo "==> Skipping patch regeneration (--no-regenerate)."
fi

echo "==> Sync complete."
