#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${1:-$DEFAULT_ROOT}"
REPO_ROOT="$ROOT/draw-things-community"
PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
PATCHES_DIR="$PATCH_ROOT/patches"
MANIFEST_FILE="$PATCHES_DIR/manifest.sh"

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

mkdir -p "$PATCHES_DIR" "$PATCH_ROOT/checkouts/s4nnc" "$PATCH_ROOT/checkouts/ccv"

copy_required_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "error: required source file missing: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

echo "==> Resolving Swift package dependencies..."
(
  cd "$REPO_ROOT"
  swift package resolve
)

echo "==> Refreshing snapshot backups..."
for rel_path in "${PATCH_ROOT_FILES[@]}"; do
  copy_required_file "$REPO_ROOT/$rel_path" "$PATCH_ROOT/$rel_path"
done

for rel_path in "${PATCH_S4NNC_FILES[@]}"; do
  copy_required_file \
    "$REPO_ROOT/.build/checkouts/s4nnc/$rel_path" \
    "$PATCH_ROOT/checkouts/s4nnc/$rel_path"
done

for rel_path in "${PATCH_CCV_FILES[@]}"; do
  copy_required_file \
    "$REPO_ROOT/.build/checkouts/ccv/$rel_path" \
    "$PATCH_ROOT/checkouts/ccv/$rel_path"
done

echo "==> Regenerating unified patch files..."
git -C "$REPO_ROOT" diff -- "${PATCH_ROOT_FILES[@]}" > "$PATCHES_DIR/draw-things-community.patch"

# Include untracked files (git diff omits them by default).
for rel_path in "${PATCH_ROOT_FILES[@]}"; do
  if git -C "$REPO_ROOT" ls-files --error-unmatch "$rel_path" >/dev/null 2>&1; then
    continue
  fi
  if [[ -f "$REPO_ROOT/$rel_path" ]]; then
    (
      cd "$REPO_ROOT"
      git diff --no-index -- /dev/null "$rel_path" || true
    ) >> "$PATCHES_DIR/draw-things-community.patch"
  fi
done

git -C "$REPO_ROOT/.build/checkouts/s4nnc" diff -- "${PATCH_S4NNC_FILES[@]}" > "$PATCHES_DIR/s4nnc.patch"
git -C "$REPO_ROOT/.build/checkouts/ccv" diff -- "${PATCH_CCV_FILES[@]}" > "$PATCHES_DIR/ccv.patch"

echo "==> Patch generation complete:"
wc -c \
  "$PATCHES_DIR/draw-things-community.patch" \
  "$PATCHES_DIR/s4nnc.patch" \
  "$PATCHES_DIR/ccv.patch"
