#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspaces/LTX2_3}"
REPO_ROOT="$ROOT/draw-things-community"
PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
PATCHES_DIR="$PATCH_ROOT/patches"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$PATCHES_DIR" "$PATCH_ROOT/checkouts/s4nnc" "$PATCH_ROOT/checkouts/ccv"

echo "==> Resolving Swift package dependencies..."
(
  cd "$REPO_ROOT"
  swift package resolve
)

echo "==> Refreshing snapshot backups..."
cp "$REPO_ROOT/Package.swift" "$PATCH_ROOT/Package.swift"
mkdir -p "$PATCH_ROOT/Apps/ModelQuantizer"
cp "$REPO_ROOT/Apps/ModelQuantizer/Quantizer.swift" "$PATCH_ROOT/Apps/ModelQuantizer/Quantizer.swift"
mkdir -p "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation"
cp \
  "$REPO_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift" \
  "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift"
cp "$REPO_ROOT/.build/checkouts/s4nnc/Package.swift" "$PATCH_ROOT/checkouts/s4nnc/Package.swift"
cp "$REPO_ROOT/.build/checkouts/ccv/Package.swift" "$PATCH_ROOT/checkouts/ccv/Package.swift"

echo "==> Regenerating unified patch files..."
git -C "$REPO_ROOT" diff -- \
  Package.swift \
  Apps/ModelQuantizer/Quantizer.swift \
  Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift \
  > "$PATCHES_DIR/draw-things-community.patch"

git -C "$REPO_ROOT/.build/checkouts/s4nnc" diff -- Package.swift > "$PATCHES_DIR/s4nnc.patch"
git -C "$REPO_ROOT/.build/checkouts/ccv" diff -- Package.swift > "$PATCHES_DIR/ccv.patch"

echo "==> Patch generation complete:"
wc -c \
  "$PATCHES_DIR/draw-things-community.patch" \
  "$PATCHES_DIR/s4nnc.patch" \
  "$PATCHES_DIR/ccv.patch"