#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${1:-$DEFAULT_ROOT}"
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
mkdir -p "$PATCH_ROOT/Apps/ModelConverter"
cp "$REPO_ROOT/Apps/ModelConverter/Converter.swift" "$PATCH_ROOT/Apps/ModelConverter/Converter.swift"
mkdir -p "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Models"
cp \
  "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift" \
  "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift"
cp \
  "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift" \
  "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift"
mkdir -p "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation"
cp \
  "$REPO_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift" \
  "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift"
cp "$REPO_ROOT/.build/checkouts/s4nnc/Package.swift" "$PATCH_ROOT/checkouts/s4nnc/Package.swift"
cp "$REPO_ROOT/.build/checkouts/ccv/Package.swift" "$PATCH_ROOT/checkouts/ccv/Package.swift"

echo "==> Regenerating unified patch files..."
PATCH_TARGETS=(
  Package.swift
  Apps/ModelConverter/Converter.swift
  Apps/ModelQuantizer/Quantizer.swift
  Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift
  Libraries/SwiftDiffusion/Sources/Models/HiDream.swift
  Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift
)

git -C "$REPO_ROOT" diff -- "${PATCH_TARGETS[@]}" > "$PATCHES_DIR/draw-things-community.patch"

# Include untracked files (git diff omits them by default).
for rel_path in "${PATCH_TARGETS[@]}"; do
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

git -C "$REPO_ROOT/.build/checkouts/s4nnc" diff -- Package.swift > "$PATCHES_DIR/s4nnc.patch"
git -C "$REPO_ROOT/.build/checkouts/ccv" diff -- Package.swift > "$PATCHES_DIR/ccv.patch"

echo "==> Patch generation complete:"
wc -c \
  "$PATCHES_DIR/draw-things-community.patch" \
  "$PATCHES_DIR/s4nnc.patch" \
  "$PATCHES_DIR/ccv.patch"