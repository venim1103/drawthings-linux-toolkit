#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${1:-$DEFAULT_ROOT}"
PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
REPO_ROOT="$ROOT/draw-things-community"
PATCHES_DIR="$PATCH_ROOT/patches"

if [[ ! -d "$PATCH_ROOT" ]]; then
  echo "error: patch root not found: $PATCH_ROOT" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

apply_git_patch() {
  local repo="$1"
  local patch_file="$2"
  local label="$3"

  if [[ ! -s "$patch_file" ]]; then
    echo "warning: patch file not found or empty for $label: $patch_file"
    return 1
  fi

  if git -C "$repo" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repo" apply --whitespace=nowarn "$patch_file"
    echo "applied patch: $label"
    return 0
  fi

  if git -C "$repo" apply --3way --whitespace=nowarn "$patch_file" >/dev/null 2>&1; then
    echo "applied patch with 3-way merge: $label"
    return 0
  fi

  echo "warning: could not apply unified patch for $label"
  return 1
}

copy_snapshot_patch() {
  local src="$1"
  local dst="$2"
  local required="${3:-0}"

  if [[ ! -f "$src" ]]; then
    if [[ "$required" == "1" ]]; then
      echo "error: required snapshot file missing: $src" >&2
      exit 1
    fi
    echo "warning: snapshot source not found: $src"
    return 1
  fi

  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "patched: ${dst#$REPO_ROOT/}"
  return 0
}

echo "==> Applying Draw Things quantization patch set..."

# Root repo patch: prefer unified diff, fallback to snapshot copy.
if ! apply_git_patch "$REPO_ROOT" "$PATCHES_DIR/draw-things-community.patch" "draw-things-community"; then
  copy_snapshot_patch "$PATCH_ROOT/Package.swift" "$REPO_ROOT/Package.swift" 1
  copy_snapshot_patch "$PATCH_ROOT/Package.resolved" "$REPO_ROOT/Package.resolved" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Apps/ModelConverter/Converter.swift" \
    "$REPO_ROOT/Apps/ModelConverter/Converter.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Libraries/ModelOp/Sources/ModelImporter.swift" \
    "$REPO_ROOT/Libraries/ModelOp/Sources/ModelImporter.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Apps/ModelQuantizer/Quantizer.swift" \
    "$REPO_ROOT/Apps/ModelQuantizer/Quantizer.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift" \
    "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Archive/SafeTensors.swift" \
    "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Archive/SafeTensors.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift" \
    "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift" 1
  copy_snapshot_patch \
    "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift" \
    "$REPO_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift" 1
fi

echo "==> Resolving Swift package dependencies to materialize checkouts..."
(
  cd "$REPO_ROOT"
  swift package resolve
)

# s4nnc checkout patch: prefer unified diff, fallback to snapshot copy.
if ! apply_git_patch \
  "$REPO_ROOT/.build/checkouts/s4nnc" \
  "$PATCHES_DIR/s4nnc.patch" \
  "s4nnc"; then
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/s4nnc/Package.swift" \
    "$REPO_ROOT/.build/checkouts/s4nnc/Package.swift"
fi

# ccv checkout patch: prefer unified diff, fallback to snapshot copy.
if ! apply_git_patch \
  "$REPO_ROOT/.build/checkouts/ccv" \
  "$PATCHES_DIR/ccv.patch" \
  "ccv"; then
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/Package.swift" \
    "$REPO_ROOT/.build/checkouts/ccv/Package.swift"
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_cnnp_model.c" \
    "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_cnnp_model.c"
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_cnnp_model_addons.c" \
    "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_cnnp_model_addons.c"
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_nnc_tensor.c" \
    "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_nnc_tensor.c"
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_nnc_cmd.c" \
    "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_nnc_cmd.c"
  copy_snapshot_patch \
    "$PATCH_ROOT/checkouts/ccv/lib/nnc/cmd/scaled_dot_product_attention/ccv_nnc_scaled_dot_product_attention.c" \
    "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/cmd/scaled_dot_product_attention/ccv_nnc_scaled_dot_product_attention.c"
fi

echo "==> Patch apply complete."