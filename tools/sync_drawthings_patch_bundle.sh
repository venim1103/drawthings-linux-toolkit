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

if [[ ! -d "$PATCH_ROOT" ]]; then
  echo "error: patch root not found: $PATCH_ROOT" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

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
copy_required "$PATCH_ROOT/Package.swift" "$REPO_ROOT/Package.swift"
copy_required "$PATCH_ROOT/Package.resolved" "$REPO_ROOT/Package.resolved"
copy_required "$PATCH_ROOT/Apps/ModelConverter/Converter.swift" "$REPO_ROOT/Apps/ModelConverter/Converter.swift"
copy_required "$PATCH_ROOT/Libraries/ModelOp/Sources/ModelImporter.swift" "$REPO_ROOT/Libraries/ModelOp/Sources/ModelImporter.swift"
copy_required "$PATCH_ROOT/Apps/ModelQuantizer/Quantizer.swift" "$REPO_ROOT/Apps/ModelQuantizer/Quantizer.swift"
copy_required "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift" "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Functional+SwishMul.swift"
copy_required "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Archive/SafeTensors.swift" "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Archive/SafeTensors.swift"
copy_required "$PATCH_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift" "$REPO_ROOT/Libraries/SwiftDiffusion/Sources/Models/HiDream.swift"
copy_required "$PATCH_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift" "$REPO_ROOT/Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift"

echo "==> Resolving Swift packages to ensure checkouts are present..."
(
  cd "$REPO_ROOT"
  swift package resolve
)

echo "==> Syncing checkout snapshots..."
copy_required "$PATCH_ROOT/checkouts/s4nnc/Package.swift" "$REPO_ROOT/.build/checkouts/s4nnc/Package.swift"
copy_required "$PATCH_ROOT/checkouts/ccv/Package.swift" "$REPO_ROOT/.build/checkouts/ccv/Package.swift"
copy_required "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_cnnp_model.c" "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_cnnp_model.c"
copy_required "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_cnnp_model_addons.c" "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_cnnp_model_addons.c"
copy_required "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_nnc_tensor.c" "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_nnc_tensor.c"
copy_required "$PATCH_ROOT/checkouts/ccv/lib/nnc/ccv_nnc_cmd.c" "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/ccv_nnc_cmd.c"
copy_required "$PATCH_ROOT/checkouts/ccv/lib/nnc/cmd/scaled_dot_product_attention/ccv_nnc_scaled_dot_product_attention.c" "$REPO_ROOT/.build/checkouts/ccv/lib/nnc/cmd/scaled_dot_product_attention/ccv_nnc_scaled_dot_product_attention.c"

if [[ "$RUN_REGENERATE" == "1" ]]; then
  echo "==> Regenerating unified patch files..."
  bash "$SCRIPT_DIR/generate_drawthings_quant_patches.sh" "$ROOT"
else
  echo "==> Skipping patch regeneration (--no-regenerate)."
fi

echo "==> Sync complete."
