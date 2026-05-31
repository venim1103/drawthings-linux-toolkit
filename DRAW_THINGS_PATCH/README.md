# Draw Things Quantization Patch Bundle

This directory stores Linux quantization fixes for `draw-things-community`.

## Patch strategy

We keep two patch representations:

- Unified diffs (preferred): `patches/*.patch`
- Snapshot backups (fallback): copied files under this folder

`tools/apply_drawthings_quant_patch.sh` applies unified diffs first using `git apply` (with 3-way fallback). If that fails due to upstream drift, it falls back to snapshot copy for the affected file.

## Files covered

- Root repo files:
  - `Package.swift`
  - `Apps/ModelQuantizer/Quantizer.swift`
  - `Vendors/ZIPFoundation/Sources/ZIPFoundation/Archive+MemoryFile.swift`
- Checkout manifests:
  - `.build/checkouts/s4nnc/Package.swift`
  - `.build/checkouts/ccv/Package.swift`

## Apply patches

```bash
bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh
```

## Regenerate patches after edits

```bash
bash /workspaces/drawthings-linux-toolkit/tools/generate_drawthings_quant_patches.sh
```

## Build quantizer

```bash
cd /workspaces/drawthings-linux-toolkit/draw-things-community
swift build -c release --product model-quantizer
```