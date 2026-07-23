# Draw Things Quantization Patch Bundle

This directory stores Linux quantization fixes for `draw-things-community`.

## Patch strategy

We keep two patch representations:

- Unified diffs (preferred): `patches/*.patch`
- Snapshot backups (fallback): copied files under this folder

`tools/apply_drawthings_quant_patch.sh` applies unified diffs first using `git apply` (with 3-way fallback). If that fails due to upstream drift, default mode falls back to snapshot copy for the affected file.

For upstream-update safety, use strict verification mode (see below), which disallows fallback copy.

## Single source of truth

Patch file coverage is defined in:

- `DRAW_THINGS_PATCH/patches/manifest.sh`

All patch tools consume that manifest:

- `tools/apply_drawthings_quant_patch.sh`
- `tools/sync_drawthings_patch_bundle.sh`
- `tools/generate_drawthings_quant_patches.sh`

## Files covered

See `DRAW_THINGS_PATCH/patches/manifest.sh` for the full list (root files + s4nnc files + ccv files).

## Apply patches (default, fallback allowed)

```bash
bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh
```

## Strict apply (fallback disabled)

```bash
bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh --strict
```

Behavior:

- Unified patch apply is required for all patch sets.
- Any would-fallback case becomes a hard failure.

## Verify mode (strict + idempotence gate)

```bash
bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh --verify
```

Behavior:

- Implies `--strict`.
- Applies patches once.
- Applies patches again and verifies the second pass is a no-op.
- Fails if second pass changes repo state.

Useful for CI / pre-merge update gates.

Convenience wrapper (recommended for CI):

```bash
bash /workspaces/drawthings-linux-toolkit/tools/verify_drawthings_patch_bundle.sh
```

Optional machine-readable report:

```bash
bash /workspaces/drawthings-linux-toolkit/tools/verify_drawthings_patch_bundle.sh --json-report /tmp/dt_patch_report.json
```

Exit codes:

- `0`: success
- `2`: unified patch apply failed in strict/verify mode
- `3`: required patch file missing/empty
- `4`: checkout preparation failure
- `5`: idempotence failure in verify mode
- `6`: verify integrity failure

## Sync toolkit snapshots into draw-things-community

Use this after editing files under `DRAW_THINGS_PATCH/` so the working
`draw-things-community` tree and unified patch files stay aligned.

```bash
bash /workspaces/drawthings-linux-toolkit/tools/sync_drawthings_patch_bundle.sh
```

Pass `--no-regenerate` to skip the patch regeneration step.

## Regenerate patches after edits

```bash
bash /workspaces/drawthings-linux-toolkit/tools/generate_drawthings_quant_patches.sh
```

## Recommended upstream update workflow (strict)

1. Update `draw-things-community` to desired upstream ref.
2. Run strict/idempotence gate:

   ```bash
   bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh --verify
   ```

3. If it fails, rebase conflicting changes and regenerate bundle:

   ```bash
   bash /workspaces/drawthings-linux-toolkit/tools/generate_drawthings_quant_patches.sh
   ```

4. Re-run `--verify` on a clean tree.

## One-command strict local safety gate

Use this when you want to be strict/careful before moving to the next task:

```bash
bash /workspaces/drawthings-linux-toolkit/tools/strict_drawthings_safety_check.sh --allow-dirty
```

What it does:

1. Shell syntax checks for patch scripts + manifest.
2. Manifest path integrity checks.
3. Strict patch verify (idempotence).
4. Patch regeneration.
5. Strict patch verify again.
6. Smoke build (`model-quantizer` by default).

Notes:

- Omit `--allow-dirty` to require a clean `DRAW_THINGS_PATCH` tree.
- Use `--skip-build` for a faster gate when you only need patch integrity.
- Use `--build-product <name>` to smoke-test a different Swift product.

## Build quantizer

```bash
cd /workspaces/drawthings-linux-toolkit/draw-things-community
swift build -c release --product model-quantizer
```
