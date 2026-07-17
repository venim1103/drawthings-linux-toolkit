# drawthings-linux-toolkit

Linux-first toolkit for Draw Things conversion, quantization, and inference workflows.

## Current recommended flow (LTX2.3 custom models)

Use the streamlined 3-command pipeline:

1. `bash tools/dt_convert.sh <model.safetensors>`
2. `bash tools/dt_quantize.sh <name> [--codec q6p|q8p|q4p]`
3. `bash tools/dt_test.sh <name>`

Details and examples are in `PIPELINE.md`.

## Key docs to keep handy

- `PIPELINE.md` - quickstart and codec notes.
- `CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md` - validated root-cause/fix history.
- `REPO_MIGRATION_GUIDE.md` - migration/sync guidance.
- `DRAW_THINGS_PATCH/README.md` - patch workflow for Draw Things source updates.

## Repository layout

- `.devcontainer/` - devcontainer build and runtime setup.
- `tools/` - active scripts used by the current pipeline.
- `tools/archive/` - legacy debug/runner scripts retained for reference.
- `docs/archive/` - historical runbooks and handoff docs.
- `DRAW_THINGS_PATCH/` - patch snapshots and generated patch files.

## Cleanup policy

- New production workflows should be added under `tools/`.
- Experimental or one-off debug scripts should go under `tools/archive/legacy_debug/` after they are no longer actively used.
- Historical session docs belong under `docs/archive/` to keep the repository root clean.
- Generated runtime artifacts should live under `output/` and can be archived/pruned regularly.

Legacy workspace notes are preserved in `docs/archive/2026_q6p_debug/WORKSPACE_MANUAL.md`.
