# drawthings-linux-toolkit

Linux-first toolkit for Draw Things conversion, quantization, and inference workflows.

## MOSS image generation

For this workspace's prepared patched candidate and independent model copies:

```bash
bash tools/dt_image.sh "A blue glass vase with white flowers, still life photograph."
bash tools/dt_image.sh --mode gpu --size 1024 --seed 123 "A lakeside cabin at sunrise."
```

The helper defaults to SFT 4-bit S, 512x512, 50 DDIMTrailing steps, CFG 3, shift 5, seed 4242, an empty negative prompt and CPU weight offload. It starts a temporary local server, validates the final PNG, prints its output path and stops the server. Original `dt-models` payloads and active catalogs are not selected for inference; the installed/default runtime is unchanged. Generated images, configs, logs and acceptance metadata remain together under the printed per-run directory in `output/`. Automatic tags include nanoseconds; existing explicit tags are rejected.

Options include `--model sft-i4x|sft-q8p|base-q8p`, `--width 768 --height 512`, `--steps`, `--seed`, `--negative-prompt`, `--port`, `--tag`, `--metrics` and `--dry-run`. Dimensions must be multiples of 64. Use `bash tools/dt_image.sh --help` for the complete interface.

`--mode gpu` omits forced CPU weight offload; it does not mean zero CPU use or guarantee no automatic host-memory use. Tokenization, file loading and orchestration still use the CPU, and upstream policy also depends on available host RAM. On this 12 GiB RTX A3000 devcontainer, SFT i4x passed full 50-step GPU-mode tests at both sizes:

| Size | Time Including Startup | Sampled Whole-Device Peak |
|---|---|---|
| 512x512 | 57.547 s | 8,133 MiB |
| 1024x1024 | 190.258 s | 8,133 MiB |

Both outputs were finite and visibly prompt-following, with complete streams/chunks and post-Echo; no telemetry readings were missing. These are individual observations, not controlled performance benchmarks or GPU-mode certification for q8p variants. Offload remains the conservative default. Evidence: `output/upgrades/moss-image-stateful-20261027/image-helper-gpu-i4x-50steps-20261028-summary.json` and the corresponding `image_helper_gpu_i4x_*_50steps_20261028` canaries. Wider model/quality limits remain in the [MOSS integration plan](docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md).

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
- `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` - CUDA runtime upgrade and MOSS-Image integration plan.
- `docs/WEELLM_HYBRID_TOOLING_PLAN_2026-10-24.md` - WeeLLM comparison and hybrid conversion/oracle tooling recipe.
- `docs/WEELLM_HYBRID_REVIEW_2026-10-26.md` - Historical defect review of the initial hybrid tooling baseline.
- `docs/MOSS_QUALITY_AND_CUDA_REVIEW_2026-10-26.md` - Review of the hardened CUDA runtime (`cf8baab..acd9b26`) and MOSS image-quality diagnosis plan.

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
