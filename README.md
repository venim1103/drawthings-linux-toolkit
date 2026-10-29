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

### Custom MOSS conversion (experimental)

Use full-precision BF16/F16/F32 safetensors from the supported 42-layer MOSS Image architecture, either a single file or an explicit sharded index. Quantized inputs, incomplete architectures and incompatible rotary frequencies are rejected. Adjacent `config.json` is checked when present; `--config` selects it explicitly.

The original BF16 transformer downloaded in this workspace is `OpenMOSS-Team/MOSS-Image-Base` at revision `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`. Use its explicit index below, or replace `SOURCE` with a custom full-precision safetensors file/index:

```bash
SOURCE=dt-models/moss-image-base-source-85975793/transformer/diffusion_pytorch_model.safetensors.index.json
.venv/bin/python tools/dt_moss_convert.py "$SOURCE" --inspect
.venv/bin/python tools/dt_moss_convert.py "$SOURCE" \
	--output dt-models/my_moss_f16.ckpt \
	--prepare-inference output/moss-models/my-moss
bash tools/dt_image.sh --bundle output/moss-models/my-moss --check "Model selection check"
```

Conversion writes 524 native tensors using only the supplied learned values. The timestep-output projection is preserved as F32; other tensors are F16. Bundle preparation packs native trailer storage offline using the raw codec's 16-byte offset/length descriptor and creates independent writable copies of the pinned Qwen encoder and Flux VAE from `dt-models`. It never registers the model in an active catalog or replaces an existing output. `bundle.json` records source and packaged hashes. To prepare an already converted checkpoint, use `--prepare-existing dt-models/my_moss_f16.ckpt --prepare-inference output/moss-models/my-moss` instead of a source/`--output`.

`--check` verifies bundle integrity and the model's decoded native catalog entry, then exits without generation. Custom bundles require `--mode offload`; it is not GPU-only certification. Remove `--check` and provide an image prompt to request generation. Native loading, finiteness and prompt-following quality are separate gates.

Original-source investigation on 2026-10-28 found a raw-trailer descriptor bug: packaging used the palettized codec's 24-byte layout where native raw decoding expects 16 bytes. Bounded post-encoding samples reached `DynamicGraph._Store.loadBytes` / `fread` / `read`; the wrong descriptor makes raw decoding use offset zero and can turn a payload offset into a huge read length. The earlier finite two-step image and 1,800-second 50-step timeout used that malformed bundle and do not establish correct loading or image quality.

Raw packaging, validation and reading now use the pinned native layout; all 66 CPU-focused checks pass. The existing private bundle was repaired by changing metadata only, with all trailer payload bytes hash-verified unchanged. On 2026-10-29, its guarded two-step request completed in 203.905 seconds including startup/cleanup: one finite 512x512 blurred image, complete stream/chunks and post-Echo, sampled GPU peak 2,305 MiB. The full 50-step request timed out at 900 seconds after 38 sampling callbacks and 25 previews, with no final image.

A separate native correctness fix now skips unsupported optional managed prefetch without clearing prior errors. Its private candidate passes all 93 CUDA-enabled tests and eight linked-runtime gates. A matched two-step request passed in 127.516 seconds; a bounded native staging experiment took 196.942 seconds with byte-identical output, so staging was rejected and removed. These are individual timings, not a benchmark. The guard-only 50-step request still timed out at the same 900-second cap after 37 sampling callbacks and 24 previews, with no final image; sustained callback intervals were approximately 14.5-28.3 seconds. Callbacks do not establish completed steps. All 1,792 GPU samples were present, peak was 2,305 MiB, post-Echo passed and cleanup left the GPU idle. Bundle and all nine protected hashes verified unchanged afterward. No installed runtime/default, model payload, precision or threading change was made.

Full-resolution prompt-following quality and practical full inference remain unaccepted. Bounded standalone matrix probes did not predict the real-model staging outcome; the next check must establish the actual weight layout/allocation path rather than repeat the same full request with a larger deadline. Resume details and local evidence are in the [MOSS handover](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md). Client progress remains unbuffered so timeouts retain received signposts.

Budget about 18 GiB for one original-sized inference bundle and about 27 GiB peak free space during preparation, in addition to the source and converted checkpoint. Preparation checks filesystem headroom, but cannot measure Windows host capacity from this container. Keep canonical weights and diagnostic reports; obsolete independent test copies are the first cleanup candidates. Deleting files inside WSL/Docker frees Linux blocks but may not immediately shrink its virtual disk on Windows. Do not delete the virtual disk or reset Docker to reclaim it.

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
