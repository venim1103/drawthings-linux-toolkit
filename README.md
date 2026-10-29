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

Base and SFT use the same supported MOSS architecture and native backend. Official SFT q8p and i4x already pass full image tests; SFT does not require a separate converter, tokenizer or inference patch. A custom full-precision SFT or weight merge must still pass source inspection and its own image acceptance test. See the [SFT results](DRAW_THINGS_PATCH/cuda/README.md#base-repeatability-and-sft-q8p-bring-up-october-27).

Use full-precision BF16/F16/F32 safetensors from the supported 42-layer MOSS Image architecture, either a single file or an explicit sharded index. Quantized inputs, incomplete architectures and incompatible rotary frequencies are rejected. Adjacent `config.json` is checked when present; `--config` selects it explicitly.

The original BF16 transformer downloaded in this workspace is `OpenMOSS-Team/MOSS-Image-Base` at revision `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`. Use its explicit index below, or replace `SOURCE` with a custom full-precision safetensors file/index:

```bash
SOURCE=dt-models/moss-image-base-source-85975793/transformer/diffusion_pytorch_model.safetensors.index.json
.venv/bin/python tools/dt_moss_convert.py "$SOURCE" --inspect
.venv/bin/python tools/dt_moss_convert.py "$SOURCE" \
	--output dt-models/my_moss_f16.ckpt \
	--prepare-inference output/moss-models/my-moss
bash tools/dt_image.sh --bundle output/moss-models/my-moss \
	--grpc-bin output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI \
	--check "Model selection check"
```

Conversion writes 524 native tensors using only the supplied learned values. The timestep-output projection is preserved as F32; other tensors are F16. Bundle preparation packs native trailer storage offline using the raw codec's 16-byte offset/length descriptor and creates independent writable copies of the pinned Qwen encoder and Flux VAE from `dt-models`. It never registers the model in an active catalog or replaces an existing output. `bundle.json` records source and packaged hashes. To prepare an already converted checkpoint, use `--prepare-existing dt-models/my_moss_f16.ckpt --prepare-inference output/moss-models/my-moss` instead of a source/`--output`.

To create genuine packed `i4x` from a validated full-precision MOSS checkpoint, use the separate experimental quantizer, not the LTX-oriented `dt_quantize.sh` or palettized `q4p`:

```bash
.venv/bin/python tools/dt_moss_quantize.py dt-models/my_moss_f16.ckpt \
	--output dt-models/my_moss_i4x.ckpt --ccv-source /path/to/pristine/pinned/ccv \
	--jobs 16 --prepare-inference output/moss-models/my-moss-i4x
```

The native CPU encoder is hash-pinned to CCV `69b1ed1f` and writes Q4_K with regular Hadamard-256 for 308 core matrices. The other 216 sensitive tensors retain their own F16/F32 bytes, a more conservative policy than the official mixed-codec artifact. This is not a renamed `q4p` checkpoint or a requantization of official low-bit weights. Outputs must be new paths; canonical inputs and active catalogs are not modified. Matrix workers affect only CPU quantization, not inference threading. An already produced artifact can be bundled with `--prepare-existing ... --quantization i4x --prepare-inference ...`; the declaration enables strict packed-schema validation, not blanket acceptance of encoded tensors. Numerical codec/storage tests do not replace a checkpoint-specific full-image test.

`--check` verifies bundle integrity and the model's decoded native catalog entry, then exits without generation. Custom bundles require `--mode offload`; it is not GPU-only certification. The explicit `--grpc-bin` selects the patched private candidate, or an executable resource guard, without changing the helper's existing default or installing anything. Build the current profile using the [CUDA integration instructions](DRAW_THINGS_PATCH/cuda/README.md#isolated-server-integration). Remove `--check` and provide an image prompt to request generation. Native loading, finiteness and prompt-following quality are separate gates.

**Original-source end-to-end bring-up now passes.** On 2026-10-29, our conversion of the pinned original BF16 source completed a 512x512, 50-step DDIMTrailing request in **289.534 seconds including startup/cleanup** on the RTX A3000 Laptop GPU with CPU offload. The viewed image follows the orange-teapot, white-table, green-plant and window-light prompt. Native `moss_image` selection, one finite final image, complete stream/chunks and post-Echo all pass. Local result: `output/q6p_canary_moss_original_raw_actual_managed_staging_50steps_20261029_a/image.png`.

The fixes cover native raw-trailer descriptors, unsupported managed prefetch, and bounded dense-GEMM staging selected from actual CUDA allocation attributes. Observed managed denoiser weights lacked the CCV mapped flag, so the earlier flag-based staging experiment missed them. The accepted candidate is SHA-256 `acafef3d7f8f87588355cca88c41b21ad975665650271e1e23e53e0e0a2bf12b`; all 95 CUDA-enabled tests and eight linked-runtime gates pass. Its two-step output is byte-identical to the preserved guard-only control. Full-run sampled peak was 2,305 MiB and maximum temperature 66 C; timing is an individual observation, not a benchmark. Bundle verification and all nine protected hashes pass after inference. Installed defaults, active catalogs, canonical weights, precision and threading remain unchanged.

**A custom merged SFT checkpoint also passes.** The local `dt-models/mosskr2_image_sft_bf16_v1.safetensors` matches the existing source contract and converts with the same F16/F32 policy, without merge-specific code changes. After an interrupted session/restart, the completed private bundle verified intact; both guarded two-step and standard 50-step requests pass. The full 512x512 image visibly follows the teapot/table/plant/window prompt and completed in **284.303 seconds including startup/cleanup**. No Turbo/DMD extras or additional LoRA were used. See the [merged-SFT result and repeat command](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#custom-merged-sft-result-october-29).

These runs verify the original source and one custom merge at one prompt/profile, not arbitrary custom checkpoints, wider prompt/resolution coverage, framework parity, GPU-only inference or custom low-bit conversion. Other compatible custom full-precision checkpoints use the same conversion path but need their own acceptance runs. The merge creator's quality-improvement and low-step Turbo claims were not tested. Failed probes and the earlier malformed-bundle/time-out results remain recorded in the [MOSS handover](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md). Client progress remains unbuffered so timeouts retain received signposts.

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
