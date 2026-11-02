# drawthings-linux-toolkit

Linux-first toolkit for Draw Things conversion, quantization, and inference workflows.

## MOSS image generation

For this workspace's prepared patched candidate and independent model copies:

```bash
bash tools/dt_image.sh "A blue glass vase with white flowers, still life photograph."
bash tools/dt_image.sh --mode gpu --size 1024 --seed 123 "A lakeside cabin at sunrise."
bash tools/dt_image.sh --preset mosskr2-lora4 "An orange ceramic teapot on a white table."
```

`--preset mosskr2-lora4` selects the fastest tested local LoRA profile: our MOSSkr2 i4x bundle, the imported four-step LoRA at weight 1.0, 512x512, four DDIMTrailing steps, CFG 1, shift 5, seed 4242, GPU policy, the patched resource-guarded server, telemetry and a 180-second generation timeout. It configures the guard environment automatically; no long paths or environment variables are needed. This requires the prepared local bundle, adapter, guard and server, not just a fresh checkout. The measured profile took about 27 seconds including startup and used about 8 GiB VRAM; new prompts/settings can differ.

Without a preset, the helper still defaults to SFT 4-bit S, 512x512, 50 DDIMTrailing steps, CFG 3, shift 5, seed 4242, an empty negative prompt and CPU weight offload. It starts a temporary local server, validates the final PNG, prints its output path and stops the server. Original `dt-models` payloads and active catalogs are not selected for inference; the installed/default runtime is unchanged. Generated images, configs, logs and acceptance metadata remain together under the printed per-run directory, normally in `output/`. Automatic tags include nanoseconds; existing explicit tags are rejected unless a different output directory is selected.

Options include `--model sft-i4x|sft-q8p|base-q8p`, `--width 768 --height 512`, `--steps`, `--guidance-scale`, repeatable `--lora FILE[:WEIGHT[:MODE]]`, `--timeout-sec`, `--seed`, `--negative-prompt`, `--port`, `--tag`, `--output-dir`, `--metrics` and `--dry-run`. Explicit sampling options override preset defaults regardless of order; `--lora` replaces its default adapter list. The preset cannot combine with `--model`, `--bundle` or `--grpc-bin`; use those generic options without a preset for a custom setup. Dimensions must be multiples of 64. Use `--dry-run` to prepare configs without starting a server, or `--check` for native catalog validation without generation. See `bash tools/dt_image.sh --help` for the complete interface.

Choose an exact new output directory with `--output-dir`:

```bash
bash tools/dt_image.sh --preset mosskr2-lora4 \
	--guidance-scale 1 --steps 4 --output-dir "./output/my teapot" \
	"An orange ceramic teapot on a white table."
```

This writes `image.png`, the config, logs and metadata directly into `./output/my teapot/`. Relative paths are resolved from your current directory; absolute paths work too, including outside the workspace. Parent directories are created, but the selected run directory must not already exist, even if empty; files and symlink destinations are rejected rather than overwritten. `--output-dir` replaces the default `output/q6p_canary_<tag>` location, not the image filename, and does not append the tag. The preset's progress log follows the selected directory. Without this option, existing output behavior is unchanged.

### Persistent Image Sessions

Keep one guarded server running for repeated requests with the prepared `mosskr2-lora4` preset:

```bash
bash tools/dt_image.sh --preset mosskr2-lora4 --session start
bash tools/dt_image.sh --preset mosskr2-lora4 --session use \
	--output-dir ./output/bard-session \
	"A photo of a medieval bard in a medieval town playing lute."
bash tools/dt_image.sh --preset mosskr2-lora4 --session use \
	"A red bicycle with a blue helmet on its seat, studio photograph."
bash tools/dt_image.sh --session status
bash tools/dt_image.sh --session stop
```

`start` verifies the full bundle once and checks native model selection; it does not render an image. `use` keeps the server alive and accepts normal prompt/sampling/output options, including `--width 384 --height 704`. Changed bundle files, catalogs or pinned runtime inputs require a stop/restart and fresh full verification. Requests are serialized: another request or lifecycle operation is rejected as busy rather than queued. Status remains available during generation. A failed/timed-out request stops the session's own server; start a new session before retrying. Use the same `--port N` on every command when overriding port 7862. Calls without `--session` retain the one-shot behavior.

Three different 512x512/four-step prompts on one server took **18.085, 11.876 and 13.010 seconds** for request preparation, generation and validation, excluding session startup/full verification. A fresh session with `--weights-cache 4` took **13.658, 12.641 and 12.037 seconds**, without a meaningful warm benefit; cache zero remains the default. Matched prompts produced byte-identical configs, raw images and PNGs. These are six local observations, not a controlled cold-disk or general performance benchmark.

This reuses the process/CUDA context and avoids repeated full-bundle hashing, **not full MOSS model retention in VRAM**. Idle VRAM was 171 MiB in both sessions. Native `--weights-cache` is a CPU RAM cache on this dGPU, not a GPU residency switch; its explicit session-start bound is 0..8 integer GiB. Keep zero unless a measured profile benefits. Session state and shared logs are under `output/image-session-<port>/`; each image run also has strict acceptance, telemetry, `session-request.json` and a request-local `session-server.log` excerpt. Existing prepared local guard/runtime/model dependencies still apply. See the [session measurements and limits](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#persistent-image-sessions-november-2).

`--mode gpu` omits forced CPU weight offload; it does not mean zero CPU use or guarantee no automatic host-memory use. Tokenization, file loading and orchestration still use the CPU, and upstream policy also depends on available host RAM. On this 12 GiB RTX A3000 devcontainer, SFT i4x passed full 50-step GPU-mode tests at both sizes:

| Size | Time Including Startup | Sampled Whole-Device Peak |
|---|---|---|
| 512x512 | 57.547 s | 8,133 MiB |
| 1024x1024 | 190.258 s | 8,133 MiB |

Both outputs were finite and visibly prompt-following, with complete streams/chunks and post-Echo; no telemetry readings were missing. These are individual observations, not controlled performance benchmarks or GPU-mode certification for q8p variants. Offload remains the conservative default. Evidence: `output/upgrades/moss-image-stateful-20261027/image-helper-gpu-i4x-50steps-20261028-summary.json` and the corresponding `image_helper_gpu_i4x_*_50steps_20261028` canaries. Wider model/quality limits remain in the [MOSS integration plan](docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md).

### Custom MOSS conversion (experimental)

Base and SFT use the same supported MOSS architecture and native backend. Official SFT q8p and i4x already pass full image tests; SFT does not require a separate converter, tokenizer or inference patch. A custom full-precision SFT or weight merge must still pass source inspection and its own image acceptance test. See the [SFT results](DRAW_THINGS_PATCH/cuda/README.md#base-repeatability-and-sft-q8p-bring-up-october-27).

Use full-precision BF16/F16/F32 safetensors from the supported 42-layer MOSS Image architecture, either a single file or an explicit sharded index. Quantized inputs, incomplete architectures and incompatible rotary frequencies are rejected. Adjacent `config.json` is checked when present; `--config` selects it explicitly.

Main-layer attention weights may be separate Q/K/V tensors or a fused `layers.N.proj.qkv.weight` tensor with Q, K and V stacked along the output rows. Fused tensors use the same validated split and Q/K rotary conversion as the existing text blocks; mixed fused/separate keys in one layer and incompatible shapes are rejected. A filename containing `Turbo` does not establish sampling settings or certify distilled-model quality.

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

`--check` verifies bundle integrity and the model's decoded native catalog entry, then exits without generation. Full-precision bundles require `--mode offload`; strictly verified i4x bundles also allow explicit `--mode gpu`. This removes forced offload, not all host-memory use. The explicit `--grpc-bin` selects the patched private candidate, or an executable resource guard, without changing the helper's existing default or installing anything. Build the current profile using the [CUDA integration instructions](DRAW_THINGS_PATCH/cuda/README.md#isolated-server-integration). Remove `--check` and provide an image prompt to request generation. Native loading, finiteness and prompt-following quality are separate gates.

**Custom i4x GPU-policy optimization, November 2:** matched 512x512/eight-step runs on MOSSkr2 reduced the observed sampling span from **11.5-12.5 s with offload to 8.51 s with GPU policy**, with byte-identical request, raw output and PNG across both repeats of each mode. The warmed offload request took **49.599 s end to end**, versus **26.005-27.002 s** for GPU mode. GPU-policy peaks were 6,103-6,111 MiB. The existing four-step LoRA also passes GPU mode at **8,133 MiB**, with byte-identical output to its accepted offload run. These are short local measurements, not 50-step or other-resolution performance certification. Offload remains the default; no automatic VRAM-fit selection or fallback is implemented. Use the [guarded command and measurement limits](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#gpu-preferred-custom-i4x-november-2), especially with LoRAs or different geometry.

**Original-source end-to-end bring-up now passes.** On 2026-10-29, our conversion of the pinned original BF16 source completed a 512x512, 50-step DDIMTrailing request in **289.534 seconds including startup/cleanup** on the RTX A3000 Laptop GPU with CPU offload. The viewed image follows the orange-teapot, white-table, green-plant and window-light prompt. Native `moss_image` selection, one finite final image, complete stream/chunks and post-Echo all pass. Local result: `output/q6p_canary_moss_original_raw_actual_managed_staging_50steps_20261029_a/image.png`.

The fixes cover native raw-trailer descriptors, unsupported managed prefetch, and bounded dense-GEMM staging selected from actual CUDA allocation attributes. Observed managed denoiser weights lacked the CCV mapped flag, so the earlier flag-based staging experiment missed them. The accepted candidate is SHA-256 `acafef3d7f8f87588355cca88c41b21ad975665650271e1e23e53e0e0a2bf12b`; all 95 CUDA-enabled tests and eight linked-runtime gates pass. Its two-step output is byte-identical to the preserved guard-only control. Full-run sampled peak was 2,305 MiB and maximum temperature 66 C; timing is an individual observation, not a benchmark. Bundle verification and all nine protected hashes pass after inference. Installed defaults, active catalogs, canonical weights, precision and threading remain unchanged.

**A custom merged SFT checkpoint also passes.** The local `dt-models/mosskr2_image_sft_bf16_v1.safetensors` matches the existing source contract and converts with the same F16/F32 policy, without merge-specific code changes. After an interrupted session/restart, the completed private bundle verified intact; both guarded two-step and standard 50-step requests pass. The full 512x512 image visibly follows the teapot/table/plant/window prompt and completed in **284.303 seconds including startup/cleanup**. No Turbo/DMD extras or additional LoRA were used. See the [merged-SFT result and repeat command](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#custom-merged-sft-result-october-29).

**Our custom i4x quantization also passes.** `dt-models/mosskr2_image_sft_v1_ourconvert_i4x.ckpt` is **3.03 GB**, 70.3% smaller than the 10.19 GB F16 conversion. Its 308 Q4_K/Hadamard-256 matrices load in the unchanged patched server, and all 216 preserved tensors match their source bytes exactly. Guarded two-step and full 512x512/50-step requests pass; the viewed full image follows the same teapot prompt in **112.769 seconds including startup/cleanup**, with sampled peak 2,305 MiB GPU memory. All 98 regression tests, post-inference bundle integrity and nine protected hashes pass. These timings are individual observations, not a controlled benchmark. See the [custom i4x result and repeat command](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#custom-merged-sft-i4x-result-october-29).

**MoodyMOSS full-precision and custom i4x conversions also pass.** The local `dt-models/moodymossv10turbobf16.safetensors` contains 42 fused main-layer QKV weights. The source-layout adapter converts its 378 BF16 tensors into the same 524 native tensors, preserving the original source. Both F16 and genuine i4x pass guarded two-step and full 512x512/50-step requests with viewed prompt-following images. The F16 full render took **255.036 seconds including startup/cleanup**; the **3.03 GB i4x** conversion, 70.3% smaller, took **141.988 seconds**, with sampled peak 2,305 MiB GPU memory. All 216 sensitive tensors match their F16 source bytes exactly; all 100 regression tests, post-inference bundle checks and nine protected hashes pass. Timings are individual observations, not a controlled benchmark. See the [MoodyMOSS i4x result and repeat command](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#moodymoss-i4x-result-october-29).

**Four-step MOSS LoRA inference works with MOSSkr2 i4x.** The iOS-imported `moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16_lora_f16.ckpt` passes a guarded 512x512, four-step DDIMTrailing/CFG 1/shift 5 request at LoRA weight 1.0 in **49.452 seconds including startup/cleanup**. Its viewed image is clean and prompt-following; the same-seed no-LoRA control is visibly less resolved. All 252 source modules pass the existing numerical audit against the imported checkpoint. A separate source-only reconstruction also renders correctly in **41.153 seconds**, demonstrating local conversion feasibility, not a general converter CLI. Its weights differ only in FP16 subnormal rounding from the iOS import, and its image is not pixel-identical. Only the private test catalog is changed; post-pause source, bundle and protected-file checks pass. See the [LoRA study and repeat command](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md#mosskr2-i4x-four-step-lora-october-30).

These runs verify the original source, MOSSkr2 and MoodyMOSS, including our own i4x conversions of both custom checkpoints, plus this specific four-step LoRA on MOSSkr2 i4x, at one prompt/profile, not arbitrary custom checkpoints/adapters, wider prompt/resolution coverage, framework parity, guaranteed full GPU residency or broad quantization-quality parity. Other compatible checkpoints and LoRAs need their own acceptance runs. General creator quality-improvement and low-step Turbo claims remain untested beyond the specific adapter result above. Failed probes and the earlier malformed-bundle/time-out results remain recorded in the [MOSS handover](docs/MOSS_CONVERSION_HANDOVER_2026-10-28.md). Client progress remains unbuffered so timeouts retain received signposts.

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
