# MOSS Conversion Handover

Date: 2026-10-28. This is the current handover, not a claim that custom MOSS inference is complete.

## User Pause And WIP Checkpoint

The user requested pausing for today and committing/pushing the work. Do not resume native testing automatically after that checkpoint. The document was created after published commit `fc26076` and is included in this WIP checkpoint.

The investigation identified a concrete packaging defect, not a threading or precision fix:

- Raw external codec `0x10000000` expects a 16-byte little-endian `<QQ>` descriptor: relative offset, byte length. Only palettized descriptors use `<IIQQ>` with a block-size/padding prefix.
- The converter and Python reader incorrectly agreed on the 24-byte palettized layout for raw tensors. The pinned native C reader passes the descriptor unchanged; Swift `decodeWithExternalEagerFile` reads offset at byte 0, and `decodeWithExternalStore` reads length at byte 8. Old bundles therefore load from relative offset zero, or request a potentially huge length equal to the intended payload offset.
- Bounded parent-launched LLDB samples initially caught `cufileread` / `cudaMemcpy` / the WSL driver. A separate read-only timing trace completed all 234 logged copies (6,508,057,600 bytes) in a summed 7.798564 seconds, so those samples did not establish the sustained bottleneck.
- Targeted post-encoding sampling reached libc `read` at file address `0x11496c`. A full stack located `DynamicGraph._Store.loadBytes` through `decodeWithExternalEagerFile`, `Store.read` and `UNetFixedEncoder`. The selected thread accumulated approximately 99% of one core. This supports the oversized-read diagnosis; it is not post-fix performance certification.
- Fixed raw packaging, validation and reading in the toolkit. A new independent descriptor test first failed `24 != 16`, then passed. Raw sidecar coverage now rejects the malformed layout while retaining 24-byte palettized handling. All 66 CPU-focused tests pass in 6.694 seconds.
- Repaired all 524 descriptors in the existing private bundle without a new model copy. Only its 114,688-byte metadata region and manifest changed; the 10,186,162,176-byte denoiser file size and complete trailer payload-region hash are unchanged. Metadata/manifest backups and rollback support were used.
- **No native inference has run on the repaired bundle.** The old two-step finite image used malformed storage and is not correct-weight loading acceptance. Full inference, prompt-following quality and resolution of the timeout remain unverified.

Current repaired denoiser SHA-256: `fa67bb5fd28c1a182567bc4f1bfd476902bd11b061b1be355fb4d60343ce2fef`.

Unchanged complete trailer payload-region SHA-256: `beffb14ab13fb4a2261ffc03e008bccf685e7e1ae08968f8801a581eab94d22a`.

New evidence under `output/upgrades/moss-image-stateful-20261027/`:

- `raw-descriptor-cpu-regression-20261028.log`: 66 CPU checks pass.
- `raw-descriptor-repair-20261028-a/report.json`: old/new file identities, identical payload hashes, 524-tensor validation.
- `raw-descriptor-repair-20261028-a/before-metadata.bin` and `before-bundle.json`: small pre-repair backups; not runnable model copies.
- `native-hot-profile-20261028-c/profile.json`: three useful LLDB samples, fourth halt timed out; report records incomplete capture and cleanup request.
- `copy-timing-20261028-a-run.log` and `output/q6p_canary_moss_original_copy_timing_20261028_a/server.log`: read-only transfer timing control.
- `output/q6p_canary_moss_original_targeted_cpu_profile_20261028_b/server.log`: delivered full post-encoding read stack.
- `output/q6p_canary_moss_original_targeted_pc_profile_20261028_a/{server.log,native-maps.txt}`: delivered read PCs and mappings for symbol resolution.

All those diagnostic runs were deliberately bounded, used the existing private bundle, and ended without an accepted image. Failed/incomplete probes remain saved. Profiling helpers and payloads are ignored local artifacts, not included in the commit. The private resource guard now has optional targeted profiling; leave `DT_SAMPLE_CPU_STACKS`, `DT_TARGETED_PROFILE_CLIENT_LOG` and `LD_PRELOAD` unset for acceptance runs. No native runtime/source patch, rebuild, installation, global ptrace change, precision change or worker-count change was made.

## Resume Here

The goal is easy, correct conversion of a custom full-precision MOSS Image model, or the original BF16 model, into Draw Things format and successful prompt-following inference on Linux/CUDA.

Conversion and private bundle tooling are implemented. The raw-descriptor defect is corrected and the existing private bundle metadata is repaired, but native acceptance has not been rerun. After the user resumes work, start with a short guarded native check of the repaired bundle before considering a bounded 50-step quality control. Do not repeat profiling or the long render merely to reproduce the superseded state.

Pre-investigation published checkpoint:

- Commit: `fc26076c09cf90106b11637be4eb3e95784235d9`.
- Subject: `feat: add experimental MOSS conversion and bundle tooling`.
- Pushed to `origin/main`; local and remote tips matched and the worktree was clean.
- This handover was added locally afterward; it is not part of `fc26076`, but is included in the subsequent WIP checkpoint.
- No native jobs were left running at the checkpoint; GPU memory/use were zero.
- Old terminal IDs and PIDs in previous chat history are not active jobs to poll.

The next chat should inspect current git/process/GPU state, then read saved evidence. Do not start with another long render, download, rebuild or model copy.

## Acceptance Status

| Area | Verified Result |
| --- | --- |
| Original BF16 source | Pinned download, source manifest and hashes verified |
| Conversion | 462 learned source tensors to 524 native tensors; exact native metadata contract |
| Precision | F16 except the timestep-output projection weight/bias, preserved as F32 from original values |
| Private bundle | Raw descriptors repaired to native format; independent payloads and pinned companions verified; post-repair native check pending |
| Native model selection | Decoded server metadata confirms `moss_image`, rather than merely finding a filename |
| Pre-repair two-step request | One finite 512x512 noisy image, complete stream; malformed raw descriptors invalidate correct-weight loading acceptance |
| Post-repair native/quality | Not run before user pause; no prompt-following acceptance |
| Pre-repair full 50-step request | Timed out at 1,800 seconds; post-request Echo passed; no accepted final image |
| Pre-repair short stage probe | Reached `textEncoded`, then `imageEncoded`; no sampling signpost within 60 seconds |
| CPU-focused tests | 66 passed in 6.694 seconds after descriptor correction |
| Preservation | Earlier 24-file hash baseline retained; metadata-only repair verifies identical full trailer payload hash and current bundle integrity |
| Full current CUDA suite | Not rerun/certified for this increment |

**Do not equate finite output, schema validity, a live CPU worker, or a passing Echo with correct image generation.** The read-path/descriptor defect is identified; post-fix loading, performance and full custom-model quality remain unverified.

## Constraints

- No subagents or delegated execution. Work directly in this workspace.
- Default native builds to 16 jobs. Build parallelism does not control inference parallelism.
- Preserve installed/default runtime, canonical models, active catalogs and the intentionally dirty legacy CPU checkout/tools.
- Do not reset unrelated changes or modify the old CPU converter checkout to enable CUDA.
- Download weights into `dt-models`, not new diagnostic output trees.
- Inference must use independent writable regular-file copies, never hardlinks or symlinks to canonical models. Reflinks are allowed.
- Reuse the current private trailer bundle. Avoid additional full-sized copies or downloads.
- Only one native/GPU workload at a time. Do not overlap inference with CUDA tests, large hashes, conversion or model copying.
- Any native fix must follow [the patch workflow](../DRAW_THINGS_PATCH/README.md) and [the CUDA profile](../DRAW_THINGS_PATCH/cuda/README.md): pinned snapshots, manifest, generated unified diff, strict apply and idempotence verification.
- Keep the deferred old LTX null-input GEMM crash out of this task's prerequisites.
- Do not install Linux NVIDIA driver/kernel/display packages in WSL.
- Do not change global Yama/ptrace settings or try privileged attach workarounds.
- Do not commit/push further work unless requested. The last commit/push authorization was completed.

## Environment

Workspace: `/workspaces/drawthings-linux-toolkit`, Ubuntu 22.04.5 devcontainer on Windows/WSL.

| Component | Current Value |
| --- | --- |
| GPU | NVIDIA RTX A3000 Laptop, 12,288 MiB, SM86 |
| Driver | 595.95 |
| Host RAM/swap | About 29 GiB / 8 GiB |
| Allowed logical CPUs | 20 in the measured native process |
| Python | `.venv/bin/python`, Python 3.10.12 |
| Torch | 2.10.0+cu128 |
| Swift | 6.0.3 |
| Bazel | 7.4.1 |
| CUDA | 12.8 at `/usr/local/cuda-12.8` |
| LLVM/Clang | 14 |

CUDA execution needs the WSL driver libraries preserved:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
```

Local weights, builds, caches and diagnostic reports are ignored artifacts, not included in the pushed commit. This handover assumes the existing workspace; a fresh clone alone will not contain them.

## Runtime Boundaries

Patched private candidate:

```text
output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI
SHA-256: 71878a22c08fe7a38a55946d993b35771406c6d13501824e516bfca26c35dcaa
```

Pins:

- Community: `0621d8198f3ed505bb7cefc71b84f8f0e1ce5936`.
- s4nnc: `b5476a77e7f67f12729872e4b9f1a649b3cd3df0`.
- CCV: `69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8`.
- Private CCV: `output/upgrades/ccv-69b1ed1f-qwen35-cuda-forward-safety-v2`.

Installed defaults remain `/usr/local/bin/gRPCServerCLI` and `/usr/local/bin/drawthings-grpc`; do not promote the candidate over them.

Legacy CPU binaries remain under `draw-things-community/.build/release/`: `model-converter`, `model-quantizer`, `lora-converter`. Preserve that checkout and its existing changes.

Official Base/SFT q8p and SFT i4x already have working native controls in earlier commits. The default [image helper](../tools/dt_image.sh) uses the independent five-model pool at `output/upgrades/moss-image-stateful-20261027/models`, not canonical weights. Do not delete that pool as obsolete.

## Code Owners

| File | Role |
| --- | --- |
| [dt_moss_convert.py](../tools/dt_moss_convert.py) | Source/config preflight, conversion CLI, offline native trailer preparation, private bundle publication/verification |
| [dt_keymap_compiler.py](../tools/dt_keymap_compiler.py) | `build_moss_image_contract`: supported architecture, native names/shapes, transforms and precision policy |
| [dt_direct_ckpt_builder.py](../tools/dt_direct_ckpt_builder.py) | Existing strict staged/atomic source-to-checkpoint writer |
| [seek_io.py](../tools/dt_weellm/seek_io.py) | Single/sharded safetensors reading and source ownership validation |
| [sqlite_store.py](../tools/dt_weellm/sqlite_store.py) | SQLite/split/sidecar reader and raw native trailer support; trailers refuse writable opens |
| [dt_image.sh](../tools/dt_image.sh) | `--bundle`, offload-only policy for unquantized bundles, `--check` metadata-only mode |
| [run_q6p_canary_once.sh](../tools/run_q6p_canary_once.sh) | Owned server/request lifecycle, decoded model gate, strict image acceptance, unbuffered client progress |
| [test_dt_weellm.py](../tools/test_dt_weellm.py) | Conversion, source, storage, bundle and precision coverage |
| [test_dt_make_config.py](../tools/test_dt_make_config.py) | Helper, sequence, decoded-model preflight and timeout-progress/cleanup coverage |

The public [README](../README.md) has the current conversion commands and experimental acceptance limits. Native server Echo handling is in the pinned candidate's `Libraries/GRPC/Server/Sources/ImageGenerationServiceImpl.swift`. Signpost emissions are in `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`; fixed MOSS model loading is controlled by `UNetFixedEncoder.swift`. The confirmed decoder contract is in pinned s4nnc `nnc/Store.swift`, not the original toolkit round-trip test.

## Model Assets

Original transformer only, not the original text encoder/VAE:

```text
Repository: OpenMOSS-Team/MOSS-Image-Base
Revision:   85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
Root:       dt-models/moss-image-base-source-85975793/
Manifest:   source-manifest.json
Index:      transformer/diffusion_pytorch_model.safetensors.index.json
Config:     transformer/config.json
Shards:     transformer/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
```

Supported architecture: dim 3072, FFN 8192, frequency dim 256, latent channels 128, 24 heads, 42 layers, text dim 4096, RoPE base 10000. Learned values are BF16 in this original source. `rope.bands` is allowed only after validating shape, finiteness and numerical agreement with native frequencies.

Canonical converted checkpoint:

```text
dt-models/moss_image_base_ourconvert_f16.ckpt
Bytes:   10194063360
SHA-256: 859b525cad201c4e6a79f98cdf75ccf1287d83c7668c0897deccb08651d7ef92
```

Current valid private bundle:

```text
output/moss-models/base-original-85975793-trailer/
  bundle.json
  models/custom.json
  models/moss_image_base_ourconvert_f16.ckpt
  models/qwen_3.5_9b_i5x.ckpt
  models/flux_2_vae_f16.ckpt

Packaged denoiser bytes:   10186162176
Packaged denoiser SHA-256: fa67bb5fd28c1a182567bc4f1bfd476902bd11b061b1be355fb4d60343ce2fef
```

Companion hashes:

- Qwen: `559d41f5e6721b4edb5a661c42b6f328a243876c2637cad36cf9937fb0d453c1`.
- VAE: `48369d4b1495aec3df0579591f4bf9c199a73cc288130df2359c4a173c5ae864`.

The canonical inline checkpoint and packaged trailer have different file hashes by design. The old malformed trailer file hash was `d406fe647fd3f70c6225d824ad0fb7f1797df8cdc1a926610bcca04eecbfb9b4`; the corrected hash above differs only because metadata changed. Packaging and repair preserve tensor payload bytes; the manifest records source and packaged identities separately. Original learned values were not repaired using official converted weights. An actual-source check verified all 3,072 timestep-bias values exactly in prepared F32, but that is not complete forward parity.

## Resolved Traps

1. The first bundle catalog omitted required `upcast_attention`. Native Codable silently dropped the specification; a finite noisy image from that attempt was not valid MOSS acceptance. Current catalog generation/validation requires `upcast_attention: false`, `version: moss_image`, the pinned companions and objective `{"u":{"condition_scale":1000}}`. The authoritative gate decodes `Echo.override.models`; `Echo.files` alone is insufficient.
2. The corrected inline bundle faulted before generation. Native loading had rewritten the private checkpoint into about 1 MiB SQLite metadata plus a roughly 10 GB `-tensordata` sidecar. Sampled GPU peak was only 115 MiB, no cgroup OOM kill was recorded, and the kernel recorded SIGSEGV. This does not establish the cause of the earlier whole-host restart, and does not prove a GPU OOM.
3. Offline raw native trailer packaging bypasses `TensorData.makeExternalData` rewriting. The original package produced a finite two-step image without a sidecar, but used incorrect 24-byte raw descriptors. That is not correct-weight loading acceptance. Current raw descriptors are corrected; native validation remains pending.
4. Client progress previously buffered to redirected stdout. Both timed and untimed generation launches now use Python `-u`; a one-second fixture verifies that received signposts survive timeout and owned-server cleanup. The old long-run four-line client log cannot establish which steps actually completed.

Native trailer details: SQLite header bytes 60:64 contain the big-endian UInt32 trailer base; base/payload alignment is 16,384. Raw external codec is `0x10000000`; its descriptor is little-endian `<QQ>`: relative offset, byte length. Palettized external codecs use `<IIQQ>`: block size, padding, relative offset, byte length. Confusing these layouts caused the defect. Source of truth is pinned s4nnc `nnc/Store.swift` and its unchanged CCV decode callback boundary.

## CPU Evidence

The long-run process had 20 allowed logical CPUs, cgroup `cpu.max = max 100000`, and no inherited OMP/OpenBLAS/MKL/BLIS thread-limit settings. It had 12 native threads; one worker accumulated about 96% of total CPU time. Approximately 101% in `top` therefore meant one core's worth of CPU, not a one-core container limit. Roughly 16 GiB resident memory matched the user's observation.

Loaded CPU libraries included `libomp.so.5` and `libcblas.so.3.10.3`. Their presence does not prove that setting `OMP_NUM_THREADS=16` would help. The later bounded profiles identified the native raw read path and malformed descriptors described above; no BLAS/threading workaround was applied. The remaining post-fix behavior is unverified.

A bounded live LLDB attach was rejected by Yama `ptrace_scope=1`. No privileged workaround was attempted. If a debugger is needed, prefer launching the owned candidate as its child through the existing parent-launch approach, with an explicit time/stop budget, detach/kill handling and final GPU/process cleanup. Do not leave an orphan inferior behind.

## Saved Evidence

Common report root: `output/upgrades/moss-image-stateful-20261027/`.

| Artifact Relative to That Root | Meaning |
| --- | --- |
| `moss-original-bf16-conversion-20261028.json` | Original conversion report |
| `moss-original-trailer-bundle-20261028.json` | Current bundle preparation report |
| `moss-original-trailer-guarded-2steps-20261028.log` | Successful strict two-step request |
| `moss-original-trailer-guarded-50steps-20261028.log` | Full request: timeout 124 at 1,800 s, post-Echo 0 |
| `moss-original-trailer-50step-progress-probe60s-20261028.log` | Deliberately bounded diagnostic: timeout 124 at 60 s, post-Echo 0 |
| `moss-conversion-cpu-regression-final-20261028.log` | 65 tests passed |
| `moss-conversion-preservation-after-cleanup-20261028.json` | Protected/source/canonical/native hash and bundle checks |
| `cleanup-evidence-20261028/deletion-manifest.json` | Approved deletions, pre-deletion hashes and retained native metadata |
| `guard-moss-converted-server.py` / `.sh` | Private resource-guarded native launcher |

Per-request directories hold `server.log`, `client.log`, `gpu.csv`, decoded model selection and any accepted image artifacts:

```text
output/q6p_canary_moss_original_trailer_guarded_2steps_20261028/
output/q6p_canary_moss_original_trailer_guarded_50steps_20261028/
output/q6p_canary_moss_original_trailer_50step_progress_probe60s_20261028/
```

Only the two-step directory has an accepted final image. Read the bounded probe's `client.log` for the actual encoding signposts.

## Resource Guard

The private guard requires `--cpu-offload`, owns its child with parent-death cleanup, and samples GPU/host memory every 250 ms. Defaults are `DT_VALIDATION_MAX_GPU_MIB=9216` and `DT_VALIDATION_MIN_HOST_MIB=4096`. It fails closed if telemetry is unavailable and logs resource aborts and native exit status.

This sampled guard cannot prevent every abrupt allocation, driver or host failure. **The ordinary image helper does not automatically use this private guard.** For experimental native diagnostics, explicitly select the guarded launcher through `run_q6p_canary_once.sh --grpc-bin ...`, or use an equivalently bounded owned debugger launch.

For reproducing the last short stage probe only when needed, the relevant runner arguments were:

```text
--grpc-bin output/upgrades/moss-image-stateful-20261027/guard-moss-converted-server.sh
--proto output/upgrades/community-0621d819-qwen35-cuda/Libraries/GRPC/Models/Sources/imageService/imageService.proto
--cache-dir output/upgrades/schemas/0621d819
--bindings-dir output/upgrades/schemas/0621d819/flatbuffers
--models-dir output/moss-models/base-original-85975793-trailer/models
--model moss_image_base_ourconvert_f16.ckpt
--require-model-version moss_image
--require-model-file qwen_3.5_9b_i5x.ckpt
--require-model-file flux_2_vae_f16.ckpt
--host 127.0.0.1:7862 --image-mode --width 512 --height 512
--steps 50 --sampler DDIMTrailing --guidance-scale 3 --shift 5 --seed 4242
--prompt 'An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph.'
--negative-prompt '' --server-cpu-offload --server-fread-preferred
--server-weights-cache 0 --gpu-metrics --final-mode --timeout-sec 60
--tag <new-unique-tag>
```

These are arguments, not a request to immediately rerun them. Confirm current resource/port/process state first and preserve the schema path choices. Existing tags must not be overwritten. `--bundle-model` integrity verification hashes about 19 GB; do not run it concurrently with inference.

## Disk State

The user reported Windows host free space falling roughly from 120 GB to 50 GB. Linux free space is a different measurement.

The user approved deleting only payloads from `output/moss-models/base-original-85975793` and `output/moss-models/base-original-85975793-v2`. Seven payload files totaling 38,205,521,920 bytes were removed; catalogs, manifests, run evidence and about 1.1 MiB native metadata were retained. Those two old bundles are intentionally not runnable.

The one replacement bundle contains 19,100,000,256 payload bytes. Net reclaimed model-payload space is 19,105,521,664 bytes, about 17.8 GiB. Last Linux free space was about 330 GiB. Deletion frees reusable Linux blocks but does not guarantee immediate Windows VHD shrinkage; the Windows disk is not mounted/accessibly measurable here. No other pool/cache deletion or Docker reset is authorized.

## Next Investigation

1. Wait for the user to resume. Confirm current worktree, candidate identity, no owned native jobs and idle GPU; do not treat historical PIDs as live jobs.
2. Read the pause checkpoint and metadata-repair report. Reuse the repaired bundle with denoiser hash `fa67bb5f...`; do not recreate/download/copy model payloads. Leave diagnostic interposition/profiling variables unset.
3. Run a short resource-guarded native two-step check using the corrected descriptors and existing schema/companion choices. Require decoded `moss_image`, complete finite output, cleanup and bundle integrity. Two-step noise alone is not quality acceptance.
4. If that passes without a new blocker, run one explicitly bounded 50-step quality control. Inspect the actual image for the prompt, not just acceptance JSON; distinguish loading, completion, practical timing and image quality.
5. If a local failure remains, form one nearby falsifiable hypothesis and immediately validate a small fix. Do not blindly change threading, precision or transforms. Any actual native implementation change must follow the pinned patch workflow.
6. Recheck protected resources and private payload integrity after native cleanup; record actual results before claiming inference accepted.

Success means a coherent prompt-following image from our original/custom source conversion, practical completion, finite/complete output, protected resources unchanged, and honest separation of what was tested from what remains open. Custom q8p/i4x quantization is later work, not implemented/certified here.

## CPU Verification

These are the last passing CPU-focused checks, not the full native/CUDA suite:

```bash
env -u DT_RUN_CUDA_TESTS .venv/bin/python -m unittest \
  tools.test_dt_weellm.CheckpointSafetyTests \
  tools.test_dt_weellm.WeeLLMTransformsTests \
  tools.test_dt_weellm.SeekerAndDequantTests \
  tools.test_dt_weellm.MossConversionBundleTests \
  tools.test_dt_weellm.ContractAndAuditorTests \
  tools.test_dt_weellm.DirectBuilderAndLoRABridgeTests \
  tools.test_dt_make_config tools.test_dt_tensor_to_playable -v
bash -n tools/dt_image.sh tools/run_q6p_canary_once.sh
git diff --check
```

Run only checks needed for the next change; do not rerun native gates merely to reconfirm this handover. More historical context is in the [upgrade/MOSS plan](UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md) and the [hybrid tooling plan](WEELLM_HYBRID_TOOLING_PLAN_2026-10-24.md); distinguish their older phases from this current checkpoint.
