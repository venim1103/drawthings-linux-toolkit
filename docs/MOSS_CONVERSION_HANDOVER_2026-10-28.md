# MOSS Conversion Handover

Created: 2026-10-28. Updated: 2026-10-29. Original-source and one custom merged-SFT end-to-end test now pass, including our own i4x quantization of the merge; arbitrary custom checkpoints and broader quality coverage are not yet certified.

## Custom Merged SFT i4x Result: October 29

Our own genuine MOSS i4x encoder now produces a working artifact from the accepted full-precision MOSSkr2 conversion. Official SFT i4x inference worked previously; this is the first accepted custom MOSS i4x generation through our tooling. Legacy palettized `q4p` is not equivalent and was not renamed or used as an intermediate.

- Canonical checkpoint: `dt-models/mosskr2_image_sft_v1_ourconvert_i4x.ckpt`, **3,026,944,000 bytes** (3.03 GB), SHA-256 `f9faa5e8e31c93fb01f31e1c52cb3d2c23c9ede9ff37a97f19f3372a7f2f9690`. This is **70.3% smaller** than the 10,194,063,360-byte full-precision conversion.
- Private bundle: `output/moss-models/mosskr2-sft-v1-i4x`. Its native-trailer denoiser is 3,028,375,552 bytes, SHA-256 `21d082d809add6dc62675da857561638293b5a9c086087ef7fb330923f43d95f`, with independent writable Qwen/VAE companions and private catalog. Explicit manifest profile `quantization: i4x` enables strict packed-schema validation.
- Policy: 308 core matrices use native Q4_K with regular Hadamard-256, inline identifier `0x018a1eab` / trailer identifier `0x118a1eab`. All 216 sensitive F16/F32 tensors retain their own source bytes; actual post-inference comparisons confirm every one exactly. The eligible keyset matches official i4x metadata, but no official learned values were used. Unlike the official mixed-codec artifact, the remaining tensors are not palettized or ezm7-encoded.
- Encoder: [tools/dt_moss_quantize.py](../tools/dt_moss_quantize.py) and [native adapter](../tools/dt_moss_quantize.c), using hash-pinned CCV `69b1ed1f` CPU sources. The run used 16 independent matrix workers without changing inference threading. Full native decode is finite; independent regular-Hadamard checks reconstruct sampled first/last rows, with maximum relative RMS error 0.067031 across matrices. This sampled numerical error is not an image-quality score.
- Regression checks: nine focused codec/bundle tests pass in 2.815 seconds; the combined CUDA/native/helper suite passes all **98 tests in 70.461 seconds**. Extended API call-site checks pass. No native runtime/profile patch, installed/default promotion, active-catalog mutation or canonical full-precision replacement was needed.

Both native requests used the unchanged `acafef3d...` patched server, the existing local resource guard, CPU offload, `--fread-preferred`, weights cache zero, 512x512, DDIMTrailing/CFG 3/shift 5/seed 4242, an empty negative prompt and the same orange-teapot prompt as the full-precision baseline:

- Two steps: **54.933 seconds including startup/cleanup**, 480-second request cap; tag `mosskr2_i4x_2steps_20261029_a`. All 88 memory/temperature samples are present; peak 2,305 MiB, maximum 60 C; two utilization readings are unavailable.
- Full 50 steps: **112.769 seconds including startup/cleanup**, 900-second request cap; tag `mosskr2_i4x_50steps_20261029_a`. All 205 memory/temperature/utilization samples are present; peak 2,305 MiB, maximum 67 C. Config SHA-256 `4b6dbeb6a8e7043c07ede854b0573c8e7e2df6d015e1c185935a6cf24d92c649`.
- Both pass decoded native `moss_image` selection, one finite final 512x512 image, zero audio, complete stream/chunks and post-Echo. The viewed full image clearly shows an orange ceramic teapot on a white table beside a green plant, with bright window light. Final image: `output/q6p_canary_mosskr2_i4x_50steps_20261029_a/image.png`.
- Neither run triggered the resource guard. Owned launchers/native processes are gone, the port is free and GPU memory/use returned to zero. Post-run private bundle integrity, the BF16 source, canonical F16/i4x conversions, canonical companions, accepted server and all nine protected hashes pass unchanged.

This verifies one custom checkpoint's quantization and one prompt/profile, not lossless output parity, broad quality parity, arbitrary checkpoints/resolutions, GPU-only custom inference or the creator's Turbo/quality claims. Individual timings are not a controlled performance benchmark. The existing CUFILE diagnostic messages still appear during successful native reads; this increment makes no file-I/O runtime change or claim of log cleanliness.

To generate from the prepared bundle without registering it or changing defaults:

```bash
env -u LD_PRELOAD -u DT_SAMPLE_CPU_STACKS -u DT_TARGETED_PROFILE_CLIENT_LOG \
  DT_VALIDATION_NATIVE="$PWD/output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI" \
  bash tools/dt_image.sh --bundle output/moss-models/mosskr2-sft-v1-i4x \
    --grpc-bin output/upgrades/moss-image-stateful-20261027/guard-moss-converted-server.sh \
    --metrics 'An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph.'
```

The guard remains an ignored local validation artifact. The helper supplies the WSL/CUDA library path and defaults to the tested offload profile; its normal 3,600-second timeout differs from the explicit acceptance caps above. Use fresh output paths for another quantization; published artifacts are never overwritten. Confirm resources first and do not rerun merely to reproduce the accepted result.

Evidence under `output/upgrades/moss-image-stateful-20261027/`: `mosskr2-i4x-quantization-20261029.{json,log}`, `mosskr2-i4x-{2steps,50steps}-20261029.log`, `mosskr2-i4x-acceptance-20261029.json`, `mosskr2-i4x-preservation-20261029.json` and `moss-i4x-regression-20261029.log`. Model artifacts and diagnostic reports remain local and are not committed. Tooling checkpoint: `cc79c493cad98dc3442013c4f4b7dde42e246345`.

## Custom Merged SFT Result: October 29

The user supplied `dt-models/mosskr2_image_sft_bf16_v1.safetensors`, described by its creator as a MOSS SFT attention/MLP merge influenced by Krea2. This local file passes the unchanged supported source contract: 462 learned tensors, expected rotary buffer and 42-layer MOSS structure. Its SHA-256 is `ba2c3713c81aa446378b1b0a863ac28f167ceb8a1f27e3bcf96bc44fb3b51806`. No remote provenance or causal explanation of the graft was independently verified.

The existing converter writes `dt-models/mosskr2_image_sft_v1_ourconvert_f16.ckpt`: 524 native tensors, F16 except original-value F32 timestep-output projection weight/bias. The canonical conversion hash is `bda9cabe3388831bfeb8694b761a22c15c4c7f4c93153ad7a1ccfca67305a49f`; packaged trailer denoiser hash is `1cb64a3e8b9a2c7b50340d653a00cbe535148f4ec600476d8c078b91d6614699`. Private bundle: `output/moss-models/mosskr2-sft-v1`, with independent pinned Qwen/VAE companions. Preparation had completed before the user's sudden restart: the final conversion report and published manifest were intact, all bundle hashes verified, no native jobs remained and the GPU was idle. No repacking or replacement model copy was needed after restart.

Using the existing `acafef3d...` patched server and local resource guard, CPU offload, `--fread-preferred`, weights cache zero, 512x512, DDIMTrailing/CFG 3/shift 5/seed 4242 and the same teapot prompt:

- Two-step smoke test passes decoded native `moss_image` selection, one finite final image, zero audio, complete stream/chunks and post-Echo. Tag: `mosskr2_sft_2steps_after_restart_20261029_a`; 480-second request cap. All 319 memory/temperature samples are present, peak 2,305 MiB and maximum 61 C; five utilization readings are unavailable.
- Standard 50-step request passes all the same strict checks in **284.303 seconds wall time including startup/cleanup**, within a 900-second request cap. The viewed final image clearly shows the requested orange ceramic teapot on a white table beside a green plant, with bright window light. Tag: `mosskr2_sft_50steps_after_restart_20261029_a`. All 523 memory/temperature samples are present, peak 2,305 MiB and maximum 66 C; five utilization readings are unavailable. Config SHA-256: `4e7fd4264e6c63db0a58eeab20fafc70b4b9b37ecddecf60122e6e5d3bb55de4`.
- Neither run triggered the resource guard; both owned server processes are gone and GPU memory/use returned to zero. Post-run private bundle integrity, source hash, canonical conversion hash, accepted server hash and all nine protected resource hashes pass.

This checkpoint works through the existing converter/backend **without merge-specific patches, Turbo/DMD extras or additional LoRA**. Base and SFT share the supported architecture/backend; official SFT q8p/i4x already passed before this test, so SFT itself does not require a separate tokenizer or runtime. This one prompt/profile is not proof of the creator's quality claims, wider prompt/geometry reliability, reference-framework parity, 8-12-step Turbo use or custom low-bit conversion. Timing is an individual observation, not a benchmark. No native implementation or installed/default/catalog change was made for this merge.

To generate from the already prepared local bundle, using the existing ignored resource guard described below:

```bash
env -u LD_PRELOAD -u DT_SAMPLE_CPU_STACKS -u DT_TARGETED_PROFILE_CLIENT_LOG \
  DT_VALIDATION_NATIVE="$PWD/output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI" \
  bash tools/dt_image.sh --bundle output/moss-models/mosskr2-sft-v1 \
    --grpc-bin output/upgrades/moss-image-stateful-20261027/guard-moss-converted-server.sh \
    --metrics 'An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph.'
```

The helper verifies bundle integrity before starting, supplies the WSL/CUDA library path, defaults to the tested offload/512/50-step profile, and uses a fresh tag. Its normal timeout is 3,600 seconds; the acceptance requests above explicitly used the canary's shorter 480/900-second caps. Confirm resources first and do not rerun merely to reproduce the accepted result.

Evidence under `output/upgrades/moss-image-stateful-20261027/`: `mosskr2-conversion-20261029.json`, `mosskr2-source-before-20261029.sha256`, `mosskr2-{2steps,50steps}-after-restart-20261029.log`, `mosskr2-acceptance-after-restart-20261029.json` and `mosskr2-preservation-after-restart-20261029.json`. Both canary output directories retain config/model selection, raw final tensor, PNG, strict acceptance, client/server logs and telemetry. Final image: `output/q6p_canary_mosskr2_sft_50steps_after_restart_20261029_a/image.png`.

## Current Result: Original-Source Full Inference Passes

The original BF16 source, converted using our own converter and the repaired private bundle, now completes the standard **512x512, 50-step DDIMTrailing/CFG 3/shift 5/seed 4242** request. Wall time was **289.534 seconds including startup/cleanup**, within the unchanged 900-second request cap. The viewed final image clearly follows the orange ceramic teapot, white table, green plant and bright window-light prompt. Native `moss_image` selection, one finite final image, zero audio, complete stream/chunks and post-Echo pass.

The controlling evidence was a bounded twelve-call raw-GEMM capture: nine real CUDA-managed weights and three device weights, with **no CCV mapped flag on any wrapper**. CUDA allocation queries and pending-before/after checks all returned zero. Managed shapes included `[3072,3072]`, `[8192,3072]` and `[3072,8192]`. The earlier flag-only staging experiment therefore missed these observed hot managed weights; its timing pair was not a measurement of effective staging for them.

The maintained native GEMM path now selects staging from actual CUDA allocation attributes, only on devices without concurrent managed access and only for non-view raw F32/F16/BF16 weights at most 64 MiB. It uses a same-stream copy into existing workspace with unchanged arithmetic, no persistent weight cache and no whole-model VRAM residency. Larger/device/view/QX paths and supported-CMA behavior are unchanged. The earlier prefetch capability guard remains. No CUDA errors are cleared.

Current candidate:

- Binary: `output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI`.
- SHA-256: `acafef3d7f8f87588355cca88c41b21ad975665650271e1e23e53e0e0a2bf12b`.
- Private CCV: `output/upgrades/ccv-69b1ed1f-qwen35-cuda-actual-managed-staging`; existing isolated Bazel root reused after preserving the control.
- The prefetch-only `dc94ffc6...` control remains at `output/upgrades/gRPCServerCLI-0621d819-before-mapped-staging`; the rejected flag-only `c51a759e...` remains separately preserved. Original `71878a22...` candidate and installed defaults remain unchanged.
- Strict application/snapshot/idempotence verification passes. The 16-job incremental build took 14.100 seconds. All eight linked native execution gates pass; the expanded mapped-GEMM gate covers 36 exact-output cases including unflagged managed allocations, both orientations, optional bias and three precisions. Final combined CUDA-enabled suite: **95 tests passed in 74.597 seconds**.
- New two-step request: **184.610 seconds**, config/raw/PNG byte-identical to the preserved guard-only control. This blurred image alone is not quality acceptance.
- Full request: 51 sampling callbacks, 37 previews, 549 telemetry samples; sampled peak 2,305 MiB, maximum 66 C, no resource abort. All memory/temperature readings are present; two utilization readings are unavailable. Subsequent callback intervals are 1.501-3.504 seconds, median 2.4955, not isolated-kernel or certified step timings. These are individual observations, not a benchmark.
- After inference cleanup, bundle verification and all nine protected hashes pass. No model payload copies/downloads, installed/default promotion, catalog change, precision change or threading change occurred.

Local evidence:

- Final image and strict acceptance: `output/q6p_canary_moss_original_raw_actual_managed_staging_50steps_20261029_a/{image.png,image_acceptance.json,image_r0056_01.bin}`.
- Full config SHA-256: `24cac2045b2c902338a615cd0b8b26a45ed937481c199144afe4c9e8129d0b60`.
- Native gate report: `output/upgrades/native-backends-qwen35-actual-managed-staging.json`.
- Common evidence root: `output/upgrades/moss-image-stateful-20261027/`.
- Under that root: `raw-gemm-layout-20261029-hot/profile.json`, `actual-managed-staging-{2steps,50steps}-summary-20261029.json`, `actual-managed-staging-preservation-20261029.json`, and `actual-managed-staging-final-regression-20261029.log`.

`tools/dt_image.sh` now accepts an explicit executable `--grpc-bin`, preserving its existing default and offload behavior. Eight hermetic helper tests pass. This actual guarded preflight also passes decoded native model selection:

```bash
env -u LD_PRELOAD -u DT_SAMPLE_CPU_STACKS -u DT_TARGETED_PROFILE_CLIENT_LOG \
  DT_VALIDATION_NATIVE="$PWD/output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI" \
  bash tools/dt_image.sh --bundle output/moss-models/base-original-85975793-trailer \
    --grpc-bin output/upgrades/moss-image-stateful-20261027/guard-moss-converted-server.sh \
    --check 'Model selection check'
```

The guard is an existing ignored local safety launcher, not a newly installed runtime. The helper supplies the WSL/CUDA library path. Remove `--check` and supply an image prompt for generation; use a new automatic tag and confirm resources first. Do not rerun merely to reproduce the accepted result. Compatible custom checkpoints can use the same tooling, but need their own source validation and inference acceptance. Wider prompts/resolutions, framework parity, GPU-only custom inference and broader low-bit quality coverage remain open; the one custom i4x result above now passes. The sections below retain historical failed probes and checkpoints.

## Managed Prefetch Follow-up

The user explicitly authorized continued work and periodic commits/pushes. The earlier completed inference investigation was committed and pushed as `207047f`. The installed defaults, original `71878a22...` candidate and model payloads remain protected.

A bounded exact-output staging probe found approximately 6.4 ms for copy-plus-GEMM versus 50 ms for direct managed weights. Separately, `cudaMemPrefetchAsync` returned 101 and left error 101 pending on this concurrent-managed-access-zero GPU. Native dense GEMM already calls the mapped-tensor prefetch helper; that helper lacked the capability check used by memory advice. The extended existing `--execute-memory-advice` gate reproduces this error on the original candidate as `DT_MEMORY_ADVICE_EXECUTION=-84`.

The pinned compatibility snapshot now skips unsupported optional prefetch without clearing prior errors. The generated diff passes strict/idempotence verification. Its C gate checks managed-buffer contents, synchronization and preservation of an injected prior error; a regression test retains the original candidate as the negative control. The isolated 16-job build succeeded at `output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard`, with fresh CCV copy `output/upgrades/ccv-69b1ed1f-qwen35-cuda-prefetch-guard` and separate Bazel root `.cache/bazel-qwen35-cuda-prefetch-guard`. Candidate SHA-256: `dc94ffc6daa972d5e43073c57ba2c981bf8141b1946167554e16dd0398185fb2`. Build log: `output/upgrades/moss-image-stateful-20261027/prefetch-guard-build-20261029.log`.

All eight linked-runtime execution gates pass, including the new opt-in `--execute-mapped-gemm`: twelve exact-output dense GEMM cases covering ordinary/managed weights, F32/F16/BF16 and optional bias. Persistent report: `output/upgrades/native-backends-qwen35-prefetch-guard.json`. The full CUDA-enabled suite passes all 93 tests in 64.246 seconds; log `prefetch-guard-regression-20261029-pristine.log` under the common evidence root. Set `DT_CCV_SOURCE_DIR` to pristine pinned CCV for that suite, and `DT_GRPC_CANDIDATE` to the new executable. The first run used applied sources and failed only the pristine-base patch fixture; the focused corrected fixture and full rerun passed. All nine protected hashes are unchanged; report `prefetch-guard-preservation-20261029.json`.

The guard checkpoint was committed and pushed as `2077700`; local and remote tips matched. Do not treat this prefetch correctness guard as a certified raw-weight performance fix or force the entire unquantized model into VRAM.

### Bounded Native Staging Follow-up

The proposed staging experiment was implemented only for non-view raw F32/F16/BF16 weights no larger than 64 MiB on devices without concurrent managed access, using the existing stream workspace and unchanged arithmetic. A fresh private CCV copy, regenerated strict patch and 16-job incremental build produced `c51a759efc1d210f2f0c8c8c5ad15e3a26aa11485f7c22aacd24a691ab2004aa` in 50.191 seconds. All eight native gates passed, and all 93 CUDA-enabled tests passed in 66.826 seconds. The existing mapped-GEMM probe now covers 24 exact-output cases including both weight orientations; the accepted guard-only candidate also passed the expanded probe before the experiment.

Matched guarded two-step results using the same repaired bundle and config:

- Staging: **196.942 seconds**, tag `moss_original_raw_mapped_staging_2steps_20261029_a`; 368 complete GPU samples.
- Guard-only control: **127.516 seconds**, tag `moss_original_raw_prefetch_control_2steps_20261029_a`; 232 complete GPU samples.
- Both passed native model selection, finite 512x512 image, zero audio, complete stream/chunks and post-Echo. Config, final raw tensor and PNG are byte-identical. Both viewed images remain blurred orange/light-green two-step outputs. Sampled peaks are 2,305 MiB and 63 C, with no missing memory/temperature/utilization readings. Timing includes startup/cleanup and is not a benchmark. Comparison: `output/upgrades/moss-image-stateful-20261027/mapped-staging-comparison-20261029.json`.

Staging was **not accepted**: the real-model pair does not demonstrate a benefit despite the earlier standalone probe. Its changes were removed from the maintained snapshot and generated diff, strict/idempotence verification passed, and the isolated build was restored to the exact accepted `dc94ffc6...` binary in 5.261 seconds. The rejected executable is preserved at `output/upgrades/gRPCServerCLI-0621d819-rejected-mapped-staging`, with its private CCV copy at `output/upgrades/ccv-69b1ed1f-qwen35-cuda-mapped-staging`. The guard-only control is preserved at `output/upgrades/gRPCServerCLI-0621d819-before-mapped-staging`.

The private resource guard now accepts explicit `DT_VALIDATION_NATIVE` without changing its original-candidate default. Optional `DT_VALIDATION_CLIENT_LOG` records bounded phase timestamps through the existing telemetry loop; it does not instrument native code or change its arithmetic. The override/help/cleanup gate and mocked eight-signpost/default/cleanup test pass. All acceptance runs continue to unset `LD_PRELOAD`, `DT_SAMPLE_CPU_STACKS` and `DT_TARGETED_PROFILE_CLIENT_LOG`.

### Guard-only Full Gate And Final Verification

The accepted guard-only candidate was then tested at the unchanged 900-second 50-step request cap: **timeout, no final image or acceptance JSON**, post-Echo passed, 910.809 seconds wall time including startup/cleanup. Tag: `moss_original_raw_prefetch_guard_50steps_20261029_a`. It emitted 37 sampling callbacks and 24 previews, not 37 certified completed steps. All 1,792 memory/temperature/utilization samples are present; peak 2,305 MiB, maximum 63 C, no resource abort. GPU memory/use returned to zero after cleanup.

Passive timestamps measured from the guard's monitor setup show `textEncoded`/`imageEncoded` at 41.445 seconds, initial `sampling` at 113.540 seconds, and last observed `sampling` at 887.830 seconds. Subsequent callback intervals range from 14.518 to 28.286 seconds (median 21.655). This is sustained denoising-time cost as well as startup/loading overhead; callback timing is not isolated kernel or completed-step timing. Structured report: `output/upgrades/moss-image-stateful-20261027/prefetch-guard-50steps-summary-20261029.json`.

After all inference cleanup, the restored guard-only candidate passes all 93 CUDA-enabled tests in 73.599 seconds and all eight linked-runtime execution gates with the expanded 24-case mapped-GEMM probe. Reports/logs: `output/upgrades/native-backends-qwen35-prefetch-final.json`, `prefetch-final-regression-20261029.log` under the common evidence root. Bundle verification passes, all nine protected hashes remain unchanged (`prefetch-final-preservation-20261029.json`), and the GPU is idle. No installed defaults, catalogs, model payloads, precision or threading policy were changed.

At that checkpoint, full-resolution prompt-following quality and practical raw-model inference remained unaccepted. The actual-allocation investigation and passing full request above supersede that status; custom low-bit conversion remains unaccepted. The standalone managed/staged probe did not identify the effective real-model path, and no deadline increase was used.

## October 29 Resume

The user resumed after restarting the computer. Commit `475912fc4e35a462398f03ba5d7ec808a296f692` matched `origin/main`, the worktree was clean, the candidate hash matched, the GPU was idle and bundle integrity passed. This supersedes the pause instruction below, not the preservation/no-subagent constraints or the requirement for new commit/push authorization.

- The first repaired-bundle two-step check timed out at 180 seconds, with post-Echo passing and no final image. It reached `textEncoded`, `imageEncoded` and the initial `sampling` callback. In `DDIMSampler`, feedback occurs before the iteration executes; that callback alone is not a completed denoising step.
- A separate 120-second read-only copy trace also timed out. Its first 512 logged copies all succeeded: 15,467,090,944 bytes and 64.518422 seconds summed inside `cudaMemcpy`. The tracer hit its 512-copy cap; those are incomplete totals, and no pending logged copy does not establish that later transfers finished. Later copies slowed markedly. Neither timeout establishes a deadlock or justifies a threading/precision workaround.
- A bounded 32 MiB transfer probe compared file-mapped, heap and pinned sources with both ordinary device and managed destinations. All 18 baseline transfers passed byte-exact readback, taking approximately 3.3-8.2 ms. With a fixed 4 GiB managed reservation, managed-destination transfers took approximately 33-39 ms regardless of source, while ordinary device transfers remained approximately 3.2-8.3 ms. This reproduces some allocation-state scaling, not the complete native slowdown, and does not justify replacing the mapped-file path.
- One unchanged guarded two-step check with a 480-second request cap completed in **203.905 seconds wall time including startup/cleanup**. Decoded model selection was `moss_image`; one finite 512x512 image, zero audio, complete stream/chunks and post-Echo all passed. Its viewed image is blurred with an orange form in a light/green scene, not full-quality acceptance. All 379 GPU memory readings were present, sampled peak was 2,305 MiB, no resource abort was logged, and cleanup returned the GPU to 0 MiB/0%.
- The unchanged guarded **50-step request timed out at 900 seconds** (`canary_rc=124`, post-Echo 0; 913.259 seconds wall time including startup/cleanup). It emitted 38 sampling callbacks and 25 previews, but no final image or acceptance JSON. All 1,794 memory/temperature readings were present: sampled peak 2,305 MiB, maximum 62 C; one utilization sample was missing. No resource abort was logged. The last preview was finite and viewed, but is a 64x64 intermediate projection, not a full-resolution VAE-decoded image or prompt-quality acceptance. Cleanup returned GPU memory/use to zero.
- A nearby source check confirms raw on-demand file tensors use `cumallocmanaged` through `ccv_nnc_tensor_new_from_file`. A bounded standalone cuBLAS probe compared the same 8192x3072 weight matrix and 3072x1055 input using FP16 inputs/output and FP32 accumulation. Every result exactly matched the analytic value. After warm-up, ordinary-device weights took 1.640-1.690 ms versus managed weights 51.519-52.534 ms. With 128 additional managed 32 MiB reservations (4 GiB total), ordinary-device weights took 45.646-46.696 ms and managed weights 81.040-84.603 ms. Timings include the GEMM call and device synchronization, not isolated GPU kernel time. This establishes allocation-related cost in the probe, not the complete native bottleneck, a production staging fix, or permission to force the entire model into VRAM.
- No native implementation, model payload, precision, threading or sampler change was made. The same private candidate, repaired bundle, CPU offload, `--fread-preferred`, weights cache zero, DDIMTrailing/CFG 3/shift 5/seed 4242 and teapot prompt were retained. No full-size copies/downloads were made. Full 50-step completion, practical full inference and prompt-following acceptance remain unverified. Do not repeat an unchanged long request merely with a larger deadline.

Local evidence:

- `output/q6p_canary_moss_original_raw_fixed_guarded_2steps_20261029_a/`: initial timeout.
- `output/q6p_canary_moss_original_raw_fixed_copy_trace_20261029_a/`: bounded transfer trace.
- `output/q6p_canary_moss_original_raw_fixed_guarded_2steps_20261029_b/`: passing strict acceptance JSON, raw image, PNG, client/server logs and GPU samples.
- `output/q6p_canary_moss_original_raw_fixed_guarded_50steps_20261029_a/`: full-request timeout, progress, GPU samples and last intermediate preview PNG; no accepted final image.
- `output/upgrades/moss-image-stateful-20261027/file-copy-probe-20261029-{a,4g}.log`: exact-readback probe results; probe source/binary remain ignored diagnostics.
- `output/upgrades/moss-image-stateful-20261027/file-copy-gemm-probe-20261029-{a,4g}.log`: exact-output GEMM/allocation-state controls.
- `output/upgrades/moss-image-stateful-20261027/raw-fixed-preservation-20261029.json`: all nine protected hashes unchanged after native cleanup, plus private-file identities and full-request outcome.

At the end of those inference runs, bundle verification passed again after cleanup; all three payload sizes and single-link regular-file identities were unchanged. All nine protected runtime/CPU/catalog/canonical-model/candidate hashes matched the October 28 baseline. Only the README and this handover had tracked changes, and no native/GPU jobs remained active. That documentation was subsequently committed/pushed as `207047f` under renewed user authorization; see the follow-up above for current work. Run hashing strictly serial to inference.

## User Pause And WIP Checkpoint

Historical October 28 checkpoint: the user requested pausing for the day and committing/pushing the work. That request was completed as `475912f`; the user explicitly resumed on October 29. The document was created after published commit `fc26076` and is included in the WIP checkpoint.

The investigation identified a concrete packaging defect, not a threading or precision fix:

- Raw external codec `0x10000000` expects a 16-byte little-endian `<QQ>` descriptor: relative offset, byte length. Only palettized descriptors use `<IIQQ>` with a block-size/padding prefix.
- The converter and Python reader incorrectly agreed on the 24-byte palettized layout for raw tensors. The pinned native C reader passes the descriptor unchanged; Swift `decodeWithExternalEagerFile` reads offset at byte 0, and `decodeWithExternalStore` reads length at byte 8. Old bundles therefore load from relative offset zero, or request a potentially huge length equal to the intended payload offset.
- Bounded parent-launched LLDB samples initially caught `cufileread` / `cudaMemcpy` / the WSL driver. A separate read-only timing trace completed all 234 logged copies (6,508,057,600 bytes) in a summed 7.798564 seconds, so those samples did not establish the sustained bottleneck.
- Targeted post-encoding sampling reached libc `read` at file address `0x11496c`. A full stack located `DynamicGraph._Store.loadBytes` through `decodeWithExternalEagerFile`, `Store.read` and `UNetFixedEncoder`. The selected thread accumulated approximately 99% of one core. This supports the oversized-read diagnosis; it is not post-fix performance certification.
- Fixed raw packaging, validation and reading in the toolkit. A new independent descriptor test first failed `24 != 16`, then passed. Raw sidecar coverage now rejects the malformed layout while retaining 24-byte palettized handling. All 66 CPU-focused tests pass in 6.694 seconds.
- Repaired all 524 descriptors in the existing private bundle without a new model copy. Only its 114,688-byte metadata region and manifest changed; the 10,186,162,176-byte denoiser file size and complete trailer payload-region hash are unchanged. Metadata/manifest backups and rollback support were used.
- **At the October 28 pause, no native inference had run on the repaired bundle.** The old two-step finite image used malformed storage and is not correct-weight loading acceptance. See the October 29 results above for the first repaired-bundle finite-output pass and subsequent full-request timeout; full-quality acceptance remains unverified.

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

Conversion and private bundle tooling are implemented. Raw descriptors are corrected, and allocation-aware native staging now completes the original-source full request with a prompt-following image. Continue from the current passing result above, not the superseded timeouts or rejected flag-only staging experiment. Do not repeat pre-repair profiling or an unchanged long render merely to reproduce the existing state.

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
| Private bundle | Raw descriptors repaired to native format; independent payloads and pinned companions verified; original-source full request passed |
| Native model selection | Decoded server metadata confirms `moss_image`, rather than merely finding a filename |
| Pre-repair two-step request | One finite 512x512 noisy image, complete stream; malformed raw descriptors invalidate correct-weight loading acceptance |
| Post-repair two-step request | One finite 512x512 blurred image; complete stream/chunks/post-Echo; 203.905 s wall time including startup/cleanup |
| Pre-staging repaired full request | Timeout at 900 s after 38 sampling callbacks/25 previews; no accepted final image |
| Allocation-aware two-step request | Strict pass in 184.610 s; config/raw/PNG byte-identical to preserved guard-only control |
| Allocation-aware full request | Strict finite/stream/post-Echo pass in 289.534 s; viewed 512x512 image follows the teapot/table/plant/window prompt |
| Custom merged SFT | MOSSkr2 local BF16 source converts without model-specific changes; two-step and full 50-step pass; viewed prompt-following image in 284.303 s |
| Pre-repair full 50-step request | Timed out at 1,800 seconds; post-request Echo passed; no accepted final image |
| Pre-repair short stage probe | Reached `textEncoded`, then `imageEncoded`; no sampling signpost within 60 seconds |
| CPU-focused tests | 66 passed in 6.694 seconds after descriptor correction |
| Preservation | Earlier 24-file baseline retained; metadata-only repair preserved payload bytes; post-run bundle integrity and all nine current protected hashes pass |
| Full current CUDA suite | 95 tests passed in 74.597 s; all eight linked execution gates pass, including 36 exact-output GEMM cases |

**Do not equate finite output, schema validity, a live CPU worker, or a passing Echo with correct image generation.** The original-source full image was actually viewed and matches this prompt/profile. Arbitrary custom-model quality, wider coverage and framework parity remain unverified.

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
3. Offline raw native trailer packaging bypasses `TensorData.makeExternalData` rewriting. The original package produced a finite two-step image without a sidecar, but used incorrect 24-byte raw descriptors. That is not correct-weight loading acceptance. Corrected raw descriptors now pass a guarded two-step finite-output check; the post-repair full request timed out, so full quality remains unverified.
4. Client progress previously buffered to redirected stdout. Both timed and untimed generation launches now use Python `-u`; a one-second fixture verifies that received signposts survive timeout and owned-server cleanup. The old long-run four-line client log cannot establish which steps actually completed.

Native trailer details: SQLite header bytes 60:64 contain the big-endian UInt32 trailer base; base/payload alignment is 16,384. Raw external codec is `0x10000000`; its descriptor is little-endian `<QQ>`: relative offset, byte length. Palettized external codecs use `<IIQQ>`: block size, padding, relative offset, byte length. Confusing these layouts caused the defect. Source of truth is pinned s4nnc `nnc/Store.swift` and its unchanged CCV decode callback boundary.

## CPU Evidence

The long-run process had 20 allowed logical CPUs, cgroup `cpu.max = max 100000`, and no inherited OMP/OpenBLAS/MKL/BLIS thread-limit settings. It had 12 native threads; one worker accumulated about 96% of total CPU time. Approximately 101% in `top` therefore meant one core's worth of CPU, not a one-core container limit. Roughly 16 GiB resident memory matched the user's observation.

Loaded CPU libraries included `libomp.so.5` and `libcblas.so.3.10.3`. Their presence does not prove that setting `OMP_NUM_THREADS=16` would help. The later bounded profiles identified the native raw read path and malformed descriptors described above; no BLAS/threading workaround was applied. Post-fix copy diagnostics and the first finite-output pass are recorded in the October 29 section.

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

1. Original-source conversion and the full 512x512 prompt-following request now pass. Next validate an actual compatible custom full-precision checkpoint through the same inspect/convert/private-bundle/preflight/short/full sequence, or expand prompt/geometry coverage. Do not reopen the superseded flag-only staging experiment or force all raw weights into VRAM.
2. Reuse denoiser hash `fa67bb5f...`; do not recreate/download/copy payloads. Leave diagnostic interposition/profiling variables unset for acceptance. Confirm current worktree, candidate identity and resource/port/process state before a new workload; historical PIDs are not live jobs.
3. If a local failure remains, form one nearby falsifiable hypothesis and immediately validate a small fix. Do not blindly change threading, precision or transforms. Any actual native implementation change must follow the pinned patch workflow.
4. For each new checkpoint/profile, require a completed finite full-resolution image and inspect it for the prompt before accepting quality. Recheck protected resources and private payload integrity after native cleanup; record actual results and practical timing separately.

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
