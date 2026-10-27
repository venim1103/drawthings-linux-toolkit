# WeeLLM Analysis and Hybrid Tooling Plan for `drawthings-linux-toolkit`

Assessment date: 2026-10-24. Reference WeeLLM commit inspected: `88233af976568bfb134f082b62434c1b3f849b28` (`jit-roy/weellm`).

## October 26 Hardening Status

This section supersedes the original completion claims and unexecuted recipes below. The [October 26 review](WEELLM_HYBRID_REVIEW_2026-10-26.md) remains the historical defect baseline. The hybrid tooling is now substantially hardened, but native end-to-end MOSS inference is **not** certified.

### Implemented And Checked

- Standalone compiler/builder/oracle/LoRA entrypoints work from outside the repository without a `PYTHONPATH` workaround.
- Contracts are schema-validated and checked for globally unique target keys, coherent counts, and valid transform/shape lists. `jsonschema>=4.18,<5` is declared in the tools requirements.
- MOSS was rebuilt from the pinned native mapping: all **524** target keys, shapes, and stored datatypes match the official checkpoint metadata. The independent native-layout SHA-256 is frozen in the test suite. The two F32 tensors are the second time projection, not blanket LTX-style norm/modulation restoration.
- All **4,448** LTX source rules have shape checks against the existing safetensors header; all **5,746** target names match the existing trusted converted checkpoint. The two learnable-register names are corrected. Bundled VAE/audio-VAE/vocoder weights are explicitly excluded, not silently discarded.
- Conversion resolves a complete mapping before materialization, rejects unmapped/ambiguous/duplicate sources and target collisions, honors source/target shapes, checks serialized F16/BF16 values for overflow, and checks/fsyncs staged SQLite output before replacement. Existing output requires `--force`; existing sidecar destinations are not silently replaced.
- Raw inline, native `tensor_splits`, and raw external descriptors are understood. Codec identifiers are read from the high bits of `type`. Missing or truncated sidecars and unsupported encoded payloads fail explicitly. Read-only inspection cannot create missing files. Native acceptance of a tiny Python-written split tensor was verified through the unchanged candidate's actual reader under LLDB.
- Restoration stages an SQLite backup, preserves per-rule matrix/norm rank, restores BF16/F32 from the model's own source, clears replaced codec metadata, validates required coverage and source finiteness, and atomically replaces only on success. Failure leaves original checkpoint bytes unchanged.
- Diff/health/structural gates reject NaNs, unchecked data, wrong shape/key coverage, unknown datatypes, malformed payloads, and empty checks. Packed/palettized codecs are not claimed to have been numerically validated when the Python decoder does not support them.
- Kohya tokenization, profile-aware aliases, FP8 output-channel scaling, shard ownership/duplicate detection, and scaled/fused LoRA output verification have regression coverage. Unsupported LoRA input/diagonal split variants fail instead of reporting zero-pair success.
- Contract-driven shell failures are fatal. LoRA verification runs before catalog registration. Catalog writes are staged and atomic. Contract quantization/restoration uses private staging and does not replace an existing inline model after a failed check; externalized output publication is explicitly rejected until paired-store publication is implemented. Temporary-workspace tests cover these failures plus valid single-file/indexed-directory direct conversion.
- Qwen fixtures now use real 16-key/32-value grouped heads, native `[B,Hv,Dv,Dk]` state, explicit log-decay, nonzero initial state, backward state checkpoints, and per-head gated RMSNorm. A C probe compiles the pinned unchanged CPU implementations and independently agrees on grouped, batched, non-square, channel-decay, checkpoint, and real-head-size cases.
- Ten GGML dequantizers have frozen three-block reference hashes independently obtained from `gguf==0.13.0`; no GGUF package install is needed to run those regression vectors.
- The Swift auditor recognizes generic/interleaved-index syntax and head mismatches and can flag unsupported MOSS importer branches, but reports its scope as heuristic rather than runtime acceptance.

### CUDA Operator Increment: October 26

The separate [Qwen3.5 CUDA patch profile](../DRAW_THINGS_PATCH/cuda/README.md) now contains forward CUDA backends for `GatedDelta` and `RMSNormGated`. Both are correctness-first implementations validated on the RTX A3000/SM86 and integrated into a separately rebuilt server. The legacy CPU bundle and pristine candidate dependencies remain unchanged.

- RMSNorm passed 60 precision/layout/affine/in-place numerical cases; observed maximum absolute error was about `9.54e-7`.
- Gated delta passed 108 grouped/batched/non-square/real-head-size precision, decay, checkpoint, and in-place-state cases. Maximum observed FP32 state error was about `4.47e-8`; output error including reduced-precision rounding was about `6.10e-5`.
- Invalid/empty operation contracts are covered. GPU tests are explicitly opt-in; CUDA compilation and real device execution were performed for this increment.
- Pinned base hashes, independent snapshots, and a generated unified diff are maintained under `DRAW_THINGS_PATCH/cuda`. Strict apply, exact snapshot equality, repeat-apply idempotence, drift rejection, and preservation of unrelated Package files are tested in temporary trees. No snapshot fallback is allowed.

```bash
DT_RUN_CUDA_TESTS=1 .venv/bin/python -m unittest \
  tools.test_dt_weellm.CudaQwenReferenceTests \
  tools.test_dt_weellm.CudaPatchBundleTests -v
.venv/bin/python tools/drawthings_cuda_patch.py verify --ccv-source /path/to/pristine/pinned/ccv
```

The gated-delta backend keeps recurrent state/decay in F32 and supports F32/F16/BF16 Q/K/V/beta, grouped heads, scalar/channel log or direct decay, backward checkpoints, and checkpoint-free in-place state. It currently supports key dimensions up to 1,024. RMSNorm supports one last-axis reduction on contiguous tensors, optional affine scale, mixed supported precisions, and in-place output. Unsupported contracts fail explicitly; backward/training and performance optimization are out of scope.

### Native Server Integration: October 26

The pinned community checkout, copied CCV source, Bazel root, and artifact tag are isolated under `output/upgrades` and `.cache/bazel-qwen35-cuda`. The build helper rejects reuse of the pristine candidate, default cache, or default artifact tag when a CUDA override is supplied. Both native builds used 16 jobs; neither installed or promoted a runtime.

The first full build compiled successfully but failed the new runtime registration gate: all six Qwen CUDA selections were absent. CCV's `lib/BUILD.bazel` consumes a pre-generated `ccv_nnc_cmd.inc` without regenerating it. The helper now runs the patch tool's private-only `prepare` action using upstream Ruby code generation, checks the generated GPU calls, and then builds. Ruby is declared and checked in the devcontainer build prerequisites. The first attempt's log, hash, and negative registry report are preserved with `before-bindings` names.

The initial operator-integrated candidate at `output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI` had SHA-256 `5cfdd16e4110e5248fcb4f37fca7da2dc26790fdddcca1f9682f22e24ba1a39d`, now preserved separately as `output/upgrades/gRPCServerCLI-0621d819-qwen35-cuda-before-empty-sdpa`. The local report `output/upgrades/native-backends-qwen35-cuda.json` records all six Qwen F32/F16/BF16 selections as GPU_REF, an SDPA GPU control, and both CPU controls. Its `qwen_cuda_execution=0` also requires real native allocation, transfer, GPU dispatch, and readback for tiny F32 gated-delta and affine RMSNorm operations. Outputs and state checkpoints must be finite and agree with the pinned CPU implementations within absolute `2e-6`.

The native probe uses explicit C debugger expression settings and permits other threads to run during CUDA calls. Missing registrations refuse execution in the same expression; a failed debugger assertion cannot authorize a later GPU call. Regression tests accept the patched candidate and reject the pristine candidate, including explicit execution refusal. The combined suite passed **55 tests with CUDA enabled**, including the existing 168 numerical CUDA cases:

```bash
DT_RUN_CUDA_TESTS=1 .venv/bin/python -m unittest \
  tools.test_dt_weellm tools.test_dt_make_config tools.test_dt_tensor_to_playable -v
```

This closes operator integration, not full-model activation parity or MOSS inference. F16/BF16 numerical execution is covered by the standalone CUDA tests; the linked-server execution probe currently uses F32 fixtures.

### Empty Attention And Guided Canary: October 26

The preserved pre-fix binary reproduces the rank-zero SDPA divide-by-zero. The CUDA profile now includes the pinned original SDPA source and a narrow guard: all required tensors and present outputs empty is a no-op; mixed empty/nonempty contracts fail. Existing assertions and nonempty compute remain intact. A fresh private CCV copy at `output/upgrades/ccv-69b1ed1f-qwen35-cuda-empty-sdpa` avoids applying an evolved profile over older snapshots.

The 16-job rebuild reused the private cache and completed in 18 seconds. The current separate candidate has SHA-256 `5cca983e5883b5e9091f24d34475af99bc6054e67aa51da1f7f6fa8e7441febd`. Its report `output/upgrades/native-backends-qwen35-cuda-empty-sdpa.json` passes the existing Qwen gate plus `--execute-sdpa`: eight empty/mixed/projected contracts and a nonempty batched, grouped, non-square FP16 GPU calculation against known output. The combined CUDA-enabled suite now passes **56 tests**.

The two-step 512x512 CFG-3/fixed-shift-5 private canary completes its stream and post-Echo without SIGFPE but still returns zero final images and one preview with **131,072 non-finite values**. This is a failed canary, not a quality result. Its 323 whole-device samples show an observed 979 MiB peak and 57 C maximum; four utilization readings are unavailable, and partial-run memory is not a model-fit measurement. No 50-step run was attempted after this failure.

Bounded LLDB diagnostics inspect actual first-layer GPU calls without changing their outputs. The first gated-delta case uses `[1,11,16,128]` Q/K, `[1,11,32,128]` V, F32 decay/state, and zero backward checkpoints. It returns success and all six inputs/two outputs are finite after synchronization. The first gated RMSNorm likewise returns success with finite three inputs/output. These observations narrow the investigation but do not establish full encoder or denoiser parity. The observer resolves versioned CUDA APIs through `dlsym` and reads back through a header-grounded C adapter; unsuccessful observer launches were excluded from evidence.

Raw canary output is under `output/q6p_canary_moss_base_q8p_512_qwen35_empty_sdpa_2steps_20261026`. Verified diagnostic logs are under `output/q6p_canary_moss_base_q8p_qwen_live_delta_finite_verified_20261026` and `output/q6p_canary_moss_base_q8p_qwen_live_rmsnorm_finite_20261026`. Runtime logs also contain `GEMM_EX` parameter-12 warnings; their connection to the remaining non-finite output is unproven. The deferred LTX GEMM investigation was not reopened.

### Packed Embedding And Finite Output: October 26

The first nonempty Qwen embedding lookup, before decoder normalization, produced 1,848 non-finite values in `[11,4096]`. Its table is Q5_K-packed (`CCV_QX`, reserved codec 9); the CUDA index-select path neither decoded packed storage nor registered QX, so release-mode dispatch read encoded bytes as dense floats. This supersedes the earlier hypothesis that the first failure must occur after the initial Qwen operation pair. A few finite early operator calls were not full-encoder validation.

The separate CUDA profile now pins the original index-select source and reuses `ccv_nnc_compat_decode_qx` before typed lookup. It registers QX support, validates decoded precision/contiguity, prefetches external data, and checks launch status. The current isolated candidate is SHA-256 `920bf727c5a27a7d1ad90735735c74b7edfbbc1459598cea018f71e2d713dd96`, built with 16 jobs against `output/upgrades/ccv-69b1ed1f-qwen35-cuda-embedding`. The prior `5cca983e...` binary is preserved at `output/upgrades/gRPCServerCLI-0621d819-before-packed-embedding`. The full dense table is currently decoded into workspace, increasing memory; row-selective decoding is a later optimization.

Native `--execute-embedding` adds tiny Q5_K/Q6_K lookup tests across F32/F16/BF16 and repeated/reordered indices. The fixture uses CCV's native encoder and its aligned row-scale footer, not an assumed raw GGML layout. GPU values are compared with native CPU decoding: exact F32, at most two ULPs for reduced precision, with strict finite/sign checks. The pre-fix runtime refuses the packed test before device execution. The combined CUDA-enabled suite passes **57 tests** and the linked Qwen/SDPA/embedding report passes all gates.

The private two-step and original 50-step 512x512 CFG-3/shift-5 requests now each produce one finite final image with complete stream/chunks/post-Echo and no audio. Raw outputs and acceptance metadata are under `output/q6p_canary_moss_base_q8p_512_packed_embedding_{2steps,50steps}_20261026`. The 50-step config hash is unchanged at `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`. Observed whole-device peaks are 2,305 MiB for both runs; telemetry spans about 120/475 seconds including startup, maximum temperatures 56/59 C, with one unavailable utilization sample in the full run.

These are finite-output passes, **not semantic-quality acceptance**: the 50-step image depicts a repeated green structure rather than the requested teapot. Full reference conditioning/denoiser/sampler/decode parity is still needed. A read-only cuBLAS trace also showed the parameter-12 warnings are zero-sized skipped GEMMs with matching FP16 types (`m=0,k=0`), ruling out the mixed-datatype hypothesis; their broader empty-operation handling remains separate. The old LTX GEMM crash investigation remains deferred.

### Quality Diagnosis And Captured CPU Parity: October 26

The runtime binary remains `920bf727...`; this increment adds diagnostic coverage, not a production correction. Real bounded pre/post captures show that MOSS receives nonzero, distinct positive/negative text features (mean-feature cosine about 0.396). Its 4096-to-3072 text projection matches CPU math with maximum unrounded absolute error about 0.03127 under declared relative/absolute tolerances. The first 24-head text attention uses scale 1, no causal mask, no variable-length mode; independent CPU attention differs by at most 0.002071 in output and 5.81e-6 in log-sum-exp. Adjacent-pair rotary multiplication and its four-axis text positions also match the reference formula within reduced-precision tolerance.

The FP32 timestep projection matches CPU math to about 6.05e-8 maximum absolute error; its rows vary between timesteps. The live timestep weight and bias exactly match the official checkpoint, including a strictly bounded read of its raw FP32 trailer descriptor. Native latent scaling and pack/unpack ordering agree at source level with the reference setup. Native Euler velocity sign is consistent, though its timestep grid is not identical to the reference's snapped schedule. These facts do not establish complete Qwen/DiT/sampler/VAE parity.

`compare-capture` makes linear, rotary, and unmasked noncausal SDPA comparisons repeatable with capture hashes, shape/type validation, finite-value gates, and memory/compute budgets. The C capture adapter caps payloads at 64 MiB, rejects noncontiguous/unsupported tensors, and atomically refuses overwrite. Captures and reports are under `output/upgrades/moss-image-base-add235e9`, with prefixes `text_projection_capture`, `text_attention_capture`, `text_rotary_capture`, and `time_projection_capture`.

Nonuniform packed Q5_K/Q6_K GEMM tests initially exposed a small difference from dense products. The controlling kernel documents an intentional int8 activation quantization for more than four rows; modeling that contract makes packed products pass, while a separate dense GPU control independently matches unquantized CPU products. New `--execute-packed-gemm` acceptance covers both paths. Nonuniform packed lookup remains verified against native CPU decoding. All **60** combined CUDA-enabled tests pass.

Two diagnostic hypotheses were rejected without changing source or defaults. Forcing dense packed-weight fallback only inside a disposable debugger inferior makes encoder conditioning non-finite. Removing the negative branch with CFG 1 still produces an unrelated structured image after 50 steps (`output/q6p_canary_moss_base_q8p_512_cfg1_quality_diagnostic_20261026`), despite strict finite-image acceptance. The special skipped negative block is therefore not the sole quality cause. Next is a complete reference layer/model boundary comparison, not a sampler or precision change justified only by these passing primitives. M1 quality remains unaccepted.

### Scaled Recurrent Normalization And Prompt-Following Base: October 27

After the complete MOSS block-0 phase, an independent reference effort moved into Qwen decoder layers. The pinned Transformers 5.3.0 package is isolated under `output/upgrades/moss-image-base-add235e9/reference-transformers-5.3.0`, runs offline from constructed configuration/captured tensors, and is absent from normal toolkit imports. No new model payloads were downloaded. Full-attention layer 3 was captured completely (21 operations); FP32 Transformers output was finite, differing by about 2.7% RMS, or about 1.3% with native activation quantization modelled. These were diagnostic differences, not proof of an architecture bug.

The first recurrent-layer trace instead stopped at a concrete failure: native RMSNorm returned `CCV_NNC_EXEC_INVALID` on Qwen's non-affine key normalization. Its CUDA implementation rejected any output `scale != 1`, including the required key `1/sqrt(128)` and query `1/128` factors. The pinned checkpoint's stored `A_log` values are already exponentiated; no extra graph exponentiation was applied. Concat's builder and CUDA copy path explicitly skip empty tensors, so no speculative concat patch was made.

The CUDA profile now supports finite non-unit RMSNorm forward output scales by reusing the existing scale-one implementation and applying a typed output-only scaling kernel. It preserves saved inverse statistics and leaves scale-one/backward behavior unchanged. The new tiny regression reproduces pre-fix rejection and validates 36 affine/non-affine F32/F16/BF16 scale cases, including zero/negative scales and zero rows. The full actual recurrent layer now completes all 29 operations with status 0; its Q/K scales are recorded explicitly. The suite passes **67 CUDA-enabled tests**.

Current candidate SHA-256 is `1d27f8d444d6211322496d3ce20871173b5d7d5942401bbe07fba9fa216a91b5`, built with 16 jobs against `output/upgrades/ccv-69b1ed1f-qwen35-cuda-scaled-rmsnorm`. Pre-fix `920bf727...` is preserved at `output/upgrades/gRPCServerCLI-0621d819-before-scaled-rmsnorm`. Both short and standard 50-step private canaries pass finite-image/complete-stream/post-Echo checks. The full result visibly follows the orange-teapot/white-table/green-plant prompt, a first quality bring-up observation. The profile/config hash is unchanged. Artifacts: `output/q6p_canary_moss_base_q8p_scaled_rmsnorm_{2steps,50steps}_20261027`; peak sampled device memory 2,305 MiB, telemetry duration about 75/129 seconds including startup, maximum temperatures 57/65 C, no missing samples.

This supersedes the earlier current-candidate quality-failure status for this specific Base canary, not historical evidence. It does not certify every prompt, seed, resolution, quantization, SFT model, performance condition, or framework parity. Installed/default runtime and canonical models/catalogs remain protected. Next: repeatability and a second prompt before any wider model testing or promotion.

### Complete Block-0 Reference: October 27

A bounded native trace now captures all 32 operations of MOSS main block 0, from its first full-width RMSNorm through the second residual addition, stopping at block 1's corresponding norm. The actual input is `[1,1055,3072]`; all native operation statuses are successful, and the block exit exactly equals the next block's input. The private capture is about 1.02 GB under `output/upgrades/moss-image-base-add235e9/block0-trace-20261027`.

The new `compare-moss-block` oracle independently propagates the original input through normalization, modulation, Q/K/V projections, rotary, attention, gated residuals, and soft-capped SiTU FFN. All 28 checked intermediate/output stages pass declared `atol=0.01, rtol=0.01`; final relative RMS error is `0.00013639`. This is a complete executed-block arithmetic comparison, not a proof of the whole trained network or original pre-conversion source mapping.

Separate named checkpoint checks close the circular-loading gap: independently compiled pinned CPU q8p decoding matches all seven runtime matrices bit-for-bit, including which FFN matrix feeds the gate/up soft caps. Four learned norm vectors also match exactly after expected runtime precision conversion. A CPU calculation of timestep 1000, the time MLP, and all six named modulation projections agrees with the captured chunks (relative RMS errors about `0.00021` to `0.00060`). Reports are `checkpoint-matrix-assignment.json`, `checkpoint-normalization-assignment.json`, and `checkpoint-modulation-assignment.json` in the capture directory.

The public block oracle rejects incomplete/unsupported topology, native failures, malformed/unsafe captures, non-finite values, unrecognized affine/biased projections, oversized work, and accidental broadcast expansion. Small analytic and corrupted-output fixtures verify that later native captures cannot reset a wrong CPU reference. The combined CUDA-enabled suite passes **66 tests**. Runtime remains `920bf727...`; no quality fix, installation, promotion, or model/catalog mutation is claimed. Next priorities remain later/special denoiser boundaries and complete Qwen hidden-state reference parity.

### Commands

```bash
.venv/bin/python -m pip install -r requirements-drawthings-tools.txt
.venv/bin/python -m unittest tools.test_dt_weellm tools.test_dt_make_config tools.test_dt_tensor_to_playable -v

# The explicit native source path must be the pinned CCV checkout.
.venv/bin/python tools/dt_layer_parity_oracle.py qwen35-linear-attn-fixture \
  --out-dir output/oracle/qwen35_verified \
  --native-ccv-source /path/to/pinned/ccv
```

Native tests can use `DT_CCV_SOURCE_DIR` and `DT_GRPC_CANDIDATE`. The CUDA-enabled native execution test defaults to the separate Qwen candidate; the native split-store test defaults to the pristine candidate. Without the compiler, pinned sources, or candidate/debugger, relevant native tests explicitly skip; CPU-only success must not be presented as native acceptance. Fixture manifests record whether validation is `numpy_reference_only` or `native_cpu_verified`, input/output conventions, fixture hash, native source hashes, tolerances, and observed errors.

### Remaining Acceptance Gates

1. Establish repeatability and broader prompt/seed/resolution coverage after the first prompt-following Base result, then continue reference activation/sampler/decode parity and SFT bring-up. Preserve the legacy CPU patches, pristine candidate, and installed runtime; one successful quality canary does not authorize promotion.
2. Add full reference/native text-encoder and denoiser activation capture. `verify-block` is a serialization/weight check, not a forward execution oracle. Native CPU operation parity is now independently verified; full WeeLLM/native model parity is not.
3. Convert a complete real model into private staging, verify native load and finite final generation, and measure actual peak memory/time before publication. No full-size conversion or full-model GPU inference was performed in this hardening pass.
4. Add explicit native decoding or a validated bridge for ezm7/palettized/packed/trailer variants before numerical diff/health checks on those artifacts. Unsupported codecs currently fail closed.
5. Validate full artifact/catalog publication under concurrency and paired external storage, broader LoRA families/input/diagonal splits, and model/tokenizer/reference-environment pins and third-party notices before calling the workbench production-complete.

No canonical checkpoint payloads were edited and no runtime was promoted. Post-integration checks matched the installed server/launcher, all three legacy CPU converters, and both active catalogs against the preservation baseline. Both detached community source trees remain clean; the original CCV dependency reports `pristine`. The original candidate retains SHA-256 `fd9f2d82119e6d79172a70b4230f79c75933958b78a36b2cd8e4799cfb265b5a`. GPU memory returned to zero and no owned server, debugger, or compiler remained active.

---

## 1. Executive Summary

**WeeLLM** is a pure-Python/PyTorch layer-streaming framework built to run large diffusion transformers (DiTs) and vision-language text encoders on GPUs with limited VRAM. Rather than compressing models into static low-bit resident graphs, WeeLLM streams model layers on demand from System RAM or NVMe disk into a small GPU VRAM buffer during each forward pass and immediately evicts them after execution.

### Core Strategic Conclusion

1. **Production Inference Engine (`drawthings-linux-toolkit` wins on speed):**
   On our target hardware (NVIDIA RTX A3000 12 GiB Laptop GPU, SM86, ~29 GiB container-visible RAM), Draw Things' compiled C/CUDA runtime (`gRPCServerCLI` backed by `s4nnc` and `ccv`) remains the superior production inference engine once a model is converted and quantized (`q6p`, `q8p`, `q4p`, `i8x`, `i4x`). Resident quantized inference is compute-bound, whereas WeeLLM is PCIe-bus-bound (transferring 10–40 GB of weights across PCIe on *every* denoising step).
2. **Bring-Up, Conversion, and Diagnostic Tooling (`WeeLLM` fills our biggest gaps):**
   While you cannot drop WeeLLM's Python layer streamer inside `gRPCServerCLI`, WeeLLM's internal modules—specifically its **zero-copy Safetensors/GGUF seekers**, **on-the-fly GGUF/ComfyUI dequantizers**, **declarative tensor keymaps**, and **bounded-VRAM PyTorch implementations of MOSS-Image, Qwen3.5, and LTX-2**—directly solve the main engineering bottlenecks in `drawthings-linux-toolkit`:
   * **Unblocking MOSS M1 (`docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md`):** Provides a low-VRAM (<6 GiB) PyTorch reference oracle on the 12 GiB laptop to isolate the `131,072` non-finite (`NaN`) latent preview values and validate CCV's missing CUDA backends for Qwen3.5 `GatedDelta` (`0x6de198ac`) and `RMSNormGated` (`0x9b8af346`), as well as the layer-22 zero-length attention bypass (`_ccv_nnc_scaled_dot_product_attention_forw`).
   * **Unblocking Custom MOSS Conversion (M3) & GGUF Ingestion:** Upstream `ModelImporter.swift` still hits `fatalError()` branches for MOSS generic import, and `s4nnc` CPU graph transforms have historically introduced out-of-bounds bugs (§17 rotary transpose) or metadata omissions (§3a BF16 embeds, §18 QK-norm de-interleave, §22 `[1,1,N]` norm rank). By combining WeeLLM's seekers/dequantizers with a contract-driven Python SQLite `.ckpt` writer, we can convert Hugging Face `.safetensors` (single or sharded), ComfyUI FP8 checkpoints, and `.gguf` files directly into validated Draw Things `.ckpt` stores.
   * **Generalizing Post-Quantization Norm Restores & LoRA Verification (M4):** Replaces LTX-hardcoded regexes in `dt_quantize.sh`, `dt_q6p_restore_f32_norms.py`, and `dt_q6p_restore_qk_norms.py` with declarative per-architecture contracts (`contracts/ltx2_3.json`, `contracts/moss_image.json`), and automates the §23/§24 static LoRA permutation/rank audit across architectures.

---

## 2. Architectural Comparison: `drawthings-linux-toolkit` vs. `WeeLLM`

| Dimension | `drawthings-linux-toolkit` (`gRPCServerCLI` + `s4nnc`/`ccv`) | `jit-roy/weellm` (`88233af9`) |
| --- | --- | --- |
| **Core Memory Strategy** | **Compress & Keep Resident (or Static Offload):** Converts weights offline into SQLite `.ckpt` (+ `-tensordata` sidecar or trailer). Uses palettized (`q4p`/`q6p`/`q8p`) or packed row-wise (`i8x`/`i4x`) codecs so the model fits in 12 GiB VRAM. | **Stream & Evict (Dynamic Block Swapping):** Keeps weights in System RAM (`ram_seek.py`) or on NVMe disk (`disk_seek.py`). Streams 1 to $K$ transformer blocks into a small GPU VRAM budget per forward pass. |
| **Per-Step Bottleneck** | **Compute-bound (Tensor Cores):** Weights stay in VRAM (or use static `externalOnDemand`); denoising steps run at native CUDA GEMM/FlashAttention speeds. | **I/O & PCIe-bound:** A 20 GB model run for 30 diffusion steps transfers $30 \times 20\text{ GB} = 600\text{ GB}$ over PCIe ($\sim 30\text{–}45\text{ s}$ of pure bus transfer overhead per generation, plus per-layer dequantization). |
| **Minimum VRAM Floor** | ~7–11 GiB for 20B+ models after `q6p`/`q4p`/`i4x` quantization. | **2–6 GiB VRAM** even for unquantized 20B–32B models (as long as System RAM + NVMe can hold the file). |
| **Model Ingestion Friction** | **High:** Requires Swift `ModelImporter` support, exact `ModelWeightMapping` rules, rotary de-interleaving, SQLite `datatype`/`dim` alignment, and F32 norm restoration. | **Near Zero:** Directly reads Hugging Face `.safetensors`, ComfyUI FP8 `.safetensors`, and `.gguf` files using Python dictionary keymaps. |
| **Supported Model Catalog** | Curated C/Swift implementations (`LTX2.swift`, `MOSSImage.swift`, `Flux2.swift`, `HiDream.swift`, SDXL/SD3, etc.). | Broad Python catalog: `ltx2` / `ltx25`, `moss_image`, `flux` / `flux2` / `flux3`, `minimax_h3`, `hidream`, `cogview4`, `ernie_image`, `ideogram4`, `krea2`, `longcat`, `lumina2`, `qwen_image` / `21`, `z_image`, plus `qwen3_5`, `qwen3_vl`, `gemma2`/`4`, `mistral3`, `glm`, `t5`/`umt5`. |
| **LoRA Execution** | Offline conversion via `lora-converter` into `_lora_f16.ckpt` (and optional `q8p`/`q6p`/`q4p` quantization via `model-quantizer`). | Dynamic on-the-fly per-layer delta application (`W + alpha * B @ A`) in scratch VRAM during block streaming (`lora_streamer.py`). |
| **Best Role on Our 12 GiB Laptop** | **Primary production runtime** for repeatable, fast video/image generation and mobile-portable `.ckpt` creation. | **Reference oracle, direct format bridge, and rapid architecture prototyping tool** when bringing up new models or debugging converter/CUDA defects. |

---

## 3. Codebase Audit of `jit-roy/weellm` & Reusable Components

Inspection of `WeeLLM-88233af976568bfb134f082b62434c1b3f849b28` identifies five subsystems that we can adapt directly into `drawthings-linux-toolkit`:

### 3.1 Zero-Copy Disk & RAM Seekers (`weellm/io/`)
* `weellm/io/safetensors/safetensors_base.py`, `disk_seek.py`, `ram_seek.py`:
  * Parses the 8-byte little-endian header length and JSON metadata once, then seeks directly to `[buffer_start + start_offset, buffer_start + end_offset]` per tensor without loading the entire checkpoint into memory.
  * `ram_seek.py` pins or caches byte buffers in System RAM so repeated layer accesses avoid disk I/O.
* `weellm/io/ggufs/gguf_seek.py`:
  * Parses GGUF v2/v3 headers, metadata KV pairs, and tensor info blocks, providing random-access byte slices for quantized GGUF tensors.
* `weellm/io/memory.py` & `seeker.py`:
  * Computes per-block memory footprints and coordinates `vram_budget` and `ram_budget` allocations.

### 3.2 On-the-Fly Dequantizers (`weellm/io/ggufs/gguf_dequant.py` & `safetensors/comfy_dequant.py`)
* `gguf_dequant.py`:
  * Pure PyTorch/NumPy block dequantization routines for standard GGML/GGUF quantization types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`).
* `comfy_dequant.py`:
  * Unpacks ComfyUI-specific FP8 (`F8_E4M3`, `F8_E5M2`) and block-scaled weights (`weight_scale` / `scale_weight`) into standard FP16/BF16/FP32 tensors.

### 3.3 Declarative Keymaps (`weellm/io/ggufs/keymaps/` & `weellm/io/safetensors/keymaps/`)
* **Transformers:** `ltx25.py`, `flux.py`, `flux2.py`, `flux3.py`, `hidream.py`, `qwen_image.py`, `qwen_image_21.py`, `sd15.py`, `sd3.py`, `sdxl.py`, `z_image.py`, `auraflow.py`, `cogview4.py`, `ernie_image.py`, `ideogram4.py`, `krea2.py`, `longcat.py`, `lumina2.py`, `minimax_h3.py`.
* **Text Encoders:** `gemma4.py`, `glm.py`, `llama.py`, `qwen3vl.py`, `t5.py`.
* **LoRAs & VAEs:** `weellm/io/safetensors/keymaps/loras/{ltx2.py, minimax.py, diffusers_fallback.py}` and `vaes/qwen_image_vae.py`.
* **Why this matters for Draw Things:** These files already solve the tedious naming-convention alignment between raw BFL/ComfyUI/GGUF checkpoints and canonical Diffusers/Transformer block structures, including where fused `qkv` or fused `mlp` weights must be split.

### 3.4 Streamed Model Implementations (`weellm/models/`)
* `weellm/models/text_encoders/qwen3_5_model.py`:
  * Implements Qwen3.5 (the exact text encoder used by MOSS-Image) with layer-by-layer streaming through `BaseTextEncoderStreamer`.
* `weellm/models/transformers/moss_image_transformer_2d_model.py`, `weellm/pipelines/image/adapters/moss_image_pipeline.py`, `weellm/schedulers/moss_image_scheduler.py`:
  * Full reference implementation of the 42-layer MOSS-Image architecture, 2-block text refiner, 4-axis centered rotary embeddings, 2x2 latent packing (128 channels per token), and flow-Euler scheduler.
* `weellm/models/transformers/ltx2_video_transformer_3d_model.py` & `ltx2_connectors.py`:
  * Streamed implementation of the LTX-2 3D video/audio DiT and 1D connectors.

---

## 4. Proposed Hybrid Tooling Architecture

All new tooling will live under `contracts/` and `tools/` (with shared library code in `tools/dt_weellm/`). It requires **zero modifications** to the production `gRPCServerCLI` binary or existing LTX 2.3 shell entrypoints until explicitly opted in.

```text
drawthings-linux-toolkit/
├── contracts/                                # Declarative model serialization & mapping contracts
│   ├── schema.json                           # JSON Schema for contract validation
│   ├── ltx2_3.json                           # 5,746-tensor LTX 2.3 contract (§3a, §17, §18, §22)
│   └── moss_image.json                       # MOSS-Image Base/SFT 42-layer + refiner contract
└── tools/
    ├── dt_weellm/                            # Core Python library adapted from WeeLLM
    │   ├── __init__.py
    │   ├── seek_io.py                        # Safetensors (single/sharded) + GGUF zero-copy seeker
    │   ├── dequant.py                        # GGUF K-quant & ComfyUI FP8 dequantizers
    │   ├── transforms.py                     # CPU NumPy/Torch: rotary de-interleave, splits, dim blobs
    │   ├── sqlite_store.py                   # s4nnc SQLite .ckpt + -tensordata reader/writer
    │   └── keymaps.py                        # Unified WeeLLM <-> Draw Things keymap registry
    ├── dt_keymap_compiler.py                 # Tool 1: Contract generator & Swift mapper static auditor
    ├── dt_direct_ckpt_builder.py             # Tool 2: Pure-Python Safetensors/GGUF -> .ckpt converter
    ├── dt_layer_parity_oracle.py             # Tool 3: Low-VRAM WeeLLM vs. Draw Things bisection oracle
    └── dt_lora_universal_bridge.py           # Tool 4: Multi-arch LoRA normalizer & §24 static verifier
```

---

## 5. Detailed Component Specifications & Recipes

### 5.1 Core Library (`tools/dt_weellm/`)

#### A. `tools/dt_weellm/seek_io.py` — Unified Zero-Copy Tensor Seeker
Adapts `weellm/io/safetensors/disk_seek.py` and `weellm/io/ggufs/gguf_seek.py` into a unified `TensorArchiveSeeker` interface supporting:
1. **Single-file `.safetensors`:** Reads the 8-byte header size, parses JSON offsets, validates `data_offsets` against file size (matching `dt_preflight_safetensors.py`), and reads individual tensors via `os.pread(fd, length, buffer_start + start)`.
2. **Sharded `.safetensors` (`model.safetensors.index.json` or directory):** Resolves `weight_map` entries to shard filenames and opens file descriptors lazily per shard. (Directly satisfies the M3 requirement in `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` to support sharded sources and reject incomplete shard sets fail-closed).
3. **`.gguf` files:** Parses GGUF v2/v3 tensor descriptors (name, dimensions, `ggml_type`, byte offset relative to aligned data section) and yields raw quantized byte blocks per tensor.

Peak host memory during streaming is bounded to `O(max_single_tensor_bytes)` ($\le 1.6\text{ GiB}$ for LTX 2.3's largest `video_aggregate_embed` tensor, and $< 200\text{ MiB}$ for standard DiT blocks).

#### B. `tools/dt_weellm/transforms.py` — Bit-Exact `TensorDescriptor.swift` Transform Engine
Re-implements in deterministic CPU NumPy/PyTorch every weight transformation performed by `DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift` and `s4nnc`:

1. **Rotary Q/K & QK-Norm De-interleaving (§17 & §18):**
   Matches `TensorDescriptor.swift` lines 81–110 (`out[h*hd + d*2 + p] = in[h*hd + p*(hd/2) + d]`):
   ```python
   def rotary_deinterleave(
       arr: np.ndarray, num_heads: int, head_dim: int
   ) -> np.ndarray:
       """Permutes axis 0 of shape [num_heads * head_dim, ...] from interleaved to split-half rotary order."""
       if arr.shape[0] != num_heads * head_dim:
           raise ValueError(
               f"rotary_deinterleave shape mismatch: {arr.shape[0]} != {num_heads} * {head_dim}"
           )
       if head_dim % 2 != 0:
           raise ValueError(f"head_dim must be even, got {head_dim}")
       orig_shape = arr.shape
       cols = int(np.prod(orig_shape[1:])) if arr.ndim > 1 else 1
       viewed = arr.reshape(num_heads, 2, head_dim // 2, cols)
       permuted = np.transpose(viewed, (0, 2, 1, 3))
       return np.ascontiguousarray(permuted.reshape(orig_shape))
   ```
   * Inverse function `rotary_interleave()` is also provided so converted `.ckpt` tensors can be mapped back to Hugging Face layout for single-layer parity tests.

2. **Fused Tensor Slicing (`split_output` / `split_input` / `diagonal_up` / `diagonal_down`):**
   Matches `TensorDescriptor.swift` lines 112–260 for `ModelWeightElement` entries where `count > 1` or `offsets != nil`:
   * Splits fused `to_qkv` projections ( applying `rotary_deinterleave` to the `q` and `k` slices while leaving `v` in identity order).
   * Splits or reorders packed FFN `up` / `gate` projections (preserving MOSS's packed FFN up/gate order and head-modulation order required by M3).

3. **`s4nnc` SQLite Metadata & `dim` Blob Packing (§3 & §22):**
   ```python
   DT_FLOAT32 = 16384    # 0x4000
   DT_FLOAT16 = 131072   # 0x20000
   DT_BFLOAT16 = 524288  # 0x80000

   DT_TYPE_CPU = 1
   DT_TYPE_GPU = 258

   DT_FORMAT_NCHW = 1
   DT_FORMAT_NHWC = 2

   def encode_dim_blob(shape: Sequence[int], ndim_slots: int = 12) -> bytes:
       padded = list(shape) + [0] * (ndim_slots - len(shape))
       return struct.pack(f"<{ndim_slots}i", *padded[:ndim_slots])
   ```
   * Enforces the §22 rank-3 norm rule: any tensor designated in the contract with `"rank_policy": "norm_1_1_n"` has its shape encoded as `[1, 1, N]` in the 48-byte `dim` blob rather than `[N]` or `[1, N]`.

#### C. `tools/dt_weellm/sqlite_store.py` — Draw Things `.ckpt` Store Reader/Writer
* Supports reading and writing `s4nnc` SQLite `tensors` tables (`name TEXT PRIMARY KEY, type INTEGER, format INTEGER, datatype INTEGER, dim BLOB, data BLOB`).
* Transparently handles externalized `-tensordata` sidecars (`TensorData.makeExternalData` discovered during U2 in `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` §7) so inspection, validation, and norm restoration work whether a checkpoint is inline or split across `<model>.ckpt` + `<model>.ckpt-tensordata`.
* Uses `PRAGMA journal_mode=DELETE; PRAGMA synchronous=OFF;` exclusively on temporary staging files (`<output>.tmp.<pid>`), followed by integrity verification and atomic `os.replace()` promotion.

---

### 5.2 Tool 1: `tools/dt_keymap_compiler.py` — Declarative Contract Compiler & Swift Mapper Auditor

#### Purpose
Eliminates hardcoded LTX 2.3 assumptions in `dt_preflight_safetensors.py`, `dt_validate_converted_ckpt.py`, and `dt_quantize.sh` by compiling a single source-of-truth JSON contract (`contracts/<profile>.json`) per model architecture, and statically auditing Draw Things Swift mappers before compilation.

#### CLI Interface
```bash
# 1. Compile or refresh a contract from WeeLLM keymaps + Swift source + optional official .ckpt
.venv/bin/python tools/dt_keymap_compiler.py compile \
  --profile moss_image \
  --weellm-keymap weellm/io/ggufs/keymaps/transformers/ \
  --swift-model DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Models/MOSSImage.swift \
  --ref-ckpt dt-models/moss_image_base_q8p.ckpt \
  --output contracts/moss_image.json

# 2. Statically audit Swift mapper code against known converter failure modes (§3a, §17, §18, §22, M3)
.venv/bin/python tools/dt_keymap_compiler.py audit-swift \
  --profile moss_image \
  --swift-model DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Models/MOSSImage.swift \
  --swift-importer DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift
```

#### Contract Schema (`contracts/<profile>.json`)
Each contract specifies:
* `profile`: Architecture identifier (`"ltx2_3"`, `"moss_image"`, `"flux2"`).
* `companions`: Default `autoencoder`, `text_encoder`, `clip_encoder`, `modifier`, `default_scale`, and `objective` (so `dt_quantize.sh` reads companion metadata from the contract rather than hardcoding LTX companions).
* `geometry_guards`:
  * `latent_unit_pixels`: `64` (or `16`).
  * `temporal_frame_rule`: `"8k+1"` for `ltx2_3` (§30 lesson) vs. `"single_frame"` for `moss_image`.
* `source_prefix_strip`: `["model.diffusion_model.", "model.", "diffusion_model."]`.
* `tensors`: Dictionary mapping canonical source keys (plus WeeLLM GGUF/ComfyUI aliases) to:
  * `dt_names`: Target SQLite key(s) (e.g., `["__dit__[t-x_q-0-0]"]`).
  * `split_policy`: `null`, `"chunk_o"`, `"chunk_i"`, or explicit `offsets`.
  * `rotary_interleaved`: `bool`, `num_heads`: `int`, `head_dim`: `int`.
  * `f16_store_datatype`: `"F16"`, `"F32"`, or `"BF16"`.
  * `quant_restore_policy`: `"none"`, `"f32_from_source"`, or `"bf16_from_source"`.
  * `dim_rank_policy`: `"preserve"` or `"norm_1_1_n"`.

#### Static Audit Rules (`audit-swift`)
1. **Unflagged Rotary QK-Norm Detector (§18):** Parses each attention block in the Swift model builder. If `to_q` / `to_k` has `interleaved: true` with `(numberOfHeads: H, headDimension: D)`, any `norm_q` / `norm_k` / `q_norm` / `k_norm` in the same scope without `interleaved: true, numberOfHeads: H, headDimension: D` triggers a hard `FAIL`.
2. **Sensitive Precision Downcast Detector (§3a & MOSS M3):** Flags any tensor whose runtime graph declaration uses `.Float32` or `.BFloat16` (such as LTX's 4 `aggregate_embed` tensors or MOSS's FP32 second timestep projection and AdaLN modulations) if its converter mapping lacks `isBF16: true` or FP32 preservation.
3. **Missing Generic Importer Branch Detector (M3):** Scans `ModelImporter.swift` for `fatalError()` or unhandled `switch modelVersion` cases for the target profile.

---

### 5.3 Tool 2: `tools/dt_direct_ckpt_builder.py` — Pure-Python Streaming Converter & Norm Restorer

#### Purpose
Provides a fast, deterministic, CPU-only conversion and post-quantization restoration engine driven by `contracts/<profile>.json`.
* **Why this is critical for MOSS M3:** As documented in `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` (§8 & §10 M3), upstream `ModelImporter.swift` has no MOSS architecture detection and hits `fatalError()` during import. `dt_direct_ckpt_builder.py` can convert official or custom MOSS `.safetensors` (single or sharded) or `.gguf` files directly into Draw Things `_f16.ckpt` files today—using the exact `MOSSImage.swift` tensor naming and precision contract—without waiting on or debugging `ModelImporter.swift` and `s4nnc` CPU graph execution.

#### CLI Interface
```bash
# Mode A: Convert Safetensors (single/sharded) or GGUF -> Draw Things f16 .ckpt
.venv/bin/python tools/dt_direct_ckpt_builder.py convert \
  --input /path/to/moss_or_ltx_source.safetensors \
  --contract contracts/moss_image.json \
  --output dt-models/moss_custom_f16.ckpt

# Mode B: Contract-driven post-quantization F32/BF16 & [1,1,N] norm restore
# (Replaces both dt_q6p_restore_f32_norms.py and dt_q6p_restore_qk_norms.py)
.venv/bin/python tools/dt_direct_ckpt_builder.py restore-norms \
  --quantized-ckpt dt-models/moss_custom_q8p.ckpt \
  --f16-source dt-models/moss_custom_f16.ckpt \
  --contract contracts/moss_image.json

# Mode C: Positive-control diff against an official Draw Things .ckpt
.venv/bin/python tools/dt_direct_ckpt_builder.py diff-ckpt \
  --candidate dt-models/moss_base_our_f16.ckpt \
  --reference dt-models/moss_image_base_q8p.ckpt \
  --contract contracts/moss_image.json
```

#### Key Safety Guarantees
* **Fail-Closed Coverage:** Any missing required source key, unexpected tensor shape, NaN/Inf in source weights, or unmapped tensor (unless explicitly marked ignored in the contract) aborts conversion with a non-zero exit code before promoting the output file.
* **Preserves Original FP32 Before Any Cast (M3 Requirement):** When converting MOSS or LTX sources, tensors marked `f16_store_datatype: "F32"` or `quant_restore_policy: "f32_from_source"` are widened directly from the source `BF16`/`F32` buffer to `F32`—never downcast to `F16` first—preventing underflow on MOSS's FP32-sensitive second timestep projection.
* **Reference-Free Own-Value Restoration:** During `restore-norms`, values always come from the model's *own* converted `_f16.ckpt` (which kept sensitive norms in `F32`), never overwriting a custom fine-tune's weights with official base-model weights.

---

### 5.4 Tool 3: `tools/dt_layer_parity_oracle.py` — Bounded-VRAM WeeLLM vs. Draw Things Bisection Harness

#### Purpose
Directly resolves our active **MOSS M1 runtime blockers** (`docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` §10 M1) and accelerates future architecture debugging on the 12 GiB RTX A3000 laptop.

#### Context of the Active MOSS M1 Blockers
In `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` (lines 495–534):
1. Running `moss_image_base_q8p.ckpt` with `qwen_3.5_9b_i5x.ckpt` at CFG 3 hits `SIGFPE` (`idiv %ebx`) in `_ccv_nnc_scaled_dot_product_attention_forw + 3614` because MOSS skips layer 22 on the unconditional branch by passing zero-length Q/K tensors (`dim = [0, ...]`).
2. When bypassing zero-length attention in LLDB, a 2-step CFG-3 run completes the stream but produces **131,072 non-finite (`NaN`) values** in the latent preview.
3. Native symbol probing (`backend-probe.c`) proved that CCV has **no CUDA backend (`0`)** for two operations used in `Qwen3_5LinearAttention`:
   * `GatedDelta` (`0x6de198ac`) — CPU reference `0x3d9883e5` exists; CUDA is `0`.
   * `RMSNormGated` (`0x9b8af346`) — CPU reference `0x3d9883e5` exists; CUDA is `0`.

#### How `dt_layer_parity_oracle.py` Uses WeeLLM to Pinpoint & Verify the Fix
Using WeeLLM's `weellm/models/text_encoders/qwen3_5_model.py`, `weellm/models/transformers/moss_image_transformer_2d_model.py`, and `weellm/pipelines/image/adapters/moss_image_pipeline.py` inside an isolated reference environment (`vram_budget=4.5` GiB, `ram_budget=18.0` GiB):

1. **Stage 1: Qwen3.5 Text Encoder & Linear Attention Oracle (`--stage text-encoder`)**
   ```bash
   .venv-weellm/bin/python tools/dt_layer_parity_oracle.py \
     --stage text-encoder \
     --model-path /path/to/qwen3.5-9b-source \
     --prompt "An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph." \
     --vram-budget 4.5 --ram-budget 18.0 \
     --out-dir output/oracle/moss_m1_qwen35
   ```
   * Dumps exact input token IDs (both positive prompt and empty negative prompt after chat template application).
   * Hooks every layer of `Qwen3_5Model` (specifically isolating the outputs of `GatedDelta` and `RMSNormGated` inside `Qwen3_5LinearAttention` vs. full attention blocks).
   * Writes `summary.json` (`min`, `max`, `mean`, `std`, `nan_count`, `inf_count`, `l2_norm` per layer) and `.npy` fixtures for Layer 0, Layer 15, final hidden state, and `GatedDelta` unit-test inputs/outputs.
   * **Immediate M1 Payoff:** Provides exact golden input/output tensors to test our CCV CUDA kernel or CPU-fallback implementation for `0x6de198ac` (`GatedDelta`) and `0x9b8af346` (`RMSNormGated`) in a standalone 1-second C/CUDA micro-test before rebuilding the full `gRPCServerCLI`.

2. **Stage 2: MOSS Text Refiner & 42-Layer DiT Step-1 Bisection (`--stage denoiser-step1`)**
   ```bash
   .venv-weellm/bin/python tools/dt_layer_parity_oracle.py \
     --stage denoiser-step1 \
     --model-path /path/to/moss-image-base \
     --embeds-dir output/oracle/moss_m1_qwen35 \
     --width 512 --height 512 --cfg 3.0 --shift 5.0 --seed 4242 \
     --vram-budget 5.0 --ram-budget 18.0 \
     --out-dir output/oracle/moss_m1_step1
   ```
   * Streams the 2-block MOSS text refiner, timestep embeddings (verifying dynamic range of the second FP32 timestep projection), and all 42 transformer blocks for both conditional and unconditional branches (verifying the exact numerical effect of skipping block 22 on the unconditional branch).
   * Records per-block activation health (`nan_count`, `max_abs`, `std`) so any divergence in Draw Things' `UNetFromNNC` can be bisected to the exact block index.

3. **Stage 3: 5-Second Single-Block Weight & Transform Verifier (`--stage verify-block`)**
   ```bash
   .venv-weellm/bin/python tools/dt_layer_parity_oracle.py \
     --stage verify-block \
     --arch moss_image --block-index 0 \
     --safetensors /path/to/source.safetensors \
     --ckpt dt-models/moss_custom_f16.ckpt \
     --contract contracts/moss_image.json
   ```
   * Reads block `0` weights from the raw `.safetensors` (via `seek_io.py`) and from the converted Draw Things `.ckpt` (via `sqlite_store.py`), applies `rotary_interleave` / un-splitting to reconstruct the source layout from the `.ckpt`, and verifies both weight parity (`max_abs_diff < 1e-3`) and single-block forward-pass output parity in under 5 seconds on CPU/GPU.

---

### 5.5 Tool 4: `tools/dt_lora_universal_bridge.py` — Multi-Architecture LoRA Normalizer & Static Verifier

#### Purpose
Generalizes our LTX 2.3 LoRA workflow (`tools/dt_lora_convert.sh`, `tools/dt_lora_quantize.sh`, and the §23/§24 static LoRA audits in `CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md`) to support MOSS-Image (M4), LTX 2.3/2.5, and Flux2 LoRAs from any training framework.

#### CLI Interface
```bash
# 1. Inspect & normalize a community LoRA (Kohya / ComfyUI / BFL -> Draw Things importer format)
.venv/bin/python tools/dt_lora_universal_bridge.py normalize \
  --input dt-models/community_moss_lora.safetensors \
  --contract contracts/moss_image.json \
  --output dt-models/community_moss_lora_normalized.safetensors

# 2. Run the §24 static mathematical verification on a converted Draw Things LoRA .ckpt
.venv/bin/python tools/dt_lora_universal_bridge.py verify \
  --source-safetensors dt-models/community_moss_lora_normalized.safetensors \
  --converted-ckpt dt-models/community_moss_lora_f16.ckpt \
  --contract contracts/moss_image.json
```

#### Automated Static Verification Checks (`verify` mode)
1. **Key Pairing & Completeness:** Confirms every `lora_A` / `lora_down` and `lora_B` / `lora_up` pair in the source `.safetensors` maps to a present `__down__` and `__up__` tensor in the SQLite `.ckpt` (zero silently dropped LoRA modules).
2. **§17/§24 Rotary Permutation Proof:**
   * For every attention projection marked `rotary_interleaved: true` in `contracts/<profile>.json` (e.g., `to_q`, `to_k` across self/cross/audio-video attentions):
     * Verifies `max_abs_diff(rotary_deinterleave(lora_B), ckpt_up) < 1e-4`.
     * Verifies `max_abs_diff(lora_A, ckpt_down) < 1e-4` (since `lora_A` acts on the input dimension and must remain unpermuted).
   * For non-rotary projections (`to_v`, `to_out`, FFN projections):
     * Verifies both `ckpt_up` and `ckpt_down` match `IDENTITY` (`max_abs_diff < 1e-4`).
3. **Fused Projection & Alpha Scaling Check (MOSS M4):**
   * Verifies that LoRAs targeting fused QKV or packed FFN up/gate layers have their `lora_B` (`__up__`) slices split at the exact contract offsets and scaled by `alpha / rank` where applicable.

---

## 6. Implementation Status & Verified Artifacts (Completed October 24, 2026)

The following is the original October 24 prototype inventory, not evidence that all five milestones (`W0`-`W4`) passed native acceptance. See the October 26 hardening status above for implemented fixes, executed checks, and remaining gates:

| Milestone | Delivered Scope | Active Code / Contract Artifacts | Verified Capabilities |
| :--- | :--- | :--- | :--- |
| **W0 (Core Primitives)** | Zero-copy I/O, block dequantizers, CPU NumPy transforms, SQLite `.ckpt` store reader/writer. | `tools/dt_weellm/transforms.py`<br>`tools/dt_weellm/seek_io.py`<br>`tools/dt_weellm/dequant.py`<br>`tools/dt_weellm/sqlite_store.py` | §17/§18 rotary de-interleave/interleave, §22 rank-3 `[1, 1, N]` `dim` packing, §3a BF16 conversions, all GGML dequantizers (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`), ComfyUI FP8, sharded safetensors, externalized `-tensordata` sidecars. |
| **W1 (Contract Compiler & Swift Auditor)** | Declarative contract schemas, LTX 2.3 & MOSS-Image contracts, preflight/validation contract hooks. | `tools/dt_keymap_compiler.py`<br>`contracts/schema.json`<br>`contracts/ltx2_3.json`<br>`contracts/moss_image.json`<br>`tools/dt_preflight_safetensors.py`<br>`tools/dt_validate_converted_ckpt.py` | Compiles contracts; statically audits Swift mappers (`audit-swift`) for §18 QK-norm unflagged rotary de-interleaving (`DT-CONV-018`) and §3a unflagged aggregate embeds (`DT-CONV-003A`); validates `.ckpt` tensor counts and required prefixes fail-closed. |
| **W2 (Parity Oracle & Qwen3.5 Golden Vectors)** | Single-block forward/weight verifier, Qwen3.5 linear attention reference implementation, checkpoint health audit. | `tools/dt_layer_parity_oracle.py` | Generates golden NPZ fixtures for CCV Qwen3.5 `GatedDelta` (`0x6de198ac`) and `RMSNormGated` (`0x9b8af346`) to resolve MOSS M1; sweeps all transformer blocks for parity; audits checkpoints for `NaN`/`Inf` and rank-3 norm policy. |
| **W3 (Direct Checkpoint Builder & Norm Restorer)** | Pure-Python streaming converter from `.safetensors`/`.gguf` to `.ckpt`, contract-driven norm restorer, checkpoint diff inspector. | `tools/dt_direct_ckpt_builder.py`<br>`tools/dt_convert.sh`<br>`tools/dt_quantize.sh` | Bypasses `ModelImporter.swift` `fatalError()` branches to convert MOSS M3 and GGUF sources; restores F32 norms and `[1, 1, N]` dims using model's own F16 source; dynamically registers MOSS companions in `custom.json`; compares candidate vs. reference stores (`diff-ckpt`). |
| **W4 (Universal LoRA Bridge & Static Verifier)** | Key normalizer (Kohya/LyCORIS/ComfyUI -> Diffusers), automated §24 mathematical static LoRA verifier. | `tools/dt_lora_universal_bridge.py`<br>`tools/dt_lora_convert.sh`<br>`tools/dt_weellm/keymaps.py` | Automatically unmangles community LoRA keys before `lora-converter`; statically proves that rotary `to_q`/`to_k` `__up__` matrices are de-interleaved while all `__down__` and non-rotary matrices remain pure identity. |

---

## 7. Practical Playbook: End-to-End MOSS Inference, Conversion & Quantization

This section provides the exact execution runbook for resolving the active MOSS milestones in `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md`.

```
[ Raw HuggingFace / GGUF / LoRA ]
                 │
                 ▼
    ┌─────────────────────────┐
    │ dt_keymap_compiler.py   │ ──► Audits Swift mappers & compiles contracts/<profile>.json
    └─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ dt_direct_ckpt_builder  │ ──► Converts SF/GGUF directly to Draw Things .ckpt (Bypasses Swift fatalError)
    └─────────────────────────┘
                 │
                 ├──► diff-ckpt ──► Byte-verifies against official Draw Things .ckpt
                 ▼
    ┌─────────────────────────┐
    │ dt_quantize.sh +        │ ──► Quantizes to Q8P/Q6P/I4X + contract-driven F32 norm restoration
    │ contracts/moss_image    │     + writes correct MOSS companions to custom.json
    └─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ dt_layer_parity_oracle  │ ──► Generates golden Qwen3.5 test vectors (fixes M1 CUDA backends)
    └─────────────────────────┘     + inspects .ckpt health for NaNs/rank bugs
```

### Phase A: Unblocking Milestone M1 (Qwen3.5 CUDA Backends & Zero-Length Attention)

In `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md` (§10 M1, lines 495–534):
1. MOSS skips block 22 on the unconditional branch using zero-length tensors, causing `SIGFPE` (`idiv %ebx`) in `_ccv_nnc_scaled_dot_product_attention_forw + 3614`.
2. When bypassed, CFG-3 produces **131,072 non-finite (`NaN`) preview values**, and CCV's registry confirms **no CUDA backend (`0`)** for Qwen3.5's `GatedDelta` (`0x6de198ac`) and `RMSNormGated` (`0x9b8af346`).

**Action Step 1: Generate Deterministic Golden Test Vectors for CCV**
Run the reference implementation to produce unit-test vectors:
```bash
.venv/bin/python tools/dt_layer_parity_oracle.py qwen35-linear-attn-fixture \
  --out-dir output/oracle/qwen35_fixtures
```
* **Output:** `output/oracle/qwen35_fixtures/qwen35_linear_attention_golden.npz` and `manifest.json`.
* **Use:** When adding a CPU-fallback or CUDA kernel in `checkouts/ccv/lib/nnc/cmd/`, write a standalone C micro-test asserting that `ccv_nnc_cmd_exec` matches `gated_delta_out` and `rmsnorm_gated_out` with `max_abs_diff < 1e-4`. This proves text-encoder correctness in isolation before launching the full server.

---

### Phase B: Milestone M3 Custom MOSS Conversion (Bypassing Swift `fatalError`)

Upstream `ModelImporter.swift` lacks MOSS architecture detection and hits `fatalError()` when attempting to import custom MOSS checkpoints.

**Action Step 2: Convert MOSS Safetensors or GGUF Directly to Draw Things `.ckpt`**
Convert directly in Python without touching Swift:
```bash
# Direct Python conversion via dt_direct_ckpt_builder.py
bash tools/dt_convert.sh dt-models/moss_image_base.safetensors \
  --direct \
  --contract contracts/moss_image.json \
  --name moss_image_base_our
```
* **What this executes:**
  1. Preflight verifies source tensors against `contracts/moss_image.json`.
  2. Streams the 42 main DiT blocks and 2 text refiner blocks without loading the 10GB file into RAM.
  3. Rotary-deinterleaves all Q and K projections ($24\text{ heads} \times 128\text{ dim}$) using the proven §17 formula.
  4. Widens `timestep_embedder.linear_2` directly from source to Float32, preventing underflow.
  5. Enforces `[1, 1, N]` rank-3 dimensions on all AdaLN and QK-norm SQLite blobs (§22).
  6. Outputs `dt-models/moss_image_base_our_f16.ckpt`.

---

### Phase C: Milestone M3 Static Parity Verification against Official Draw Things Weights

Before spending 2 hours quantizing, prove that the newly converted F16 checkpoint is numerically and structurally identical to the official Draw Things baseline.

**Action Step 3: Run Checkpoint Diff**
```bash
.venv/bin/python tools/dt_direct_ckpt_builder.py diff-ckpt \
  --candidate dt-models/moss_image_base_our_f16.ckpt \
  --reference dt-models/moss_image_base_q8p.ckpt \
  --contract contracts/moss_image.json
```
* **Acceptance:**
  * `missing_in_candidate = 0`, `extra_in_candidate = 0`.
  * `metadata_mismatch_count = 0` (verifying datatype, format, and dimension alignment).
  * `max_observed_diff < 1e-3` across all matching float tensors.

---

### Phase D: Milestone M3 MOSS Quantization, Contract Norm Restoration & Alias Registration

Quantizing to `q8p`, `q6p`, or `i4x` palettizes the weights. If sensitive modulation tables and QK-norms are palettized, the DiT diverges to NaN.

**Action Step 4: Quantize and Restore Norms Using Contract**
```bash
bash tools/dt_quantize.sh moss_image_base_our \
  --contract contracts/moss_image.json \
  --codec q8p \
  --model-version moss_image
```
* **What this executes:**
  1. Quantizes the large linear projections to `q8p` (or `q6p` / `q4p`).
  2. Invokes `dt_direct_ckpt_builder.py restore-norms` with `contracts/moss_image.json`: restores all AdaLN modulations, QK-norms, and timestep embedders to Float32 with `[1, 1, N]` rank-3 dims from `moss_image_base_our_f16.ckpt`.
  3. Auto-registers the alias in `dt-models/custom.json` with official MOSS companions:
     * `"version": "moss_image"`
     * `"autoencoder": "flux_2_vae_f16.ckpt"`
     * `"text_encoder": "qwen_3.5_9b_i5x.ckpt"`
     * `"default_scale": 16`
     * `"objective": {"u": {"condition_scale": 1000}}`

---

### Phase E: Pre-Inference Checkpoint Health Audit

Verify checkpoint integrity with zero GPU cost before running canary inference:
```bash
.venv/bin/python tools/dt_layer_parity_oracle.py inspect-ckpt-health \
  --ckpt dt-models/moss_image_base_our_q8p.ckpt \
  --contract contracts/moss_image.json
```
* **Acceptance:**
  * `finite_checked == total_tensors` (0 `NaN` or `Inf` values in stored weights).
  * 0 rank-3 policy violations on AdaLN and QK-norm weights.

---

### Phase F: Milestone M4 Universal MOSS LoRA Ingestion & Static Verification

When community LoRAs arrive in Kohya, ComfyUI, or BFL formats:

**Action Step 5: Normalize, Convert & Statically Audit LoRA**
```bash
bash tools/dt_lora_convert.sh dt-models/community_moss_lora.safetensors \
  --name moss_style_lora \
  --version moss_image \
  --normalize \
  --verify-post \
  --contract contracts/moss_image.json
```
* **What this executes:**
  1. `--normalize`: Unmangles Kohya underscore keys (e.g. `lora_unet_transformer_blocks_0_attn_to_q...`) into standard `diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight` format so `lora-converter` imports 100% of the parameters without dropping keys.
  2. `--verify-post`: Runs `dt_lora_universal_bridge.py verify` to mathematically prove that every rotary `__up__` matrix was de-interleaved ($24\text{ heads} \times 128\text{ dim}$) while all `__down__` matrices remain identity (`max_abs_diff < 1e-4`).

---

## 8. Command Reference & Quick Index

| Purpose | Command |
| :--- | :--- |
| **Run WeeLLM Unit Tests** | `.venv/bin/python -m unittest tools.test_dt_weellm -v` |
| **Compile/Refresh Contract** | `.venv/bin/python tools/dt_keymap_compiler.py compile --profile {ltx2_3\|moss_image} --output contracts/{profile}.json` |
| **Audit Swift Mappers for §18/§3a Bugs** | `.venv/bin/python tools/dt_keymap_compiler.py audit-swift DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Models/LTX2.swift DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift` |
| **Generate Qwen3.5 Golden Fixtures** | `.venv/bin/python tools/dt_layer_parity_oracle.py qwen35-linear-attn-fixture --out-dir output/oracle/qwen35` |
| **Direct Safetensors/GGUF Conversion** | `bash tools/dt_convert.sh <model.safetensors> --direct --contract contracts/<profile>.json` |
| **Checkpoint Parity Diff** | `.venv/bin/python tools/dt_direct_ckpt_builder.py diff-ckpt --candidate <cand.ckpt> --reference <ref.ckpt> --contract contracts/<profile>.json` |
| **Contract-Driven Quantize & Restore** | `bash tools/dt_quantize.sh <name> --contract contracts/<profile>.json --codec q8p --model-version <version>` |
| **Inspect Checkpoint Health** | `.venv/bin/python tools/dt_layer_parity_oracle.py inspect-ckpt-health --ckpt <model.ckpt> --contract contracts/<profile>.json` |
| **Universal LoRA Convert & Verify** | `bash tools/dt_lora_convert.sh <lora.safetensors> --version <version> --normalize --verify-post --contract contracts/<profile>.json` |
| **Single-Block Weight/Transform Check** | `.venv/bin/python tools/dt_layer_parity_oracle.py verify-block --safetensors <src.safetensors> --ckpt <out.ckpt> --contract contracts/<profile>.json` |

