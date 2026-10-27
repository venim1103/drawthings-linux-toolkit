# Qwen3.5 CUDA Forward Patch Profile

This profile is separate from the legacy CPU quantization bundle. It adds CUDA forward backends for `GatedDelta` and `RMSNormGated`, safe empty-forward SDPA handling, decoded packed-embedding lookup, and scaled RMSNorm forward support without changing Package files, assertion settings, CPU backends, or the installed runtime.

The manifest pins community, s4nnc, and CCV revisions and SHA-256 hashes for the two modified registration files and original CUDA SDPA/index-select/RMSNorm sources. The `ccv/` tree contains patched snapshots; `patches/ccv.patch` is the generated unified diff. The patch tool never falls back to copying snapshots over unexpected upstream files.

## Validate

```bash
.venv/bin/python tools/drawthings_cuda_patch.py verify --ccv-source /path/to/pinned/ccv
DT_RUN_CUDA_TESTS=1 .venv/bin/python -m unittest tools.test_dt_weellm.CudaQwenReferenceTests -v
```

`verify` applies the unified diff in a temporary tree, checks snapshot equality, and verifies a second application is a no-op. `check` reports pristine/applied state without modifying the source. Mixed files or upstream drift are rejected.

GPU tests compile the snapshots with `nvcc` and compare against the NumPy reference already validated against the pinned native CPU implementation. Use `DT_CCV_SOURCE_DIR` for pinned CCV headers and `DT_CUDA_ARCH` to override the default `sm_86`. Explicit GPU opt-in requires an available compiler/device; CPU-only tests do not imply CUDA acceptance.

## Apply To An Isolated Candidate

```bash
.venv/bin/python tools/drawthings_cuda_patch.py apply --ccv-source /path/to/isolated/pinned/ccv
```

Do not point this at the legacy converter checkout or the only pristine candidate dependency tree. Preserve the original candidate binary and use a separate build/output root for integration. The native build must regenerate CCV backend bindings after registration changes. No server build or promotion is performed by this tool.

To regenerate after snapshot changes, supply pristine pinned base files:

```bash
.venv/bin/python tools/drawthings_cuda_patch.py generate --ccv-source /path/to/pristine/pinned/ccv
```

## Isolated Server Integration

The candidate helper accepts `DRAWTHINGS_CCV_OVERRIDE` only for an applied, hash-verified profile in a private directory under `output/upgrades`. A distinct candidate directory, Bazel root, and artifact tag are required, preventing reuse of the pristine candidate's build outputs and provenance files.

Ruby is a native-build prerequisite. Before Bazel runs, the helper automatically invokes the patch tool's `prepare` action, which regenerates bindings with upstream `ruby build-cmd.rb .` inside the private CCV copy and verifies both GPU registration calls. `prepare` refuses non-private source paths before modifying files. Plain `apply` does not regenerate bindings: CCV's Bazel rules consume a pre-generated command table, so compiling the new CUDA files alone is not sufficient.

```bash
DRAWTHINGS_CANDIDATE_DIR="$PWD/output/upgrades/community-0621d819-qwen35-cuda" \
DRAWTHINGS_BAZEL_ROOT="$PWD/.cache/bazel-qwen35-cuda" \
DRAWTHINGS_CCV_OVERRIDE="$PWD/output/upgrades/ccv-69b1ed1f-qwen35-cuda-scaled-rmsnorm" \
DRAWTHINGS_CANDIDATE_TAG=0621d819-qwen35-cuda-scaled-rmsnorm \
DT_BUILD_JOBS=16 bash tools/drawthings_build_candidate.sh build

.venv/bin/python tools/dt_layer_parity_oracle.py native-backends \
	--binary output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI \
	--ccv-source output/upgrades/ccv-69b1ed1f-qwen35-cuda-scaled-rmsnorm \
	--require-qwen-cuda --execute-qwen --execute-sdpa --execute-embedding --execute-packed-gemm --execute-scaled-rmsnorm \
	--report output/upgrades/native-backends-qwen35-scaled-rmsnorm.json
```

Create the detached community checkout and independent CCV copy before this command; apply the profile only to that private CCV copy. An earlier applied profile is not an upgrade base: create a fresh copy of the pinned pristine CCV source when the profile changes. The report checks six CUDA precision selections, CPU/SDPA controls, the binary hash, and optionally real linked-runtime GPU operations against CPU values. The pristine server is a tested negative control: the required-Qwen-CUDA gate rejects it even though CPU and SDPA controls are present. Integration is accepted only after both the full build and the positive linked-runtime check pass.

### Verified Integration

The initial operator-integrated candidate built with 16 jobs and passed the linked-runtime gate. Its SHA-256 was `5cfdd16e4110e5248fcb4f37fca7da2dc26790fdddcca1f9682f22e24ba1a39d`; the local acceptance report is `output/upgrades/native-backends-qwen35-cuda.json`. A preserved copy now lives at `output/upgrades/gRPCServerCLI-0621d819-qwen35-cuda-before-empty-sdpa`.

All six Qwen F32/F16/BF16 selections resolve to GPU_REF, with SDPA and both CPU controls also matching. Actual native allocation, transfer, command dispatch, and GPU-to-CPU transfer passed for tiny F32 gated-delta and affine RMSNorm cases; all outputs and recurrent-state checkpoints were finite and within absolute `2e-6` of the pinned CPU implementations. The pristine candidate also passes a negative test that explicitly refuses GPU execution when registrations are absent. The combined CUDA-enabled suite passed all 55 tests.

The first full build correctly failed the registration gate because bindings were stale. Its log, hash, and negative registry report are retained with `before-bindings` names under `output/upgrades`. Automatic upstream regeneration fixed the integration; no generated file was hand-edited. This candidate is not installed or promoted. The original candidate, installed runtime, legacy CPU tools, and active catalogs remain unchanged.

### Empty SDPA Increment

An all-empty call against the preserved pre-fix binary reproduced the original SIGFPE at `_ccv_nnc_scaled_dot_product_attention_forw + 3614`. The profile now returns success before shape/head arithmetic only when Q, K, V, the attention output, and all present auxiliary outputs have rank zero. Partially empty contracts or nonempty auxiliary outputs return `CCV_NNC_EXEC_INVALID`; normal nonempty attention is unchanged.

The rebuilt candidate has SHA-256 `5cca983e5883b5e9091f24d34475af99bc6054e67aa51da1f7f6fa8e7441febd`. The 16-job incremental build reused the isolated cache and completed in about 18 seconds. `--execute-sdpa` checks eight empty/mixed/projected-output contracts and actual nonempty grouped FP16 attention with two batches, four query heads, two key/value heads, unequal query/key lengths, and analytically known output. The native report records `sdpa_contracts=[0,-1,-1,-1,-1,-1,0,-1]` and `sdpa_cuda_execution=0`. All **56** combined tests passed with CUDA enabled, including rejection of the pre-fix binary and the existing Qwen numerical/dispatch gates.

The private two-step guided MOSS canary no longer crashes, but still produces no final image: its sole preview contains 131,072 non-finite values. Separate bounded live diagnostics show that the first real FP16 gated-delta and gated-RMSNorm calls return success with finite inputs and outputs. This does not prove later layers remain finite. No 50-step run or image-quality acceptance is claimed; the next task is locating the first later non-finite activation.

### Packed Embedding Increment

The next diagnostic identified non-finite values already in the first nonempty Qwen token lookup: a packed Q5_K table reached CUDA index-select, which only handled ordinary dense types and interpreted the packed bytes as floats in release builds. The selected `[11,4096]` output contained 1,848 non-finite values before any decoder normalization. The lookup now prefetches and decodes QX data with CCV's existing decoder, selects rows using the decoded datatype, registers QX support, and rejects unsupported contracts. This correctness-first path materializes the dense embedding table in the stream workspace; it is not an optimized row-only decoder.

The current separate candidate has SHA-256 `920bf727c5a27a7d1ad90735735c74b7edfbbc1459598cea018f71e2d713dd96`; its pre-fix control is `output/upgrades/gRPCServerCLI-0621d819-before-packed-embedding`. A fresh private CCV copy and 16-job incremental build preserve the previous controls and default runtime. `--execute-embedding` tests Q5_K/Q6_K tables with repeated/reordered indices across F32/F16/BF16. Fixtures use the pinned native CPU quantizer, including CCV's aligned row-scale footer, and native CPU decode as reference. F32 comparison is exact; reduced precision permits at most two ULPs while rejecting non-finite values and sign errors. The pre-fix binary explicitly refuses the packed test. All **57** combined CUDA-enabled tests pass.

Both private guided MOSS canaries now produce exactly one finite final image and complete stream/chunks/post-Echo: two steps and the original 50-step CFG-3/shift-5 profile. Their acceptance metadata records zero non-finite values. The full run's config hash remains `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`; output is `output/q6p_canary_moss_base_q8p_512_packed_embedding_50steps_20261026`. Whole-device telemetry covers about 475 seconds, with a 2,305 MiB observed peak and 59 C maximum; one utilization reading is unavailable. These include startup and are not allocator-exact performance results.

Finite output is **not image-quality acceptance**. The 50-step result is a repeated green structure rather than the requested orange teapot. Reference activation/sampler/decoder parity and prompt fidelity remain unresolved; no default promotion or SFT claim follows from these results. The remaining `GEMM_EX` warnings were separately traced to zero-sized skipped operations with matching FP16 operands, not a mixed-datatype failure.

### Scaled RMSNorm And First Prompt-Following Base Result: October 27

A complete Qwen recurrent-layer trace, unlike finiteness-only checks, stopped at `CCV_NNC_RMSNORM_FORWARD` with status `-1`. The CUDA implementation explicitly rejected every `scale != 1`; Qwen's non-affine key/query L2 normalization requires `1/sqrt(128)` and `1/128`. This was a real model-execution gap despite finite downstream artifacts. The preserved `output/upgrades/gRPCServerCLI-0621d819-before-scaled-rmsnorm` remains a negative control.

The profile now pins the original RMSNorm CUDA source and supports finite non-unit forward scales on contiguous F32/F16/BF16 output: run the existing scale-one implementation, then scale only the normalized output in place. Saved inverse-standard-deviation values are unchanged. Scale-one behavior and backward rejection remain intact. The numerical gate covers **36** affine/non-affine dtype/scale cases, including zero and negative output scales, zero input rows, and saved-statistics checks; the pre-fix binary is explicitly rejected. The complete recurrent-layer replay now executes all 29 operations successfully, including both formerly rejected Q/K scales.

The current separate candidate is SHA-256 `1d27f8d444d6211322496d3ce20871173b5d7d5942401bbe07fba9fa216a91b5`, built with 16 jobs in about 21 seconds against a fresh private CCV copy. All existing native gates plus `--execute-scaled-rmsnorm` pass; the full CUDA-enabled suite passes **67 tests**.

Both standard private canaries pass: two steps and the original 50-step CFG-3/shift-5/seed-4242 512x512 profile, with unchanged full config hash `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`. The full image now visibly depicts the requested orange teapot on a white table beside a green plant, instead of the previous unrelated texture. This is the first prompt-following Base bring-up observation for this profile, not a cross-framework quality benchmark or universal acceptance. Raw/PNG/acceptance artifacts are under `output/q6p_canary_moss_base_q8p_scaled_rmsnorm_{2steps,50steps}_20261027`. Observed whole-device peak is 2,305 MiB for both, telemetry spans about 75/129 seconds including startup, and maximum temperatures are 57/65 C; no samples are missing.

The isolated offline Transformers 5.3.0 reference is installed only under the private output tree, with no change to normal toolkit imports and no new weights downloaded. A pre-fix full-attention-layer comparison produced finite outputs with roughly 2.7% RMS difference from FP32 reference, or 1.3% when native activation quantization was modelled. This did not expose the later-discovered unsupported recurrent normalization. The main lesson is to check operation return statuses and complete recurrent paths, not treat finite early primitives as encoder certification. Repeatability, other prompts/resolutions, SFT, backward behavior, and default promotion remain separate gates.

### Base Repeatability And SFT q8p Bring-Up: October 27

The same native candidate now passes a fresh-process Base repeat with identical config, raw final tensor bytes, decoded values, and PNG, plus a distinct prompt/seed depicting a blue vase, three white daisies, and a red book. Separately pinned official SFT q8p (`55d7bdb2...`, denoiser SHA-256 `e5631ac1...`) passes two-step and standard 50-step image/stream gates using independent writable model copies; the full image visibly matches the orange-teapot prompt. Its Qwen/VAE companions are byte-identical to Base's, and all private payload hashes remain unchanged afterward. No native patch or default/runtime/catalog change was needed for SFT. Exact provenance, configs, telemetry, artifact paths, and reproduction instructions are in the [upgrade/MOSS plan](../../docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md).

These are bounded prompt-following observations, not universal quality or cross-framework equivalence. Official packed 4-bit S, other resolutions, backward behavior, and promotion remain separate gates.

### Official 4-bit S Numerical And Image Gates: October 27

Official SFT `i4x` at the pinned revision passes separate two-step and 50-step image/stream gates; the full image visibly follows the same teapot scene as q8p. The actual metadata inventory contains 308 Q4_K tensors with the Hadamard-256 modifier, eight retained q8p tensors, 199 ezm7 tensors and nine raw tensors. It is not a renamed palettized q4p file. All private model hashes remain unchanged.

`--execute-packed-gemm` now includes Q4_K both plain and with the **regular normalized H256** used by the official artifact. Its independent CPU activation reference uses the pinned H4-based convention, not a Sylvester/Walsh transform; rotated activations are quantized per row for the packed expectation, while the separate dense-GPU control remains unquantized. Q5_K/Q6_K coverage stays intact. The linked report `output/upgrades/native-backends-qwen35-q4k-h256.json` and all **67 CUDA-enabled tests** pass.

The i4x denoiser uses about 44.1% less disk space, but this offloaded run has the same observed 2,305 MiB whole-device peak as q8p. A single matched image supports this profile, not universal quality equivalence, performance certification, custom quantization, or default promotion. Exact hashes, configs, telemetry caveats and reproduction are in the upgrade/MOSS plan. Other resolutions remain the next gate.

## Coverage And Limits

### Complete MOSS Block Capture

`compare-moss-block --capture-dir <directory>` propagates an independent CPU calculation through the supported 32-operation captured MOSS block graph, starting from the original block input. It does not reset reference intermediates to native outputs. The command validates topology, status, tensor headers, paths, optional inputs, modulation shapes, and CPU work limits; reports contain per-stage errors and capture hashes. It is an executed-block arithmetic check, not full-model or source-checkpoint parity.

```bash
OPENBLAS_NUM_THREADS=16 .venv/bin/python tools/dt_layer_parity_oracle.py compare-moss-block \
	--capture-dir output/upgrades/moss-image-base-add235e9/block0-trace-20261027 \
	--report output/upgrades/moss-image-base-add235e9/block0-trace-20261027/oracle-block-reference.json
```

On the real `[1,1055,3072]` block-0 capture, all 28 checked stages pass `atol=0.01, rtol=0.01`; final relative RMS error is about `1.364e-4` (0.014%). The captured block exit exactly equals block 1's input. Separate read-only checks independently CPU-decode the named official checkpoint: all seven matrix weights and all four learned normalization vectors match runtime values exactly; the FFN up/gate roles agree, and all six modulation chunks match a CPU evaluation from timestep 1000 and named stored parameters. No runtime patch follows from this passing result. Tiny analytic, corrupted-output, incomplete-capture, unsafe-path, and non-finite tests bring the combined CUDA-enabled suite to **66 passing tests**.

The private trace is about 1.02 GB and was bounded to 64 MiB per tensor and 2 GiB total. It includes typed command metadata, original tensor identities, decoded packed weights, and explicit inferior cleanup. The q8p CPU decoder is compiled from the pinned CCV source and independently checked against a known palette. The private checkpoint check also decodes only needed raw/ezm7 values with strict bounds; this does not add general ezm7 support to the public checkpoint reader.

### Captured Activation Checks

`tools/dt_layer_parity_oracle.py compare-capture` compares bounded raw native captures with independent CPU linear, adjacent-pair rotary, or unmasked noncausal grouped-attention math. The prefix identifies `<prefix>_input_0.bin`, further inputs, and `<prefix>_output_0.bin`; optional attention output 1 is log-sum-exp. Reports record capture hashes, shapes, formats, datatypes, tolerances, and measured errors. Non-finite/malformed data, unsupported masked/projected attention, and excessive reference allocations fail explicitly. Capture scope is an individual operation, not full-model parity.

```bash
.venv/bin/python tools/dt_layer_parity_oracle.py compare-capture \
	--prefix output/upgrades/moss-image-base-add235e9/text_attention_capture \
	--operation sdpa --scale 1 --atol 0.005 --rtol 0.005 \
	--report output/upgrades/moss-image-base-add235e9/text-attention-parity.json
```

Real MOSS captures pass text projection, text attention/log-sum-exp, rotary multiplication, and FP32 timestep projection checks. The latter's loaded weights and bias match the checkpoint exactly. `--execute-packed-gemm` additionally checks nonuniform Q4_K/Q5_K/Q6_K weights and activations against native CPU decoding, including the regular-H256 Q4_K variant, with a separate same-weight dense GPU control. For more than four rows, the native packed fast path intentionally quantizes activations to int8; its CPU expectation explicitly models that rounding in the appropriate activation space, while the dense control uses unquantized activations. A constant-row-only fixture would miss this distinction. The combined CUDA-enabled suite passes **67 tests**.

A 50-step CFG-1 diagnostic also returns a finite but unrelated image, so the special negative branch is not the sole cause of the quality failure. A debugger-only dense-fallback A/B produces non-finite encoder conditioning and was rejected, not promoted. The on-disk candidate stays `920bf727...`; neither experiment changes model files or runtime defaults. Quality remains unaccepted.

- RMSNorm: one last-axis reduction on contiguous tensors, optional affine scale, mixed F32/F16/BF16 value/gate/scale types, FP32 accumulation, in-place output, zero-rank empty operations.
- Gated delta: grouped key/value heads, FP32 recurrent state and decay, F32/F16/BF16 Q/K/V/beta, scalar-per-head or per-key-channel decay, explicit log-decay, nonzero initial state, checkpoints, and in-place state when checkpoint count is zero. Key dimension is limited to 1,024.
- Unsupported layouts, types, or contracts return `CCV_NNC_EXEC_INVALID`; no backward/training implementation is added.
- Tests include real Qwen3.5 dimensions (16 key heads, 32 value heads, 128-dimensional heads), batched and non-square cases. These are correctness-first kernels, not tuned performance results.

On the RTX A3000/SM86, 60 RMSNorm and 108 gated-delta numerical cases passed, plus invalid/empty contract tests. Observed maximum errors were approximately `9.54e-7` for RMSNorm, `4.47e-8` for FP32 gated-delta state, and `6.10e-5` for gated-delta output including reduced-precision rounding. FP32 checks use absolute tolerance `2e-6` and relative tolerance `2e-5`; reduced-precision output checks use separately declared tolerances in the tests.

These results establish tested operator correctness, linked-server integration, complete block-0 comparison, exact Base repeatability for one fixed profile, and finite prompt-following Base/SFT q8p and official SFT 4-bit S images. Complete cross-framework model parity, broader prompt/resolution/codec/layout coverage, and production readiness remain separate gates.