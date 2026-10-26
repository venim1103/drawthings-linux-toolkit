# Qwen3.5 CUDA Forward Patch Profile

This profile is separate from the legacy CPU quantization bundle. It adds CUDA forward backends for `GatedDelta` and `RMSNormGated` and safe empty-forward SDPA handling without changing Package files, assertion settings, CPU backends, or the installed runtime.

The manifest pins community, s4nnc, and CCV revisions and SHA-256 hashes for the two modified registration files and original CUDA SDPA source. The `ccv/` tree contains patched snapshots; `patches/ccv.patch` is the generated unified diff. The patch tool never falls back to copying snapshots over unexpected upstream files.

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
DRAWTHINGS_CCV_OVERRIDE="$PWD/output/upgrades/ccv-69b1ed1f-qwen35-cuda-empty-sdpa" \
DRAWTHINGS_CANDIDATE_TAG=0621d819-qwen35-cuda-empty-sdpa \
DT_BUILD_JOBS=16 bash tools/drawthings_build_candidate.sh build

.venv/bin/python tools/dt_layer_parity_oracle.py native-backends \
	--binary output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI \
	--ccv-source output/upgrades/ccv-69b1ed1f-qwen35-cuda-empty-sdpa \
	--require-qwen-cuda --execute-qwen --execute-sdpa \
	--report output/upgrades/native-backends-qwen35-cuda-empty-sdpa.json
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

## Coverage And Limits

- RMSNorm: one last-axis reduction on contiguous tensors, optional affine scale, mixed F32/F16/BF16 value/gate/scale types, FP32 accumulation, in-place output, zero-rank empty operations.
- Gated delta: grouped key/value heads, FP32 recurrent state and decay, F32/F16/BF16 Q/K/V/beta, scalar-per-head or per-key-channel decay, explicit log-decay, nonzero initial state, checkpoints, and in-place state when checkpoint count is zero. Key dimension is limited to 1,024.
- Unsupported layouts, types, or contracts return `CCV_NNC_EXEC_INVALID`; no backward/training implementation is added.
- Tests include real Qwen3.5 dimensions (16 key heads, 32 value heads, 128-dimensional heads), batched and non-square cases. These are correctness-first kernels, not tuned performance results.

On the RTX A3000/SM86, 60 RMSNorm and 108 gated-delta numerical cases passed, plus invalid/empty contract tests. Observed maximum errors were approximately `9.54e-7` for RMSNorm, `4.47e-8` for FP32 gated-delta state, and `6.10e-5` for gated-delta output including reduced-precision rounding. FP32 checks use absolute tolerance `2e-6` and relative tolerance `2e-5`; reduced-precision output checks use separately declared tolerances in the tests.

These results establish standalone operator correctness, linked-server operator integration, and the empty SDPA regression fix, not full MOSS inference. Locating the remaining non-finite activation, complete text-encoder/denoiser parity, and an accepted final image remain separate gates.