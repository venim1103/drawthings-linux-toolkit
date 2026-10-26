# Qwen3.5 CUDA Forward Patch Profile

This profile is separate from the legacy CPU quantization bundle. It adds CUDA forward backends for `GatedDelta` and `RMSNormGated` without changing Package files, assertion settings, CPU backends, or the installed runtime.

The manifest pins community, s4nnc, and CCV revisions and SHA-256 hashes for the two modified registration files. The `ccv/` tree contains patched snapshots; `patches/ccv.patch` is the generated unified diff. The patch tool never falls back to copying snapshots over unexpected upstream files.

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
DRAWTHINGS_CCV_OVERRIDE="$PWD/output/upgrades/ccv-69b1ed1f-qwen35-cuda" \
DRAWTHINGS_CANDIDATE_TAG=0621d819-qwen35-cuda \
DT_BUILD_JOBS=16 bash tools/drawthings_build_candidate.sh build

.venv/bin/python tools/dt_layer_parity_oracle.py native-backends \
	--binary output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI \
	--ccv-source output/upgrades/ccv-69b1ed1f-qwen35-cuda \
	--require-qwen-cuda --execute-qwen \
	--report output/upgrades/native-backends-qwen35-cuda.json
```

Create the detached community checkout and independent CCV copy before this command; apply the profile only to that private CCV copy. The report checks six CUDA precision selections, CPU/SDPA controls, the binary hash, and optionally real linked-runtime GPU operations against CPU values. The pristine server is a tested negative control: the required-Qwen-CUDA gate rejects it even though CPU and SDPA controls are present. Integration is accepted only after both the full build and the positive linked-runtime check pass.

### Verified Integration

The separate candidate built with 16 jobs and passed the linked-runtime gate. Its SHA-256 is `5cfdd16e4110e5248fcb4f37fca7da2dc26790fdddcca1f9682f22e24ba1a39d`; the local acceptance report is `output/upgrades/native-backends-qwen35-cuda.json`.

All six Qwen F32/F16/BF16 selections resolve to GPU_REF, with SDPA and both CPU controls also matching. Actual native allocation, transfer, command dispatch, and GPU-to-CPU transfer passed for tiny F32 gated-delta and affine RMSNorm cases; all outputs and recurrent-state checkpoints were finite and within absolute `2e-6` of the pinned CPU implementations. The pristine candidate also passes a negative test that explicitly refuses GPU execution when registrations are absent. The combined CUDA-enabled suite passed all 55 tests.

The first full build correctly failed the registration gate because bindings were stale. Its log, hash, and negative registry report are retained with `before-bindings` names under `output/upgrades`. Automatic upstream regeneration fixed the integration; no generated file was hand-edited. This candidate is not installed or promoted. The original candidate, installed runtime, legacy CPU tools, and active catalogs remain unchanged.

## Coverage And Limits

- RMSNorm: one last-axis reduction on contiguous tensors, optional affine scale, mixed F32/F16/BF16 value/gate/scale types, FP32 accumulation, in-place output, zero-rank empty operations.
- Gated delta: grouped key/value heads, FP32 recurrent state and decay, F32/F16/BF16 Q/K/V/beta, scalar-per-head or per-key-channel decay, explicit log-decay, nonzero initial state, checkpoints, and in-place state when checkpoint count is zero. Key dimension is limited to 1,024.
- Unsupported layouts, types, or contracts return `CCV_NNC_EXEC_INVALID`; no backward/training implementation is added.
- Tests include real Qwen3.5 dimensions (16 key heads, 32 value heads, 128-dimensional heads), batched and non-square cases. These are correctness-first kernels, not tuned performance results.

On the RTX A3000/SM86, 60 RMSNorm and 108 gated-delta numerical cases passed, plus invalid/empty contract tests. Observed maximum errors were approximately `9.54e-7` for RMSNorm, `4.47e-8` for FP32 gated-delta state, and `6.10e-5` for gated-delta output including reduced-precision rounding. FP32 checks use absolute tolerance `2e-6` and relative tolerance `2e-5`; reduced-precision output checks use separately declared tolerances in the tests.

These results establish standalone operator correctness and linked-server operator integration, not full MOSS inference. The distinct zero-length scaled-dot-product-attention crash, complete text-encoder/denoiser parity, and an accepted final image remain separate gates.