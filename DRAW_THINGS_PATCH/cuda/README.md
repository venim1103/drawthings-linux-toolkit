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

## Coverage And Limits

- RMSNorm: one last-axis reduction on contiguous tensors, optional affine scale, mixed F32/F16/BF16 value/gate/scale types, FP32 accumulation, in-place output, zero-rank empty operations.
- Gated delta: grouped key/value heads, FP32 recurrent state and decay, F32/F16/BF16 Q/K/V/beta, scalar-per-head or per-key-channel decay, explicit log-decay, nonzero initial state, checkpoints, and in-place state when checkpoint count is zero. Key dimension is limited to 1,024.
- Unsupported layouts, types, or contracts return `CCV_NNC_EXEC_INVALID`; no backward/training implementation is added.
- Tests include real Qwen3.5 dimensions (16 key heads, 32 value heads, 128-dimensional heads), batched and non-square cases. These are correctness-first kernels, not tuned performance results.

On the RTX A3000/SM86, 60 RMSNorm and 108 gated-delta numerical cases passed, plus invalid/empty contract tests. Observed maximum errors were approximately `9.54e-7` for RMSNorm, `4.47e-8` for FP32 gated-delta state, and `6.10e-5` for gated-delta output including reduced-precision rounding. FP32 checks use absolute tolerance `2e-6` and relative tolerance `2e-5`; reduced-precision output checks use separately declared tolerances in the tests.

These results do not establish full MOSS inference. Server integration, the distinct zero-length scaled-dot-product-attention crash, complete text-encoder/denoiser parity, and an accepted final image remain separate gates.