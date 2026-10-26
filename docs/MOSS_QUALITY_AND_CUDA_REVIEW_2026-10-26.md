# MOSS-Image CUDA Enablement & Quality Boundary Review (`cf8baab..acd9b26`)

**Review Date:** 2026-10-26  
**Baseline Reviewed:** `venim1103/drawthings-linux-toolkit` (`main` across `cf8baab..acd9b26`)  
**Active Isolated CUDA Candidate:** `920bf727c5a27a7d1ad90735735c74b7edfbbc1459598cea018f71e2d713dd96`

---

## 1. Executive Summary

Across the seven commits from `cf8baab` to `acd9b26`, the project completed two major engineering milestones:

1. **Full Hardening of the Hybrid Tooling Suite (`560bc13`, `19d9bbd`):**
   Every defect identified in `docs/WEELLM_HYBRID_REVIEW_2026-10-26.md` (`F01`–`F17`) was resolved and locked down with 60 unit, native-CPU, and CUDA-enabled tests in `tools/test_dt_weellm.py`:
   * `contracts/moss_image.json` and `build_moss_image_contract()` were rebuilt from the pinned native `MOSSImage.swift` mapper, matching all **524** official target keys, shapes, and datatypes (`15df82ca64fb5a9469be36868e502f00fd0b0957aad242bb7af43aec9085931d`).
   * `DrawThingsCkptStore` (`tools/dt_weellm/sqlite_store.py`) now extracts codec identifiers from the upper 32 bits of `type`, validates external descriptors fail-closed, and implements native `tensor_splits` chunking (`format | (1 << 32)`), verified through CCV's `ccv_nnc_tensor_read` under LLDB.
   * `dt_direct_ckpt_builder.py`, `dt_layer_parity_oracle.py`, and `dt_lora_universal_bridge.py` work as standalone scripts outside the repository, validate JSON Schemas, check post-cast F16/BF16 finiteness, and stage all writes atomically.

2. **Native CUDA Enablement & Finite 50-Step MOSS Generation (`193603b`..`acd9b26`):**
   An isolated CUDA patch profile (`DRAW_THINGS_PATCH/cuda/`) was created, verified, and compiled into a private candidate server (`920bf727...`) without modifying the legacy CPU converter bundle or active catalogs:
   * **`GatedDelta` (`0x6de198ac`) & `RMSNormGated` (`0x9b8af346`) CUDA Backends (`193603b`, `acef1cc`):** Implemented forward CUDA reference kernels supporting grouped heads (`16` key heads, `32` value heads, `128` head dim), `F32`/`F16`/`BF16` operands, `F32` state/decay, `log_decay`, and state checkpoints, plus automated Ruby command-table regeneration (`ruby build-cmd.rb .`).
   * **Rank-0 Empty SDPA Guard (`5aed2ba`):** Added a narrow pre-arithmetic guard in `ccv_nnc_scaled_dot_product_attention_flash_attn.cu` that returns `CCV_NNC_EXEC_SUCCESS` when all Q/K/V/O tensors have rank 0 (eliminating the `SIGFPE` on MOSS block 22's skipped unconditional branch).
   * **Packed `CCV_QX` Embedding Lookup (`4bb0ba0`):** Identified that `qwen_3.5_9b_i5x.ckpt` stores its token embedding table in `Q5_K` (`CCV_QX`), which `ccv_nnc_index_select_gpu_ref.cu` previously read as raw float bytes (producing `1,848` NaNs in the `[11, 4096]` token embeddings before layer 0). Decoding via `ccv_nnc_compat_decode_qx` before lookup eliminated all `131,072` non-finite preview values.
   * **Current M1 State:** Both the 2-step smoke canary and the full 50-step CFG-3 (`512x512`, `seed=4242`, `shift=5.0`) MOSS canary complete cleanly with **0 NaNs/Infs** and a **2,305 MiB** whole-device peak. However, the 50-step image depicts a **repeated green structure** rather than the requested *"orange ceramic teapot"*.

---

## 2. Commit-by-Commit Audit (`cf8baab..acd9b26`)

| Commit | Summary | Key Technical Verification |
| :--- | :--- | :--- |
| `560bc13` | `docs: review WeeLLM hybrid tooling and record validation gaps` | Established the 17-item defect baseline (`F01`–`F17`) with executable reproducers. |
| `19d9bbd` | `feat: harden hybrid tooling and add native validation gates` | Repaired `F01`–`F17`: exact 524-key MOSS contract, `tensor_splits` support, `dt_qwen35_cpu_probe.c` native CPU verification, FP8 row-scale broadcasting, Kohya unmangling, and atomic shell staging. |
| `193603b` | `feat: add verified Qwen3.5 CUDA forward patch profile` | Added `ccv_nnc_gated_delta_gpu_ref.cu` and `ccv_nnc_rmsnorm_gated_gpu_ref.cu` under `DRAW_THINGS_PATCH/cuda/` + `tools/drawthings_cuda_patch.py`. Passed 60 RMSNorm + 108 GatedDelta CUDA tests on SM86. |
| `acef1cc` | `feat: integrate and verify isolated Qwen CUDA runtime` | Automated `ruby build-cmd.rb .` binding generation during isolated candidate builds and added `native-backends --execute-qwen` LLDB verification. |
| `5aed2ba` | `fix: handle empty CUDA attention and add native regression gates` | Patched `ccv_nnc_scaled_dot_product_attention_flash_attn.cu` for rank-0 skipped layers; verified 8 empty/mixed SDPA contracts and nonempty grouped FP16 attention. |
| `4bb0ba0` | `fix: decode packed CUDA embeddings and validate MOSS output` | Patched `ccv_nnc_index_select_gpu_ref.cu` to decode `CCV_QX` (`Q5_K`/`Q6_K`) tables; achieved first 100% finite 2-step and 50-step MOSS images. |
| `acd9b26` | `feat: add captured activation parity and packed GEMM diagnostics` | Added `compare-capture` (`linear`, `rotary`, `sdpa`) and `--execute-packed-gemm`; verified MOSS `text_projection`, first `text_attention`, `text_rotary`, and `F32` `time_projection` captures against CPU math. |

---

## 3. Code Audit Findings: Bugs, Subtle Risks & Optimization Opportunities

### 3.1 `ccv_nnc_index_select_gpu_ref.cu`: 1.24 GiB Full-Table Workspace Decode vs. Row-Selective Decode
In `DRAW_THINGS_PATCH/cuda/ccv/lib/nnc/cmd/index/gpu/ccv_nnc_index_select_gpu_ref.cu` (lines 73–78):
```cuda
if (is_quantized)
{
    void* const workspace = ccv_nnc_stream_context_get_workspace(
        stream_context, ccv_nnc_compat_qx_dense_data_size(a->info), CCV_TENSOR_GPU_MEMORY);
    ccv_nnc_compat_decode_qx(a->data.u8, a->info, workspace, stream_context);
    a_data = (const unsigned char*)workspace;
}
```
* **Impact:** Correctness-wise, this is completely sound and passes `--execute-embedding`. However, for Qwen3.5's `[151936, 4096]` vocabulary table, `ccv_nnc_compat_qx_dense_data_size` allocates **$151,936 \times 4096 \times 2 = 1.24\text{ GiB}$** of GPU stream workspace and launches a full-table decode kernel just to extract $11\text{–}77$ prompt rows.
* **Future Improvement (Post-M1):** Because `CCV_NNC_QX_8I_ROWWISE_Q5_K` and `Q6_K` are row-wise super-block formats (`256` elements per super-block, `16` super-blocks per `4096`-dim row) with an aligned per-row scale footer at the end of the buffer, a follow-up optimization can decode only the `b_rows` selected rows directly into `b`, reducing workspace from $1.24\text{ GiB}$ to $0\text{ B}$.

### 3.2 `ccv_nnc_index_select_gpu_ref.cu`: `CCV_16BF` Type Aliasing in `CCV_32S` vs `CCV_32F` Branches
In `ccv_nnc_index_select_gpu_ref.cu` lines 85–86:
* When `indices->info.datatype == CCV_32S` (integer indices), `CCV_16F` and `CCV_16BF` share the `(const __half*)a_data` branch. Because the integer-index kernel only performs a bitwise 16-bit assignment (`b[dest * b_inc + j] = a[src * a_inc + j]`), `CCV_16BF` bits are copied without alteration.
* When `indices->info.datatype == CCV_32F` (interpolated float indices, lines 91–95), `CCV_16BF` properly uses `const __nv_bfloat16*` so arithmetic conversion `(float)a[...]` uses `__nv_bfloat16` rather than `__half`.

### 3.3 Diagnostic Tooling Enhancement: `inspect-capture` for Latent & Conditioning Tensors
While `compare-capture` (`tools/dt_layer_parity_oracle.py`) verifies an operation's input-to-output math (`linear`, `rotary`, `sdpa`), diagnosing the "repeated green structure" also requires inspecting standalone captured tensors (such as the final 36-layer Qwen3.5 hidden states, the step-50 packed `[1, 32, 32, 128]` denoiser output, or the unpacked `[1, 64, 64, 32]` VAE latent) for:
* Per-channel mean/std collapse or outlier spikes,
* Token-to-token cosine similarity (detecting if all tokens collapsed to a constant vector), and
* $2\times 2$ spatial block periodicity (detecting if `2x2` patch unpacking transposed spatial and channel axes, which directly produces a repeated $2\times 2$ grid artifact).

We have added the `inspect-capture` subcommand (`inspect_capture_tensor()`) to `tools/dt_layer_parity_oracle.py` to compute these exact diagnostics on any captured `.bin` tensor.

---

## 4. Root-Cause Analysis & Action Plan for the MOSS "Repeated Green Structure"

Let us summarize what has been **proven working** versus where the remaining semantic/structural bug can hide:

### What Is Proven Working (Ruled Out)
1. **Qwen3.5 Token Embedding Lookup (`Q5_K`):** Verified exact against CPU decode (`--execute-embedding`).
2. **Qwen3.5 `GatedDelta` & `RMSNormGated` Kernels:** Verified on SM86 across 168 test cases and verified finite on Layer 0 live calls.
3. **MOSS `text_embedder.proj` (`4096 -> 3072`), First Text Refiner Attention, & Text 4-Axis Rotary:** Verified via `compare-capture` against CPU float64 math (`acd9b26`).
4. **MOSS `time_embedding.2` (`F32` Second Timestep Projection):** Verified via `compare-capture` (`max_abs_err = 6.05e-8`) and confirmed distinct across timesteps.
5. **MOSS Negative Branch Layer-22 Skip:** Ruled out as the sole cause by the 50-step CFG-1 run (`output/q6p_canary_moss_base_q8p_512_cfg1_quality_diagnostic_20261026`), which does not execute the negative branch yet still produces an unrelated structured image.

### The Four Remaining Suspects (Ordered by Diagnostic Speed)

#### Suspect A: $2\times 2$ Patchify / Unpatchify Permutation or `flux_2_vae_f16.ckpt` Latent Scaling
* **Symptom Match:** Very High. A "repeated green structure" across a 512x512 image is the classic signature of a $2\times 2$ spatial-to-channel permutation mismatch (`[H, W, 2, 2, C]` vs. `[H, W, C, 2, 2]` or `[H, 2, W, 2, C]`) between MOSS's `final_linear` (`128` channels = $2 \times 2 \times 32$) and the `flux_2_vae_f16.ckpt` decoder, or a missing latent scale/shift factor.
* **Fast Verification:**
  1. Capture the tensor immediately before and after unpatchify (at the end of the 50th denoising step, just before `FirstStage` / VAE decode) and run:
     ```bash
     .venv/bin/python tools/dt_layer_parity_oracle.py inspect-capture \
       --file output/upgrades/moss-image-base-add235e9/vae_input_capture.bin
     ```
  2. Check `spatial_periodicity_2x2` in the report: if adjacent even/odd rows or columns have a large step jump compared to stride-2 neighbors, the $2\times 2$ unpack reshape order in `MOSSImage.swift` / `UNetProtocol.swift` does not match the checkpoint's `final_linear_video.weight` row ordering!

#### Suspect B: Full 36-Layer Qwen3.5 Output Drift (`Conv1d` / RoPE / Layer Index / Token Slicing)
* **Symptom Match:** High. While Layer 0 `GatedDelta` and `RMSNormGated` are verified, Qwen3.5-9B runs 36 layers mixing linear attention (`Conv1d` causal depthwise convolution + `GatedDelta`) and full attention (every 4th layer with partial RoPE). Furthermore, `text_projection_capture_input_0.bin` captures the final Qwen3.5 output entering MOSS.
* **Fast Verification:**
  1. Run `inspect-capture` on the already-captured `text_projection_capture_input_0.bin`:
     ```bash
     .venv/bin/python tools/dt_layer_parity_oracle.py inspect-capture \
       --file output/upgrades/moss-image-base-add235e9/text_projection_capture_input_0.bin
     ```
  2. Compare `text_projection_capture_input_0.bin` against the hidden states produced by running Qwen3.5-9B on the identical token IDs in CPU/WeeLLM reference mode. Specifically check:
     * Which hidden state layer index is extracted?
     * Are system-template prefix tokens stripped or kept?
     * Does CCV's `Conv1d` causal padding in `Qwen3_5LinearAttention` match PyTorch's `F.conv1d(..., padding=kernel_size-1)[..., :seq_len]`?

#### Suspect C: MOSS Shared Modulation Chunk Order or `ffn_up_gate` Half Order
* **Symptom Match:** High. In `MOSSImage.swift`:
  * `shared_block_modulation` is split into 6 chunks (`x_ada_ln_0..5`), and `video_head_modulation` is split into 2 chunks (`ada_ln_1, ada_ln_0`).
  * `layers.{i}.proj.ffn_up_gate.weight` (`[16384, 3072]`) is split into two `[8192, 3072]` matrices (`ffn_up_proj` and `ffn_gate_proj`).
  * If the official `moss_image_base_q8p.ckpt` was exported with `(gate, up)` order while the runtime graph computes `up * SiLU(gate)` (or if two of the 6 modulation chunks are swapped), every operation is 100% finite and locally valid, yet the DiT denoises toward a meaningless textural attractor.
* **Fast Verification:**
  * Capture the inputs and output of **MOSS Main Block 0** (`layers.0`) on step 0 and compare against WeeLLM's `moss_image_transformer_2d_model.py` forward pass on the exact same inputs.

#### Suspect D: Flow-Matching Timestep Scale (`condition_scale = 1000` vs `[0, 1]`) & Shift Formula
* **Symptom Match:** Medium-High. If `time_embedding.0` sinusoidal embedding receives $t \in [0, 1]$ when it expects $t \in [0, 1000]$ (or vice-versa), or if the flow update step sign/scale mismatches, the trajectory fails to converge to the data manifold.
* **Fast Verification:**
  * Inspect the captured `time_projection_capture_input_0.bin` values across steps 0, 25, and 49 to confirm the sinusoidal frequencies span the expected $[0, 1000]$ timestep range of OpenMOSS's reference scheduler.
