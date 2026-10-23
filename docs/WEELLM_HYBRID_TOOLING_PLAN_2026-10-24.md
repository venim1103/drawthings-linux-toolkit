# WeeLLM Analysis and Hybrid Tooling Plan for `drawthings-linux-toolkit`

Assessment date: 2026-10-24. Reference WeeLLM commit inspected: `88233af976568bfb134f082b62434c1b3f849b28` (`jit-roy/weellm`).

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

## 6. Phased Implementation Roadmap & Acceptance Gates

To preserve the stability of our existing LTX 2.3 pipeline and follow the strict upgrade discipline established in `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md`, implementation is divided into five gated milestones (`W0`–`W4`).

### Milestone W0: Isolated Reference Environment & WeeLLM Seeker Extraction
* **Scope:**
  1. Create an optional, isolated Python virtual environment (`.venv-weellm`) or reuse `.venv` for pure-NumPy/SQLite tools without altering pinned runtime dependencies.
  2. Vendor/adapt the clean, license-compatible seeker and dequantizer primitives from `jit-roy/weellm` (`88233af9`) into `tools/dt_weellm/{seek_io.py, dequant.py, transforms.py, sqlite_store.py}`.
* **Acceptance Gate:**
  * Unit test suite (`tools/test_dt_weellm_core.py`) passes 100% on CPU:
    * `rotary_deinterleave` and `rotary_interleave` round-trip and match the §17 formula on `[4096, 4096]` (`32` heads $\times$ `128` dim) and `[3072, 3072]` (`24` heads $\times$ `128` dim).
    * `encode_dim_blob` produces byte-identical 48-byte blobs to `dt_q6p_restore_qk_norms.py`.
    * `seek_io.py` parses headers and slices tensors from existing `.safetensors` files in `< 50 ms` without full-file reads.

### Milestone W1: Contract Compiler (`dt_keymap_compiler.py`) & LTX 2.3 Baseline Parity
* **Scope:**
  1. Build `tools/dt_keymap_compiler.py` and generate `contracts/ltx2_3.json` and `contracts/moss_image.json`.
  2. Add `--contract` support to `tools/dt_preflight_safetensors.py` and `tools/dt_validate_converted_ckpt.py` (with `--fail-closed` mode).
* **Acceptance Gate:**
  * `contracts/ltx2_3.json` accurately classifies all 5,746 LTX 2.3 checkpoint tensors (4,200 `F16`, 1,540 `ada_ln` `F32`, 608 `q_norm`/`k_norm` `F32` with `[1,1,N]` rank-3 dims, and 4 `BF16` aggregate embeds).
  * `dt_keymap_compiler.py audit-swift` run against an unpatched `LTX2.swift` fixture automatically flags all 17 missing `q_norm`/`k_norm` `interleaved: true` sites from §18, and passes cleanly against our patched `DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Models/LTX2.swift`.

### Milestone W2: MOSS M1 Backend & NaN Diagnosis (`dt_layer_parity_oracle.py`)
* **Scope:**
  1. Implement `tools/dt_layer_parity_oracle.py` wrapping WeeLLM's `qwen3_5_model.py` and `moss_image_transformer_2d_model.py`.
  2. Generate reference fixtures for `Qwen3_5LinearAttention` (`GatedDelta` `0x6de198ac` and `RMSNormGated` `0x9b8af346`) and step-1 MOSS conditional/unconditional denoiser blocks.
* **Acceptance Gate:**
  * Peak GPU VRAM during oracle execution stays $\le 6.0\text{ GiB}$ on the RTX A3000 12 GiB Laptop GPU.
  * Produces finite golden reference tensors for the M1 teapot prompt (`seed=4242`, `512x512`, `CFG=3.0`, `shift=5.0`).
  * Pinpoints the exact origin of the M1 `131,072` non-finite preview values and validates the CCV backend fix for `GatedDelta` / `RMSNormGated` and zero-length attention.

### Milestone W3: Direct Python Checkpoint Builder (`dt_direct_ckpt_builder.py`) for MOSS M3 & GGUF
* **Scope:**
  1. Implement `tools/dt_direct_ckpt_builder.py` (`convert`, `restore-norms`, `diff-ckpt`).
  2. Add `--profile` / `--contract` support to `tools/dt_convert.sh` and `tools/dt_quantize.sh` so MOSS and future models use their own contract companions and norm-restore rules rather than LTX 2.3 defaults.
* **Acceptance Gate:**
  * Converting the official LTX 2.3 or MOSS-Image Base source `.safetensors` via `dt_direct_ckpt_builder.py` produces a `.ckpt` whose tensor names, `datatype` codes, `dim` blobs, and transformed payload values match the official Draw Things reference checkpoint (`max_abs_diff == 0.0` for FP16/BF16/FP32 direct casts; `< 1e-3` where official is palettized `q8p`).
  * `restore-norms` reproduces the exact output of `dt_q6p_restore_f32_norms.py` + `dt_q6p_restore_qk_norms.py` on LTX 2.3, and preserves FP32 timestep/modulation norms on MOSS-Image.

### Milestone W4: Universal LoRA Bridge (`dt_lora_universal_bridge.py`) for MOSS M4
* **Scope:**
  1. Implement `tools/dt_lora_universal_bridge.py` with WeeLLM LoRA keymaps and the automated §24 static permutation/rank verifier.
  2. Integrate as a pre-conversion normalizer and post-conversion gate in `tools/dt_lora_convert.sh`.
* **Acceptance Gate:**
  * Passes the §24 static audit (`3,368 / 3,368` tensors verified) on `ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe_ourconvert_lora_f16.ckpt`.
  * Normalizes and statically verifies a MOSS-Image LoRA checkpoint against `contracts/moss_image.json`.

---

## 7. Summary of Immediate Next Actions

1. **While resolving MOSS M1 (`docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md`):**
   Build **Milestone W0 + W2 (`dt_layer_parity_oracle.py`)** first if you want golden input/output tensors from WeeLLM's `Qwen3_5Model` (`qwen3_5_model.py`) and `MossImageTransformer2DModel` (`moss_image_transformer_2d_model.py`) to test CCV's `GatedDelta` (`0x6de198ac`), `RMSNormGated` (`0x9b8af346`), and empty-attention fixes.
2. **When moving to MOSS M3 (Custom MOSS Conversion) & M4 (LoRA):**
   Build **Milestone W1 + W3 (`dt_keymap_compiler.py` + `dt_direct_ckpt_builder.py`)** and **Milestone W4 (`dt_lora_universal_bridge.py`)** to bypass `ModelImporter.swift`'s `fatalError()` branches and convert `.safetensors` / `.gguf` checkpoints directly into validated Draw Things `.ckpt` stores.
