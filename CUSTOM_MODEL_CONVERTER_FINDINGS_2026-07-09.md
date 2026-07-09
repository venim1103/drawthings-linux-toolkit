# Custom Model Converter — Findings Log (2026-07-09)

## 0) Purpose

This document records everything established in the 2026-07-08/09 investigation
session, in which we finally isolated the true root cause of the custom LTX2.3
"convert → quantize → inference" failure.

It supersedes the working hypothesis of Runs 001–117 (which treated the custom
model as the suspect) and should be read together with:

- `CUSTOM_MODEL_ROOT_CAUSE_ANALYSIS_AND_FORWARD_PLAN_2026-07-07.md` (strategy)
- `Q6P_RUN_LOG.md` (historical run ledger)

---

## 1) Headline Finding (Proven)

**The converter is the root cause, not the custom finetune.**

We converted Draw Things' *own official* `ltx_2.3_22b_distilled_1.1.safetensors`
with our tooling — **zero custom weights involved** — and the resulting f16
checkpoint crashes at model-load with the *exact* signature that has plagued the
custom model since Run 001:

```
Thread 2 crashed:
0  ccv_nnc_tensor_read + 5309
1  ccv_cnnp_model_read + 638
*** Program crashed: Bad pointer dereference at 0x0000000000000008 ***
rax = 0x0
```

Because the input was Draw Things' own model, this removes the custom finetune as
a variable entirely. The failure is in the **conversion pipeline**.

This is the single most important test in the whole project, and it was never run
in Runs 001–117.

---

## 2) The Positive Control That Cracked It

### 2.1 Why it matters

For 117 runs there was no positive control: every comparison was
*our-converted-custom-ckpt* vs *their-already-converted-official-ckpt*. Nobody ever
asked the decisive question:

> Can our converter turn **any** LTX2.3 safetensors — including Draw Things' own —
> into a working checkpoint?

### 2.2 What we did

- User downloaded `dt-models/ltx-2.3-22b-distilled-1.1.safetensors` (43G) — the
  official base model's own source.
- Converted it with the new `tools/run_convert_inference.sh` in **RAW mode**
  (no align-to-official step, so we test the converter's true output).
- Preflight passed cleanly: `tensor_count=5947`, `BF16: 5657, F32: 290`, all
  structural checks PASS, no missing required keys.
- Conversion completed successfully (~11 min): `version=ltx2_3 modifier=kontext`,
  output `ltx23_control_f16.ckpt` (42G), and the converter's own validator PASSed
  (`tensor_count=5746`, all LTX2.3 required prefixes present).

### 2.3 Result

- Structure looked correct, but **inference crashed at model-load** with the
  `ccv_nnc_tensor_read` signature above.
- Meanwhile the official pre-converted `ltx_2.3_22b_distilled_f16.ckpt`
  **loads and runs** on the identical setup (`RESULT=PASS`, response #1 produced).

Conclusion: a correct-serialization f16 LTX2.3 works here; our converter's f16 does
not. The converter is at fault.

---

## 3) The Exact Mechanism (Serialization Policy Mismatch)

Row-wise comparison (`dt_probe_ckpt_meta_len_rowwise.py`) of our converted f16 vs
the official f16:

| Field | Ours | Official | Count affected | Meaning |
|---|---|---|---|---|
| keyset | 5746 | 5746 | identical | **Not a naming/mapping gap** |
| `type` | 1 | 258 | 5745 (all readable) | CPU-memory vs GPU-tagged |
| `format` | 1 | 2 | 5743 | NCHW vs NHWC |
| `datatype` | F16 (131072) | F32 (16384) | ~1545 | norm/ada_ln/modulation kept F32 by official |
| `data_len` | half-size | full | 1542 | consequence of F16 downcast on norms |

Key deductions:

- **Keysets are identical (5746/5746).** The crash is *not* a missing-tensor
  problem — this rules out the mapping-name hypothesis.
- The converter applies the wrong per-tensor **datatype policy** and wrong dim
  **ranks**. The single crash trigger is the datatype of 4 specific tensors
  (Section 3a); the rank/norm-precision issues (Section 3b) are additional
  correctness defects that must also be fixed.

### 3a) The Exact Crash Tensor (bfloat16 vs float16)

Using the high-limit SQLite build, the official datatype distribution across all
5746 tensors is:

| datatype code | meaning | official count | our converter |
|---|---|---|---|
| 16384 (0x4000) | float32 | 1542 (norms / ada_ln) | wrote as F16 |
| 131072 (0x20000) | float16 | 4200 | F16 (ok) |
| 524288 (0x80000) | **bfloat16** | **4** (aggregate_embed) | **wrote as F16 ← crash** |

The 4 bfloat16 tensors are:

- `__text_feature_extractor__[t-video_aggregate_embed-0-0]` dim `[4096, 188160]` (1.54GB)
- `__text_feature_extractor__[t-video_aggregate_embed-0-1]` dim `[4096]`
- `__text_feature_extractor__[t-audio_aggregate_embed-0-0]` dim `[2048, 188160]`
- `__text_feature_extractor__[t-audio_aggregate_embed-0-1]` dim `[2048]`

The crash is **provably** on `video_aggregate_embed-0-0`: the faulting register
`r15 = 770703360 = 4096 × 188160` is that tensor's exact element count. The runtime
model declares this parameter bfloat16; our converter stored it float16, so
`ccv_nnc_tensor_read` dereferences a null and SIGSEGVs (`bad deref 0x8`, `rax=0`).

Proof by elimination: a post-process that fixed the norm precision (F16→F32) and
the dim ranks but left these 4 as F16 produced a **byte-identical crash** (same
registers) — confirming the datatype of these 4 tensors is the trigger.

### 3b) Additional Correctness Defects (not the crash, but wrong)

Exact `dim`/datatype diff vs official (2756 dim mismatches, all element-count
preserving — pure reshapes):

- `1D → 3D` ×592: norms `[N]` → `[1,1,N]`
- `2D → 1D` ×592: biases `[N,1]` → `[N]`
- `2D → 3D` ×1540: ada_ln `[1,N]` → `[1,1,N]` (these are also the F16→F32 set)

The official has **no 4D+ tensors** (ranks are only 1/2/3D, every 3D is `[1,1,N]`),
so `format` NCHW→NHWC needs **no transpose** — it is contiguous-equivalent metadata.

---

## 4) A Nuance On The Oversized Row (Corrects Runs 114–117)

The oversized (>1GB) row `__text_feature_extractor__[t-video_aggregate_embed-0-0]`
drove Runs 114–117 and repeated high-limit SQLite surgery.

**What was actually true:** the official f16 contains the *same* oversized row
(same dim, same length), so the row's *existence and size* are **expected** — the
tensor itself is not corrupt and not a structural bug. Runs 114–117 were right that
it is special but wrong about *why*: they tried to patch its bytes/rows. The real
issue is only its **datatype** — official stores it BF16, our converter stored it
F16. It is the crash site precisely because of that datatype downcast, not because
of its size or contents (see Sections 3a/3b).

---

## 5) Code-Level Root Cause

Location: `draw-things-community/Libraries/ModelOp/Sources/ModelImporter.swift`
(patched LTX2.3 import path, roughly lines 1975–2245), which serializes via
`Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift`.

Findings:

- The LTX2.3 path performs **raw tensor-name mapping**: it builds
  `Tensor<FloatType>` values (with `FloatType = Float16` in this build) in
  `.NCHW` format and calls `value.write(...) → internalWrite(...) →
  store.write(name, tensor:)`.
- `internalWrite` has only an `isBF16` branch. There is **no F32 path** and no
  per-layer datatype policy. So every tensor is written F16 / CPU / NCHW.
- The official checkpoint's F32 norms + GPU / NHWC layout are characteristic of a
  **model-based write** (`store.write(key, model:)`), where each parameter is
  serialized with the model's real per-layer dtype and inference layout. The
  patched LTX2.3 importer **bypasses** that by writing raw tensors.

So the patched importer is structurally incomplete: it reproduces the right *keys*
and *values*, but not the right *serialization policy* the runtime loader expects.

### 5a) The Fix Applied (converter, via DRAW_THINGS_PATCH)

`ModelWeightElement` already carries an `isBF16: Bool` flag; when set,
`TensorDescriptor.write` stores the tensor as bfloat16. The 4 aggregate_embed
mapping entries were bare array literals (`isBF16 = false` → downcast to F16). The
fix sets `isBF16: true` on all 4, in
`Libraries/ModelOp/Sources/ModelImporter.swift` (the `ltx23TextFeatureAliases`
block):

```swift
ltx23TextFeatureAliases["...video_aggregate_embed.weight"] =
  ModelWeightElement(["t-video_aggregate_embed-0-0"], isBF16: true)
// ...bias, audio weight, audio bias likewise isBF16: true
```

This preserves bfloat16 straight from the source safetensors (full precision, no
1.5GB blob surgery, no lossy F16→BF16 relabel). The change was captured into the
patch bundle with `tools/generate_drawthings_quant_patches.sh` — both the unified
`DRAW_THINGS_PATCH/patches/draw-things-community.patch` and the snapshot copy under
`DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift`. The working
`draw-things-community/` tree is never committed to.

The norm-precision (F16→F32) and dim-rank defects from Section 3b are handled by a
fast post-process tool (Section 8) applied after conversion; porting them into the
converter is future work.

---

## 6) Why The Previous Approach Could Never Work

- Runs 001–117 chased **row-level parity** by copying metadata/data **out of the
  official ckpt into the custom ckpt** (`dt_align_ckpt_*`, `dt_patch_ckpt_*`,
  high-limit row repair). That can only ever approximate Draw Things' *base*
  weights — never a working *custom* finetune — and it masked whether the converter
  was correct.
- `dt_align_ckpt_metadata.py` only rewrites the metadata columns and **never
  touches the blob**. Relabeling an F16 norm blob as F32 guarantees a loader
  over-read. This explains why the align / parity-copy era produced either crashes
  or garbled-but-loadable output.
- In-place SQLite surgery on the 42G file is unreliable: the expected >1GB row
  breaks full-table statements; per-row edits work but are slow and fragile. This
  path is closed.

---

## 7) Environment / Operational Facts

- **Battery is dangerous for inference.** Running inference on battery triggered a
  Windows BSOD `VIDEO_MEMORY_MANAGEMENT_INTERNAL (0x10E)` (dGPU reset under load,
  42G model streamed through a 12GB RTX A3000). Keep the charger plugged in for any
  model-load / inference step.
- **Conversion is CPU-only** (math threads, no GPU) and is safe to run on battery.
- GPU: NVIDIA RTX A3000 12GB Laptop GPU; model is offloaded/streamed, not
  fully resident.
- Disk: real constraint is the Windows host (<30GB free); the container's large
  "free" figure is inside an already-grown WSL VHDX. Deleting inside the container
  does not shrink the Windows disk without a VHDX compact.

---

## 8) Tooling Produced This Session

- `tools/run_convert_inference.sh` — one control-first orchestrator chaining the
  existing primitives: preflight → convert (RAW by default; `--aligned` opt-in) →
  validate-vs-source → optional `--quantize` → upsert a correct `ltx2.3`
  `custom.json` alias → first-frame canary → render PNG. Resumable: skips convert
  if the f16 exists unless `--force-convert`. Acceptance is a **coherent frame**,
  not merely `rc=0`.
- `tools/dt_fix_ltx23_serialization.py` — post-process that fixes the Section 3b
  defects using the official f16 as an *architecture template* (copies zero
  weights): reshapes dim ranks and widens the 1542 norm tensors F16→F32. Runs in
  <1s (touches only small tensors). Element-count parity is verified per row.
- `output/official_f16_keyset.txt` — the 5746 official tensor names, for exact
  keyset/dim diffing.

---

## 9) What Is Proven vs. Still Open

Proven:
- Converter (not the custom model) is the root cause.
- Correct-serialization f16 LTX2.3 loads and runs here.
- Keysets are identical; the crash trigger is the **bfloat16→float16 downcast of the
  4 aggregate_embed tensors** (crash register = `video_aggregate_embed-0-0` element
  count).
- Additional (non-crash) defects: 2756 wrong dim ranks + 1542 norms wrongly F16.
- The oversized row's size/contents are expected; only its datatype was wrong.

Fix status:
- Converter patched (`isBF16: true` on the 4 aggregate_embed entries) and captured
  into `DRAW_THINGS_PATCH`. `model-converter` rebuilt successfully.
- **VERIFIED**: reconverted control now stores the 4 embeds as bfloat16 (524288);
  post-process fixed norms/dims. Inference **no longer crashes at load** — it now
  reaches `textEncoded` and `imageEncoded` (previously an instant SIGSEGV). The
  `isBF16` fix is confirmed end-to-end for model loading.

Still open (now a *different*, downstream problem — see §12/§13):
- The 42G f16 cannot stream through the 12GB GPU → a cuDNN weight-eviction crash.
  Resolution is to **quantize to q6p** (the working official format), not another
  checkpoint edit.
- Port the norm-F32 + dim-rank fixes into the converter so no post-process is
  needed, then apply the whole proven pipeline to `10_e_v1_bf16.safetensors`.

---

## 10) Forward Plan (Unchanged Objective: A Coherent Frame)

1. Reconvert the control with the fixed converter (in progress) → the 4 aggregate
   tensors are now bfloat16 from source.
2. Apply `tools/dt_fix_ltx23_serialization.py` (norm F16→F32 + dim ranks).
3. Run the first-frame canary (charger plugged in). If it loads and renders a
   coherent frame → the pipeline is proven end-to-end.
4. If a crash persists, it can only be `type`/`format`; address then.
5. Apply the identical, proven pipeline to `10_e_v1_bf16.safetensors`.

Acceptance rule remains: runtime PASS is insufficient; the frame must be coherent
under matched settings.

---

## 11) One-Line Summary

The converter downcast four `aggregate_embed` tensors from bfloat16 to float16
(plus wrong dim ranks and F16 norms); the runtime expects bfloat16 there, so it
crashed in `ccv_nnc_tensor_read` — fixed with `isBF16: true` in the converter
(persisted to `DRAW_THINGS_PATCH`), and the custom finetune was never the problem.

---

## 12) BF16 Fix Verification (2026-07-09, later)

After rebuilding the converter and reconverting the control:

- The 4 aggregate_embed tensors are now **bfloat16 (524288)**; full datatype
  distribution `131072=5742, 524288=4` — matching the official policy.
- Inference progressed from an **instant load SIGSEGV** to producing
  `response #1: textEncoded` and `response #2: imageEncoded`, with **zero crash
  lines** in the loader. The model now loads and runs.

Conclusion: the `isBF16` converter fix is **confirmed working**. The historical
`ccv_nnc_tensor_read -> ccv_cnnp_model_read` load crash is eliminated.

---

## 13) Downstream Discovery: f16 Too Large For The 12GB GPU → Quantize

After the load crash was fixed, a **different** crash appeared, later in the
pipeline (during DiT weight streaming):

```
ccv_nnc_tensor_read + 4622
_ccv_nnc_device_local_drain + 71
libcudnn_graph.so.9.8.0        (cuDNN)
```

This is a GPU **weight-streaming / eviction** crash, not a checkpoint bug:

- Our control is **f16 = 42G** (`-tensordata`); the working official model is
  **q6p = 20G**. The GPU is a 12GB RTX A3000.
- A 44G-class f16 cannot stream through 12GB, so ccv's device-local drain path
  crashes inside cuDNN. `ccv_nnc_tensor.c` normalizes stored `type` to CPU on read
  (lines 96–98), so `type`/`format` metadata is **not** the cause.
- Control experiment: `--cpu-offload --no-flash-attention` **did not crash**
  (`post_echo_rc=0`) but **timed out at 1200s with no output** — pure-CPU execution
  is too slow (this is also why single generations take 15+ min and GPU shows
  ~0 MiB).

The resolution is the missing third step of the original goal: **quantize the
fixed f16 to q6p**, matching the streamable official model. The official q6p keeps
the identical datatype policy (4 BF16 embeds, 1542 F32 norms, 4200 F16), so the
quantized output must preserve BF16 on the 4 embeds too (verify after quantizing;
the quantizer may re-downcast — if so, apply the same fix pattern).

### 13.1 Updated Pipeline (control-proven target)

1. Convert safetensors → f16 with the **fixed** converter (BF16 embeds correct).
2. `tools/dt_fix_ltx23_serialization.py` (norm F16→F32 + dim ranks).
3. **Quantize f16 → q6p** (`tools/dt_quantize_model.sh -m ltx2.3 --target-codec q6p`).
4. Verify q6p datatypes vs official q6p (BF16 embeds, F32 norms).
5. Inference on the **GPU** (no `--cpu-offload`) — q6p (~20G) streams within 12GB.
6. Accept only a **coherent frame**; then repeat for `10_e_v1_bf16.safetensors`.

### 13.2 GPU Utilization Note (open follow-up)

The user correctly flagged that generations are not using the GPU. That is a
*consequence* of the two states above: the GPU path (fast) crashed on the 42G f16,
so we fell back to CPU offload (0 GPU). Once the model is q6p and streams on the
GPU, utilization returns. If a q6p GPU run still hits memory pressure, tune
`--weights-cache <gib>` rather than reverting to `--cpu-offload`. Verify real
usage with `nvidia-smi dmon` during a run (WSL2 single-snapshot memory reporting
is unreliable).
