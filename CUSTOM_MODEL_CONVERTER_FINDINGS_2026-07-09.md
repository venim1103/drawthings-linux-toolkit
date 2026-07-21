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

---

## 14) Quantization Results + The Sampling Milestone (2026-07-10)

### 14.1 Quantized q6p — a second precision bug (in the quantizer)

Quantized the fixed f16 → q6p (20G, ~1h53m; the quantizer is slow, likely
single-threaded on palettization). Results vs official q6p:

- **BF16 embeds preserved** (4 × `524288`) — the converter fix survives quantization.
- **But the quantizer PALETTIZED the 1542 norm/ada_ln tensors to 6-bit**
  (`datatype 131072`, `datalen≈2954`) instead of keeping them **F32** like the
  official q6p (`16384`, `datalen 8192`). This is the quantizer's analog of the
  converter's BF16 bug: it does not apply the LTX2.3 keep-F32 precision policy to
  our model. 6-bit layernorm/ada_ln weights are numerically unstable.

Our f16 norms are correct F32 (identical to official f16), so the defect is purely
in the quantizer's per-tensor keep-F32 decision.

### 14.2 Workaround + the milestone

New tool `tools/dt_q6p_restore_f32_norms.py` restores the 1542 norms to F32 by
copying **our own f16's** F32 values into the q6p (uses the official q6p only as a
template for *which* names are F32 and their dim encoding — **copies zero official
weights**). After this, the datatype distribution exactly matches official
(`131072=4200, 16384=1542, 524288=4`).

Result of GPU inference on the norm-restored q6p (no `--cpu-offload`):

- **No stall, no crash.** GPU utilized (~12G).
- Pipeline advanced through `textEncoded → imageEncoded → sampling` and **wrote 5
  preview frames**.

This is the first time in the entire investigation (117+ runs) that a
custom-converted model **ran the full sampling loop and produced frames**. The
`isBF16` + norm-F32 fixes together clear every crash/stall.

### 14.3 The remaining gap: gray output (weights, not norms)

Despite sampling successfully, the output is **gray / low-variance**, and the final
decoded payload fails (`Image processed failed`; only previews arrive). Side-by-side
with the official q6p (same 1.1 model, same settings, GPU):

| | official q6p | our q6p |
|---|---|---|
| samples w/o stall | ✓ | ✓ (after norm fix) |
| preview | colorful/structured | uniform gray |
| final image | coherent (red car, matches prompt) | fails (gray latents) |

**Ruled out as the cause of gray output:**
- Norm datatype (now F32) **and** norm *values*: our F16-widened norms are within
  ~1% max / ~0.001% mean of the official true F32 — negligible for layernorm.
- Keyset, BF16 embeds, dim ranks, load crash, GPU streaming.

**Remaining suspect: DiT weight values.** Either the converter's hand-written
LTX2.3 mapping (`makeLTX23ConnectorMapping` / `ltx23EmbedderAliases`) produces some
semantically-wrong weights, or our quantizer's 6-bit palettization degrades them.
Distinguishing requires weight-level work (each quantize iteration is ~2h).

### 14.4 Status Summary

Journey: `instant crash (117 runs)` → `loads` → `streams on GPU` → `samples full
frames` → **output not yet coherent (gray)**. The pipeline is fundamentally working;
the last mile is weight-value fidelity.

Proposed next steps (pick per cost/appetite):
1. **Isolate converter vs quantizer**: quantize the *official* f16 with *our*
   quantizer → if gray, the quantizer is the culprit; if coherent, the converter
   mapping is. (~2h)
2. **Audit the converter mapping** (cheap, read-only): diff our LTX2.3 mapping vs
   the upstream LTX2 model's expected weight layout for wrong/missing entries.
3. **Also fix the quantizer** to keep norms F32 natively (analog of the converter
   `isBF16` fix), so no post-process is needed.

Acceptance rule unchanged: only a **coherent frame** counts. Then apply the whole
proven pipeline to `10_e_v1_bf16.safetensors`.

### 14.5 Tools Added This Phase

- `tools/dt_q6p_restore_f32_norms.py` — restore F32 norms into a q6p from our own
  f16 (q6p-safe; does not touch palettized weights). Needed because
  `dt_fix_ltx23_serialization.py` is f16-only (q6p palettized data breaks its
  length math).

---

## 15) Gray-Output Root-Cause Audit (2026-07-10, later) — It's The Quantizer

### 15.1 The converter mapping is SOUND (ruled out)

A read-only audit of the hand-written LTX2.3 import mapping (initially suspected as
the source of wrong weights) concluded the **structure is correct**:

- The mapping is `source_key → [DrawThings target name]` (e.g.
  `video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight →
  t-to_q-0-0`), resolved through `reverseMapping`. The `t-to_q-i-0` strings are
  **target** store-keys, not phantom source keys.
- The importer handles the `model.diffusion_model.` source prefix
  (`ModelImporter.swift` L317-327), and the keyset matches official 5746/5746.

So the gray output is **not** a broken converter mapping. (An automated agent
initially mis-reported "phantom keys"; that was a misread and is refuted here.)

### 15.2 Direct weight-data comparison: our q6p vs official q6p

Comparing data length of all shared tensors (via high-limit sqlite):
`datalen_mismatch = 2933` (~half). Per-block dump (block 0) shows a stark pattern:

- **Every weight matrix** (`-0-0`) is **16-24 bytes** in the official q6p but the
  **full palettized blob** in ours (e.g. `x_q-0-0`: official 24B vs ours 13MB;
  `x_out_proj-0-0`: official 24B vs ours 52MB).
- Biases/norms (`-0-1`) have real data in both.

### 15.3 Root difference: sidecar vs inline storage

| checkpoint | storage | `x_q-0-0` `data` |
|---|---|---|
| official f16 | inline | 33MB |
| **official q6p** | **`-tensordata` sidecar** | 24-byte reference (offset+len) |
| our f16 | sidecar (42G tensordata) | — |
| **our q6p** | **inline** (single 20G `.ckpt`) | full 13MB palettized blob |

The official q6p keeps weights in a `-tensordata` sidecar (the DB row is a 24-byte
pointer). **Our quantizer wrote every weight inline.** Cause in
`Apps/ModelQuantizer/Quantizer.swift`:

- L123 opens the **input** store *with*
  `externalStore: TensorData.externalStore(filePath: inputFile)`,
- L126 opens the **output** store as `graph.openStore(outputFile)` **without** any
  external store → all writes land inline.

Our f16 (converter output) *does* use a sidecar; the quantizer collapses it inline.

### 15.4 Quantizer LTX2.3 precision policy (for reference)

`Quantizer.swift` L150-215 (forced-codec path), per tensor:

- `text_feature_extractor` / `text_video_connector` / `text_audio_connector`
  → **preserve original** (write as-is)
- `embedder` / `pos_embed` / `-linear-` / `scale_shift_table` /
  `caption_projection` / `patchify_proj` / `proj_out` → **fp16**
- `squeezedDims>1` and (`ada_ln`|`adaln`|4D) → **[q8p, ezm7]**
- other 2D → **[q6p, ezm7]**
- 1D scalar → **ezm7**

Note the policy quantizes `ada_ln` to q8p, but the official q6p keeps norms F32 —
another mismatch (already worked around by `dt_q6p_restore_f32_norms.py`).

### 15.5 Conclusion + decisive next step

After the F32-norm restore, the model samples but stays gray, so the gray comes
from the **q6p-palettized attention/MLP weight values**, produced by either:
- our converter emitting subtly different weight *values* than Draw Things', or
- our quantizer's inline/palettization path.

The mapping structure is sound, so suspicion is on the **quantizer**. Two moves:

1. **Isolate (definitive, ~2h):** quantize the *official* f16 with *our* quantizer
   → gray ⇒ quantizer; coherent ⇒ our converter weight values.
2. **Fix the quantizer sidecar output** (`Quantizer.swift` L126: give the output
   store an `externalStore`), regenerate the patch, rebuild, re-quantize, re-test.

---

## 16) DECISIVE ISOLATION RESULT (2026-07-10) — The Converter Is The Bug

The isolation test (option 1 above) was run and is **conclusive**.

Procedure:
- Quantized the **official** f16 (`ltx_2.3_22b_distilled_f16.ckpt`, Draw Things'
  known-good weights) with **our** quantizer → `official_via_ourquant_q6p.ckpt`.
- Restored F32 norms (from the official f16's own norms), aliased `ltx23_isolation`.
- Ran GPU inference (384x384, 8 steps, no cpu-offload).

Result: **`RESULT=PASS`, 23 responses, 9 images + 1 audio, `imageDecoded`,
"Image processed successfully".** The rendered image is a **coherent sports car at
sunset** — same quality as the official q6p control.

### 16.1 What this proves

| stage | verdict |
|---|---|
| our **quantizer** | ✅ correct — turns known-good weights into a working, coherent q6p |
| inline vs sidecar storage | ✅ irrelevant — the isolation q6p is also inline and works |
| our **converter — structure** (names/keyset/dims/norms) | ✅ correct |
| our **converter — weight values** (transforms) | ❌ **THE BUG** |

Truth table:

| f16 source → our quantizer → q6p | output |
|---|---|
| **official** f16 (good weights) | ✅ coherent |
| **our** converter's f16 | ❌ gray |

Same quantizer, same pipeline — the only variable is the f16 weight **values**.
Therefore our converter mis-transforms the DiT/attention weights while writing them
(the mapping *names* are correct; the *values* are scrambled). This also means the
gray output on the custom `10_e_v1` has the **same** cause — fixing the converter
fixes both.

### 16.2 Prime suspects (converter weight transforms)

In `ModelImporter.swift` (LTX2.3 mapping) via
`SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift`:

- `interleaved: true` rotary handling on Q/K
  (`TensorDescriptor.swift` L66-82 reshapes `[numHeads, 2, headDim/2, -1]` then
  `.transposed(1,2)`) — a wrong rotary de-interleave scrambles attention.
- transpose / orientation (`[out,in]` vs `[in,out]`).
- `format` (`.O`/`.I`) and QKV split/`offsets`.
- `numberOfHeads` / `headDimension` mismatch vs the source `[4096,4096]` Q layout.

### 16.3 Side benefit

`official_via_ourquant_q6p.ckpt` is a **working** model (official weights, our
quantize) — proof the convert-agnostic part of the pipeline (quantize → q6p → GPU
inference) is fully functional on this hardware.

### 16.4 Plan to the fix

1. Pinpoint the exact wrong transform by comparing a Q/K weight **value-for-value**
   between our f16 and the official f16 (same tensor, same 1.1 model).
2. Fix the converter mapping/transform in `ModelImporter.swift`.
3. Regenerate `DRAW_THINGS_PATCH`, rebuild `model-converter`.
4. Reconvert (~11 min) → requantize (~2h) → restore norms → GPU test.
5. Accept only a coherent frame; then apply to `10_e_v1_bf16.safetensors`.

Recommended: option 2 (a concrete, identified divergence from the official path).

## 17) EXACT ROOT CAUSE + FIX (2026-07-10) — Interleaved Rotary Transpose Did A GPU Round-Trip On A CPU-Only Converter

Step 16.4.1 (value-for-value comparison) was run and **pinpointed the exact bug**.

### 17.1 The evidence (our f16 vs official f16, same 1.1 model, same tensor)

Compared identical tensors between our converter's `ltx23_control_f16.ckpt` and the
official `ltx_2.3_22b_distilled_f16.ckpt`. The pattern is unambiguous:

| tensor | interleaved? | ours | official | verdict |
|---|---|---|---|---|
| `__dit__[t-x_q-0-1]` (Q bias) | **yes** | min −1.95 / max **1.96**, many exact `0.0` | min −0.38 / max 0.44 | ❌ scrambled |
| `__text_video_connector__[t-to_q-0-1]` (Q bias) | **yes** | min −8192 / max **65024** (F16 overflow!) | min −0.074 / max 0.059 | ❌ garbage |
| `__dit__[t-x_norm_q-0-0]` (norm) | no | min −0.181 / max 1.133 / mean 0.17479 | **same** min/max/mean 0.17482 | ✅ same values (order differs) |
| `__text_video_connector__[t-norm_q-0-0]` (norm) | no | min 0.738 / max 1.000 / mean 0.92283 | **same** | ✅ same values |

Two decisive facts:
- **Only the `interleaved: true` Q/K tensors are wrong.** Their values are not a
  permutation of the correct ones — they are genuine garbage (zeros interleaved with
  overflowed `65024` = the largest normal F16). That is the fingerprint of reading
  **uninitialized / wrong memory**, not a transpose-orientation mistake.
- **The non-interleaved norms have identical value sets** (same min/max/mean, just a
  different order) → same weights, same model version. This rules out "wrong source
  file" or "1.0 vs 1.1" and isolates the fault to the interleaved code path.

Both the **DiT** (`__dit__`) and the **connectors** (`__text_video_connector__`)
are affected → a **single shared transform** is the culprit, not per-mapper logic.

### 17.2 The culprit code

`Libraries/SwiftDiffusion/Sources/Models/LTX2.swift` marks every attention Q/K with
`interleaved: true` (DiT `attn1`, self-attention, cross-attention, connectors). That
flag is consumed in **one** place — the serializer:

`Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift`,
`internalWrite(...)`, `case .O:` (the **only** `interleaved` / `transposed` block in
the file). Original code:

```swift
tensor = graph.withNoGrad {
  Tensor<FloatType>(
    from: graph.variable(
      tensor.reshaped(
        format: tensor.format, shape: [numberOfHeads, 2, headDimension / 2, -1]
      ).toGPU()                       // <-- move to GPU
    ).transposed(1, 2).reshaped(      // <-- transpose on GPU
      format: tensor.format, shape: [numberOfHeads * headDimension, -1]
    ).toCPU().rawValue)               // <-- move back to CPU
}
```

The rotary de-interleave transpose is performed via a **`.toGPU()` → `.transposed()`
→ `.toCPU()` round-trip**. This is correct on Draw Things' Mac/Metal build (where the
converter has a working GPU). But **our Linux `model-converter` is CPU-only by
design** (`dt_convert_model.sh` forces CPU math threads; `Converter.swift` sets up no
GPU `DynamicGraph`; the only `.transposed()` in the whole file sits right after this
`.toGPU()`). On that CPU-only path the GPU round-trip silently returns garbage
instead of the transposed tensor — producing the scrambled/overflowed Q/K weights and
the gray output.

This is why:
- the **official** f16 (converted on Mac) is correct,
- **our** f16 (converted on Linux) is garbage **only** on interleaved Q/K,
- the quantizer, storage format, mapping names, dims, and norms are all fine, and
- the same bug hits both the control model and the custom `10_e_v1`.

### 17.3 The fix (two graph attempts FAILED; a manual CPU permutation WORKS)

**First attempt (WRONG):** do the same reshape/transpose on the CPU, dropping the
`.toGPU()`/`.toCPU()` and adding `.contiguous()`:

```swift
graph.variable(tensor.reshaped([numHeads,2,headDim/2,-1]))
  .transposed(1, 2).contiguous().reshaped([numHeads*headDim,-1]).rawValue
```

This compiled but **still produced garbage** — different garbage: `__dit__[t-x_q-0-1]`
max `23952`, and `__text_video_connector__[t-to_q-0-1]` came back byte-identical to the
*norm* tensor's memory. Conclusion: **s4nnc's graph `transposed()` is broken on the
CPU-only converter both ways** — with or without the GPU round-trip it reads out of
bounds. Do not use graph transpose in the converter.

**Working fix:** the transform is a **pure row-permutation** of the output dimension
(verified value-for-value against the raw safetensors and the official f16 — the raw
and official share the same value set at 100%, so no arithmetic is involved). So do it
by hand on the CPU with raw pointers, no graph ops:

```swift
case .O:
  if interleaved, numberOfHeads > 0, headDimension > 0 {
    let shape = tensor.shape
    let outDim = shape[0]
    if outDim == numberOfHeads * headDimension {
      var cols = 1
      for i in 1..<shape.count { cols *= shape[i] }
      let hd = headDimension
      let half = headDimension / 2
      let source = Tensor<FloatType>(from: tensor)  // guarantee contiguous
      var result = Tensor<FloatType>(.CPU, format: tensor.format, shape: shape)
      source.withUnsafeBytes { srcRaw in
        let src = srcRaw.bindMemory(to: FloatType.self)
        result.withUnsafeMutableBytes { dstRaw in
          let dst = dstRaw.bindMemory(to: FloatType.self)
          for h in 0..<numberOfHeads {
            let base = h * hd
            for d in 0..<half {
              for p in 0..<2 {
                let dstOff = (base + d * 2 + p) * cols
                let srcOff = (base + p * half + d) * cols
                for c in 0..<cols { dst[dstOff + c] = src[srcOff + c] }
              }
            }
          }
        }
      }
      tensor = result
    }
  }
```

The exact index map (verified 100% vs official, `headDim=128`, `numHeads=32`):

$$\text{out}[h \cdot hd + d \cdot 2 + p] = \text{in}[h \cdot hd + p \cdot \tfrac{hd}{2} + d]$$

### 17.4 Captured in DRAW_THINGS_PATCH (never touch upstream)

`TensorDescriptor.swift` was **added to** `tools/generate_drawthings_quant_patches.sh`
(both the snapshot-copy section and the `PATCH_TARGETS` array) — it was not previously
tracked there. Running that script snapshots the file into
`DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift`
and regenerates `DRAW_THINGS_PATCH/patches/draw-things-community.patch`, so the fix is
reproducible from a clean upstream checkout without ever editing the
`draw-things-community` repo. (Both done — the diff hunk is present in the unified
patch.)

### 17.5 Verification method (cheap first, before the 2h requantize)

1. Rebuild `model-converter` (~6 min).
2. Reconvert the control (~11 min, CPU).
3. **Cheap gate:** re-run the value comparison for `__dit__[t-x_q-0-1]` and
   `__text_video_connector__[t-to_q-0-1]`. Fix confirmed iff the biases are now small
   (≈ ±0.4 / ±0.07), not `65024`.
4. Only then: requantize (~2h) → restore F32 norms → GPU canary → render a coherent
   frame.
5. Apply the same proven pipeline to `10_e_v1_bf16.safetensors`.

### 17.6 One-line summary (§17)

The converter mangled attention weights because the `interleaved` rotary transpose ran
through s4nnc's graph `transposed()`, which is broken on our GPU-less CPU converter
(both the `.toGPU()` round-trip and a pure-CPU `.transposed().contiguous()` read out of
bounds). Replacing it with a **manual raw-pointer row-permutation** (no graph ops)
produces weights that match Draw Things' official f16 exactly, and the change is
captured in `DRAW_THINGS_PATCH`.

### 17.7 CONFIRMED RESULT (2026-07-10) — the fix produces correct weights

Rebuilt `model-converter` with the manual permutation, reconverted the control
(`RESULT=PASS`, 5746 tensors), and ran the cheap gate
(`tools/dt_verify_interleaved_fix.py`, ours vs official f16):

| tensor | ours (fixed) | official | \|max\| ratio |
|---|---|---|---|
| `__dit__[t-x_q-0-1]` | −0.3828 … **0.4375** | −0.3828 … 0.4395 | **1.00** |
| `__text_video_connector__[t-to_q-0-1]` | −0.0737 … **0.0588** | −0.0737 … 0.0588 | **1.00** |

`RESULT=PASS interleaved Q/K weights are sane`. The garbage (`65024` / `23952`) is
gone; the values now match the official f16. **The converter root cause is fixed** and
the corrected `dt-models/ltx23_control_f16.ckpt` is on disk.

Downstream (in progress): requantize the corrected f16 → q6p → restore F32 norms →
GPU render → confirm a coherent frame → apply the same to `10_e_v1_bf16.safetensors`.

**Performance note (to optimize later):** the manual reconvert took ~63 min vs ~12 min
for the graph version (mostly `sys`/I-O wait, `user` only ~1.5 min — the extra
per-tensor allocations/copies, not the arithmetic). The element-wise inner copy can be
replaced with a per-row bulk copy (`UnsafeMutablePointer.update(from:count:)`) and the
`Tensor(from:)` source copy can be dropped once the input is known-contiguous. Not on
the critical path to correctness; revisit before large custom-model runs.

## 18) SECOND INTERLEAVE BUG — QK-norms Not De-interleaved (2026-07-10, end-to-end render)

The Q/K interleave fix (§17) was necessary and produced a **huge** jump: gray → richly
structured output. But it is **not sufficient** on its own. Full requantize + norm
restore + GPU render revealed a **second, sibling bug**: the QK-normalization weights
need the same de-interleave and our converter isn't applying it.

### 18.1 End-to-end render result (corrected q6p)

- Requantized the corrected f16 → q6p (2h10m, 20 GB, 4 BF16 embeds preserved).
- Restored 1540 F32 norms (official f16 as value source; norm values are identical to
  ours, verified). Datatype dist matches the proven-coherent isolation q6p (4202/1540/4
  vs 4200/1542/4).
- GPU render (`ltx23_control`, 384×384, 8 steps, seed 4242): **`RESULT=PASS`,
  RENDER_EXIT=0, 23 responses, 9 images, 1 audio, "Image processed successfully"**, ~5.5 min.
- Decoded images (`.bin` = 68-byte tensor header + 384×384×3 F16): **all 9 frames have
  `std ≈ 0.48`** — richly structured, **definitively NOT gray** (a gray frame is
  `std ≈ 0`). Saved PNGs in `output/q6p_canary_ltx23_control_fixed_*/image_r00XX_01.png`.

### 18.2 But not yet coherent — vs the isolation control

Side-by-side, same seed/steps/pipeline:

- **Isolation (official weights):** a clean, coherent **sports car at sunset**.
- **Ours (corrected converter):** structured, warm-colored, reflective — but
  **abstract**, no coherent scene (vertical banding / panel seams).

So a residual converter error remains, beyond the §17 Q/K interleave.

### 18.3 Diagnostic — attention *projections* are now perfect

Compared our f16 vs official f16 (both store these inline, directly readable):

| weight | match |
|---|---|
| `__dit__[t-x_q-0-0]` self-attn Q | **100.0%** |
| `__dit__[t-x_k-0-0]` self-attn K | **100.0%** |
| `__dit__[t-x_v-0-0]` self-attn V | **100.0%** |
| `__dit__[t-xa_q-0-0]` cross-attn Q | **100.0%** |
| `__dit__[t-xa_k-0-0]` cross-attn K | **100.0%** |
| `__text_video_connector__[t-to_q/to_k-0-0]` | **100.0%** |

The §17 fix fully corrected the Q/K/V **weights** (not just the biases). Attention
projections are done.

### 18.4 The remaining bug — QK-norm tensors are permuted

A type-level sweep (one instance of every tensor kind, layer 0, our f16 vs official)
found **138 types match, 1 skipped (oversize), and exactly 17 mismatched — every one a
`q_norm`/`k_norm`**:

```
  6.2%  __dit__[t-x_norm_q-0-0]        (self-attn query-norm)
  6.4%  __dit__[t-x_norm_k-0-0]
  7.4%  __dit__[t-xa_norm_q-0-0]       (cross-attn)
  4.9%  __dit__[t-ca_norm_q/k-0-0]
 24.9%  __dit__[t-cv_norm_q/k-0-0]
  7.9%  __dit__[t-a_norm_q/k-0-0]      (audio)
  7.9%  __dit__[t-ax_norm_q/k-0-0]
 10.9%  __text_video_connector__[t-norm_q/k-0-0]
  8.2%  __text_audio_connector__[t-norm_q/k-0-0]
```

These are the **QK-normalization** weights — `RMSNorm(axis:[2], name:"norm_q"/"norm_k")`
applied to the query/key vectors before attention (LTX2.swift L123-127, L250-254, etc.).

Applying the **same de-interleave permutation** as Q/K makes them match official:

| norm | raw match | de-interleaved match | headDim |
|---|---|---|---|
| `__dit__[t-x_norm_q-0-0]` | 6.2% | **99.3%** | 128 |
| `__dit__[t-x_norm_k-0-0]` | 6.4% | **99.3%** | 128 |
| `__dit__[t-xa_norm_q-0-0]` | 7.4% | **99.9%** | 64 |
| `__text_video_connector__[t-norm_q-0-0]` | 10.9% | **100.0%** | 128 |

Note **headDim varies by attention type** (self-attn 128, cross-attn 64) — the same
`(numberOfHeads, headDimension)` as that block's `to_q`/`to_k`.

### 18.5 Root cause

The Q/K **projections** carry `interleaved: true` in the mapping, so they hit the §17
de-interleave. The **QK-norms do not** — they're plain name arrays:

- `Libraries/SwiftDiffusion/Sources/Models/LTX2.swift`: e.g. L177-178
  `mapping["\(prefix).attn1.k_norm.weight"] = [normK.weight.name]` (and the mirror
  `q_norm`), repeated for cross-attn / block / DiT-single / audio variants
  (L312-313, L428-431, L1328-1329, L1472-1473, …).
- `Libraries/ModelOp/Sources/ModelImporter.swift` `makeLTX23ConnectorMapping` L1225-1226
  `mapping["\(base).attn1.k_norm.weight"] = ["t-norm_k-\(i)-0"]` (and `q_norm`).

At runtime the norm is applied to the **already de-interleaved** q/k (because `to_q`/
`to_k` are stored de-interleaved), so the norm weight **must also be de-interleaved**.
Draw Things' official f16 has them de-interleaved; ours are left in raw (interleaved)
order → the QK-norm scales the wrong channels → attention is subtly scrambled →
structured-but-incoherent output.

### 18.6 The fix (final piece)

Flag every `q_norm.weight` / `k_norm.weight` mapping as `interleaved: true` with the
**same `numberOfHeads` / `headDimension` as the adjacent `to_q`/`to_k`** (all in scope
at each site), so they flow through the §17 manual row-permutation in
`TensorDescriptor.swift`. That permutation already handles a 1-D `[numHeads*headDim]`
tensor (`cols = 1`).

Sites:
- `LTX2.swift`: all `q_norm.weight` / `k_norm.weight` mapping lines (self-attn, cross-
  attn, block, DiT-single, audio) — use the block's local `h` / `k`.
- `ModelImporter.swift` `makeLTX23ConnectorMapping` L1225-1226 — use `numberOfHeads` /
  headDimension already used by the connector `to_q`/`to_k` just above (L1208-1217).

Both files are captured via `DRAW_THINGS_PATCH` (LTX2.swift will need adding to the
patch-target list, like TensorDescriptor.swift was). Then: rebuild → reconvert →
verify all 17 norm types now match official (cheap gate) → requantize → restore norms →
render → expect a **coherent** frame.

### 18.7 Why this should be the last piece

The type-level sweep found **only** the QK-norms mismatched; every other tensor kind
(projections, FF/MLP, ada_ln, embedders, scale_shift, out-proj) already matches official
100%. Fixing the QK-norm de-interleave closes the last identified gap between our
converter output and Draw Things' known-good weights.

### 18.8 One-line summary (§18)

The §17 fix corrected the Q/K/V *projections* (gray → structured), but the sibling
**QK-norm** weights (`q_norm`/`k_norm`) were left interleaved because their mappings lack
`interleaved: true`; flagging them (same head config as `to_q`/`to_k`) routes them through
the same de-interleave and should yield a coherent frame.

## 19) COHERENT FRAME ACHIEVED + QK-norm fix applied (2026-07-13)

### 19.1 Fast validation (no requantize) — CONFIRMED coherent

The QK-norms are palettized in the q6p and are **not** in the official q6p's F32 set, so
`dt_q6p_restore_f32_norms.py` skips them. That means we can overwrite them **directly in
the q6p** with the correct (de-interleaved) values as F32 — validating the fix WITHOUT a
2h requantize:

- `tools/dt_q6p_restore_qk_norms.py --q6p ltx23_control_q6p.ckpt --f16-source <official f16>`
  copied all **608** QK-norm tensors (every layer) as F32 (dim `[1,1,N]` from the source
  dim blob). Verified 100% match to official for spot-checked tensors.
- GPU render (`ltx23_control`, 384×384, 8 steps, seed 4242): **`RESULT=PASS`**, `std` rose
  0.47 → **0.62** (matching the coherent isolation control), and the image is a **clean,
  coherent red sports car on a road at sunset** — mountains, sky, road markings, headlights.
  `output/q6p_canary_ltx23_control_qknorm_*/image_r0022_01.png`.

**This proves the full fix**: §17 (projections) + §18 (QK-norms) → a coherent frame from
OUR converted model, matching the official-weights control.

### 19.2 Head rule

Every LTX2.3 QK-norm has `numberOfHeads = 32`; `headDimension = N/32` (self-attn N=4096→128,
cross-attn N=2048→64). The converter fix uses each block's own `h`/`k`, so it is exact
without relying on this rule; a post-process would use `headDim = N/32`.

### 19.3 Permanent converter fix (applied, in DRAW_THINGS_PATCH)

The importer uses the non-LoRA `LTX2` / `LTX2Fixed` (verified: `ModelImporter.swift`
L1490/1496/1516/1522) and the hand-written `makeLTX23ConnectorMapping`. So the QK-norm
mappings were flagged `interleaved: true` (same head config as the sibling `to_q`/`to_k`)
at exactly the **used** sites:

- `LTX2.swift` `LTX2SelfAttention` (`k_norm`/`q_norm`, `h`/`k`) — video+audio self-attn.
- `LTX2.swift` `LTX2CrossAttention` (`k_norm` optional/`q_norm`, `h`/`k.1`) — cross / a2v / v2a.
- `LTX2.swift` `LTX2CrossAttentionFixed` (`k_norm`, `h`/`k.1`) — the KV-fixed cross path.
- `ModelImporter.swift` `makeLTX23ConnectorMapping` L1225-26 (`k_norm`/`q_norm`,
  `numberOfHeads`/`headDimension`).

The LoRA twins (`LoRALTX2*`) and `BasicTransformerBlock1D`/`Embedding1DConnector` were left
unchanged — the importer does not use them for the raw-mapping conversion (harmless). Both
`LTX2.swift` and `TensorDescriptor.swift` are now in
`tools/generate_drawthings_quant_patches.sh` (snapshot + `PATCH_TARGETS`); the upstream
`draw-things-community` repo is never edited directly.

### 19.4 Remaining to reach a fully converter-native coherent frame

Rebuild `model-converter` → reconvert control → cheap-verify the 17 norm types now match
official (they should, like the projections did) → requantize → restore F32 norms → render.
Then run the same pipeline on `10_e_v1_bf16.safetensors` (the actual custom target).

### 19.5 Converter-native fix VERIFIED (2026-07-13)

Rebuilt + reconverted the control. Compared all 16 QK-norm types to official:

- 14 types at **99-100%**; `ca_norm_q`/`ca_norm_k` at 96.2%/96.8% @ `atol=3e-3`, but
  **maxdiff is only 0.0078** and they hit **100% @ `atol=1e-2`** — pure F16 rounding, not a
  wrong transform (`deint(hd=64)`=96% while every other headDim scores 24-49%, so hd=64 is
  provably correct; `ca` is a 2048-dim / headDim-64 attention).
- Projection weights still **100%**.

So the converter now emits correctly de-interleaved Q/K/V projections **and** QK-norms
natively — no post-processing needed. The importer uses the non-LoRA `LTX2`/`LTX2Fixed`
paths, so only `LTX2SelfAttention`, `LTX2CrossAttention`, `LTX2CrossAttentionFixed` and the
connector mapping needed flags; the LoRA twins were left untouched (unused, harmless).

## 20) CONVERSION SPEEDUP — `PRAGMA synchronous=OFF` (2026-07-13)

### 20.1 The bottleneck

Conversion was **disk-I/O-bound, not CPU-bound**. A 44 GB convert took ~56-63 min but
used only ~1.5 min of `user` CPU; the rest was `sys` + I/O wait. Evidence while running:
process in `D` (uninterruptible I/O wait), a **22 GB SQLite rollback journal**, and only
~12-15 MB/s throughput. s4nnc opens the checkpoint store with SQLite's defaults
(`journal_mode=DELETE`, `synchronous=FULL`), so every commit `fsync`s and the journal is
flushed synchronously — pathological on the WSL2 virtual disk.

### 20.2 The fix

Add `PRAGMA synchronous=OFF` on the **write** path of s4nnc's `openStore`
(`checkouts/s4nnc/nnc/Store.swift`, right after the write-path `auto_vacuum`). Model
conversion/quantization outputs are one-shot and fully regenerable, so per-commit fsync
durability is pointless. Journaling stays on (so `VACUUM`/rollback still work); only the
fsync/synchronous flush is skipped, letting writes stream through the page cache.

### 20.3 Measured result (custom `10_e_v1` convert)

| metric | before (`synchronous=FULL`) | after (`synchronous=OFF`) |
|---|---|---|
| read rate | ~15 MB/s | **~110 MB/s** |
| write rate | ~12 MB/s | **~42 MB/s** |
| SQLite journal | **22 GB** | **~4.6 KB** |
| process state | `D` (I/O wait) | `R` (running) |
| wall-clock (44 GB) | ~56 min | **~7 min (≈8×)** |

### 20.4 Captured in DRAW_THINGS_PATCH

`checkouts/s4nnc/nnc/Store.swift` was added to `tools/generate_drawthings_quant_patches.sh`
(snapshot copy + the `s4nnc.patch` git-diff now covers `Package.swift nnc/Store.swift`).
The s4nnc checkout files are read-only, so applying edits there requires `chmod +w` first.
The upstream repos are never edited directly.

## 21) CUSTOM 10_e_v1 END-TO-END — sampled but DIVERGED (2026-07-13, OPEN)

First full run of the actual target `10_e_v1` through the fixed pipeline. It got a lot
further than ever before but did **not** yield a frame — the DiT **diverges** during
sampling.

### 21.1 Pipeline run (all succeeded up to render)

- Convert `10_e_v1_bf16.safetensors` → f16 with the fixed converter: **RESULT=PASS**,
  15m15s (~3.7× faster thanks to §20), Q/K biases sane, no garbage.
- Requantize → `10_e_v1_q6p.ckpt` (20 GB), 1h41m (CPU-bound palettization).
- Restore norms from `10_e_v1`'s **own** values (widened F16→F32): 1540 ada_ln + 608
  QK-norms → q6p now has 2148 F32 + 4 BF16 embeds (matches the coherent control's shape).
- Alias `10_e_v1` → q6p (mirrors `ltx23_control`).

### 21.2 The failure

GPU render (384×384, 8 steps, seed 4242): the model loads and **samples** (5 preview
frames emitted) but the **final decode produced no images** — `RESULT=FAIL, no final
generated output payloads`, "Image processed failed", 0 images.

Decoding the latent previews shows the DiT is **numerically diverging**:

| preview | NaN | Inf | min / max | std |
|---|---|---|---|---|
| 0004 | 62 | 4 | −64512 / 65024 | 4447 |
| 0010 | 364 | 5 | −64512 / 65024 | 6031 |

Latents blow up to F16 overflow (±65024) and NaN count **grows** over steps → classic
divergence, so the decode has nothing valid to render.

### 21.3 What's ruled out — the custom weights are fine

Scanned the custom f16: **no tensor has |max|>50, and none contain NaN/Inf.** Compared
every tensor type to the control f16: differences are tiny (maxdiff 0.01–0.098) and every
magnitude matches the control exactly — `10_e_v1` is a **light fine-tune of the same base**
and its converted weights are sane. So divergence is **not** an anomalous/garbage weight.

### 21.4 Prime suspect — the converter-native path was never validated end-to-end

The coherent control frame (§19.1) used the **q6p-surgery** path: quantize the *old*
control f16 + overwrite norms with the **official** f16's F32 values. The
**converter-native** path (reconvert with the §18 QK-norm fix → requantize → restore the
model's **own** F16→F32-widened norms → render) was first exercised here, on the custom
model, and it failed. So the bug is most likely in that path, not in `10_e_v1`:

- **H1 — F16→F32 widened norms.** Control restored *official F32-native* norms; custom
  restored *F16→F32-widened* norms. If norm precision/values matter, this differs.
- **H2 — QK-norm F32 restore may be wrong or unnecessary here.** After quantizing an
  already-de-interleaved f16, the q6p's palettized QK-norms are already correct (like the
  coherent *isolation* run, which never restored them). Overwriting them with F32 (via
  `dt_q6p_restore_qk_norms.py`, using the custom f16's dim blob) may have introduced a
  dim/precision problem. Worth testing WITHOUT the QK-norm restore.
- **H3 — the 1540-norm restore for a converter-native f16.** Same tool, but the source
  norms are the model's own (fine-tune) values; a wrong dim/widen could destabilize.

### 21.5 Next steps (for the continuation)

1. **Isolate the path, not the model:** requantize the on-disk **converter-native control
   f16** (`ltx23_control_f16.ckpt`, already has de-interleaved norms) → restore norms →
   render. If it also diverges, the bug is in the converter-native quantize/restore path
   (not `10_e_v1`). If it's coherent, the issue is specific to the custom norms.
2. **Ablate the QK-norm restore:** re-quantize the custom f16 and render WITHOUT running
   `dt_q6p_restore_qk_norms.py` (leave the palettized de-interleaved QK-norms, as in the
   coherent isolation run). If that renders, the QK-norm F32 restore is the culprit.
3. **Compare norm values** custom-vs-official for the 1540 F32 set and the QK-norms — did
   the fine-tune change norms enough to matter, or is a restored dim/value off?
4. Consider whether F16→F32 widening needs the exact F32 dim encoding the runtime expects.

Nothing here is captured in DRAW_THINGS_PATCH (no source change) — this is a pipeline /
q6p-post-process investigation. The converter fixes (§17–20) stand; the open item is the
q6p norm-restore path for a converter-native model.

## 22) RESOLVED — QK-norm tensor RANK ([N] vs [1,1,N]) (2026-07-14) — CUSTOM MODEL COHERENT

The §21 divergence was the QK-norm **`dim` (tensor rank), not the values**. Comparing dim
blobs:

| tensor source | `x_norm_q` dim |
|---|---|
| Draw Things **official** f16 | `[1, 1, 4096]` (rank-3) ✓ |
| **our converter** f16 (custom AND control) | `[4096]` (rank-1) ✗ |
| control q6p (coherent) | `[1,1,4096]` — the surgery sourced dim from the *official* f16 |
| custom q6p (diverged) | `[4096]` — `dt_q6p_restore_qk_norms.py` copied *our* f16's dim |

So the coherent control frame worked **by accident**: its QK-norm surgery copied the dim
from the official f16 (rank-3). The custom model, restored from our own f16, inherited the
rank-1 `[4096]` dim. The runtime's RMSNorm expects `[1,1,N]`; given `[N]` it mis-applies
the norm, the DiT latents overflow F16, and sampling diverges (§21.2).

### 22.1 The fix

`tools/dt_q6p_restore_qk_norms.py` now rebuilds the dim as `[1,1,N]` (rank-3) regardless
of the source, using the same blob width:

```python
ndim = len(s_dim) // 4
n_elem = len(f32_bytes) // 4
dim_blob = struct.pack("<%di" % ndim, *([1, 1, n_elem] + [0] * (ndim - 3)))
```

Re-ran it on the existing custom q6p (no requantize) → all 608 QK-norms now `[1,1,4096]` /
`[1,1,2048]`, matching the coherent control exactly.

### 22.2 Result — coherent custom frame

Re-render (384×384, 8 steps, seed 4242): **`RESULT=PASS`, 9 images, no NaN, no F16
overflow, `std ≈ 0.53`** (coherent range). The image is a **clean red car on a mountain
road at sunset** — the first coherent frame from the custom `10_e_v1` model. **Original
objective achieved.**

### 22.3 The full working recipe (converter-native, any LTX2.3 fine-tune)

1. Convert safetensors → f16 with the fixed converter (§17 + §18). Fast (§20).
2. Requantize f16 → q6p (~1h40m, CPU-bound).
3. `dt_q6p_restore_f32_norms.py --q6p <q6p> --f16-source <our f16> --ref-q6p <official q6p>`
   (1540 ada_ln norms; ref-q6p supplies the correct dim).
4. `dt_q6p_restore_qk_norms.py --q6p <q6p> --f16-source <our f16>` (608 QK-norms, now with
   the `[1,1,N]` dim fix).
5. Alias → q6p; render.

### 22.4 Follow-up (optional cleanup)

Our converter writes ALL norms rank-1 `[N]`; the restores paper over it (ada_ln via the
ref-q6p dim, QK via the new `[1,1,N]` reconstruction). A cleaner long-term fix would make
the converter emit `[1,1,N]` norms directly (TensorDescriptor/norm write path), removing
the need for the dim reconstruction — but the current pipeline works end-to-end.

---

## 23) LoRA §17 de-interleave — statically PROVEN correct (2026-07-16)

Open question from §22: does the §17 Q/K de-interleave permutation (which the base-model
converter applies via `TensorDescriptor.internalWrite` `case .O:`) also correctly apply to
the **LoRA up-matrix**? If the LoRA up were left interleaved while the base is de-interleaved,
attention math would be corrupted → the NaN we see in q4p+LoRA final decode.

### 23.1 Method (static, no 25-min render)

Compared, for `transformer_blocks.0.attn1.to_q`:
- **Raw** safetensors up = `diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight`
  (bf16, `[4096,72]`) from `ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors`.
- **Stored** ckpt up = `__dit__[t-x_q-0-0]__up__` (f16, `[4096,72]`) from the converted
  `ltx_2.3_22b_distilled_lora_fro90_ceil72_lora_f16.ckpt` (SQLite `tensors` table).

Applied the §17 row permutation on the out dim (numHeads=32, hd=128, half=64):
`out[h*128 + d*2 + p] = in[h*128 + p*64 + d]`.

### 23.2 Result — §17 IS applied to the LoRA

| Relationship        | max abs diff |
|---------------------|--------------|
| identity (raw==stored)     | `0.0769`     |
| **§17 permuted (raw→stored)** | **`5.9e-08`** |
| §17 inverse         | `0.0740`     |

The permuted match is exact to bf16→f16 rounding. **The converter correctly de-interleaves
the LoRA up-matrix, consistent with the base model.** §17 is NOT the source of the q4p+LoRA
NaN. Candidate (a) is eliminated; the remaining hypothesis is (b) runtime numeric divergence
(q4p quant error + distillation LoRA at weight 1.0 pushing the final latent/VAE decode to
NaN, which TAESD previews mask by clamping non-finite to 0).

### 23.3 LoRA weight 0.6 — still NaN (2026-07-16)

Re-ran q4p base (alias `10_e_v1_4`) + LoRA at **weight 0.6** (5 frames, 384², 8 steps, seed
4242), timeout raised to 2700s so the final decode can complete.

- Previews progressed normally (0004→0010), no divergence visible in TAESD.
- Final decode → **`Image processed failed`** (server logs `Image processed` then the failure).
  Config confirms `"loras": [["file": ..., "mode": "all", "weight": 0.6]]`. Same NaN signature
  as weight 1.0 (`guard !isNaN` at `LocalImageGenerator.swift:4471` returns nil).
- Run: `output/q6p_canary_10_e_v1_4_test_20260716_130421/`.

**Conclusion:** reducing the LoRA weight does NOT avoid the NaN. The "weight-1.0 overdrive"
sub-hypothesis is weakened — the final-latent/VAE NaN persists at 0.6. The divergence is not a
simple magnitude effect. Remaining candidates: (b1) quantization (q4p too lossy) interacting
with *any* LoRA delta — test via **f16 base + LoRA** to isolate quant vs LoRA-conversion; or
(b2) a subtler LoRA-conversion error beyond §17 (other projections / down-matrix / scaling).

> NOTE: first 0.6 attempt (`..._123206/`) hit the **1800s harness timeout** during final decode
> (sampling was ~2–3 min slower than the weight-1.0 run, leaving too little decode headroom).
> Use `--timeout-sec 2700` for LoRA runs. Inconclusive; superseded by the 2700s run above.

---

## 24) FULL static LoRA-conversion audit — conversion is CORRECT (2026-07-16)

To decide between (b1) quantization interaction vs (b2) a LoRA-conversion bug **without** more
25-min renders, we statically audited the entire converted LoRA against the raw safetensors.

### 24.1 Method

For every attention q/k/v/out projection in `transformer_blocks.0`, compare:
- **Raw**: `diffusion_model.transformer_blocks.0.<module>.lora_B.weight` (up) / `.lora_A.weight`
  (down), bf16, from `dt-models/ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors`.
- **Stored**: `__dit__[t-<token>-0-0]__up__` / `__down__`, f16, from the converted
  `dt-models/ltx_2.3_22b_distilled_lora_fro90_ceil72_lora_f16.ckpt` (SQLite `tensors` table).

Classify each up-matrix as **IDENTITY** (raw==stored) or **PERMUTED** (raw→stored under the
§17 row de-interleave `out[h*hd + d*2 + p] = in[h*hd + p*(hd/2) + d]`, numHeads=32). Modules were
auto-linked ckpt-token ↔ sf-module by matching the (identity) down-matrices by content.
Tool: ad-hoc `.venv/bin/python` + `safetensors` + `sqlite3` (re-runnable; see §24.4).

### 24.2 Result — every projection converts exactly as the base model does

| ckpt token | sf module | up shape | up verdict (max diff) | down |
|-----------|-----------|----------|-----------------------|------|
| `x_q`  | attn1.to_q              | (4096,72) | **PERMUTED** (p17=5.9e-8) | IDENTITY |
| `x_k`  | attn1.to_k              | (4096,60) | **PERMUTED** (5.9e-8)     | IDENTITY |
| `x_v`  | attn1.to_v              | (4096,72) | IDENTITY (0.0)            | IDENTITY |
| `x_o`  | attn1.to_out.0          | (4096,72) | IDENTITY (0.0)            | IDENTITY |
| `a_q`  | audio_attn1.to_q        | (2048,72) | **PERMUTED** (5.6e-8)     | IDENTITY |
| `a_k`  | audio_attn1.to_k        | (2048,72) | **PERMUTED** (5.2e-8)     | IDENTITY |
| `cv_q` | attn2.to_q (text cross) | (4096,11) | **PERMUTED** (5.2e-8)     | IDENTITY |
| `cv_k` | attn2.to_k (text cross) | (4096, 6) | **PERMUTED** (5.6e-8)     | IDENTITY |
| `ca_q` | audio_attn2.to_q        | (2048,72) | **PERMUTED** (5.6e-8)     | IDENTITY |
| `ca_k` | audio_attn2.to_k        | (2048,72) | **PERMUTED** (5.9e-8)     | IDENTITY |
| `xa_q` | video_to_audio_attn.to_q| (2048,52) | **PERMUTED** (5.2e-8)     | IDENTITY |
| `xa_k` | video_to_audio_attn.to_k| (2048,36) | **PERMUTED** (5.6e-8)     | IDENTITY |
| `ax_q` | audio_to_video_attn.to_q| (2048,72) | **PERMUTED** (5.9e-8)     | IDENTITY |
| `ax_k` | audio_to_video_attn.to_k| (2048,48) | **PERMUTED** (5.8e-8)     | IDENTITY |

Facts established:
1. **All rotary q/k up-matrices are §17-permuted** (match raw under the permutation to ~5e-8),
   at all 6 attention sites including both text cross-attentions (`attn2`/`audio_attn2`).
2. **Non-rotary v/out up-matrices are identity** (correctly NOT permuted).
3. **All down-matrices are identity** (correct — the §17 row-permutation acts on the *output*
   dim; the down-matrix `[rank, in]` has no output dim to permute).
4. **No alpha/scale tensors exist** in the safetensors (0 keys outside `lora_A`/`lora_B`), so
   there is no LoRA alpha for the converter to mishandle. Per-projection **ranks vary**
   (q=72, k=60, cv_k=6 …) and are all preserved correctly.
5. The LoRA uses the **identical `LTX2`/`LTX2Fixed` mapper** as the base model (`interleaved:
   true` at every attention site — `LTX2.swift:162/297/413/823/1324/1468/1588/2012`). The base
   model conversion is **proven coherent (§22)**, so these flags are correct for LTX-2.3, and
   the LoRA up-matrix permutation matches them exactly.

### 24.3 Conclusion — (b2) eliminated; the failure is (b1) numeric divergence

The converted LoRA is **structurally and numerically correct**: it applies the §17 de-interleave
to exactly the projections the (coherent) base model does, to bf16→f16 tolerance, with identity
down-matrices and correct per-projection ranks. **There is no LoRA-conversion bug.**

Combined with §23.3 (weight 0.6 still NaN), the q4p+LoRA failure is **(b1): the q4p quantization
error, compounded by the distillation-LoRA delta, drives the *final latent* to NaN/inf.** The
base q4p alone is coherent (§22); adding the LoRA tips it over. TAESD previews hide it (they clamp
non-finite → 0); the full VAE decode does not, so `guard !isNaN` at
`LocalImageGenerator.swift:4471` returns nil → server logs `Image processed` → `Image processed
failed`. The f32 high-precision VAE fallback (`FirstStage.swift:1018-1034`, already active on
Linux since `isMaxPerformance==true`) also NaNs, confirming the NaN is *in the latent*, not the
decode arithmetic.

### 24.4 Re-run the audit (copy/paste)

```bash
.venv/bin/python - <<'PY'
from safetensors import safe_open
import numpy as np, sqlite3, struct
sf="dt-models/ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors"
db="dt-models/ltx_2.3_22b_distilled_lora_fro90_ceil72_lora_f16.ckpt"
c=sqlite3.connect(db)
def stored(n):
    r=c.execute("SELECT data,dim FROM tensors WHERE name=?", (n,)).fetchone()
    if not r: return None
    d,dim=r; k=len(dim)//4; dims=[x for x in struct.unpack("<%di"%k,dim) if x>0]
    return np.frombuffer(d,dtype=np.float16).astype(np.float32).reshape(dims)
def perm(out,h):
    hd=out//h; hf=hd//2; p=np.empty(out,np.int64)
    for i in range(h):
        for d in range(hf):
            for q in range(2): p[i*hd+d*2+q]=i*hd+q*hf+d
    return p
with safe_open(sf,framework="pt") as f:
    b=f.get_tensor("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight").float().numpy()
u=stored("__dit__[t-x_q-0-0]__up__")
print("identity",abs(b-u).max(),"perm17",abs(b[perm(4096,32)]-u).max())
PY
```

---

## 25) HANDOFF — state, constraints, and next steps (2026-07-16)

### 25.1 One-paragraph status

**Goal:** apply the distilled LoRA to the custom quantized LTX-2.3 model and produce a coherent
frame via `dt_test.sh`. **Where we are:** base q4p (alias `10_e_v1_4`) renders coherently
(§22, std≈0.53). The **LoRA converts correctly** (§23/§24 — §17 de-interleave applied to exactly
the right projections, verified numerically to ~5e-8). But **q4p + LoRA NaNs at the final VAE
decode**, at both weight 1.0 and 0.6. Root cause is **numeric divergence from q4p quantization
loss + the distillation-LoRA delta pushing the final latent to NaN** — NOT a converter/LoRA bug.

### 25.2 Hard constraints (read before running anything)

- **f16 base + LoRA is NOT runnable** — `10_e_v1_4_f16.ckpt` is 44 GB, exceeds available VRAM.
  Do not attempt; it will OOM.
- **`frames=1` CRASHES** (SIGSEGV in libcuda — invalid temporal size for a video model). Use
  `frames=5` (or more).
- Each render is **~25–30 min** on this GPU/WSL setup regardless of LoRA. Use
  **`--timeout-sec 2700`** for LoRA runs (final decode needs headroom; 1800s can cut it off —
  see §23.3 note).
- Always render via the **alias `10_e_v1_4`** (as `dt_test.sh` does), never a raw ckpt filename —
  the alias pulls the correct VAE/text-encoder companions from `custom.json`; a raw filename
  bypasses them and produces noise (a harness artifact, not a model bug).
- Any edit under `draw-things-community/` MUST be mirrored into `DRAW_THINGS_PATCH/`.

### 25.3 Recommended next steps, in priority order

1. **q6p base + LoRA** (best fit for the VRAM constraint; directly tests b1).
   We have `dt-models/10_e_v1_4_q6p.ckpt` (21 GB, less lossy than q4p, and proven coherent
   standalone in §22). q6p gives more numeric headroom than q4p.
   - **Prereq:** point the alias at the q6p files. Edit `dt-models/custom.json` entry
     `10_e_v1_4`: set `file` and `clip_encoder` to `10_e_v1_4_q6p.ckpt` (keep the VAE/
     text-encoder/condition_scale/modifier fields). NOTE: q6p needs the §22 norm restores
     applied (`dt_q6p_restore_f32_norms.py` + `dt_q6p_restore_qk_norms.py`) — verify they were
     run on this q6p file before trusting a result (the standalone q6p coherence in §22 implies
     they were, but confirm).
   - **Run:** `bash tools/dt_test.sh 10_e_v1_4 --lora ltx-2.3-22b-distilled-lora-fro90_ceil72:1.0 --frames 5 --timeout-sec 2700`
   - **Read-out:** COHERENT → confirms q4p-specific quant loss is the culprit; q6p (or q8p) is the
     shippable base for LoRA use. Still NaN → the divergence is not purely q4p; go to step 2/3.

2. **Sampler/step-count for a distillation LoRA.** `fro90_ceil72` is a *distillation/acceleration*
   LoRA — it changes the sampling trajectory and is typically meant for a specific (low) step
   count / scheduler. The current config is a **two-stage** schedule (`stage2Steps:10,
   imagePriorSteps:5, sampler:17, teaCacheStart:5`, 8 stage-1 steps). Applying an accel-LoRA on
   top of an already-distilled two-stage schedule may over-accelerate and diverge. Try a plain
   step count / disable stage-2 / different sampler and see whether the latent stays finite.
   (Cheap-ish variations, still ~25 min each.)

3. **Locate the first NaN step (requires a debug build).** Add temporary NaN-scan logging in the
   sampler loop and in `FirstStage.decode` to find whether the NaN appears mid-sampling (which
   step) or only at the final decode. Build via the existing Swift toolchain; mirror any
   `draw-things-community/` change into `DRAW_THINGS_PATCH/`. This pinpoints whether it's the DiT
   forward (LoRA math) or the VAE, and at which timestep.

4. **q4p requantization sanity (lower priority).** If q6p also fails, re-examine whether the q4p/
   q6p quantization of the *attention* weights is well-conditioned for LoRA addition (LoRA delta
   is added to a quantized base at runtime). Consider q8p as an intermediate.

### 25.4 Key files / anchors for the next session

- `dt-models/custom.json` (alias `10_e_v1_4`, line ~180) — companions; repoint to q6p for step 1.
- `dt-models/custom_lora.json` — LoRA registry (schema confirmed correct: `name/file/prefix/
  version:"ltx2.3"/is_lo_ha:false`).
- `dt-models/10_e_v1_4_q4p.ckpt` (coherent base, NaN w/ LoRA) · `..._q6p.ckpt` (untested w/ LoRA)
  · `..._f16.ckpt` (VRAM-blocked).
- `dt-models/ltx-2.3-22b-distilled-lora-fro90_ceil72.safetensors` (raw) ·
  `ltx_2.3_22b_distilled_lora_fro90_ceil72_lora_f16.ckpt` (converted, audited correct).
- `tools/dt_test.sh` — entry point (`--lora NAME[:WEIGHT[:MODE]]`, `--frames`, `--timeout-sec`).
- `draw-things-community/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:4471` —
  `guard !isNaN(...)` → nil (the observed failure point).
- `draw-things-community/Libraries/SwiftDiffusion/Sources/FirstStage.swift:1018-1034` — f32 VAE
  fallback (active, still NaNs → NaN is in the latent).
- `draw-things-community/Libraries/SwiftDiffusion/Sources/Models/LTX2.swift` —
  `interleaved: true` at 162/297/413/823/1324/1468/1588/2012 (shared base+LoRA mapper).
- `draw-things-community/Libraries/SwiftDiffusion/Sources/Extensions/TensorDescriptor.swift:54-111`
  — §17 de-interleave in `internalWrite` `case .O:` (applies to LoRA up via `format:.O`).
- Runs: `output/q6p_canary_10_e_v1_4_test_20260716_130421/` (q4p+LoRA@0.6, NaN) ·
  `..._114316/` (q4p+LoRA@1.0, NaN) · `..._111743/` (base q4p, COHERENT std=0.533).

### 25.5 What is DEFINITIVELY ruled out (don't re-litigate)

- ❌ §17 de-interleave mis-applied to the LoRA (proven applied correctly, §23/§24).
- ❌ Any LoRA-conversion bug — up/down/rank/permutation all correct at every projection (§24).
- ❌ LoRA alpha/scale mishandling (no alpha tensors exist, §24.2.4).
- ❌ Weight-1.0 overdrive as sole cause (0.6 still NaN, §23.3).
- ❌ Missing `.ltx2_3` LoRA branch / structural code gap (prior subagent static analysis).
- ❌ VAE-decode arithmetic (f32 fallback active and still NaNs → NaN is upstream, in the latent).

---

## 26) Distilled LoRA schedule alignment — BREAKTHROUGH (2026-07-21)

### 26.1 What changed

User clarified that `ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe` is a **distillation /
acceleration LoRA** intended to shift the base model toward a distilled (TCD-like) sampling
regime. Prior failing runs used the existing canary defaults (`sampler=17`, `steps=8`), which are
not a distilled schedule.

### 26.2 Control re-validation after restoring official q6p file

`dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt` had temporarily become `0` bytes, which invalidated
earlier official-control attempts (`Illegal instruction` in `TextEncoder.encodeLTX2`). After user
restored the file (valid SQLite header, non-zero size), controls became meaningful again.

- `official_q6p_via_custom` **without LoRA**: PASS/COHERENT.
  - Run: `output/q6p_canary_official_q6p_via_custom_test_20260720_133252/`
  - Final: `image_r0014_01.bin`, std≈0.6909, NaN=0.

- Same alias + iOS-converted distilled LoRA @1.0, default sampler path (`sampler=17`, `steps=8`):
  FAIL (no final image payload).
  - Run: `output/q6p_canary_official_q6p_via_custom_test_20260720_134524/`

This proves the base path is healthy and the failure is schedule-sensitive with LoRA enabled.

### 26.3 Signature of the failing schedule (sampler 17 path)

Across multiple failing LoRA runs (`official` and `custom` q6p/q4p) the preview signature was
identical:

- `preview_0004..0010`: `NaN=18560`, finite std≈`0.965333`.
- No final `image_r*.bin` payload.

This includes mode split tests:
- `mode=all`, `mode=base`, `mode=refiner` all fail under sampler 17 schedule.
- Lowering LoRA weight to `0.1` still fails with the same signature.

Interpretation: this is not merely weight magnitude; the default scheduler/step regime is the wrong
operating point for this distilled LoRA.

### 26.4 Distilled-aligned run — SUCCESS

Using a distilled-style schedule on the same official q6p base + same iOS LoRA:

- sampler: `19` (`TCDTrailing`)
- steps: `2`
- frames: default canary stream path (`num_frames` from harness)
- LoRA: `ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe_lora_f16.ckpt:1.0`

Result: **PASS**, full payloads returned.

- Run: `output/q6p_canary_official_q6p_tcdtrailing_s2_lora52_20260721_074416/`
- Stream summary: images written `9`, audio `1`, generation stream finished.
- Preview: `preview_0004.bin` NaN=0, std≈1.1837.
- Final: `image_r0016_01.bin` NaN=0, std≈0.6822.

### 26.5 q4p + distilled LoRA under distilled schedule — SUCCESS

Added a dedicated q4p alias with correct companions:

- `10_e_v1_4_q4p_alias` in `dt-models/custom.json`
  - file/clip_encoder: `10_e_v1_4_q4p.ckpt`
  - autoencoder/text-encoder/objective same as `10_e_v1_4`.

Run with the same distilled schedule:

- model: `10_e_v1_4_q4p_alias`
- sampler: `19` (`TCDTrailing`)
- steps: `2`
- LoRA: `...ceil52_condsafe...:1.0`

Result: **PASS**, full payloads returned.

- Run: `output/q6p_canary_q4p_tcdtrailing_s2_lora52_20260721_080815/`
- Stream summary: images written `9`, audio `1`.
- Final: `image_r0016_01.bin` NaN=0, std≈0.6353.
- PNG saved: `image_r0016_01.png`.

### 26.6 Net conclusion

For this LoRA, the prior failures were primarily **schedule mismatch** (using default sampler 17 /
higher-step path with a distilled acceleration LoRA), not file corruption or converter defects.

Working recipe established:

- Use `TCDTrailing` (`sampler=19`) + low steps (`2` to start) for
  `ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe`.

### 26.7 Tooling note

`tools/run_q6p_canary_once.sh` had an internal `SAMPLER` variable wired into config generation but
no CLI flag parsing for `--sampler`. Added:

- `--sampler <n>` to usage,
- argument parsing,
- validation,
- and run-time logging (`sampler=<n>`).

This enables reproducible distilled-schedule canary runs from the command line.

### 26.8 Convenience preset for repeatable validation

Added `tools/dt_test_distilled_lora.sh` to run the now-proven safe preset in one
command:

```bash
bash tools/dt_test_distilled_lora.sh
```

Defaults:
- model: `official_q6p_via_custom`
- lora: `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0`
- steps: `2`
- sampler: `19` (`TCDTrailing`)
- frames: `5`
- timeout: `2700`

It also accepts optional overrides:

```bash
bash tools/dt_test_distilled_lora.sh 10_e_v1_4_q4p_alias \
  ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0
```

Additionally, `tools/dt_test.sh` now accepts `--sampler <n>` and forwards it to
`run_q6p_canary_once.sh` so scheduler selection is explicit at the top-level test
entrypoint.

---

## 27) Local LoRA conversion parity vs iOS conversion — PROVEN (2026-07-21)

### 27.1 Objective

Now that the downloaded iOS-converted LoRA (`...ceil52_condsafe_lora_f16.ckpt`) is known-good,
verify whether our local conversion pipeline produces an equivalent ckpt and equivalent runtime
behavior.

### 27.2 Conversion performed (local tooling)

Command:

```bash
bash tools/dt_lora_convert.sh \
  dt-models/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe.safetensors \
  --name ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe_ourconvert \
  --version ltx2.3 \
  --force
```

Output:

- `dt-models/ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe_ourconvert_lora_f16.ckpt`

Registration (local `custom_lora.json`):

- `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe_ourconvert`

### 27.3 Static parity check (iOS ckpt vs our ckpt)

Compared both ckpts by reading the SQLite `tensors` table and checking:

- tensor names
- datatype / format / dims
- raw float16 payload values

Result:

- tensor count: `3368` vs `3368`
- missing keys both directions: `0`
- metadata mismatches: `0`
- value mismatches: `0`
- **exact-equal tensors: `3368/3368`**

This is full byte-level parity for all stored tensor payloads.

### 27.4 Runtime parity check under distilled-safe schedule

Used the proven distilled recipe (`sampler=19` TCDTrailing, `steps=2`) on
`official_q6p_via_custom` and compared iOS-converted vs locally-converted LoRA in the same
harness (`run_q6p_canary_once.sh`, same prompt/seed/resolution).

iOS-converted LoRA recheck:

- Run: `output/q6p_canary_official_q6p_tcdtrailing_s2_ios52_recheck_20260721_083617/`
- RESULT: PASS (images `9`, audio `1`)

Locally-converted LoRA recheck:

- Run: `output/q6p_canary_official_q6p_tcdtrailing_s2_our52_recheck_20260721_084644/`
- RESULT: PASS (images `9`, audio `1`)

Final-frame comparison:

- compared `image_r0016_01.bin` payloads
- **exact-equal bytes**
- max abs diff: `0.0`

### 27.5 Conclusion

For `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe`, our local converter output is
equivalent to the iOS-converted ckpt both statically and at runtime when tested under the correct
distilled schedule.

Therefore, local LoRA conversion is validated for this case.

---

## 28) LoRA quantization — WORKING (q8p / q6p / q4p) (2026-07-21)

### 28.1 Objective

Now that local LoRA conversion is validated (§27), extend the pipeline to **quantize** the LoRA
ckpt (f16 → q8p/q6p/q4p) to shrink it, and verify each still renders coherently.

### 28.2 Revisiting the "LoRA specifics" from the base-model work

During base-model quantization (§17–§22) the sensitive tensors (Q/K de-interleave, ada_ln
norms, QK-norms) needed converter fixes and post-quant F32 restores. The open question was
whether LoRAs have analogous specifics.

Two things were checked here:

1. **Runtime LoRA read codecs.** `LoRALoader.swift` reads every LoRA tensor with
   `codec: [.q6p, .q8p, .ezm7, .externalData]` at all 25 read sites (no `q4p`/`q5p`/`q7p`
   literally listed). This initially looked like a hard "no q4p for LoRA" limit.
   **Empirically disproven** (§28.4): s4nnc dequantizes the *self-describing* palettized blob
   regardless of that hint list, so q4p LoRA loads fine. The codec list is not a hard filter for
   dequantizing reads.

2. **Sensitive-tensor handling by the quantizer.** The existing `model-quantizer -m ltx2.3`
   forced path already routes the LoRA's sensitive tensors to safe precision **by key
   substring**, and the LoRA's abbreviated key names line up well enough:
   - `*embedder*`, `*proj_out*` up+down → **fp16** (40 tensors preserved verbatim).
   - rank-1 projections + all `adaln_single`/scalar rows → **ezm7** (near-lossless; 1984 tensors).
     (Confirmed: all 64 LoRA `adaln_single` tensors are rank-1 `[1,N]`/`[N]` → ezm7, none quantized.)
   - rank>1 attention/FFN deltas → **[codec, ezm7]** (1344 tensors).
   So **no separate F32 norm-restore is needed for LoRAs** (unlike the base model).

### 28.3 Quantization performed

Tool: existing `tools/dt_quantize_model.sh` → `model-quantizer -m ltx2.3 --target-codec <c>`
with `--ltx-trace-output` for the per-tensor decision log. Source:
`ltx_2.3_22b_distilled_lora_1.1_fro90_ceil52_condsafe_ourconvert_lora_f16.ckpt` (§27).

| variant | file size | bytes/elem (rank-52 `a_q` up) | decision histogram |
|---------|-----------|-------------------------------|--------------------|
| f16     | 489 MB    | 2.00 (fp16)                   | —                  |
| q8p     | 256 MB    | 1.03                          | ezm7×1984, fp16×40, [q8p,ezm7]×1344 |
| q6p     | 196 MB    | 0.78                          | ezm7×1984, fp16×40, [q6p,ezm7]×1344 |
| q4p     | 144 MB    | 0.56                          | ezm7×1984, fp16×40, [q4p,ezm7]×1344 |

Note: the SQLite `datatype` column reads `131072` (FP16 logical type) for all tensors in every
variant; the actual palettization is visible only via the stored blob bytes/element (above), not
that column.

### 28.4 Runtime validation (official q6p base, distilled schedule, sampler 19 / steps 2)

All three quantized LoRAs **PASS** (9 images + audio, no NaN, coherent), compared against the
f16 LoRA reference frame (`output/q6p_canary_official_q6p_tcdtrailing_s2_our52_recheck_20260721_084644/`):

| variant | RESULT | final std | mean abs diff vs f16 |
|---------|--------|-----------|----------------------|
| f16     | PASS   | 0.6822    | 0.000000 (reference) |
| q8p     | PASS   | 0.6801    | 0.021773             |
| q6p     | PASS   | 0.6806    | 0.033328             |
| q4p     | PASS   | 0.6810    | 0.051412             |

The drift is smooth and monotonic (q8p < q6p < q4p), the signature of progressive quantization
loss — **not** of dropped/unreadable tensors. This confirms all quantized LoRA tensors are
genuinely applied at runtime, including q4p.

Runs:
- q8p: `output/q6p_canary_lora_q8p_test_20260721_092034/`
- q6p: `output/q6p_canary_lora_q6p_test_20260721_093453/`
- q4p: `output/q6p_canary_lora_q4p_test_20260721_095054/`

### 28.5 Conclusion + recommendation

LoRA quantization works with the existing `model-quantizer` and needs **no** converter change and
**no** post-quant norm restore. Recommended default: **q8p** (best quality/size, ~0.022 drift,
~2× smaller). q6p is a good balance (~2.5× smaller). q4p is usable (~3.4× smaller) with more drift.

### 28.6 Tooling added

`tools/dt_lora_quantize.sh` — one-command LoRA quantize + register:

```bash
bash tools/dt_lora_quantize.sh <lora_f16.ckpt|NAME> --codec q8p   # q8p (default) | q6p | q4p
# then:
bash tools/dt_test_distilled_lora.sh official_q6p_via_custom <registered_name>:1.0
```

It wraps `dt_quantize_model.sh` with `-m ltx2.3`, opts into q4p for LoRAs (safe here; the base-
model q4p gate does not apply), registers the output in `dt-models/custom_lora.json`, and skips
the base-model-only F32 norm restores.
