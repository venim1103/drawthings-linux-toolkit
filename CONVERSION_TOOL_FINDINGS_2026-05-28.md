# Conversion Findings Report (2026-05-28)

## Scope

This report captures the full investigation and patch work for converting custom LTX-2.3 model weights into Draw Things checkpoints, including structural fixes in the importer, quantization outcomes, and runtime comparison against the official LTX-2.3 q6p baseline.

## Current Status (Latest Verified)

- Converter-side structural defects for this custom LTX-2.3 source were fixed.
- The fixed F16 and derived q6p checkpoints now match the official LTX-2.3 q6p keyset exactly (`5746` tensors, no missing/extra keys).
- Runtime with custom q6p now returns final image payloads (not preview-only), but image quality remains noisy/unstable versus official q6p.
- Runtime with custom q4p (quantized from fixed F16) caused Draw Things server crashes (`Bad pointer dereference` in ccv read path), so q4p is currently unsafe for this model.
- Custom model metadata now points to q6p in `dt-models/custom.json`.

## Executive Summary

- Initial failures were caused by importer mapping gaps (missing embedder/connector/feature tensors and placeholder keys).
- After importer patching, converted outputs passed strict structural validation and reached key parity with official q6p.
- Quantization behavior diverged by codec:
  - q4p: server crash on inference start (`UNAVAILABLE: Socket closed` on client).
  - q6p: inference runs and emits final image payloads.
- Remaining issue is runtime quality mismatch (custom q6p output noisy vs official q6p), not missing key families.

## Applied Patch Summary

### 1) Importer Mapping Fixes (Community + Toolkit Mirror)

Files:

- `draw-things-community/Libraries/ModelOp/Sources/ModelImporter.swift`
- `DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift`

Key changes:

1. Expanded LTX diffusers detection and key lookup fallback to include `diffusion_model.*` prefixed layouts.
2. Added normalization for `diffusion_model.` prefix stripping in LTX branches.
3. Added explicit LTX2.3 connector mapping builder (`makeLTX23ConnectorMapping`) for deterministic source-to-target mapping.
4. Added explicit alias mappings for:
   - `patchify_proj.* -> __dit__[t-x_embedder-*]`
   - `audio_patchify_proj.* -> __dit__[t-a_embedder-*]`
5. Added explicit text feature extractor aggregate aliases:
   - `text_embedding_projection.video_aggregate_embed.{weight,bias}`
   - `text_embedding_projection.audio_aggregate_embed.{weight,bias}`
6. Skipped empty destination names during mapping writes to prevent placeholder keys.
7. Relaxed additional mapping consumption guard to ensure aliases are applied when source keys exist.

### 2) Conversion Wrapper Validation

Files:

- `tools/dt_convert_model.sh`
- `tools/dt_validate_converted_ckpt.py`

Key changes:

1. Added automatic post-conversion validation controls (`DRAWTHINGS_CONVERTER_VALIDATE*`).
2. Added profile-aware structural checks for LTX2.3:
   - required prefix families,
   - placeholder key rejection,
   - source safetensors-assisted profile detection.

### 3) Runtime Wrapper Guardrails

File:

- `tools/dt_generate_video.sh`

Key changes:

1. Added LTX-2.3 upscaler presence checks.
2. Avoided implicit hires-fix enable when required upscalers are absent.
3. Added warnings for manually enabled hires-fix without required upscalers.

### 4) Model Metadata Update

File:

- `dt-models/custom.json`

Key changes:

1. Main custom entry switched from q4p to q6p:
   - `file: 10_e_v1_bf16_q6p.ckpt`
   - `clip_encoder: 10_e_v1_bf16_q6p.ckpt`

## Reproduction Results

### A) Historical pre-fix failure (custom q4p)

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q4p.ckpt
```

Output folder:

- `output/dt_video_20260528_095514`

Observed stream summary:

- `responses: 11`
- `images written: 0`
- `audio written: 0`
- `preview frames seen: 5`
- stream ended without final `generatedImages` payloads

### B) Post-fix structural pass (custom fix2 f16)

Command:

```bash
python tools/dt_validate_converted_ckpt.py \
  --file dt-models/10_e_v1_bf16_fix2_f16.ckpt \
  --source-safetensors dt-models/10_e_v1_bf16.safetensors
```

Observed:

- `tensor_count=5746`
- `RESULT=PASS`

### C) q4p from fixed F16 (post-fix quantized) crash

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q4p.ckpt
```

Observed:

- gRPC server crash with segmentation fault:
  - `Bad pointer dereference`
  - stack includes `ccv_nnc_tensor_read`, `ccv_cnnp_model_read`
- client error: `UNAVAILABLE: Socket closed`

### D) q6p from fixed F16 (post-fix quantized) runtime

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q6p.ckpt
```

Output folder:

- `output/dt_video_20260528_171805`

Observed stream summary:

- `responses: 14`
- `images written: 1`
- `audio written: 0`
- `preview frames seen: 5`

Result:

- final image payload returned, but output remains visually noisy.

### E) Official LTX-2.3 q6p control

Command:

```bash
./tools/dt_video_test_run.sh ltx_2.3_22b_distilled_1.1_q6p.ckpt
```

Output folder:

- `output/dt_video_20260528_171821`

Observed stream summary:

- `responses: 15`
- `images written: 1`
- `audio written: 1`
- `preview frames seen: 5`

Result:

- coherent final image and audio payloads returned.

## Structural Comparison (Final State)

Compared files:

- `dt-models/10_e_v1_bf16_q6p.ckpt` (custom, post-fix)
- `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt` (official)

Counts:

- `custom_count = 5746`
- `official_count = 5746`
- `official_minus_custom = 0`
- `custom_minus_official = 0`

Interpretation:

- Structural/key-family compatibility has been achieved.
- Remaining discrepancy is behavioral/runtime quality, not missing tensors.

## Runtime Numeric Snapshot (Image Tensor Distribution)

Custom q6 image tensor (`output/dt_video_20260528_171805/image_r0014_01.bin`):

- min/max: `-3.275391 / 4.406250`
- p1/p50/p99: `-1.636719 / 0.154053 / 1.933594`

Official q6 image tensor (`output/dt_video_20260528_171821/image_r0014_01.bin`):

- min/max: `-1.233398 / 1.217773`
- p1/p50/p99: `-0.958496 / -0.616211 / 1.006836`

Interpretation:

- Custom output distribution remains wider and inconsistent with official behavior, matching observed noisy decode.

## Conclusion

1. Converter/importer structural issues were real and are now fixed.
2. q6p conversion path is structurally sound and runtime-stable enough to emit final image payloads.
3. q4p is currently unstable for this custom model (server crash).
4. Current blocker is model/runtime quality parity (custom q6p output quality), not converter tensor-family completeness.

## Immediate Operational Guidance

- Use `10_e_v1_bf16_q6p.ckpt` as the current custom test artifact (q4p replaced).
- Keep using official `ltx_2.3_22b_distilled_1.1_q6p.ckpt` as quality baseline.
- Treat custom q4p as non-production for now due to reproducible server crash.
- Next debugging focus should be runtime behavior/calibration differences (sampling/config/model semantics), not key mapping coverage.
