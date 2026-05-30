# Conversion Findings Report (2026-05-28)

## Scope

This report captures the full investigation and patch work for converting custom LTX-2.3 model weights into Draw Things checkpoints, including structural fixes in the importer, quantization outcomes, and runtime comparison against the official LTX-2.3 q6p baseline.

## Current Status (Latest Verified)

- Converter-side structural defects for this custom LTX-2.3 source were fixed.
- The fixed F16 and derived q6p checkpoints now match the official LTX-2.3 q6p keyset exactly (`5746` tensors, no missing/extra keys).
- Controlled one-response probes show official q6/f16 controls reach `textEncoded`, while current custom F16/q6 checkpoints reproduce runtime failure (`ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`, server exit 139).
- Runtime with custom q4p (quantized from fixed F16) caused Draw Things server crashes (`Bad pointer dereference` in ccv read path), so q4p is currently unsafe for this model.
- Official exact F16 baseline file (`ltx_2.3_22b_distilled_f16.ckpt`) was downloaded and checksum-verified against the published value in-session.
- Custom model metadata now points to q6p in `dt-models/custom.json`.

## Executive Summary

- Initial failures were caused by importer mapping gaps (missing embedder/connector/feature tensors and placeholder keys).
- After importer patching, converted outputs passed strict structural validation and reached key parity with official q6p.
- Runtime behavior now diverges by artifact provenance, not by keyset parity:
   - official q6/f16 controls: stable to first signpost in one-response probes.
   - custom q4p/q6/f16: current runtime reader path crashes (`ccv_nnc_tensor_read` / `ccv_cnnp_model_read`).
- Remaining blocker is serialization compatibility (type/format/datatype policy), not missing key families.

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
- Structural key parity is necessary but not sufficient for runtime compatibility.

## Detailed Metadata / Serialization Diff (2026-05-29)

### Reproduction update (post-structural-fix)

Control (official q6) with one-response probe:

- Reached first signpost (`textEncoded`) and remained stable.

Custom probes:

- `10_e_v1_bf16_q6p.ckpt`: server crashed in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.
- `10_e_v1_bf16_fix2_f16.ckpt`: same crash path.

Implication:

- The crash is not q6-only; current custom F16 and derived q6 are both incompatible with the active runtime reader path.

### Controlled A/B update with official exact F16 (2026-05-30)

Probe setup (single-frame, low-cost):

- width/height: `384x704`
- steps: `8`
- sampler: `19`
- shift: `5.0`
- `--max-responses 1`

Official exact F16 control (`ltx_2.3_22b_distilled_f16.ckpt`):

- Probe returned `response #1` with signpost `textEncoded`.
- Client exit: `PROBE_OFFICIAL_F16_EXIT=0`.

Custom fixed F16 under same probe style (`10_e_v1_bf16_fix2_f16.ckpt`):

- Request started, then client failed with `gRPC error: UNAVAILABLE: Socket closed`.
- Client exit: `PROBE_CUSTOM_F16_AGAIN_EXIT=1`.
- Server crashed with `Bad pointer dereference` and backtrace rooted at:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
   - process exit: `139` (`Segmentation fault`).

Guidance confound check (custom F16, guidance forced to `1.0`):

- Re-ran one-response probe with guidance matching official control.
- Client again failed with `gRPC error: UNAVAILABLE: Socket closed`.
- Client exit: `PROBE_CUSTOM_F16_G1_EXIT=1`.
- Server again crashed in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.

Interpretation:

- This session's official-vs-custom exact-F16 A/B confirms runtime instability is specific to custom serialization, not just q6 quantization or guidance-scale mismatch.

### Metadata diff evidence (no regeneration)

Report artifacts:

- `output/probe_metadata_diff_20260529.json`
- `output/probe_metadata_diff_20260529.md`

Key counts from `custom_q6` vs `official_q6`:

- `format_diff_cq6_vs_oq6 = 5743`
- `type_diff_cq6_vs_oq6 = 1837`
- `datatype_diff_cq6_vs_oq6 = 1545`
- `large_data_len_delta_cq6_vs_oq6 = 1689`
- `quantized_only_in_custom_q6_vs_cf16 = 260`

Key counts from `custom_f16` vs `official_q6`:

- `format_diff_cf16_vs_oq6 = 5743`
- `type_diff_cf16_vs_oq6 = 3904`
- `datatype_diff_cf16_vs_oq6 = 1545`

Top mismatch families:

- `__dit__`
- `__text_video_connector__`
- `__text_audio_connector__`
- `__text_feature_extractor__`

Representative mismatches:

- Many custom tensors are stored with `format=1` where official q6 uses `format=2`.
- Many custom tensors keep `datatype=131072` where official q6 uses `datatype=16384` or `datatype=524288`.
- `custom_q6` quantizes 260 tensors that official q6 effectively keeps at F16-style metadata behavior (notably connector `down_proj` blocks and `proj_out` blocks).

### Toolchain compatibility note

Attempt to run a source-matched local `gRPCServerCLI` build was blocked in this Linux container:

- Swift build for `gRPCServerCLI` failed with `no such module 'CoreML'` while resolving `LocalImageGenerator` dependencies.

This prevented direct same-source runtime validation in the current environment.

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
2. Current runtime still crashes when loading custom checkpoints (both fixed F16 and derived q6) in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.
3. The blocker is now metadata/serialization compatibility (type/format/datatype policy divergence), not missing tensor key families.
4. q4p remains unstable and should be treated as non-usable.
5. No new long quantization/conversion run should be started until serialization compatibility checks are enforced.

## Immediate Operational Guidance

- Keep using official `ltx_2.3_22b_distilled_1.1_q6p.ckpt` as the runtime baseline for day-to-day generation.
- Use official `ltx_2.3_22b_distilled_f16.ckpt` as a control when validating loader stability/signpost progression.
- Treat custom `10_e_v1_bf16_fix2_f16.ckpt`, `10_e_v1_bf16_q6p.ckpt`, and `10_e_v1_bf16_q4p.ckpt` as non-runnable in the current runtime environment.
- Before any new 4+ hour regeneration, require metadata diff checks against the official q6 baseline (see `output/probe_metadata_diff_20260529.*`).
- Next fix target is serialization policy alignment (type/format/datatype decisions), not additional key-name mapping coverage.
