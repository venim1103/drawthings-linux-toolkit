# Custom Model Pipeline — Quickstart

Convert, quantize, and test any LTX2.3 custom model with three commands. All the
converter fixes are baked into the output files, so the quantized model is
self-contained and portable (including to Draw Things on mobile).

## The three commands

```bash
# 1) Convert safetensors -> f16 checkpoint (CPU, ~15 min)
bash tools/dt_convert.sh dt-models/my_model.safetensors

# 2) Quantize f16 -> q6p + restore norms + register alias (CPU, ~1-2 h)
bash tools/dt_quantize.sh my_model            # default codec: q6p
#   bash tools/dt_quantize.sh my_model --codec q8p
#   bash tools/dt_quantize.sh my_model --codec q4p

# 3) Render a test frame and decode it to PNG (GPU)
bash tools/dt_test.sh my_model
```

The name (`my_model`) is derived from the safetensors filename in step 1 and
reused in steps 2 and 3. Override it with `dt_convert.sh ... --name my_model`.

Every script has `--help`.

## What each step does

| Step | Script | Output | Notes |
|------|--------|--------|-------|
| 1 | `dt_convert.sh` | `dt-models/<name>_f16.ckpt` | Preflight → convert (fixed rotary + QK-norm de-interleave) → validate |
| 2 | `dt_quantize.sh` | `dt-models/<name>_<codec>.ckpt` + `custom.json` alias | Quantize → restore F32 norms → restore F32 QK-norms (`[1,1,N]` dim fix) → alias |
| 3 | `dt_test.sh` | `output/q6p_canary_<name>_.../*.png` | GPU render + coherence check (std/NaN) + PNG decode |

Skipping the norm restores in step 2 makes the model diverge — they are the
step that turns a structured-but-wrong image into a coherent one. Use
`--no-restore` only for debugging comparisons.

## Codecs

| Codec | Size (22B model) | Use |
|-------|------------------|-----|
| `q8p` | largest  | Highest quality |
| `q6p` | ~20 GB (default) | Balanced; matches official Draw Things q6p format |
| `q4p` | smallest | Lowest precision (auto-enabled for ltx2.3 by the wrapper) |

Norm restores are codec-independent (norms are always stored F32), so all three
codecs go through the identical restore path.

## Portability to mobile Draw Things

The quantized `.ckpt` is a single self-contained file — the §17/§18 weight
fixes, the §22 `[1,1,N]` dims, and the F32 norm precision are all written into
it. It matches the official Draw Things q6p format, so you import the
**pre-quantized** file directly on the device (do not re-convert on the phone).

Copy alongside the model:

- `dt-models/ltx_2.3_audio_video_vae_f16.ckpt` (autoencoder, ~1.7 GB)
- `dt-models/gemma_3_12b_it_qat_q8p.ckpt` **and** its `-tensordata` sidecar (~13 GB)

Then configure it on the device as an ltx2.3 model with those two companions and
`modifier=kontext` (the same fields `dt_quantize.sh` writes into `custom.json`).

## Notes

- **Convert and quantize are CPU-only** — safe to run on battery.
- **`dt_test.sh` uses the GPU.** On a laptop, run it on AC power. GPU inference on
  battery has triggered a Windows/WSL `VIDEO_MEMORY_MANAGEMENT` BSOD.
- The norm restore is **reference-free** — no official checkpoint needed. The
  ada_ln F32 set is reconstructed from the model itself and reshaped to `[1,1,N]`.
  Optionally pass `--ref-q6p PATH` to use an official q6p as the F32 template.
- If a quantize finished but you want to re-run the norm restores (or supply a
  reference), use `dt_quantize.sh <NAME> --restore-only` — it skips the ~1–2 h
  quantize and just fixes the existing `.ckpt`.
