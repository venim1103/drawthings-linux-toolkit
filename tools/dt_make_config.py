#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import flatbuffers

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PY_BINDINGS = WORKSPACE_ROOT / ".cache/dt_flatbuf_py"


def _parse_bool(v: str) -> bool:
    x = v.strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {v}")


def _load_generated_modules(bindings_dir: Path):
    if not bindings_dir.exists():
        raise FileNotFoundError(
            f"FlatBuffer python bindings directory not found: {bindings_dir}\n"
            "Generate it with:\n"
            "  flatc --python -o .cache/dt_flatbuf_py .cache/config_for_flatc.fbs"
        )

    sys.path.insert(0, str(bindings_dir))
    import GenerationConfiguration as GC  # type: ignore

    return GC


def build_config(args) -> bytes:
    GC = _load_generated_modules(Path(args.bindings_dir))

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    start_width = max(1, args.width // 64)
    start_height = max(1, args.height // 64)

    builder = flatbuffers.Builder(4096)

    model = builder.CreateString(args.model)
    upscaler = builder.CreateString(args.upscaler)
    face_restoration = builder.CreateString(args.face_restoration)
    refiner_model = builder.CreateString(args.refiner_model)
    name = builder.CreateString(args.name)
    clip_l_text = builder.CreateString(args.clip_l_text)
    open_clip_g_text = builder.CreateString(args.open_clip_g_text)
    t5_text = builder.CreateString(args.t5_text)

    GC.GenerationConfigurationStart(builder)
    GC.GenerationConfigurationAddId(builder, 0)
    GC.GenerationConfigurationAddStartWidth(builder, start_width)
    GC.GenerationConfigurationAddStartHeight(builder, start_height)
    GC.GenerationConfigurationAddSeed(builder, seed)
    GC.GenerationConfigurationAddSteps(builder, args.steps)
    GC.GenerationConfigurationAddGuidanceScale(builder, args.guidance_scale)
    GC.GenerationConfigurationAddStrength(builder, args.strength)
    GC.GenerationConfigurationAddModel(builder, model)
    GC.GenerationConfigurationAddSampler(builder, args.sampler)
    GC.GenerationConfigurationAddBatchCount(builder, args.batch_count)
    GC.GenerationConfigurationAddBatchSize(builder, args.batch_size)
    GC.GenerationConfigurationAddHiresFix(builder, args.hires_fix)
    GC.GenerationConfigurationAddUpscaler(builder, upscaler)
    GC.GenerationConfigurationAddSeedMode(builder, args.seed_mode)
    GC.GenerationConfigurationAddClipSkip(builder, args.clip_skip)
    GC.GenerationConfigurationAddMaskBlur(builder, args.mask_blur)
    GC.GenerationConfigurationAddFaceRestoration(builder, face_restoration)
    GC.GenerationConfigurationAddRefinerModel(builder, refiner_model)
    GC.GenerationConfigurationAddTargetImageHeight(builder, args.height)
    GC.GenerationConfigurationAddTargetImageWidth(builder, args.width)
    GC.GenerationConfigurationAddName(builder, name)
    GC.GenerationConfigurationAddFpsId(builder, args.fps_id)
    GC.GenerationConfigurationAddMotionBucketId(builder, args.motion_bucket_id)
    GC.GenerationConfigurationAddCondAug(builder, args.cond_aug)
    GC.GenerationConfigurationAddStartFrameCfg(builder, args.start_frame_cfg)
    GC.GenerationConfigurationAddNumFrames(builder, args.num_frames)
    GC.GenerationConfigurationAddMaskBlurOutset(builder, args.mask_blur_outset)
    GC.GenerationConfigurationAddSharpness(builder, args.sharpness)
    GC.GenerationConfigurationAddShift(builder, args.shift)
    GC.GenerationConfigurationAddTiledDecoding(builder, args.tiled_decoding)
    GC.GenerationConfigurationAddTiledDiffusion(builder, args.tiled_diffusion)
    GC.GenerationConfigurationAddPreserveOriginalAfterInpaint(
        builder, args.preserve_original_after_inpaint
    )
    GC.GenerationConfigurationAddUpscalerScaleFactor(builder, args.upscaler_scale_factor)
    GC.GenerationConfigurationAddSeparateClipL(builder, args.separate_clip_l)
    GC.GenerationConfigurationAddClipLText(builder, clip_l_text)
    GC.GenerationConfigurationAddSeparateOpenClipG(builder, args.separate_open_clip_g)
    GC.GenerationConfigurationAddOpenClipGText(builder, open_clip_g_text)
    GC.GenerationConfigurationAddResolutionDependentShift(
        builder, args.resolution_dependent_shift
    )
    GC.GenerationConfigurationAddTeaCache(builder, args.tea_cache)
    GC.GenerationConfigurationAddTeaCacheThreshold(builder, args.tea_cache_threshold)
    GC.GenerationConfigurationAddTeaCacheStart(builder, args.tea_cache_start)
    GC.GenerationConfigurationAddTeaCacheEnd(builder, args.tea_cache_end)
    GC.GenerationConfigurationAddTeaCacheMaxSkipSteps(builder, args.tea_cache_max_skip_steps)
    GC.GenerationConfigurationAddCausalInference(builder, args.causal_inference)
    GC.GenerationConfigurationAddSeparateT5(builder, args.separate_t5)
    GC.GenerationConfigurationAddT5Text(builder, t5_text)

    config = GC.GenerationConfigurationEnd(builder)
    builder.Finish(config)
    return bytes(builder.Output()), seed, start_width, start_height


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Draw Things GenerationConfiguration flatbuffer")
    p.add_argument("--bindings-dir", default=str(DEFAULT_PY_BINDINGS))
    p.add_argument("--out", required=True, help="Output .bin path")

    p.add_argument("--model", required=True, help="Model file name, e.g. z_image_turbo_1.0_q6p.ckpt")
    p.add_argument("--name", default="cli-config")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=-1, help="-1 for random")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--sampler", type=int, default=17, help="SamplerType enum value")
    p.add_argument("--seed-mode", type=int, default=2, help="SeedMode enum value")

    p.add_argument("--batch-count", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--clip-skip", type=int, default=1)

    p.add_argument("--hires-fix", type=_parse_bool, default=False)
    p.add_argument("--upscaler", default="")
    p.add_argument("--upscaler-scale-factor", type=int, default=0)

    p.add_argument("--mask-blur", type=float, default=1.5)
    p.add_argument("--mask-blur-outset", type=int, default=0)
    p.add_argument("--sharpness", type=float, default=0.0)

    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--resolution-dependent-shift", type=_parse_bool, default=False)

    p.add_argument("--tiled-decoding", type=_parse_bool, default=False)
    p.add_argument("--tiled-diffusion", type=_parse_bool, default=False)
    p.add_argument("--preserve-original-after-inpaint", type=_parse_bool, default=True)

    p.add_argument("--face-restoration", default="")
    p.add_argument("--refiner-model", default="")

    p.add_argument("--separate-clip-l", type=_parse_bool, default=False)
    p.add_argument("--clip-l-text", default="")
    p.add_argument("--separate-open-clip-g", type=_parse_bool, default=False)
    p.add_argument("--open-clip-g-text", default="")

    p.add_argument("--separate-t5", type=_parse_bool, default=False)
    p.add_argument("--t5-text", default="")

    p.add_argument("--tea-cache", type=_parse_bool, default=False)
    p.add_argument("--tea-cache-threshold", type=float, default=0.2)
    p.add_argument("--tea-cache-start", type=int, default=5)
    p.add_argument("--tea-cache-end", type=int, default=-1)
    p.add_argument("--tea-cache-max-skip-steps", type=int, default=3)

    p.add_argument("--causal-inference", type=int, default=0)

    p.add_argument("--fps-id", type=int, default=5)
    p.add_argument("--motion-bucket-id", type=int, default=127)
    p.add_argument("--cond-aug", type=float, default=0.02)
    p.add_argument("--start-frame-cfg", type=float, default=1.0)
    p.add_argument("--num-frames", type=int, default=14)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    blob, seed, start_w, start_h = build_config(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    print(f"wrote config bytes: {out_path}")
    print(f"size bytes: {len(blob)}")
    print(f"model: {args.model}")
    print(f"seed: {seed}")
    print(f"start_width/start_height (64x blocks): {start_w}/{start_h}")
    print(f"pixel size: {start_w * 64}x{start_h * 64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
