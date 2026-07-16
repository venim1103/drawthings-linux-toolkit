#!/usr/bin/env python3
import argparse
import json
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
    import LoRA  # type: ignore
    import LoRAMode  # type: ignore

    return GC, LoRA, LoRAMode


def _optional_string_offset(builder: flatbuffers.Builder, value: str):
    txt = value.strip()
    if not txt:
        return None
    return builder.CreateString(txt)


def _resolve_model_name_to_file(model: str) -> tuple[str, str | None]:
    """Resolve a custom model display name to its file key when possible.

    GenerationConfiguration.model is expected to be a downloadable file key.
    Custom aliases in dt-models/custom.json are user-friendly names and need
    to be mapped back to their file field.
    """
    query = model.strip()
    if not query:
        return model, None

    custom_json = WORKSPACE_ROOT / "dt-models" / "custom.json"
    if not custom_json.exists():
        return model, None

    try:
        payload = json.loads(custom_json.read_text(encoding="utf-8"))
    except Exception:
        return model, None

    if not isinstance(payload, list):
        return model, None

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name", "")).strip() != query:
            continue

        model_file = str(entry.get("file", "")).strip()
        if model_file:
            return model_file, "custom.json:name"

    return model, None


def _resolve_lora_name_to_file(lora: str) -> tuple[str, str | None]:
    query = lora.strip()
    if not query:
        return lora, None

    custom_lora_json = WORKSPACE_ROOT / "dt-models" / "custom_lora.json"
    if not custom_lora_json.exists():
        return lora, None

    try:
        payload = json.loads(custom_lora_json.read_text(encoding="utf-8"))
    except Exception:
        return lora, None

    if not isinstance(payload, list):
        return lora, None

    for entry in payload:
        if not isinstance(entry, dict):
            continue

        file_name = str(entry.get("file", "")).strip()
        name = str(entry.get("name", "")).strip()
        if query == name and file_name:
            return file_name, "custom_lora.json:name"
        if query == file_name:
            return file_name, None

    return lora, None


def _parse_lora_mode(raw: str, fallback_mode: int) -> int:
    txt = raw.strip().lower()
    if not txt:
        return fallback_mode

    mode_by_name = {
        "all": 0,
        "base": 1,
        "refiner": 2,
    }
    if txt in mode_by_name:
        return mode_by_name[txt]

    try:
        mode = int(txt)
    except ValueError as exc:
        raise ValueError(f"Invalid LoRA mode: {raw}") from exc
    if mode not in (0, 1, 2):
        raise ValueError(f"LoRA mode must be one of 0,1,2 or all/base/refiner, got: {raw}")
    return mode


def _parse_lora_specs(args) -> list[dict]:
    specs: list[dict] = []
    default_mode = _parse_lora_mode(args.lora_mode, 0)

    for raw in args.lora:
        text = raw.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(
                f"Invalid --lora value '{raw}'. Expected FILE[:WEIGHT[:MODE]]"
            )

        lora_name = parts[0].strip()
        if not lora_name:
            raise ValueError(f"Invalid --lora value '{raw}': missing file/name")

        weight = 0.6
        mode = default_mode
        if len(parts) >= 2 and parts[1].strip():
            try:
                weight = float(parts[1].strip())
            except ValueError as exc:
                raise ValueError(f"Invalid LoRA weight in '{raw}'") from exc

        if len(parts) >= 3 and parts[2].strip():
            mode = _parse_lora_mode(parts[2], default_mode)

        resolved_file, resolved_from = _resolve_lora_name_to_file(lora_name)
        specs.append(
            {
                "input": lora_name,
                "file": resolved_file,
                "weight": weight,
                "mode": mode,
                "resolved_from": resolved_from,
            }
        )
    return specs


def build_config(args) -> bytes:
    GC, LoRA, _LoRAMode = _load_generated_modules(Path(args.bindings_dir))

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    start_width = max(1, args.width // 64)
    start_height = max(1, args.height // 64)

    hires_fix_start_width = int(args.hires_fix_start_width)
    hires_fix_start_height = int(args.hires_fix_start_height)

    # Pixel-space overrides are converted to 64x latent blocks.
    if hires_fix_start_width <= 0 and args.hires_fix_width > 0:
        hires_fix_start_width = max(1, args.hires_fix_width // 64)
    if hires_fix_start_height <= 0 and args.hires_fix_height > 0:
        hires_fix_start_height = max(1, args.hires_fix_height // 64)

    if args.hires_fix:
        # If hires-fix is enabled but no explicit start size is provided,
        # default to a half-resolution first pass.
        if hires_fix_start_width <= 0:
            hires_fix_start_width = max(1, (start_width + 1) // 2)
        if hires_fix_start_height <= 0:
            hires_fix_start_height = max(1, (start_height + 1) // 2)

        # Runtime requires hires-fix start sizes to be strictly smaller than start size.
        hires_fix_start_width = min(hires_fix_start_width, max(1, start_width - 1))
        hires_fix_start_height = min(hires_fix_start_height, max(1, start_height - 1))
    else:
        hires_fix_start_width = 0
        hires_fix_start_height = 0

    builder = flatbuffers.Builder(4096)

    model = builder.CreateString(args.model)
    upscaler = _optional_string_offset(builder, args.upscaler)
    face_restoration = _optional_string_offset(builder, args.face_restoration)
    refiner_model = _optional_string_offset(builder, args.refiner_model)
    name = builder.CreateString(args.name)
    clip_l_text = _optional_string_offset(builder, args.clip_l_text)
    open_clip_g_text = _optional_string_offset(builder, args.open_clip_g_text)
    t5_text = _optional_string_offset(builder, args.t5_text)

    lora_offsets = []
    lora_specs = _parse_lora_specs(args)
    for lora in lora_specs:
        lora_file = builder.CreateString(lora["file"])
        LoRA.LoRAStart(builder)
        LoRA.LoRAAddFile(builder, lora_file)
        LoRA.LoRAAddWeight(builder, lora["weight"])
        LoRA.LoRAAddMode(builder, lora["mode"])
        lora_offsets.append(LoRA.LoRAEnd(builder))

    loras_vector = None
    if lora_offsets:
        GC.GenerationConfigurationStartLorasVector(builder, len(lora_offsets))
        for lora_offset in reversed(lora_offsets):
            builder.PrependUOffsetTRelative(lora_offset)
        loras_vector = builder.EndVector()

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
    GC.GenerationConfigurationAddHiresFixStartWidth(builder, hires_fix_start_width)
    GC.GenerationConfigurationAddHiresFixStartHeight(builder, hires_fix_start_height)
    GC.GenerationConfigurationAddHiresFixStrength(builder, args.hires_fix_strength)
    if upscaler is not None:
        GC.GenerationConfigurationAddUpscaler(builder, upscaler)
    GC.GenerationConfigurationAddSeedMode(builder, args.seed_mode)
    GC.GenerationConfigurationAddClipSkip(builder, args.clip_skip)
    if loras_vector is not None:
        GC.GenerationConfigurationAddLoras(builder, loras_vector)
    GC.GenerationConfigurationAddMaskBlur(builder, args.mask_blur)
    if face_restoration is not None:
        GC.GenerationConfigurationAddFaceRestoration(builder, face_restoration)
    if refiner_model is not None:
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
    if clip_l_text is not None:
        GC.GenerationConfigurationAddClipLText(builder, clip_l_text)
    GC.GenerationConfigurationAddSeparateOpenClipG(builder, args.separate_open_clip_g)
    if open_clip_g_text is not None:
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
    if t5_text is not None:
        GC.GenerationConfigurationAddT5Text(builder, t5_text)

    config = GC.GenerationConfigurationEnd(builder)
    builder.Finish(config)
    return (
        bytes(builder.Output()),
        seed,
        start_width,
        start_height,
        hires_fix_start_width,
        hires_fix_start_height,
        lora_specs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Draw Things GenerationConfiguration flatbuffer")
    p.add_argument("--bindings-dir", default=str(DEFAULT_PY_BINDINGS))
    p.add_argument("--out", required=True, help="Output .bin path")

    p.add_argument(
        "--model",
        required=True,
        help=(
            "Model file key (e.g. z_image_turbo_1.0_q6p.ckpt) or a custom.json "
            "model name alias."
        ),
    )
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
    p.add_argument("--hires-fix-start-width", type=int, default=0, help="Hires-fix start width in 64x blocks")
    p.add_argument("--hires-fix-start-height", type=int, default=0, help="Hires-fix start height in 64x blocks")
    p.add_argument("--hires-fix-width", type=int, default=0, help="Hires-fix start width in pixels")
    p.add_argument("--hires-fix-height", type=int, default=0, help="Hires-fix start height in pixels")
    p.add_argument("--hires-fix-strength", type=float, default=0.7)
    p.add_argument("--upscaler", default="")
    p.add_argument("--upscaler-scale-factor", type=int, default=0)

    p.add_argument("--mask-blur", type=float, default=1.5)
    p.add_argument("--mask-blur-outset", type=int, default=0)
    p.add_argument("--sharpness", type=float, default=0.0)

    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--resolution-dependent-shift", type=_parse_bool, default=True)

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

    p.add_argument(
        "--lora",
        action="append",
        default=[],
        help="Repeatable LoRA spec: FILE[:WEIGHT[:MODE]] where MODE is all/base/refiner or 0/1/2",
    )
    p.add_argument(
        "--lora-mode",
        default="all",
        help="Default mode for --lora entries without explicit mode (all/base/refiner or 0/1/2)",
    )

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

    resolved_model, resolved_from = _resolve_model_name_to_file(args.model)
    if resolved_model != args.model:
        print(f"resolved model alias: {args.model} -> {resolved_model} ({resolved_from})")
        args.model = resolved_model

    if args.width < 64 or args.width % 64 != 0:
        raise ValueError(f"--width must be a positive multiple of 64, got: {args.width}")
    if args.height < 64 or args.height % 64 != 0:
        raise ValueError(f"--height must be a positive multiple of 64, got: {args.height}")
    if args.hires_fix_width > 0 and args.hires_fix_width % 64 != 0:
        raise ValueError(
            f"--hires-fix-width must be a multiple of 64 when set, got: {args.hires_fix_width}"
        )
    if args.hires_fix_height > 0 and args.hires_fix_height % 64 != 0:
        raise ValueError(
            f"--hires-fix-height must be a multiple of 64 when set, got: {args.hires_fix_height}"
        )

    blob, seed, start_w, start_h, hires_fix_start_w, hires_fix_start_h, lora_specs = build_config(
        args
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    print(f"wrote config bytes: {out_path}")
    print(f"size bytes: {len(blob)}")
    print(f"model: {args.model}")
    print(f"seed: {seed}")
    print(f"start_width/start_height (64x blocks): {start_w}/{start_h}")
    print(f"pixel size: {start_w * 64}x{start_h * 64}")
    print(f"loras: {len(lora_specs)}")
    for lora in lora_specs:
        suffix = ""
        if lora["resolved_from"]:
            suffix = f" ({lora['input']} -> {lora['file']} via {lora['resolved_from']})"
        print(
            f"  - file={lora['file']} weight={lora['weight']} mode={lora['mode']}{suffix}"
        )
    print(f"hires_fix: {args.hires_fix}")
    if args.hires_fix:
        print(
            "hires_fix_start_width/start_height (64x blocks): "
            f"{hires_fix_start_w}/{hires_fix_start_h}"
        )
        print(f"hires_fix pixel size: {hires_fix_start_w * 64}x{hires_fix_start_h * 64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
