#!/usr/bin/env python3
"""Universal Multi-Architecture LoRA Normalizer & §24 Static Verifier (Milestone W4).

Adapts WeeLLM's LoRA keymaps (`weellm/io/safetensors/keymaps/loras/`) and generalizes
the static mathematical audit from `CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md` (§23/§24)
to verify any converted Draw Things LoRA `.ckpt` against its source `.safetensors` and
`contracts/<profile>.json`:
  - Verifies that every rotary `to_q` / `to_k` `__up__` matrix in the `.ckpt` matches
    `rotary_deinterleave(lora_B)` (`max_abs_diff <= atol`).
  - Verifies that non-rotary `__up__` matrices (`to_v`, `to_out`, FFN) and all `__down__`
    matrices (`lora_A`) match `IDENTITY` (`max_abs_diff <= atol`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_direct_ckpt_builder import load_contract
from tools.dt_weellm.keymaps import canonicalize_lora_key
from tools.dt_weellm.seek_io import SafetensorsSeeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import apply_contract_rule_to_tensor, write_safetensors_file

LORA_SOURCE_PREFIXES = (
    "",
    "diffusion_model.",
    "model.diffusion_model.",
    "transformer.",
    "lycoris_",
    "lora_unet_",
)

LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
LORA_UP_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight")


def strip_lora_key(raw_key: str) -> Optional[Tuple[str, str]]:
    """Return `(base_module_weight_key, direction)` where direction is `'down'` or `'up'`."""
    key = raw_key
    for prefix in ("diffusion_model.", "model.diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    for down_sfx in LORA_DOWN_SUFFIXES:
        if key.endswith(down_sfx):
            return key[: -len(down_sfx)] + ".weight", "down"
    for up_sfx in LORA_UP_SUFFIXES:
        if key.endswith(up_sfx):
            return key[: -len(up_sfx)] + ".weight", "up"
    return None


def dt_base_name_to_lora_names(dt_base_name: str) -> Tuple[str, str]:
    """Map a base `.ckpt` tensor name like `__dit__[t-x_q-0-0]` to `(__up__, __down__)`."""
    return f"{dt_base_name}__up__", f"{dt_base_name}__down__"


def normalize_lora_safetensors(
    input_path: str | Path,
    output_path: str | Path,
    profile: str = "ltx2_3",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Inspect and normalize any Kohya, LyCORIS, ComfyUI, or Diffusers LoRA into canonical Diffusers format."""
    in_p = Path(input_path).expanduser().resolve()
    out_p = Path(output_path).expanduser().resolve()
    if in_p == out_p or (out_p.exists() and not overwrite):
        raise ValueError("Normalization needs a distinct output and explicit overwrite permission")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    normalized_tensors: Dict[str, Tuple[str, Sequence[int], bytes]] = {}
    stats = {
        "input_tensors": 0,
        "normalized_up_tensors": 0,
        "normalized_down_tensors": 0,
        "normalized_alpha_tensors": 0,
        "skipped_unrecognized": 0,
    }

    with SafetensorsSeeker(in_p) as seeker:
        for key in seeker.keys():
            stats["input_tensors"] += 1
            info, raw = seeker.read_raw_bytes(key)
            parsed = canonicalize_lora_key(key, profile=profile)
            if parsed is None:
                raise ValueError(f"Unsupported LoRA tensor key: {key}")

            canonical_weight, kind = parsed
            stem = (
                canonical_weight[: -len(".weight")]
                if canonical_weight.endswith(".weight")
                else canonical_weight
            )
            if kind == "down":
                target_key = f"diffusion_model.{stem}.lora_A.weight"
                stats["normalized_down_tensors"] += 1
            elif kind == "up":
                target_key = f"diffusion_model.{stem}.lora_B.weight"
                stats["normalized_up_tensors"] += 1
            elif kind == "alpha":
                target_key = f"diffusion_model.{stem}.alpha"
                stats["normalized_alpha_tensors"] += 1
            else:
                target_key = key

            if target_key in normalized_tensors:
                raise ValueError(f"LoRA normalization collision: {target_key}")
            normalized_tensors[target_key] = (info.dtype, info.shape, raw)
        metadata = dict(seeker.metadata)
    if not stats["normalized_down_tensors"] or not stats["normalized_up_tensors"]:
        raise ValueError("LoRA contains no complete weight directions")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{out_p.name}.tmp.", dir=out_p.parent)
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        write_safetensors_file(staging, normalized_tensors, metadata=metadata)
        os.replace(staging, out_p)
    finally:
        if staging.exists():
            staging.unlink()
    return stats


def verify_converted_lora(
    source_safetensors: str | Path,
    converted_ckpt: str | Path,
    contract_path: str | Path,
    atol: float = 1e-2,
    scale_factor: float = 1.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the §24 static LoRA verification across all mapped projections."""
    contract = load_contract(contract_path)
    if not math.isfinite(atol) or atol < 0 or not math.isfinite(scale_factor) or scale_factor < 0:
        raise ValueError("Tolerance and scale factor must be finite and nonnegative")
    tensor_rules: Dict[str, Dict[str, Any]] = contract.get("tensors", {})

    paired_source: Dict[str, Dict[str, str]] = {}
    failures: List[str] = []
    with SafetensorsSeeker(source_safetensors) as sf:
        for raw_key in sf.keys():
            parsed = canonicalize_lora_key(raw_key, profile=contract["profile"])
            if parsed is None:
                failures.append(f"Unrecognized LoRA tensor: {raw_key}")
                continue
            base_weight_key, direction = parsed
            if direction in paired_source.get(base_weight_key, {}):
                raise ValueError(f"Duplicate canonical LoRA direction: {base_weight_key} {direction}")
            paired_source.setdefault(base_weight_key, {})[direction] = raw_key

        with DrawThingsCkptStore(converted_ckpt, writable=False) as ckpt:
            ckpt_keys = set(ckpt.list_keys())
            verified_pairs = 0
            rotary_permuted_up_verified = 0
            identity_up_verified = 0
            identity_down_verified = 0
            expected_ckpt_keys = set()

            for base_weight_key, pair in sorted(paired_source.items()):
                if "up" not in pair or "down" not in pair:
                    failures.append(f"Incomplete LoRA A/B pair in source for {base_weight_key}")
                    continue

                rule = tensor_rules.get(base_weight_key)
                if rule is None:
                    failures.append(f"Unmapped LoRA module in contract: {base_weight_key}")
                    continue

                dt_names = rule.get("dt_names", [])
                if rule.get("split_axis", 0) != 0 or rule.get("split_policy") not in (None, "chunk_o"):
                    failures.append(f"Unsupported input/diagonal LoRA split: {base_weight_key}")
                    continue
                raw_up = sf.read_f32(pair["up"])
                raw_down = sf.read_f32(pair["down"])
                if raw_up.ndim != 2 or raw_down.ndim != 2 or raw_up.shape[1] != raw_down.shape[0] or not raw_up.shape[1]:
                    failures.append(f"Invalid LoRA rank/shape: {base_weight_key}")
                    continue
                alpha = float(raw_up.shape[1])
                if "alpha" in pair:
                    alpha_values = sf.read_f32(pair["alpha"])
                    if alpha_values.size != 1:
                        failures.append(f"LoRA alpha must be scalar: {base_weight_key}")
                        continue
                    alpha = float(alpha_values.reshape(-1)[0])
                if not math.isfinite(alpha) or not np.isfinite(raw_up).all() or not np.isfinite(raw_down).all():
                    failures.append(f"Non-finite LoRA values: {base_weight_key}")
                    continue
                transform = {key: value for key, value in rule.items()
                             if key not in ("source_shape", "target_shapes", "dim_rank_policy")}
                expected_up_groups = apply_contract_rule_to_tensor(
                    raw_up * (alpha / raw_up.shape[1]) * math.sqrt(scale_factor), transform)
                expected_down = raw_down * math.sqrt(scale_factor)
                for base_name, expected_up in expected_up_groups:
                    up_name, down_name = dt_base_name_to_lora_names(base_name)
                    expected_ckpt_keys.update((up_name, down_name))
                    if up_name not in ckpt_keys or down_name not in ckpt_keys:
                        failures.append(f"Missing converted LoRA tensors: {base_name}")
                        continue
                    up_result, down_result = ckpt.read_f32(up_name), ckpt.read_f32(down_name)
                    if up_result is None or down_result is None:
                        failures.append(f"Unreadable converted LoRA tensors: {base_name}")
                        continue
                    stored_up, stored_down = up_result[1], down_result[1]
                    for label, expected, stored in ((up_name, expected_up, stored_up), (down_name, expected_down, stored_down)):
                        if expected.shape != stored.shape or not np.isfinite(stored).all() or not np.allclose(expected, stored, rtol=0, atol=atol):
                            failures.append(f"LoRA shape/value mismatch: {label}")
                    identity_down_verified += 1
                    if rule.get("rotary_interleaved") or any(rule.get("rotary_interleaved_slices", [])):
                        rotary_permuted_up_verified += 1
                    else:
                        identity_up_verified += 1
                verified_pairs += 1
            if ckpt_keys != expected_ckpt_keys:
                failures.append("Converted LoRA key coverage differs from source mappings")
            if not verified_pairs or verified_pairs != len(paired_source):
                failures.append("No complete LoRA verification coverage")

    report = {
        "source_pairs_found": len(paired_source),
        "verified_pairs": verified_pairs,
        "rotary_permuted_up_verified": rotary_permuted_up_verified,
        "identity_up_verified": identity_up_verified,
        "identity_down_verified": identity_down_verified,
        "failures": failures,
    }
    return len(failures) == 0, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-architecture LoRA Static Verifier")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_ver = sub.add_parser("verify", help="Verify converted Draw Things LoRA .ckpt against source .safetensors")
    p_ver.add_argument("--source-safetensors", required=True)
    p_ver.add_argument("--converted-ckpt", required=True)
    p_ver.add_argument("--contract", required=True)
    p_ver.add_argument("--atol", type=float, default=1e-2)
    p_ver.add_argument("--scale-factor", type=float, default=1.0)

    p_norm = sub.add_parser("normalize", help="Normalize community LoRA keys (Kohya/LyCORIS/ComfyUI -> Diffusers)")
    p_norm.add_argument("--input", required=True, help="Input raw LoRA .safetensors")
    p_norm.add_argument("--output", required=True, help="Output normalized LoRA .safetensors")
    p_norm.add_argument("--profile", default="ltx2_3", choices=["ltx2_3", "moss_image", "flux2"])
    p_norm.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.subcommand == "verify":
        ok, report = verify_converted_lora(
            args.source_safetensors,
            args.converted_ckpt,
            args.contract,
            atol=args.atol,
            scale_factor=args.scale_factor,
        )
        print("== Static LoRA Verification (§24 Audit) ==")
        for k, v in report.items():
            if k != "failures":
                print(f"  {k}={v}")
        if not ok:
            for f in report["failures"][:15]:
                print(f"  FAIL: {f}")
            print("RESULT=FAIL")
            return 1
        print("RESULT=PASS")
        return 0
    if args.subcommand == "normalize":
        stats = normalize_lora_safetensors(args.input, args.output, profile=args.profile, overwrite=args.force)
        print("== Universal LoRA Normalizer ==")
        for k, v in stats.items():
            print(f"  {k}={v}")
        print("RESULT=PASS")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
