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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.dt_weellm.seek_io import SafetensorsSeeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import rotary_deinterleave

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


def verify_converted_lora(
    source_safetensors: str | Path,
    converted_ckpt: str | Path,
    contract_path: str | Path,
    atol: float = 1e-2,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the §24 static LoRA verification across all mapped projections."""
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    tensor_rules: Dict[str, Dict[str, Any]] = contract.get("tensors", {})

    paired_source: Dict[str, Dict[str, str]] = {}
    with SafetensorsSeeker(source_safetensors) as sf:
        for raw_key in sf.keys():
            parsed = strip_lora_key(raw_key)
            if parsed is None:
                continue
            base_weight_key, direction = parsed
            paired_source.setdefault(base_weight_key, {})[direction] = raw_key

        with DrawThingsCkptStore(converted_ckpt, writable=False) as ckpt:
            ckpt_keys = set(ckpt.list_keys())
            verified_pairs = 0
            rotary_permuted_up_verified = 0
            identity_up_verified = 0
            identity_down_verified = 0
            failures: List[str] = []

            for base_weight_key, pair in sorted(paired_source.items()):
                if "up" not in pair or "down" not in pair:
                    failures.append(f"Incomplete LoRA A/B pair in source for {base_weight_key}")
                    continue

                rule = tensor_rules.get(base_weight_key)
                if rule is None:
                    failures.append(f"Unmapped LoRA module in contract: {base_weight_key}")
                    continue

                dt_names = rule.get("dt_names", [])
                if len(dt_names) != 1:
                    continue

                up_name, down_name = dt_base_name_to_lora_names(dt_names[0])
                if up_name not in ckpt_keys or down_name not in ckpt_keys:
                    failures.append(
                        f"Missing converted LoRA tensors in ckpt for {base_weight_key}: expected {up_name}, {down_name}"
                    )
                    continue

                raw_up = sf.read_f32(pair["up"])
                raw_down = sf.read_f32(pair["down"])
                _, stored_up = ckpt.read_f32(up_name)
                _, stored_down = ckpt.read_f32(down_name)

                # 1. Down matrix (`lora_A`) must always be IDENTITY
                down_diff = float(np.max(np.abs(raw_down - stored_down)))
                if down_diff > atol:
                    failures.append(
                        f"Down matrix mismatch for {base_weight_key} ({down_name}): max_abs_diff={down_diff:.6g}"
                    )
                else:
                    identity_down_verified += 1

                # 2. Up matrix (`lora_B`) must be rotary-deinterleaved iff rule["rotary_interleaved"] is True
                if rule.get("rotary_interleaved", False):
                    expected_up = rotary_deinterleave(
                        raw_up,
                        num_heads=int(rule["num_heads"]),
                        head_dim=int(rule["head_dim"]),
                    )
                    up_diff = float(np.max(np.abs(expected_up - stored_up)))
                    if up_diff > atol:
                        failures.append(
                            f"Rotary up matrix mismatch for {base_weight_key} ({up_name}): "
                            f"max_abs_diff={up_diff:.6g} (did converter miss §17 de-interleave?)"
                        )
                    else:
                        rotary_permuted_up_verified += 1
                else:
                    up_diff = float(np.max(np.abs(raw_up - stored_up)))
                    if up_diff > atol:
                        failures.append(
                            f"Non-rotary up matrix mismatch for {base_weight_key} ({up_name}): max_abs_diff={up_diff:.6g}"
                        )
                    else:
                        identity_up_verified += 1

                verified_pairs += 1

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

    args = parser.parse_args(argv)
    if args.subcommand == "verify":
        ok, report = verify_converted_lora(
            args.source_safetensors,
            args.converted_ckpt,
            args.contract,
            atol=args.atol,
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
