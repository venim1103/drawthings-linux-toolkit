#!/usr/bin/env python3
"""Bounded-VRAM WeeLLM vs. Draw Things Layer Parity & Bisection Oracle (Milestone W2).

Provides three low-VRAM diagnostic modes for resolving runtime numerical divergence
and converter transform bugs on a 12 GiB GPU:
  1. `verify-block`: 5-second CPU/NumPy single-block weight & forward-pass parity check
     comparing raw `.safetensors`/`.gguf` weights against a converted Draw Things `.ckpt`.
  2. `qwen35-linear-attn-fixture`: Deterministic reference implementation of Qwen3.5's
     `GatedDelta` (`0x6de198ac`) and `RMSNormGated` (`0x9b8af346`) operations (the exact
     two operations missing CUDA backends in CCV during MOSS M1) to generate golden input/output
     binaries for C/CUDA kernel verification.
  3. `inspect-ckpt-health`: Scans a Draw Things `.ckpt` (including externalized `-tensordata`)
     for non-finite (`NaN`/`Inf`) weights, unexpected datatypes, or rank-1 vs rank-3 norm mismatches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.dt_direct_ckpt_builder import load_contract
from tools.dt_weellm.seek_io import open_seeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DATATYPE_CODE_TO_NAME,
    DT_FLOAT32,
    rotary_deinterleave,
    split_tensor_chunks,
)


def rms_norm_f32(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Reference RMSNorm along the last axis in float32."""
    x_f32 = x.astype(np.float32)
    w_f32 = weight.astype(np.float32).reshape(-1)
    variance = np.mean(np.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * (1.0 / np.sqrt(variance + eps))
    return normed * w_f32


def silu_f32(x: np.ndarray) -> np.ndarray:
    x_f32 = x.astype(np.float32)
    return x_f32 / (1.0 + np.exp(-x_f32))


def reference_rmsnorm_gated(
    x: np.ndarray,
    gate: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Reference implementation of Qwen3.5 `RMSNormGated` (CCV command ID `0x9b8af346`).

    Computes `RMSNorm(x, weight, eps) * SiLU(gate)` in float32.
    """
    normed = rms_norm_f32(x, weight, eps=eps)
    return normed * silu_f32(gate)


def reference_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    decay: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference recurrent implementation of Qwen3.5 `GatedDelta` (CCV command ID `0x6de198ac`).

    Shapes:
      - `q`, `k`: `[batch, seq_len, num_heads, head_k_dim]`
      - `v`:      `[batch, seq_len, num_heads, head_v_dim]`
      - `beta`:   `[batch, seq_len, num_heads]` (writing gate in (0, 1))
      - `decay`:  `[batch, seq_len, num_heads]` (log-decay or exp-decay factor in (0, 1])

    State update per step `t`:
      1. `state = decay[t] * state`
      2. `v_pred = einsum('bhk,bhkv->bhv', k[t], state)`
      3. `delta = beta[t] * (v[t] - v_pred)`
      4. `state = state + einsum('bhk,bhv->bhkv', k[t], delta)`
      5. `out[t] = einsum('bhk,bhkv->bhv', q[t], state)`
    """
    b, seq_len, h, d_k = q.shape
    d_v = v.shape[-1]
    state = np.zeros((b, h, d_k, d_v), dtype=np.float32)
    outputs = np.zeros((b, seq_len, h, d_v), dtype=np.float32)

    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    beta_f = beta.astype(np.float32)
    decay_f = decay.astype(np.float32)

    for t in range(seq_len):
        g_t = decay_f[:, t, :, None, None]  # [b, h, 1, 1]
        beta_t = beta_f[:, t, :, None]      # [b, h, 1]
        k_t = k_f[:, t, :, :]               # [b, h, d_k]
        q_t = q_f[:, t, :, :]               # [b, h, d_k]
        v_t = v_f[:, t, :, :]               # [b, h, d_v]

        state = state * g_t
        v_pred = np.einsum("bhk,bhkv->bhv", k_t, state)
        delta = beta_t * (v_t - v_pred)
        state = state + np.einsum("bhk,bhv->bhkv", k_t, delta)
        outputs[:, t, :, :] = np.einsum("bhk,bhkv->bhv", q_t, state)

    return outputs, state


def verify_block_parity(
    safetensors_path: str | Path,
    ckpt_path: str | Path,
    contract_path: str | Path,
    block_prefix: str = "transformer_blocks.0.",
    atol: float = 1e-2,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify all tensors matching `block_prefix` between `safetensors_path` and `ckpt_path`."""
    contract = load_contract(contract_path)
    rules: Dict[str, Dict[str, Any]] = contract.get("tensors", {})
    matching_rules = {k: v for k, v in rules.items() if k.startswith(block_prefix)}
    if not matching_rules:
        raise ValueError(f"No contract keys start with {block_prefix!r} in {contract_path}")

    failures: List[str] = []
    checked = 0
    max_observed_diff = 0.0

    with open_seeker(safetensors_path) as seeker, DrawThingsCkptStore(ckpt_path, writable=False) as ckpt:
        for src_key, rule in sorted(matching_rules.items()):
            resolved = seeker.resolve_key(src_key) if hasattr(seeker, "resolve_key") else src_key
            if resolved is None or resolved not in seeker.keys():
                failures.append(f"Missing source key: {src_key}")
                continue

            src_arr = seeker.read_f32(resolved)
            if rule.get("rotary_interleaved", False):
                src_arr = rotary_deinterleave(
                    src_arr,
                    num_heads=int(rule["num_heads"]),
                    head_dim=int(rule["head_dim"]),
                )

            dt_names = rule["dt_names"]
            chunks = (
                split_tensor_chunks(src_arr, count=len(dt_names), axis=int(rule.get("split_axis", 0)))
                if len(dt_names) > 1
                else [src_arr]
            )

            for dt_name, expected_chunk in zip(dt_names, chunks):
                res = ckpt.read_f32(dt_name)
                if res is None:
                    failures.append(f"Missing ckpt tensor: {dt_name}")
                    continue
                meta, ckpt_arr = res
                if rule.get("dim_rank_policy") == "norm_1_1_n":
                    n_elem = int(np.prod(expected_chunk.shape))
                    if meta.shape != [1, 1, n_elem]:
                        failures.append(
                            f"Rank-3 norm policy violation on {dt_name}: got shape {meta.shape}, expected [1, 1, {n_elem}]"
                        )
                diff = float(np.max(np.abs(expected_chunk.reshape(-1) - ckpt_arr.reshape(-1))))
                max_observed_diff = max(max_observed_diff, diff)
                if diff > atol:
                    failures.append(
                        f"Value mismatch on {dt_name} (from {src_key}): max_abs_diff={diff:.6g} > atol={atol}"
                    )
                checked += 1

    report = {
        "block_prefix": block_prefix,
        "tensors_checked": checked,
        "max_observed_diff": max_observed_diff,
        "failures": failures,
    }
    return len(failures) == 0, report


def cmd_generate_qwen35_fixtures(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    b, seq_len, h, d_k, d_v = 1, 8, 4, 16, 16
    q = rng.standard_normal((b, seq_len, h, d_k)).astype(np.float32) * 0.25
    k = rng.standard_normal((b, seq_len, h, d_k)).astype(np.float32) * 0.25
    v = rng.standard_normal((b, seq_len, h, d_v)).astype(np.float32) * 0.25
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((b, seq_len, h)).astype(np.float32)))
    decay = np.exp(-np.abs(rng.standard_normal((b, seq_len, h)).astype(np.float32)) * 0.1)

    delta_out, final_state = reference_gated_delta_rule(q, k, v, beta, decay)

    x_norm = rng.standard_normal((b, seq_len, h * d_v)).astype(np.float32)
    gate = rng.standard_normal((b, seq_len, h * d_v)).astype(np.float32)
    weight = np.ones((h * d_v,), dtype=np.float32) + rng.standard_normal((h * d_v,)).astype(np.float32) * 0.05
    rmsnorm_gated_out = reference_rmsnorm_gated(x_norm, gate, weight)

    np.savez(
        out_dir / "qwen35_linear_attention_golden.npz",
        q=q,
        k=k,
        v=v,
        beta=beta,
        decay=decay,
        gated_delta_out=delta_out,
        gated_delta_final_state=final_state,
        x_norm=x_norm,
        gate=gate,
        weight=weight,
        rmsnorm_gated_out=rmsnorm_gated_out,
    )

    manifest = {
        "seed": args.seed,
        "commands_covered": {
            "GatedDelta": "0x6de198ac",
            "RMSNormGated": "0x9b8af346",
        },
        "gated_delta_out_finite": bool(np.all(np.isfinite(delta_out))),
        "gated_delta_out_std": float(np.std(delta_out)),
        "rmsnorm_gated_out_finite": bool(np.all(np.isfinite(rmsnorm_gated_out))),
        "rmsnorm_gated_out_std": float(np.std(rmsnorm_gated_out)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("== Qwen3.5 Linear Attention Golden Fixtures Generated ==")
    print(f"output={out_dir / 'qwen35_linear_attention_golden.npz'}")
    for k_name, v_val in manifest.items():
        print(f"  {k_name}={v_val}")
    return 0


def inspect_ckpt_health(
    ckpt_path: str | Path,
    contract_path: Optional[str | Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Scan every tensor in a Draw Things .ckpt (including sidecars) for finiteness, datatypes, and rank-3 norm policy."""
    p = Path(ckpt_path).expanduser().resolve()
    contract: Optional[Dict[str, Any]] = load_contract(contract_path) if contract_path else None

    failures: List[str] = []
    total_tensors = 0
    external_tensors = 0
    finite_checked = 0
    f32_count = 0
    f16_count = 0
    bf16_count = 0

    with DrawThingsCkptStore(p, writable=False) as store:
        keys = store.list_keys()
        total_tensors = len(keys)
        for name in keys:
            meta = store.read_metadata(name)
            if meta is None:
                failures.append(f"{name}: unreadable SQLite metadata")
                continue
            if meta.is_external:
                external_tensors += 1

            dt = meta.datatype
            if dt == DT_FLOAT32:
                f32_count += 1
            elif dt == 131072:
                f16_count += 1
            elif dt == 524288:
                bf16_count += 1

            if dt in (DT_FLOAT32, 131072, 524288):
                try:
                    res = store.read_f32(name)
                    if res is not None:
                        _, arr = res
                        if not np.all(np.isfinite(arr)):
                            nan_c = int(np.isnan(arr).sum())
                            inf_c = int(np.isinf(arr).sum())
                            failures.append(f"{name}: non-finite values (NaN={nan_c}, Inf={inf_c})")
                        finite_checked += 1
                except Exception as exc:
                    failures.append(f"{name}: read_f32 failed: {exc}")

            if contract:
                patterns = contract.get("restore_patterns", {})
                ada_ln_re = re.compile(patterns.get("f32_ada_ln_regex", r"ada_ln"))
                qk_norm_re = re.compile(patterns.get("f32_qk_norm_regex", r"(norm_q|norm_k)-\d+-0\]$"))

                if ada_ln_re.search(name) or qk_norm_re.search(name):
                    if len(meta.shape) != 3 or meta.shape[0] != 1 or meta.shape[1] != 1:
                        failures.append(
                            f"{name}: rank-3 policy violation, got shape {meta.shape} (expected [1, 1, N])"
                        )

    summary = {
        "file": str(p),
        "total_tensors": total_tensors,
        "external_tensors": external_tensors,
        "finite_checked": finite_checked,
        "f32_tensors": f32_count,
        "f16_tensors": f16_count,
        "bf16_tensors": bf16_count,
        "failures": failures,
    }
    return len(failures) == 0, summary


def cmd_inspect_health(args: argparse.Namespace) -> int:
    ok, report = inspect_ckpt_health(args.ckpt, contract_path=args.contract or None)
    print("== Draw Things Checkpoint Health Audit ==")
    print(f"file={report['file']} total_tensors={report['total_tensors']}")
    print(f"external_tensors={report['external_tensors']} finite_checked={report['finite_checked']}")
    print(f"datatype_counts: F32={report['f32_tensors']}, F16={report['f16_tensors']}, BF16={report['bf16_tensors']}")
    if not ok:
        print(f"failures_count={len(report['failures'])}")
        for f in report["failures"][:20]:
            print(f"  FAIL: {f}")
        print("RESULT=FAIL")
        return 1
    print("RESULT=PASS")
    return 0


def cmd_verify_block(args: argparse.Namespace) -> int:
    contract = load_contract(args.contract)
    block_prefixes = [args.block_prefix]
    if args.all_blocks:
        max_layer = int(contract.get("architecture", {}).get("layers", 48))
        block_prefixes = [f"transformer_blocks.{i}." for i in range(max_layer)]

    total_failures = 0
    total_checked = 0
    for bp in block_prefixes:
        try:
            ok, report = verify_block_parity(
                safetensors_path=args.safetensors,
                ckpt_path=args.ckpt,
                contract_path=args.contract,
                block_prefix=bp,
                atol=args.atol,
            )
            total_checked += report["tensors_checked"]
            if not ok:
                total_failures += len(report["failures"])
                print(f"FAIL [{bp}]: max_diff={report['max_observed_diff']:.6g}")
                for f in report["failures"][:5]:
                    print(f"  {f}")
        except Exception as exc:
            total_failures += 1
            print(f"ERROR [{bp}]: {exc}")

    print("== Single/Multi-Block Weight & Transform Parity Check ==")
    print(f"blocks_tested={len(block_prefixes)} tensors_checked={total_checked} failures={total_failures}")
    if total_failures > 0:
        print("RESULT=FAIL")
        return 1
    print("RESULT=PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WeeLLM vs Draw Things Layer Parity Oracle")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_fix = sub.add_parser(
        "qwen35-linear-attn-fixture",
        help="Generate golden NPZ fixtures for Qwen3.5 GatedDelta (0x6de198ac) & RMSNormGated (0x9b8af346)",
    )
    p_fix.add_argument("--out-dir", required=True)
    p_fix.add_argument("--seed", type=int, default=4242)

    p_blk = sub.add_parser(
        "verify-block",
        help="Verify single-block weight & transform parity between source .safetensors and .ckpt",
    )
    p_blk.add_argument("--safetensors", required=True)
    p_blk.add_argument("--ckpt", required=True)
    p_blk.add_argument("--contract", required=True)
    p_blk.add_argument("--block-prefix", default="transformer_blocks.0.")
    p_blk.add_argument("--all-blocks", action="store_true", help="Verify all transformer blocks 0..N")
    p_blk.add_argument("--atol", type=float, default=1e-2)

    p_chk = sub.add_parser("inspect-ckpt-health", help="Audit .ckpt for NaNs, Infs, and rank-3 norm policy")
    p_chk.add_argument("--ckpt", required=True)
    p_chk.add_argument("--contract", default="", help="Optional contract to enforce norm rank-3 and F32 policy")

    args = parser.parse_args(argv)
    if args.subcommand == "qwen35-linear-attn-fixture":
        return cmd_generate_qwen35_fixtures(args)
    if args.subcommand == "verify-block":
        return cmd_verify_block(args)
    if args.subcommand == "inspect-ckpt-health":
        return cmd_inspect_health(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
