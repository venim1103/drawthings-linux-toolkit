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
import ctypes
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_direct_ckpt_builder import load_contract
from tools.dt_tensor_to_playable import _read_tensor_header, _tensor_to_numpy
from tools.dt_weellm.seek_io import open_seeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DATATYPE_CODE_TO_NAME,
    DT_FLOAT32,
    apply_contract_rule_to_tensor,
    decode_dim_blob,
    encode_dim_blob,
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
    initial_state: Optional[np.ndarray] = None,
    log_decay: bool = False,
    state_checkpoint_count: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grouped recurrence with native [B,Hv,Dv,Dk] state and explicit decay mode."""
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Q/K must share [B,T,Hk,Dk]; V must be [B,T,Hv,Dv]")
    batch, sequence, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if min(q.shape + v.shape) < 1 or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("Invalid grouped attention dimensions")
    gate_shape = (batch, sequence, value_heads)
    if beta.shape != gate_shape or decay.shape not in (gate_shape, gate_shape + (key_dim,)):
        raise ValueError("Decay/beta shape does not match grouped value heads")
    if not 0 <= state_checkpoint_count < sequence:
        raise ValueError("Invalid state checkpoint count")
    if any(not np.isfinite(values).all() for values in (q, k, v, beta, decay)):
        raise ValueError("Attention inputs must be finite")
    state_shape = (batch, value_heads, value_dim, key_dim)
    if initial_state is not None and (initial_state.shape != state_shape or not np.isfinite(initial_state).all()):
        raise ValueError("Initial state must be finite [B,Hv,Dv,Dk]")
    state = np.zeros(state_shape, dtype=np.float32) if initial_state is None else initial_state.astype(np.float32).copy()
    outputs = np.zeros((batch, sequence, value_heads, value_dim), dtype=np.float32)
    groups = np.arange(value_heads) // (value_heads // key_heads)
    q_f = np.take(q.astype(np.float32), groups, axis=2)
    k_f = np.take(k.astype(np.float32), groups, axis=2)
    v_f = v.astype(np.float32)
    beta_f = beta.astype(np.float32)
    decay_f = decay.astype(np.float32)
    checkpoints = {}
    for timestep in range(sequence):
        gate = decay_f[:, timestep]
        if log_decay:
            gate = np.exp(gate)
        state *= gate[:, :, None, None] if decay.ndim == 3 else gate[:, :, None, :]
        keys = k_f[:, timestep]
        prediction = np.einsum("bhvk,bhk->bhv", state, keys)
        delta = beta_f[:, timestep, :, None] * (v_f[:, timestep] - prediction)
        state += delta[:, :, :, None] * keys[:, :, None, :]
        outputs[:, timestep] = np.einsum("bhvk,bhk->bhv", state, q_f[:, timestep])
        history_index = sequence - 1 - timestep
        if 0 < history_index <= state_checkpoint_count:
            checkpoints[history_index] = state.copy()
    history = [state] + [checkpoints[index] for index in range(1, state_checkpoint_count + 1)]
    return outputs, np.concatenate(history, axis=1)


def verify_block_parity(
    safetensors_path: str | Path,
    ckpt_path: str | Path,
    contract_path: str | Path,
    block_prefix: str = "transformer_blocks.0.",
    atol: float = 1e-2,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify all tensors matching `block_prefix` between `safetensors_path` and `ckpt_path`."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
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
            resolved = seeker.resolve_key(
                src_key, aliases=rule.get("aliases"), profile=contract["profile"]
            ) if hasattr(seeker, "resolve_key") else src_key
            if resolved is None or resolved not in seeker.keys():
                failures.append(f"Missing source key: {src_key}")
                continue

            src_arr = seeker.read_f32(resolved)
            if not np.isfinite(src_arr).all():
                failures.append(f"Non-finite source tensor: {src_key}")
                continue
            for dt_name, expected_chunk in apply_contract_rule_to_tensor(src_arr, rule):
                res = ckpt.read_f32(dt_name)
                if res is None:
                    failures.append(f"Missing ckpt tensor: {dt_name}")
                    continue
                meta, ckpt_arr = res
                expected_shape, _ = decode_dim_blob(encode_dim_blob(
                    expected_chunk.shape, rank_policy=rule.get("dim_rank_policy", "preserve")))
                if meta.shape != expected_shape:
                    failures.append(f"Shape mismatch on {dt_name}: {meta.shape} != {expected_shape}")
                    continue
                if not np.isfinite(ckpt_arr).all() or not np.isfinite(expected_chunk).all():
                    failures.append(f"Non-finite values on {dt_name}")
                    continue
                diff = float(np.max(np.abs(expected_chunk.reshape(-1).astype(np.float64) - ckpt_arr.reshape(-1).astype(np.float64))))
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
    return bool(checked) and len(failures) == 0, report


def validate_qwen35_native_fixture(fixture: Path, source_dir: Path) -> Dict[str, Any]:
    compiler = shutil.which("cc")
    native_sources = [source_dir / "lib/nnc/cmd" / relative for relative in (
        "gated_delta/ccv_nnc_gated_delta_cpu_ref.c",
        "rmsnorm_gated/ccv_nnc_rmsnorm_gated_cpu_ref.c")]
    if not compiler or not all(path.is_file() for path in native_sources):
        raise ValueError("Native fixture validation requires a C compiler and CCV operator sources")
    with tempfile.TemporaryDirectory() as temporary:
        library_path = Path(temporary) / "qwen35.so"
        probe_source = Path(__file__).with_name("dt_qwen35_cpu_probe.c")
        subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-I", str(source_dir / "lib"),
                        str(probe_source), "-lm", "-o", str(library_path)], check=True,
                       capture_output=True, text=True, timeout=60)
        library = ctypes.CDLL(str(library_path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        library.dt_probe_gated_delta.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                               *([pointer] * 8), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        library.dt_probe_rmsnorm.argtypes = [ctypes.c_int, ctypes.c_int, *([pointer] * 4)]
        with np.load(fixture, allow_pickle=False) as arrays:
            queries, values = arrays["q"], arrays["v"]
            dimensions = np.array([queries.shape[0], queries.shape[1], queries.shape[2],
                                   values.shape[2], queries.shape[3], values.shape[3]], dtype=np.int32)
            actual = np.empty_like(arrays["gated_delta_out"])
            state = np.empty_like(arrays["gated_delta_final_state"])
            status = library.dt_probe_gated_delta(dimensions, queries, arrays["k"], values,
                                                 arrays["decay"], arrays["beta"], arrays["initial_state"],
                                                 actual, state, 1, 2, 0)
            if status != 0:
                raise ValueError(f"Native gated delta failed with status {status}")
            np.testing.assert_allclose(actual, arrays["gated_delta_out"], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(state, arrays["gated_delta_final_state"], atol=2e-6, rtol=2e-5)
            norm_input = arrays["x_norm"]
            norm_actual = np.empty_like(norm_input)
            status = library.dt_probe_rmsnorm(*norm_input.shape, norm_input, arrays["gate"], arrays["weight"], norm_actual)
            if status != 0:
                raise ValueError(f"Native gated RMSNorm failed with status {status}")
            np.testing.assert_allclose(norm_actual, arrays["rmsnorm_gated_out"], atol=2e-6, rtol=2e-5)
            return {"result": "PASS", "atol": 2e-6, "rtol": 2e-5,
                    "gated_delta_max_abs": float(np.max(np.abs(actual - arrays["gated_delta_out"]))),
                    "state_max_abs": float(np.max(np.abs(state - arrays["gated_delta_final_state"]))),
                    "rmsnorm_max_abs": float(np.max(np.abs(norm_actual - arrays["rmsnorm_gated_out"]))),
                    "source_sha256": {str(path.relative_to(source_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in native_sources}}


def cmd_generate_qwen35_fixtures(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    batch, sequence, key_heads, value_heads, key_dim, value_dim = 1, 8, 16, 32, 128, 128
    q = rng.standard_normal((batch, sequence, key_heads, key_dim)).astype(np.float32) * 0.025
    k = rng.standard_normal(q.shape).astype(np.float32) * 0.025
    v = rng.standard_normal((batch, sequence, value_heads, value_dim)).astype(np.float32) * 0.25
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((batch, sequence, value_heads)).astype(np.float32)))
    decay = -np.abs(rng.standard_normal(beta.shape).astype(np.float32)) * 0.1
    initial_state = rng.standard_normal((batch, value_heads, value_dim, key_dim)).astype(np.float32) * 0.01
    delta_out, final_state = reference_gated_delta_rule(
        q, k, v, beta, decay, initial_state=initial_state, log_decay=True, state_checkpoint_count=2)

    x_norm = rng.standard_normal((batch * sequence * value_heads, value_dim)).astype(np.float32)
    gate = rng.standard_normal(x_norm.shape).astype(np.float32)
    weight = np.ones((value_dim,), dtype=np.float32) + rng.standard_normal((value_dim,)).astype(np.float32) * 0.05
    rmsnorm_gated_out = reference_rmsnorm_gated(x_norm, gate, weight)

    np.savez(
        out_dir / "qwen35_linear_attention_golden.npz",
        q=q,
        k=k,
        v=v,
        beta=beta,
        decay=decay,
        initial_state=initial_state,
        gated_delta_out=delta_out,
        gated_delta_final_state=final_state,
        x_norm=x_norm,
        gate=gate,
        weight=weight,
        rmsnorm_gated_out=rmsnorm_gated_out,
    )

    manifest = {
        "schema_version": 1,
        "dtype": "float32",
        "state_layout": "B,Hv,Dv,Dk",
        "key_heads": key_heads,
        "value_heads": value_heads,
        "log_decay": True,
        "state_checkpoint_count": 2,
        "rmsnorm_axis": 1,
        "rmsnorm_epsilon": 1e-6,
        "validation": "numpy_reference_only",
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
    fixture = out_dir / "qwen35_linear_attention_golden.npz"
    manifest["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if args.native_ccv_source:
        manifest["native_cpu"] = validate_qwen35_native_fixture(fixture, Path(args.native_ccv_source).resolve())
        manifest["validation"] = "native_cpu_verified"
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
        contract_rules = {
            name: (rule, index)
            for rule in contract["tensors"].values()
            for index, name in enumerate(rule["dt_names"])
        } if contract else {}
        if contract and set(keys) != set(contract_rules):
            failures.append("Checkpoint keyset does not match the contract")
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
            else:
                failures.append(f"{name}: unsupported datatype {dt}")

            if name in contract_rules:
                rule, index = contract_rules[name]
                if "target_shapes" in rule and meta.shape != rule["target_shapes"][index]:
                    failures.append(f"{name}: target shape mismatch")
                if rule.get("dim_rank_policy") == "norm_1_1_n" and (len(meta.shape) != 3 or meta.shape[:2] != [1, 1]):
                    failures.append(f"{name}: rank-3 policy violation")

    if not total_tensors or finite_checked != total_tensors:
        failures.append("Finiteness validation did not cover every tensor")

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


def compare_native_capture(prefix: Path, operation: str, atol: float, rtol: float,
                           scale: float = 1.0) -> Dict[str, Any]:
    if not all(math.isfinite(value) for value in (atol, rtol, scale)) or atol < 0 or rtol < 0:
        raise ValueError("Capture tolerances and scale must be finite; tolerances must be nonnegative")
    captures = {}

    def load(role):
        path = prefix.with_name(prefix.name + "_" + role + ".bin")
        with path.open("rb") as stream:
            blob = stream.read(64 * 1024 * 1024 + 69)
        if len(blob) > 64 * 1024 * 1024 + 68:
            raise ValueError("Capture exceeds the 64 MiB payload limit")
        header = _read_tensor_header(blob)
        if header.identifier != 0 or header.datatype not in (16384, 131072, 524288) or not header.dims:
            raise ValueError("Capture requires a nonempty raw floating-point tensor")
        dimensions = struct.unpack("<12I", blob[20:68])
        if dimensions != header.dims + (0,) * (12 - len(header.dims)):
            raise ValueError("Capture dimensions must have contiguous positive axes and zero padding")
        values = _tensor_to_numpy(blob, header, math.prod(header.dims)).reshape(header.dims).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Capture contains non-finite values: " + role)
        captures[role] = {"shape": list(values.shape), "format": header.tensor_format,
                  "datatype": header.datatype, "sha256": hashlib.sha256(blob).hexdigest()}
        return values

    values = load("input_0")
    right = load("input_1")
    actual = load("output_0")
    extra = []
    if operation == "linear":
        if values.ndim < 2 or right.ndim != 2 or values.shape[-1] != right.shape[1]:
            raise ValueError("Linear capture has incompatible activation/weight shapes")
        elements = math.prod(values.shape[:-1]) * right.shape[0]
        if elements > 16_000_000 or elements * right.shape[1] > 2_000_000_000:
            raise ValueError("Linear capture exceeds the CPU reference budget")
        expected = values @ right.T
        if prefix.with_name(prefix.name + "_input_2.bin").exists():
            bias = load("input_2")
            if bias.shape != (right.shape[0],):
                raise ValueError("Linear bias must match the output channels")
            expected += bias
    elif operation == "rotary":
        if values.shape[-1] % 2 or right.shape[-1] != values.shape[-1] or np.broadcast_shapes(values.shape, right.shape) != values.shape:
            raise ValueError("Rotary capture requires broadcast-compatible adjacent complex pairs")
        expected = np.empty_like(values)
        expected[..., 0::2] = values[..., 0::2] * right[..., 0::2] - values[..., 1::2] * right[..., 1::2]
        expected[..., 1::2] = values[..., 0::2] * right[..., 1::2] + values[..., 1::2] * right[..., 0::2]
    elif operation == "sdpa":
        third = load("input_2")
        if any(tensor.ndim != 4 for tensor in (values, right, third)):
            raise ValueError("SDPA capture requires NHWC rank-four tensors")
        if (values.shape[0] != right.shape[0] or right.shape[:3] != third.shape[:3]
                or values.shape[3] != right.shape[3] or values.shape[2] % right.shape[2]):
            raise ValueError("SDPA capture has incompatible batch/head dimensions")
        if (any(prefix.with_name(prefix.name + "_input_" + str(index) + ".bin").exists() for index in range(3, 8))
                or prefix.with_name(prefix.name + "_output_2.bin").exists()):
            raise ValueError("Masked or projected SDPA capture is not supported")
        if any(capture["format"] != 2 for capture in captures.values()):
            raise ValueError("SDPA capture requires NHWC tensor metadata")
        groups = values.shape[2] // right.shape[2]
        elements = values.shape[0] * values.shape[2] * values.shape[1] * right.shape[1]
        if (elements > 16_000_000 or elements * values.shape[3] > 2_000_000_000
                or max(right.size, third.size) * groups > 16_000_000):
            raise ValueError("SDPA capture exceeds the CPU reference budget")
        keys = np.repeat(right, groups, axis=2)
        third = np.repeat(third, groups, axis=2)
        scores = (values.transpose(0, 2, 1, 3) @ keys.transpose(0, 2, 3, 1)) * scale
        maximum = scores.max(axis=-1, keepdims=True)
        exponent = np.exp(scores - maximum)
        denominator = exponent.sum(axis=-1, keepdims=True)
        expected = ((exponent / denominator) @ third.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        if prefix.with_name(prefix.name + "_output_1.bin").exists():
            extra.append(("logsumexp", load("output_1"), (maximum + np.log(denominator)).squeeze(-1)))
    else:
        raise ValueError("Unknown capture operation: " + operation)
    comparisons = []
    for name, observed, reference in [("output", actual, expected)] + extra:
        if observed.shape != reference.shape or not np.isfinite(reference).all():
            raise ValueError("Capture output shape mismatch or non-finite reference: " + name)
        difference = np.abs(observed - reference)
        comparisons.append({"name": name, "max_absolute_error": float(difference.max()),
                            "mean_absolute_error": float(difference.mean()),
                            "matches": bool(np.allclose(observed, reference, atol=atol, rtol=rtol))})
    return {"operation": operation, "scope": "captured_operation_not_full_model_parity",
            "result": "PASS" if all(check["matches"] for check in comparisons) else "FAIL",
            "atol": atol, "rtol": rtol, "scale": scale, "captures": captures, "comparisons": comparisons}


def inspect_capture_tensor(path: Path) -> Dict[str, Any]:
    """Inspect a raw 68-byte-header native capture (.bin) for distribution, token similarity, and 2x2 grid artifacts."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Capture file not found: {resolved}")
    with resolved.open("rb") as stream:
        blob = stream.read(64 * 1024 * 1024 + 69)
    if len(blob) > 64 * 1024 * 1024 + 68:
        raise ValueError("Capture exceeds the 64 MiB payload limit")
    header = _read_tensor_header(blob)
    if header.identifier != 0 or header.datatype not in (16384, 131072, 524288) or not header.dims:
        raise ValueError("Capture requires a nonempty raw floating-point tensor")
    dimensions = struct.unpack("<12I", blob[20:68])
    if dimensions != header.dims + (0,) * (12 - len(header.dims)):
        raise ValueError("Capture dimensions must have contiguous positive axes and zero padding")
    values = _tensor_to_numpy(blob, header, math.prod(header.dims)).reshape(header.dims).astype(np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        raise ValueError(
            f"Capture contains non-finite values: nan={int(np.isnan(values).sum())}, inf={int(np.isinf(values).sum())}"
        )

    flat = values.reshape(-1)
    quantiles = np.quantile(flat, [0.01, 0.5, 0.99]).tolist()
    report: Dict[str, Any] = {
        "file": str(resolved),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "shape": list(values.shape),
        "format": header.tensor_format,
        "datatype": header.datatype,
        "count": int(values.size),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "l2_norm": float(np.linalg.norm(flat)),
        "quantiles_p01_p50_p99": [float(q) for q in quantiles],
        "zero_fraction": float(np.mean(flat == 0.0)),
    }

    # If 2D/3D sequence-like [..., T, C], compute adjacent token cosine similarity to detect token collapse
    if values.ndim in (2, 3) and values.shape[-2] >= 2 and values.shape[-1] >= 2:
        seq = values.reshape(-1, values.shape[-2], values.shape[-1])[0]
        norms = np.linalg.norm(seq, axis=-1)
        valid = norms > 1e-12
        if np.all(valid):
            normalized = seq / norms[:, None]
            adj_cos = np.sum(normalized[:-1] * normalized[1:], axis=-1)
            report["sequence_diagnostics"] = {
                "tokens": int(seq.shape[0]),
                "channels": int(seq.shape[1]),
                "token_norm_min": float(norms.min()),
                "token_norm_max": float(norms.max()),
                "adjacent_token_cosine_mean": float(adj_cos.mean()),
                "adjacent_token_cosine_max": float(adj_cos.max()),
            }

    # If 4D spatial NHWC [B, H, W, C] with H, W >= 4, check 2x2 checkerboard/unpatchify periodicity
    if values.ndim == 4 and values.shape[1] >= 4 and values.shape[2] >= 4:
        h_adj = float(np.mean(np.abs(values[:, 1:, :, :] - values[:, :-1, :, :])))
        w_adj = float(np.mean(np.abs(values[:, :, 1:, :] - values[:, :, :-1, :])))
        h_stride2 = float(np.mean(np.abs(values[:, 2:, :, :] - values[:, :-2, :, :])))
        w_stride2 = float(np.mean(np.abs(values[:, :, 2:, :] - values[:, :, :-2, :])))
        report["spatial_periodicity_2x2"] = {
            "height_diff_stride1": h_adj,
            "height_diff_stride2": h_stride2,
            "width_diff_stride1": w_adj,
            "width_diff_stride2": w_stride2,
            "checkerboard_ratio_h": h_adj / max(h_stride2, 1e-12),
            "checkerboard_ratio_w": w_adj / max(w_stride2, 1e-12),
        }

    return report


def native_backend_report(binary: Path, source: Path, execute_qwen: bool = False,
                          execute_sdpa: bool = False, execute_embedding: bool = False,
                          execute_packed_gemm: bool = False) -> Dict[str, Any]:
    compiler, debugger = shutil.which("cc"), shutil.which("lldb")
    if not compiler or not debugger or not binary.is_file():
        raise ValueError("Native backend inspection requires a compiler, LLDB, and candidate executable")
    checks = [(operation, datatype, 2, code) for operation, code in (
        ("gated_delta", 0x6de198ac), ("rmsnorm_gated", 0x9b8af346))
        for datatype in (16384, 131072, 524288)]
    checks += [("sdpa_control", 131072, 2, 0x284ed926),
               ("gated_delta_cpu_control", 16384, 1, 0x6de198ac),
               ("rmsnorm_cpu_control", 16384, 1, 0x9b8af346)]
    with tempfile.TemporaryDirectory() as temporary:
        library = Path(temporary) / "backend_probe.so"
        subprocess.run([compiler, "-shared", "-fPIC", "-g", "-O2", "-I", str(source / "lib"),
                        str(Path(__file__).with_name("dt_qwen35_cpu_probe.c")), "-lm", "-o", str(library)],
                       check=True, capture_output=True, text=True, timeout=60)
        commands = ["settings set target.disable-aslr false", "breakpoint set --name main", "run",
                "breakpoint disable 1",
                    f'expression -- ((void *(*)(const char *, int))dlopen)({json.dumps(str(library))}, 2)',
                    'script addresses = [lldb.target.FindSymbols(name).GetContextAtIndex(0).GetSymbol().GetStartAddress().GetLoadAddress(lldb.target) for name in ["ccv_nnc_init", "ccv_nnc_cmd_find_backend"]]',
                    'script assert all(address != lldb.LLDB_INVALID_ADDRESS for address in addresses), "Native backend symbol missing"',
                    'script result = lldb.frame.EvaluateExpression("dt_probe_native_init((void*)%d)" % addresses[0]); assert result.GetError().Success(), str(result.GetError())',
                    f'script checks = {checks!r}; selected = []',
                    'script [selected.append(lldb.frame.EvaluateExpression("dt_probe_native_backend((void*)%d,%dU,%d,%d)" % (addresses[1], code, memory, datatype))) for operation, datatype, memory, code in checks]',
                    'script assert all(result.GetError().Success() for result in selected), str([str(result.GetError()) for result in selected])',
                    'script print("DT_BACKENDS=" + __import__("json").dumps([result.GetValueAsUnsigned(0) for result in selected]))',
                    "process kill"]
        if execute_qwen or execute_sdpa or execute_embedding or execute_packed_gemm:
            commands[-1:-1] = [
                'script execution_addresses = [lldb.target.FindSymbols(name).GetContextAtIndex(0).GetSymbol().GetStartAddress().GetLoadAddress(lldb.target) for name in ["ccv_nnc_tensor_new", "ccv_nnc_tensor_free", "ccv_nnc_cmd_exec"]]',
                'script execution_options = lldb.SBExpressionOptions(); execution_options.SetLanguage(lldb.eLanguageTypeC); execution_options.SetIgnoreBreakpoints(True); execution_options.SetStopOthers(False); execution_options.SetUnwindOnError(False); execution_options.SetTimeoutInMicroSeconds(60000000)'
            ]
        if execute_qwen:
            commands[-1:-1] = [
                'script execution_ready = all(address != lldb.LLDB_INVALID_ADDRESS for address in execution_addresses) and all(result.GetError().Success() and result.GetValueAsUnsigned(0) == 0x5f19790a for result in selected[:6]); execution = lldb.frame.EvaluateExpression("dt_probe_native_qwen_execution(" + ",".join("(void*)%d" % address for address in execution_addresses) + ")", execution_options) if execution_ready else None; print("DT_QWEN_EXECUTION=" + str(execution.GetValueAsSigned(-99) if execution is not None and execution.GetError().Success() else -98))',
                'script print("DT_QWEN_EXECUTION_ERROR=" + (str(execution.GetError()) if execution is not None else "CUDA registrations or symbols missing"))'
            ]
        if execute_sdpa:
            commands[-1:-1] = [
                'script sdpa_forward = lldb.target.FindSymbols("_ccv_nnc_scaled_dot_product_attention_forw").GetContextAtIndex(0).GetSymbol().GetStartAddress().GetLoadAddress(lldb.target)',
                'script sdpa_ready = sdpa_forward != lldb.LLDB_INVALID_ADDRESS and all(address != lldb.LLDB_INVALID_ADDRESS for address in execution_addresses); sdpa_contracts = [lldb.frame.EvaluateExpression("dt_probe_native_sdpa_contract((void*)%d,%d)" % (sdpa_forward, mode), execution_options) for mode in range(8)] if sdpa_ready else []; sdpa_statuses = [result.GetValueAsSigned(-99) if result.GetError().Success() else -99 for result in sdpa_contracts]; print("DT_SDPA_CONTRACTS=" + __import__("json").dumps(sdpa_statuses)); print("DT_SDPA_ERRORS=" + str([str(result.GetError()) for result in sdpa_contracts]))',
                'script sdpa_execution = lldb.frame.EvaluateExpression("dt_probe_native_sdpa_execution(" + ",".join("(void*)%d" % address for address in execution_addresses) + ")", execution_options) if sdpa_statuses == [0,-1,-1,-1,-1,-1,0,-1] else None; print("DT_SDPA_EXECUTION=" + str(sdpa_execution.GetValueAsSigned(-99) if sdpa_execution is not None and sdpa_execution.GetError().Success() else -98)); print("DT_SDPA_EXECUTION_ERROR=" + (str(sdpa_execution.GetError()) if sdpa_execution is not None else "SDPA contracts failed"))'
            ]
        if execute_embedding or execute_packed_gemm:
            commands[-1:-1] = [
                'script packing_addresses = [lldb.target.FindSymbols(name).GetContextAtIndex(0).GetSymbol().GetStartAddress().GetLoadAddress(lldb.target) for name in ["ccv_nnc_quantize_8i_rowwise_x", "ccv_nnc_dequantize_8i_rowwise_x"]]'
            ]
        if execute_embedding:
            commands[-1:-1] = [
                'script embedding_ready = all(address != lldb.LLDB_INVALID_ADDRESS for address in execution_addresses + packing_addresses) and addresses[1] != lldb.LLDB_INVALID_ADDRESS; embedding = lldb.frame.EvaluateExpression("dt_probe_native_embedding_execution(" + ",".join("(void*)%d" % address for address in execution_addresses + [addresses[1]] + packing_addresses) + ")", execution_options) if embedding_ready else None; print("DT_EMBEDDING_EXECUTION=" + str(embedding.GetValueAsSigned(-99) if embedding is not None and embedding.GetError().Success() else -98)); print("DT_EMBEDDING_ERROR=" + (str(embedding.GetError()) if embedding is not None else "Native symbols missing"))'
            ]
        if execute_packed_gemm:
            commands[-1:-1] = [
                'script packed_ready = all(address != lldb.LLDB_INVALID_ADDRESS for address in execution_addresses + packing_addresses); packed = lldb.frame.EvaluateExpression("dt_probe_native_packed_gemm(" + ",".join("(void*)%d" % address for address in execution_addresses + packing_addresses) + ")", execution_options) if packed_ready else None; print("DT_PACKED_GEMM_EXECUTION=" + str(packed.GetValueAsSigned(-99) if packed is not None and packed.GetError().Success() else -98)); print("DT_PACKED_GEMM_ERROR=" + (str(packed.GetError()) if packed is not None else "Native symbols missing"))'
            ]
        invocation = [debugger, "--batch"]
        for command in commands:
            invocation.extend(["-o", command])
        invocation.extend(["--", str(binary), "--help"])
        result = subprocess.run(invocation, capture_output=True, text=True, timeout=120)
        match = re.search(r"^DT_BACKENDS=(\[.*\])$", result.stdout, re.M)
        if result.returncode or match is None:
            raise ValueError("Native backend inspection failed: " + result.stdout[-6000:] + result.stderr[-2000:])
        selected = json.loads(match.group(1))
        execution_match = re.search(r"^DT_QWEN_EXECUTION=(-?\d+)$", result.stdout, re.M)
        execution_status = int(execution_match.group(1)) if execution_match else None
        if execute_qwen and execution_status != 0:
            raise ValueError("Native Qwen execution did not pass: " + result.stdout[-6000:])
        sdpa_contract_match = re.search(r"^DT_SDPA_CONTRACTS=(\[.*\])$", result.stdout, re.M)
        sdpa_contracts = json.loads(sdpa_contract_match.group(1)) if sdpa_contract_match else None
        sdpa_execution_match = re.search(r"^DT_SDPA_EXECUTION=(-?\d+)$", result.stdout, re.M)
        sdpa_status = int(sdpa_execution_match.group(1)) if sdpa_execution_match else None
        if execute_sdpa and (sdpa_contracts != [0, -1, -1, -1, -1, -1, 0, -1] or sdpa_status != 0):
            raise ValueError("Native SDPA validation did not pass: " + result.stdout[-10000:])
        embedding_match = re.search(r"^DT_EMBEDDING_EXECUTION=(-?\d+)$", result.stdout, re.M)
        embedding_status = int(embedding_match.group(1)) if embedding_match else None
        if execute_embedding and embedding_status != 0:
            raise ValueError("Native packed embedding validation did not pass: " + result.stdout[-6000:])
        packed_match = re.search(r"^DT_PACKED_GEMM_EXECUTION=(-?\d+)$", result.stdout, re.M)
        packed_status = int(packed_match.group(1)) if packed_match else None
        if execute_packed_gemm and packed_status != 0:
            raise ValueError("Native packed GEMM validation did not pass: " + result.stdout[-10000:])
    expected = [0x5f19790a] * 7 + [0x3d9883e5] * 2
    return {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "qwen_cuda_registered": selected == expected,
            "qwen_cuda_execution": execution_status,
            "sdpa_contracts": sdpa_contracts, "sdpa_cuda_execution": sdpa_status,
            "packed_embedding_cuda_execution": embedding_status,
            "packed_gemm_cuda_execution": packed_status,
            "checks": [{"operation": operation, "datatype": datatype, "memory": memory,
                        "backend": hex(backend), "expected": hex(wanted)}
                       for (operation, datatype, memory, _), backend, wanted in zip(checks, selected, expected)]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WeeLLM vs Draw Things Layer Parity Oracle")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    p_capture = sub.add_parser("compare-capture", help="Compare captured native linear, rotary, or unmasked SDPA operations with CPU math")
    p_capture.add_argument("--prefix", required=True, type=Path)
    p_capture.add_argument("--operation", required=True, choices=("linear", "rotary", "sdpa"))
    p_capture.add_argument("--atol", type=float, default=0.001)
    p_capture.add_argument("--rtol", type=float, default=0.001)
    p_capture.add_argument("--scale", type=float, default=1.0)
    p_capture.add_argument("--report", type=Path)
    p_backends = sub.add_parser("native-backends", help="Inspect actual compiled CUDA/CPU backend selections under LLDB")
    p_backends.add_argument("--binary", required=True, type=Path)
    p_backends.add_argument("--ccv-source", required=True, type=Path)
    p_backends.add_argument("--report", type=Path)
    p_backends.add_argument("--require-qwen-cuda", action="store_true")
    p_backends.add_argument("--execute-qwen", action="store_true", help="Run linked-runtime CUDA operators and compare with native CPU values")
    p_backends.add_argument("--execute-sdpa", action="store_true", help="Check empty SDPA contracts and a nonempty grouped FP16 CUDA operation")
    p_backends.add_argument("--execute-embedding", action="store_true", help="Validate packed Q5_K/Q6_K row lookup across F32/F16/BF16")
    p_backends.add_argument("--execute-packed-gemm", action="store_true", help="Compare packed Q5_K/Q6_K FP16 matrix multiplication with native CPU decoding")

    p_fix = sub.add_parser(
        "qwen35-linear-attn-fixture",
        help="Generate golden NPZ fixtures for Qwen3.5 GatedDelta (0x6de198ac) & RMSNormGated (0x9b8af346)",
    )
    p_fix.add_argument("--out-dir", required=True)
    p_fix.add_argument("--seed", type=int, default=4242)
    p_fix.add_argument("--native-ccv-source", default="", help="Compile pinned CCV CPU operators and validate the generated fixtures")

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

    p_cap_stat = sub.add_parser(
        "inspect-capture",
        help="Inspect a single captured native .bin tensor for distribution, token collapse, and 2x2 grid periodicity",
    )
    p_cap_stat.add_argument("--file", required=True, type=Path)
    p_cap_stat.add_argument("--report", type=Path)

    args = parser.parse_args(argv)
    if args.subcommand == "inspect-capture":
        report = inspect_capture_tensor(args.file)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.subcommand == "compare-capture":
        report = compare_native_capture(args.prefix, args.operation, args.atol, args.rtol, args.scale)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["result"] == "PASS" else 1
    if args.subcommand == "native-backends":
        report = native_backend_report(args.binary.resolve(), args.ccv_source.resolve(), args.execute_qwen, args.execute_sdpa, args.execute_embedding, args.execute_packed_gemm)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["qwen_cuda_registered"] or not args.require_qwen_cuda else 1
    if args.subcommand == "qwen35-linear-attn-fixture":
        return cmd_generate_qwen35_fixtures(args)
    if args.subcommand == "verify-block":
        return cmd_verify_block(args)
    if args.subcommand == "inspect-ckpt-health":
        return cmd_inspect_health(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
