"""Deterministic CPU NumPy transformations matching Draw Things TensorDescriptor.swift and s4nnc."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# s4nnc / ccv datatype codes stored in SQLite `tensors.datatype`
DT_FLOAT32 = 16384    # 0x4000 (CCV_32F)
DT_FLOAT16 = 131072   # 0x20000 (CCV_16F)
DT_BFLOAT16 = 524288  # 0x80000 (CCV_16BF)

# s4nnc / ccv memory type codes stored in SQLite `tensors.type`
DT_TYPE_CPU = 1
DT_TYPE_GPU = 258

# s4nnc / ccv tensor format codes stored in SQLite `tensors.format`
DT_FORMAT_NCHW = 1
DT_FORMAT_NHWC = 2

DATATYPE_NAME_TO_CODE = {
    "F32": DT_FLOAT32,
    "FLOAT32": DT_FLOAT32,
    "F16": DT_FLOAT16,
    "FLOAT16": DT_FLOAT16,
    "BF16": DT_BFLOAT16,
    "BFLOAT16": DT_BFLOAT16,
}

DATATYPE_CODE_TO_NAME = {
    DT_FLOAT32: "F32",
    DT_FLOAT16: "F16",
    DT_BFLOAT16: "BF16",
}


def rotary_deinterleave(arr: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Permute axis 0 from interleaved rotary layout to split-half rotary layout.

    Exact CPU NumPy equivalent of `TensorDescriptor.swift` (§17 & §18):
        out[h * hd + d * 2 + p] = in[h * hd + p * (hd / 2) + d]

    Works for 2-D projection weights `[num_heads * head_dim, in_dim]`,
    1-D biases `[num_heads * head_dim]`, and 1-D QK-norm weights `[num_heads * head_dim]`.
    """
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError(f"num_heads ({num_heads}) and head_dim ({head_dim}) must be positive")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for rotary de-interleaving")
    if arr.ndim < 1:
        raise ValueError("Input array must be at least 1-D")

    out_dim = arr.shape[0]
    expected_out = num_heads * head_dim
    if out_dim != expected_out:
        raise ValueError(
            f"rotary_deinterleave axis-0 mismatch: got {out_dim}, expected {expected_out} "
            f"(num_heads={num_heads}, head_dim={head_dim})"
        )

    orig_shape = arr.shape
    cols = int(np.prod(orig_shape[1:])) if arr.ndim > 1 else 1
    half = head_dim // 2

    viewed = np.ascontiguousarray(arr).reshape(num_heads, 2, half, cols)
    permuted = np.transpose(viewed, (0, 2, 1, 3))
    return np.ascontiguousarray(permuted.reshape(orig_shape))


def rotary_interleave(arr: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Inverse of `rotary_deinterleave` (maps Draw Things split-half order back to HF interleaved)."""
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError(f"num_heads ({num_heads}) and head_dim ({head_dim}) must be positive")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for rotary interleaving")
    if arr.ndim < 1:
        raise ValueError("Input array must be at least 1-D")

    out_dim = arr.shape[0]
    expected_out = num_heads * head_dim
    if out_dim != expected_out:
        raise ValueError(
            f"rotary_interleave axis-0 mismatch: got {out_dim}, expected {expected_out} "
            f"(num_heads={num_heads}, head_dim={head_dim})"
        )

    orig_shape = arr.shape
    cols = int(np.prod(orig_shape[1:])) if arr.ndim > 1 else 1
    half = head_dim // 2

    viewed = np.ascontiguousarray(arr).reshape(num_heads, half, 2, cols)
    permuted = np.transpose(viewed, (0, 2, 1, 3))
    return np.ascontiguousarray(permuted.reshape(orig_shape))


def split_tensor_chunks(
    arr: np.ndarray,
    count: int,
    axis: int = 0,
    offsets: Optional[Sequence[int]] = None,
    squeeze_unit_dims: bool = True,
    split_policy: Optional[str] = None,
) -> List[np.ndarray]:
    """Split a fused weight/modulation tensor matching `TensorDescriptor.swift` lines 112-260.

    Supports:
      - `split_policy=None` or `"chunk_o"`: split along axis 0 (or `axis`).
      - `split_policy="chunk_i"`: split along axis 1 (when 2-D+) or axis 0 (when 1-D).
      - `split_policy="diagonal_up"` / `"diagonal_down"`: block-diagonal extraction along both
        axis 0 and axis 1 (`[i*c0:(i+1)*c0, i*c1:(i+1)*c1]`).
      - `offsets`: explicit start offsets along the split axis.
    """
    if count <= 1:
        return [np.ascontiguousarray(arr)]

    work = arr
    if squeeze_unit_dims and work.ndim > 1:
        squeezed_shape = [d for d in work.shape if d != 1]
        if not squeezed_shape:
            squeezed_shape = [1]
        work = work.reshape(squeezed_shape)

    if split_policy in ("diagonal_up", "diagonal_down") and work.ndim >= 2:
        rows, cols = work.shape[0], work.shape[1]
        j_step = cols // count
        r_slices: List[Tuple[int, int]] = []
        if offsets is not None:
            if len(offsets) != count:
                raise ValueError(f"Expected {count} offsets, got {len(offsets)}")
            for i in range(count):
                r_start = offsets[i]
                r_end = offsets[i + 1] if i + 1 < count else rows
                r_slices.append((r_start, r_end))
        else:
            r_step = rows // count
            for i in range(count):
                r_slices.append((i * r_step, (i + 1) * r_step))

        diag_chunks: List[np.ndarray] = []
        for i, (r0, r1) in enumerate(r_slices):
            c0, c1 = i * j_step, (i + 1) * j_step
            diag_chunks.append(np.ascontiguousarray(work[r0:r1, c0:c1]))
        return diag_chunks

    effective_axis = axis
    if split_policy == "chunk_i":
        effective_axis = 1 if work.ndim > 1 else 0
    elif split_policy == "chunk_o":
        effective_axis = 0

    if effective_axis < 0 or effective_axis >= work.ndim:
        raise ValueError(f"Invalid split axis {effective_axis} for shape {work.shape}")

    dim_len = work.shape[effective_axis]
    slices: List[Tuple[int, int]] = []
    if offsets is not None:
        if len(offsets) != count:
            raise ValueError(f"Expected {count} offsets, got {len(offsets)}: {offsets}")
        for i in range(count):
            start = int(offsets[i])
            end = int(offsets[i + 1]) if i + 1 < count else dim_len
            if start < 0 or end < start or end > dim_len:
                raise ValueError(f"Invalid slice [{start}:{end}] for axis length {dim_len}")
            slices.append((start, end))
    else:
        if dim_len % count != 0:
            raise ValueError(
                f"Dimension length {dim_len} along axis {effective_axis} is not divisible by count {count}"
            )
        step = dim_len // count
        for i in range(count):
            slices.append((i * step, (i + 1) * step))

    chunks: List[np.ndarray] = []
    for start, end in slices:
        idx = [slice(None)] * work.ndim
        idx[effective_axis] = slice(start, end)
        chunks.append(np.ascontiguousarray(work[tuple(idx)]))
    return chunks


def apply_contract_rule_to_tensor(
    arr_f32: np.ndarray,
    rule: Dict[str, Any],
) -> List[Tuple[str, np.ndarray]]:
    """Apply scaling, rotary de-interleaving, and chunk splitting according to a contract rule.

    Handles both:
      1. Single/pre-split rotary de-interleaving (`rotary_interleaved: True`).
      2. Fused QKV splitting (`dt_names` has length > 1 and `rotary_interleaved_slices` is e.g.
         `[True, True, False]`), where Q and K slices are rotary-deinterleaved after splitting
         while V remains in identity order.
    """
    work = np.ascontiguousarray(arr_f32, dtype=np.float32)
    scale = float(rule.get("scale", 1.0))
    if scale != 1.0:
        work = work * scale

    dt_names: List[str] = list(rule["dt_names"])
    rotary_slices: Optional[List[bool]] = rule.get("rotary_interleaved_slices")
    heads = int(rule.get("num_heads", 0))
    head_dim = int(rule.get("head_dim", 0))

    # If the whole tensor is rotary-interleaved (and not a mixed fused QKV tensor), deinterleave first
    if rule.get("rotary_interleaved", False) and not rotary_slices:
        work = rotary_deinterleave(work, num_heads=heads, head_dim=head_dim)

    if len(dt_names) > 1:
        split_axis = int(rule.get("split_axis", 0))
        split_policy = rule.get("split_policy")
        offsets = rule.get("offsets")
        squeeze = bool(rule.get("squeeze_unit_dims", True))
        chunks = split_tensor_chunks(
            work,
            count=len(dt_names),
            axis=split_axis,
            offsets=offsets,
            squeeze_unit_dims=squeeze,
            split_policy=split_policy,
        )
    else:
        chunks = [work]

    if rotary_slices:
        if len(rotary_slices) != len(chunks):
            raise ValueError(
                f"rotary_interleaved_slices length ({len(rotary_slices)}) != chunk count ({len(chunks)})"
            )
        processed_chunks: List[np.ndarray] = []
        for chunk, should_deinterleave in zip(chunks, rotary_slices):
            if should_deinterleave:
                processed_chunks.append(
                    rotary_deinterleave(chunk, num_heads=heads, head_dim=head_dim)
                )
            else:
                processed_chunks.append(chunk)
        chunks = processed_chunks

    return list(zip(dt_names, chunks))


def encode_dim_blob(
    shape: Sequence[int],
    ndim_slots: int = 12,
    rank_policy: str = "preserve",
) -> bytes:
    """Encode a tensor shape into an `s4nnc` SQLite `dim` blob (default 12x int32 = 48 bytes).

    `rank_policy`:
      - `"preserve"`: writes `shape` followed by zeros.
      - `"norm_1_1_n"`: enforces the §22 rank-3 `[1, 1, N]` layout required by Draw Things'
        RMSNorm and AdaLN modulation loaders.
    """
    if ndim_slots < 3:
        raise ValueError(f"ndim_slots must be >= 3, got {ndim_slots}")

    clean_shape = [int(d) for d in shape if int(d) > 0]
    if not clean_shape:
        clean_shape = [1]

    if rank_policy == "norm_1_1_n":
        n_elem = int(np.prod(clean_shape))
        active_dims = [1, 1, n_elem]
    elif rank_policy == "preserve":
        active_dims = clean_shape
    else:
        raise ValueError(f"Unknown rank_policy: {rank_policy!r}")

    if len(active_dims) > ndim_slots:
        raise ValueError(f"Shape {active_dims} exceeds ndim_slots={ndim_slots}")

    padded = active_dims + [0] * (ndim_slots - len(active_dims))
    return struct.pack(f"<{ndim_slots}i", *padded)


def decode_dim_blob(dim_blob: bytes) -> Tuple[List[int], int]:
    """Decode an `s4nnc` SQLite `dim` blob into `(non_zero_dims, ndim_slots)`."""
    if len(dim_blob) % 4 != 0 or len(dim_blob) == 0:
        raise ValueError(f"Invalid dim_blob byte length: {len(dim_blob)}")
    ndim_slots = len(dim_blob) // 4
    raw = struct.unpack(f"<{ndim_slots}i", dim_blob)
    active = [int(d) for d in raw if d > 0]
    return active, ndim_slots


def bf16_bytes_to_f32(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Convert raw little-endian BFloat16 bytes into a float32 NumPy array of `shape`."""
    u16 = np.frombuffer(raw_bytes, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.reshape(shape).copy()


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Convert a float32/float16 NumPy array into raw little-endian BFloat16 bytes (round-to-nearest-even)."""
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    nan_mask = ((u32 & 0x7F800000) == 0x7F800000) & ((u32 & 0x007FFFFF) != 0)
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x00007FFF) + lsb
    rounded = np.where(nan_mask, u32 | np.uint32(0x00400000), u32 + rounding_bias)
    u16 = (rounded >> 16).astype("<u2")
    return u16.tobytes()


def tensor_to_ckpt_bytes(arr_f32: np.ndarray, target_datatype: int) -> bytes:
    """Serialize a float32 NumPy array into raw `s4nnc` inline tensor bytes for `target_datatype`."""
    if target_datatype == DT_FLOAT32:
        return np.ascontiguousarray(arr_f32, dtype="<f4").tobytes()
    if target_datatype == DT_FLOAT16:
        return np.ascontiguousarray(arr_f32, dtype="<f2").tobytes()
    if target_datatype == DT_BFLOAT16:
        return f32_to_bf16_bytes(arr_f32)
    raise ValueError(f"Unsupported target_datatype code: {target_datatype}")


def write_safetensors_file(
    output_path: str | Path,
    tensors: Dict[str, Tuple[str, Sequence[int], bytes]],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Write a standard `.safetensors` file from `{key: (dtype_str, shape, raw_bytes)}`."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    blobs: List[bytes] = []
    offset = 0
    for name in sorted(tensors.keys()):
        dtype_str, shape, raw_bytes = tensors[name]
        end = offset + len(raw_bytes)
        header[name] = {
            "dtype": str(dtype_str).upper(),
            "shape": [int(d) for d in shape],
            "data_offsets": [offset, end],
        }
        blobs.append(raw_bytes)
        offset = end

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - (len(header_json) % 8)) % 8
    if pad:
        header_json += b" " * pad

    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for b in blobs:
            f.write(b)
