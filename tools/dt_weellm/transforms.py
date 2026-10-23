"""Deterministic CPU NumPy transformations matching Draw Things TensorDescriptor.swift and s4nnc."""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

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
) -> List[np.ndarray]:
    """Split a fused weight/modulation tensor along `axis` matching `TensorDescriptor.swift`.

    When `count > 1` in `TensorDescriptor.swift`, unit dimensions are squeezed before slicing
    along axis 0 (for `.O` format) or axis 1 (for `.I` format).
    """
    if count <= 1:
        return [np.ascontiguousarray(arr)]

    work = arr
    if squeeze_unit_dims and work.ndim > 1:
        squeezed_shape = [d for d in work.shape if d != 1]
        if not squeezed_shape:
            squeezed_shape = [1]
        work = work.reshape(squeezed_shape)

    if axis < 0 or axis >= work.ndim:
        raise ValueError(f"Invalid split axis {axis} for shape {work.shape}")

    dim_len = work.shape[axis]
    slices: List[Tuple[int, int]] = []
    if offsets is not None:
        if len(offsets) != count:
            raise ValueError(f"Expected {count} offsets, got {len(offsets)}: {offsets}")
        for i in range(count):
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < count else dim_len
            if start < 0 or end < start or end > dim_len:
                raise ValueError(f"Invalid slice [{start}:{end}] for axis length {dim_len}")
            slices.append((start, end))
    else:
        if dim_len % count != 0:
            raise ValueError(
                f"Dimension length {dim_len} along axis {axis} is not divisible by count {count}"
            )
        step = dim_len // count
        for i in range(count):
            slices.append((i * step, (i + 1) * step))

    chunks: List[np.ndarray] = []
    for start, end in slices:
        idx = [slice(None)] * work.ndim
        idx[axis] = slice(start, end)
        chunks.append(np.ascontiguousarray(work[tuple(idx)]))
    return chunks


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
    # Round-to-nearest-even for BFloat16 ( preserving NaN payload bits )
    nan_mask = (u32 & 0x7F800000) == 0x7F800000 and_not_zero(u32 & 0x007FFFFF)
    lsb = (u32 >> 16) & 1
    rounding_bias = 0x00007FFF + lsb
    rounded = np.where(nan_mask, u32 | 0x00400000, u32 + rounding_bias)
    u16 = (rounded >> 16).astype("<u2")
    return u16.tobytes()


def and_not_zero(arr: np.ndarray) -> np.ndarray:
    return arr != 0


def tensor_to_ckpt_bytes(arr_f32: np.ndarray, target_datatype: int) -> bytes:
    """Serialize a float32 NumPy array into raw `s4nnc` inline tensor bytes for `target_datatype`."""
    if target_datatype == DT_FLOAT32:
        return np.ascontiguousarray(arr_f32, dtype="<f4").tobytes()
    if target_datatype == DT_FLOAT16:
        return np.ascontiguousarray(arr_f32, dtype="<f2").tobytes()
    if target_datatype == DT_BFLOAT16:
        return f32_to_bf16_bytes(arr_f32)
    raise ValueError(f"Unsupported target_datatype code: {target_datatype}")
