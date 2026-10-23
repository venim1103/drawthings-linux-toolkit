"""Pure-NumPy dequantizers for GGUF blocks and ComfyUI FP8 tensors adapted from WeeLLM."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .transforms import bf16_bytes_to_f32

# GGML type constants matching seek_io.py
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30


def dequantize_fp8_e4m3(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Convert raw FP8 E4M3FN bytes into float32 (IEEE-like 1 sign, 4 exp, 3 mantissa, bias=7)."""
    u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    sign = ((u8 >> 7) & 1).astype(np.float32)
    exp = ((u8 >> 3) & 0x0F).astype(np.int32)
    mant = (u8 & 0x07).astype(np.float32)

    sign_factor = 1.0 - 2.0 * sign
    # Subnormal: exp == 0 -> (-1)^s * 2^-6 * (mant / 8)
    subnormal = sign_factor * (mant / 8.0) * (2.0 ** -6)
    # Normal: 0 < exp < 15 or (exp == 15 and mant < 7)
    normal = sign_factor * (1.0 + mant / 8.0) * np.power(2.0, exp - 7, dtype=np.float32)
    out = np.where(exp == 0, subnormal, normal)
    # E4M3FN NaN is 0x7F or 0xFF (exp == 15 and mant == 7)
    out = np.where((exp == 15) & (u8 & 0x07 == 7), np.nan, out)
    return out.astype(np.float32).reshape(shape)


def dequantize_fp8_e5m2(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Convert raw FP8 E5M2 bytes into float32 (same exponent layout as IEEE FP16, so shift left by 8)."""
    u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    u16 = (u8.astype(np.uint16)) << 8
    f16 = u16.view(np.float16)
    return f16.astype(np.float32).reshape(shape)


def dequantize_gguf_q4_0(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q4_0 blocks (32 elements per 18-byte block: 2B fp16 d + 16B nibbles)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)  # [n_blocks, 1]
    qs = blocks[:, 2:]  # [n_blocks, 16]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    combined = np.concatenate([lo, hi], axis=1).astype(np.float32)  # [n_blocks, 32]
    out = combined * d
    return out.reshape(shape)


def dequantize_gguf_q8_0(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q8_0 blocks (32 elements per 34-byte block: 2B fp16 d + 32B int8)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 34)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)  # [n_blocks, 1]
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)  # [n_blocks, 32]
    out = qs * d
    return out.reshape(shape)


def dequantize_gguf_q4_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q4_K super-blocks (256 elements per 144-byte block) adapted from WeeLLM."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 144)

    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)  # [n_blocks, 1]
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)  # [n_blocks, 1]
    scales = blocks[:, 4:16]  # [n_blocks, 12]
    qs = blocks[:, 16:144]  # [n_blocks, 128]

    # Unpack 8 sub-block scales and mins (6-bit packed into 12 bytes)
    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    m = np.zeros((n_blocks, 8), dtype=np.float32)
    sc[:, 0:4] = scales[:, 0:4] & 63
    m[:, 0:4] = scales[:, 4:8] & 63
    sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)

    out = np.zeros((n_blocks, 256), dtype=np.float32)
    for j in range(4):
        q_chunk = qs[:, j * 32 : (j + 1) * 32]
        q_lo = (q_chunk & 0x0F).astype(np.float32)
        q_hi = (q_chunk >> 4).astype(np.float32)
        d1 = d[:, 0:1] * sc[:, 2 * j : 2 * j + 1]
        m1 = dmin[:, 0:1] * m[:, 2 * j : 2 * j + 1]
        d2 = d[:, 0:1] * sc[:, 2 * j + 1 : 2 * j + 2]
        m2 = dmin[:, 0:1] * m[:, 2 * j + 1 : 2 * j + 2]
        out[:, j * 64 : j * 64 + 32] = d1 * q_lo - m1
        out[:, j * 64 + 32 : (j + 1) * 64] = d2 * q_hi - m2

    return out.reshape(shape)


def dequantize_gguf_q6_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q6_K super-blocks (256 elements per 210-byte block) adapted from WeeLLM."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 210)

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view("<f2").astype(np.float32)

    out = np.zeros((n_blocks, 256), dtype=np.float32)
    for n in range(2):
        ql_part = ql[:, n * 64 : (n + 1) * 64]
        qh_part = qh[:, n * 32 : (n + 1) * 32]
        sc_part = scales[:, n * 8 : (n + 1) * 8]

        q1 = ((ql_part[:, 0:32] & 0x0F) | (((qh_part >> 0) & 3) << 4)).astype(np.int8) - 32
        q2 = ((ql_part[:, 32:64] & 0x0F) | (((qh_part >> 2) & 3) << 4)).astype(np.int8) - 32
        q3 = ((ql_part[:, 0:32] >> 4) | (((qh_part >> 4) & 3) << 4)).astype(np.int8) - 32
        q4 = ((ql_part[:, 32:64] >> 4) | (((qh_part >> 6) & 3) << 4)).astype(np.int8) - 32

        base = n * 128
        for l_idx in range(2):
            s = slice(l_idx * 16, (l_idx + 1) * 16)
            out[:, base + 0 + l_idx * 16 : base + 16 + l_idx * 16] = (
                d * sc_part[:, l_idx + 0 : l_idx + 1] * q1[:, s]
            )
            out[:, base + 32 + l_idx * 16 : base + 48 + l_idx * 16] = (
                d * sc_part[:, l_idx + 2 : l_idx + 3] * q2[:, s]
            )
            out[:, base + 64 + l_idx * 16 : base + 80 + l_idx * 16] = (
                d * sc_part[:, l_idx + 4 : l_idx + 5] * q3[:, s]
            )
            out[:, base + 96 + l_idx * 16 : base + 112 + l_idx * 16] = (
                d * sc_part[:, l_idx + 6 : l_idx + 7] * q4[:, s]
            )

    return out.reshape(shape)


def dequantize_gguf_tensor(
    raw_bytes: bytes, ggml_type: Optional[int], shape: Sequence[int]
) -> np.ndarray:
    """Dispatch GGUF tensor block dequantization to float32."""
    if ggml_type == GGML_TYPE_F32:
        return np.frombuffer(raw_bytes, dtype="<f4").reshape(shape).copy()
    if ggml_type == GGML_TYPE_F16:
        return np.frombuffer(raw_bytes, dtype="<f2").astype(np.float32).reshape(shape)
    if ggml_type == GGML_TYPE_BF16:
        return bf16_bytes_to_f32(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q4_0:
        return dequantize_gguf_q4_0(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q8_0:
        return dequantize_gguf_q8_0(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q4_K:
        return dequantize_gguf_q4_k(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q6_K:
        return dequantize_gguf_q6_k(raw_bytes, shape)
    raise ValueError(f"Unsupported GGML quantization type code: {ggml_type}")
