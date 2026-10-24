"""Pure-NumPy dequantizers for all standard GGUF blocks and ComfyUI FP8 tensors adapted from WeeLLM."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .transforms import bf16_bytes_to_f32

# GGML type constants matching seek_io.py
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30


def dequantize_fp8_e4m3(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Convert raw FP8 E4M3FN bytes into float32 (IEEE-like 1 sign, 4 exp, 3 mantissa, bias=7)."""
    u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    sign = ((u8 >> 7) & 1).astype(np.float32)
    exp = ((u8 >> 3) & 0x0F).astype(np.int32)
    mant = (u8 & 0x07).astype(np.float32)

    sign_factor = 1.0 - 2.0 * sign
    subnormal = sign_factor * (mant / 8.0) * (2.0 ** -6)
    normal = sign_factor * (1.0 + mant / 8.0) * np.power(2.0, exp - 7, dtype=np.float32)
    out = np.where(exp == 0, subnormal, normal)
    # E4M3FN NaN is 0x7F or 0xFF (exp == 15 and mant == 7)
    out = np.where((exp == 15) & ((u8 & 0x07) == 7), np.nan, out)
    return out.astype(np.float32).reshape(shape)


def dequantize_fp8_e5m2(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Convert raw FP8 E5M2 bytes into float32 (same exponent layout as IEEE FP16, shift left by 8)."""
    u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    u16 = (u8.astype(np.uint16)) << 8
    f16 = u16.view(np.float16)
    return f16.astype(np.float32).reshape(shape)


def dequantize_gguf_q4_0(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q4_0 blocks (32 elements per 18-byte block: 2B fp16 d + 16B nibbles)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    qs = blocks[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    combined = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (combined * d).reshape(shape)


def dequantize_gguf_q4_1(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q4_1 blocks (32 elements per 20-byte block: 2B fp16 d + 2B fp16 m + 16B nibbles)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 20)
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    m = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    qs = blocks[:, 4:]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    combined = np.concatenate([lo, hi], axis=1)
    return (combined * d + m).reshape(shape)


def dequantize_gguf_q5_0(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q5_0 blocks (32 elements per 22-byte block: 2B fp16 d + 4B qh + 16B qs)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 22)
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    qh = blocks[:, 2:6].copy().view("<u4")  # [n_blocks, 1]
    qs = blocks[:, 6:22]  # [n_blocks, 16]

    bit_pos_lo = np.arange(16, dtype=np.uint32)[None, :]
    bit_pos_hi = np.arange(16, 32, dtype=np.uint32)[None, :]
    xh_0 = ((qh >> bit_pos_lo) & 1).astype(np.uint8) << 4
    xh_1 = ((qh >> bit_pos_hi) & 1).astype(np.uint8) << 4

    x0 = ((qs & 0x0F) | xh_0).astype(np.int8) - 16
    x1 = ((qs >> 4) | xh_1).astype(np.int8) - 16
    combined = np.concatenate([x0, x1], axis=1).astype(np.float32)
    return (combined * d).reshape(shape)


def dequantize_gguf_q5_1(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q5_1 blocks (32 elements per 24-byte block: 2B d + 2B m + 4B qh + 16B qs)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 24)
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    m = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    qh = blocks[:, 4:8].copy().view("<u4")
    qs = blocks[:, 8:24]

    bit_pos_lo = np.arange(16, dtype=np.uint32)[None, :]
    bit_pos_hi = np.arange(16, 32, dtype=np.uint32)[None, :]
    xh_0 = ((qh >> bit_pos_lo) & 1).astype(np.uint8) << 4
    xh_1 = ((qh >> bit_pos_hi) & 1).astype(np.uint8) << 4

    x0 = ((qs & 0x0F) | xh_0).astype(np.float32)
    x1 = ((qs >> 4) | xh_1).astype(np.float32)
    combined = np.concatenate([x0, x1], axis=1)
    return (combined * d + m).reshape(shape)


def dequantize_gguf_q8_0(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q8_0 blocks (32 elements per 34-byte block: 2B fp16 d + 32B int8)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 32
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 34)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (qs * d).reshape(shape)


def dequantize_gguf_q2_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q2_K super-blocks (256 elements per 84-byte block)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 84)
    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]
    d = blocks[:, 80:82].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 82:84].copy().view("<f2").astype(np.float32)

    out = np.zeros((n_blocks, 256), dtype=np.float32)
    is_idx = 0
    for n in range(2):
        q_chunk = qs[:, n * 32 : (n + 1) * 32]
        for shift in (0, 2, 4, 6):
            sc = (scales[:, is_idx : is_idx + 1] & 0x0F).astype(np.float32)
            mn = (scales[:, is_idx : is_idx + 1] >> 4).astype(np.float32)
            q_part1 = ((q_chunk[:, 0:16] >> shift) & 3).astype(np.float32)
            start1 = n * 128 + (shift // 2) * 32
            out[:, start1 : start1 + 16] = d * sc * q_part1 - dmin * mn
            is_idx += 1

            sc = (scales[:, is_idx : is_idx + 1] & 0x0F).astype(np.float32)
            mn = (scales[:, is_idx : is_idx + 1] >> 4).astype(np.float32)
            q_part2 = ((q_chunk[:, 16:32] >> shift) & 3).astype(np.float32)
            out[:, start1 + 16 : start1 + 32] = d * sc * q_part2 - dmin * mn
            is_idx += 1
    return out.reshape(shape)


def dequantize_gguf_q3_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q3_K super-blocks (256 elements per 110-byte block)."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 110)
    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    scales_raw = blocks[:, 96:108]
    d = blocks[:, 108:110].copy().view("<f2").astype(np.float32)

    # Unpack 16 6-bit scales from 12 bytes
    sc = np.zeros((n_blocks, 16), dtype=np.int8)
    sc[:, 0:4] = (scales_raw[:, 0:4] & 0x0F) | (((scales_raw[:, 8:12] >> 0) & 3) << 4)
    sc[:, 4:8] = (scales_raw[:, 4:8] & 0x0F) | (((scales_raw[:, 8:12] >> 2) & 3) << 4)
    sc[:, 8:12] = (scales_raw[:, 0:4] >> 4) | (((scales_raw[:, 8:12] >> 4) & 3) << 4)
    sc[:, 12:16] = (scales_raw[:, 4:8] >> 4) | (((scales_raw[:, 8:12] >> 6) & 3) << 4)
    sc_f = (sc.astype(np.int16) - 32).astype(np.float32)

    out = np.zeros((n_blocks, 256), dtype=np.float32)
    m_bit = 0
    is_idx = 0
    for n in range(2):
        q_chunk = qs[:, n * 32 : (n + 1) * 32]
        for shift in (0, 2, 4, 6):
            hm_lo = (((hmask[:, 0:16] >> m_bit) & 1) ^ 1) << 2
            q_lo = (((q_chunk[:, 0:16] >> shift) & 3).astype(np.int8) - hm_lo.astype(np.int8)).astype(np.float32)
            base = n * 128 + (shift // 2) * 32
            out[:, base : base + 16] = d * sc_f[:, is_idx : is_idx + 1] * q_lo
            is_idx += 1

            hm_hi = (((hmask[:, 16:32] >> m_bit) & 1) ^ 1) << 2
            q_hi = (((q_chunk[:, 16:32] >> shift) & 3).astype(np.int8) - hm_hi.astype(np.int8)).astype(np.float32)
            out[:, base + 16 : base + 32] = d * sc_f[:, is_idx : is_idx + 1] * q_hi
            is_idx += 1
            m_bit += 1
    return out.reshape(shape)


def dequantize_gguf_q4_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q4_K super-blocks (256 elements per 144-byte block) adapted from WeeLLM."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 144)

    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]

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


def dequantize_gguf_q5_k(raw_bytes: bytes, shape: Sequence[int]) -> np.ndarray:
    """Dequantize GGML Q5_K super-blocks (256 elements per 176-byte block) adapted from WeeLLM."""
    nelem = int(np.prod(shape))
    n_blocks = nelem // 256
    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, 176)

    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]

    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    m = np.zeros((n_blocks, 8), dtype=np.float32)
    sc[:, 0:4] = scales[:, 0:4] & 63
    m[:, 0:4] = scales[:, 4:8] & 63
    sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)

    out = np.zeros((n_blocks, 256), dtype=np.float32)
    for j in range(4):
        q_chunk = qs[:, j * 32 : (j + 1) * 32]
        hb_lo = ((qh >> (2 * j)) & 1) << 4
        hb_hi = ((qh >> (2 * j + 1)) & 1) << 4
        q_lo = ((q_chunk & 0x0F) | hb_lo).astype(np.float32)
        q_hi = ((q_chunk >> 4) | hb_hi).astype(np.float32)
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
    if ggml_type == GGML_TYPE_Q4_1:
        return dequantize_gguf_q4_1(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q5_0:
        return dequantize_gguf_q5_0(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q5_1:
        return dequantize_gguf_q5_1(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q8_0:
        return dequantize_gguf_q8_0(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q2_K:
        return dequantize_gguf_q2_k(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q3_K:
        return dequantize_gguf_q3_k(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q4_K:
        return dequantize_gguf_q4_k(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q5_K:
        return dequantize_gguf_q5_k(raw_bytes, shape)
    if ggml_type == GGML_TYPE_Q6_K:
        return dequantize_gguf_q6_k(raw_bytes, shape)
    raise ValueError(f"Unsupported GGML quantization type code: {ggml_type}")
