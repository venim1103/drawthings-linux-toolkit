#!/usr/bin/env python3
import argparse
import struct
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import fpzip  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fpzip = None


HEADER_BYTES = 68

CCV_8U = 0x01000
CCV_32S = 0x02000
CCV_32F = 0x04000
CCV_64S = 0x08000
CCV_64F = 0x10000
CCV_16F = 0x20000
CCV_16BF = 0x80000

CCV_TENSOR_FORMAT_NCHW = 0x01
CCV_TENSOR_FORMAT_NHWC = 0x02

IDENTIFIER_RAW = 0x0
IDENTIFIER_FPZIP = 0xF7217
IDENTIFIER_ZIP = 0x217


@dataclass(frozen=True)
class TensorHeader:
    identifier: int
    tensor_type: int
    tensor_format: int
    datatype: int
    dims: tuple[int, ...]


def _read_tensor_header(blob: bytes) -> TensorHeader:
    if len(blob) < HEADER_BYTES:
        raise ValueError(f"Tensor blob too small: {len(blob)} bytes")

    values = struct.unpack("<17I", blob[:HEADER_BYTES])
    identifier, tensor_type, tensor_format, datatype, _reserved, *dims = values
    nonzero_dims = tuple(int(d) for d in dims if d > 0)
    return TensorHeader(
        identifier=int(identifier),
        tensor_type=int(tensor_type),
        tensor_format=int(tensor_format),
        datatype=int(datatype),
        dims=nonzero_dims,
    )


def _dtype_from_header(datatype: int):
    if datatype == CCV_8U:
        return np.uint8
    if datatype == CCV_32S:
        return np.int32
    if datatype == CCV_64S:
        return np.int64
    if datatype == CCV_16F:
        return np.float16
    if datatype == CCV_32F:
        return np.float32
    if datatype == CCV_64F:
        return np.float64
    if datatype == CCV_16BF:
        return np.uint16  # bfloat16 is unpacked separately.
    raise ValueError(f"Unsupported tensor datatype flag: {datatype}")


def _decode_raw_or_zip_payload(header: TensorHeader, payload: bytes, expected_bytes: int) -> bytes:
    if header.identifier == IDENTIFIER_RAW:
        decoded = payload
    elif header.identifier == IDENTIFIER_ZIP:
        decoded = zlib.decompress(payload)
    else:
        raise ValueError(f"Unsupported payload identifier for byte decode: 0x{header.identifier:x}")

    if len(decoded) != expected_bytes:
        raise ValueError(
            f"Decoded payload size mismatch: got {len(decoded)}, expected {expected_bytes}"
        )
    return decoded


def _decode_fpzip_payload(payload: bytes, expected_count: int) -> np.ndarray:
    if fpzip is None:
        raise ValueError(
            "Tensor payload is fpzip-compressed but Python package 'fpzip' is unavailable"
        )

    arr = np.asarray(fpzip.decompress(payload, order="C"), dtype=np.float32).reshape(-1)
    if arr.size != expected_count:
        raise ValueError(f"fpzip element count mismatch: got {arr.size}, expected {expected_count}")
    return arr


def _tensor_to_numpy(blob: bytes, header: TensorHeader, expected_count: int) -> np.ndarray:
    payload = blob[HEADER_BYTES:]

    if header.identifier == IDENTIFIER_FPZIP:
        return _decode_fpzip_payload(payload, expected_count)

    np_dtype = _dtype_from_header(header.datatype)
    expected_bytes = expected_count * np.dtype(np_dtype).itemsize
    decoded = _decode_raw_or_zip_payload(header, payload, expected_bytes)

    if header.datatype == CCV_16BF:
        bf16 = np.frombuffer(decoded, dtype="<u2")
        return (bf16.astype(np.uint32) << 16).view("<f4")

    if np_dtype is np.float16:
        return np.frombuffer(decoded, dtype="<f2")
    if np_dtype is np.float32:
        return np.frombuffer(decoded, dtype="<f4")
    if np_dtype is np.float64:
        return np.frombuffer(decoded, dtype="<f8")
    if np_dtype is np.int32:
        return np.frombuffer(decoded, dtype="<i4")
    if np_dtype is np.int64:
        return np.frombuffer(decoded, dtype="<i8")
    return np.frombuffer(decoded, dtype=np_dtype)


def _float_image_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    safe = np.where(finite, arr, 0.0)
    valid = safe[finite]
    p1, p99 = np.percentile(valid, [1.0, 99.0])

    if p1 >= -0.05 and p99 <= 1.05:
        scaled = safe * 255.0
    elif p1 >= -1.05 and p99 <= 1.05:
        scaled = (safe + 1.0) * 127.5
    else:
        lo, hi = float(p1), float(p99)
        if hi <= lo:
            lo, hi = float(valid.min()), float(valid.max())
        if hi <= lo:
            scaled = np.zeros_like(safe)
        else:
            scaled = (safe - lo) * (255.0 / (hi - lo))

    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def decode_image_tensor_to_uint8(blob: bytes) -> np.ndarray:
    header = _read_tensor_header(blob)
    dims = header.dims

    if len(dims) < 3:
        raise ValueError(f"Expected image tensor dims, got {dims}")

    if header.tensor_format == CCV_TENSOR_FORMAT_NHWC:
        if len(dims) == 3:
            n, h, w, c = 1, dims[0], dims[1], dims[2]
        else:
            n, h, w, c = dims[:4]
        arr = _tensor_to_numpy(blob, header, n * h * w * c).reshape(n, h, w, c)[0]
    elif header.tensor_format == CCV_TENSOR_FORMAT_NCHW:
        if len(dims) == 3:
            n, c, h, w = 1, dims[0], dims[1], dims[2]
        else:
            n, c, h, w = dims[:4]
        arr = _tensor_to_numpy(blob, header, n * h * w * c).reshape(n, c, h, w)[0].transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported image tensor format: {header.tensor_format}")

    if arr.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels for image tensor, got shape {arr.shape}")

    rgb = arr[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.integer):
        info = np.iinfo(rgb.dtype)
        scaled = (rgb.astype(np.float32) - info.min) * (255.0 / (info.max - info.min))
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return _float_image_to_uint8(rgb)


def decode_audio_tensor_to_stereo_f32(blob: bytes) -> np.ndarray:
    header = _read_tensor_header(blob)
    dims = header.dims

    if len(dims) < 2:
        raise ValueError(f"Expected 2D audio tensor, got dims={dims}")

    if len(dims) >= 3 and dims[0] == 1 and dims[1] == 2:
        channels = 2
        samples = int(np.prod(dims[2:]))
    else:
        channels, samples = dims[:2]

    if channels != 2:
        raise ValueError(f"Expected stereo audio with 2 channels, got {channels}")

    expected = channels * samples
    arr = _tensor_to_numpy(blob, header, expected)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = float(max(abs(info.min), abs(info.max)))
        if denom <= 0:
            denom = 1.0
        arr = arr.astype(np.float32) / denom
    else:
        arr = arr.astype(np.float32)

    # Return shape: [samples, channels]
    arr = arr.reshape(channels, samples).T
    arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32)


def write_wav_f32(stereo: np.ndarray, out_path: Path, sample_rate: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(4)
        wf.setframerate(sample_rate)
        wf.setcomptype("NONE", "not compressed")
        wf.writeframes(stereo.astype("<f4").tobytes())


def write_animated_gif(frames: list[np.ndarray], out_path: Path, fps: int, seconds: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / max(1, fps)))
    if len(frames) == 1:
        frame_count = max(1, int(round(fps * seconds)))
        pil_frame = Image.fromarray(frames[0], mode="RGB")
        pil_frames = [pil_frame.copy() for _ in range(frame_count)]
    else:
        pil_frames = [Image.fromarray(frame, mode="RGB") for frame in frames]

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def write_mp4(frames: list[np.ndarray], out_path: Path, fps: int, seconds: float) -> bool:
    if cv2 is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(max(1, fps)), (w, h))
    if not writer.isOpened():
        return False

    if len(frames) == 1:
        frame_count = max(1, int(round(fps * seconds)))
        bgr = frames[0][..., ::-1]
        for i in range(frame_count):
            # Add minimal motion so players treat this as a short clip, not a still.
            if i % 2 == 0:
                out = bgr
            else:
                out = np.roll(bgr, shift=1, axis=1)
            writer.write(out)
    else:
        for frame in frames:
            writer.write(frame[..., ::-1])

    writer.release()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Draw Things tensor blobs to playable media")
    parser.add_argument(
        "--image-bin",
        action="append",
        required=True,
        help="Path to generated image .bin tensor (repeat for multiple frames)",
    )
    parser.add_argument("--audio-bin", help="Path to generated audio .bin tensor")
    parser.add_argument("--out-dir", required=True, help="Output directory for playable artifacts")
    parser.add_argument("--base-name", default="dt_playable", help="Base name for output files")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-seconds", type=float, default=2.0)
    parser.add_argument("--mp4-fps", type=int, default=12)
    parser.add_argument("--mp4-seconds", type=float, default=2.0)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    args = parser.parse_args()

    frames = []
    for image_bin in args.image_bin:
        image_blob = Path(image_bin).read_bytes()
        frames.append(decode_image_tensor_to_uint8(image_blob))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{args.base_name}.png"
    Image.fromarray(frames[0], mode="RGB").save(png_path)

    gif_path = out_dir / f"{args.base_name}.gif"
    write_animated_gif(frames, gif_path, fps=args.gif_fps, seconds=args.gif_seconds)

    print(f"wrote image: {png_path}")
    print(f"wrote playable gif: {gif_path}")
    print(f"frames used: {len(frames)}")

    mp4_path = out_dir / f"{args.base_name}.mp4"
    if write_mp4(frames, mp4_path, fps=args.mp4_fps, seconds=args.mp4_seconds):
        print(f"wrote playable mp4: {mp4_path}")
    else:
        print("skipped mp4: OpenCV not available or failed to initialize encoder")

    if args.audio_bin:
        audio_blob = Path(args.audio_bin).read_bytes()
        stereo = decode_audio_tensor_to_stereo_f32(audio_blob)
        wav_path = out_dir / f"{args.base_name}.wav"
        write_wav_f32(stereo, wav_path, sample_rate=args.audio_sample_rate)
        print(f"wrote audio: {wav_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
