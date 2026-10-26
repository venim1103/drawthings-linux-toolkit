"""Zero-copy Safetensors (single/sharded) and GGUF seeker adapted from WeeLLM (`weellm/io/`)."""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .dequant import dequantize_fp8_e4m3, dequantize_fp8_e5m2, dequantize_gguf_tensor
from .keymaps import STANDARD_PREFIXES, get_architecture_aliases
from .transforms import bf16_bytes_to_f32

SAFETENSORS_DTYPE_BYTES: Dict[str, int] = {
    "F16": 2,
    "FLOAT16": 2,
    "BF16": 2,
    "BFLOAT16": 2,
    "F32": 4,
    "FLOAT32": 4,
    "FLOAT": 4,
    "F64": 8,
    "FLOAT64": 8,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

# Standard GGML/GGUF type IDs
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

GGML_BLOCK_INFO: Dict[int, Tuple[int, int, str]] = {
    # type_id: (block_elements, block_bytes, label)
    GGML_TYPE_F32: (1, 4, "F32"),
    GGML_TYPE_F16: (1, 2, "F16"),
    GGML_TYPE_BF16: (1, 2, "BF16"),
    GGML_TYPE_Q4_0: (32, 18, "Q4_0"),
    GGML_TYPE_Q4_1: (32, 20, "Q4_1"),
    GGML_TYPE_Q5_0: (32, 22, "Q5_0"),
    GGML_TYPE_Q5_1: (32, 24, "Q5_1"),
    GGML_TYPE_Q8_0: (32, 34, "Q8_0"),
    GGML_TYPE_Q2_K: (256, 84, "Q2_K"),
    GGML_TYPE_Q3_K: (256, 110, "Q3_K"),
    GGML_TYPE_Q4_K: (256, 144, "Q4_K"),
    GGML_TYPE_Q5_K: (256, 176, "Q5_K"),
    GGML_TYPE_Q6_K: (256, 210, "Q6_K"),
}


@dataclass(frozen=True)
class TensorSliceInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    file_path: Path
    byte_offset: int
    byte_length: int
    ggml_type: Optional[int] = None


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class SafetensorsSeeker:
    """Zero-copy seeker for single-file or sharded `.safetensors` checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.root_path = Path(path).expanduser().resolve()
        self.tensors: Dict[str, TensorSliceInfo] = {}
        self.metadata: Dict[str, str] = {}
        self._fds: Dict[Path, int] = {}

        if self.root_path.is_dir():
            index_file = self.root_path / "model.safetensors.index.json"
            single_file = self.root_path / "model.safetensors"
            if index_file.is_file():
                self._load_sharded_index(index_file)
            elif single_file.is_file():
                self._load_single_file(single_file)
            else:
                raise FileNotFoundError(
                    f"Directory {self.root_path} contains neither model.safetensors.index.json nor model.safetensors"
                )
        elif self.root_path.name.endswith(".index.json"):
            self._load_sharded_index(self.root_path)
        else:
            self._load_single_file(self.root_path)

    def _load_sharded_index(self, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in sharded index: {index_path}")

        base_dir = index_path.parent
        shard_files = sorted(set(str(v) for v in weight_map.values()))
        for shard_rel in shard_files:
            shard_path = (base_dir / shard_rel).resolve()
            if not shard_path.is_relative_to(base_dir.resolve()):
                raise ValueError(f"Shard path escapes the index directory: {shard_rel}")
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing safetensors shard referenced by index: {shard_path}")
            self._load_single_file(shard_path)

        missing = [k for k in weight_map.keys() if k not in self.tensors]
        if missing:
            raise ValueError(f"Sharded index keys missing from shard headers: {missing[:5]}")
        for key, info in self.tensors.items():
            if key not in weight_map or info.file_path != (base_dir / weight_map[key]).resolve():
                raise ValueError(f"Tensor does not belong to its indexed shard: {key}")

    def _load_single_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Safetensors file not found: {file_path}")
        file_size = file_path.stat().st_size
        with file_path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"Cannot read 8-byte safetensors header length: {file_path}")
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len + 8 > file_size:
                raise ValueError(f"Invalid safetensors header length {header_len} in {file_path}")
            if header_len > 100 * 1024 * 1024:
                raise ValueError("Safetensors header exceeds the supported size limit")
            header_obj = json.loads(f.read(header_len), object_pairs_hook=_unique_json_object)

        if not isinstance(header_obj, dict):
            raise ValueError(f"Safetensors header is not a JSON object: {file_path}")

        buffer_start = 8 + header_len
        data_bytes = file_size - buffer_start

        intervals = []
        for key, info in header_obj.items():
            if key == "__metadata__":
                if isinstance(info, dict):
                    for mk, mv in info.items():
                        self.metadata[str(mk)] = str(mv)
                continue
            if not isinstance(info, dict):
                raise ValueError(f"Invalid tensor entry for {key!r} in {file_path}")
            dtype = str(info.get("dtype", "")).upper()
            shape = tuple(int(d) for d in info.get("shape", []))
            if len(shape) > 12 or any(dimension < 0 for dimension in shape):
                raise ValueError(f"Invalid tensor dimensions for {key}")
            offsets = info.get("data_offsets", [])
            if not (isinstance(offsets, list) and len(offsets) == 2):
                raise ValueError(f"Invalid data_offsets for {key!r} in {file_path}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or end > data_bytes:
                raise ValueError(
                    f"Out-of-bounds data_offsets ({start}, {end}) for {key!r} in {file_path} (data_bytes={data_bytes})"
                )
            if dtype not in SAFETENSORS_DTYPE_BYTES:
                raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {key!r}")
            nelem = math.prod(shape) if shape else 1
            expected_bytes = nelem * SAFETENSORS_DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                raise ValueError(
                    f"Byte length mismatch for {key!r}: expected {expected_bytes}, got {end - start}"
                )
            if key in self.tensors:
                raise ValueError(f"Duplicate safetensors key across shards: {key}")
            intervals.append((start, end))
            self.tensors[key] = TensorSliceInfo(
                name=key,
                dtype=dtype,
                shape=shape,
                file_path=file_path,
                byte_offset=buffer_start + start,
                byte_length=end - start,
            )
        end_of_previous = 0
        for start, end in sorted(intervals):
            if start < end_of_previous:
                raise ValueError(f"Overlapping safetensors payloads in {file_path}")
            end_of_previous = end

    def keys(self) -> List[str]:
        return list(self.tensors.keys())

    def resolve_key(
        self,
        key: str,
        prefixes: Sequence[str] = ("", *STANDARD_PREFIXES),
        aliases: Optional[Sequence[str]] = None,
        profile: str = "ltx2_3",
    ) -> Optional[str]:
        candidates: List[str] = []
        for p in prefixes:
            candidates.append(f"{p}{key}" if p else key)
        if aliases:
            for a in aliases:
                candidates.append(a)
                for p in prefixes:
                    if p:
                        candidates.append(f"{p}{a}")
        for auto_alias in get_architecture_aliases(key, profile=profile):
            candidates.append(auto_alias)
        matches = list(dict.fromkeys(candidate for candidate in candidates if candidate in self.tensors))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source aliases for {key}: {matches}")
        return matches[0] if matches else None

    def read_raw_bytes(self, key: str) -> Tuple[TensorSliceInfo, bytes]:
        info = self.tensors[key]
        fd = self._fds.get(info.file_path)
        if fd is None:
            fd = os.open(str(info.file_path), os.O_RDONLY)
            self._fds[info.file_path] = fd
        raw = os.pread(fd, info.byte_length, info.byte_offset)
        if len(raw) != info.byte_length:
            raise IOError(
                f"Short read for {key!r}: expected {info.byte_length} bytes, got {len(raw)}"
            )
        return info, raw

    def read_f32(self, key: str, scale_key: Optional[str] = None) -> np.ndarray:
        """Read tensor `key` directly into a contiguous float32 NumPy array (with optional ComfyUI FP8 scale)."""
        info, raw = self.read_raw_bytes(key)
        dtype = info.dtype
        shape = info.shape

        if dtype in ("F32", "FLOAT32", "FLOAT"):
            arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        elif dtype in ("F16", "FLOAT16"):
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        elif dtype in ("BF16", "BFLOAT16"):
            arr = bf16_bytes_to_f32(raw, shape)
        elif dtype == "F8_E4M3":
            arr = dequantize_fp8_e4m3(raw, shape)
        elif dtype == "F8_E5M2":
            arr = dequantize_fp8_e5m2(raw, shape)
        elif dtype in ("F64", "FLOAT64"):
            arr = np.frombuffer(raw, dtype="<f8").astype(np.float32).reshape(shape)
        else:
            raise ValueError(f"Cannot convert dtype {dtype!r} for {key!r} to float32")

        # Apply ComfyUI FP8 scale_weight if present
        auto_scale_key = scale_key
        if auto_scale_key is None and key.endswith(".weight"):
            candidate = key[: -len(".weight")] + ".scale_weight"
            if candidate in self.tensors:
                auto_scale_key = candidate
            else:
                candidate2 = key[: -len(".weight")] + ".weight_scale"
                if candidate2 in self.tensors:
                    auto_scale_key = candidate2

        if auto_scale_key and auto_scale_key in self.tensors:
            scale_arr = self.read_f32(auto_scale_key)
            if scale_arr.size == 1:
                scale_arr = scale_arr.reshape(())
            elif arr.ndim >= 2 and scale_arr.size == arr.shape[0]:
                scale_arr = scale_arr.reshape((arr.shape[0],) + (1,) * (arr.ndim - 1))
            elif scale_arr.shape != arr.shape:
                raise ValueError(f"Unsupported scale layout {scale_arr.shape} for {key} {arr.shape}")
            arr = arr * scale_arr

        return arr

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __enter__(self) -> "SafetensorsSeeker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GGUFSeeker:
    """Zero-copy GGUF v2/v3 header parser and block seeker adapted from `weellm/io/ggufs/gguf_seek.py`."""

    # GGUF metadata value types
    _VAL_UINT8 = 0
    _VAL_INT8 = 1
    _VAL_UINT16 = 2
    _VAL_INT16 = 3
    _VAL_UINT32 = 4
    _VAL_INT32 = 5
    _VAL_FLOAT32 = 6
    _VAL_BOOL = 7
    _VAL_STRING = 8
    _VAL_ARRAY = 9
    _VAL_UINT64 = 10
    _VAL_INT64 = 11
    _VAL_FLOAT64 = 12

    _SCALAR_SIZES = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    def __init__(self, path: str | Path) -> None:
        self.file_path = Path(path).expanduser().resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
        self.tensors: Dict[str, TensorSliceInfo] = {}
        self.alignment: int = 32
        self._fd: Optional[int] = None
        self._parse_header()

    def _read_exact(self, f, n: int) -> bytes:
        b = f.read(n)
        if len(b) != n:
            raise ValueError(f"Unexpected EOF in GGUF header of {self.file_path}")
        return b

    def _read_string(self, f) -> str:
        length = struct.unpack("<Q", self._read_exact(f, 8))[0]
        return self._read_exact(f, length).decode("utf-8", errors="replace")

    def _skip_or_read_value(self, f, vtype: int):
        if vtype in self._SCALAR_SIZES:
            raw = self._read_exact(f, self._SCALAR_SIZES[vtype])
            if vtype == self._VAL_UINT32:
                return struct.unpack("<I", raw)[0]
            return None
        if vtype == self._VAL_STRING:
            return self._read_string(f)
        if vtype == self._VAL_ARRAY:
            elem_type, count = struct.unpack("<IQ", self._read_exact(f, 12))
            if elem_type in self._SCALAR_SIZES:
                f.seek(count * self._SCALAR_SIZES[elem_type], os.SEEK_CUR)
            else:
                for _ in range(count):
                    self._skip_or_read_value(f, elem_type)
            return None
        raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")

    def _parse_header(self) -> None:
        file_size = self.file_path.stat().st_size
        with self.file_path.open("rb") as f:
            magic = self._read_exact(f, 4)
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file (magic={magic!r}): {self.file_path}")
            version, tensor_count, kv_count = struct.unpack("<IQQ", self._read_exact(f, 20))
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version {version} in {self.file_path}")

            for _ in range(kv_count):
                key = self._read_string(f)
                vtype = struct.unpack("<I", self._read_exact(f, 4))[0]
                val = self._skip_or_read_value(f, vtype)
                if key == "general.alignment" and isinstance(val, int) and val > 0:
                    self.alignment = val

            raw_entries = []
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", self._read_exact(f, 4))[0]
                dims_gguf = [
                    struct.unpack("<Q", self._read_exact(f, 8))[0] for _ in range(n_dims)
                ]
                # GGUF stores dimensions in reverse order compared to PyTorch/Safetensors
                shape = tuple(int(d) for d in reversed(dims_gguf))
                ggml_type = struct.unpack("<I", self._read_exact(f, 4))[0]
                rel_offset = struct.unpack("<Q", self._read_exact(f, 8))[0]
                raw_entries.append((name, shape, ggml_type, rel_offset))

            header_end = f.tell()
            rem = header_end % self.alignment
            data_start = header_end + ((self.alignment - rem) if rem != 0 else 0)

            for name, shape, ggml_type, rel_offset in raw_entries:
                if ggml_type not in GGML_BLOCK_INFO:
                    raise ValueError(f"Unsupported GGML tensor type {ggml_type} for {name!r}")
                block_elems, block_bytes, label = GGML_BLOCK_INFO[ggml_type]
                if len(shape) > 12 or any(dimension <= 0 for dimension in shape):
                    raise ValueError(f"Invalid GGUF dimensions for {name}")
                nelem = math.prod(shape) if shape else 1
                if nelem % block_elems != 0:
                    raise ValueError(
                        f"GGUF tensor {name!r} element count {nelem} not divisible by block size {block_elems}"
                    )
                byte_length = (nelem // block_elems) * block_bytes
                abs_offset = data_start + rel_offset
                if abs_offset + byte_length > file_size:
                    raise ValueError(f"GGUF tensor {name!r} exceeds file bounds in {self.file_path}")
                if name in self.tensors:
                    raise ValueError(f"Duplicate GGUF tensor name: {name}")
                self.tensors[name] = TensorSliceInfo(
                    name=name,
                    dtype=label,
                    shape=shape,
                    file_path=self.file_path,
                    byte_offset=abs_offset,
                    byte_length=byte_length,
                    ggml_type=ggml_type,
                )

    def keys(self) -> List[str]:
        return list(self.tensors.keys())

    def resolve_key(
        self,
        key: str,
        prefixes: Sequence[str] = ("", *STANDARD_PREFIXES),
        aliases: Optional[Sequence[str]] = None,
        profile: str = "ltx2_3",
    ) -> Optional[str]:
        candidates: List[str] = [f"{prefix}{key}" for prefix in prefixes]
        if aliases:
            candidates.extend(aliases)
        candidates.extend(get_architecture_aliases(key, profile=profile))
        matches = list(dict.fromkeys(candidate for candidate in candidates if candidate in self.tensors))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous GGUF aliases for {key}: {matches}")
        return matches[0] if matches else None

    def read_f32(self, key: str) -> np.ndarray:
        info = self.tensors[key]
        if self._fd is None:
            self._fd = os.open(str(self.file_path), os.O_RDONLY)
        raw = os.pread(self._fd, info.byte_length, info.byte_offset)
        if len(raw) != info.byte_length:
            raise IOError(f"Short read for GGUF tensor {key!r}")
        return dequantize_gguf_tensor(raw, info.ggml_type, info.shape)

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "GGUFSeeker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_seeker(path: str | Path) -> SafetensorsSeeker | GGUFSeeker:
    """Auto-detect and open either a `GGUFSeeker` or `SafetensorsSeeker`."""
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return GGUFSeeker(p)
    return SafetensorsSeeker(p)
