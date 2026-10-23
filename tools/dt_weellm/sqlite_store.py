"""Draw Things `s4nnc` SQLite `.ckpt` (and `-tensordata` sidecar) reader and staged writer."""

from __future__ import annotations

import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .transforms import (
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
    DT_FORMAT_NHWC,
    DT_TYPE_CPU,
    bf16_bytes_to_f32,
    decode_dim_blob,
    encode_dim_blob,
    tensor_to_ckpt_bytes,
)


@dataclass
class CkptTensorRow:
    name: str
    tensor_type: int
    tensor_format: int
    datatype: int
    dim_blob: bytes
    shape: List[int]
    ndim_slots: int
    data_len: int
    is_external: bool = False


class DrawThingsCkptStore:
    """Reader and atomic staged writer for Draw Things SQLite `.ckpt` files."""

    def __init__(self, ckpt_path: str | Path, writable: bool = False) -> None:
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.sidecar_path = Path(f"{self.ckpt_path}-tensordata")
        self.writable = writable
        self.con = sqlite3.connect(str(self.ckpt_path))
        if writable:
            self.con.execute("PRAGMA journal_mode=DELETE")
            self.con.execute("PRAGMA synchronous=OFF")
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS tensors ("
                "name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
                "datatype INTEGER, dim BLOB, data BLOB)"
            )
        self._sidecar_fd: Optional[int] = None
        if self.sidecar_path.is_file():
            self._sidecar_fd = os.open(str(self.sidecar_path), os.O_RDONLY)

    def list_keys(self) -> List[str]:
        cur = self.con.cursor()
        cur.execute("SELECT name FROM tensors")
        return [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]

    def read_metadata(self, name: str) -> Optional[CkptTensorRow]:
        cur = self.con.cursor()
        try:
            row = cur.execute(
                "SELECT type, format, datatype, dim, length(data) FROM tensors WHERE name=?",
                (name,),
            ).fetchone()
        except sqlite3.DataError:
            return None
        if row is None:
            return None
        t_type, t_fmt, dt, dim_raw, dlen = row
        dim_bytes = bytes(dim_raw)
        shape, slots = decode_dim_blob(dim_bytes)
        is_ext = (int(t_type) & 0x1000_0000) != 0
        return CkptTensorRow(
            name=name,
            tensor_type=int(t_type),
            tensor_format=int(t_fmt),
            datatype=int(dt),
            dim_blob=dim_bytes,
            shape=shape,
            ndim_slots=slots,
            data_len=int(dlen) if dlen is not None else 0,
            is_external=is_ext,
        )

    def read_raw_tensor(self, name: str) -> Optional[Tuple[CkptTensorRow, bytes]]:
        cur = self.con.cursor()
        try:
            row = cur.execute(
                "SELECT type, format, datatype, dim, data FROM tensors WHERE name=?",
                (name,),
            ).fetchone()
        except sqlite3.DataError:
            return None
        if row is None:
            return None
        t_type, t_fmt, dt, dim_raw, data_raw = row
        dim_bytes = bytes(dim_raw)
        data_bytes = bytes(data_raw) if data_raw is not None else b""
        shape, slots = decode_dim_blob(dim_bytes)
        is_ext = (int(t_type) & 0x1000_0000) != 0

        if is_ext and len(data_bytes) >= 24 and self._sidecar_fd is not None:
            # Externalized store format in s4nnc Store.swift:
            # offset at +8 (uint64), length at +16 (uint64)
            offset = struct.unpack("<Q", data_bytes[8:16])[0]
            length = struct.unpack("<Q", data_bytes[16:24])[0]
            payload = os.pread(self._sidecar_fd, length, offset)
        else:
            payload = data_bytes

        meta = CkptTensorRow(
            name=name,
            tensor_type=int(t_type),
            tensor_format=int(t_fmt),
            datatype=int(dt),
            dim_blob=dim_bytes,
            shape=shape,
            ndim_slots=slots,
            data_len=len(payload),
            is_external=is_ext,
        )
        return meta, payload

    def read_f32(self, name: str) -> Optional[Tuple[CkptTensorRow, np.ndarray]]:
        """Read unquantized F16, BF16, or F32 tensor from SQLite store into float32 NumPy array."""
        res = self.read_raw_tensor(name)
        if res is None:
            return None
        meta, payload = res
        nelem = int(np.prod(meta.shape)) if meta.shape else 1
        if meta.datatype == DT_FLOAT32 and len(payload) == nelem * 4:
            arr = np.frombuffer(payload, dtype="<f4").reshape(meta.shape).copy()
        elif meta.datatype == DT_FLOAT16 and len(payload) == nelem * 2:
            arr = np.frombuffer(payload, dtype="<f2").astype(np.float32).reshape(meta.shape)
        elif meta.datatype == DT_BFLOAT16 and len(payload) == nelem * 2:
            arr = bf16_bytes_to_f32(payload, meta.shape)
        else:
            raise ValueError(
                f"Tensor {name!r} is palettized/compressed or has unexpected payload size "
                f"(datatype={meta.datatype}, len={len(payload)}, expected_nelem={nelem})"
            )
        return meta, arr

    def write_tensor(
        self,
        name: str,
        arr_f32: np.ndarray,
        datatype: int = DT_FLOAT16,
        tensor_type: int = DT_TYPE_CPU,
        tensor_format: int = DT_FORMAT_NHWC,
        ndim_slots: int = 12,
        rank_policy: str = "preserve",
    ) -> None:
        if not self.writable:
            raise RuntimeError(f"Store {self.ckpt_path} was opened read-only")
        dim_blob = encode_dim_blob(arr_f32.shape, ndim_slots=ndim_slots, rank_policy=rank_policy)
        payload = tensor_to_ckpt_bytes(arr_f32, datatype)
        self.con.execute(
            "INSERT OR REPLACE INTO tensors (name, type, format, datatype, dim, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                int(tensor_type),
                int(tensor_format),
                int(datatype),
                sqlite3.Binary(dim_blob),
                sqlite3.Binary(payload),
            ),
        )

    def commit(self) -> None:
        if self.writable:
            self.con.commit()

    def close(self) -> None:
        if self.writable:
            self.con.commit()
        self.con.close()
        if self._sidecar_fd is not None:
            try:
                os.close(self._sidecar_fd)
            except OSError:
                pass
            self._sidecar_fd = None

    def __enter__(self) -> "DrawThingsCkptStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
