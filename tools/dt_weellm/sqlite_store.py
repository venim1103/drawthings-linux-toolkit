"""Draw Things `s4nnc` SQLite `.ckpt` (and `-tensordata` sidecar) reader and staged writer."""

from __future__ import annotations

import os
import math
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
    has_splits: bool = False
    reserved: int = 0

    @property
    def codec_identifier(self) -> int:
        return (self.tensor_type >> 32) & 0xFFFFFFFF


class DrawThingsCkptStore:
    """Reader and atomic staged writer for Draw Things SQLite `.ckpt` files."""

    def __init__(self, ckpt_path: str | Path, writable: bool = False,
                 chunk_bytes: int = 64 * 1024 * 1024) -> None:
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.sidecar_path = Path(f"{self.ckpt_path}-tensordata")
        self.writable = writable
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self.chunk_bytes = chunk_bytes
        self.con = sqlite3.connect(
            str(self.ckpt_path) if writable else self.ckpt_path.as_uri() + "?mode=ro",
            uri=not writable,
        )
        if writable:
            self.con.execute("PRAGMA journal_mode=DELETE")
            self.con.execute("PRAGMA synchronous=FULL")
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS tensors ("
                "name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
                "datatype INTEGER, dim BLOB, data BLOB)"
            )
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS tensor_splits ("
                "name TEXT, part INTEGER, data BLOB, PRIMARY KEY (name, part))"
            )
        self._sidecar_fd: Optional[int] = None
        if self.sidecar_path.is_file():
            self._sidecar_fd = os.open(str(self.sidecar_path), os.O_RDONLY)

    def list_keys(self) -> List[str]:
        cur = self.con.cursor()
        cur.execute("SELECT name FROM tensors INDEXED BY sqlite_autoindex_tensors_1")
        return [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]

    def read_metadata(self, name: str) -> Optional[CkptTensorRow]:
        cur = self.con.cursor()
        try:
            row = cur.execute(
                "SELECT type, format, datatype, dim, length(data) FROM tensors WHERE name=?",
                (name,),
            ).fetchone()
        except sqlite3.DataError as error:
            raise ValueError(f"Unsupported oversized SQLite row: {name}") from error
        if row is None:
            return None
        t_type, t_fmt, dt, dim_raw, dlen = row
        dim_bytes = bytes(dim_raw)
        shape, slots = decode_dim_blob(dim_bytes)
        is_ext = ((int(t_type) >> 32) & 0x1000_0000) != 0
        has_splits = bool(int(t_fmt) & (1 << 32))
        if has_splits:
            tail_length = self.con.execute(
                "SELECT SUM(length(data)) FROM tensor_splits WHERE name=?", (name,)
            ).fetchone()[0]
            if tail_length is None:
                raise ValueError(f"Split tensor has no tail: {name}")
            dlen = int(dlen) + int(tail_length)
        return CkptTensorRow(
            name=name,
            tensor_type=int(t_type),
            tensor_format=int(t_fmt) & 0xFFFFFFFF,
            datatype=int(dt) & 0xFFFFFFFF,
            dim_blob=dim_bytes,
            shape=shape,
            ndim_slots=slots,
            data_len=int(dlen) if dlen is not None else 0,
            is_external=is_ext,
            has_splits=has_splits,
            reserved=(int(dt) >> 32) & 0xFFFFFFFF,
        )

    def read_raw_tensor(self, name: str) -> Optional[Tuple[CkptTensorRow, bytes]]:
        meta = self.read_metadata(name)
        if meta is None:
            return None
        cur = self.con.cursor()
        try:
            row = cur.execute(
                "SELECT type, format, datatype, dim, data FROM tensors WHERE name=?",
                (name,),
            ).fetchone()
        except sqlite3.DataError as error:
            raise ValueError(f"Unsupported oversized SQLite row: {name}") from error
        if row is None:
            return None
        t_type, t_fmt, dt, dim_raw, data_raw = row
        dim_bytes = bytes(dim_raw)
        data_bytes = bytes(data_raw) if data_raw is not None else b""
        shape, slots = decode_dim_blob(dim_bytes)
        if meta.has_splits:
            parts = self.con.execute(
                "SELECT part, data FROM tensor_splits WHERE name=? ORDER BY part", (name,)
            )
            chunks = [data_bytes]
            for expected_part, (part, chunk) in enumerate(parts):
                if part != expected_part or not chunk:
                    raise ValueError(f"Invalid split sequence for {name}")
                chunks.append(bytes(chunk))
            data_bytes = b"".join(chunks)

        if meta.is_external:
            if len(data_bytes) != 24 or self._sidecar_fd is None:
                raise ValueError(f"Missing sidecar or invalid external descriptor for {name}")
            offset = struct.unpack("<Q", data_bytes[8:16])[0]
            length = struct.unpack("<Q", data_bytes[16:24])[0]
            if offset + length > os.fstat(self._sidecar_fd).st_size:
                raise ValueError(f"External tensor exceeds sidecar bounds: {name}")
            payload = os.pread(self._sidecar_fd, length, offset)
            if len(payload) != length:
                raise ValueError(f"Short external tensor read: {name}")
        else:
            payload = data_bytes

        meta.data_len = len(payload)
        return meta, payload

    def read_f32(self, name: str) -> Optional[Tuple[CkptTensorRow, np.ndarray]]:
        """Read unquantized F16, BF16, or F32 tensor from SQLite store into float32 NumPy array."""
        res = self.read_raw_tensor(name)
        if res is None:
            return None
        meta, payload = res
        if meta.codec_identifier & ~0x1000_0000 or meta.reserved:
            raise ValueError(f"Unsupported encoded tensor codec for {name}: {meta.codec_identifier:#x}")
        nelem = math.prod(meta.shape) if meta.shape else 1
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
        if not name or not arr_f32.size or not np.isfinite(arr_f32).all():
            raise ValueError("Tensor names must be nonempty and values finite")
        payload = tensor_to_ckpt_bytes(arr_f32, datatype)
        validation_bytes = max(8, self.chunk_bytes - self.chunk_bytes % 8)
        for offset in range(0, len(payload), validation_bytes):
            chunk = payload[offset:offset + validation_bytes]
            if datatype == DT_BFLOAT16:
                finite = ((np.frombuffer(chunk, dtype="<u2") & 0x7F80) != 0x7F80).all()
            else:
                finite = np.isfinite(np.frombuffer(chunk, dtype="<f2" if datatype == DT_FLOAT16 else "<f4")).all()
            if not finite:
                raise ValueError(f"Floating-point overflow while serializing {name}")
        if tensor_type >> 32 or tensor_format >> 32:
            raise ValueError("Raw writes cannot preserve encoded codec or split markers")
        self.con.execute("DELETE FROM tensor_splits WHERE name=?", (name,))
        split = len(payload) > self.chunk_bytes
        self.con.execute(
            "INSERT OR REPLACE INTO tensors (name, type, format, datatype, dim, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                int(tensor_type),
                int(tensor_format) | ((1 << 32) if split else 0),
                int(datatype),
                sqlite3.Binary(dim_blob),
                sqlite3.Binary(payload[:self.chunk_bytes]),
            ),
        )
        if split:
            self.con.executemany(
                "INSERT INTO tensor_splits (name, part, data) VALUES (?, ?, ?)",
                ((name, part, sqlite3.Binary(payload[offset:offset + self.chunk_bytes]))
                 for part, offset in enumerate(range(self.chunk_bytes, len(payload), self.chunk_bytes))),
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
        if exc_type is not None and self.writable:
            self.con.rollback()
        self.close()
