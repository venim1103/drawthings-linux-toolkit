#!/usr/bin/env python3
"""Quantize supported MOSS core matrices with pinned native Q4_K/Hadamard-256."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_moss_convert import (
    MOSS_I4X_IDENTIFIER, ROOT, file_digest, moss_i4x_bytes, moss_i4x_eligible,
    prepare_moss_bundle, validate_moss_checkpoint,
)
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore

NATIVE_HASHES = {
    "lib/nnc/ccv_nnc_8i_rowwise.c": "3687be9175ad42c65e40ef3416a1830facc45ef9eb86785a6c2ff8d70fa0dd58",
    "lib/ccv_util.c": "dc95565feae6024ec7c0e469dab8c0bf135408709f8429884d09ce8f03aeeaf5",
    "lib/nnc/ccv_nnc.h": "825e313f7e8180fd81326058e78caa98e425baf6d96fff98fd961f8ed9d5071c",
    "lib/ccv.h": "6ba9df337453e06e017a3cbe5e49aaf3a50b00801384dca362bd7823a3e6b3e2",
    "lib/ccv_internal.h": "0819373daa817651a6c3ac7df9cda8b01b060b43590bf55a17c93fbcbd21ffbd",
    "lib/nnc/ccv_nnc_internal.h": "b774844de0a6403673795da5f9e5cd9ccdbca5a386fc7059fdedf5090476da9c",
}


def regular_hadamard_256(values):
    result = np.array(values, dtype=np.float32, copy=True).reshape(-1, 256)
    for stride in (1, 4, 16, 64):
        groups = result.reshape(-1, 4, stride)
        first, second, third, fourth = (groups[:, index].copy() for index in range(4))
        groups[:, 0] = first + second + third - fourth
        groups[:, 1] = first + second - third + fourth
        groups[:, 2] = first - second + third + fourth
        groups[:, 3] = -first + second + third + fourth
    return (result / 16).reshape(values.shape)


class NativeI4xEncoder:
    def __init__(self, source, build_directory):
        source = Path(source).resolve()
        for name, expected in NATIVE_HASHES.items():
            if file_digest(source / name) != expected:
                raise ValueError(f"Pinned native encoder source differs: {name}")
        self.source = source
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if compiler is None:
            raise ValueError("A C compiler is required for native i4x encoding")
        library = Path(build_directory) / "moss-i4x.so"
        command = [compiler, "-O3", "-shared", "-fPIC", "-ffunction-sections", "-fdata-sections",
                   "-fvisibility=hidden", "-I", str(source / "lib"), str(ROOT / "tools/dt_moss_quantize.c"),
                   str(source / "lib/ccv_util.c"), "-Wl,--gc-sections", "-lm", "-o", str(library)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Native i4x encoder compilation failed: {result.stderr}")
        self.library = ctypes.CDLL(str(library))
        self.library.dt_moss_i4x_encode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                                   ctypes.c_void_p, ctypes.c_size_t]
        self.library.dt_moss_i4x_encode.restype = ctypes.c_size_t
        self.library.dt_moss_i4x_decode_rotated.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                                           ctypes.c_size_t, ctypes.c_void_p]
        self.library.dt_moss_i4x_decode_rotated.restype = ctypes.c_int

    def decode_rotated(self, payload, shape):
        if len(shape) != 2 or not all(shape) or shape[1] % 256 or np.prod(shape) > 33554432:
            raise ValueError("Native i4x matrices require bounded rows and 256-column groups")
        if len(payload) != moss_i4x_bytes(shape):
            raise ValueError("Invalid native i4x payload length")
        packed = np.frombuffer(payload, dtype=np.uint8)
        decoded = np.empty(shape, dtype=np.float16)
        status = self.library.dt_moss_i4x_decode_rotated(packed.ctypes.data, packed.nbytes,
                                                       shape[0], shape[1], decoded.ctypes.data)
        if status or not np.isfinite(decoded).all():
            raise ValueError("Native i4x decoding failed or produced nonfinite values")
        return decoded

    def encode(self, values):
        values = np.ascontiguousarray(values, dtype=np.float16)
        if (values.ndim != 2 or not values.size or values.shape[1] % 256
                or values.nbytes > 64 * 1024 * 1024 or not np.isfinite(values).all()):
            raise ValueError("Native i4x matrices must be finite, at most 64 MiB and use 256-column groups")
        packed = np.zeros(moss_i4x_bytes(values.shape), dtype=np.uint8)
        length = self.library.dt_moss_i4x_encode(values.ctypes.data, values.shape[0], values.shape[1],
                                               packed.ctypes.data, packed.nbytes)
        if length != packed.nbytes:
            raise ValueError("Native i4x encoder returned an invalid payload length")
        payload = packed.tobytes()
        decoded = self.decode_rotated(payload, values.shape)
        indices = [0, values.shape[0] - 1]
        restored = regular_hadamard_256(decoded[indices])
        reference = values[indices].astype(np.float32)
        denominator = float(np.linalg.norm(reference))
        difference = float(np.linalg.norm(restored - reference))
        relative_rms = difference / denominator if denominator else difference
        if not np.isfinite(relative_rms) or relative_rms > 0.25:
            raise ValueError(f"Native i4x sampled reconstruction error is excessive: {relative_rms}")
        return payload, relative_rms


def quantize_moss_checkpoint(source, output, encoder, jobs=1):
    source = Path(source).resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Quantized output already exists: {output}")
    if output.suffix != ".ckpt" or jobs not in range(1, 17):
        raise ValueError("Output must end in .ckpt and jobs must be between 1 and 16")
    validate_moss_checkpoint(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < source.stat().st_size + 1024 ** 3:
        raise ValueError("Insufficient space for staged i4x output and 1 GiB reserve")
    source_hash = file_digest(source)
    quantized = 0
    preserved = 0
    maximum_sampled_error = 0.0
    with tempfile.TemporaryDirectory(prefix=".moss-i4x.", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with DrawThingsCkptStore(source) as original, DrawThingsCkptStore(staged, writable=True) as target, ThreadPoolExecutor(max_workers=jobs) as workers:
            pending = deque()

            def publish_next():
                nonlocal quantized, preserved, maximum_sampled_error
                metadata, payload, future = pending.popleft()
                identifier = 0
                if future is not None:
                    payload, error = future.result()
                    identifier = MOSS_I4X_IDENTIFIER
                    quantized += 1
                    maximum_sampled_error = max(maximum_sampled_error, error)
                else:
                    preserved += 1
                target.con.execute("INSERT INTO tensors (name, type, format, datatype, dim, data) VALUES (?, ?, ?, ?, ?, ?)",
                                   (metadata.name, (metadata.tensor_type & 0xFFFFFFFF) | (identifier << 32),
                                    metadata.tensor_format, metadata.datatype, metadata.dim_blob, payload))
                print(f"i4x_tensor={metadata.name} codec={'Q4_K_H256' if identifier else 'preserved'}", file=sys.stderr, flush=True)

            for name in original.list_keys():
                metadata, payload = original.read_raw_tensor(name)
                dtype = "<f4" if metadata.datatype == 16384 else "<f2"
                values = np.frombuffer(payload, dtype=dtype).reshape(metadata.shape)
                if not np.isfinite(values).all():
                    raise ValueError(f"Cannot quantize nonfinite tensor: {name}")
                future = None
                if moss_i4x_eligible(name, metadata.shape, metadata.datatype):
                    future = workers.submit(encoder.encode, values)
                    payload = None
                pending.append((metadata, payload, future))
                if len(pending) >= jobs:
                    publish_next()
            while pending:
                publish_next()
        validation = validate_moss_checkpoint(staged, allow_i4x=True)
        if file_digest(source) != source_hash:
            raise ValueError("Source checkpoint changed during quantization")
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(staged, output)
    return {"result": "PASS", "quantization": "i4x", "policy": "Q4_K_H256 core matrices; sensitive tensors preserved verbatim",
            "source": str(source), "source_sha256": source_hash, "output": str(output),
            "output_sha256": file_digest(output), "bytes": output.stat().st_size, "validation": validation,
            "quantized_tensors": quantized, "preserved_tensors": preserved,
            "maximum_sampled_relative_rms_error": maximum_sampled_error,
            "native_source_hashes": NATIVE_HASHES, "native_wrapper_sha256": file_digest(ROOT / "tools/dt_moss_quantize.c")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Validated full-precision MOSS .ckpt; encoded inputs are rejected")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccv-source", type=Path, required=True, help="Pinned CCV 69b1ed1f CPU encoder sources")
    parser.add_argument("--jobs", type=int, default=1, help="Independent CPU matrix workers, 1-16; inference threading is unchanged")
    parser.add_argument("--prepare-inference", type=Path)
    parser.add_argument("--companions-dir", type=Path, default=ROOT / "dt-models")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="dt-moss-i4x-native-") as temporary:
        encoder = NativeI4xEncoder(args.ccv_source, temporary)
        report = quantize_moss_checkpoint(args.source, args.output, encoder, jobs=args.jobs)
    if args.prepare_inference is not None:
        report["inference"] = prepare_moss_bundle(args.output, args.prepare_inference, args.companions_dir, quantization="i4x")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
