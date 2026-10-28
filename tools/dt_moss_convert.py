#!/usr/bin/env python3
"""Strict source-to-checkpoint conversion for the supported MOSS Image architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_direct_ckpt_builder import convert_source_to_ckpt
from tools.dt_keymap_compiler import build_moss_image_contract
from tools.dt_weellm.seek_io import open_seeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import DATATYPE_NAME_TO_CODE

ROOT = Path(__file__).resolve().parents[1]
COMPANIONS = {
    "qwen_3.5_9b_i5x.ckpt": "559d41f5e6721b4edb5a661c42b6f328a243876c2637cad36cf9937fb0d453c1",
    "flux_2_vae_f16.ckpt": "48369d4b1495aec3df0579591f4bf9c199a73cc288130df2359c4a173c5ae864",
}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_moss_checkpoint(path):
    path = Path(path).resolve()
    if not path.is_file() or Path(str(path) + "-tensordata").exists():
        raise ValueError("Expected an existing converted checkpoint without a sidecar")
    contract = build_moss_image_contract()
    expected = {name: (rule["target_shapes"][index], DATATYPE_NAME_TO_CODE[rule["f16_store_datatype"]])
                for rule in contract["tensors"].values() for index, name in enumerate(rule["dt_names"])}
    with DrawThingsCkptStore(path) as store:
        if set(store.list_keys()) != set(expected):
            raise ValueError("Converted MOSS keyset differs from the complete native contract")
        for name, (shape, datatype) in expected.items():
            metadata = store.read_metadata(name)
            expected_bytes = math.prod(shape) * (4 if datatype == DATATYPE_NAME_TO_CODE["F32"] else 2)
            data_len = metadata.data_len
            if metadata.is_external:
                descriptor = store.con.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
                if not store.trailer_offset or len(descriptor) != 16 or metadata.has_splits:
                    raise ValueError(f"Invalid native trailer descriptor: {name}")
                offset, data_len = struct.unpack("<QQ", descriptor)
                if offset % 16384 or store.trailer_offset + offset + data_len > path.stat().st_size:
                    raise ValueError(f"Invalid native trailer payload range: {name}")
            if metadata.shape != shape or metadata.datatype != datatype or metadata.codec_identifier not in (0, 0x10000000) or metadata.reserved or data_len != expected_bytes:
                raise ValueError(f"Converted MOSS tensor schema mismatch: {name}")
        if store.con.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Converted checkpoint failed SQLite checking")
        storage_format = "native_trailer" if store.trailer_offset else "inline_sqlite"
    return {"result": "PASS", "tensor_count": len(expected), "storage_format": storage_format}


def prepare_native_trailer(source, target):
    if target.exists():
        raise FileExistsError(f"Trailer output already exists: {target}")
    alignment = 16384
    with tempfile.TemporaryDirectory(prefix=".moss-trailer.", dir=target.parent) as temporary:
        metadata_path = Path(temporary) / "metadata.ckpt"
        payload_path = Path(temporary) / "payload.bin"
        with DrawThingsCkptStore(source) as original, DrawThingsCkptStore(metadata_path, writable=True) as metadata_store, payload_path.open("xb") as payload_stream:
            for name in original.list_keys():
                metadata, payload = original.read_raw_tensor(name)
                if metadata.codec_identifier & ~0x10000000 or metadata.reserved:
                    raise ValueError(f"Trailer packaging requires raw tensors: {name}")
                if metadata.datatype not in (DATATYPE_NAME_TO_CODE["F16"], DATATYPE_NAME_TO_CODE["F32"]):
                    raise ValueError(f"Unsupported trailer tensor precision: {name}")
                dtype = "<f4" if metadata.datatype == DATATYPE_NAME_TO_CODE["F32"] else "<f2"
                if not np.isfinite(np.frombuffer(payload, dtype=dtype)).all():
                    raise ValueError(f"Cannot package nonfinite tensor: {name}")
                offset = (payload_stream.tell() + alignment - 1) // alignment * alignment
                payload_stream.write(b"\0" * (offset - payload_stream.tell()))
                payload_stream.write(payload)
                descriptor = struct.pack("<QQ", offset, len(payload))
                metadata_store.con.execute(
                    "INSERT INTO tensors (name, type, format, datatype, dim, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (name, metadata.tensor_type | (0x10000000 << 32), metadata.tensor_format,
                     metadata.datatype, metadata.dim_blob, descriptor),
                )
        sqlite_bytes = metadata_path.stat().st_size
        trailer_offset = (sqlite_bytes + alignment - 1) // alignment * alignment
        if trailer_offset > 0xFFFFFFFF:
            raise ValueError("Native trailer metadata exceeds its 32-bit offset")
        with target.open("xb") as output_stream, metadata_path.open("rb") as metadata_stream, payload_path.open("rb") as payload_stream:
            shutil.copyfileobj(metadata_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.write(b"\0" * (trailer_offset - sqlite_bytes))
            shutil.copyfileobj(payload_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.seek(60)
            output_stream.write(struct.pack(">I", trailer_offset))
            output_stream.flush()
            os.fsync(output_stream.fileno())


def prepare_moss_bundle(checkpoint, destination, companions_dir):
    checkpoint = Path(checkpoint).resolve()
    destination = Path(destination).expanduser().resolve()
    companions_dir = Path(companions_dir).resolve()
    if not destination.is_relative_to(ROOT / "output") or destination == ROOT / "output":
        raise ValueError("Inference bundles must be new directories under this workspace's output/")
    if destination.exists():
        raise FileExistsError(f"Inference bundle already exists: {destination}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.ckpt", checkpoint.name):
        raise ValueError("Bundle checkpoint basename must use letters, digits, dot, underscore or hyphen")
    if checkpoint.name in COMPANIONS:
        raise ValueError("Converted model basename collides with a companion")
    validation = validate_moss_checkpoint(checkpoint)
    sources = [checkpoint] + [companions_dir / name for name in COMPANIONS]
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing regular model payload: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimated_peak_bytes = 2 * checkpoint.stat().st_size + sum(source.stat().st_size for source in sources[1:])
    if shutil.disk_usage(destination.parent).free < estimated_peak_bytes + 1024 ** 3:
        raise ValueError(f"Insufficient filesystem space: bundle preparation needs approximately {estimated_peak_bytes / 1024 ** 3:.1f} GiB plus 1 GiB reserve")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        models = staging / "models"
        models.mkdir()
        records = []
        for source in sources:
            digest = file_digest(source)
            if source.name in COMPANIONS and digest != COMPANIONS[source.name]:
                raise ValueError(f"Companion hash differs from the validated payload: {source.name}")
            target = models / source.name
            if source == checkpoint:
                prepare_native_trailer(source, target)
                validation = validate_moss_checkpoint(target)
            else:
                subprocess.run(["cp", "--reflink=auto", "--no-clobber", "--", str(source), str(target)], check=True)
            target.chmod(0o600)
            if target.stat().st_nlink != 1 or (source.stat().st_dev, source.stat().st_ino) == (target.stat().st_dev, target.stat().st_ino):
                raise ValueError("Inference payloads must be independent writable copies")
            target_digest = file_digest(target)
            if source != checkpoint and target_digest != digest:
                raise ValueError(f"Inference copy hash mismatch: {source.name}")
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            records.append({"file": source.name, "source": str(source), "source_sha256": digest, "sha256": target_digest,
                            "bytes": target.stat().st_size})
        catalog = [{"name": checkpoint.stem, "file": checkpoint.name, "version": "moss_image",
                    "prefix": "", "upcast_attention": False, "modifier": "none", "default_scale": 16, "hires_fix_scale": 24,
                    "text_encoder": "qwen_3.5_9b_i5x.ckpt", "autoencoder": "flux_2_vae_f16.ckpt",
                    "objective": {"u": {"condition_scale": 1000}}}]
        catalog_path = models / "custom.json"
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
        report = {"schema_version": 1, "profile": "moss_image", "model": checkpoint.name,
                  "validation": validation, "files": records, "catalog_sha256": file_digest(catalog_path)}
        (staging / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
        for path in (catalog_path, staging / "bundle.json"):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"result": "PASS", "bundle": str(destination), "model": checkpoint.name,
            "inference_command": f'bash tools/dt_image.sh --bundle "{destination}" "Your prompt"'}


def verify_moss_bundle(destination):
    destination = Path(destination).expanduser().resolve()
    if not destination.is_relative_to(ROOT / "output"):
        raise ValueError("Only private inference bundles under output/ can be selected")
    report = json.loads((destination / "bundle.json").read_text())
    if report.get("schema_version") != 1 or report.get("profile") != "moss_image":
        raise ValueError("Unsupported MOSS inference bundle")
    model = report.get("model", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.ckpt", model):
        raise ValueError("Invalid bundle model basename")
    records = report.get("files", [])
    if len(records) != 3 or {record["file"] for record in records} != {model, *COMPANIONS}:
        raise ValueError("Incomplete bundle payload list")
    models = destination / "models"
    if models.is_symlink():
        raise ValueError("Bundle models directory cannot be a symlink")
    for record in records:
        path = models / record["file"]
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or not os.access(path, os.W_OK):
            raise ValueError(f"Bundle payload is not an independent writable file: {path.name}")
        if path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
            raise ValueError(f"Bundle payload changed: {path.name}")
        if path.name in COMPANIONS and record["sha256"] != COMPANIONS[path.name]:
            raise ValueError(f"Unvalidated companion: {path.name}")
        original = Path(record["source"])
        if original.exists() and (original.stat().st_dev, original.stat().st_ino) == (path.stat().st_dev, path.stat().st_ino):
            raise ValueError("Bundle payload aliases its published source")
    catalog = models / "custom.json"
    if catalog.is_symlink() or file_digest(catalog) != report["catalog_sha256"]:
        raise ValueError("Private model catalog changed")
    specifications = json.loads(catalog.read_text())
    if not isinstance(specifications, list) or len(specifications) != 1:
        raise ValueError("Private model catalog must contain exactly one specification")
    specification = specifications[0]
    required = {"file": model, "prefix": "", "version": "moss_image", "upcast_attention": False,
                "default_scale": 16, "text_encoder": "qwen_3.5_9b_i5x.ckpt", "autoencoder": "flux_2_vae_f16.ckpt",
                "objective": {"u": {"condition_scale": 1000}}}
    if not isinstance(specification, dict) or not isinstance(specification.get("name"), str) or any(specification.get(key) != value for key, value in required.items()):
        raise ValueError("Private model catalog is incompatible with the native MOSS specification")
    if validate_moss_checkpoint(models / model)["storage_format"] != "native_trailer":
        raise ValueError("Regenerate the inference bundle with offline native trailer storage")
    return model


def inspect_moss_source(seeker, contract=None):
    contract = contract or build_moss_image_contract()
    available = set(seeker.keys())
    consumed = set()
    generated_buffers = []
    missing = []
    mismatched = []
    for source, rule in contract["tensors"].items():
        resolved = seeker.resolve_key(source, aliases=rule.get("aliases"), profile="moss_image")
        if resolved is None:
            missing.append(source)
            continue
        if resolved in consumed:
            raise ValueError(f"Multiple MOSS rules resolve to the same source: {resolved}")
        consumed.add(resolved)
        source_dtype = getattr(seeker.tensors[resolved], "dtype", None)
        if source_dtype is not None and source_dtype not in ("BF16", "F16", "F32"):
            raise ValueError(f"Unsupported learned-weight precision for {resolved}: {source_dtype}")
        actual = list(seeker.tensors[resolved].shape)
        if actual != rule["source_shape"]:
            mismatched.append({"tensor": resolved, "expected": rule["source_shape"], "actual": actual})
    rotary_buffer = seeker.resolve_key("rope.bands", profile="moss_image")
    if rotary_buffer is not None:
        expected = np.power(10000.0, -np.arange(16, dtype=np.float64) / 16).astype(np.float32)
        values = seeker.read_f32(rotary_buffer)
        if values.shape != (16,) or not np.isfinite(values).all() or not np.allclose(values, expected, rtol=1e-6, atol=1e-9):
            raise ValueError("Unsupported MOSS rope.bands: source rotary frequencies differ from the native runtime")
        consumed.add(rotary_buffer)
        generated_buffers.append(rotary_buffer)
    extra = sorted(available - consumed)
    if missing or mismatched or extra:
        raise ValueError("Unsupported or incomplete MOSS architecture: " + json.dumps({
            "missing_count": len(missing), "missing_examples": missing[:8],
            "shape_mismatches": mismatched[:8], "unmapped_examples": extra[:8],
        }))
    return {"result": "PASS", "profile": "moss_image", "architecture": contract["architecture"],
            "source_tensors": len(consumed) - len(generated_buffers), "target_tensors": contract["expected_ckpt_tensor_count"],
            "validated_generated_buffers": generated_buffers,
            "fp32_preserved_tensors": [source for source, rule in contract["tensors"].items()
                                       if rule["f16_store_datatype"] == "F32"]}


def inspect_moss_config(path):
    config = json.loads(Path(path).read_text())
    expected = {"dim": 3072, "ffn_dim": 8192, "freq_dim": 256, "latent_channels": 128,
                "num_heads": 24, "num_layers": 42, "rope_base": 10000.0, "text_dim": 4096}
    if not isinstance(config, dict) or any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("MOSS config is incompatible with the supported native architecture")
    return {"file": str(Path(path).resolve()), "validated": expected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="Supported single-file or indexed safetensors source")
    parser.add_argument("--inspect", action="store_true", help="Validate source metadata without converting")
    parser.add_argument("--output", type=Path, help="New mixed F16/F32 Draw Things .ckpt path")
    parser.add_argument("--config", type=Path, help="Source transformer config (otherwise validate adjacent config.json if present)")
    parser.add_argument("--prepare-inference", type=Path, help="Publish an independent inference bundle under output/")
    parser.add_argument("--prepare-existing", type=Path, help="Prepare a previously converted inline MOSS checkpoint")
    parser.add_argument("--companions-dir", type=Path, default=ROOT / "dt-models")
    parser.add_argument("--bundle-model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.bundle_model is not None:
        if args.source is not None or args.output is not None or args.prepare_existing is not None:
            parser.error("bundle verification cannot be combined with conversion")
        print(verify_moss_bundle(args.bundle_model))
        return 0
    if args.prepare_existing is not None:
        if args.source is not None or args.output is not None or args.prepare_inference is None:
            parser.error("--prepare-existing requires --prepare-inference and no source/--output")
        print(json.dumps(prepare_moss_bundle(args.prepare_existing, args.prepare_inference, args.companions_dir), indent=2))
        return 0
    if args.source is None:
        parser.error("source is required")
    if args.inspect and (args.output is not None or args.prepare_inference is not None):
        parser.error("--inspect cannot publish output or an inference bundle")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error("source must be an existing checkpoint file or safetensors index; directories are not inferred")
    if source.suffix != ".safetensors" and not source.name.endswith(".index.json"):
        parser.error("this converter accepts only full-precision safetensors or an explicit safetensors index")
    if not args.inspect and args.output is None:
        parser.error("--output is required unless --inspect is used")
    if args.output is not None and args.output.suffix != ".ckpt":
        parser.error("--output must end in .ckpt")
    contract = build_moss_image_contract()
    with open_seeker(source) as seeker:
        report = inspect_moss_source(seeker, contract)
    config = args.config or source.parent / "config.json"
    if config.is_file():
        report["source_config"] = inspect_moss_config(config)
    elif args.config is not None:
        parser.error("explicit source config does not exist")
    contract["ignored_source_patterns"] = ["^" + re.escape(key) + "$" for key in report["validated_generated_buffers"]]
    report["source"] = str(source)
    if not args.inspect:
        output = args.output.expanduser().resolve()
        with tempfile.TemporaryDirectory(prefix="dt-moss-contract-") as temporary:
            contract_path = Path(temporary) / "contract.json"
            contract_path.write_text(json.dumps(contract, indent=2) + "\n")
            report["conversion"] = convert_source_to_ckpt(source, contract_path, output)
        report["validation"] = validate_moss_checkpoint(output)
        report["output"] = str(output)
        report["storage_policy"] = "F16 with original-value F32 timestep-output projection"
        if args.prepare_inference is not None:
            report["inference"] = prepare_moss_bundle(output, args.prepare_inference, args.companions_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)