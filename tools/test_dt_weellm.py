import json
import hashlib
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.dt_direct_ckpt_builder import (
    convert_source_to_ckpt,
    diff_ckpt_stores,
    load_contract,
    restore_norms_from_contract,
)
from tools.dt_keymap_compiler import (
    audit_swift_mapper_source,
    build_ltx23_contract,
    build_moss_image_contract,
)
from tools.dt_layer_parity_oracle import (
    compare_native_capture,
    compare_moss_block_capture,
    inspect_capture_tensor,
    inspect_ckpt_health,
    reference_gated_delta_rule,
    reference_rmsnorm_gated,
    verify_block_parity,
)
from tools.dt_lora_universal_bridge import normalize_lora_safetensors, verify_converted_lora
from tools.dt_validate_converted_ckpt import validate_ckpt
from tools.dt_weellm.dequant import (
    dequantize_fp8_e4m3,
    dequantize_fp8_e5m2,
    dequantize_gguf_q4_0,
    dequantize_gguf_q8_0,
    dequantize_gguf_tensor,
)
from tools.dt_weellm.seek_io import GGUFSeeker, SafetensorsSeeker, GGML_BLOCK_INFO
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
    bf16_bytes_to_f32,
    decode_dim_blob,
    encode_dim_blob,
    f32_to_bf16_bytes,
    rotary_deinterleave,
    rotary_interleave,
    split_tensor_chunks,
)


def write_mock_safetensors(path: Path, tensors: dict) -> None:
    """Write a valid `.safetensors` file from `{name: (dtype_str, shape, raw_bytes)}`."""
    header = {}
    blobs = []
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        blobs.append(raw)
        offset = end
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


class CheckpointSafetyTests(unittest.TestCase):
    def test_comparison_enforces_explicit_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "c.json"
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {
                "source": {"dt_names": ["expected"]}}}))
            paths = [root / name for name in ("candidate.ckpt", "reference.ckpt")]
            for path in paths:
                with DrawThingsCkptStore(path, writable=True) as store:
                    store.write_tensor("different", np.ones(2))
            self.assertTrue(diff_ckpt_stores(*paths)["is_equivalent"])
            self.assertFalse(diff_ckpt_stores(*paths, contract_path=contract)["is_equivalent"])

    def test_writer_rejects_bfloat16_overflow_and_empty_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            with DrawThingsCkptStore(Path(temporary) / "values.ckpt", writable=True) as store:
                with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError, "overflow"):
                    store.write_tensor("huge", np.array([np.finfo(np.float32).max]), datatype=DT_BFLOAT16)
                with self.assertRaises(ValueError):
                    store.write_tensor("empty", np.array([], dtype=np.float32))

    def test_shell_contract_failures_leave_catalogs_unchanged(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".venv").symlink_to(repository / ".venv", target_is_directory=True)
            tools = root / "tools"
            tools.mkdir()
            models = root / "dt-models"
            models.mkdir()
            for path in (repository / "tools").glob("*.py"):
                (tools / path.name).symlink_to(path)
            (tools / "dt_weellm").symlink_to(repository / "tools" / "dt_weellm", target_is_directory=True)
            for name in ("dt_convert.sh", "dt_quantize.sh", "dt_lora_convert.sh"):
                shutil.copyfile(repository / "tools" / name, tools / name)
            source = root / "lora.safetensors"
            values = np.ones((2, 2), dtype="<f4")
            write_mock_safetensors(source, {
                "module.lora_A.weight": ("F32", values.shape, values.tobytes()),
                "module.lora_B.weight": ("F32", values.shape, values.tobytes())})
            contract = root / "contract.json"
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "model_version": "ltx2.3",
                                           "tensors": {"module.weight": {"dt_names": ["__dit__[module]"]}}}))
            with DrawThingsCkptStore(models / "style_lora_f16.ckpt", writable=True):
                pass
            original = '[{"name":"preserve-me"}]\n'
            for name in ("custom.json", "custom_lora.json"):
                (models / name).write_text(original)
            commands = [
                ["dt_convert.sh", str(source), "--direct", "--contract", str(root / "missing.json")],
                ["dt_quantize.sh", "missing", "--contract", str(root / "missing.json")],
                ["dt_lora_convert.sh", str(source), "--name", "style", "--verify-post"],
                ["dt_lora_convert.sh", str(source), "--name", "style", "--verify-post", "--contract", str(contract)],
            ]
            for command in commands:
                result = subprocess.run(["bash", str(tools / command[0]), *command[1:]],
                                        cwd=root, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                if command[-1] == str(contract):
                    self.assertIn("Missing converted LoRA tensors", result.stdout)
                for name in ("custom.json", "custom_lora.json"):
                    self.assertEqual((models / name).read_text(), original)
            model_source = root / "model.safetensors"
            write_mock_safetensors(model_source, {"module.weight": ("F32", values.shape, values.tobytes())})
            shards = root / "shards"
            shards.mkdir()
            shutil.copyfile(model_source, shards / "part.safetensors")
            (shards / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"module.weight": "part.safetensors"}}))
            for name, input_path in (("single", model_source), ("sharded", shards)):
                result = subprocess.run(
                    ["bash", str(tools / "dt_convert.sh"), str(input_path), "--direct", "--contract", str(contract), "--name", name],
                    cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((models / f"{name}_f16.ckpt").exists())
                self.assertEqual((models / "custom.json").read_text(), original)
            quant_contract = root / "quant_contract.json"
            quant_contract.write_text(json.dumps({
                "schema_version": 1, "profile": "test", "model_version": "ltx2.3",
                "companions": {"autoencoder": "vae.ckpt", "text_encoder": "text.ckpt", "modifier": "none",
                               "default_scale": 1, "objective": {"u": {"condition_scale": 1000}}},
                "tensors": {"module.weight": {"dt_names": ["__dit__[module]"], "quant_restore_policy": "f32_from_source"}}}))
            with DrawThingsCkptStore(models / "preserved_f16.ckpt", writable=True) as store:
                store.write_tensor("__dit__[module]", values)
            quantized = models / "preserved_q6p.ckpt"
            with DrawThingsCkptStore(quantized, writable=True) as store:
                store.write_tensor("wrong_target", values)
            original_checkpoint = quantized.read_bytes()
            result = subprocess.run(["bash", str(tools / "dt_quantize.sh"), "preserved", "--restore-only",
                                     "--contract", str(quant_contract)], cwd=root,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Required restoration tensors are missing", result.stderr)
            self.assertEqual(quantized.read_bytes(), original_checkpoint)
            self.assertEqual((models / "custom.json").read_text(), original)
            self.assertFalse(list(models.glob(".quant.*")))

    def test_structural_validation_requires_complete_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, contract = root / "w.ckpt", root / "c.json"
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {
                "source": {"dt_names": ["__dit__[expected]"], "target_shapes": [[2]]}}}))
            with DrawThingsCkptStore(path, writable=True) as store:
                store.write_tensor("__dit__[arbitrary]", np.ones(2))
            self.assertEqual(validate_ckpt(path, "none", None, None, "any", 10, contract), 1)
            with DrawThingsCkptStore(path, writable=True) as store:
                store.con.execute("DELETE FROM tensors")
                store.write_tensor("__dit__[expected]", np.ones(2))
            self.assertEqual(validate_ckpt(path, "none", None, None, "any", 10, contract), 0)

    def test_health_rejects_empty_unknown_and_missing_contract_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "health.ckpt"
            with DrawThingsCkptStore(path, writable=True):
                pass
            self.assertFalse(inspect_ckpt_health(path)[0])
            with DrawThingsCkptStore(path, writable=True) as store:
                store.write_tensor("weight", np.ones(2))
                store.con.execute("UPDATE tensors SET datatype=123456")
            self.assertFalse(inspect_ckpt_health(path)[0])

    def test_mixed_fused_conversion_and_weight_check_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, contract, output = root / "s.safetensors", root / "c.json", root / "o.ckpt"
            rule = {"dt_names": ["q", "k", "v"], "rotary_interleaved_slices": [True, True, False],
                    "num_heads": 1, "head_dim": 4}
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {"block.qkv": rule}}))
            values = np.arange(24, dtype="<f4").reshape(12, 2)
            write_mock_safetensors(source, {"block.qkv": ("F32", values.shape, values.tobytes())})
            convert_source_to_ckpt(source, contract, output)
            passed, report = verify_block_parity(source, output, contract, block_prefix="block.")
            self.assertTrue(passed, report)

    def test_restore_preserves_matrix_rank_restores_bf16_and_clears_codec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, quantized, contract = root / "s.ckpt", root / "q.ckpt", root / "c.json"
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {
                "matrix": {"dt_names": ["matrix"], "quant_restore_policy": "f32_from_source", "dim_rank_policy": "preserve"},
                "embed": {"dt_names": ["embed"], "quant_restore_policy": "bf16_from_source"}}}))
            for path in (source, quantized):
                with DrawThingsCkptStore(path, writable=True) as store:
                    store.write_tensor("matrix", np.arange(6, dtype=np.float32).reshape(2, 3), datatype=DT_FLOAT32)
                    store.write_tensor("embed", np.ones(2), datatype=DT_BFLOAT16 if path == source else DT_FLOAT16)
                    if path == quantized:
                        store.con.execute("UPDATE tensors SET type=? WHERE name='matrix'", ((0x8a1e8b << 32) | 1,))
            restore_norms_from_contract(quantized, source, contract)
            with DrawThingsCkptStore(quantized) as store:
                self.assertEqual(store.read_metadata("matrix").shape, [2, 3])
                self.assertEqual(store.read_metadata("matrix").codec_identifier, 0)
                self.assertEqual(store.read_metadata("embed").datatype, DT_BFLOAT16)
            before = quantized.read_bytes()
            with DrawThingsCkptStore(source, writable=True) as store:
                store.con.execute("DELETE FROM tensors WHERE name='embed'")
            with self.assertRaisesRegex(ValueError, "missing"):
                restore_norms_from_contract(quantized, source, contract)
            self.assertEqual(quantized.read_bytes(), before)

    def test_conversion_rejects_unmapped_shapes_overflow_and_existing_output(self):
        for case in ("unmapped", "shape", "overflow", "existing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output, contract = root / "source.safetensors", root / "out.ckpt", root / "c.json"
                rule = {"dt_names": ["weight"], "source_shape": [2], "f16_store_datatype": "F16"}
                contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {"w": rule}}))
                values = np.array([1e10, 2.0] if case == "overflow" else [1.0, 2.0], dtype="<f4")
                tensors = {"w": ("F32", (1, 2) if case == "shape" else (2,), values.tobytes())}
                if case == "unmapped":
                    tensors["extra"] = ("F32", (2,), values.tobytes())
                write_mock_safetensors(source, tensors)
                if case == "existing":
                    output.write_bytes(b"preserve existing output")
                with np.errstate(over="ignore"), self.assertRaises((ValueError, FileExistsError)):
                    convert_source_to_ckpt(source, contract, output)
                if case == "existing":
                    self.assertEqual(output.read_bytes(), b"preserve existing output")
                else:
                    self.assertFalse(output.exists())
                self.assertFalse(list(root.glob(".out.ckpt.tmp.*")))

    def test_contract_rejects_duplicate_targets_and_invalid_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            base = {"schema_version": 1, "profile": "test", "tensors": {
                "a": {"dt_names": ["same"]}, "b": {"dt_names": ["same"]}}}
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "globally unique"):
                load_contract(path)
            base["tensors"] = {"a": {"dt_names": ["a"], "source_shape": [-1]}}
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "Invalid contract"):
                load_contract(path)

    def test_native_external_marker_and_split_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "store.ckpt"
            expected = np.arange(12, dtype=np.float32)
            with DrawThingsCkptStore(path, writable=True, chunk_bytes=8) as store:
                store.write_tensor("split", expected, datatype=DT_FLOAT32)
                store.write_tensor("external", np.zeros(4))
                store.con.execute(
                    "UPDATE tensors SET type=?, data=? WHERE name='external'",
                    ((0x10000000 << 32) | 1, struct.pack("<QQQ", 0, 16, 8)),
                )
            Path(str(path) + "-tensordata").write_bytes(
                bytes(16) + np.arange(4, dtype="<f2").tobytes()
            )
            with DrawThingsCkptStore(path) as store:
                self.assertTrue(store.read_metadata("split").has_splits)
                np.testing.assert_array_equal(store.read_f32("split")[1], expected)
                self.assertTrue(store.read_metadata("external").is_external)
                np.testing.assert_array_equal(store.read_f32("external")[1], np.arange(4))
            Path(str(path) + "-tensordata").unlink()
            with DrawThingsCkptStore(path) as store:
                with self.assertRaisesRegex(ValueError, "Missing sidecar"):
                    store.read_f32("external")

    def test_readonly_open_cannot_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.ckpt"
            with self.assertRaises(Exception):
                DrawThingsCkptStore(path)
            self.assertFalse(path.exists())

    def test_cli_help_from_outside_repository(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        for name in ("dt_keymap_compiler.py", "dt_direct_ckpt_builder.py",
                     "dt_layer_parity_oracle.py", "dt_lora_universal_bridge.py"):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [sys.executable, str(root / "tools" / name), "--help"],
                    cwd=temporary, env=environment, capture_output=True, text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_comparison_rejects_unchecked_or_incomplete_data(self):
        for case in ("empty", "malformed", "nan", "extra", "shape"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                candidate = Path(temporary) / "candidate.ckpt"
                reference = Path(temporary) / "reference.ckpt"
                for path in (candidate, reference):
                    with DrawThingsCkptStore(path, writable=True) as store:
                        if case != "empty":
                            store.write_tensor("weight", np.array([1.0, 2.0]))
                        if case == "malformed":
                            store.con.execute("UPDATE tensors SET data=?", (b"\x00",))
                        if path == candidate and case == "nan":
                            store.con.execute(
                                "UPDATE tensors SET data=?",
                                (np.array([np.nan, np.nan], dtype="<f2").tobytes(),),
                            )
                        if path == candidate and case == "extra":
                            store.write_tensor("extra", np.array([1.0]))
                        if path == candidate and case == "shape":
                            store.write_tensor("weight", np.array([[1.0, 2.0]]))
                self.assertFalse(diff_ckpt_stores(candidate, reference)["is_equivalent"])

    def test_comparison_requires_full_finite_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("candidate.ckpt", "reference.ckpt")]
            for path in paths:
                with DrawThingsCkptStore(path, writable=True) as store:
                    store.write_tensor("weight", np.array([1.0, 2.0]))
            report = diff_ckpt_stores(*paths)
            self.assertTrue(report["is_equivalent"])
            self.assertEqual(report["value_comparison_count"], 1)
            self.assertEqual(report["unreadable_tensor_count"], 0)


class CudaPatchBundleTests(unittest.TestCase):
    def test_prepare_generates_bindings_only_in_private_copy(self):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("ruby") or not (source / "lib/nnc/cmd/build-cmd.rb").is_file():
            self.skipTest("Binding preparation requires Ruby and pinned CCV sources")
        manifest = json.loads((root / "DRAW_THINGS_PATCH/cuda/manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary)
            for relative, baseline in manifest["files"].items():
                if baseline is not None:
                    destination = outside / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, destination)
            before = {relative: (outside / relative).read_bytes() for relative, baseline in manifest["files"].items() if baseline is not None}
            result = subprocess.run([sys.executable, str(root / "tools/drawthings_cuda_patch.py"), "prepare", "--ccv-source", str(outside)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, {relative: (outside / relative).read_bytes() for relative in before})
            self.assertTrue(all(not (outside / relative).exists() for relative, baseline in manifest["files"].items() if baseline is None))
        with tempfile.TemporaryDirectory(prefix="bindings-test-", dir=root / "output/upgrades") as temporary:
            subprocess.run(["cp", "-a", "--reflink=auto", str(source) + "/.", temporary], check=True, timeout=60)
            result = subprocess.run([sys.executable, str(root / "tools/drawthings_cuda_patch.py"), "prepare", "--ccv-source", temporary],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["state"], "applied_and_bindings_generated")

    def test_build_helper_rejects_pristine_targets_and_unsafe_tags(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tools").mkdir()
            helper = root / "tools/drawthings_build_candidate.sh"
            shutil.copyfile(repository / "tools/drawthings_build_candidate.sh", helper)
            private = root / "output/upgrades/private-ccv"
            private.mkdir(parents=True)
            cases = [
                ({"DRAWTHINGS_CANDIDATE_TAG": "../unsafe"}, "artifact tag"),
                ({"DRAWTHINGS_CCV_OVERRIDE": str(private),
                  "DRAWTHINGS_CANDIDATE_DIR": str(root / "output/upgrades/community-0621d819")}, "separate candidate directory"),
            ]
            for overrides, message in cases:
                with self.subTest(overrides=overrides):
                    environment = dict(os.environ)
                    for key in ("DRAWTHINGS_CANDIDATE_TAG", "DRAWTHINGS_CCV_OVERRIDE", "DRAWTHINGS_CANDIDATE_DIR", "DRAWTHINGS_BAZEL_ROOT"):
                        environment.pop(key, None)
                    environment.update(overrides)
                    result = subprocess.run(["bash", str(helper), "--check"], env=environment,
                                            capture_output=True, text=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)

    def test_strict_apply_is_idempotent_and_rejects_drift(self):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        manifest = json.loads((root / "DRAW_THINGS_PATCH/cuda/manifest.json").read_text())
        if not all((source / relative).is_file() for relative, baseline in manifest["files"].items() if baseline is not None):
            self.skipTest("CUDA patch verification needs the pinned CCV base files")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for relative, baseline in manifest["files"].items():
                if baseline is not None:
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, destination)
            untouched = target / "Package.swift"
            untouched.write_text("preserve this CPU/build configuration\n")
            command = [sys.executable, str(root / "tools/drawthings_cuda_patch.py"), "apply", "--ccv-source", str(target)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = {relative: hashlib.sha256((target / relative).read_bytes()).hexdigest() for relative in manifest["files"]}
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, {relative: hashlib.sha256((target / relative).read_bytes()).hexdigest() for relative in manifest["files"]})
            changed = target / next(iter(manifest["files"]))
            changed.write_bytes(changed.read_bytes() + b"\ndrift\n")
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing fallback", result.stderr)
            self.assertEqual(untouched.read_text(), "preserve this CPU/build configuration\n")


@unittest.skipUnless(os.environ.get("DT_RUN_CUDA_TESTS") == "1", "Set DT_RUN_CUDA_TESTS=1 for opt-in CUDA tests")
class CudaQwenReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        compiler = shutil.which("nvcc")
        if not compiler:
            raise RuntimeError("CUDA tests explicitly requested, but nvcc is unavailable")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        library_path = Path(temporary.name) / "qwen35_cuda.so"
        build = subprocess.run(
            [compiler, "-shared", "-Xcompiler=-fPIC", "-arch=" + os.environ.get("DT_CUDA_ARCH", "sm_86"),
             "-std=c++17", "-DHAVE_CUDA", "-I", str(root / "DRAW_THINGS_PATCH/cuda/ccv/lib"),
             "-I", str(source / "lib"), str(root / "tools/dt_qwen35_cuda_probe.cu"), "-o", str(library_path)],
            capture_output=True, text=True, timeout=120)
        if build.returncode:
            raise RuntimeError(build.stderr)
        cls.library = ctypes.CDLL(str(library_path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        cls.library.dt_cuda_rmsnorm.argtypes = [ctypes.c_int] * 7 + [pointer] * 4
        cls.library.dt_cuda_rmsnorm.restype = ctypes.c_int
        cls.library.dt_cuda_rmsnorm_contract.argtypes = [ctypes.c_int]
        cls.library.dt_cuda_rmsnorm_contract.restype = ctypes.c_int
        cls.library.dt_cuda_gated_delta.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            *([ctypes.c_int] * 5), *([pointer] * 8)]
        cls.library.dt_cuda_gated_delta.restype = ctypes.c_int
        cls.library.dt_cuda_gated_delta_contract.argtypes = [ctypes.c_int]
        cls.library.dt_cuda_gated_delta_contract.restype = ctypes.c_int

    @staticmethod
    def rounded(values, datatype):
        if datatype == DT_FLOAT16:
            return values.astype(np.float16).astype(np.float32)
        if datatype == DT_BFLOAT16:
            return bf16_bytes_to_f32(f32_to_bf16_bytes(values), values.shape)
        return values

    def test_gated_rmsnorm_matches_reference(self):
        rng = np.random.default_rng(4242)
        for columns in (1, 3, 33, 128, 257):
            for datatype in (DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16):
                values = rng.normal(size=(7, columns)).astype(np.float32)
                values[0] = 0
                gate = rng.normal(size=values.shape).astype(np.float32)
                weight = rng.normal(size=(columns,)).astype(np.float32)
                for affine in (0, 1):
                    for inplace in (0, 1):
                        with self.subTest(columns=columns, datatype=datatype, affine=affine, inplace=inplace):
                            actual = np.empty_like(values)
                            status = self.library.dt_cuda_rmsnorm(7, columns, datatype, DT_FLOAT32, DT_FLOAT16,
                                                                 affine, inplace, values, gate, weight, actual)
                            self.assertEqual(status, 0)
                            expected = reference_rmsnorm_gated(self.rounded(values, datatype), gate,
                                self.rounded(weight, DT_FLOAT16) if affine else np.ones_like(weight))
                            expected = self.rounded(expected, datatype)
                            self.assertTrue(np.isfinite(actual).all())
                            np.testing.assert_allclose(actual, expected,
                                atol=2e-6 if datatype == DT_FLOAT32 else 2e-3,
                                rtol=2e-5 if datatype == DT_FLOAT32 else 1e-2)

    def test_gated_rmsnorm_rejects_invalid_layout_and_accepts_empty(self):
        self.assertEqual(self.library.dt_cuda_rmsnorm_contract(0), 0)
        for mode in (1, 2, 3, 4):
            self.assertNotEqual(self.library.dt_cuda_rmsnorm_contract(mode), 0)

    def test_gated_delta_matches_reference(self):
        rng = np.random.default_rng(4242)
        for dimensions in ((1, 5, 2, 4, 3, 5), (2, 3, 1, 2, 4, 2), (1, 4, 16, 32, 128, 128)):
            batch, sequence, key_heads, value_heads, key_dim, value_dim = dimensions
            for datatype in (DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16):
                queries = rng.normal(size=(batch, sequence, key_heads, key_dim)).astype(np.float32) * 0.025
                keys = rng.normal(size=queries.shape).astype(np.float32) * 0.025
                values = rng.normal(size=(batch, sequence, value_heads, value_dim)).astype(np.float32)
                beta = np.full((batch, sequence, value_heads), 0.5, dtype=np.float32)
                initial = rng.normal(size=(batch, value_heads, value_dim, key_dim)).astype(np.float32) * 0.01
                for channel_decay in (0, 1):
                    shape = beta.shape + ((key_dim,) if channel_decay else ())
                    for log_decay in (0, 1):
                        decay = -np.abs(rng.normal(size=shape)).astype(np.float32) * 0.1 if log_decay else np.full(shape, 0.9, dtype=np.float32)
                        for checkpoints, inplace in ((0, 0), (0, 1), (2, 0)):
                            with self.subTest(dimensions=dimensions, datatype=datatype, channel_decay=channel_decay,
                                              log_decay=log_decay, checkpoints=checkpoints, inplace=inplace):
                                expected, state = reference_gated_delta_rule(
                                    self.rounded(queries, datatype), self.rounded(keys, datatype),
                                    self.rounded(values, datatype), self.rounded(beta, datatype),
                                    decay, initial, bool(log_decay), checkpoints)
                                expected = self.rounded(expected, datatype)
                                actual, final = np.empty_like(expected), np.empty_like(state)
                                status = self.library.dt_cuda_gated_delta(
                                    np.array(dimensions, dtype=np.int32), datatype, log_decay, channel_decay, checkpoints, inplace,
                                    queries, keys, values, decay, beta, initial, actual, final)
                                self.assertEqual(status, 0)
                                self.assertTrue(np.isfinite(actual).all() and np.isfinite(final).all())
                                np.testing.assert_allclose(actual, expected, atol=2e-6 if datatype == DT_FLOAT32 else 1e-4,
                                                           rtol=2e-5 if datatype == DT_FLOAT32 else 1e-2)
                                np.testing.assert_allclose(final, state, atol=2e-6, rtol=2e-5)

    def test_gated_delta_rejects_invalid_contract_and_accepts_empty(self):
        self.assertEqual(self.library.dt_cuda_gated_delta_contract(0), 0)
        for mode in (1, 2, 3, 4, 5):
            self.assertNotEqual(self.library.dt_cuda_gated_delta_contract(mode), 0)


class NativeCaptureTests(unittest.TestCase):
    @staticmethod
    def write_capture(prefix, role, values):
        values = np.asarray(values, dtype="<f4")
        header = struct.pack("<17I", 0, 1, 2, DT_FLOAT32, 0,
                             *values.shape, *([0] * (12 - values.ndim)))
        prefix.with_name(prefix.name + "_" + role + ".bin").write_bytes(header + values.tobytes())

    def test_linear_capture_is_finite_complete_and_cli_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "linear"
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            weights = np.array([[0.5, -1, 2]], dtype=np.float32)
            expected = values @ weights.T + 0.25
            for role, array in (("input_0", values), ("input_1", weights),
                                ("input_2", np.array([0.25])), ("output_0", expected)):
                self.write_capture(prefix, role, array)
            report = compare_native_capture(prefix, "linear", 1e-6, 1e-6)
            self.assertEqual(report["result"], "PASS")
            self.assertEqual(len(report["captures"]), 4)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in report["captures"].values()))
            destination = Path(temporary) / "report.json"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("dt_layer_parity_oracle.py")),
                                     "compare-capture", "--prefix", str(prefix), "--operation", "linear",
                                     "--atol", "0.000001", "--rtol", "0.000001", "--report", str(destination)],
                                    cwd=temporary, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout), json.loads(destination.read_text()))
            self.write_capture(prefix, "output_0", expected + 1)
            self.assertEqual(compare_native_capture(prefix, "linear", 1e-6, 1e-6)["result"], "FAIL")
            self.write_capture(prefix, "output_0", np.full_like(expected, np.nan))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare_native_capture(prefix, "linear", 1e-6, 1e-6)

    def test_grouped_attention_capture_and_mask_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "attention"
            queries = np.zeros((1, 2, 2, 2), dtype=np.float32)
            keys = np.zeros((1, 3, 1, 2), dtype=np.float32)
            values = np.arange(6, dtype=np.float32).reshape(1, 3, 1, 2)
            expected = np.broadcast_to(values.mean(axis=1, keepdims=True), queries.shape).copy()
            for role, array in (("input_0", queries), ("input_1", keys), ("input_2", values),
                                ("output_0", expected), ("output_1", np.full((1, 2, 2), np.log(3)))):
                self.write_capture(prefix, role, array)
            report = compare_native_capture(prefix, "sdpa", 1e-6, 1e-6)
            self.assertEqual(report["result"], "PASS")
            self.assertEqual(len(report["comparisons"]), 2)
            self.write_capture(prefix, "input_3", np.zeros((1, 1, 2, 3)))
            with self.assertRaisesRegex(ValueError, "Masked"):
                compare_native_capture(prefix, "sdpa", 1e-6, 1e-6)

    def test_rotary_capture_and_invalid_metadata_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "rotary"
            values = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 2, 4)
            rotation = np.array([0, -1, 1, 0], dtype=np.float32).reshape(1, 1, 1, 4)
            expected = values.copy()
            expected[..., 0] = values[..., 1]
            expected[..., 1] = -values[..., 0]
            for role, array in (("input_0", values), ("input_1", rotation), ("output_0", expected)):
                self.write_capture(prefix, role, array)
            self.assertEqual(compare_native_capture(prefix, "rotary", 1e-6, 1e-6)["result"], "PASS")
            with self.assertRaisesRegex(ValueError, "finite"):
                compare_native_capture(prefix, "rotary", float("nan"), 0)
            path = prefix.with_name(prefix.name + "_input_0.bin")
            malformed = bytearray(path.read_bytes())
            struct.pack_into("<12I", malformed, 20, 1, 0, 1, 2, 4, *([0] * 7))
            path.write_bytes(malformed)
            with self.assertRaisesRegex(ValueError, "zero padding"):
                compare_native_capture(prefix, "rotary", 1e-6, 1e-6)

    def test_inspect_capture_tensor_reports_sequence_and_spatial_periodicity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seq_prefix = root / "seq"
            seq_vals = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4)
            self.write_capture(seq_prefix, "input_0", seq_vals)
            seq_file = seq_prefix.with_name("seq_input_0.bin")
            seq_report = inspect_capture_tensor(seq_file)
            self.assertEqual(seq_report["shape"], [1, 3, 4])
            self.assertIn("sequence_diagnostics", seq_report)
            self.assertEqual(seq_report["sequence_diagnostics"]["tokens"], 3)

            # Create a 4D NHWC tensor with deliberate 2x2 alternating row checkerboard artifact
            grid_prefix = root / "grid"
            grid = np.zeros((1, 4, 4, 2), dtype=np.float32)
            grid[:, 0::2, :, :] = 1.0
            grid[:, 1::2, :, :] = -1.0
            self.write_capture(grid_prefix, "output_0", grid)
            grid_file = grid_prefix.with_name("grid_output_0.bin")
            grid_report = inspect_capture_tensor(grid_file)
            self.assertIn("spatial_periodicity_2x2", grid_report)
            self.assertGreater(grid_report["spatial_periodicity_2x2"]["checkerboard_ratio_h"], 100.0)

    def test_inspect_capture_checks_all_batches_and_zero_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "batch"
            values = np.array([[[1, 0], [0, 1], [1, 1]], [[2, 2], [2, 2], [0, 0]]], dtype=np.float32)
            self.write_capture(prefix, "input_0", values)
            report = inspect_capture_tensor(prefix.with_name("batch_input_0.bin"))
            sequence = report["sequence_diagnostics"]
            self.assertEqual(sequence["batches"], 2)
            self.assertEqual(sequence["zero_norm_tokens"], 1)
            self.assertEqual(len(sequence["per_batch"]), 2)
            self.assertAlmostEqual(sequence["per_batch"][1]["adjacent_token_cosine_mean"], 1)
            self.assertIn("channel_statistics", report)

    def test_inspect_capture_does_not_label_nchw_as_spatial_nhwc(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "layout"
            self.write_capture(prefix, "input_0", np.ones((1, 4, 5, 6)))
            path = prefix.with_name("layout_input_0.bin")
            blob = bytearray(path.read_bytes())
            struct.pack_into("<I", blob, 8, 1)
            path.write_bytes(blob)
            report = inspect_capture_tensor(path)
            self.assertNotIn("spatial_periodicity_2x2", report)
            self.assertEqual(report["channel_statistics"]["axis"], 1)
            self.assertEqual(report["channel_statistics"]["channels"], 4)


class MossBlockCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "block"
        self.directory.mkdir()
        self.operations = []
        initial = np.arange(1, 9, dtype=np.float32).reshape(1, 2, 4) / 8
        normalized = initial / np.sqrt(np.mean(initial ** 2, axis=-1, keepdims=True) + 1e-6)
        zero = np.zeros_like(initial)
        heads = zero.reshape(1, 2, 1, 4)
        wide = np.zeros((1, 2, 8), dtype=np.float32)
        ones = np.ones((1, 1, 4), dtype=np.float32)
        zeros = np.zeros_like(ones)
        square = np.zeros((4, 4), dtype=np.float32)
        expansion = np.zeros((8, 4), dtype=np.float32)
        rotation = np.array([1, 0, 1, 0], dtype=np.float32).reshape(1, 1, 1, 4)

        def add(name, inputs, output, cap=0, source=None):
            ordinal = len(self.operations)
            prefix = self.directory / ("op_%03d" % ordinal)
            tensors = {}
            for role, array in [("input_" + str(index), value) for index, value in enumerate(inputs)] + [("output_0", output)]:
                NativeCaptureTests.write_capture(prefix, role, array)
                tensors[role] = {"file": prefix.name + "_" + role + ".bin", "shape": list(array.shape),
                                 "data": ordinal * 100 + (90 if role == "output_0" else int(role[-1]))}
            if source is not None:
                tensors["input_0"]["data"] = self.operations[source]["tensors"]["output_0"]["data"]
            command = {"name": "CCV_NNC_" + name + "_FORWARD", "soft_cap": cap,
                       "rms_epsilon": 1e-6, "rms_axis_count": 1, "rms_axes": [inputs[0].ndim - 1, 0, 0],
                       "transpose_a": [0, 0], "transpose_b": [0, 1], "attention_scale": 1,
                       "scalars": [1, 1] if name == "ADD" else [1, 0] if name == "MUL" else [0.5, 0]}
            self.operations.append({"ordinal": ordinal, "status": 0, "command": command, "tensors": tensors})

        add("RMSNORM", [initial, ones], normalized)
        add("MUL", [normalized, ones], normalized)
        add("ADD", [normalized, zeros], normalized)
        add("DATATYPE_CONVERSION", [normalized], normalized)
        add("GEMM", [normalized, square], zero)
        add("RMSNORM", [heads, ones.reshape(1, 1, 1, 4)], heads)
        add("CMUL", [heads, rotation], heads)
        add("SCALAR_MUL", [heads], heads)
        add("GEMM", [normalized, square], zero)
        add("RMSNORM", [heads, ones.reshape(1, 1, 1, 4)], heads)
        add("CMUL", [heads, rotation], heads)
        add("GEMM", [normalized, square], zero)
        add("SCALED_DOT_PRODUCT_ATTENTION", [heads, heads, heads], heads)
        add("GEMM", [zero, square], zero)
        add("DATATYPE_CONVERSION", [zero], zero)
        add("MUL", [zero, ones], zero)
        add("ADD", [initial, zero], initial)
        add("RMSNORM", [initial, ones], normalized)
        add("MUL", [normalized, ones], normalized)
        add("ADD", [normalized, zeros], normalized)
        add("DATATYPE_CONVERSION", [normalized], normalized)
        add("GEMM", [normalized, expansion], wide)
        add("GEMM", [normalized, expansion], wide)
        add("TANH", [wide], wide, cap=25, source=22)
        add("SIGMOID", [wide], np.full_like(wide, 0.5))
        add("TANH", [wide], wide, cap=4, source=21)
        add("MUL", [wide, np.full_like(wide, 0.5)], wide)
        add("MUL", [wide, wide], wide)
        add("GEMM", [wide, expansion.T.copy()], zero)
        add("DATATYPE_CONVERSION", [zero], zero)
        add("MUL", [zero, ones], zero)
        add("ADD", [initial, zero], initial)
        NativeCaptureTests.write_capture(self.directory / "block", "output", initial)
        self.manifest = {"complete": True, "operations": self.operations,
                         "block_output": {"file": "block_output.bin"}}
        self.save_manifest()

    def save_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def test_complete_block_formula_and_cli(self):
        report = compare_moss_block_capture(self.directory, 1e-6, 1e-6)
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(len(report["checks"]), 28)
        self.assertEqual(report["block_relative_rms_error"], 0)
        self.assertFalse(report["checkpoint_assignment_verified_by_this_command"])
        destination = self.root / "report.json"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("dt_layer_parity_oracle.py")),
                                 "compare-moss-block", "--capture-dir", str(self.directory), "--report", str(destination)],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), json.loads(destination.read_text()))

    def test_block_output_corruption_fails_without_reference_reset(self):
        changed = np.full((1, 2, 4), 7, dtype=np.float32)
        NativeCaptureTests.write_capture(self.directory / "op_031", "output_0", changed)
        NativeCaptureTests.write_capture(self.directory / "block", "output", changed)
        report = compare_moss_block_capture(self.directory)
        self.assertEqual(report["result"], "FAIL")
        self.assertFalse(report["checks"][-1]["matches"])

    def test_incomplete_unsafe_and_nonfinite_block_refusal(self):
        self.manifest["complete"] = False
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "complete"):
            compare_moss_block_capture(self.directory)
        self.manifest["complete"] = True
        self.operations[0]["tensors"]["input_0"]["file"] = "../outside.bin"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "path"):
            compare_moss_block_capture(self.directory)
        self.operations[0]["tensors"]["input_0"]["file"] = "op_000_input_0.bin"
        self.save_manifest()
        NativeCaptureTests.write_capture(self.directory / "op_000", "input_0", np.full((1, 2, 4), np.nan))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare_moss_block_capture(self.directory)


class NativeQwenReferenceTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DT_RUN_CUDA_TESTS") == "1", "Set DT_RUN_CUDA_TESTS=1 for native CUDA execution")
    def test_native_backend_cli_accepts_qwen_cuda_and_executes(self):
        root = Path(__file__).resolve().parents[1]
        candidate = Path(os.environ.get("DT_GRPC_CANDIDATE", str(root / "output/upgrades/community-0621d819-qwen35-cuda/bazel-bin/Apps/gRPCServerCLI")))
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("lldb") or not shutil.which("cc") or not candidate.is_file() or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("Native CUDA integration requires LLDB, compiler, and separately built candidate/source")
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "native-backends.json"
            result = subprocess.run(
                [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "native-backends",
                 "--binary", str(candidate), "--ccv-source", str(source), "--require-qwen-cuda",
                 "--execute-qwen", "--execute-sdpa", "--execute-embedding", "--execute-packed-gemm", "--report", str(report_path)],
                capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report, json.loads(report_path.read_text()))
            self.assertTrue(report["qwen_cuda_registered"])
            self.assertEqual(report["qwen_cuda_execution"], 0)
            self.assertEqual(report["sdpa_contracts"], [0, -1, -1, -1, -1, -1, 0, -1])
            self.assertEqual(report["sdpa_cuda_execution"], 0)
            self.assertEqual(report["packed_embedding_cuda_execution"], 0)
            self.assertEqual(report["packed_gemm_cuda_execution"], 0)
            self.assertEqual(len(report["checks"]), 9)
            self.assertTrue(all(check["backend"] == check["expected"] for check in report["checks"]))
            self.assertEqual(report["binary_sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())

    def test_native_embedding_gate_rejects_pre_fix_binary(self):
        root = Path(__file__).resolve().parents[1]
        candidate = root / "output/upgrades/gRPCServerCLI-0621d819-before-packed-embedding"
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("lldb") or not shutil.which("cc") or not candidate.is_file() or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("Packed-embedding negative control requires the preserved pre-fix candidate/source")
        result = subprocess.run(
            [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "native-backends",
             "--binary", str(candidate), "--ccv-source", str(source), "--execute-embedding"],
            capture_output=True, text=True, timeout=150)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Native packed embedding validation did not pass", result.stderr)
        self.assertIn("\nDT_EMBEDDING_EXECUTION=-40\n", result.stderr)

    def test_native_sdpa_gate_rejects_pre_fix_binary(self):
        root = Path(__file__).resolve().parents[1]
        candidate = root / "output/upgrades/gRPCServerCLI-0621d819-qwen35-cuda-before-empty-sdpa"
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("lldb") or not shutil.which("cc") or not candidate.is_file() or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("SDPA negative control requires the preserved pre-fix candidate and pinned source")
        result = subprocess.run(
            [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "native-backends",
             "--binary", str(candidate), "--ccv-source", str(source), "--execute-sdpa"],
            capture_output=True, text=True, timeout=150)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Native SDPA validation did not pass", result.stderr)
        self.assertIn("SIGFPE", result.stderr)
        self.assertIn("\nDT_SDPA_EXECUTION=-98\n", result.stderr)

    def test_native_backend_cli_rejects_pristine_qwen_cuda(self):
        root = Path(__file__).resolve().parents[1]
        candidate = root / "output/upgrades/community-0621d819/bazel-bin/Apps/gRPCServerCLI"
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("lldb") or not shutil.which("cc") or not candidate.is_file() or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("Pristine negative control requires LLDB, compiler, and pinned candidate/source")
        result = subprocess.run(
            [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "native-backends",
             "--binary", str(candidate), "--ccv-source", str(source), "--require-qwen-cuda"],
            capture_output=True, text=True, timeout=150)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["qwen_cuda_registered"])
        self.assertTrue(all(check["backend"] == "0x0" for check in report["checks"][:6]))
        self.assertTrue(all(check["backend"] == check["expected"] for check in report["checks"][6:]))
        refused = subprocess.run(
            [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "native-backends",
             "--binary", str(candidate), "--ccv-source", str(source), "--require-qwen-cuda", "--execute-qwen"],
            capture_output=True, text=True, timeout=150)
        self.assertEqual(refused.returncode, 1, refused.stdout + refused.stderr)
        self.assertIn("\nDT_QWEN_EXECUTION=-98\n", refused.stderr)
        self.assertIn("\nDT_QWEN_EXECUTION_ERROR=CUDA registrations or symbols missing\n", refused.stderr)

    def test_fixture_cli_records_native_validation(self):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        if not shutil.which("cc") or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("Native fixture generation requires pinned CCV sources")
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(root / "tools/dt_layer_parity_oracle.py"), "qwen35-linear-attn-fixture",
                 "--out-dir", temporary, "--native-ccv-source", str(source)],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(temporary) / "manifest.json").read_text())
            self.assertEqual(manifest["validation"], "native_cpu_verified")
            self.assertEqual(manifest["native_cpu"]["result"], "PASS")

    def test_python_split_store_is_readable_by_native_candidate(self):
        root = Path(__file__).resolve().parents[1]
        candidate = Path(os.environ.get("DT_GRPC_CANDIDATE", str(root / "output/upgrades/community-0621d819/bazel-bin/Apps/gRPCServerCLI")))
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        debugger, compiler = shutil.which("lldb"), shutil.which("cc")
        if not debugger or not compiler or not candidate.is_file() or not (source / "lib/nnc/ccv_nnc.h").is_file():
            self.skipTest("Native split-store probe requires LLDB, a C compiler, and the pinned candidate/source")
        with tempfile.TemporaryDirectory() as temporary:
            library_path, checkpoint = Path(temporary) / "probe.so", Path(temporary) / "split.ckpt"
            with DrawThingsCkptStore(checkpoint, writable=True, chunk_bytes=8) as store:
                store.write_tensor("split_matrix", np.arange(12, dtype=np.float32).reshape(3, 4), datatype=DT_FLOAT32)
            build = subprocess.run([compiler, "-shared", "-fPIC", "-g", "-O2", "-I", str(source / "lib"),
                                    str(root / "tools/dt_qwen35_cpu_probe.c"), "-lm", "-o", str(library_path)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            symbols = ["ccv_nnc_init", "sqlite3_open_v2", "ccv_nnc_tensor_read", "sqlite3_close", "ccv_nnc_tensor_free"]
            commands = ["settings set target.disable-aslr false", "breakpoint set --name main", "run",
                        f'expression -- ((void *(*)(const char *, int))dlopen)({json.dumps(str(library_path))}, 2)',
                        f'script addresses = [lldb.target.FindSymbols(name).GetContextAtIndex(0).GetSymbol().GetStartAddress().GetLoadAddress(lldb.target) for name in {symbols!r}]',
                        'script assert all(address != lldb.LLDB_INVALID_ADDRESS for address in addresses), "Native probe symbol missing"',
                        'script expression = "dt_probe_native_store(" + ",".join("(void*)%d" % address for address in addresses) + "," + ' + repr(json.dumps(str(checkpoint))) + ' + ")"',
                        'script result = lldb.frame.EvaluateExpression(expression); print("DT_NATIVE_STORE_RESULT=" + str(result.GetValueAsSigned(-99))); print(result.GetError())',
                        "process kill"]
            invocation = [debugger, "--batch"]
            for command in commands:
                invocation.extend(["-o", command])
            invocation.extend(["--", str(candidate), "--help"])
            result = subprocess.run(invocation, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DT_NATIVE_STORE_RESULT=0", result.stdout)

    def test_numpy_matches_pinned_native_cpu_operations(self):
        root = Path(__file__).resolve().parents[1]
        source = Path(os.environ.get("DT_CCV_SOURCE_DIR", str(root / ".cache/bazel-upgrade/de9c699fb740fbb90831ca87b34ee139/external/ccv")))
        compiler = shutil.which("cc")
        if not compiler or not (source / "lib/nnc/cmd/gated_delta/ccv_nnc_gated_delta_cpu_ref.c").exists():
            self.skipTest("Set DT_CCV_SOURCE_DIR to the pinned CCV checkout and install a C compiler")
        with tempfile.TemporaryDirectory() as temporary:
            library_path = Path(temporary) / "probe.so"
            build = subprocess.run(
                [compiler, "-shared", "-fPIC", "-O2", "-I", str(source / "lib"),
                 str(root / "tools/dt_qwen35_cpu_probe.c"), "-lm", "-o", str(library_path)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            library = ctypes.CDLL(str(library_path))
            pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
            library.dt_probe_gated_delta.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                *([pointer] * 8), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            library.dt_probe_gated_delta.restype = ctypes.c_int
            library.dt_probe_rmsnorm.argtypes = [ctypes.c_int, ctypes.c_int, *([pointer] * 4)]
            library.dt_probe_rmsnorm.restype = ctypes.c_int
            rng = np.random.default_rng(4242)
            for dimensions in ((1, 5, 2, 4, 3, 5), (2, 3, 1, 2, 4, 2), (1, 3, 16, 32, 128, 128)):
                batch, sequence, key_heads, value_heads, key_dim, value_dim = dimensions
                for channel_decay in (False, True):
                    queries = rng.normal(size=(batch, sequence, key_heads, key_dim)).astype(np.float32) * 0.025
                    keys = rng.normal(size=queries.shape).astype(np.float32) * 0.025
                    values = rng.normal(size=(batch, sequence, value_heads, value_dim)).astype(np.float32)
                    beta = np.full((batch, sequence, value_heads), 0.5, dtype=np.float32)
                    gate_shape = beta.shape + ((key_dim,) if channel_decay else ())
                    decay = -np.abs(rng.normal(size=gate_shape)).astype(np.float32) * 0.1
                    initial = rng.normal(size=(batch, value_heads, value_dim, key_dim)).astype(np.float32) * 0.01
                    expected, state = reference_gated_delta_rule(
                        queries, keys, values, beta, decay, initial, True, state_checkpoint_count=2)
                    output, final = np.empty_like(expected), np.empty_like(state)
                    result = library.dt_probe_gated_delta(np.array(dimensions, dtype=np.int32), queries, keys, values,
                                                         decay, beta, initial, output, final, 1, 2, int(channel_decay))
                    self.assertEqual(result, 0)
                    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-5)
                    np.testing.assert_allclose(final, state, atol=2e-6, rtol=2e-5)
            values = rng.normal(size=(8, 128)).astype(np.float32)
            gate = rng.normal(size=values.shape).astype(np.float32)
            weight = rng.normal(size=(128,)).astype(np.float32)
            actual = np.empty_like(values)
            self.assertEqual(library.dt_probe_rmsnorm(8, 128, values, gate, weight, actual), 0)
            np.testing.assert_allclose(actual, reference_rmsnorm_gated(values, gate, weight), atol=2e-6, rtol=2e-5)


class WeeLLMTransformsTests(unittest.TestCase):
    def test_rotary_deinterleave_matches_explicit_loop_and_inverts(self):
        # Verify exact match with TensorDescriptor.swift L95-106 formula:
        # out[h*hd + d*2 + p] = in[h*hd + p*(hd/2) + d]
        num_heads, head_dim, cols = 32, 128, 16
        rng = np.random.default_rng(42)
        src = rng.standard_normal((num_heads * head_dim, cols)).astype(np.float32)

        expected = np.empty_like(src)
        half = head_dim // 2
        for h in range(num_heads):
            base = h * head_dim
            for d in range(half):
                for p in range(2):
                    expected[base + d * 2 + p] = src[base + p * half + d]

        got = rotary_deinterleave(src, num_heads=num_heads, head_dim=head_dim)
        np.testing.assert_array_equal(got, expected)

        # Verify 1-D QK-norm vector [4096]
        norm_1d = src[:, 0]
        got_1d = rotary_deinterleave(norm_1d, num_heads=num_heads, head_dim=head_dim)
        np.testing.assert_array_equal(got_1d, expected[:, 0])

        # Verify exact round-trip with rotary_interleave
        recovered = rotary_interleave(got, num_heads=num_heads, head_dim=head_dim)
        np.testing.assert_array_equal(recovered, src)

    def test_dim_blob_norm_rank3_encoding(self):
        # §22 rule: QK-norms and ada_ln modulations must be stored [1, 1, N] in 48-byte dim blob
        blob = encode_dim_blob([4096], ndim_slots=12, rank_policy="norm_1_1_n")
        self.assertEqual(len(blob), 48)
        unpacked = struct.unpack("<12i", blob)
        self.assertEqual(unpacked[:4], (1, 1, 4096, 0))
        dims, slots = decode_dim_blob(blob)
        self.assertEqual(dims, [1, 1, 4096])
        self.assertEqual(slots, 12)

    def test_bf16_roundtrip(self):
        vals = np.array([[1.0, -2.5, 0.125, 65024.0]], dtype=np.float32)
        raw = f32_to_bf16_bytes(vals)
        self.assertEqual(len(raw), 8)
        recovered = bf16_bytes_to_f32(raw, vals.shape)
        np.testing.assert_allclose(recovered, vals, rtol=1e-2)

    def test_split_tensor_chunks_squeezes_unit_dims(self):
        # Matches scale_shift_table stored [1, 9, 4096] -> squeezed [9, 4096] -> 9 chunks of [4096]
        arr = np.arange(9 * 16, dtype=np.float32).reshape(1, 9, 16)
        chunks = split_tensor_chunks(arr, count=9, axis=0, squeeze_unit_dims=True)
        self.assertEqual(len(chunks), 9)
        for i, c in enumerate(chunks):
            self.assertEqual(c.shape, (1, 16))
            np.testing.assert_array_equal(c[0], arr[0, i])


class SeekerAndDequantTests(unittest.TestCase):
    def test_all_ggml_decoders_match_frozen_gguf_013_vectors(self):
        expected = {
            2: "ec9cb8593ecd8177026e031e548fd775c2ad0aa7b1e7f92119fcb8c885124d93",
            3: "bb609678e017f7e0be0a87e36e9a16490e73256c745327009d54b47167ad516f",
            6: "5b177880341f486159d48c556c40f14289b8fa09a54160e05d5b7520c9ae5e61",
            7: "3f2612f7d7ce89b07a94c522dfcf6dce551cae98e08e1a6e6e07ed80fedeb8a6",
            8: "c833ffe5f5760892ae5fc6adcc87513d73d62927fc072a62a4cfc2cfd0266ac5",
            10: "ec570ef297c609e4d874a57cbb9bdee95f91e8f950df47b318699d877150d8f1",
            11: "d05cb4cd7873aef0ea9e38469dde407e35a1877940c60589751a419786ba8530",
            12: "dff728086f02a8636b1acf515707e5fce74a4e8207d4e537c753a080bac575a6",
            13: "6160baeaa7490d409cf08af6a163a63f84406fdbb969d472cc614b859968e9cc",
            14: "20d0d5ae2cd8706e06a50f239bfd1c13df3d854e5aea9d26b831f50f75cb4f5b",
        }
        rng = np.random.default_rng(4242)
        for code, digest in expected.items():
            with self.subTest(ggml_type=code):
                elements, size, _ = GGML_BLOCK_INFO[code]
                raw = rng.integers(0, 256, size=(3, size), dtype=np.uint8)
                offset = {10: 80, 11: 108, 14: 208}.get(code, 0)
                raw[:, offset:offset + 2] = np.frombuffer(np.float16(0.25).tobytes(), dtype=np.uint8)
                if code in (3, 7, 12, 13):
                    raw[:, 2:4] = np.frombuffer(np.float16(0.125).tobytes(), dtype=np.uint8)
                if code == 10:
                    raw[:, 82:84] = np.frombuffer(np.float16(0.125).tobytes(), dtype=np.uint8)
                actual = dequantize_gguf_tensor(raw.tobytes(), code, (3, elements))
                self.assertEqual(hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest(), digest)

    def test_fp8_row_scales_and_duplicate_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "a.safetensors", root / "b.safetensors"
            write_mock_safetensors(first, {
                "linear.weight": ("F8_E4M3", (2, 2), bytes([0x38] * 4)),
                "linear.weight_scale": ("F32", (2,), np.array([2, 3], dtype="<f4").tobytes())})
            with SafetensorsSeeker(first) as seeker:
                np.testing.assert_array_equal(seeker.read_f32("linear.weight"), [[2, 2], [3, 3]])
            write_mock_safetensors(second, {"linear.weight": ("F32", (1,), np.ones(1, dtype="<f4").tobytes())})
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"linear.weight": first.name, "linear.weight_scale": second.name}}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                SafetensorsSeeker(index)

    def test_single_and_sharded_safetensors_seeker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a = np.array([1.5, -2.0, 3.25, 4.0], dtype="<f2")
            b = np.array([10.0, 20.0], dtype="<f4")

            shard1 = tmp_dir / "model-00001-of-00002.safetensors"
            shard2 = tmp_dir / "model-00002-of-00002.safetensors"
            write_mock_safetensors(shard1, {"w1": ("F16", (2, 2), a.tobytes())})
            write_mock_safetensors(shard2, {"w2": ("F32", (2,), b.tobytes())})

            index_path = tmp_dir / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "w1": shard1.name,
                            "w2": shard2.name,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with SafetensorsSeeker(tmp_dir) as seeker:
                self.assertEqual(sorted(seeker.keys()), ["w1", "w2"])
                np.testing.assert_allclose(seeker.read_f32("w1"), a.reshape(2, 2), atol=1e-3)
                np.testing.assert_allclose(seeker.read_f32("w2"), b, atol=1e-6)

    def test_gguf_seeker_and_q8_0_dequant(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf_path = Path(tmp) / "test.gguf"
            # Build a minimal valid GGUF v3 file with 1 Q8_0 tensor of shape (32,)
            scale = np.float16(0.5).tobytes()
            qs = np.arange(-16, 16, dtype=np.int8).tobytes()
            q8_block = scale + qs  # 34 bytes

            name_bytes = b"blk.0.attn_q.weight"
            header = bytearray()
            header += b"GGUF"
            header += struct.pack("<IQQ", 3, 1, 0)  # v3, 1 tensor, 0 kv
            header += struct.pack("<Q", len(name_bytes)) + name_bytes
            header += struct.pack("<I", 1)  # 1D
            header += struct.pack("<Q", 32)  # len 32
            header += struct.pack("<I", 8)  # GGML_TYPE_Q8_0 = 8
            header += struct.pack("<Q", 0)  # rel offset 0
            while len(header) % 32 != 0:
                header += b"\x00"

            gguf_path.write_bytes(bytes(header) + q8_block)

            with GGUFSeeker(gguf_path) as seeker:
                self.assertEqual(seeker.keys(), ["blk.0.attn_q.weight"])
                arr = seeker.read_f32("blk.0.attn_q.weight")
                expected = np.arange(-16, 16, dtype=np.float32) * 0.5
                np.testing.assert_allclose(arr, expected, atol=1e-5)

    def test_fp8_dequantizers(self):
        # 0x38 in E4M3 (sign=0, exp=7, mant=0) -> 1.0 * 2^(7-7) = 1.0
        arr_e4m3 = dequantize_fp8_e4m3(bytes([0x38, 0xB8]), (2,))
        np.testing.assert_allclose(arr_e4m3, np.array([1.0, -1.0], dtype=np.float32))
        # 0x3C in E5M2 (0x3C00 in FP16) -> 1.0
        arr_e5m2 = dequantize_fp8_e5m2(bytes([0x3C, 0xBC]), (2,))
        np.testing.assert_allclose(arr_e5m2, np.array([1.0, -1.0], dtype=np.float32))


class ContractAndAuditorTests(unittest.TestCase):
    def test_auditor_checks_generic_head_parameters_and_importer_support(self):
        text = (
            "func Block<T: Numeric>() {\n"
            'mapping["proj.q.weight"] = ModelWeightElement([q], interleavedIndices: [0], numberOfHeads: 24, headDimension: 128)\n'
            'mapping["q_norm.weight"] = ModelWeightElement([n], interleavedIndices: [0], numberOfHeads: 999, headDimension: 2)\n'
            "}\n"
        )
        self.assertTrue(audit_swift_mapper_source(text))
        self.assertTrue(audit_swift_mapper_source(
            "case .mossImage: fatalError()", "ModelImporter.swift", profile="moss_image"))

    def test_moss_contract_uses_native_layout(self):
        contract = build_moss_image_contract()
        rules = contract["tensors"]
        names = [name for rule in rules.values() for name in rule["dt_names"]]
        self.assertEqual(len(names), 524)
        self.assertEqual(len(set(names)), 524)
        self.assertEqual(rules["layers.0.q_norm.weight"]["source_shape"], [128])
        self.assertEqual(rules["layers.0.q_norm.weight"]["target_shapes"], [[1, 1, 1, 128]])
        self.assertEqual(rules["time_embedding.2.weight"]["f16_store_datatype"], "F32")
        self.assertNotIn("layers.0.proj.q.bias", rules)
        metadata = {name: (2, {"F16": DT_FLOAT16, "F32": DT_FLOAT32}[rule["f16_store_datatype"]], rule["target_shapes"][index])
                for rule in rules.values() for index, name in enumerate(rule["dt_names"])}
        digest = hashlib.sha256(json.dumps(sorted(metadata.items()), separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "15df82ca64fb5a9469be36868e502f00fd0b0957aad242bb7af43aec9085931d")

    def test_ltx23_contract_matches_exact_verified_counts(self):
        contract = build_ltx23_contract()
        self.assertEqual(contract["tensors"]["transformer_blocks.0.audio_to_video_attn.to_q.weight"]["source_shape"], [2048, 4096])
        self.assertEqual(contract["tensors"]["transformer_blocks.0.video_to_audio_attn.to_k.weight"]["source_shape"], [2048, 4096])
        summary = contract["expected_counts_summary"]
        self.assertEqual(summary["total_ckpt_tensors"], 5746)
        self.assertEqual(summary["f32_ada_ln_tensors"], 1540)
        self.assertEqual(summary["qk_norm_tensors"], 608)
        self.assertEqual(summary["bf16_aggregate_embed_tensors"], 4)

    def test_swift_auditor_passes_patched_ltx2_and_catches_unpatched_qknorm(self):
        repo_root = Path(__file__).resolve().parents[1]
        patched_ltx2 = (
            repo_root / "DRAW_THINGS_PATCH/Libraries/SwiftDiffusion/Sources/Models/LTX2.swift"
        )
        patched_importer = (
            repo_root / "DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift"
        )
        self.assertEqual(
            audit_swift_mapper_source(patched_ltx2.read_text(encoding="utf-8"), str(patched_ltx2)),
            [],
        )
        self.assertEqual(
            audit_swift_mapper_source(
                patched_importer.read_text(encoding="utf-8"), str(patched_importer)
            ),
            [],
        )

        # Simulate the pre-§18 bug where q_norm / k_norm lacked `interleaved: true`
        broken_swift = """
        private func BrokenSelfAttention(prefix: String, k: Int, h: Int) {
          let mapper: ModelWeightMapper = { _ in
            var mapping = ModelWeightMapping()
            mapping["\\(prefix).to_q.weight"] = ModelWeightElement(
              [toQueries.weight.name], interleaved: true, numberOfHeads: h, headDimension: k)
            mapping["\\(prefix).q_norm.weight"] = [normQ.weight.name]
            return mapping
          }
        }
        """
        findings = audit_swift_mapper_source(broken_swift, "Broken.swift")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].rule_id, "DT-CONV-018")


class DirectBuilderAndLoRABridgeTests(unittest.TestCase):
    def test_grouped_delta_native_layout_and_chunked_initial_state(self):
        rng = np.random.default_rng(42)
        queries = rng.normal(size=(1, 4, 2, 3)).astype(np.float32) * 0.1
        keys = rng.normal(size=queries.shape).astype(np.float32) * 0.1
        values = rng.normal(size=(1, 4, 4, 5)).astype(np.float32)
        beta = np.full((1, 4, 4), 0.5, dtype=np.float32)
        decay = np.full(beta.shape, -0.2, dtype=np.float32)
        initial = rng.normal(size=(1, 4, 5, 3)).astype(np.float32)
        output, state = reference_gated_delta_rule(queries, keys, values, beta, decay, initial, True)
        first, intermediate = reference_gated_delta_rule(
            queries[:, :2], keys[:, :2], values[:, :2], beta[:, :2], decay[:, :2], initial, True)
        second, final = reference_gated_delta_rule(
            queries[:, 2:], keys[:, 2:], values[:, 2:], beta[:, 2:], decay[:, 2:], intermediate, True)
        self.assertEqual(state.shape, (1, 4, 5, 3))
        np.testing.assert_allclose(output, np.concatenate([first, second], axis=1), atol=1e-6)
        np.testing.assert_allclose(state, final, atol=1e-6)

    def test_lora_fused_scaled_coverage_and_empty_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, contract, output = root / "l.safetensors", root / "c.json", root / "o.ckpt"
            contract.write_text(json.dumps({"schema_version": 1, "profile": "test", "tensors": {
                "fused.weight": {"dt_names": ["q", "k", "v"]}}}))
            up = np.arange(12, dtype="<f4").reshape(6, 2)
            down = np.ones((2, 3), dtype="<f4")
            write_mock_safetensors(source, {
                "fused.lora_A.weight": ("F32", down.shape, down.tobytes()),
                "fused.lora_B.weight": ("F32", up.shape, up.tobytes()),
                "fused.alpha": ("F32", (1,), np.array([4.0], dtype="<f4").tobytes())})
            with DrawThingsCkptStore(output, writable=True):
                pass
            self.assertFalse(verify_converted_lora(source, output, contract)[0])
            with DrawThingsCkptStore(output, writable=True) as store:
                for name, chunk in zip(("q", "k", "v"), np.split(up * 2, 3)):
                    store.write_tensor(name + "__up__", chunk)
                    store.write_tensor(name + "__down__", down)
            passed, report = verify_converted_lora(source, output, contract)
            self.assertTrue(passed, report)

    def test_end_to_end_convert_restore_and_lora_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            contract_path = tmp_dir / "mini_contract.json"
            contract = {
                "schema_version": 1,
                "profile": "mini_test",
                "restore_patterns": {
                    "f32_ada_ln_regex": r"ada_ln",
                    "f32_qk_norm_regex": r"(norm_q|norm_k)-\d+-0\]$",
                },
                "tensors": {
                    "blocks.0.attn.to_q.weight": {
                        "dt_names": ["__dit__[t-x_q-0-0]"],
                        "rotary_interleaved": True,
                        "num_heads": 2,
                        "head_dim": 4,
                        "f16_store_datatype": "F16",
                        "quant_restore_policy": "none",
                        "dim_rank_policy": "preserve",
                    },
                    "blocks.0.attn.to_v.weight": {
                        "dt_names": ["__dit__[t-x_v-0-0]"],
                        "rotary_interleaved": False,
                        "num_heads": 0,
                        "head_dim": 0,
                        "f16_store_datatype": "F16",
                        "quant_restore_policy": "none",
                        "dim_rank_policy": "preserve",
                    },
                    "blocks.0.attn.q_norm.weight": {
                        "dt_names": ["__dit__[t-x_norm_q-0-0]"],
                        "rotary_interleaved": True,
                        "num_heads": 2,
                        "head_dim": 4,
                        "f16_store_datatype": "F16",
                        "quant_restore_policy": "f32_from_source",
                        "dim_rank_policy": "norm_1_1_n",
                    },
                    "blocks.0.scale_shift_table": {
                        "dt_names": [
                            "__dit__[t-attn1_ada_ln_0-0-0]",
                            "__dit__[t-attn1_ada_ln_1-0-0]",
                        ],
                        "rotary_interleaved": False,
                        "num_heads": 0,
                        "head_dim": 0,
                        "f16_store_datatype": "F32",
                        "quant_restore_policy": "f32_from_source",
                        "dim_rank_policy": "norm_1_1_n",
                    },
                },
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            q_w = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
            v_w = np.ones((8, 4), dtype=np.float32) * 0.5
            q_norm = np.arange(8, dtype=np.float32)
            scale_shift = np.arange(16, dtype=np.float32).reshape(1, 2, 8)

            sf_path = tmp_dir / "source.safetensors"
            write_mock_safetensors(
                sf_path,
                {
                    "blocks.0.attn.to_q.weight": ("F32", q_w.shape, q_w.tobytes()),
                    "blocks.0.attn.to_v.weight": ("F32", v_w.shape, v_w.tobytes()),
                    "blocks.0.attn.q_norm.weight": ("F32", q_norm.shape, q_norm.tobytes()),
                    "blocks.0.scale_shift_table": ("F32", scale_shift.shape, scale_shift.tobytes()),
                },
            )

            f16_ckpt = tmp_dir / "model_f16.ckpt"
            stats = convert_source_to_ckpt(sf_path, contract_path, f16_ckpt)
            self.assertEqual(stats["ckpt_tensors_written"], 5)

            # Simulate quantizer palettizing the norms in a mock q6p ckpt, then restore them
            q6p_ckpt = tmp_dir / "model_q6p.ckpt"
            q6p_ckpt.write_bytes(f16_ckpt.read_bytes())
            with DrawThingsCkptStore(q6p_ckpt, writable=True) as store:
                # Corrupt datatype and rank to simulate raw quantizer output
                store.write_tensor(
                    "__dit__[t-x_norm_q-0-0]",
                    np.zeros((8,), dtype=np.float32),
                    datatype=DT_FLOAT16,
                    rank_policy="preserve",
                )

            r_stats = restore_norms_from_contract(q6p_ckpt, f16_ckpt, contract_path)
            self.assertGreaterEqual(r_stats["restored_f32"], 1)

            with DrawThingsCkptStore(q6p_ckpt, writable=False) as store:
                meta, arr = store.read_f32("__dit__[t-x_norm_q-0-0]")
                self.assertEqual(meta.datatype, DT_FLOAT32)
                self.assertEqual(meta.shape, [1, 1, 8])
                np.testing.assert_allclose(
                    arr.reshape(8), rotary_deinterleave(q_norm, 2, 4), atol=1e-3
                )

            # Test Universal LoRA Bridge static verification (§24)
            lora_sf = tmp_dir / "lora.safetensors"
            lora_ckpt = tmp_dir / "lora_f16.ckpt"
            q_up = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
            q_down = np.ones((2, 4), dtype=np.float32) * 0.25
            v_up = np.ones((8, 2), dtype=np.float32) * 0.75
            v_down = np.ones((2, 4), dtype=np.float32) * 0.125

            write_mock_safetensors(
                lora_sf,
                {
                    "diffusion_model.blocks.0.attn.to_q.lora_B.weight": (
                        "F32",
                        q_up.shape,
                        q_up.tobytes(),
                    ),
                    "diffusion_model.blocks.0.attn.to_q.lora_A.weight": (
                        "F32",
                        q_down.shape,
                        q_down.tobytes(),
                    ),
                    "diffusion_model.blocks.0.attn.to_v.lora_B.weight": (
                        "F32",
                        v_up.shape,
                        v_up.tobytes(),
                    ),
                    "diffusion_model.blocks.0.attn.to_v.lora_A.weight": (
                        "F32",
                        v_down.shape,
                        v_down.tobytes(),
                    ),
                },
            )
            with DrawThingsCkptStore(lora_ckpt, writable=True) as store:
                store.write_tensor(
                    "__dit__[t-x_q-0-0]__up__",
                    rotary_deinterleave(q_up, 2, 4),
                    datatype=DT_FLOAT16,
                )
                store.write_tensor("__dit__[t-x_q-0-0]__down__", q_down, datatype=DT_FLOAT16)
                store.write_tensor("__dit__[t-x_v-0-0]__up__", v_up, datatype=DT_FLOAT16)
                store.write_tensor("__dit__[t-x_v-0-0]__down__", v_down, datatype=DT_FLOAT16)

            ok, report = verify_converted_lora(lora_sf, lora_ckpt, contract_path)
            self.assertTrue(ok, msg=str(report))
            self.assertEqual(report["rotary_permuted_up_verified"], 1)
            self.assertEqual(report["identity_up_verified"], 1)
            self.assertEqual(report["identity_down_verified"], 2)

            # Test single-block weight & transform parity check (`dt_layer_parity_oracle.py`)
            blk_ok, blk_report = verify_block_parity(
                sf_path, f16_ckpt, contract_path, block_prefix="blocks.0."
            )
            self.assertTrue(blk_ok, msg=str(blk_report))
            self.assertEqual(blk_report["tensors_checked"], 5)

    def test_qwen35_gated_delta_and_rmsnorm_gated_reference(self):
        rng = np.random.default_rng(123)
        q = rng.standard_normal((1, 4, 2, 8)).astype(np.float32) * 0.1
        k = rng.standard_normal((1, 4, 2, 8)).astype(np.float32) * 0.1
        v = rng.standard_normal((1, 4, 2, 8)).astype(np.float32) * 0.1
        beta = np.full((1, 4, 2), 0.5, dtype=np.float32)
        decay = np.full((1, 4, 2), 0.9, dtype=np.float32)

        out, state = reference_gated_delta_rule(q, k, v, beta, decay)
        self.assertEqual(out.shape, (1, 4, 2, 8))
        self.assertEqual(state.shape, (1, 2, 8, 8))
        self.assertTrue(np.all(np.isfinite(out)))

        x = rng.standard_normal((1, 4, 16)).astype(np.float32)
        gate = rng.standard_normal((1, 4, 16)).astype(np.float32)
        w = np.ones((16,), dtype=np.float32)
        rg_out = reference_rmsnorm_gated(x, gate, w)
        self.assertEqual(rg_out.shape, (1, 4, 16))
        self.assertTrue(np.all(np.isfinite(rg_out)))

    def test_diff_ckpt_stores_and_inspect_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            c1_path = tmp_dir / "candidate.ckpt"
            c2_path = tmp_dir / "reference.ckpt"

            with DrawThingsCkptStore(c1_path, writable=True) as s1:
                s1.write_tensor("t1", np.array([1.0, 2.0], dtype=np.float32), datatype=DT_FLOAT32)
                s1.write_tensor(
                    "__dit__[t-x_norm_q-0-0]",
                    np.array([0.5, 0.5], dtype=np.float32),
                    datatype=DT_FLOAT32,
                    rank_policy="norm_1_1_n",
                )

            with DrawThingsCkptStore(c2_path, writable=True) as s2:
                s2.write_tensor("t1", np.array([1.0, 2.0], dtype=np.float32), datatype=DT_FLOAT32)
                s2.write_tensor(
                    "__dit__[t-x_norm_q-0-0]",
                    np.array([0.5, 0.5], dtype=np.float32),
                    datatype=DT_FLOAT32,
                    rank_policy="norm_1_1_n",
                )

            diff = diff_ckpt_stores(c1_path, c2_path)
            self.assertTrue(diff["is_equivalent"])
            self.assertEqual(diff["shared_tensor_count"], 2)

            health_ok, report = inspect_ckpt_health(c1_path)
            self.assertTrue(health_ok, msg=str(report))
            self.assertEqual(report["total_tensors"], 2)
            self.assertEqual(report["f32_tensors"], 2)

    def test_lora_normalize_unmangles_kohya_keys_and_hydrates_contracts(self):
        repo_root = Path(__file__).resolve().parents[1]
        ltx_contract = load_contract(repo_root / "contracts/ltx2_3.json")
        self.assertEqual(len(ltx_contract["tensors"]), 4448)
        moss_contract = load_contract(repo_root / "contracts/moss_image.json")
        self.assertGreater(len(moss_contract["tensors"]), 400)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_kohya = tmp_dir / "kohya_lora.safetensors"
            w_down = np.ones((4, 8), dtype=np.float32)
            w_up = np.ones((8, 4), dtype=np.float32)
            write_mock_safetensors(
                raw_kohya,
                {
                    "lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight": (
                        "F32",
                        w_down.shape,
                        w_down.tobytes(),
                    ),
                    "lora_unet_transformer_blocks_0_attn1_to_q.lora_up.weight": (
                        "F32",
                        w_up.shape,
                        w_up.tobytes(),
                    ),
                },
            )

            norm_out = tmp_dir / "normalized_lora.safetensors"
            stats = normalize_lora_safetensors(raw_kohya, norm_out, profile="ltx2_3")
            self.assertEqual(stats["normalized_down_tensors"], 1)
            self.assertEqual(stats["normalized_up_tensors"], 1)

            with SafetensorsSeeker(norm_out) as s:
                self.assertIn("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", s.keys())
                self.assertIn("diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight", s.keys())


if __name__ == "__main__":
    unittest.main()
