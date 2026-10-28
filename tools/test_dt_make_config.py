import json
import os
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import dt_tensor_to_playable as decoder
from tools.dt_make_config import _resolve_sampler


class SamplerResolutionTests(unittest.TestCase):
    def test_numeric_values_remain_unchanged(self):
        for value in (0, 17, 19):
            with self.subTest(value=value):
                self.assertEqual(_resolve_sampler(value), value)
                self.assertEqual(_resolve_sampler(str(value)), value)

    def test_names_use_selected_generated_enum(self):
        module = SimpleNamespace(SamplerType=SimpleNamespace(EulerATrailing=73))
        with patch.dict(sys.modules, {"SamplerType": module}):
            self.assertEqual(_resolve_sampler("EulerATrailing"), 73)
            with self.assertRaisesRegex(ValueError, "Unknown SamplerType"):
                _resolve_sampler("unknown")


class ImageCanarySequenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.script = self.tools / "run_q6p_canary_once.sh"
        shutil.copyfile(Path(__file__).with_name(self.script.name), self.script)
        shutil.copyfile(Path(__file__).with_name("dt_tensor_to_playable.py"),
                self.tools / "dt_tensor_to_playable.py")
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)
        self.models = self.root / "models"
        self.models.mkdir()
        for name in ("first.ckpt", "second.ckpt"):
            (self.models / name).write_bytes(b"fixture")
        (self.root / "escape.ckpt").write_bytes(b"fixture")
        self.proto = self.root / "imageService.proto"
        self.proto.write_text("")
        self.bindings = self.root / "bindings"
        self.bindings.mkdir()
        (self.tools / "dt_make_config.py").write_text(
            "import argparse\nfrom pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--out', required=True)\n"
            "parser.add_argument('--model', required=True)\n"
            "arguments, remaining = parser.parse_known_args()\n"
            "Path(arguments.out).write_bytes(arguments.model.encode())\n"
        )
        self.server = self.root / "fixture-server"
        self.server.write_text(f"#!{sys.executable}\nimport signal\nsignal.pause()\n")
        self.server.chmod(0o755)
        self.client = self.tools / "dt_api_client.py"
        self.client.write_text(
            "import sys\n"
            "if 'echo' not in sys.argv:\n"
            "    print('response #1')\n"
            "    print('generation stream finished')\n"
            "    print('images written: 0')\n"
            "    print('audio written: 0')\n"
        )

    def run_canary(self, *, models=("second.ckpt",), image_mode=True, dry_run=True, extra=()):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            "bash", str(self.script), "--model", "first.ckpt",
            "--models-dir", str(self.models), "--grpc-bin", str(self.server),
            "--proto", str(self.proto), "--bindings-dir", str(self.bindings),
            "--cache-dir", str(self.root / "cache"), "--tag", "sequence-test",
            "--host", f"127.0.0.1:{port}",
        ]
        if image_mode:
            command.append("--image-mode")
        if dry_run:
            command.append("--dry-run")
        for model in models:
            command.extend(("--next-model", model))
        return subprocess.run(command + list(extra), cwd=self.root, capture_output=True,
                              text=True, timeout=20)

    def test_sequence_dry_run_prepares_every_config(self):
        result = self.run_canary(models=("second.ckpt", "first.ckpt"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        work = self.root / "output/q6p_canary_sequence-test"
        self.assertEqual((work / "config.bin").read_bytes(), b"first.ckpt")
        self.assertEqual((work / "request_001/config.bin").read_bytes(), b"second.ckpt")
        self.assertEqual((work / "request_002/config.bin").read_bytes(), b"first.ckpt")
        self.assertEqual(result.stdout.count("server_command="), 1)
        self.assertIn("RESULT=DRY_RUN", result.stdout)
        self.assertNotIn("server_pid=", result.stdout)

    def test_sequence_rejects_missing_or_escaping_model_before_output(self):
        for model in ("missing.ckpt", "../escape.ckpt", ""):
            with self.subTest(model=model):
                result = self.run_canary(models=(model,))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sequence model file missing or invalid", result.stderr)
                self.assertFalse((self.root / "output").exists())

    def test_sequence_requires_strict_image_mode(self):
        for image_mode, extra in ((False, ()), (True, ("--soft-fail",)),
                                  (True, ("--allow-missing-model",))):
            with self.subTest(image_mode=image_mode, extra=extra):
                result = self.run_canary(image_mode=image_mode, extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--next-model requires --image-mode", result.stderr)
                self.assertFalse((self.root / "output").exists())

    def test_single_request_dry_run_remains_available(self):
        result = self.run_canary(models=(), image_mode=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        work = self.root / "output/q6p_canary_sequence-test"
        self.assertEqual((work / "config.bin").read_bytes(), b"first.ckpt")
        self.assertFalse((work / "request_001").exists())

    def test_successful_images_share_one_owned_process_and_validate_each_request(self):
        fixture = self.root / "fixture_image.bin"
        header = struct.pack("<17I", decoder.IDENTIFIER_RAW, 0,
                             decoder.CCV_TENSOR_FORMAT_NHWC, decoder.CCV_16F, 0,
                             1, 256, 256, 3, *([0] * 8))
        fixture.write_bytes(header + bytes(256 * 256 * 3 * 2))
        self.client.write_text(
            "import shutil, sys\nfrom pathlib import Path\n"
            "if 'echo' not in sys.argv:\n"
            "    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
            f"    shutil.copyfile({str(fixture)!r}, output / 'image_fixture.bin')\n"
            "    print('response #1')\n"
            "    print('generation stream finished')\n"
            "    print('images written: 1')\n"
            "    print('audio written: 0')\n"
        )
        result = self.run_canary(models=("second.ckpt", "first.ckpt"), dry_run=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        work = self.root / "output/q6p_canary_sequence-test"
        report = json.loads((work / "sequence_acceptance.json").read_text())
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["scope"], "finite_image_stream_same_server_only")
        self.assertFalse(report["repeatability_checked"])
        self.assertEqual([request["model"] for request in report["requests"]],
                         ["first.ckpt", "second.ckpt", "first.ckpt"])
        self.assertEqual({request["server_pid"] for request in report["requests"]},
                         {report["server_pid"]})
        for directory in (work, work / "request_001", work / "request_002"):
            self.assertTrue((directory / "image.png").is_file())
        with self.assertRaises(ProcessLookupError):
            os.kill(report["server_pid"], 0)

    def test_failed_request_cleans_up_owned_server_and_has_no_sequence_pass(self):
        result = self.run_canary(dry_run=False)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT=FAIL no final generated output", result.stdout)
        self.assertNotIn("RESULT=PASS", result.stdout)
        process_id = int(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                              if line.startswith("server_pid=")))
        with self.assertRaises(ProcessLookupError):
            os.kill(process_id, 0)
        self.assertFalse((self.root / "output/q6p_canary_sequence-test/sequence_acceptance.json").exists())

    def test_timed_out_request_preserves_progress_and_cleans_up_owned_server(self):
        self.client.write_text(
            "import signal, sys\n"
            "if 'echo' not in sys.argv:\n"
            "    print('current signpost: fixture_progress')\n"
            "    signal.pause()\n"
        )
        result = self.run_canary(models=(), dry_run=False, extra=("--timeout-sec", "1"))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT=FAIL canary timed out (1s)", result.stdout)
        work = self.root / "output/q6p_canary_sequence-test"
        self.assertIn("current signpost: fixture_progress", (work / "client.log").read_text())
        self.assertFalse((work / "image_acceptance.json").exists())
        process_id = int(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                              if line.startswith("server_pid=")))
        with self.assertRaises(ProcessLookupError):
            os.kill(process_id, 0)

    def test_native_model_preflight_checks_decoded_metadata_without_generation(self):
        self.client.write_text(
            "import base64, json, sys\n"
            "if 'echo' not in sys.argv: raise SystemExit('generation must not run')\n"
            "models = [{'file': 'first.ckpt', 'version': 'moss_image'}]\n"
            "print(json.dumps({'override': {'models': base64.b64encode(json.dumps(models).encode()).decode()}}))\n"
        )
        result = self.run_canary(models=(), dry_run=False,
                                 extra=("--preflight-only", "--require-model-version", "moss_image"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT=PREFLIGHT_PASS", result.stdout)
        work = self.root / "output/q6p_canary_sequence-test"
        self.assertEqual(json.loads((work / "model-selection.json").read_text())["result"], "PASS")
        self.assertFalse((work / "image_acceptance.json").exists())

    def test_native_model_preflight_rejects_missing_catalog_entry(self):
        self.client.write_text(
            "import base64, json\n"
            "print(json.dumps({'override': {'models': base64.b64encode(b'[]').decode()}}))\n"
        )
        result = self.run_canary(models=(), dry_run=False,
                                 extra=("--preflight-only", "--require-model-version", "moss_image"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not decode", result.stderr)
        self.assertNotIn("RESULT=PREFLIGHT_PASS", result.stdout)


class ImageHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tools = self.root / "tools"
        tools.mkdir()
        self.script = tools / "dt_image.sh"
        shutil.copyfile(Path(__file__).with_name(self.script.name), self.script)
        code = "import json, os, sys; print(json.dumps({'args': sys.argv[1:], 'library_path': os.environ.get('LD_LIBRARY_PATH', '')}))"
        (tools / "run_q6p_canary_once.sh").write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(code)} "$@"\n'
        )
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)
        (tools / "dt_moss_convert.py").write_text("print('my_moss_f16.ckpt')\n")

    def run_helper(self, *arguments):
        return subprocess.run(["bash", str(self.script), *arguments], cwd=self.root,
                              capture_output=True, text=True, timeout=10)

    def test_default_profile_uses_private_candidate_and_offload(self):
        result = self.run_helper("A blue vase")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.splitlines()[-1])
        arguments = report["args"]
        self.assertEqual(arguments[arguments.index("--model") + 1], "moss_image_sft_i4x.ckpt")
        self.assertEqual(arguments[arguments.index("--sampler") + 1], "DDIMTrailing")
        self.assertEqual(arguments[arguments.index("--steps") + 1], "50")
        self.assertEqual(arguments[arguments.index("--negative-prompt") + 1], "")
        self.assertIn("--server-cpu-offload", arguments)
        self.assertIn("--image-mode", arguments)
        self.assertRegex(arguments[arguments.index("--tag") + 1], r"^image_\d{8}_\d{6}_\d+$")
        self.assertEqual(arguments[arguments.index("--models-dir") + 1],
                         str(self.root / "output/upgrades/moss-image-stateful-20261027/models"))
        self.assertTrue(arguments[arguments.index("--grpc-bin") + 1].startswith(str(self.root / "output/upgrades")))
        self.assertIn("/usr/lib/wsl/lib", report["library_path"])

    def test_gpu_mode_and_models_forward_without_forced_offload(self):
        for model, filename in (("sft-i4x", "moss_image_sft_i4x.ckpt"),
                                ("sft-q8p", "moss_image_sft_q8p.ckpt"),
                                ("base-q8p", "moss_image_base_q8p.ckpt")):
            with self.subTest(model=model):
                result = self.run_helper("--mode", "gpu", "--model", model,
                                         "--size", "1024", "--seed", "123", "--metrics", "--dry-run", "A blue vase")
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = json.loads(result.stdout.splitlines()[-1])["args"]
                self.assertNotIn("--server-cpu-offload", arguments)
                self.assertEqual(arguments[arguments.index("--model") + 1], filename)
                self.assertEqual(arguments[arguments.index("--width") + 1], "1024")
                self.assertEqual(arguments[arguments.index("--height") + 1], "1024")
                self.assertIn("--dry-run", arguments)
                self.assertIn("--gpu-metrics", arguments)

    def test_prompt_metacharacters_and_negative_text_remain_literal(self):
        marker = self.root / "must-not-exist"
        prompt = f"A blue vase; $(touch {marker}), 'glass' and $HOME"
        result = self.run_helper("--negative-prompt", "blur; $HOME", "--width", "768", "--height", "512", prompt)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(result.stdout.splitlines()[-1])["args"]
        self.assertEqual(arguments[arguments.index("--prompt") + 1], prompt)
        self.assertEqual(arguments[arguments.index("--negative-prompt") + 1], "blur; $HOME")
        self.assertFalse(marker.exists())

    def test_invalid_options_and_dimensions_are_rejected(self):
        cases = (("",), ("  ",), ("", "second"), ("--mode", "invalid", "prompt"),
                 ("--model", "../model.ckpt", "prompt"), ("--size", "513", "prompt"),
                 ("--width", "0", "prompt"), ("--tag", "../escape", "prompt"),
                 ("--unknown", "prompt"), ("--mode",), ("first", "second"))
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_helper(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)

    def test_existing_output_tag_is_not_reused(self):
        directory = self.root / "output/q6p_canary_keep"
        directory.mkdir(parents=True)
        payload = directory / "image.png"
        payload.write_bytes(b"existing image")
        result = self.run_helper("--tag", "keep", "prompt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output tag already exists", result.stderr)
        self.assertEqual(payload.read_bytes(), b"existing image")

    def test_verified_bundle_selects_its_private_model_directory(self):
        bundle = self.root / "output/moss-models/custom"
        (bundle / "models").mkdir(parents=True)
        result = self.run_helper("--bundle", str(bundle), "--dry-run", "A blue vase")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(result.stdout.splitlines()[-1])["args"]
        self.assertEqual(arguments[arguments.index("--model") + 1], "my_moss_f16.ckpt")
        self.assertEqual(arguments[arguments.index("--models-dir") + 1], str(bundle / "models"))
        self.assertEqual(arguments[arguments.index("--require-model-version") + 1], "moss_image")
        result = self.run_helper("--bundle", str(bundle), "--model", "base-q8p", "A blue vase")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot be combined", result.stderr)
        result = self.run_helper("--bundle", str(bundle), "--mode", "gpu", "A blue vase")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require --mode offload", result.stderr)


if __name__ == "__main__":
    unittest.main()