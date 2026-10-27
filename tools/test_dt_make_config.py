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


if __name__ == "__main__":
    unittest.main()