import contextlib
import io
import struct
import sys
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import dt_tensor_to_playable as decoder


def tensor_blob(values, identifier=decoder.IDENTIFIER_RAW):
    values = np.asarray(values, dtype="<f2")
    header = struct.pack(
        "<17I", identifier, 0, decoder.CCV_TENSOR_FORMAT_NHWC,
        decoder.CCV_16F, 0, 1, 1, 1, len(values), *([0] * 8),
    )
    payload = values.tobytes()
    if identifier == decoder.IDENTIFIER_ZIP:
        payload = zlib.compress(payload)
    return header + payload


class TensorValidationTests(unittest.TestCase):
    def test_finite_raw_and_zipped_tensors(self):
        for identifier in (decoder.IDENTIFIER_RAW, decoder.IDENTIFIER_ZIP):
            with self.subTest(identifier=identifier):
                decoder.validate_finite_tensor(tensor_blob([0.0, 0.5, -1.0], identifier))

    def test_rejects_nonfinite_values_before_sanitization(self):
        for identifier in (decoder.IDENTIFIER_RAW, decoder.IDENTIFIER_ZIP):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(identifier=identifier, invalid=invalid):
                    blob = tensor_blob([0.0, invalid, 0.5], identifier)
                    with self.assertRaisesRegex(ValueError, "1 non-finite values"):
                        decoder.validate_finite_tensor(blob)
                    self.assertEqual(decoder.decode_image_tensor_to_uint8(blob).shape, (1, 1, 3))

    def test_rejects_truncated_payload(self):
        with self.assertRaisesRegex(ValueError, "payload size mismatch"):
            decoder.validate_finite_tensor(tensor_blob([0.0, 0.5, 1.0])[:-1])

    def test_rejects_frame_count_mismatch_before_reading(self):
        for expected in ("0", "9"):
            with self.subTest(expected=expected):
                arguments = [
                    "dt_tensor_to_playable.py", "--image-bin", "unused.bin",
                    "--out-dir", "/unused", "--expected-frames", expected,
                ]
                with patch.object(sys, "argv", arguments), patch.object(Path, "read_bytes") as read:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            decoder.main()
                    self.assertEqual(raised.exception.code, 2)
                    read.assert_not_called()

    def test_strict_image_failure_writes_no_outputs(self):
        arguments = [
            "dt_tensor_to_playable.py", "--image-bin", "bad.bin",
            "--out-dir", "/unused", "--strict",
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(Path, "read_bytes", return_value=tensor_blob([0.0, np.nan, 1.0])):
                with patch.object(Path, "mkdir") as mkdir:
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        decoder.main()
                    mkdir.assert_not_called()

    def test_strict_audio_failure_writes_no_outputs(self):
        arguments = [
            "dt_tensor_to_playable.py", "--image-bin", "good.bin",
            "--audio-bin", "bad_audio.bin", "--out-dir", "/unused", "--strict",
        ]
        payloads = [tensor_blob([0.0, 0.5, 1.0]), tensor_blob([0.0, np.inf, 1.0])]
        with patch.object(sys, "argv", arguments):
            with patch.object(Path, "read_bytes", side_effect=payloads):
                with patch.object(Path, "mkdir") as mkdir:
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        decoder.main()
                    mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()