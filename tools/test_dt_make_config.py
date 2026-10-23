import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()