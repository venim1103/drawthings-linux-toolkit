import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.dt_lora_merge_ckpt import LoraSpec, merge_loras_into_ckpt, parse_lora_spec
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
    DT_FORMAT_NHWC,
    DT_TYPE_CPU,
    encode_dim_blob,
    tensor_to_ckpt_bytes,
)


def _write_sidecar_ckpt(ckpt_path: Path, tensors: dict) -> None:
    """Write a base `.ckpt` with an external `-tensordata` sidecar like `ModelConverter` produces."""
    sidecar_path = Path(f"{ckpt_path}-tensordata")
    offset = 0
    with sidecar_path.open("wb") as sidecar, DrawThingsCkptStore(ckpt_path, writable=True) as store:
        for name, (arr, dt, rank_policy) in tensors.items():
            raw = tensor_to_ckpt_bytes(arr, dt)
            sidecar.write(raw)
            dim_blob = encode_dim_blob(arr.shape, ndim_slots=12, rank_policy=rank_policy)
            descriptor = struct.pack("<QQQ", 0, offset, len(raw))
            store.con.execute(
                "INSERT INTO tensors (name, type, format, datatype, dim, data) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    name,
                    DT_TYPE_CPU | (0x1000_0000 << 32),
                    DT_FORMAT_NHWC,
                    dt,
                    dim_blob,
                    descriptor,
                ),
            )
            offset += len(raw)


class LoraMergeCkptTests(unittest.TestCase):
    def test_parse_lora_spec(self):
        s1 = parse_lora_spec("dt-models/foo_lora_f16.ckpt")
        self.assertEqual(s1.weight, 1.0)
        self.assertTrue(str(s1.path).endswith("dt-models/foo_lora_f16.ckpt"))

        s2 = parse_lora_spec("dt-models/foo_lora_f16.ckpt:0.65")
        self.assertAlmostEqual(s2.weight, 0.65)

        s3 = parse_lora_spec("dt-models/foo_lora_f16.ckpt:-0.25")
        self.assertAlmostEqual(s3.weight, -0.25)

        with self.assertRaises(ValueError):
            parse_lora_spec("dt-models/foo_lora_f16.ckpt:nan")

    def test_merge_single_and_multi_lora_preserves_dtypes_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_ckpt = root / "base_f16.ckpt"
            lora1_ckpt = root / "lora1_f16.ckpt"
            lora2_ckpt = root / "lora2_f16.ckpt"
            out_ckpt = root / "merged_f16.ckpt"

            w_q = np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=np.float32)
            w_v = np.full((2, 3), 0.5, dtype=np.float32)
            ada_ln = np.array([1.25, -0.75, 2.5, 0.125], dtype=np.float32)
            agg_embed = np.array([[3.5, -1.25], [0.5, 4.0]], dtype=np.float32)

            _write_sidecar_ckpt(
                base_ckpt,
                {
                    "__dit__[t-x_q-0-0]": (w_q, DT_FLOAT16, "preserve"),
                    "__dit__[t-x_v-0-0]": (w_v, DT_FLOAT16, "preserve"),
                    "__dit__[t-ada_ln-0-0]": (ada_ln, DT_FLOAT32, "norm_1_1_n"),
                    "__text_feature_extractor__[t-video_aggregate_embed-0-0]": (
                        agg_embed,
                        DT_BFLOAT16,
                        "preserve",
                    ),
                },
            )

            u1 = np.array([[0.5, -0.25], [1.0, 0.5]], dtype=np.float32)
            d1 = np.array([[0.2, 0.4, -0.6], [-0.5, 0.25, 0.75]], dtype=np.float32)
            with DrawThingsCkptStore(lora1_ckpt, writable=True) as s1:
                s1.write_tensor("__dit__[t-x_q-0-0]__up__", u1, datatype=DT_FLOAT16)
                s1.write_tensor("__dit__[t-x_q-0-0]__down__", d1, datatype=DT_FLOAT16)

            u2 = np.array([[0.1], [-0.2]], dtype=np.float32)
            d2 = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
            with DrawThingsCkptStore(lora2_ckpt, writable=True) as s2:
                s2.write_tensor("__dit__[t-x_q-0-0]__up__", u2, datatype=DT_FLOAT16)
                s2.write_tensor("__dit__[t-x_q-0-0]__down__", d2, datatype=DT_FLOAT16)

            stats = merge_loras_into_ckpt(
                base_ckpt=base_ckpt,
                loras=[LoraSpec(lora1_ckpt, 0.5), LoraSpec(lora2_ckpt, 1.25)],
                output_ckpt=out_ckpt,
            )
            self.assertEqual(stats["base_tensors_total"], 4)
            self.assertEqual(stats["tensors_modified"], 1)
            self.assertEqual(stats["tensors_copied_verbatim"], 3)
            self.assertEqual(stats["pair_deltas_applied"], 2)
            self.assertFalse(Path(f"{out_ckpt}-tensordata").exists())

            expected_q = (w_q + 0.5 * (u1 @ d1) + 1.25 * (u2 @ d2)).astype(np.float16).astype(np.float32)

            with DrawThingsCkptStore(out_ckpt, writable=False) as merged:
                q_meta, q_val = merged.read_f32("__dit__[t-x_q-0-0]")
                self.assertEqual(q_meta.datatype, DT_FLOAT16)
                np.testing.assert_allclose(q_val, expected_q, rtol=0, atol=1e-3)

                v_meta, v_val = merged.read_f32("__dit__[t-x_v-0-0]")
                self.assertEqual(v_meta.datatype, DT_FLOAT16)
                np.testing.assert_array_equal(v_val, w_v)

                norm_meta, norm_val = merged.read_f32("__dit__[t-ada_ln-0-0]")
                self.assertEqual(norm_meta.datatype, DT_FLOAT32)
                self.assertEqual(norm_meta.shape, [1, 1, 4])
                np.testing.assert_array_equal(norm_val.reshape(-1), ada_ln)

                agg_meta, agg_val = merged.read_f32("__text_feature_extractor__[t-video_aggregate_embed-0-0]")
                self.assertEqual(agg_meta.datatype, DT_BFLOAT16)
                np.testing.assert_array_equal(agg_val, agg_embed)

    def test_merge_conv4d_and_direct_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_ckpt = root / "base.ckpt"
            lora_ckpt = root / "lora.ckpt"
            out_ckpt = root / "out.ckpt"

            conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
            bias_w = np.array([0.25, -0.5], dtype=np.float32)
            with DrawThingsCkptStore(base_ckpt, writable=True) as b:
                b.write_tensor("__dit__[t-conv-0-0]", conv_w, datatype=DT_FLOAT16)
                b.write_tensor("__dit__[t-conv-0-1]", bias_w, datatype=DT_FLOAT32)

            u_conv = np.full((2, 2, 1, 1), 0.5, dtype=np.float32)
            d_conv = np.full((2, 3, 3, 3), 0.1, dtype=np.float32)
            diff_bias = np.array([0.5, 0.25], dtype=np.float32)
            with DrawThingsCkptStore(lora_ckpt, writable=True) as l_store:
                l_store.write_tensor("__dit__[t-conv-0-0]__up__", u_conv, datatype=DT_FLOAT16)
                l_store.write_tensor("__dit__[t-conv-0-0]__down__", d_conv, datatype=DT_FLOAT16)
                l_store.write_tensor("__dit__[t-conv-0-1]__diff__", diff_bias, datatype=DT_FLOAT32)

            merge_loras_into_ckpt(base_ckpt, [LoraSpec(lora_ckpt, 2.0)], out_ckpt)
            with DrawThingsCkptStore(out_ckpt, writable=False) as merged:
                _, c_out = merged.read_f32("__dit__[t-conv-0-0]")
                # Each output element gets 2.0 * sum_{r=0..1}(0.5 * 0.1) = 0.2 -> 1.2
                np.testing.assert_allclose(c_out, np.full((2, 3, 3, 3), 1.2, dtype=np.float32), atol=1e-3)
                b_meta, b_out = merged.read_f32("__dit__[t-conv-0-1]")
                self.assertEqual(b_meta.datatype, DT_FLOAT32)
                np.testing.assert_allclose(b_out, np.array([1.25, 0.0], dtype=np.float32), atol=1e-6)

    def test_fail_closed_guards_and_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_ckpt = root / "base.ckpt"
            bad_lora = root / "bad_lora.ckpt"
            good_lora = root / "good_lora.ckpt"
            out_ckpt = root / "merged.ckpt"

            with DrawThingsCkptStore(base_ckpt, writable=True) as b:
                b.write_tensor("__dit__[t-x_q-0-0]", np.ones((4, 4), dtype=np.float32), datatype=DT_FLOAT16)

            # Incomplete LoRA pair (missing __down__)
            with DrawThingsCkptStore(bad_lora, writable=True) as l_bad:
                l_bad.write_tensor("__dit__[t-x_q-0-0]__up__", np.ones((4, 2), dtype=np.float32), datatype=DT_FLOAT16)

            with self.assertRaisesRegex(ValueError, "Incomplete LoRA"):
                merge_loras_into_ckpt(base_ckpt, [LoraSpec(bad_lora, 1.0)], out_ckpt)
            self.assertFalse(out_ckpt.exists())

            # CLI dry-run smoke test
            with DrawThingsCkptStore(good_lora, writable=True) as l_good:
                l_good.write_tensor("__dit__[t-x_q-0-0]__up__", np.ones((4, 2), dtype=np.float32) * 0.1)
                l_good.write_tensor("__dit__[t-x_q-0-0]__down__", np.ones((2, 4), dtype=np.float32) * 0.1)

            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dt_lora_merge_ckpt",
                    "--base",
                    str(base_ckpt),
                    "--lora",
                    f"{good_lora}:0.5",
                    "--output",
                    str(out_ckpt),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("RESULT=PASS (dry-run)", res.stdout)
            self.assertFalse(out_ckpt.exists())


if __name__ == "__main__":
    unittest.main()
