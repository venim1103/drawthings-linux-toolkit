import json
import struct
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
    inspect_ckpt_health,
    reference_gated_delta_rule,
    reference_rmsnorm_gated,
    verify_block_parity,
)
from tools.dt_lora_universal_bridge import normalize_lora_safetensors, verify_converted_lora
from tools.dt_weellm.dequant import (
    dequantize_fp8_e4m3,
    dequantize_fp8_e5m2,
    dequantize_gguf_q4_0,
    dequantize_gguf_q8_0,
)
from tools.dt_weellm.seek_io import GGUFSeeker, SafetensorsSeeker
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
    def test_ltx23_contract_matches_exact_verified_counts(self):
        contract = build_ltx23_contract()
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
        self.assertEqual(len(ltx_contract["tensors"]), 1252)
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
