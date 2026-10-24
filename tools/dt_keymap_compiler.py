#!/usr/bin/env python3
"""Universal Keymap & Contract Compiler + Static Swift Mapper Auditor (Milestone W1).

Generates declarative model conversion/validation contracts (`contracts/<profile>.json`)
and statically audits Draw Things Swift model builders (`LTX2.swift`, `MOSSImage.swift`,
`ModelImporter.swift`) against known converter failure modes:
  - Rule 1 (§18): Unflagged `q_norm` / `k_norm` / `norm_q` / `norm_k` rotary de-interleaving.
  - Rule 2 (§3a / M3): Missing BF16/F32 preservation on aggregate embeds / timestep / modulation tensors.
  - Rule 3 (M3): `fatalError()` or missing branches in `ModelImporter.swift`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class SwiftAuditFinding:
    rule_id: str
    severity: str  # "ERROR" | "WARN"
    file_path: str
    line_number: int
    function_name: str
    key_pattern: str
    message: str


def build_ltx23_contract() -> Dict[str, Any]:
    """Generate the declarative 5,746-tensor contract for LTX 2.3 (`ltx2_3`).

    Matches every verified rule from `CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md`:
      - 48 DiT blocks (`__dit__`) + 2 connectors (`__text_video_connector__`, `__text_audio_connector__`)
        + `__text_feature_extractor__`
      - §3a: 4 BF16 aggregate embedding tensors (`isBF16: true`, code 524288)
      - §17: Rotary de-interleaving on all `to_q` / `to_k` weights and biases (32 heads)
      - §18: Rotary de-interleaving on all 608 `q_norm` / `k_norm` weights (32 heads)
      - §22: `[1, 1, N]` rank-3 `dim` layout and F32 restoration on all 1,540 `ada_ln`
        modulation tensors and 608 `q_norm` / `k_norm` weights.
    """
    tensors: Dict[str, Dict[str, Any]] = {}

    def add_entry(
        src_key: str,
        dt_names: List[str],
        *,
        rotary: bool = False,
        heads: int = 0,
        head_dim: int = 0,
        f16_dtype: str = "F16",
        quant_restore: str = "none",
        rank_policy: str = "preserve",
        split_axis: int = 0,
    ) -> None:
        tensors[src_key] = {
            "dt_names": dt_names,
            "rotary_interleaved": rotary,
            "num_heads": heads,
            "head_dim": head_dim,
            "f16_store_datatype": f16_dtype,
            "quant_restore_policy": quant_restore,
            "dim_rank_policy": rank_policy,
            "split_axis": split_axis,
        }

    # 1. Embedders & Output Projections (`__dit__`)
    add_entry("patchify_proj.weight", ["__dit__[t-x_embedder-0-0]"])
    add_entry("patchify_proj.bias", ["__dit__[t-x_embedder-0-1]"])
    add_entry("audio_patchify_proj.weight", ["__dit__[t-a_embedder-0-0]"])
    add_entry("audio_patchify_proj.bias", ["__dit__[t-a_embedder-0-1]"])
    add_entry("proj_out.weight", ["__dit__[t-proj_out-0-0]"])
    add_entry("proj_out.bias", ["__dit__[t-proj_out-0-1]"])
    add_entry("audio_proj_out.weight", ["__dit__[t-audio_proj_out-0-0]"])
    add_entry("audio_proj_out.bias", ["__dit__[t-audio_proj_out-0-1]"])

    # 2. Text Feature Extractor BF16 Aggregate Embeds (§3a: 4 BF16 tensors)
    add_entry(
        "text_embedding_projection.video_aggregate_embed.weight",
        ["__text_feature_extractor__[t-video_aggregate_embed-0-0]"],
        f16_dtype="BF16",
        quant_restore="bf16_from_source",
    )
    add_entry(
        "text_embedding_projection.video_aggregate_embed.bias",
        ["__text_feature_extractor__[t-video_aggregate_embed-0-1]"],
        f16_dtype="BF16",
        quant_restore="bf16_from_source",
    )
    add_entry(
        "text_embedding_projection.audio_aggregate_embed.weight",
        ["__text_feature_extractor__[t-audio_aggregate_embed-0-0]"],
        f16_dtype="BF16",
        quant_restore="bf16_from_source",
    )
    add_entry(
        "text_embedding_projection.audio_aggregate_embed.bias",
        ["__text_feature_extractor__[t-audio_aggregate_embed-0-1]"],
        f16_dtype="BF16",
        quant_restore="bf16_from_source",
    )

    # 3. Global AdaLN Single Timestep Embedders (`LTX2Fixed`)
    adaln_singles = [
        ("adaln_single", "tx", 9),
        ("audio_adaln_single", "ta", 9),
        ("prompt_adaln_single", "ptx", 2),
        ("audio_prompt_adaln_single", "pta", 2),
        ("av_ca_video_scale_shift_adaln_single", "tcx", 4),
        ("av_ca_audio_scale_shift_adaln_single", "tca", 4),
        ("av_ca_a2v_gate_adaln_single", "a2v", 1),
        ("av_ca_v2a_gate_adaln_single", "v2a", 1),
    ]
    for src_prefix, dt_tok, count in adaln_singles:
        add_entry(
            f"{src_prefix}.emb.timestep_embedder.linear_1.weight",
            [f"__dit__[t-{dt_tok}_embedder_0-0-0]"],
        )
        add_entry(
            f"{src_prefix}.emb.timestep_embedder.linear_1.bias",
            [f"__dit__[t-{dt_tok}_embedder_0-0-1]"],
        )
        add_entry(
            f"{src_prefix}.emb.timestep_embedder.linear_2.weight",
            [f"__dit__[t-{dt_tok}_embedder_1-0-0]"],
        )
        add_entry(
            f"{src_prefix}.emb.timestep_embedder.linear_2.bias",
            [f"__dit__[t-{dt_tok}_embedder_1-0-1]"],
        )
        add_entry(
            f"{src_prefix}.linear.weight",
            [f"__dit__[t-{dt_tok}_adaln_single_{c}-0-0]" for c in range(count)],
        )
        add_entry(
            f"{src_prefix}.linear.bias",
            [f"__dit__[t-{dt_tok}_adaln_single_{c}-0-1]" for c in range(count)],
        )

    # Output scale-shift modulations (4 F32ada_ln tensors)
    add_entry(
        "scale_shift_table",
        ["__dit__[t-norm_out_ada_ln_0-0-0]", "__dit__[t-norm_out_ada_ln_1-0-0]"],
        f16_dtype="F32",
        quant_restore="f32_from_source",
        rank_policy="norm_1_1_n",
    )
    add_entry(
        "audio_scale_shift_table",
        [
            "__dit__[t-audio_norm_out_ada_ln_0-0-0]",
            "__dit__[t-audio_norm_out_ada_ln_1-0-0]",
        ],
        f16_dtype="F32",
        quant_restore="f32_from_source",
        rank_policy="norm_1_1_n",
    )

    # 4. 1D Connectors (`video_embeddings_connector` & `audio_embeddings_connector`, 8 blocks each)
    connectors = [
        ("video_embeddings_connector", "__text_video_connector__", 32, 128),
        ("audio_embeddings_connector", "__text_audio_connector__", 32, 64),
    ]
    for src_conn, dt_conn, heads, hd in connectors:
        add_entry(
            f"{src_conn}.learnable_registers",
            [f"{dt_conn}[t-learnable_registers-0-0]"],
        )
        for i in range(8):
            base = f"{src_conn}.transformer_1d_blocks.{i}"
            add_entry(
                f"{base}.attn1.to_q.weight",
                [f"{dt_conn}[t-to_q-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{base}.attn1.to_q.bias",
                [f"{dt_conn}[t-to_q-{i}-1]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{base}.attn1.to_k.weight",
                [f"{dt_conn}[t-to_k-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{base}.attn1.to_k.bias",
                [f"{dt_conn}[t-to_k-{i}-1]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(f"{base}.attn1.to_v.weight", [f"{dt_conn}[t-to_v-{i}-0]"])
            add_entry(f"{base}.attn1.to_v.bias", [f"{dt_conn}[t-to_v-{i}-1]"])
            add_entry(f"{base}.attn1.to_gate_logits.weight", [f"{dt_conn}[t-to_gate-{i}-0]"])
            add_entry(f"{base}.attn1.to_gate_logits.bias", [f"{dt_conn}[t-to_gate-{i}-1]"])
            add_entry(f"{base}.attn1.to_out.0.weight", [f"{dt_conn}[t-to_o-{i}-0]"])
            add_entry(f"{base}.attn1.to_out.0.bias", [f"{dt_conn}[t-to_o-{i}-1]"])
            add_entry(
                f"{base}.attn1.q_norm.weight",
                [f"{dt_conn}[t-norm_q-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
                quant_restore="f32_from_source",
                rank_policy="norm_1_1_n",
            )
            add_entry(
                f"{base}.attn1.k_norm.weight",
                [f"{dt_conn}[t-norm_k-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
                quant_restore="f32_from_source",
                rank_policy="norm_1_1_n",
            )
            add_entry(f"{base}.ff.net.0.proj.weight", [f"{dt_conn}[t-up_proj-{i}-0]"])
            add_entry(f"{base}.ff.net.0.proj.bias", [f"{dt_conn}[t-up_proj-{i}-1]"])
            add_entry(f"{base}.ff.net.2.weight", [f"{dt_conn}[t-down_proj-{i}-0]"])
            add_entry(f"{base}.ff.net.2.bias", [f"{dt_conn}[t-down_proj-{i}-1]"])

    # 5. 48 Main DiT Transformer Blocks
    attn_specs = [
        ("attn1", "x", 32, 128),
        ("attn2", "cv", 32, 128),
        ("audio_attn1", "a", 32, 64),
        ("audio_attn2", "ca", 32, 64),
        ("audio_to_video_attn", "ax", 32, 64),
        ("video_to_audio_attn", "xa", 32, 64),
    ]
    for i in range(48):
        p = f"transformer_blocks.{i}"
        for attn_sub, tok, heads, hd in attn_specs:
            ap = f"{p}.{attn_sub}"
            add_entry(
                f"{ap}.to_q.weight",
                [f"__dit__[t-{tok}_q-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{ap}.to_q.bias",
                [f"__dit__[t-{tok}_q-{i}-1]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{ap}.to_k.weight",
                [f"__dit__[t-{tok}_k-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(
                f"{ap}.to_k.bias",
                [f"__dit__[t-{tok}_k-{i}-1]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
            )
            add_entry(f"{ap}.to_v.weight", [f"__dit__[t-{tok}_v-{i}-0]"])
            add_entry(f"{ap}.to_v.bias", [f"__dit__[t-{tok}_v-{i}-1]"])
            add_entry(f"{ap}.to_gate_logits.weight", [f"__dit__[t-{tok}_gate-{i}-0]"])
            add_entry(f"{ap}.to_gate_logits.bias", [f"__dit__[t-{tok}_gate-{i}-1]"])
            add_entry(f"{ap}.to_out.0.weight", [f"__dit__[t-{tok}_o-{i}-0]"])
            add_entry(f"{ap}.to_out.0.bias", [f"__dit__[t-{tok}_o-{i}-1]"])
            add_entry(
                f"{ap}.q_norm.weight",
                [f"__dit__[t-{tok}_norm_q-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
                quant_restore="f32_from_source",
                rank_policy="norm_1_1_n",
            )
            add_entry(
                f"{ap}.k_norm.weight",
                [f"__dit__[t-{tok}_norm_k-{i}-0]"],
                rotary=True,
                heads=heads,
                head_dim=hd,
                quant_restore="f32_from_source",
                rank_policy="norm_1_1_n",
            )

        # Feed-forward layers (video & audio)
        add_entry(f"{p}.ff.net.0.proj.weight", [f"__dit__[t-x_linear1-{i}-0]"])
        add_entry(f"{p}.ff.net.0.proj.bias", [f"__dit__[t-x_linear1-{i}-1]"])
        add_entry(f"{p}.ff.net.2.weight", [f"__dit__[t-x_out_proj-{i}-0]"])
        add_entry(f"{p}.ff.net.2.bias", [f"__dit__[t-x_out_proj-{i}-1]"])
        add_entry(f"{p}.audio_ff.net.0.proj.weight", [f"__dit__[t-a_linear1-{i}-0]"])
        add_entry(f"{p}.audio_ff.net.0.proj.bias", [f"__dit__[t-a_linear1-{i}-1]"])
        add_entry(f"{p}.audio_ff.net.2.weight", [f"__dit__[t-a_out_proj-{i}-0]"])
        add_entry(f"{p}.audio_ff.net.2.bias", [f"__dit__[t-a_out_proj-{i}-1]"])

        # Block AdaLN Modulation Tables (32 F32 tensors per block -> 48 * 32 = 1536 tensors)
        add_entry(
            f"{p}.scale_shift_table",
            [f"__dit__[t-attn1_ada_ln_{c}-{i}-0]" for c in range(9)],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )
        add_entry(
            f"{p}.audio_scale_shift_table",
            [f"__dit__[t-audio_attn1_ada_ln_{c}-{i}-0]" for c in range(9)],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )
        add_entry(
            f"{p}.prompt_scale_shift_table",
            [f"__dit__[t-prompt_scale_shift_ada_ln_{c}-{i}-0]" for c in range(2)],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )
        add_entry(
            f"{p}.audio_prompt_scale_shift_table",
            [f"__dit__[t-audio_prompt_scale_shift_ada_ln_{c}-{i}-0]" for c in range(2)],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )
        add_entry(
            f"{p}.scale_shift_table_a2v_ca_audio",
            [
                f"__dit__[t-audio_to_video_attn_ada_ln_3-{i}-0]",
                f"__dit__[t-audio_to_video_attn_ada_ln_2-{i}-0]",
                f"__dit__[t-video_to_audio_attn_ada_ln_3-{i}-0]",
                f"__dit__[t-video_to_audio_attn_ada_ln_2-{i}-0]",
                f"__dit__[t-video_to_audio_attn_ada_ln_4-{i}-0]",
            ],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )
        add_entry(
            f"{p}.scale_shift_table_a2v_ca_video",
            [
                f"__dit__[t-audio_to_video_attn_ada_ln_1-{i}-0]",
                f"__dit__[t-audio_to_video_attn_ada_ln_0-{i}-0]",
                f"__dit__[t-video_to_audio_attn_ada_ln_1-{i}-0]",
                f"__dit__[t-video_to_audio_attn_ada_ln_0-{i}-0]",
                f"__dit__[t-audio_to_video_attn_ada_ln_4-{i}-0]",
            ],
            f16_dtype="F32",
            quant_restore="f32_from_source",
            rank_policy="norm_1_1_n",
        )

    total_dt_tensors = sum(len(v["dt_names"]) for v in tensors.values())
    f32_ada_ln_count = sum(
        len(v["dt_names"])
        for v in tensors.values()
        if v["f16_store_datatype"] == "F32"
    )
    qk_norm_count = sum(
        len(v["dt_names"])
        for k, v in tensors.items()
        if ("q_norm" in k or "k_norm" in k)
    )
    bf16_count = sum(
        len(v["dt_names"])
        for v in tensors.values()
        if v["f16_store_datatype"] == "BF16"
    )

    return {
        "schema_version": 1,
        "profile": "ltx2_3",
        "model_version": "ltx2.3",
        "expected_ckpt_tensor_count": total_dt_tensors,
        "expected_counts_summary": {
            "total_ckpt_tensors": total_dt_tensors,
            "f32_ada_ln_tensors": f32_ada_ln_count,
            "qk_norm_tensors": qk_norm_count,
            "bf16_aggregate_embed_tensors": bf16_count,
        },
        "companions": {
            "autoencoder": "ltx_2.3_audio_video_vae_f16.ckpt",
            "text_encoder": "gemma_3_12b_it_qat_q8p.ckpt",
            "modifier": "kontext",
            "default_scale": 12,
            "hires_fix_scale": 24,
            "objective": {"u": {"condition_scale": 1000}},
        },
        "geometry_guards": {
            "latent_unit_pixels": 64,
            "temporal_frame_rule": "8k+1",
            "min_video_frames": 9,
        },
        "source_required_keys": [
            "transformer_blocks.47.attn1.to_q.weight",
            "audio_patchify_proj.weight",
            "video_embeddings_connector.learnable_registers",
            "audio_embeddings_connector.learnable_registers",
            "video_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
            "audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
        ],
        "ckpt_required_prefixes": {
            "__dit__[t-a_embedder-": 1,
            "__dit__[t-x_embedder-": 1,
            "__text_audio_connector__[t-down_proj-": 1,
            "__text_video_connector__[t-down_proj-": 1,
        },
        "restore_patterns": {
            "f32_ada_ln_regex": r"ada_ln",
            "f32_qk_norm_regex": r"(norm_q|norm_k)-\d+-0\]$",
            "bf16_embed_regex": r"_aggregate_embed-",
        },
        "tensors": tensors,
    }


def build_moss_image_contract(ref_ckpt: Optional[Path] = None) -> Dict[str, Any]:
    """Generate the declarative contract for MOSS-Image (`moss_image`, 42 layers, 24 heads x 128 dim).

    Incorporates verified architecture specs from `docs/UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md`
    (§8, §9, §10 M1/M3) and WeeLLM's `moss_image_transformer_2d_model.py`.
    If `--ref-ckpt` (such as `dt-models/moss_image_base_q8p.ckpt`) is provided, also snapshots the
    exact SQLite tensor keyset and F32 preservation list from the official checkpoint.
    """
    tensors: Dict[str, Dict[str, Any]] = {}
    num_heads = 24
    head_dim = 128

    # 1. Embedders, Output Projections, Time Embedders
    tensors["patchify_proj.weight"] = {
        "dt_names": ["__dit__[t-x_embedder-0-0]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }
    tensors["patchify_proj.bias"] = {
        "dt_names": ["__dit__[t-x_embedder-0-1]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }
    tensors["timestep_embedder.linear_1.weight"] = {
        "dt_names": ["__dit__[t-t_embedder_0-0-0]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }
    tensors["timestep_embedder.linear_1.bias"] = {
        "dt_names": ["__dit__[t-t_embedder_0-0-1]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }
    # Second timestep projection is FP32-sensitive in Draw Things (MOSS M3 requirement)
    tensors["timestep_embedder.linear_2.weight"] = {
        "dt_names": ["__dit__[t-t_embedder_1-0-0]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F32",
        "quant_restore_policy": "f32_from_source",
        "dim_rank_policy": "preserve",
    }
    tensors["timestep_embedder.linear_2.bias"] = {
        "dt_names": ["__dit__[t-t_embedder_1-0-1]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F32",
        "quant_restore_policy": "f32_from_source",
        "dim_rank_policy": "preserve",
    }
    tensors["norm_out.weight"] = {
        "dt_names": ["__dit__[t-norm_out-0-0]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F32",
        "quant_restore_policy": "f32_from_source",
        "dim_rank_policy": "norm_1_1_n",
    }
    tensors["norm_out.bias"] = {
        "dt_names": ["__dit__[t-norm_out-0-1]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F32",
        "quant_restore_policy": "f32_from_source",
        "dim_rank_policy": "norm_1_1_n",
    }
    tensors["proj_out.weight"] = {
        "dt_names": ["__dit__[t-proj_out-0-0]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }
    tensors["proj_out.bias"] = {
        "dt_names": ["__dit__[t-proj_out-0-1]"],
        "rotary_interleaved": False,
        "f16_store_datatype": "F16",
        "quant_restore_policy": "none",
        "dim_rank_policy": "preserve",
    }

    # 2. Context Refiner Blocks (2 layers)
    for i in range(2):
        p = f"text_refiner_blocks.{i}"
        tensors[f"{p}.attn.to_q.weight"] = {
            "dt_names": [f"__dit__[t-refiner_q-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_k.weight"] = {
            "dt_names": [f"__dit__[t-refiner_k-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_v.weight"] = {
            "dt_names": [f"__dit__[t-refiner_v-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_out.0.weight"] = {
            "dt_names": [f"__dit__[t-refiner_o-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.ff.net.0.proj.weight"] = {
            "dt_names": [f"__dit__[t-refiner_ff1-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.ff.net.2.weight"] = {
            "dt_names": [f"__dit__[t-refiner_ff2-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }

    # 3. 42 Main DiT Blocks
    for i in range(42):
        p = f"transformer_blocks.{i}"
        tensors[f"{p}.attn.to_q.weight"] = {
            "dt_names": [f"__dit__[t-x_q-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_q.bias"] = {
            "dt_names": [f"__dit__[t-x_q-{i}-1]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_k.weight"] = {
            "dt_names": [f"__dit__[t-x_k-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_k.bias"] = {
            "dt_names": [f"__dit__[t-x_k-{i}-1]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_v.weight"] = {
            "dt_names": [f"__dit__[t-x_v-{i}-0]"],
            "rotary_interleaved": False,
            "num_heads": 0,
            "head_dim": 0,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_v.bias"] = {
            "dt_names": [f"__dit__[t-x_v-{i}-1]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_out.0.weight"] = {
            "dt_names": [f"__dit__[t-x_o-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.to_out.0.bias"] = {
            "dt_names": [f"__dit__[t-x_o-{i}-1]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.attn.norm_q.weight"] = {
            "dt_names": [f"__dit__[t-norm_q-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F32",
            "quant_restore_policy": "f32_from_source",
            "dim_rank_policy": "norm_1_1_n",
        }
        tensors[f"{p}.attn.norm_k.weight"] = {
            "dt_names": [f"__dit__[t-norm_k-{i}-0]"],
            "rotary_interleaved": True,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "f16_store_datatype": "F32",
            "quant_restore_policy": "f32_from_source",
            "dim_rank_policy": "norm_1_1_n",
        }
        tensors[f"{p}.ff.net.0.proj.weight"] = {
            "dt_names": [f"__dit__[t-x_linear1-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.ff.net.0.proj.bias"] = {
            "dt_names": [f"__dit__[t-x_linear1-{i}-1]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.ff.net.2.weight"] = {
            "dt_names": [f"__dit__[t-x_out_proj-{i}-0]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.ff.net.2.bias"] = {
            "dt_names": [f"__dit__[t-x_out_proj-{i}-1]"],
            "rotary_interleaved": False,
            "f16_store_datatype": "F16",
            "quant_restore_policy": "none",
            "dim_rank_policy": "preserve",
        }
        tensors[f"{p}.scale_shift_table"] = {
            "dt_names": [f"__dit__[t-attn1_ada_ln_{c}-{i}-0]" for c in range(6)],
            "rotary_interleaved": False,
            "f16_store_datatype": "F32",
            "quant_restore_policy": "f32_from_source",
            "dim_rank_policy": "norm_1_1_n",
        }

    ref_f32_keys: List[str] = []
    ref_total_keys = 0
    if ref_ckpt is not None and ref_ckpt.is_file():
        from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
        from tools.dt_weellm.transforms import DT_FLOAT32

        with DrawThingsCkptStore(ref_ckpt, writable=False) as store:
            all_keys = store.list_keys()
            ref_total_keys = len(all_keys)
            for k in all_keys:
                meta = store.read_metadata(k)
                if meta is not None and meta.datatype == DT_FLOAT32:
                    ref_f32_keys.append(k)

    return {
        "schema_version": 1,
        "profile": "moss_image",
        "model_version": "moss_image",
        "architecture": {
            "layers": 42,
            "text_refiner_layers": 2,
            "hidden_size": 3072,
            "num_heads": 24,
            "head_dim": 128,
            "ffn_hidden_size": 8192,
            "vae_channels": 32,
            "packed_patch_channels": 128,
            "unconditional_skip_layer": 22,
        },
        "companions": {
            "autoencoder": "flux_2_vae_f16.ckpt",
            "text_encoder": "qwen_3.5_9b_i5x.ckpt",
            "modifier": "none",
            "default_scale": 16,
            "objective": {"u": {"condition_scale": 1000}},
        },
        "geometry_guards": {
            "latent_unit_pixels": 64,
            "temporal_frame_rule": "single_frame",
            "min_video_frames": 1,
        },
        "source_required_keys": [
            "transformer_blocks.41.attn.to_q.weight",
        ],
        "ckpt_required_prefixes": {
            "__dit__[": 1,
        },
        "restore_patterns": {
            "f32_ada_ln_regex": r"(ada_ln|t_embedder|time_embed|modulation)",
            "f32_qk_norm_regex": r"(norm_q|norm_k)-\d+-0\]$",
        },
        "reference_ckpt_snapshot": {
            "source_file": str(ref_ckpt) if ref_ckpt else None,
            "total_tensors": ref_total_keys,
            "f32_preserved_tensors": sorted(ref_f32_keys),
        },
        "tensors": tensors,
    }


# --- Static Swift Mapper Auditor (`audit-swift`) ---

_FUNC_DECL_RE = re.compile(r"^\s*(?:public\s+|private\s+|internal\s+)?func\s+([A-Za-z0-9_]+)\s*\(")
_MAPPING_ASSIGN_RE = re.compile(
    r'mapping\[\s*"([^"]+)"\s*\]\s*=\s*(.+)'
)


def audit_swift_mapper_source(
    swift_text: str,
    file_label: str = "<swift>",
    include_lora_and_unused_1d: bool = False,
) -> List[SwiftAuditFinding]:
    """Scan a Draw Things Swift model file for §18 QK-norm interleave omissions.

    In Draw Things Swift model files (`LTX2.swift`, `MOSSImage.swift`, `ModelImporter.swift`),
    each block builder function defines a `ModelWeightMapper` closure. Whenever a scope maps
    `to_q` or `to_k` with `interleaved: true`, any `q_norm` / `k_norm` / `norm_q` / `norm_k`
    mapped in the same function MUST also use `ModelWeightElement(..., interleaved: true, ...)`
    or else the runtime RMSNorm scales scrambled rotary channels (§18).
    """
    lines = swift_text.splitlines()
    findings: List[SwiftAuditFinding] = []

    current_func = "<top>"
    # Collect per-function mapping assignments: (line_no, key_pattern, rhs_text)
    func_mappings: Dict[str, List[Tuple[int, str, str]]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m_func = _FUNC_DECL_RE.match(line)
        if m_func:
            current_func = m_func.group(1)
            func_mappings.setdefault(current_func, [])

        m_map = _MAPPING_ASSIGN_RE.search(line)
        if m_map:
            key_pat = m_map.group(1)
            rhs_lines = [m_map.group(2).strip()]
            # Join multi-line ModelWeightElement(...) calls up to closing paren or next mapping
            j = i + 1
            while (
                j < len(lines)
                and "ModelWeightElement" in rhs_lines[0]
                and ")" not in rhs_lines[-1]
                and j < i + 6
            ):
                rhs_lines.append(lines[j].strip())
                j += 1
            rhs_full = " ".join(rhs_lines)
            func_mappings.setdefault(current_func, []).append((i + 1, key_pat, rhs_full))
        i += 1

    for func_name, entries in func_mappings.items():
        if not include_lora_and_unused_1d:
            # As documented in CUSTOM_MODEL_CONVERTER_FINDINGS §19.3, LoRA* and BasicTransformerBlock1D
            # are not used by the base model importer (`LTX2`, `LTX2Fixed`, `makeLTX23ConnectorMapping` are).
            if func_name.startswith("LoRA") or func_name == "BasicTransformerBlock1D":
                continue

        interleaved_qk_heads: Optional[Tuple[str, str]] = None
        for _, key_pat, rhs in entries:
            if any(tok in key_pat for tok in ("to_q.weight", "to_k.weight", "to_qkv.weight")):
                if "interleaved: true" in rhs:
                    m_h = re.search(r"numberOfHeads:\s*([A-Za-z0-9_.]+)", rhs)
                    m_d = re.search(r"headDimension:\s*([A-Za-z0-9_.]+)", rhs)
                    interleaved_qk_heads = (
                        m_h.group(1) if m_h else "?",
                        m_d.group(1) if m_d else "?",
                    )
                    break

        if interleaved_qk_heads is not None:
            exp_h, exp_d = interleaved_qk_heads
            for line_no, key_pat, rhs in entries:
                if any(
                    norm_tok in key_pat
                    for norm_tok in ("q_norm.weight", "k_norm.weight", "norm_q.weight", "norm_k.weight")
                ):
                    if "interleaved: true" not in rhs:
                        findings.append(
                            SwiftAuditFinding(
                                rule_id="DT-CONV-018",
                                severity="ERROR",
                                file_path=file_label,
                                line_number=line_no,
                                function_name=func_name,
                                key_pattern=key_pat,
                                message=(
                                    f"QK-norm {key_pat!r} in {func_name}() is plain-mapped without "
                                    f"`interleaved: true` while sibling Q/K projection is rotary-interleaved "
                                    f"(expected numberOfHeads: {exp_h}, headDimension: {exp_d})."
                                ),
                            )
                        )

    # Rule 2 (§3a): Aggregate embedding tensors in ModelImporter.swift must specify `isBF16: true`
    for func_name, entries in func_mappings.items():
        for line_no, key_pat, rhs in entries:
            if "aggregate_embed" in key_pat and "isBF16: true" not in rhs:
                findings.append(
                    SwiftAuditFinding(
                        rule_id="DT-CONV-003A",
                        severity="ERROR",
                        file_path=file_label,
                        line_number=line_no,
                        function_name=func_name,
                        key_pattern=key_pat,
                        message=(
                            f"Aggregate embedding {key_pat!r} is mapped without `isBF16: true`; "
                            "downcasting to Float16 causes ccv_nnc_tensor_read SIGSEGV at model load (§3a)."
                        ),
                    )
                )

    return findings


def cmd_compile(args: argparse.Namespace) -> int:
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_ckpt = Path(args.ref_ckpt).expanduser().resolve() if args.ref_ckpt else None

    if args.profile == "ltx2_3":
        contract = build_ltx23_contract()
    elif args.profile == "moss_image":
        contract = build_moss_image_contract(ref_ckpt=ref_ckpt)
    else:
        print(f"error: unsupported profile: {args.profile}", file=sys.stderr)
        return 2

    out_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"== Contract Compiled ({args.profile}) ==")
    print(f"output={out_path}")
    if "expected_counts_summary" in contract:
        for k, v in contract["expected_counts_summary"].items():
            print(f"  {k}={v}")
    return 0


def cmd_audit_swift(args: argparse.Namespace) -> int:
    all_findings: List[SwiftAuditFinding] = []
    for path_str in args.swift_files:
        p = Path(path_str).expanduser().resolve()
        if not p.is_file():
            print(f"error: swift file not found: {p}", file=sys.stderr)
            return 2
        text = p.read_text(encoding="utf-8")
        findings = audit_swift_mapper_source(
            text,
            file_label=str(p),
            include_lora_and_unused_1d=args.include_unused_mappers,
        )
        all_findings.extend(findings)

    print("== Swift Mapper Static Audit ==")
    print(f"files_scanned={len(args.swift_files)} findings={len(all_findings)}")
    for f in all_findings:
        print(
            f"[{f.severity}] {f.rule_id} {f.file_path}:{f.line_number} ({f.function_name}): {f.message}"
        )

    if any(f.severity == "ERROR" for f in all_findings):
        print("RESULT=FAIL")
        return 1
    print("RESULT=PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Draw Things Contract Compiler & Swift Mapper Auditor")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_compile = sub.add_parser("compile", help="Compile a declarative model contract JSON")
    p_compile.add_argument("--profile", choices=["ltx2_3", "moss_image"], required=True)
    p_compile.add_argument("--ref-ckpt", default="", help="Optional reference .ckpt to snapshot F32 set")
    p_compile.add_argument("--output", required=True, help="Output JSON contract path")

    p_audit = sub.add_parser("audit-swift", help="Audit Swift model/importer files for §18/§3a bugs")
    p_audit.add_argument("swift_files", nargs="+", help="Swift source files to audit")
    p_audit.add_argument(
        "--include-unused-mappers",
        action="store_true",
        help="Also flag unused BasicTransformerBlock1D / LoRA* mappers",
    )

    args = parser.parse_args(argv)
    if args.subcommand == "compile":
        return cmd_compile(args)
    if args.subcommand == "audit-swift":
        return cmd_audit_swift(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
