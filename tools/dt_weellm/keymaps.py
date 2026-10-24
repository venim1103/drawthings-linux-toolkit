"""Unified WeeLLM <-> Draw Things Keymap Registry (`tools/dt_weellm/keymaps.py`).

Adapts key-mapping tables from `jit-roy/weellm`:
  - `weellm/io/ggufs/keymaps/transformers/{ltx25.py, flux2.py, flux3.py, ...}`
  - `weellm/io/safetensors/keymaps/loras/{ltx2.py, minimax.py, diffusers_fallback.py}`
  - `weellm/models/transformers/{moss_image_transformer_2d_model.py, ltx2_video_transformer_3d_model.py}`
  - `weellm/models/text_encoders/qwen3_5_model.py`
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STANDARD_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "transformer.",
    "dit.",
)

# GGUF -> Diffusers regex rules adapted from WeeLLM `weellm/io/ggufs/keymaps/transformers/`
GGUF_TO_DIFFUSERS_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = {
    "ltx2_3": [
        (re.compile(r"^blk\.(\d+)\.attn1\.q\.(weight|bias)$"), r"transformer_blocks.\1.attn1.to_q.\2"),
        (re.compile(r"^blk\.(\d+)\.attn1\.k\.(weight|bias)$"), r"transformer_blocks.\1.attn1.to_k.\2"),
        (re.compile(r"^blk\.(\d+)\.attn1\.v\.(weight|bias)$"), r"transformer_blocks.\1.attn1.to_v.\2"),
        (re.compile(r"^blk\.(\d+)\.attn1\.out\.(weight|bias)$"), r"transformer_blocks.\1.attn1.to_out.0.\2"),
        (re.compile(r"^blk\.(\d+)\.attn1\.q_norm\.weight$"), r"transformer_blocks.\1.attn1.q_norm.weight"),
        (re.compile(r"^blk\.(\d+)\.attn1\.k_norm\.weight$"), r"transformer_blocks.\1.attn1.k_norm.weight"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.(weight|bias)$"), r"transformer_blocks.\1.ff.net.0.proj.\2"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.(weight|bias)$"), r"transformer_blocks.\1.ff.net.2.\2"),
    ],
    "moss_image": [
        (re.compile(r"^blk\.(\d+)\.attn_q\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_q.\2"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_k.\2"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_v.\2"),
        (re.compile(r"^blk\.(\d+)\.attn_qkv\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_qkv.\2"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_out.0.\2"),
        (re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$"), r"transformer_blocks.\1.attn.norm_q.weight"),
        (re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$"), r"transformer_blocks.\1.attn.norm_k.weight"),
        (re.compile(r"^blk\.(\d+)\.ffn_up\.(weight|bias)$"), r"transformer_blocks.\1.ff.w13.\2"),
        (re.compile(r"^blk\.(\d+)\.ffn_down\.(weight|bias)$"), r"transformer_blocks.\1.ff.w2.\2"),
        (re.compile(r"^blk\.(\d+)\.adaLN_modulation\.(weight|bias)$"), r"transformer_blocks.\1.adaLN_modulation.1.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.wq\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_q.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.wk\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_k.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.wv\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_v.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.wqkv\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_qkv.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.wo\.(weight|bias)$"), r"transformer_blocks.\1.attn.to_out.0.\2"),
        (re.compile(r"^layers\.(\d+)\.attention\.q_norm\.weight$"), r"transformer_blocks.\1.attn.norm_q.weight"),
        (re.compile(r"^layers\.(\d+)\.attention\.k_norm\.weight$"), r"transformer_blocks.\1.attn.norm_k.weight"),
        (re.compile(r"^layers\.(\d+)\.feed_forward\.w13\.(weight|bias)$"), r"transformer_blocks.\1.ff.w13.\2"),
        (re.compile(r"^layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$"), r"transformer_blocks.\1.ff.w2.\2"),
        (re.compile(r"^layers\.(\d+)\.adaLN_modulation\.(weight|bias)$"), r"transformer_blocks.\1.adaLN_modulation.1.\2"),
    ],
    "qwen3_5": [
        (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"), r"model.layers.\1.self_attn.q_proj.weight"),
        (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"), r"model.layers.\1.self_attn.k_proj.weight"),
        (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"), r"model.layers.\1.self_attn.v_proj.weight"),
        (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"), r"model.layers.\1.self_attn.o_proj.weight"),
    ],
}


def get_architecture_aliases(canonical_key: str, profile: str = "ltx2_3") -> List[str]:
    """Return candidate source keys (Diffusers, ComfyUI, BFL, GGUF) for `canonical_key`."""
    aliases = [canonical_key]
    for p in STANDARD_PREFIXES:
        aliases.append(f"{p}{canonical_key}")

    if profile == "moss_image":
        # Map transformer_blocks.{i}.* <-> layers.{i}.* and blk.{i}.*
        m = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", canonical_key)
        if m:
            idx, rest = m.group(1), m.group(2)
            moss_map = {
                "attn.to_q.weight": (f"layers.{idx}.attention.wq.weight", f"blk.{idx}.attn_q.weight"),
                "attn.to_k.weight": (f"layers.{idx}.attention.wk.weight", f"blk.{idx}.attn_k.weight"),
                "attn.to_v.weight": (f"layers.{idx}.attention.wv.weight", f"blk.{idx}.attn_v.weight"),
                "attn.to_qkv.weight": (f"layers.{idx}.attention.wqkv.weight", f"blk.{idx}.attn_qkv.weight"),
                "attn.to_out.0.weight": (f"layers.{idx}.attention.wo.weight", f"blk.{idx}.attn_output.weight"),
                "attn.norm_q.weight": (
                    f"layers.{idx}.attention.q_norm.weight",
                    f"blk.{idx}.attn_q_norm.weight",
                    f"transformer_blocks.{idx}.attn.q_norm.weight",
                ),
                "attn.norm_k.weight": (
                    f"layers.{idx}.attention.k_norm.weight",
                    f"blk.{idx}.attn_k_norm.weight",
                    f"transformer_blocks.{idx}.attn.k_norm.weight",
                ),
                "ff.w13.weight": (
                    f"layers.{idx}.feed_forward.w13.weight",
                    f"blk.{idx}.ffn_up.weight",
                    f"transformer_blocks.{idx}.ff.net.0.proj.weight",
                ),
                "ff.w2.weight": (
                    f"layers.{idx}.feed_forward.w2.weight",
                    f"blk.{idx}.ffn_down.weight",
                    f"transformer_blocks.{idx}.ff.net.2.weight",
                ),
                "adaLN_modulation.1.weight": (
                    f"layers.{idx}.adaLN_modulation.weight",
                    f"blk.{idx}.adaLN_modulation.weight",
                ),
                "adaLN_modulation.1.bias": (
                    f"layers.{idx}.adaLN_modulation.bias",
                    f"blk.{idx}.adaLN_modulation.bias",
                ),
            }
            for alt in moss_map.get(rest, ()):
                aliases.append(alt)
                aliases.append(f"model.diffusion_model.{alt}")
    elif profile == "ltx2_3":
        m = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", canonical_key)
        if m:
            idx, rest = m.group(1), m.group(2)
            ltx_gguf_map = {
                "attn1.to_q.weight": f"blk.{idx}.attn1.q.weight",
                "attn1.to_k.weight": f"blk.{idx}.attn1.k.weight",
                "attn1.to_v.weight": f"blk.{idx}.attn1.v.weight",
                "attn1.to_out.0.weight": f"blk.{idx}.attn1.out.weight",
                "attn1.q_norm.weight": f"blk.{idx}.attn1.q_norm.weight",
                "attn1.k_norm.weight": f"blk.{idx}.attn1.k_norm.weight",
            }
            if rest in ltx_gguf_map:
                aliases.append(ltx_gguf_map[rest])

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def canonicalize_source_key(raw_key: str, profile: str = "ltx2_3") -> str:
    """Normalize a raw Safetensors/GGUF tensor key into its canonical contract key."""
    key = raw_key
    for p in STANDARD_PREFIXES:
        if key.startswith(p):
            key = key[len(p) :]
            break

    for pat, repl in GGUF_TO_DIFFUSERS_PATTERNS.get(profile, []):
        if pat.match(key):
            return pat.sub(repl, key)
    return key


# Kohya / LyCORIS underscore-mangled module tokens
_KOHYA_COMPOUND_TOKENS = (
    "transformer_blocks",
    "single_transformer_blocks",
    "text_refiner_blocks",
    "context_refiner",
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "to_gate_logits",
    "to_out",
    "to_qkv",
    "to_q",
    "to_k",
    "to_v",
    "q_norm",
    "k_norm",
    "norm_q",
    "norm_k",
    "adaLN_modulation",
    "feed_forward",
)


def _unmangle_kohya_stem(stem: str) -> str:
    """Convert Kohya/LyCORIS `lora_unet_transformer_blocks_0_attn1_to_q` -> `transformer_blocks.0.attn1.to_q`."""
    s = stem
    for prefix in ("lora_unet_", "lora_te_", "lycoris_", "lora_transformer_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break

    # Protect compound tokens with placeholders before replacing `_` with `.`
    placeholders: Dict[str, str] = {}
    for idx, tok in enumerate(_KOHYA_COMPOUND_TOKENS):
        ph = f"__TOK{idx}__"
        placeholders[ph] = tok
        s = s.replace(tok, ph)

    s = s.replace("_", ".")
    for ph, tok in placeholders.items():
        s = s.replace(ph, tok)
    return s


def canonicalize_lora_key(
    raw_key: str,
    profile: str = "ltx2_3",
) -> Optional[Tuple[str, str]]:
    """Parse any Kohya, LyCORIS, ComfyUI, BFL, or Diffusers LoRA key into `(canonical_base_weight_key, kind)`.

    `kind` is one of `"down"` (`lora_A`), `"up"` (`lora_B`), or `"alpha"`.
    """
    key = raw_key
    for prefix in STANDARD_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    kind: Optional[str] = None
    stem: str = ""
    if key.endswith(".alpha") or key.endswith(".lora_alpha"):
        kind = "alpha"
        stem = key[: key.rfind(".")]
    else:
        for down_sfx in (".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".lora_A.default.weight"):
            if key.endswith(down_sfx):
                kind = "down"
                stem = key[: -len(down_sfx)]
                break
        if kind is None:
            for up_sfx in (".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".lora_B.default.weight"):
                if key.endswith(up_sfx):
                    kind = "up"
                    stem = key[: -len(up_sfx)]
                    break

    if kind is None:
        return None

    if stem.startswith(("lora_unet_", "lora_te_", "lycoris_", "lora_transformer_")) or "." not in stem:
        stem = _unmangle_kohya_stem(stem)

    canonical_weight = canonicalize_source_key(f"{stem}.weight", profile=profile)
    return canonical_weight, kind


def load_external_weellm_keymap(py_path: str | Path) -> Dict[str, str]:
    """Safely extract dictionary literals from a WeeLLM keymap `.py` file via `ast.literal_eval`."""
    path = Path(py_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WeeLLM keymap file not found: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    extracted: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            try:
                d = ast.literal_eval(node)
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(k, str) and isinstance(v, str):
                            extracted[k] = v
            except Exception:
                continue
    return extracted
