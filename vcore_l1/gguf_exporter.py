# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from safetensors.torch import load_file
from .safetensors_quantizer import (
    unpack_uint8_to_ternary,
    unpack_uint8_to_int4,
)


def _map_hf_to_gguf_name(hf_name: str, arch: str) -> Optional[str]:
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"

    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_id = parts [2]
        rest = ".".join(parts[3:])
        mapping = {
            "input_layernorm.weight": f"blk.{layer_id}.attn_norm.weight",
            "self_attn.q_proj.weight": f"blk.{layer_id}.attn_q.weight",
            "self_attn.q_proj.bias": f"blk.{layer_id}.attn_q.bias",
            "self_attn.k_proj.weight": f"blk.{layer_id}.attn_k.weight",
            "self_attn.k_proj.bias": f"blk.{layer_id}.attn_k.bias",
            "self_attn.v_proj.weight": f"blk.{layer_id}.attn_v.weight",
            "self_attn.v_proj.bias": f"blk.{layer_id}.attn_v.bias",
            "self_attn.o_proj.weight": f"blk.{layer_id}.attn_output.weight",
            "post_attention_layernorm.weight": f"blk.{layer_id}.ffn_norm.weight",
            "mlp.gate_proj.weight": f"blk.{layer_id}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer_id}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer_id}.ffn_down.weight",
        }
        return mapping.get(rest, None)
    return None


def _pack_tq2_0_numpy(ternary_grid: np.ndarray, block_scales: np.ndarray, group_size: int) -> Optional[np.ndarray]:
    """
    Прямая упаковка 1.58-bit весов {-1, 0, +1} в нативный формат llama.cpp TQ2_0
    (блок 256 элементов: 64 байта упакованных 2-битных кодов + 2 байта float16 scale = 66 байт).
    """
    if group_size != 256 or ternary_grid.ndim != 2:
        return None
    out_f, in_f = ternary_grid.shape
    if in_f % 256 != 0:
        return None

    num_blocks = (out_f * in_f) // 256
    if block_scales.size != num_blocks:
        return None

    codes = (np.rint(ternary_grid).astype(np.int8) + 1).astype(np.uint8).reshape(num_blocks, 64, 4)
    packed_bytes = (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    ).astype(np.uint8)

    scales_bytes = block_scales.astype(np.float16).reshape(num_blocks, 1).view(np.uint8).reshape(num_blocks, 2)
    raw = np.concatenate([packed_bytes, scales_bytes], axis=1).reshape(-1)
    return raw


def _restore_vcore_tensor_to_fp32(
    state_dict: Dict[str, torch.Tensor],
    base_key: str,
) -> Tuple[np.ndarray, int, bool]:
    packed = state_dict[f"{base_key}.vcore_packed"]
    scales = state_dict[f"{base_key}.vcore_scales"].to(torch.float32)
    shape = tuple(map(int, state_dict[f"{base_key}.vcore_shape"].tolist()))
    g_size = int(state_dict[f"{base_key}.vcore_group"].item())
    bits = int(state_dict[f"{base_key}.vcore_bits"].item()) if f"{base_key}.vcore_bits" in state_dict else 2
    perm = state_dict.get(f"{base_key}.vcore_perm", None)

    total_elems = 1
    for d in shape:
        total_elems *= d
    padded_elems = ((total_elems + g_size - 1) // g_size) * g_size

    if bits == 2:
        grid = unpack_uint8_to_ternary(packed, (padded_elems,))
    else:
        grid = unpack_uint8_to_int4(packed, (padded_elems,))

    restored = (grid.reshape(-1, g_size) * scales.unsqueeze(1)).reshape(-1)[:total_elems].reshape(shape)
    has_perm = perm is not None
    if has_perm and restored.ndim == 2:
        inv_perm = torch.argsort(perm.to(torch.int64))
        restored = restored.index_select(1, inv_perm)

    return restored.cpu().numpy().astype(np.float32), bits, has_perm

def _load_hf_tokenizer_to_gguf(writer, model_dir: str, target_vocab_size: int = 0) -> None:
    tok_json = os.path.join(model_dir, "tokenizer.json")
    tok_cfg = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.isfile(tok_json):
        return

    with open(tok_json, "r", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj.get("model", {}).get("vocab", {})
    added = {item["content"]: item["id"] for item in tj.get("added_tokens", []) if "content" in item and "id" in item}
    full_vocab = {**vocab, **added}
    if not full_vocab:
        return

    max_id = max(max(full_vocab.values()), target_vocab_size - 1)
    tokens = [f"<|extra_{i}|>" for i in range(max_id + 1)]
    scores = [0.0] * (max_id + 1)
    token_types = [1] * (max_id + 1)

    added_ids = set(added.values())
    for tok_str, tid in full_vocab.items():
        if 0 <= tid <= max_id:
            tokens[tid] = tok_str
            token_types[tid] = 3 if tid in added_ids else 1

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    merges = tj.get("model", {}).get("merges", [])
    if merges:
        merges_str = [" ".join(m) if isinstance(m, list) else str(m) for m in merges]
        writer.add_token_merges(merges_str)

    if os.path.isfile(tok_cfg):
        with open(tok_cfg, "r", encoding="utf-8") as f:
            tc = json.load(f)
        eos_tok = tc.get("eos_token")
        bos_tok = tc.get("bos_token")
        pad_tok = tc.get("pad_token")
        if isinstance(eos_tok, dict):
            eos_tok = eos_tok.get("content")
        if isinstance(bos_tok, dict):
            bos_tok = bos_tok.get("content")
        if isinstance(pad_tok, dict):
            pad_tok = pad_tok.get("content")
        if eos_tok in full_vocab:
            writer.add_eos_token_id(int(full_vocab[eos_tok]))
        if bos_tok in full_vocab:
            writer.add_bos_token_id(int(full_vocab[bos_tok]))
        if pad_tok in full_vocab:
            writer.add_pad_token_id(int(full_vocab[pad_tok]))


def export_vcore_to_gguf(
    model_dir: str,
    output_gguf: str,
    quant_mode: str = "auto",
) -> Dict[str, int]:
    import gguf

    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"В папке {model_dir} не найден config.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "llama").lower()
    arch_map = {
        "qwen2": "qwen2",
        "llama": "llama",
        "mistral": "llama",
        "gemma": "gemma",
        "gemma2": "gemma2",
    }
    arch = arch_map.get(model_type, model_type)

    os.makedirs(os.path.dirname(os.path.abspath(output_gguf)), exist_ok=True)
    writer = gguf.GGUFWriter(output_gguf, arch)

    writer.add_name(os.path.basename(os.path.abspath(model_dir)))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
    writer.add_context_length(int(cfg.get("max_position_embeddings", 4096)))
    writer.add_embedding_length(int(cfg["hidden_size"]))
    writer.add_block_count(int(cfg["num_hidden_layers"]))
    writer.add_feed_forward_length(int(cfg["intermediate_size"]))
    writer.add_head_count(int(cfg["num_attention_heads"]))
    if "num_key_value_heads" in cfg:
        writer.add_head_count_kv(int(cfg["num_key_value_heads"]))
    if "rms_norm_eps" in cfg:
        writer.add_layer_norm_rms_eps(float(cfg["rms_norm_eps"]))
    if "rope_theta" in cfg:
        writer.add_rope_freq_base(float(cfg["rope_theta"]))

    _load_hf_tokenizer_to_gguf(writer, model_dir, target_vocab_size=int(cfg.get("vocab_size", 0)))

    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {model_dir} не найдены файлы .safetensors")

    merged_sd: Dict[str, torch.Tensor] = {}
    for sf in shard_files:
        merged_sd.update(load_file(sf))

    vcore_bases = sorted({k.split(".vcore_") [0] for k in merged_sd.keys() if ".vcore_" in k})
    plain_keys = sorted([k for k in merged_sd.keys() if ".vcore_" not in k])

    stats = {"tq2_0_tensors": 0, "q4_0_tensors": 0, "f16_tensors": 0}
    total = len(vcore_bases) + len(plain_keys)
    step = 0

    for base_k in vcore_bases:
        step += 1
        g_name = _map_hf_to_gguf_name(base_k, arch)
        if g_name is None:
            continue
        print(f"\r[V-CORE -> GGUF] [{step}/{total}] {g_name:<42}", end="", flush=True)

        w_fp32, bits, has_perm = _restore_vcore_tensor_to_fp32(merged_sd, base_k)
        g_size = int(merged_sd[f"{base_k}.vcore_group"].item())

        if (
            quant_mode in ("auto", "tq2_0")
            and bits == 2
            and not has_perm
            and g_size == 256
            and hasattr(gguf.GGMLQuantizationType, "TQ2_0")
        ):
            packed = merged_sd[f"{base_k}.vcore_packed"]
            scales = merged_sd[f"{base_k}.vcore_scales"].numpy().astype(np.float16)
            shape = tuple(map(int, merged_sd[f"{base_k}.vcore_shape"].tolist()))
            grid_np = unpack_uint8_to_ternary(packed, shape).numpy()
            tq2_raw = _pack_tq2_0_numpy(grid_np, scales, g_size)
            if tq2_raw is not None:
                writer.add_tensor(g_name, tq2_raw, raw_dtype=gguf.GGMLQuantizationType.TQ2_0)
                stats["tq2_0_tensors"] += 1
                continue

        if quant_mode != "f16" and w_fp32.ndim == 2 and (w_fp32.shape[1] % 32 == 0):
            q_data = gguf.quants.quantize(w_fp32, gguf.GGMLQuantizationType.Q4_0)
            writer.add_tensor(g_name, q_data, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
            stats["q4_0_tensors"] += 1
        else:
            writer.add_tensor(g_name, w_fp32.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
            stats["f16_tensors"] += 1

    for pk in plain_keys:
        step += 1
        g_name = _map_hf_to_gguf_name(pk, arch)
        if g_name is None:
            continue
        print(f"\r[V-CORE -> GGUF] [{step}/{total}] {g_name:<42}", end="", flush=True)
        arr = merged_sd[pk].to(torch.float32).cpu().numpy()
        if arr.ndim == 1:
            writer.add_tensor(g_name, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(g_name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        stats["f16_tensors"] += 1

    if "output.weight" not in [_map_hf_to_gguf_name(k, arch) for k in plain_keys] and "model.embed_tokens.weight" in merged_sd:
        emb = merged_sd["model.embed_tokens.weight"].to(torch.float16).cpu().numpy()
        writer.add_tensor("output.weight", emb, raw_dtype=gguf.GGMLQuantizationType.F16)
        stats["f16_tensors"] += 1

    print(flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Export V-CORE L1-Certified compressed model to standard GGUF")
    parser.add_argument("--input", "-i", required=True, help="Папка модели, сжатой через vcore-pack")
    parser.add_argument("--output", "-o", required=True, help="Путь к выходному .gguf файлу")
    parser.add_argument(
        "--mode",
        choices=["auto", "q4_0", "f16"],
        default="auto",
        help="Режим упаковки весов GGUF (auto = TQ2_0/Q4_0 + F16)",
    )
    args = parser.parse_args()

    stats = export_vcore_to_gguf(args.input, args.output, quant_mode=args.mode)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print("=== V-CORE GGUF EXPORT REPORT ===")
    print(f"  output_file   : {args.output} ({size_mb:.2f} MB)")
    print(f"  tq2_0_tensors : {stats['tq2_0_tensors']}")
    print(f"  q4_0_tensors  : {stats['q4_0_tensors']}")
    print(f"  f16_tensors   : {stats['f16_tensors']}")


if __name__ == "__main__":
    main()
    
