# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import glob
import json
import os
import shutil
from typing import Dict, Tuple
import torch
from safetensors.torch import load_file, save_file
from .cuda_quantizer import VCoreTernaryQuantizer


def pack_ternary_to_uint8(ternary_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Физическая 2-битная упаковка тернарных весов {-1, 0, +1} в байты uint8
    (4 веса в 1 байт: -1 -> 0, 0 -> 1, +1 -> 2).
    """
    orig_shape = tuple(ternary_mask.shape)
    flat = (ternary_mask.reshape(-1).to(torch.int8) + 1).to(torch.uint8)
    pad_len = (4 - (flat.numel() % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.ones(pad_len, dtype=torch.uint8, device=flat.device)])

    grouped = flat.view(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    )
    return packed.contiguous(), orig_shape


def unpack_uint8_to_ternary(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Мгновенная распаковка байтов uint8 обратно в тернарную маску {-1, 0, +1}."""
    total_elems = 1
    for d in orig_shape:
        total_elems *= d

    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03

    unpacked = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1)[:total_elems]
    return unpacked.to(torch.float32) - 1.0


def _pad_to_group(flat: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    orig_numel = flat.numel()
    rem = orig_numel % group_size
    if rem == 0:
        return flat, orig_numel
    pad = group_size - rem
    padded = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    return padded, orig_numel


def repack_safetensors_layerwise(
    input_path: str,
    output_path: str,
    group_size: int = 32,
    theta: float = 1e-4,
    max_rel_error: float = 0.40,
    include_aux_layers: bool = True,
    auto_theta: bool = True,
) -> Dict[str, float]:
    """
    Адаптивная послойная переупаковка одного .safetensors файла в 1.58-bit формат V-CORE
    с автоподбором порога theta_b* на каждом блоке и автоматическим откатом (rollback)
    в FP16 при превышении допустимой относительной L1-ошибки max_rel_error.
    """
    state_dict = load_file(input_path)
    out_tensors: Dict[str, torch.Tensor] = {}

    quantized_layers = 0
    rolled_back_layers = 0
    total_l1_orig = 0.0
    total_l1_err = 0.0

    for name, tensor in state_dict.items():
        if not tensor.is_floating_point() or tensor.numel() == 0:
            out_tensors[name] = tensor.cpu()
            continue

        if not include_aux_layers and tensor.ndim < 2:
            out_tensors[name] = tensor.cpu()
            continue

        w_f32 = tensor.to(torch.float32)
        orig_l1 = torch.sum(torch.abs(w_f32)).item()

        eff_group = min(group_size, w_f32.numel())
        padded_flat, orig_numel = _pad_to_group(w_f32.reshape(-1), eff_group)

        local_q = VCoreTernaryQuantizer(group_size=eff_group, theta=theta, auto_theta=auto_theta)
        w_quant_pad, mask_pad, scales = local_q.quantize(padded_flat)

        w_quant = w_quant_pad[:orig_numel].reshape(tensor.shape)
        mask = mask_pad[:orig_numel].reshape(tensor.shape)

        layer_err = torch.sum(torch.abs(w_f32 - w_quant)).item()
        rel_err = layer_err / (orig_l1 + 1e-8)

        total_l1_orig += orig_l1

        if rel_err <= max_rel_error:
            packed_bytes, orig_shape = pack_ternary_to_uint8(mask)
            out_tensors[f"{name}.vcore_packed"] = packed_bytes.cpu()
            out_tensors[f"{name}.vcore_scales"] = scales.to(torch.float16).cpu()
            out_tensors[f"{name}.vcore_shape"] = torch.tensor(orig_shape, dtype=torch.int64)
            out_tensors[f"{name}.vcore_group"] = torch.tensor([eff_group], dtype=torch.int64)
            quantized_layers += 1
            total_l1_err += layer_err
        else:
            out_tensors[name] = tensor.cpu()
            rolled_back_layers += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out_tensors, output_path)

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "quantized_layers": quantized_layers,
        "rolled_back_layers": rolled_back_layers,
        "orig_size_mb": orig_mb,
        "packed_size_mb": new_mb,
        "compression_ratio": orig_mb / max(new_mb, 1e-6),
        "total_l1_orig": total_l1_orig,
        "total_l1_err": total_l1_err,
    }


def repack_sharded_model(
    input_dir: str,
    output_dir: str,
    group_size: int = 32,
    theta: float = 1e-4,
    max_rel_error: float = 0.40,
    include_aux_layers: bool = True,
    auto_theta: bool = True,
) -> Dict[str, float]:
    """
    Послойная переупаковка целой папки модели Hugging Face (включая шардированные
    чекпоинты model-00001-of-0000N.safetensors + обновление model.safetensors.index.json
    и копирование tokenizer/config.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {input_dir} не найдены файлы .safetensors")

    agg = {
        "shards_processed": len(shard_files),
        "quantized_layers": 0,
        "rolled_back_layers": 0,
        "orig_size_mb": 0.0,
        "packed_size_mb": 0.0,
        "total_l1_orig": 0.0,
        "total_l1_err": 0.0,
    }

    new_weight_map: Dict[str, str] = {}

    for shard_path in shard_files:
        fname = os.path.basename(shard_path)
        out_shard = os.path.join(output_dir, fname)
        st = repack_safetensors_layerwise(
            shard_path,
            out_shard,
            group_size=group_size,
            theta=theta,
            max_rel_error=max_rel_error,
            include_aux_layers=include_aux_layers,
            auto_theta=auto_theta,
        )
        agg["quantized_layers"] += st["quantized_layers"]
        agg["rolled_back_layers"] += st["rolled_back_layers"]
        agg["orig_size_mb"] += st["orig_size_mb"]
        agg["packed_size_mb"] += st["packed_size_mb"]
        agg["total_l1_orig"] += st["total_l1_orig"]
        agg["total_l1_err"] += st["total_l1_err"]

        # Обновляем карту весов для индексного файла Hugging Face
        saved_keys = load_file(out_shard).keys()
        for k in saved_keys:
            new_weight_map[k] = fname

    # Копируем конфигурационные файлы и токенизатор, обновляем index.json
    for item in os.listdir(input_dir):
        src_item = os.path.join(input_dir, item)
        dst_item = os.path.join(output_dir, item)
        if item == "model.safetensors.index.json":
            with open(src_item, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            idx_data["weight_map"] = new_weight_map
            idx_data.setdefault("metadata", {})["vcore_quantized"] = "1.58-bit-delta1"
            with open(dst_item, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)
        elif not item.endswith(".safetensors") and os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)

    agg["compression_ratio"] = agg["orig_size_mb"] / max(agg["packed_size_mb"], 1e-6)
    return agg
    
