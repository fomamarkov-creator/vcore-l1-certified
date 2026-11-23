# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import os
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
    flat = (ternary_mask.reshape(-1).to(torch.int8) + 1).to(torch.uint8)  # {0, 1, 2}
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


def repack_safetensors_layerwise(
    input_path: str,
    output_path: str,
    group_size: int = 32,
    theta: float = 1e-4,
) -> Dict[str, float]:
    """
    Послойная переупаковка всей модели из .safetensors в сжатый 1.58-bit формат V-CORE.
    Сохраняет packed-байты (uint8) и оптимальные Δ=1 медианные масштабы (float16).
    """
    state_dict = load_file(input_path)
    out_tensors: Dict[str, torch.Tensor] = {}
    quantizer = VCoreTernaryQuantizer(group_size=group_size, theta=theta)

    skip_keywords = ("norm", "bias", "embed", "lm_head")
    quantized_layers = 0
    total_l1_orig = 0.0
    total_l1_err = 0.0

    for name, tensor in state_dict.items():
        is_linear_weight = (
            tensor.ndim == 2
            and tensor.numel() % group_size == 0
            and not any(kw in name.lower() for kw in skip_keywords)
        )

        if is_linear_weight:
            w_f32 = tensor.to(torch.float32)
            w_quant, mask, scales = quantizer.quantize(w_f32)

            total_l1_orig += torch.sum(torch.abs(w_f32)).item()
            total_l1_err += torch.sum(torch.abs(w_f32 - w_quant)).item()

            packed_bytes, orig_shape = pack_ternary_to_uint8(mask)
            out_tensors[f"{name}.vcore_packed"] = packed_bytes.cpu()
            out_tensors[f"{name}.vcore_scales"] = scales.to(torch.float16).cpu()
            out_tensors[f"{name}.vcore_shape"] = torch.tensor(orig_shape, dtype=torch.int64)
            quantized_layers += 1
        else:
            # Служебные слои оставляем в исходном формате
            out_tensors[name] = tensor.cpu()

    save_file(out_tensors, output_path)

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "quantized_layers": quantized_layers,
        "orig_size_mb": orig_mb,
        "packed_size_mb": new_mb,
        "compression_ratio": orig_mb / max(new_mb, 1e-6),
        "total_l1_orig": total_l1_orig,
        "total_l1_err": total_l1_err,
    }
    
