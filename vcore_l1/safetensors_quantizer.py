# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import glob
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
import torch
from safetensors.torch import load_file, save_file
from .cuda_quantizer import VCoreTernaryQuantizer


def pack_ternary_to_uint8(ternary_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """2-битная упаковка {-1, 0, +1} в байты uint8 (4 веса в 1 байт)."""
    orig_shape = tuple(ternary_mask.shape)
    flat = (ternary_mask.reshape(-1).to(torch.int8) + 1).to(torch.uint8)
    pad_len = (4 - (flat.numel() % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.ones(pad_len, dtype=torch.uint8, device=flat.device)])
    grouped = flat.view(-1, 4)
    packed = grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)
    return packed.contiguous(), orig_shape


def unpack_uint8_to_ternary(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Распаковка 2-битных байтов uint8 обратно в {-1, 0, +1}."""
    total_elems = 1
    for d in orig_shape:
        total_elems *= d
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    unpacked = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1)[:total_elems]
    return unpacked.to(torch.float32) - 1.0


def pack_int4_to_uint8(int4_vals: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """4-битная упаковка целых весов {-7..+7} в байты uint8 (2 веса в 1 байт)."""
    orig_shape = tuple(int4_vals.shape)
    flat = (torch.clamp(int4_vals.reshape(-1), -7, 7).to(torch.int8) + 7).to(torch.uint8)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.tensor([7], dtype=torch.uint8, device=flat.device)])
    grouped = flat.view(-1, 2)
    packed = (grouped[:, 0] & 0x0F) | ((grouped[:, 1] & 0x0F) << 4)
    return packed.contiguous(), orig_shape


def unpack_uint8_to_int4(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """Распаковка 4-битных байтов uint8 обратно в {-7..+7}."""
    total_elems = 1
    for d in orig_shape:
        total_elems *= d
    low = (packed & 0x0F).to(torch.float32) - 7.0
    high = ((packed >> 4) & 0x0F).to(torch.float32) - 7.0
    unpacked = torch.stack([low, high], dim=1).reshape(-1)[:total_elems]
    return unpacked


def _pad_to_group(flat: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    orig_numel = flat.numel()
    rem = orig_numel % group_size
    if rem == 0:
        return flat, orig_numel
    pad = group_size - rem
    return torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)]), orig_numel


def _quantize_int4_blockwise(w_f32: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Поблочное 4-битное квантование {-7..+7} с медианной калибровкой Δ=1."""
    flat = w_f32.reshape(-1)
    padded, orig_numel = _pad_to_group(flat, group_size)
    blocks = padded.view(-1, group_size)
    abs_b = torch.abs(blocks)
    med_b = torch.median(abs_b, dim=1, keepdim=True).values
    max_b = torch.max(abs_b, dim=1, keepdim=True).values
    # Шаг сетки согласован с Δ=1 медианой и защищён от переполнения [-7, 7]
    step = torch.maximum(med_b / 2.5, max_b / 7.0)
    step = torch.clamp(step, min=1e-8)

    q_int = torch.clamp(torch.round(blocks / step), -7.0, 7.0)
    w_hat = (q_int * step).reshape(-1)[:orig_numel].reshape(w_f32.shape)
    err = torch.sum(torch.abs(w_f32 - w_hat)).item()
    return q_int.reshape(-1)[:orig_numel].reshape(w_f32.shape), step.reshape(-1), err


def _quantize_tensor_cascade(
    w_f32: torch.Tensor,
    cascade_groups: List[int],
    theta: float,
    max_rel_error: float,
    auto_theta: bool,
    allow_int4: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int, float]]:
    """
    Каскад точности: 1.58-bit (64 -> 32 -> 16) -> 4-bit INT4 (32) -> откат в FP16.
    Возвращает (quant_grid, scales, chosen_group_size, bits_per_weight, l1_err).
    """
    orig_l1 = torch.sum(torch.abs(w_f32)).item()
    flat = w_f32.reshape(-1)

    for g in cascade_groups:
        eff_g = min(g, flat.numel())
        if eff_g <= 0:
            continue
        padded_flat, orig_numel = _pad_to_group(flat, eff_g)
        q = VCoreTernaryQuantizer(group_size=eff_g, theta=theta, auto_theta=auto_theta)
        w_q_pad, mask_pad, scales = q.quantize(padded_flat)

        w_q = w_q_pad[:orig_numel].reshape(w_f32.shape)
        mask = mask_pad[:orig_numel].reshape(w_f32.shape)
        err = torch.sum(torch.abs(w_f32 - w_q)).item()
        if (err / (orig_l1 + 1e-8)) <= max_rel_error:
            return mask, scales, eff_g, 2, err

    # Промежуточный уровень каскада: 4-битный INT4 блок перед откатом в FP16
    if allow_int4 and flat.numel() >= 16:
        eff_g4 = min(32, flat.numel())
        q_int4, scales4, err4 = _quantize_int4_blockwise(w_f32, group_size=eff_g4)
        if (err4 / (orig_l1 + 1e-8)) <= max_rel_error:
            return q_int4, scales4, eff_g4, 4, err4

    return None
    
