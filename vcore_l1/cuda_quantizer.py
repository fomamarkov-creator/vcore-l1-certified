# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import subprocess
from typing import Optional
import torch
import torch.nn as nn


def _weighted_median_1d(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Точная поблочная взвешенная медиана (реализация exactWeightedMedianQ
    из модуля MarkovL1WeightedTernary.lean):
    values: (num_blocks, g), weights: (num_blocks, g) >= 0.
    """
    sorted_vals, idx = torch.sort(values, dim=1)
    sorted_w = torch.gather(weights, 1, idx)
    cum_w = torch.cumsum(sorted_w, dim=1)
    total_w = cum_w.narrow(1, cum_w.size(1) - 1, 1)
    half_w = 0.5 * total_w

    mask = cum_w >= half_w
    # Первый индекс, где накопленный вес >= половины суммарного веса активных элементов
    first_idx = torch.argmax(mask.to(torch.int64), dim=1, keepdim=True)
    med = torch.gather(sorted_vals, 1, first_idx)
    return torch.where(total_w > 0, med, torch.zeros_like(med))


class VCoreTernaryQuantizer(nn.Module):
    """
    Гибридный 1.58-bit квантователь {-1, 0, +1} с поддержкой:
    1. Точного перебора критических порогов {0} U {|w_i|} (MarkovL1TernaryQuantizer.lean).
    2. Активационно-взвешенной Δ=1 медианы (MarkovL1WeightedTernary.lean) при передаче
       важности каналов activation_scales (E[|x_j|]).
    """
    def __init__(self, group_size: int = 32, theta: float = 1e-4, auto_theta: bool = True):
        super().__init__()
        self.group_size = group_size
        self.theta = theta
        self.auto_theta = auto_theta

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor, activation_scales: Optional[torch.Tensor] = None):
        orig_shape = weights.shape
        if weights.numel() % self.group_size != 0:
            raise ValueError(f"Число элементов ({weights.numel()}) должно делиться на group_size={self.group_size}")

        blocks = weights.reshape(-1, self.group_size)
        num_blocks = blocks.size(0)
        abs_blocks = torch.abs(blocks)

        if activation_scales is not None:
            if activation_scales.numel() == weights.numel():
                imp_blocks = torch.abs(activation_scales).reshape(-1, self.group_size).to(blocks.device, dtype=blocks.dtype)
            elif weights.ndim == 2 and activation_scales.numel() == weights.size(1):
                imp_blocks = torch.abs(activation_scales).view(1, -1).expand_as(weights).reshape(-1, self.group_size).to(blocks.device, dtype=blocks.dtype)
            else:
                imp_blocks = torch.ones_like(blocks)
        else:
            imp_blocks = torch.ones_like(blocks)

        if not self.auto_theta:
            pruned = torch.where(abs_blocks > self.theta, blocks, torch.zeros_like(blocks))
            mask = torch.sign(pruned)
            active_w = torch.where(abs_blocks > self.theta, imp_blocks, torch.zeros_like(imp_blocks))
            scales = _weighted_median_1d(abs_blocks, active_w)
            w_quant = (mask * scales).reshape(orig_shape)
            return w_quant, mask.reshape(orig_shape), scales.reshape(-1)

        sorted_abs, _ = torch.sort(abs_blocks, dim=1)
        best_err = torch.full((num_blocks,), float("inf"), device=blocks.device, dtype=blocks.dtype)
        best_mask = torch.zeros_like(blocks)
        best_scales = torch.zeros((num_blocks, 1), device=blocks.device, dtype=blocks.dtype)

        for k in range(self.group_size):
            if k == 0:
                th = torch.zeros((num_blocks, 1), device=blocks.device, dtype=blocks.dtype)
            else:
                th = sorted_abs.narrow(1, k - 1, 1)

            active = abs_blocks > th
            mask = torch.where(active, torch.sign(blocks), torch.zeros_like(blocks))
            active_imp = torch.where(active, imp_blocks, torch.zeros_like(imp_blocks))
            scales = _weighted_median_1d(abs_blocks, active_imp)

            w_cand = mask * scales
            err = torch.sum(imp_blocks * torch.abs(blocks - w_cand), dim=1)

            improve = err < best_err
            best_err = torch.where(improve, err, best_err)
            best_mask = torch.where(improve.unsqueeze(1), mask, best_mask)
            best_scales = torch.where(improve.unsqueeze(1), scales, best_scales)

        w_quant = (best_mask * best_scales).reshape(orig_shape)
        return w_quant, best_mask.reshape(orig_shape), best_scales.reshape(-1)


def verify_with_lean4_engine() -> str:
    res = subprocess.run(
        ["lake", "exe", "vcore_certifier", "--verify-bench"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout
    
