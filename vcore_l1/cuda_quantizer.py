# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import subprocess
import torch
import torch.nn as nn


class VCoreTernaryQuantizer(nn.Module):
    def __init__(self, group_size: int = 32, theta: float = 1e-4, auto_theta: bool = True):
        super().__init__()
        self.group_size = group_size
        self.theta = theta
        self.auto_theta = auto_theta

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor):
        orig_shape = weights.shape
        if weights.numel() % self.group_size != 0:
            raise ValueError(f"Число элементов ({weights.numel()}) должно делиться на group_size={self.group_size}")

        blocks = weights.reshape(-1, self.group_size)
        num_blocks = blocks.shape[0]
        abs_blocks = torch.abs(blocks)
        base_med = torch.median(abs_blocks, dim=1, keepdim=True).values

        if not self.auto_theta:
            pruned = torch.where(abs_blocks > self.theta, blocks, torch.zeros_like(blocks))
            mask = torch.sign(pruned)
            scales = torch.median(torch.abs(pruned), dim=1, keepdim=True).values
            w_quant = (mask * scales).reshape(orig_shape)
            return w_quant, mask.reshape(orig_shape), scales.reshape(-1)

        candidates = [
            torch.full_like(base_med, self.theta),
            0.25 * base_med,
            0.50 * base_med,
        ]

        best_err = torch.full((num_blocks,), float("inf"), device=blocks.device, dtype=blocks.dtype)
        best_mask = torch.zeros_like(blocks)
        best_scales = torch.zeros_like(base_med)

        for th in candidates:
            active = abs_blocks > th
            pruned = torch.where(active, blocks, torch.zeros_like(blocks))
            mask = torch.sign(pruned)

            masked_abs = torch.where(active, abs_blocks, torch.tensor(float("nan"), device=blocks.device, dtype=blocks.dtype))
            scales = torch.nanmedian(masked_abs, dim=1, keepdim=True).values
            scales = torch.nan_to_num(scales, nan=0.0)

            w_cand = mask * scales
            err = torch.sum(torch.abs(blocks - w_cand), dim=1)

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
    
