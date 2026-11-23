# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import subprocess
import numpy as np
import torch
import torch.nn as nn

try:
    import cupy as cp
    from torch.utils.dlpack import to_dlpack
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


SELECTOR_CUDA_CODE = r'''
extern "C" __global__
void vcore_selector_kernel(const float* __restrict__ X,
                           float* __restrict__ Y,
                           int rows,
                           int cols,
                           float theta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float val = X[idx];
    Y[idx] = (fabsf(val) > theta) ? val : 0.0f;
}
'''


class VCoreTernaryQuantizer(nn.Module):
    """
    Гибридный 1.58-bit квантователь {-1, 0, +1}:
    1. Пороговое отсечение шума (LinearSelector CUDA kernel, |w| <= theta -> 0).
    2. Поблочное Δ = 1 медианное масштабирование (alpha_b* = median(|w_i|))
       по верифицированной модели MarkovL1Quantizer.lean.
    """
    def __init__(self, group_size: int = 32, theta: float = 1e-4):
        super().__init__()
        self.group_size = group_size
        self.theta = theta

    def _apply_selector(self, w: torch.Tensor) -> torch.Tensor:
        if HAS_CUPY and w.is_cuda:
            orig_shape, orig_dtype = w.shape, w.dtype
            w_f32 = w.view(-1, orig_shape[-1]).contiguous().to(torch.float32)
            rows, cols = w_f32.shape
            y_f32 = torch.empty_like(w_f32)

            kernel = cp.RawKernel(SELECTOR_CUDA_CODE, "vcore_selector_kernel")
            cp_x = cp.from_dlpack(to_dlpack(w_f32))
            cp_y = cp.from_dlpack(to_dlpack(y_f32))

            block = (16, 16)
            grid = ((cols + 15) // 16, (rows + 15) // 16)
            kernel(grid, block, (cp_x, cp_y, rows, cols, float(self.theta)))
            cp.cuda.Device().synchronize()
            return y_f32.to(orig_dtype).view(orig_shape)

        return torch.where(torch.abs(w) > self.theta, w, torch.zeros_like(w))

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor):
        orig_shape = weights.shape
        if weights.numel() % self.group_size != 0:
            raise ValueError(f"Число элементов ({weights.numel()}) должно делиться на group_size={self.group_size}")

        # Шаг 1: Линейный селектор обнуляет шум (|w| <= theta -> 0)
        pruned = self._apply_selector(weights)

        # Шаг 2: Разбиение на непересекающиеся блоки (режим Δ = 1)
        blocks = pruned.reshape(-1, self.group_size)
        ternary_mask = torch.sign(blocks)  # {-1.0, 0.0, +1.0}

        # Оптимальный L1-масштаб по медиане модулей ненулевых весов блока
        abs_blocks = torch.abs(blocks)
        scales = torch.median(abs_blocks, dim=1, keepdim=True).values

        w_quant = (ternary_mask * scales).reshape(orig_shape)
        return w_quant, ternary_mask.reshape(orig_shape), scales.reshape(-1)


def verify_with_lean4_engine() -> str:
    """Вызывает скомпилированный сертификатор Lean 4 из связанного репозитория."""
    res = subprocess.run(
        ["lake", "exe", "vcore_certifier", "--verify-bench"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout
