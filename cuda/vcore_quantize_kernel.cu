/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE Hybrid CUDA Kernel:
 * Combines Markov's hard-threshold static filter (|x| <= theta -> 0)
 * with decoupled blockwise L1 median scaling (Delta = 1 regime).
 */

#include <cuda_runtime.h>
#include <math.h>

#define EPS 1e-6f

// 1. Оригинальный резонансный оптимизатор с фильтром Маркова
extern "C" __global__
void vcore_optimize(const float* __restrict__ X,
                    const float* __restrict__ Q,
                    float* __restrict__ V,
                    int n,
                    float zeta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;

        #pragma unroll 4
        for (int i = 0; i < n; ++i) {
            sum += Q[row * n + i] * X[i];
        }

        // Фильтр Маркова (устранение статики)
        if (fabsf(sum) < EPS) {
            sum = 0.0f;
        }

        V[row] = sum * zeta;
    }
}

// 2. Ядро деквантования и применения тернарной маски {-1, 0, +1} с масштабом блока alpha_b*
extern "C" __global__
void vcore_ternary_block_apply(const float* __restrict__ W,
                               const float* __restrict__ block_scales,
                               float* __restrict__ W_quant,
                               int total_elements,
                               int group_size,
                               float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = W[idx];
    // Фильтр Маркова: обнуление подпорогового шума
    if (fabsf(val) <= theta) {
        W_quant[idx] = 0.0f;
        return;
    }

    int block_id = idx / group_size;
    float alpha = block_scales[block_id];
    float sign = (val > 0.0f) ? 1.0f : -1.0f;
    W_quant[idx] = sign * alpha;
}
