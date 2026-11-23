/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE Hybrid CUDA Kernels:
 * 1. vcore_optimize: Resonance filter with Markov static suppression.
 * 2. vcore_fused_ternary_gemv: Direct matrix-vector inference over 2-bit packed
 *    uint8 ternary weights {-1, 0, +1} without unpacking to float32 in VRAM.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define EPS 1e-6f
#define DEFAULT_ZETA 1.024f

// 1. Ядро резонансной оптимизации с Фильтром Маркова
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
        if (fabsf(sum) < EPS) {
            sum = 0.0f;
        }
        V[row] = sum * zeta;
    }
}

// 2. Прямой Fused CUDA-инференс: Y = (W_packed_2bit * scales) * X
// Читает 4 тернарных веса из 1 байта uint8 прямо в регистрах GPU без распаковки!
extern "C" __global__
void vcore_fused_ternary_gemv(const uint8_t* __restrict__ W_packed,
                              const float* __restrict__ block_scales,
                              const float* __restrict__ X,
                              float* __restrict__ Y,
                              int out_features,
                              int in_features,
                              int group_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) return;

    float total_acc = 0.0f;
    int blocks_per_row = in_features / group_size;
    int bytes_per_row = in_features / 4;
    int base_byte_offset = row * bytes_per_row;
    int base_scale_offset = row * blocks_per_row;

    for (int b = 0; b < blocks_per_row; ++b) {
        float block_acc = 0.0f;
        int col_start = b * group_size;
        int byte_start = base_byte_offset + (col_start / 4);
        int num_bytes = group_size / 4;

        for (int j = 0; j < num_bytes; ++j) {
            uint8_t packed = W_packed[byte_start + j];
            int c = col_start + j * 4;

            // Декодирование 4 весов {0,1,2} -> {-1,0,+1} в регистрах
            float s0 = (float)((int)(packed & 0x03) - 1);
            float s1 = (float)((int)((packed >> 2) & 0x03) - 1);
            float s2 = (float)((int)((packed >> 4) & 0x03) - 1);
            float s3 = (float)((int)((packed >> 6) & 0x03) - 1);

            block_acc += s0 * X[c] + s1 * X[c + 1] + s2 * X[c + 2] + s3 * X[c + 3];
        }
        total_acc += block_acc * block_scales[base_scale_offset + b];
    }

    Y[row] = total_acc;
}

extern "C" void run_vcore_resonance(const float* d_X, const float* d_Q, float* d_V, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vcore_optimize<<<gridSize, blockSize>>>(d_X, d_Q, d_V, n, DEFAULT_ZETA);
}

extern "C" void run_vcore_fused_gemv(const uint8_t* d_W_packed,
                                     const float* d_scales,
                                     const float* d_X,
                                     float* d_Y,
                                     int out_features,
                                     int in_features,
                                     int group_size) {
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;
    vcore_fused_ternary_gemv<<<gridSize, blockSize>>>(
        d_W_packed, d_scales, d_X, d_Y, out_features, in_features, group_size
    );
}
