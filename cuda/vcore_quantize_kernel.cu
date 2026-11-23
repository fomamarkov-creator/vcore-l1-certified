/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE High-Throughput CUDA Kernels v3:
 * - Warp-parallel (32 threads/row) 32-bit uint32 vectorized GEMV with __shfl_down_sync
 * - Batched GEMM kernel for long-context prefill
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define EPS 1e-6f
#define DEFAULT_ZETA 1.024f
#define WARP_SIZE 32

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

// Warp-параллельное GEMV-ядро: 1 варп (32 потока) на 1 строку матрицы весов
extern "C" __global__
void vcore_fused_ternary_gemv_warp(const uint8_t* __restrict__ W_packed,
                                   const float* __restrict__ block_scales,
                                   const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int out_features,
                                   int in_features,
                                   int group_size) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    if (warp_id >= out_features) return;

    int blocks_per_row = in_features / group_size;
    int bytes_per_row = in_features / 4;
    int base_byte_offset = warp_id * bytes_per_row;
    int base_scale_offset = warp_id * blocks_per_row;

    float lane_acc = 0.0f;

    // Каждый поток варпа берёт свой блок b, b+32, b+64...
    for (int b = lane_id; b < blocks_per_row; b += WARP_SIZE) {
        float block_acc = 0.0f;
        int col_start = b * group_size;
        int byte_start = base_byte_offset + (col_start / 4);
        int num_bytes = group_size / 4;

        int num_words = num_bytes / 4;
        const uint32_t* word_ptr = reinterpret_cast<const uint32_t*>(W_packed + byte_start);
        for (int w = 0; w < num_words; ++w) {
            uint32_t packed32 = word_ptr[w];
            int c = col_start + w * 16;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                int code = (packed32 >> (k * 2)) & 0x03;
                float s = (float)(code - 1);
                block_acc += s * X[c + k];
            }
        }

        for (int j = num_words * 4; j < num_bytes; ++j) {
            uint8_t packed = W_packed[byte_start + j];
            int c = col_start + j * 4;
            block_acc += ((float)((int)(packed & 0x03) - 1)) * X[c]
                       + ((float)((int)((packed >> 2) & 0x03) - 1)) * X[c + 1]
                       + ((float)((int)((packed >> 4) & 0x03) - 1)) * X[c + 2]
                       + ((float)((int)((packed >> 6) & 0x03) - 1)) * X[c + 3];
        }

        lane_acc += block_acc * block_scales[base_scale_offset + b];
    }

    // Регистровая варп-редукция без использования shared memory
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        lane_acc += __shfl_down_sync(0xffffffff, lane_acc, offset);
    }

    if (lane_id == 0) {
        Y[warp_id] = lane_acc;
    }
}

// Пакетное GEMM-ядро для длинных промптов (prefill)
extern "C" __global__
void vcore_fused_ternary_gemm(const uint8_t* __restrict__ W_packed,
                              const float* __restrict__ block_scales,
                              const float* __restrict__ X,
                              float* __restrict__ Y,
                              int batch_tokens,
                              int out_features,
                              int in_features,
                              int group_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= out_features || token >= batch_tokens) return;

    const float* x_row = X + token * in_features;
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
            float s0 = (float)((int)(packed & 0x03) - 1);
            float s1 = (float)((int)((packed >> 2) & 0x03) - 1);
            float s2 = (float)((int)((packed >> 4) & 0x03) - 1);
            float s3 = (float)((int)((packed >> 6) & 0x03) - 1);
            block_acc += s0 * x_row[c] + s1 * x_row[c + 1] + s2 * x_row[c + 2] + s3 * x_row[c + 3];
        }
        total_acc += block_acc * block_scales[base_scale_offset + b];
    }

    Y[token * out_features + row] = total_acc;
}

extern "C" void run_vcore_resonance(const float* d_X, const float* d_Q, float* d_V, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vcore_optimize<<<gridSize, blockSize>>>(d_X, d_Q, d_V, n, DEFAULT_ZETA);
}

extern "C" void run_vcore_fused_gemm(const uint8_t* d_W_packed,
                                     const float* d_scales,
                                     const float* d_X,
                                     float* d_Y,
                                     int batch_tokens,
                                     int out_features,
                                     int in_features,
                                     int group_size) {
    if (batch_tokens == 1) {
        int threadsPerBlock = 256; // 8 варпов по 32 потока на блок
        int totalThreads = out_features * WARP_SIZE;
        int gridSize = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        vcore_fused_ternary_gemv_warp<<<gridSize, threadsPerBlock>>>(
            d_W_packed, d_scales, d_X, d_Y, out_features, in_features, group_size
        );
    } else {
        dim3 block(32, 8);
        dim3 grid((out_features + block.x - 1) / block.x, (batch_tokens + block.y - 1) / block.y);
        vcore_fused_ternary_gemm<<<grid, block>>>(
            d_W_packed, d_scales, d_X, d_Y, batch_tokens, out_features, in_features, group_size
        );
    }
}
