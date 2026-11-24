/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE High-Throughput CUDA Kernels v4:
 * - Warp-parallel (32 threads/row) 32-bit uint32 vectorized GEMV with __shfl_down_sync
 * - Shared-memory tiled uint32 vectorized GEMM for long-context prefill (batch > 1)
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define EPS 1e-6f
#define DEFAULT_ZETA 1.024f
#define WARP_SIZE 32
#define TILE_ROWS 16
#define TILE_TOKENS 16

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

// Warp-параллельное GEMV-ядро: 1 варп (32 потока) на 1 строку матрицы весов (batch = 1)
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

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        lane_acc += __shfl_down_sync(0xffffffff, lane_acc, offset);
    }

    if (lane_id == 0) {
        Y[warp_id] = lane_acc;
    }
}
// Тайловое пакетное GEMM-ядро для длинных промптов (prefill, batch > 1):
// кооперативно загружает упакованные 32-битные слова весов в __shared__ память
// и переиспользует их сразу для TILE_TOKENS = 16 токенов без повторных чтений из VRAM.
extern "C" __global__
void vcore_fused_ternary_gemm_tiled(const uint8_t* __restrict__ W_packed,
                                    const float* __restrict__ block_scales,
                                    const float* __restrict__ X,
                                    float* __restrict__ Y,
                                    int batch_tokens,
                                    int out_features,
                                    int in_features,
                                    int group_size) {
    __shared__ uint32_t s_W_words[TILE_ROWS][16];
    __shared__ float s_scales[TILE_ROWS];

    int local_row = threadIdx.x;
    int local_tok = threadIdx.y;
    int row = blockIdx.x * TILE_ROWS + local_row;
    int token = blockIdx.y * TILE_TOKENS + local_tok;

    int blocks_per_row = in_features / group_size;
    int bytes_per_row = in_features / 4;
    int words_per_block = group_size / 16;

    float total_acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        if (local_tok == 0 && row < out_features) {
            s_scales[local_row] = block_scales[row * blocks_per_row + b];
        }
        if (row < out_features && local_tok < words_per_block && words_per_block <= 16) {
            int byte_offset = row * bytes_per_row + (b * group_size) / 4;
            const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(W_packed + byte_offset);
            s_W_words[local_row][local_tok] = w_ptr[local_tok];
        }
        __syncthreads();

        if (row < out_features && token < batch_tokens) {
            const float* x_row = X + token * in_features + b * group_size;
            float block_acc = 0.0f;

            if (words_per_block > 0 && words_per_block <= 16) {
                for (int w = 0; w < words_per_block; ++w) {
                    uint32_t packed32 = s_W_words[local_row][w];
                    int c = w * 16;
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        int code = (packed32 >> (k * 2)) & 0x03;
                        block_acc += (float)(code - 1) * x_row[c + k];
                    }
                }
            } else {
                int byte_offset = row * bytes_per_row + (b * group_size) / 4;
                int num_bytes = group_size / 4;
                for (int j = 0; j < num_bytes; ++j) {
                    uint8_t packed = W_packed[byte_offset + j];
                    int c = j * 4;
                    block_acc += ((float)((int)(packed & 0x03) - 1)) * x_row[c]
                               + ((float)((int)((packed >> 2) & 0x03) - 1)) * x_row[c + 1]
                               + ((float)((int)((packed >> 4) & 0x03) - 1)) * x_row[c + 2]
                               + ((float)((int)((packed >> 6) & 0x03) - 1)) * x_row[c + 3];
                }
            }
            total_acc += block_acc * s_scales[local_row];
        }
        __syncthreads();
    }

    if (row < out_features && token < batch_tokens) {
        Y[token * out_features + row] = total_acc;
    }
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
        int threadsPerBlock = 256;
        int totalThreads = out_features * WARP_SIZE;
        int gridSize = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        vcore_fused_ternary_gemv_warp<<<gridSize, threadsPerBlock>>>(
            d_W_packed, d_scales, d_X, d_Y, out_features, in_features, group_size
        );
    } else {
        dim3 block(TILE_ROWS, TILE_TOKENS);
        dim3 grid((out_features + TILE_ROWS - 1) / TILE_ROWS,
                  (batch_tokens + TILE_TOKENS - 1) / TILE_TOKENS);
        vcore_fused_ternary_gemm_tiled<<<grid, block>>>(
            d_W_packed, d_scales, d_X, d_Y, batch_tokens, out_features, in_features, group_size
        );
    }
}
