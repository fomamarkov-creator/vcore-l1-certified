/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * CUDA Shared-Memory Sub-1-Bit E8 LUT Inference Kernel (~0.989 b/w, zero weight unpacking)
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

// Построение таблицы 241 точки решётки E8 (1 нулевой + 112 корневых + 128 полуцелых)
static std::vector<float> build_e8_cb_host() {
    std::vector<float> cb(241 * 8, 0.0f);
    int idx = 1;
    const float signs = {-1.0f, 1.0f};
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            for (float s1 : signs) {
                for (float s2 : signs) {
                    cb[idx * 8 + i] = s1;
                    cb[idx * 8 + j] = s2;
                    idx++;
                }
            }
        }
    }
    for (int mask = 0; mask < 256; ++mask) {
        int bits = 0;
        for (int b = 0; b < 8; ++b) bits += (mask >> b) & 1;
        if (bits % 2 == 0) {
            for (int k = 0; k < 8; ++k) {
                cb[idx * 8 + k] = ((mask >> k) & 1) ? -0.5f : 0.5f;
            }
            idx++;
        }
    }
    return cb;
}

__constant__ float d_e8_codebook[241 * 8];
static bool g_e8_cb_initialized = false;

static void ensure_e8_constant_cb() {
    if (!g_e8_cb_initialized) {
        std::vector<float> h_cb = build_e8_cb_host();
        cudaMemcpyToSymbol(d_e8_codebook, h_cb.data(), 241 * 8 * sizeof(float));
        g_e8_cb_initialized = true;
    }
}

// Этап 1: Построение LUT откликов (num_e8_blocks x 241) для каждого токена батча
__global__ void vcore_build_e8_lut_kernel(
    const float* __restrict__ X,
    float* __restrict__ LUT,
    int batch_tokens,
    int in_features
) {
    int tok = blockIdx.y;
    int eb = blockIdx.x;
    int code = threadIdx.x; // 0..240
    int num_e8_blocks = in_features / 8;

    if (tok >= batch_tokens || eb >= num_e8_blocks || code >= 241) return;

    if (code == 0) {
        LUT[(tok * num_e8_blocks + eb) * 241] = 0.0f;
        return;
    }

    const float* xb = X + (size_t)tok * in_features + eb * 8;
    const float* cw = d_e8_codebook + code * 8;
    float dot = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        dot += xb[k] * cw[k];
    }
    LUT[(tok * num_e8_blocks + eb) * 241 + code] = dot;
}

// Этап 2: Чистый табличный GEMM без умножений весов (O(1) выборка из LUT)
__global__ void vcore_e8_lut_gemm_kernel(
    const uint8_t* __restrict__ E8_indices,
    const float* __restrict__ block_scales,
    const float* __restrict__ LUT,
    float* __restrict__ Y,
    int batch_tokens,
    int out_features,
    int in_features,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (row >= out_features || tok >= batch_tokens) return;

    int num_e8_blocks = in_features / 8;
    int num_groups = in_features / group_size;
    int e8_per_group = group_size / 8;

    const uint8_t* idx_row = E8_indices + (size_t)row * num_e8_blocks;
    const float* s_row = block_scales + (size_t)row * num_groups;
    const float* lut_tok = LUT + (size_t)tok * num_e8_blocks * 241;

    float total = 0.0f;
    for (int g = 0; g < num_groups; ++g) {
        float grp_sum = 0.0f;
        int eb_start = g * e8_per_group;
        for (int k = 0; k < e8_per_group; ++k) {
            int eb = eb_start + k;
            uint8_t c = idx_row[eb];
            if (c >= 241) c = 0;
            grp_sum += lut_tok[eb * 241 + c];
        }
        total += grp_sum * s_row[g];
    }
    Y[(size_t)tok * out_features + row] = total;
}

extern "C" void run_vcore_e8_lut_gemm(
    const uint8_t* d_e8_indices,
    const float* d_scales,
    const float* d_X,
    float* d_Y,
    int batch_tokens,
    int out_features,
    int in_features,
    int group_size
) {
    ensure_e8_constant_cb();
    int num_e8_blocks = in_features / 8;
    float* d_lut = nullptr;
    cudaMalloc(&d_lut, (size_t)batch_tokens * num_e8_blocks * 241 * sizeof(float));

    dim3 grid_lut(num_e8_blocks, batch_tokens);
    vcore_build_e8_lut_kernel<<<grid_lut, 241>>>(d_X, d_lut, batch_tokens, in_features);

    int threads = 256;
    dim3 grid_gemm((out_features + threads - 1) / threads, batch_tokens);
    vcore_e8_lut_gemm_kernel<<<grid_gemm, threads>>>(
        d_e8_indices, d_scales, d_lut, d_Y, batch_tokens, out_features, in_features, group_size
    );
    cudaFree(d_lut);
}
