/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE Native CPU Engine (AVX2 SIMD):
 * Direct 1.58-bit {-1, 0, +1} matrix-vector & batched matrix multiplication
 * without unpacking weights to float32 in RAM.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float dot_ternary_block_scalar(const uint8_t* __restrict__ p,
                                             const float* __restrict__ x,
                                             int g) {
    int nb = g / 4;
    float sum = 0.0f;
    for (int j = 0; j < nb; ++j) {
        uint8_t v = p[j];
        int c = j * 4;
        sum += ((float)((int)(v & 0x03) - 1)) * x[c]
             + ((float)((int)((v >> 2) & 0x03) - 1)) * x[c + 1]
             + ((float)((int)((v >> 4) & 0x03) - 1)) * x[c + 2]
             + ((float)((int)((v >> 6) & 0x03) - 1)) * x[c + 3];
    }
    return sum;
}

extern "C" void vcore_cpu_fused_gemm(const uint8_t* __restrict__ W_packed,
                                     const float* __restrict__ block_scales,
                                     const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     int batch_tokens,
                                     int out_features,
                                     int in_features,
                                     int group_size) {
    int num_blocks = in_features / group_size;
    int bytes_per_row = in_features / 4;
    int bytes_per_block = group_size / 4;

#if defined(__AVX2__)
    if (group_size % 32 == 0) {
        std::vector<int8_t> xq(in_features);
        std::vector<float> xs(num_blocks);

        const __m256i shuf_idx = _mm256_setr_epi8(
            0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3,
            4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7
        );
        const __m256i mask3 = _mm256_set1_epi8(3);
        const __m256i one8 = _mm256_set1_epi8(1);
        const __m256i one16 = _mm256_set1_epi16(1);
        const __m256i blend02 = _mm256_set1_epi16((short)0xFF00);
        const __m256i blend46 = _mm256_set1_epi32((int)0xFFFF0000);

        for (int t = 0; t < batch_tokens; ++t) {
            const float* x_tok = X + (size_t)t * in_features;
            float* y_tok = Y + (size_t)t * out_features;

            for (int b = 0; b < num_blocks; ++b) {
                float mx = 1e-8f;
                const float* xb = x_tok + b * group_size;
                for (int k = 0; k < group_size; ++k) {
                    float a = fabsf(xb[k]);
                    if (a > mx) mx = a;
                }
                float inv = 127.0f / mx;
                xs[b] = mx / 127.0f;
                int8_t* qb = xq.data() + b * group_size;
                for (int k = 0; k < group_size; ++k) {
                    qb[k] = (int8_t)lrintf(xb[k] * inv);
                }
            }

            int chunks_per_block = group_size / 32;
            for (int r = 0; r < out_features; ++r) {
                const uint8_t* wr = W_packed + (size_t)r * bytes_per_row;
                const float* sr = block_scales + (size_t)r * num_blocks;
                float total = 0.0f;

                for (int b = 0; b < num_blocks; ++b) {
                    const uint8_t* wb = wr + b * bytes_per_block;
                    const int8_t* xb_q = xq.data() + b * group_size;
                    int32_t block_i32 = 0;

                    for (int ch = 0; ch < chunks_per_block; ++ch) {
                        uint64_t raw8 = *(const uint64_t*)(wb + ch * 8);
                        __m256i vraw = _mm256_set1_epi64x((long long)raw8);
                        __m256i bytes = _mm256_shuffle_epi8(vraw, shuf_idx);

                        __m256i lo = _mm256_and_si256(bytes, mask3);
                        __m256i hi2 = _mm256_and_si256(_mm256_srli_epi16(bytes, 2), mask3);
                        __m256i hi4 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask3);
                        __m256i hi6 = _mm256_and_si256(_mm256_srli_epi16(bytes, 6), mask3);

                        __m256i b02 = _mm256_blendv_epi8(lo, hi2, blend02);
                        __m256i b46 = _mm256_blendv_epi8(hi4, hi6, blend02);
                        __m256i vw = _mm256_sub_epi8(_mm256_blendv_epi8(b02, b46, blend46), one8);

                        __m256i va = _mm256_loadu_si256((const __m256i*)(xb_q + ch * 32));
                        __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(vw), _mm256_sign_epi8(va, vw));
                        __m256i s32 = _mm256_madd_epi16(p16, one16);

                        __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
                        s128 = _mm_hadd_epi32(s128, s128);
                        s128 = _mm_hadd_epi32(s128, s128);
                        block_i32 += _mm_cvtsi128_si32(s128);
                    }
                    total += (float)block_i32 * (sr[b] * xs[b]);
                }
                y_tok[r] = total;
            }
        }
        return;
    }
#endif

    for (int t = 0; t < batch_tokens; ++t) {
        const float* x_tok = X + (size_t)t * in_features;
        float* y_tok = Y + (size_t)t * out_features;
        for (int r = 0; r < out_features; ++r) {
            const uint8_t* wr = W_packed + (size_t)r * bytes_per_row;
            const float* sr = block_scales + (size_t)r * num_blocks;
            float total = 0.0f;
            for (int b = 0; b < num_blocks; ++b) {
                total += dot_ternary_block_scalar(wr + b * bytes_per_block, x_tok + b * group_size, group_size) * sr[b];
            }
            y_tok[r] = total;
        }
    }
}
