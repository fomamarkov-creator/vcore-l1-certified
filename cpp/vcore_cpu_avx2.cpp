/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * V-CORE Native CPU Engine (AVX2 SIMD + OpenMP):
 * Zero-multiplication 1.58-bit {-1, 0, +1} matrix-vector & matrix-matrix kernels
 * with native support for V-CORE block sizes (16, 32, 64) and INT4 cascade blocks.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

// Быстрое скалярное произведение одного 1.58-bit тернарного блока на вектор X
static inline float dot_ternary_block_cpu(const uint8_t* __restrict__ packed_block,
                                          const float* __restrict__ x_block,
                                          int group_size) {
    int num_bytes = group_size / 4;

#if defined(__AVX2__)
    // Векторная обработка по 16 весов (4 байта = 1 uint32_t) за итерацию AVX2
    __m256 acc_vec = _mm256_setzero_ps();
    const __m256i one_vec = _mm256_set1_epi32(1);
    const __m256i mask2_vec = _mm256_set1_epi32(0x03);
    const __m256i shifts_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i shifts_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);

    int num_words = num_bytes / 4;
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(packed_block);

    for (int w = 0; w < num_words; ++w) {
        uint32_t word = w_ptr[w];
        __m256i w_broadcast = _mm256_set1_epi32((int)word);

        // Первые 8 весов блока
        __m256i codes_lo = _mm256_and_si256(_mm256_srlv_epi32(w_broadcast, shifts_lo), mask2_vec);
        __m256i signs_lo_i = _mm256_sub_epi32(codes_lo, one_vec);
        __m256 signs_lo_f = _mm256_cvtepi32_ps(signs_lo_i);
        __m256 x_lo = _mm256_loadu_ps(x_block + w * 16);
        acc_vec = _mm256_fmadd_ps(signs_lo_f, x_lo, acc_vec);

        // Следующие 8 весов блока
        __m256i codes_hi = _mm256_and_si256(_mm256_srlv_epi32(w_broadcast, shifts_hi), mask2_vec);
        __m256i signs_hi_i = _mm256_sub_epi32(codes_hi, one_vec);
        __m256 signs_hi_f = _mm256_cvtepi32_ps(signs_hi_i);
        __m256 x_hi = _mm256_loadu_ps(x_block + w * 16 + 8);
        acc_vec = _mm256_fmadd_ps(signs_hi_f, x_hi, acc_vec);
    }

    // Горизонтальная редукция 8 float в регистре AVX2
    __m128 low128 = _mm256_castps256_ps128(acc_vec);
    __m128 high128 = _mm256_extractf128_ps(acc_vec, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // Хвост (если group_size не кратен 16)
    for (int j = num_words * 4; j < num_bytes; ++j) {
        uint8_t p = packed_block[j];
        int c = j * 4;
        sum += ((float)((int)(p & 0x03) - 1)) * x_block[c]
             + ((float)((int)((p >> 2) & 0x03) - 1)) * x_block[c + 1]
             + ((float)((int)((p >> 4) & 0x03) - 1)) * x_block[c + 2]
             + ((float)((int)((p >> 6) & 0x03) - 1)) * x_block[c + 3];
    }
    return sum;
#else
    // Переносимый C++ фоллбэк (ARM NEON / RISC-V / без AVX2)
    float sum = 0.0f;
    for (int j = 0; j < num_bytes; ++j) {
        uint8_t p = packed_block[j];
        int c = j * 4;
        sum += ((float)((int)(p & 0x03) - 1)) * x_block[c]
             + ((float)((int)((p >> 2) & 0x03) - 1)) * x_block[c + 1]
             + ((float)((int)((p >> 4) & 0x03) - 1)) * x_block[c + 2]
             + ((float)((int)((p >> 6) & 0x03) - 1)) * x_block[c + 3];
    }
    return sum;
#endif
}

// Нативный 4-битный блок (INT4 {-7..+7}) для каскадных слоев V-CORE
static inline float dot_int4_block_cpu(const uint8_t* __restrict__ packed_block,
                                       const float* __restrict__ x_block,
                                       int group_size) {
    int num_bytes = group_size / 2;
    float sum = 0.0f;
    for (int j = 0; j < num_bytes; ++j) {
        uint8_t p = packed_block[j];
        int q0 = (int)(p & 0x0F) - 7;
        int q1 = (int)((p >> 4) & 0x0F) - 7;
        sum += (float)q0 * x_block[j * 2] + (float)q1 * x_block[j * 2 + 1];
    }
    return sum;
}

// Главная C-функция движка vcore.cpp для умножения сжатой матрицы на батч векторов (CPU)
extern "C" void vcore_cpu_matmul(const uint8_t* __restrict__ W_packed,
                                 const float* __restrict__ block_scales,
                                 const float* __restrict__ X,
                                 float* __restrict__ Y,
                                 int batch_tokens,
                                 int out_features,
                                 int in_features,
                                 int group_size,
                                 int bits) {
    int blocks_per_row = in_features / group_size;
    int bytes_per_row = (bits == 2) ? (in_features / 4) : (in_features / 2);
    int bytes_per_block = (bits == 2) ? (group_size / 4) : (group_size / 2);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < batch_tokens; ++t) {
        for (int row = 0; row < out_features; ++row) {
            const uint8_t* w_row = W_packed + (size_t)row * bytes_per_row;
            const float* s_row = block_scales + (size_t)row * blocks_per_row;
            const float* x_row = X + (size_t)t * in_features;

            float total = 0.0f;
            if (bits == 2) {
                for (int b = 0; b < blocks_per_row; ++b) {
                    float dot = dot_ternary_block_cpu(
                        w_row + b * bytes_per_block,
                        x_row + b * group_size,
                        group_size
                    );
                    total += dot * s_row[b];
                }
            } else {
                for (int b = 0; b < blocks_per_row; ++b) {
                    float dot = dot_int4_block_cpu(
                        w_row + b * bytes_per_block,
                        x_row + b * group_size,
                        group_size
                    );
                    total += dot * s_row[b];
                }
            }
            Y[(size_t)t * out_features + row] = total;
        }
    }
}
