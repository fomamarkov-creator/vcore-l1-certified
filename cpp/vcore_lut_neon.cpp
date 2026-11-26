/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * Native ARM NEON 1.58-bit Kernel & Direct Sub-1-Bit E8 LUT Inference Engine
 */

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VCORE_HAS_NEON 1
#else
#define VCORE_HAS_NEON 0
#endif

static std::vector<float> build_e8_codebook_241() {
    std::vector<float> cb(241 * 8, 0.0f);
    int idx = 1; // 0 = нулевой вектор
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
        if (__builtin_popcount((unsigned)mask) % 2 == 0) {
            for (int k = 0; k < 8; ++k) {
                cb[idx * 8 + k] = ((mask >> k) & 1) ? -0.5f : 0.5f;
            }
            idx++;
        }
    }
    return cb;
}

extern "C" {

// Возвращает 1, если скомпилировано с аппаратным ARM NEON, иначе 0
int vcore_has_arm_neon() {
    return VCORE_HAS_NEON;
}

// Нативный ARM NEON 1.58-bit Fused GEMM (для MacBook M-серии, Android Termux, Raspberry Pi)
void vcore_neon_fused_gemm(
    const uint8_t* W_packed,
    const float* block_scales,
    const float* X,
    float* Y,
    int batch_tokens,
    int out_features,
    int in_features,
    int group_size
) {
    const int bytes_per_row = in_features / 4;
    const int blocks_per_row = in_features / group_size;
    const int bytes_per_block = group_size / 4;

    for (int t = 0; t < batch_tokens; ++t) {
        const float* x_row = X + (size_t)t * in_features;
        float* y_row = Y + (size_t)t * out_features;

        for (int r = 0; r < out_features; ++r) {
            const uint8_t* w_row = W_packed + (size_t)r * bytes_per_row;
            const float* s_row = block_scales + (size_t)r * blocks_per_row;
            float total_acc = 0.0f;

            for (int b = 0; b < blocks_per_row; ++b) {
                const uint8_t* w_blk = w_row + b * bytes_per_block;
                const float* x_blk = x_row + b * group_size;

#if VCORE_HAS_NEON
                float32x4_t v_acc = vdupq_n_f32(0.0f);
                for (int j = 0; j < bytes_per_block; ++j) {
                    uint8_t p = w_blk[j];
                    float w_vals;
                    w_vals = (float)((int)(p & 0x03) - 1);
                    w_vals = (float)((int)((p >> 2) & 0x03) - 1);
                    w_vals = (float)((int)((p >> 4) & 0x03) - 1);
                    w_vals = (float)((int)((p >> 6) & 0x03) - 1);
                    float32x4_t vw = vld1q_f32(w_vals);
                    float32x4_t vx = vld1q_f32(x_blk + j * 4);
                    v_acc = vmlaq_f32(v_acc, vw, vx);
                }
                float blk_sum = vaddvq_f32(v_acc);
#else
                float blk_sum = 0.0f;
                for (int j = 0; j < bytes_per_block; ++j) {
                    uint8_t p = w_blk[j];
                    const float* xp = x_blk + j * 4;
                    blk_sum += (float)((int)(p & 0x03) - 1) * xp
                             + (float)((int)((p >> 2) & 0x03) - 1) * xp
                             + (float)((int)((p >> 4) & 0x03) - 1) * xp
                             + (float)((int)((p >> 6) & 0x03) - 1) * xp;
                }
#endif
                total_acc += blk_sum * s_row[b];
            }
            y_row[r] = total_acc;
        }
    }
}

// Прямой Sub-1-Bit E8 LUT GEMM (0.989 b/w без распаковки весов в FP16)
void vcore_e8_lut_gemm(
    const uint8_t* E8_indices,   // (out_features, in_features / 8)
    const float* block_scales,   // (out_features, in_features / group_size)
    const float* X,              // (batch_tokens, in_features)
    float* Y,                    // (batch_tokens, out_features)
    int batch_tokens,
    int out_features,
    int in_features,
    int group_size
) {
    static const std::vector<float> cb = build_e8_codebook_241();
    const int num_e8_blocks = in_features / 8;
    const int e8_per_group = group_size / 8;

    // L1-таблица откликов для одного токена: (num_e8_blocks * 241) float
    std::vector<float> lut((size_t)num_e8_blocks * 241, 0.0f);

    for (int t = 0; t < batch_tokens; ++t) {
        const float* x_row = X + (size_t)t * in_features;
        float* y_row = Y + (size_t)t * out_features;

        // Шаг 1: строим LUT (241 скалярное произведение на каждые 8 каналов входа)
        for (int eb = 0; eb < num_e8_blocks; ++eb) {
            const float* xb = x_row + eb * 8;
            float* lut_eb = lut.data() + (size_t)eb * 241;
            lut_eb = 0.0f;
            for (int c = 1; c < 241; ++c) {
                const float* cw = cb.data() + c * 8;
#if VCORE_HAS_NEON
                float32x4_t v0 = vmulq_f32(vld1q_f32(xb), vld1q_f32(cw));
                float32x4_t v1 = vmulq_f32(vld1q_f32(xb + 4), vld1q_f32(cw + 4));
                lut_eb[c] = vaddvq_f32(vaddq_f32(v0, v1));
#else
                lut_eb[c] = xb * cw + xb * cw + xb * cw + xb * cw
                          + xb * cw + xb * cw + xb * cw + xb * cw;
#endif
            }
        }

        // Шаг 2: чистый табличный поиск (O(1) lookup) без умножений весов
        const int num_groups = in_features / group_size;
        for (int r = 0; r < out_features; ++r) {
            const uint8_t* idx_row = E8_indices + (size_t)r * num_e8_blocks;
            const float* s_row = block_scales + (size_t)r * num_groups;
            float acc = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                float grp_sum = 0.0f;
                int eb_start = g * e8_per_group;
                for (int k = 0; k < e8_per_group; ++k) {
                    int eb = eb_start + k;
                    uint8_t code = idx_row[eb];
                    if (code >= 241) code = 0;
                    grp_sum += lut[(size_t)eb * 241 + code];
                }
                acc += grp_sum * s_row[g];
            }
            y_row[r] = acc;
        }
    }
}

} // extern "C"
