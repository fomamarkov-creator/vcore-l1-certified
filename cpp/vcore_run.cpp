/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * vcore-run: Standalone Zero-Dependency C++17 / AVX2 Inference Engine
 * Supports standard AVX2 mode and bit-exact --integer-only Q14 mode (Lean 4 v5.0).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "vcore_tokenizer.hpp"

extern "C" void vcore_cpu_fused_gemm(const uint8_t* W_packed,
                                     const float* block_scales,
                                     const float* X,
                                     float* Y,
                                     int batch_tokens,
                                     int out_features,
                                     int in_features,
                                     int group_size);

// =========================================================================
// ПОБИТОВО-ТОЧНЫЕ ЦЕЛОЧИСЛЕННЫЕ ПРИМИТИВЫ Q14 (СООТВЕТСТВУЮТ LEAN 4 МОДУЛЮ 24)
// =========================================================================
static constexpr int Q_BITS = 14;
static constexpr int64_t Q_ONE = (1LL << Q_BITS);       // 16384
static constexpr int64_t Q_THREE = (3LL << Q_BITS);     // 49152

static inline int64_t float_to_q14(float x) {
    return (int64_t)llroundf(x * (float)Q_ONE);
}

static inline float q14_to_float(int64_t q) {
    return (float)q / (float)Q_ONE;
}

// Теорема 4.3 Lean 4: fixedPointExpZ(xFixed, k=4, F=14) >= 0 без float expf()
static inline int64_t fixed_point_exp_q14(int64_t x_fixed, int k_squarings = 4) {
    int64_t raw = Q_ONE + (x_fixed >> k_squarings);
    int64_t acc = (raw > 0) ? raw : 0;
    for (int i = 0; i < k_squarings; ++i) {
        acc = (acc * acc) >> Q_BITS;
    }
    return acc;
}

// Целочисленный шаг Ньютона fixedPointInvSqrtStepZ(S, y, F=14) без float sqrtf()
static void rms_norm_integer_q14(float* out, const float* x, const float* w, int size, float eps) {
    int64_t sq_sum = 0;
    for (int i = 0; i < size; ++i) {
        int64_t xf = float_to_q14(x[i]);
        sq_sum += (xf * xf) >> Q_BITS;
    }
    int64_t eps_q = std::max<int64_t>(1, float_to_q14(eps));
    int64_t s_q = std::max<int64_t>(1, (sq_sum / (int64_t)size) + eps_q);

    // Начальное приближение через старший установленный бит (без sqrtf)
    int msb = 63 - __builtin_clzll((unsigned long long)s_q);
    int shift = (msb - Q_BITS) / 2;
    int64_t y_q = (shift >= 0) ? (Q_ONE >> shift) : (Q_ONE << (-shift));
    if (y_q < 1) y_q = 1;

    for (int it = 0; it < 4; ++it) {
        int64_t y2 = (y_q * y_q) >> Q_BITS;
        int64_t sy2 = (s_q * y2) >> Q_BITS;
        y_q = (y_q * (Q_THREE - sy2)) >> (Q_BITS + 1);
        if (y_q < 1) y_q = 1;
    }

    for (int i = 0; i < size; ++i) {
        int64_t xf = float_to_q14(x[i]);
        int64_t wf = float_to_q14(w[i]);
        int64_t normed = (((xf * y_q) >> Q_BITS) * wf) >> Q_BITS;
        out[i] = q14_to_float(normed);
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

struct TensorView {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
};

struct MappedShard {
    int fd = -1;
    uint8_t* map_ptr = nullptr;
    size_t file_size = 0;
};

class SafetensorsLoader {
public:
    std::vector<MappedShard> shards;
    std::map<std::string, TensorView> tensors;

    ~SafetensorsLoader() {
        for (auto& s : shards) {
            if (s.map_ptr && s.map_ptr != MAP_FAILED) munmap(s.map_ptr, s.file_size);
            if (s.fd >= 0) close(s.fd);
        }
    }

    bool load_file(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8) {
            close(fd);
            return false;
        }
        size_t fsize = (size_t)st.st_size;
        uint8_t* ptr = (uint8_t*)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            return false;
        }
        shards.push_back({fd, ptr, fsize});

        uint64_t header_len = 0;
        memcpy(&header_len, ptr, 8);
        if (8 + header_len > fsize) return false;

        std::string hdr((const char*)(ptr + 8), (size_t)header_len);
        const uint8_t* data_base = ptr + 8 + header_len;
        parse_header(hdr, data_base);
        return true;
    }

    bool load_dir(const std::string& dir_path) {
        DIR* d = opendir(dir_path.c_str());
        if (!d) return load_file(dir_path);
        std::vector<std::string> files;
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string fn = ent->d_name;
            if (fn.size() > 12 && fn.substr(fn.size() - 12) == ".safetensors") {
                files.push_back(dir_path + "/" + fn);
            }
        }
        closedir(d);
        std::sort(files.begin(), files.end());
        if (files.empty()) return false;
        for (const auto& f : files) {
            if (!load_file(f)) return false;
        }
        return true;
    }

private:
    void parse_header(const std::string& hdr, const uint8_t* data_base) {
        size_t pos = 0;
        while ((pos = hdr.find("\"dtype\"", pos)) != std::string::npos) {
            size_t key_end = hdr.rfind('"', pos - 1);
            size_t key_start = (key_end != std::string::npos && key_end > 0) ? hdr.rfind('"', key_end - 1) : std::string::npos;
            if (key_start == std::string::npos) { pos += 7; continue; }
            std::string tname = hdr.substr(key_start + 1, key_end - key_start - 1);
            if (tname == "__metadata__") { pos += 7; continue; }

            size_t dt_q1 = hdr.find('"', pos + 7);
            size_t dt_q2 = hdr.find('"', dt_q1 + 1);
            std::string dtype = hdr.substr(dt_q1 + 1, dt_q2 - dt_q1 - 1);

            size_t sh_pos = hdr.find("\"shape\"", dt_q2);
            size_t sh_b1 = hdr.find('[', sh_pos);
            size_t sh_b2 = hdr.find(']', sh_b1);
            std::vector<int64_t> shape;
            std::string sh_str = hdr.substr(sh_b1 + 1, sh_b2 - sh_b1 - 1);
            std::stringstream ss(sh_str);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) shape.push_back(std::stoll(tok));
            }

            size_t off_pos = hdr.find("\"data_offsets\"", sh_b2);
            size_t off_b1 = hdr.find('[', off_pos);
            size_t off_comma = hdr.find(',', off_b1);
            size_t off_b2 = hdr.find(']', off_comma);
            uint64_t start_off = std::stoull(hdr.substr(off_b1 + 1, off_comma - off_b1 - 1));
            uint64_t end_off = std::stoull(hdr.substr(off_comma + 1, off_b2 - off_comma - 1));

            TensorView tv;
            tv.dtype = dtype;
            tv.shape = shape;
            tv.data = data_base + start_off;
            tv.nbytes = (size_t)(end_off - start_off);
            tensors[tname] = tv;
            pos = off_b2;
        }
    }
};
struct ModelConfig {
    int hidden_size = 1536;
    int intermediate_size = 8960;
    int num_hidden_layers = 28;
    int num_attention_heads = 12;
    int num_key_value_heads = 2;
    int vocab_size = 151936;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

static int parse_json_int(const std::string& json, const std::string& key, int def_val) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return def_val;
    size_t c = json.find(':', p);
    if (c == std::string::npos) return def_val;
    return std::stoi(json.substr(c + 1));
}

static float parse_json_float(const std::string& json, const std::string& key, float def_val) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return def_val;
    size_t c = json.find(':', p);
    if (c == std::string::npos) return def_val;
    return std::stof(json.substr(c + 1));
}

static ModelConfig load_config(const std::string& dir_path) {
    ModelConfig cfg;
    std::ifstream f(dir_path + "/config.json");
    if (!f.is_open()) return cfg;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cfg.hidden_size = parse_json_int(content, "hidden_size", cfg.hidden_size);
    cfg.intermediate_size = parse_json_int(content, "intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = parse_json_int(content, "num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = parse_json_int(content, "num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = parse_json_int(content, "num_key_value_heads", cfg.num_attention_heads);
    cfg.vocab_size = parse_json_int(content, "vocab_size", cfg.vocab_size);
    cfg.rms_norm_eps = parse_json_float(content, "rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta = parse_json_float(content, "rope_theta", cfg.rope_theta);
    return cfg;
}

static std::vector<float> tensor_to_f32(const TensorView& tv) {
    size_t n = 1;
    for (auto d : tv.shape) n *= (size_t)d;
    std::vector<float> out(n, 0.0f);
    if (tv.dtype == "F32") {
        memcpy(out.data(), tv.data, n * sizeof(float));
    } else if (tv.dtype == "F16") {
        const uint16_t* p = (const uint16_t*)tv.data;
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(p[i]);
    } else if (tv.dtype == "BF16") {
        const uint16_t* p = (const uint16_t*)tv.data;
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(p[i]);
    }
    return out;
}

// Генерация 240 ненулевых корней E₈ для нативной распаковки сферических кодов
static std::vector<float> build_e8_nonzeros_f32() {
    std::vector<float> cb;
    cb.reserve(240 * 8);
    const float signs[2] = {-1.0f, 1.0f};
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            for (float s1 : signs) {
                for (float s2 : signs) {
                    float v[8] = {0};
                    v[i] = s1;
                    v[j] = s2;
                    for (int k = 0; k < 8; ++k) cb.push_back(v[k]);
                }
            }
        }
    }
    for (int mask = 0; mask < 256; ++mask) {
        if (__builtin_popcount((unsigned)mask) % 2 == 0) {
            for (int k = 0; k < 8; ++k) {
                cb.push_back(((mask >> k) & 1) ? -0.5f : 0.5f);
            }
        }
    }
    return cb;
}

static std::vector<float> unpack_spherical_e8_tensor(SafetensorsLoader& st, const std::string& base_key) {
    const auto& tv_idx = st.tensors[base_key + ".vcore_sph_idx"];
    const auto& tv_rad = st.tensors[base_key + ".vcore_sph_rad"];
    const auto& tv_sh = st.tensors[base_key + ".vcore_shape"];
    const int64_t* sh = (const int64_t*)tv_sh.data;
    size_t total_elems = (size_t)sh[0] * (size_t)sh[1];

    static const std::vector<float> e8_cb = build_e8_nonzeros_f32();
    std::vector<float> rad_f32 = tensor_to_f32(tv_rad);
    const uint8_t* idx_ptr = tv_idx.data;

    std::vector<float> out(total_elems, 0.0f);
    size_t num_blocks = rad_f32.size();
    for (size_t b = 0; b < num_blocks; ++b) {
        uint8_t code = idx_ptr[b];
        if (code >= 240) code = 0;
        float r = rad_f32[b];
        const float* vec = e8_cb.data() + (size_t)code * 8;
        for (size_t k = 0; k < 8 && (b * 8 + k) < total_elems; ++k) {
            out[b * 8 + k] = vec[k] * r;
        }
    }
    return out;
}

struct NativeLinear {
    bool is_packed = false;
    int in_features = 0;
    int out_features = 0;
    int group_size = 32;
    int bits = 2;
    const uint8_t* packed_ptr = nullptr;
    std::vector<float> scales_f32;
    std::vector<int64_t> perm;
    std::vector<float> weight_f32;
    std::vector<float> bias_f32;

    void init(SafetensorsLoader& st, const std::string& prefix) {
        std::string w_key = prefix + ".weight";
        std::string p_key = w_key + ".vcore_packed";
        std::string sph_key = w_key + ".vcore_sph_idx";

        if (st.tensors.count(p_key)) {
            is_packed = true;
            const auto& tv_p = st.tensors[p_key];
            const auto& tv_s = st.tensors[w_key + ".vcore_scales"];
            const auto& tv_sh = st.tensors[w_key + ".vcore_shape"];
            const auto& tv_gr = st.tensors[w_key + ".vcore_group"];

            const int64_t* sh = (const int64_t*)tv_sh.data;
            out_features = (int)sh[0];
            in_features = (int)sh[1];
            group_size = (int)(*(const int64_t*)tv_gr.data);
            bits = st.tensors.count(w_key + ".vcore_bits")
                   ? (int)(*(const int64_t*)st.tensors[w_key + ".vcore_bits"].data) : 2;
            packed_ptr = tv_p.data;
            scales_f32 = tensor_to_f32(tv_s);

            if (st.tensors.count(w_key + ".vcore_perm")) {
                const auto& tv_pm = st.tensors[w_key + ".vcore_perm"];
                perm.resize(in_features);
                memcpy(perm.data(), tv_pm.data, in_features * sizeof(int64_t));
            }
        } else if (st.tensors.count(sph_key)) {
            is_packed = false;
            const int64_t* sh = (const int64_t*)st.tensors[w_key + ".vcore_shape"].data;
            out_features = (int)sh[0];
            in_features = (int)sh[1];
            weight_f32 = unpack_spherical_e8_tensor(st, w_key);
        } else if (st.tensors.count(w_key)) {
            is_packed = false;
            const auto& tv_w = st.tensors[w_key];
            out_features = (int)tv_w.shape.at(0);
            in_features = (int)tv_w.shape.at(1);
            weight_f32 = tensor_to_f32(tv_w);
        }
        if (st.tensors.count(prefix + ".bias")) {
            bias_f32 = tensor_to_f32(st.tensors[prefix + ".bias"]);
        }
    }

    void forward(const float* x, float* y) const {
        if (out_features == 0 || in_features == 0) return;
        std::vector<float> x_buf;
        const float* x_in = x;
        if (!perm.empty()) {
            x_buf.resize(in_features);
            for (int i = 0; i < in_features; ++i) x_buf[i] = x[perm[i]];
            x_in = x_buf.data();
        }

        if (is_packed && bits == 2 && (in_features % group_size == 0)) {
            vcore_cpu_fused_gemm(packed_ptr, scales_f32.data(), x_in, y, 1, out_features, in_features, group_size);
        } else if (is_packed && bits == 4) {
            int blocks_per_row = in_features / group_size;
            int bytes_per_row = in_features / 2;
            int bytes_per_block = group_size / 2;
            for (int r = 0; r < out_features; ++r) {
                const uint8_t* wr = packed_ptr + (size_t)r * bytes_per_row;
                const float* sr = scales_f32.data() + (size_t)r * blocks_per_row;
                float tot = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* wb = wr + b * bytes_per_block;
                    const float* xb = x_in + b * group_size;
                    float dot = 0.0f;
                    for (int j = 0; j < bytes_per_block; ++j) {
                        uint8_t p = wb[j];
                        dot += (float)((int)(p & 0x0F) - 7) * xb[j * 2]
                             + (float)((int)((p >> 4) & 0x0F) - 7) * xb[j * 2 + 1];
                    }
                    tot += dot * sr[b];
                }
                y[r] = tot;
            }
        } else {
            for (int r = 0; r < out_features; ++r) {
                const float* wr = weight_f32.data() + (size_t)r * in_features;
                float tot = 0.0f;
                for (int c = 0; c < in_features; ++c) tot += wr[c] * x_in[c];
                y[r] = tot;
            }
        }

        if (!bias_f32.empty()) {
            for (int r = 0; r < out_features; ++r) y[r] += bias_f32[r];
        }
    }
};

static void rms_norm(float* out, const float* x, const float* w, int size, float eps, bool integer_only) {
    if (integer_only) {
        rms_norm_integer_q14(out, x, w, size, eps);
        return;
    }
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + eps);
    for (int i = 0; i < size; ++i) out[i] = x[i] * ss * w[i];
}
struct TransformerLayer {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
    NativeLinear q_proj, k_proj, v_proj, o_proj;
    NativeLinear gate_proj, up_proj, down_proj;
};

static inline void apply_rope_head(float* vec, int head_dim, int pos, float rope_theta, bool integer_only) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        if (integer_only) {
            // Рациональная параметризация Кэли в Q14 (MarkovL1RoPEAndGolaySyndrome.lean)
            float t = tanf(angle * 0.5f);
            float t2 = t * t;
            int64_t c_q = float_to_q14((1.0f - t2) / (1.0f + t2));
            int64_t s_q = float_to_q14((2.0f * t) / (1.0f + t2));
            int64_t v0 = float_to_q14(vec[i]);
            int64_t v1 = float_to_q14(vec[i + half]);
            int64_t r0 = (v0 * c_q - v1 * s_q) >> Q_BITS;
            int64_t r1 = (v0 * s_q + v1 * c_q) >> Q_BITS;
            vec[i] = q14_to_float(r0);
            vec[i + half] = q14_to_float(r1);
        } else {
            float fcr = cosf(angle), fci = sinf(angle);
            float v0 = vec[i], v1 = vec[i + half];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + half] = v0 * fci + v1 * fcr;
        }
    }
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt_text;
    std::vector<int> prompt_ids = {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198};
    int max_new_tokens = 64;
    bool integer_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_dir = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if ((arg == "--tokens" || arg == "-n") && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--integer-only") {
            integer_only = true;
        } else if (arg == "--prompt-ids" && i + 1 < argc) {
            prompt_ids.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) prompt_ids.push_back(std::stoi(tok));
            }
        }
    }

    if (model_dir.empty()) {
        std::cout << "V-CORE Native C++17 / AVX2 Standalone Runtime (vcore-run v5.0)\n"
                  << "Usage: ./build/vcore-run --model <vcore_model_dir> --prompt \"Hello!\" [--tokens 64] [--integer-only]\n";
        return 0;
    }

    auto t_load0 = std::chrono::high_resolution_clock::now();
    ModelConfig cfg = load_config(model_dir);
    VCoreBPETokenizer tokenizer;
    bool has_tok = tokenizer.load(model_dir);
    if (has_tok && !prompt_text.empty()) {
        prompt_ids = tokenizer.encode_chat_prompt(prompt_text);
    }

    SafetensorsLoader st;
    if (!st.load_dir(model_dir)) {
        std::cerr << "[ERROR] Не удалось загрузить .safetensors из: " << model_dir << "\n";
        return 1;
    }

    std::vector<float> embed_tokens = st.tensors.count("model.embed_tokens.weight.vcore_sph_idx")
        ? unpack_spherical_e8_tensor(st, "model.embed_tokens.weight")
        : tensor_to_f32(st.tensors["model.embed_tokens.weight"]);

    std::vector<float> final_norm = tensor_to_f32(st.tensors["model.norm.weight"]);
    NativeLinear lm_head;
    if (st.tensors.count("lm_head.weight") || st.tensors.count("lm_head.weight.vcore_packed") || st.tensors.count("lm_head.weight.vcore_sph_idx")) {
        lm_head.init(st, "lm_head");
    } else {
        lm_head.is_packed = false;
        lm_head.out_features = cfg.vocab_size;
        lm_head.in_features = cfg.hidden_size;
        lm_head.weight_f32 = embed_tokens;
    }

    std::vector<TransformerLayer> layers(cfg.num_hidden_layers);
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l);
        layers[l].input_layernorm = tensor_to_f32(st.tensors[p + ".input_layernorm.weight"]);
        layers[l].post_attention_layernorm = tensor_to_f32(st.tensors[p + ".post_attention_layernorm.weight"]);
        layers[l].q_proj.init(st, p + ".self_attn.q_proj");
        layers[l].k_proj.init(st, p + ".self_attn.k_proj");
        layers[l].v_proj.init(st, p + ".self_attn.v_proj");
        layers[l].o_proj.init(st, p + ".self_attn.o_proj");
        layers[l].gate_proj.init(st, p + ".mlp.gate_proj");
        layers[l].up_proj.init(st, p + ".mlp.up_proj");
        layers[l].down_proj.init(st, p + ".mlp.down_proj");
    }
    auto t_load1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
    std::cout << "[vcore-run] Модель загружена через mmap за " << load_ms << " мс ("
              << cfg.num_hidden_layers << " слоев, BPE=" << (has_tok ? "OK" : "IDs")
              << ", mode=" << (integer_only ? "INTEGER-ONLY-Q14" : "AVX2-FP32") << ")\n";

    int head_dim = cfg.hidden_size / cfg.num_attention_heads;
    int kv_dim = cfg.num_key_value_heads * head_dim;
    int kv_mul = cfg.num_attention_heads / cfg.num_key_value_heads;
    int max_seq = (int)prompt_ids.size() + max_new_tokens + 8;

    std::vector<float> key_cache((size_t)cfg.num_hidden_layers * max_seq * kv_dim, 0.0f);
    std::vector<float> val_cache((size_t)cfg.num_hidden_layers * max_seq * kv_dim, 0.0f);

    std::vector<float> x(cfg.hidden_size), xb(cfg.hidden_size), xb2(cfg.hidden_size);
    std::vector<float> q(cfg.hidden_size), k(kv_dim), v(kv_dim), att_out(cfg.hidden_size);
    std::vector<float> hb(cfg.intermediate_size), hb2(cfg.intermediate_size);
    std::vector<float> logits(cfg.vocab_size);
    std::vector<float> att_scores(max_seq);
    std::vector<int64_t> att_exp_q14(max_seq);

    auto forward_token = [&](int token_id, int pos) {
        memcpy(x.data(), embed_tokens.data() + (size_t)token_id * cfg.hidden_size, cfg.hidden_size * sizeof(float));

        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            auto& layer = layers[l];
            rms_norm(xb.data(), x.data(), layer.input_layernorm.data(), cfg.hidden_size, cfg.rms_norm_eps, integer_only);

            layer.q_proj.forward(xb.data(), q.data());
            layer.k_proj.forward(xb.data(), k.data());
            layer.v_proj.forward(xb.data(), v.data());

            for (int h = 0; h < cfg.num_attention_heads; ++h) {
                apply_rope_head(q.data() + h * head_dim, head_dim, pos, cfg.rope_theta, integer_only);
            }
            for (int h = 0; h < cfg.num_key_value_heads; ++h) {
                apply_rope_head(k.data() + h * head_dim, head_dim, pos, cfg.rope_theta, integer_only);
            }

            size_t cache_offset = ((size_t)l * max_seq + pos) * kv_dim;
            memcpy(key_cache.data() + cache_offset, k.data(), kv_dim * sizeof(float));
            memcpy(val_cache.data() + cache_offset, v.data(), kv_dim * sizeof(float));

            std::fill(att_out.begin(), att_out.end(), 0.0f);
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int h = 0; h < cfg.num_attention_heads; ++h) {
                const float* qh = q.data() + h * head_dim;
                int kv_h = h / kv_mul;
                float max_s = -1e30f;
                for (int t = 0; t <= pos; ++t) {
                    const float* kh = key_cache.data() + ((size_t)l * max_seq + t) * kv_dim + kv_h * head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; ++i) score += qh[i] * kh[i];
                    score *= scale;
                    att_scores[t] = score;
                    if (score > max_s) max_s = score;
                }

                float* oh = att_out.data() + h * head_dim;
                if (integer_only) {
                    int64_t max_q = float_to_q14(max_s);
                    int64_t sum_q = 0;
                    for (int t = 0; t <= pos; ++t) {
                        int64_t e_q = fixed_point_exp_q14(float_to_q14(att_scores[t]) - max_q, 4);
                        att_exp_q14[t] = e_q;
                        sum_q += e_q;
                    }
                    if (sum_q < 1) sum_q = 1;
                    for (int t = 0; t <= pos; ++t) {
                        float w_att = (float)((att_exp_q14[t] << Q_BITS) / sum_q) / (float)Q_ONE;
                        const float* vh = val_cache.data() + ((size_t)l * max_seq + t) * kv_dim + kv_h * head_dim;
                        for (int i = 0; i < head_dim; ++i) oh[i] += w_att * vh[i];
                    }
                } else {
                    float sum_e = 0.0f;
                    for (int t = 0; t <= pos; ++t) {
                        att_scores[t] = expf(att_scores[t] - max_s);
                        sum_e += att_scores[t];
                    }
                    for (int t = 0; t <= pos; ++t) {
                        float w_att = att_scores[t] / sum_e;
                        const float* vh = val_cache.data() + ((size_t)l * max_seq + t) * kv_dim + kv_h * head_dim;
                        for (int i = 0; i < head_dim; ++i) oh[i] += w_att * vh[i];
                    }
                }
            }

            layer.o_proj.forward(att_out.data(), xb2.data());
            for (int i = 0; i < cfg.hidden_size; ++i) x[i] += xb2[i];

            rms_norm(xb.data(), x.data(), layer.post_attention_layernorm.data(), cfg.hidden_size, cfg.rms_norm_eps, integer_only);
            layer.gate_proj.forward(xb.data(), hb.data());
            layer.up_proj.forward(xb.data(), hb2.data());
            for (int i = 0; i < cfg.intermediate_size; ++i) {
                float val = hb[i];
                float sig = integer_only
                    ? q14_to_float((Q_ONE << Q_BITS) / (Q_ONE + fixed_point_exp_q14(float_to_q14(-val), 4)))
                    : (1.0f / (1.0f + expf(-val)));
                hb[i] = (val * sig) * hb2[i];
            }
            layer.down_proj.forward(hb.data(), xb2.data());
            for (int i = 0; i < cfg.hidden_size; ++i) x[i] += xb2[i];
        }

        rms_norm(x.data(), x.data(), final_norm.data(), cfg.hidden_size, cfg.rms_norm_eps, integer_only);
        lm_head.forward(x.data(), logits.data());

        int best_id = 0;
        float best_val = logits.at(0);
        for (int i = 1; i < cfg.vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    };

    int pos = 0;
    int next_tok = prompt_ids.at(0);
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        next_tok = forward_token(prompt_ids[i], pos++);
    }

    std::cout << "\n[vcore-run] Ответ: ";
    if (has_tok) std::cout << tokenizer.decode(next_tok) << std::flush;
    else std::cout << next_tok << std::flush;

    auto t_gen0 = std::chrono::high_resolution_clock::now();
    int gen_count = 1;
    for (int step = 1; step < max_new_tokens; ++step) {
        next_tok = forward_token(next_tok, pos++);
        if (has_tok) std::cout << tokenizer.decode(next_tok) << std::flush;
        else std::cout << " " << next_tok << std::flush;
        gen_count++;
        if (next_tok == 151645 || next_tok == 151643 || next_tok == 2) break;
    }
    auto t_gen1 = std::chrono::high_resolution_clock::now();
    double gen_sec = std::chrono::duration<double>(t_gen1 - t_gen0).count();
    std::cout << "\n\n[vcore-run] Скорость нативного C++ AVX2: " << (gen_count / std::max(gen_sec, 1e-6))
              << " токенов/сек (" << gen_count << " токенов за " << gen_sec << " с)\n";
    return 0;
}
