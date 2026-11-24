/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * Zero-dependency C++17 BPE Tokenizer for Qwen2 / Llama tokenizer.json
 */

#pragma once

#include <stdint.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class VCoreBPETokenizer {
public:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_ranks;
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
    std::string byte_to_unicode[256];
    bool is_byte_level = false;

    VCoreBPETokenizer() {
        init_gpt2_byte_tables();
    }

    bool load(const std::string& model_dir) {
        std::ifstream f(model_dir + "/tokenizer.json");
        if (!f.is_open()) return false;
        std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        // 1. Парсим словарь "vocab": { "token": id, ... }
        size_t vocab_pos = json.find("\"vocab\"");
        if (vocab_pos == std::string::npos) return false;
        size_t brace_start = json.find('{', vocab_pos);
        if (brace_start == std::string::npos) return false;

        int max_id = -1;
        std::vector<std::pair<int, std::string>> entries;
        size_t i = brace_start + 1;
        while (i < json.size()) {
            while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t' || json[i] == ',')) i++;
            if (i >= json.size() || json[i] == '}') break;
            if (json[i] != '"') { i++; continue; }

            std::string tok = parse_json_string(json, i);
            while (i < json.size() && json[i] != ':') i++;
            if (i < json.size()) i++;
            while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t')) i++;

            size_t num_start = i;
            while (i < json.size() && (isdigit((unsigned char)json[i]) || json[i] == '-')) i++;
            if (num_start < i) {
                int id = std::stoi(json.substr(num_start, i - num_start));
                entries.push_back({id, tok});
                if (id > max_id) max_id = id;
            }
        }

        // 2. Парсим специальные токены из "added_tokens"
        size_t added_pos = json.find("\"added_tokens\"");
        if (added_pos != std::string::npos) {
            size_t arr_start = json.find('[', added_pos);
            size_t arr_end = find_matching_bracket(json, arr_start, '[', ']');
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                size_t p = arr_start;
                while ((p = json.find("\"id\"", p)) != std::string::npos && p < arr_end) {
                    size_t c = json.find(':', p);
                    int id = std::stoi(json.substr(c + 1));
                    size_t cp = json.find("\"content\"", c);
                    if (cp != std::string::npos && cp < arr_end) {
                        size_t q = json.find('"', cp + 9);
                        std::string content = parse_json_string(json, q);
                        entries.push_back({id, content});
                        if (id > max_id) max_id = id;
                    }
                    p = c + 1;
                }
            }
        }

        if (max_id < 0) return false;
        id_to_token.assign((size_t)max_id + 1, "");
        for (const auto& kv : entries) {
            if (kv.first >= 0 && kv.first <= max_id) {
                id_to_token[kv.first] = kv.second;
                token_to_id[kv.second] = kv.first;
            }
        }

        // Определяем тип BPE: Qwen2/GPT2 (содержит 'Ġ' = U+0120) или SentencePiece ('▁' = U+2581)
        if (token_to_id.count("\xC4\xA0") || json.find("\"ByteLevel\"") != std::string::npos) {
            is_byte_level = true;
        }

        // 3. Парсим правила слияния "merges": [ "a b", ... ]
        size_t merges_pos = json.find("\"merges\"");
        if (merges_pos != std::string::npos) {
            size_t m_start = json.find('[', merges_pos);
            size_t m_end = find_matching_bracket(json, m_start, '[', ']');
            if (m_start != std::string::npos && m_end != std::string::npos) {
                size_t p = m_start + 1;
                int rank = 0;
                while (p < m_end) {
                    while (p < m_end && json[p] != '"') p++;
                    if (p >= m_end) break;
                    std::string m_str = parse_json_string(json, p);
                    merge_ranks[m_str] = rank++;
                }
            }
        }
        return !id_to_token.empty();
    }
        std::string decode(int token_id) const {
        if (token_id < 0 || token_id >= (int)id_to_token.size()) return "";
        const std::string& raw = id_to_token[token_id];
        if (raw.empty()) return "";
        // Пропускаем служебные теги вида <|im_end|>
        if (raw.size() >= 4 && raw.substr(0, 2) == "<|" && raw.substr(raw.size() - 2) == "|>") {
            return "";
        }

        if (is_byte_level) {
            std::string out;
            size_t i = 0;
            while (i < raw.size()) {
                uint32_t cp = decode_utf8(raw, i);
                auto it = unicode_to_byte.find(cp);
                if (it != unicode_to_byte.end()) {
                    out.push_back((char)it->second);
                } else {
                    out += encode_utf8(cp);
                }
            }
            return out;
        } else {
            // SentencePiece: заменяем U+2581 ('▁') на пробел
            std::string out;
            for (size_t i = 0; i < raw.size();) {
                if (i + 2 < raw.size() && (uint8_t)raw[i] == 0xE2 && (uint8_t)raw[i+1] == 0x96 && (uint8_t)raw[i+2] == 0x81) {
                    out.push_back(' ');
                    i += 3;
                } else {
                    out.push_back(raw[i++]);
                }
            }
            return out;
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::string pre;
        if (is_byte_level) {
            for (unsigned char c : text) pre += byte_to_unicode[c];
        } else {
            pre = "\xE2\x96\x81";
            for (char c : text) {
                if (c == ' ') pre += "\xE2\x96\x81";
                else pre.push_back(c);
            }
        }

        // Разбиваем на начальные UTF-8 символы
        std::vector<std::string> syms;
        for (size_t i = 0; i < pre.size();) {
            size_t prev = i;
            decode_utf8(pre, i);
            syms.push_back(pre.substr(prev, i - prev));
        }

        // Жадное BPE-слияние по рангам
        while (syms.size() >= 2) {
            int best_rank = 1e9;
            int best_idx = -1;
            for (int i = 0; i + 1 < (int)syms.size(); ++i) {
                std::string pair = syms[i] + " " + syms[i + 1];
                auto it = merge_ranks.find(pair);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx < 0) break;
            syms[best_idx] += syms[best_idx + 1];
            syms.erase(syms.begin() + best_idx + 1);
        }

        std::vector<int> ids;
        for (const auto& s : syms) {
            auto it = token_to_id.find(s);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                for (unsigned char c : s) {
                    auto it_b = token_to_id.find(std::string(1, (char)c));
                    if (it_b != token_to_id.end()) ids.push_back(it_b->second);
                }
            }
        }
        return ids;
    }

    std::vector<int> encode_chat_prompt(const std::string& user_prompt) const {
        std::vector<int> ids;
        auto it_im_start = token_to_id.find("<|im_start|>");
        auto it_im_end = token_to_id.find("<|im_end|>");
        if (it_im_start != token_to_id.end() && it_im_end != token_to_id.end()) {
            // Шаблон Qwen2 ChatML
            auto add_msg = [&](const std::string& role, const std::string& content) {
                ids.push_back(it_im_start->second);
                auto t_role = encode(role + "\n" + content);
                ids.insert(ids.end(), t_role.begin(), t_role.end());
                ids.push_back(it_im_end->second);
                auto t_nl = encode("\n");
                ids.insert(ids.end(), t_nl.begin(), t_nl.end());
            };
            add_msg("system", "You are a helpful assistant.");
            add_msg("user", user_prompt);
            ids.push_back(it_im_start->second);
            auto t_asst = encode("assistant\n");
            ids.insert(ids.end(), t_asst.begin(), t_asst.end());
        } else {
            // Базовый шаблон для Llama / прочих моделей
            auto it_bos = token_to_id.find("<s>");
            if (it_bos != token_to_id.end()) ids.push_back(it_bos->second);
            auto t_usr = encode(user_prompt);
            ids.insert(ids.end(), t_usr.begin(), t_usr.end());
        }
        return ids;
    }

private:
    static std::string encode_utf8(uint32_t cp) {
        std::string s;
        if (cp <= 0x7F) { s.push_back((char)cp); }
        else if (cp <= 0x7FF) {
            s.push_back((char)(0xC0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            s.push_back((char)(0xE0 | (cp >> 12)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            s.push_back((char)(0xF0 | (cp >> 18)));
            s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        }
        return s;
    }

    static uint32_t decode_utf8(const std::string& s, size_t& i) {
        uint8_t c = (uint8_t)s[i++];
        if (c < 0x80) return c;
        if ((c >> 5) == 0x6 && i < s.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | ((uint8_t)s[i++] & 0x3F);
            return cp;
        }
        if ((c >> 4) == 0xE && i + 1 < s.size()) {
            uint32_t cp = ((c & 0x0F) << 12) | (((uint8_t)s[i] & 0x3F) << 6) | ((uint8_t)s[i+1] & 0x3F);
            i += 2;
            return cp;
        }
        if ((c >> 3) == 0x1E && i + 2 < s.size()) {
            uint32_t cp = ((c & 0x07) << 18) | (((uint8_t)s[i] & 0x3F) << 12) | (((uint8_t)s[i+1] & 0x3F) << 6) | ((uint8_t)s[i+2] & 0x3F);
            i += 3;
            return cp;
        }
        return c;
    }

    void init_gpt2_byte_tables() {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            uint8_t b = (uint8_t)bs[i];
            uint32_t cp = (uint32_t)cs[i];
            byte_to_unicode[b] = encode_utf8(cp);
            unicode_to_byte[cp] = b;
        }
    }

    static size_t find_matching_bracket(const std::string& s, size_t start, char open_c, char close_c) {
        if (start == std::string::npos || start >= s.size()) return std::string::npos;
        int depth = 0;
        bool in_str = false;
        for (size_t i = start; i < s.size(); ++i) {
            if (s[i] == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
            if (!in_str) {
                if (s[i] == open_c) depth++;
                else if (s[i] == close_c) {
                    depth--;
                    if (depth == 0) return i;
                }
            }
        }
        return std::string::npos;
    }

    static std::string parse_json_string(const std::string& s, size_t& i) {
        std::string res;
        if (i >= s.size() || s[i] != '"') return res;
        i++;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') break;
            if (c == '\\' && i < s.size()) {
                char esc = s[i++];
                if (esc == '"') res.push_back('"');
                else if (esc == '\\') res.push_back('\\');
                else if (esc == '/') res.push_back('/');
                else if (esc == 'n') res.push_back('\n');
                else if (esc == 'r') res.push_back('\r');
                else if (esc == 't') res.push_back('\t');
                else if (esc == 'u' && i + 4 <= s.size()) {
                    uint32_t cp = (uint32_t)std::stoul(s.substr(i, 4), nullptr, 16);
                    i += 4;
                    res += encode_utf8(cp);
                }
            } else {
                res.push_back(c);
            }
        }
        return res;
    }
};
