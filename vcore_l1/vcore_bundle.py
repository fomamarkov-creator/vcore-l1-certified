/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 *
 * Zero-Copy mmap Reader for Single-File .vcore Bundles (VCORE5)
 */

#pragma once

#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <map>

struct VCoreBundleSlice {
    const uint8_t* ptr = nullptr;
    size_t size = 0;
};

class VCoreBundleMmap {
public:
    int fd = -1;
    uint8_t* map_ptr = nullptr;
    size_t file_size = 0;
    std::map<std::string, VCoreBundleSlice> entries;
    std::string cert_summary;

    ~VCoreBundleMmap() {
        if (map_ptr && map_ptr != MAP_FAILED) munmap(map_ptr, file_size);
        if (fd >= 0) close(fd);
    }

    static bool is_vcore_bundle(const std::string& path) {
        if (path.size() >= 6 && path.substr(path.size() - 6) == ".vcore") return true;
        int f = open(path.c_str(), O_RDONLY);
        if (f < 0) return false;
        std::string magic(8, '\0');
        ssize_t n = read(f, &magic[0], 8);
        close(f);
        return (n == 8 && memcmp(magic.data(), "VCORE5\x00\x01", 8) == 0);
    }

    bool open_bundle(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 24) {
            close(fd);
            fd = -1;
            return false;
        }
        file_size = (size_t)st.st_size;
        map_ptr = (uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_ptr == MAP_FAILED) {
            close(fd);
            fd = -1;
            return false;
        }

        if (memcmp(map_ptr, "VCORE5\x00\x01", 8) != 0) return false;

        uint64_t manifest_len = 0;
        memcpy(&manifest_len, map_ptr + 8, 8);
        if (16 + manifest_len > file_size) return false;

        std::string manifest((const char*)(map_ptr + 16), (size_t)manifest_len);
        size_t header_end = 16 + (size_t)manifest_len;
        size_t data_base = (header_end % 64 == 0) ? header_end : (header_end + (64 - (header_end % 64)));

        size_t d_pos = manifest.find("\"certified_max_block_drift\"");
        if (d_pos != std::string::npos) {
            size_t col = manifest.find(':', d_pos);
            size_t comma = manifest.find_first_of(",}", col);
            if (col != std::string::npos && comma != std::string::npos) {
                cert_summary = manifest.substr(col + 1, comma - col - 1);
            }
        }

        size_t ent_pos = manifest.find("\"entries\"");
        if (ent_pos == std::string::npos) return false;

        size_t pos = ent_pos;
        while ((pos = manifest.find("\"name\"", pos)) != std::string::npos) {
            size_t q1 = manifest.find('"', pos + 6);
            size_t q2 = manifest.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) break;
            std::string fname = manifest.substr(q1 + 1, q2 - q1 - 1);

            size_t off_p = manifest.find("\"offset\"", q2);
            size_t off_c = manifest.find(':', off_p);
            uint64_t off_val = std::stoull(manifest.substr(off_c + 1));

            size_t sz_p = manifest.find("\"size\"", off_c);
            size_t sz_c = manifest.find(':', sz_p);
            uint64_t sz_val = std::stoull(manifest.substr(sz_c + 1));

            if (data_base + off_val + sz_val <= file_size) {
                entries[fname] = {map_ptr + data_base + off_val, (size_t)sz_val};
            }
            pos = sz_c;
        }
        return !entries.empty();
    }

    std::string read_text_entry(const std::string& name) const {
        auto it = entries.find(name);
        if (it == entries.end() || !it->second.ptr) return "";
        return std::string((const char*)it->second.ptr, it->second.size);
    }
};
