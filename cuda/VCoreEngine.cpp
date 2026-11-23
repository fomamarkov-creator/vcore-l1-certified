/*
 * Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
 * Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#define DEFAULT_ZETA 1.024f

extern "C" void run_vcore_resonance(const float* X, const float* Q, float* V, int n);
extern "C" void run_vcore_quantize(const float* d_W, const float* d_scales, float* d_W_quant,
                                   int total_elements, int group_size, float theta);

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "[CUDA ERROR]: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

extern "C" int vcore_host_quantize_copy(const float* h_W, const float* h_scales, float* h_W_quant,
                                        int total_elements, int group_size, float theta) {
    int num_blocks = total_elements / group_size;
    float *d_W = nullptr, *d_scales = nullptr, *d_W_quant = nullptr;

    checkCuda(cudaMalloc(&d_W, total_elements * sizeof(float)));
    checkCuda(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
    checkCuda(cudaMalloc(&d_W_quant, total_elements * sizeof(float)));

    checkCuda(cudaMemcpy(d_W, h_W, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_scales, h_scales, num_blocks * sizeof(float), cudaMemcpyHostToDevice));

    run_vcore_quantize(d_W, d_scales, d_W_quant, total_elements, group_size, theta);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_W_quant, d_W_quant, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_W);
    cudaFree(d_scales);
    cudaFree(d_W_quant);
    return 0;
}
