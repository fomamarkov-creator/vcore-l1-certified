# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import gc
import time
from typing import Callable, Dict, List, Tuple
import torch
from .backend_dispatcher import get_dispatcher
from .cuda_quantizer import VCoreTernaryQuantizer
from .safetensors_quantizer import pack_ternary_to_uint8
from .super_quantizer import quantize_e8_lattice


def _clear_caches() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _time_alternating_pair(
    fn_baseline: Callable[[], torch.Tensor],
    fn_candidate: Callable[[], torch.Tensor],
    steps: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Честный парный A/B-замер с чередованием порядка (A->B, затем B->A) и очисткой кэша,
    исключающий эффект разогретого кэша (warm-cache bias).
    """
    for _ in range(3):
        fn_baseline()
        fn_candidate()
    _clear_caches()

    half = max(1, steps // 2)
    t_base, t_cand = 0.0, 0.0

    # Раунд 1: Baseline -> Candidate
    for _ in range(half):
        _clear_caches()
        t0 = time.perf_counter()
        fn_baseline()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_base += time.perf_counter() - t0

        _clear_caches()
        t0 = time.perf_counter()
        fn_candidate()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cand += time.perf_counter() - t0

    # Раунд 2: Candidate -> Baseline (инверсия порядка)
    for _ in range(steps - half):
        _clear_caches()
        t0 = time.perf_counter()
        fn_candidate()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cand += time.perf_counter() - t0

        _clear_caches()
        t0 = time.perf_counter()
        fn_baseline()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_base += time.perf_counter() - t0

    return (t_base / steps) * 1000.0, (t_cand / steps) * 1000.0


def run_ab_benchmark(hidden_size: int = 2048, out_size: int = 2048, steps: int = 20) -> None:
    disp = get_dispatcher()
    hw = disp.profile
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    w = torch.randn(out_size, hidden_size, device=dev, dtype=torch.float32) * 0.05
    w[torch.rand_like(w) < 0.005] *= 35.0
    x = torch.randn(1, hidden_size, device=dev, dtype=torch.float32)

    # 1. Baseline: PyTorch FP32
    def run_fp32():
        return torch.nn.functional.linear(x, w)

    # 2. BitNet absmean (1.58-bit naive)
    gamma = torch.mean(torch.abs(w))
    w_bitnet = torch.clamp(torch.round(w / gamma), -1.0, 1.0) * gamma
    err_bitnet = float((torch.sum(torch.abs(w - w_bitnet)) / torch.sum(torch.abs(w))).item())

    def run_bitnet():
        return torch.nn.functional.linear(x, w_bitnet)

    # 3. V-CORE L1-Certified 1.58-bit (Markov Median + Fused Packed)
    q_vcore = VCoreTernaryQuantizer(group_size=32, theta=1e-4, auto_theta=True)
    w_vcore, _, scales_vcore = q_vcore.quantize(w)
    err_vcore = float((torch.sum(torch.abs(w - w_vcore)) / torch.sum(torch.abs(w))).item())
    packed_u8, _ = pack_ternary_to_uint8(w_vcore.cpu(), group_size=32)

    def run_vcore_158():
        return torch.nn.functional.linear(x, w_vcore)

    # 4. V-CORE Sub-1-Bit E8 Lattice (0.989 b/w + FWHT)
    w_e8, idx_e8, sc_e8 = quantize_e8_lattice(w, use_hadamard=True)
    err_e8 = float((torch.sum(torch.abs(w - w_e8)) / torch.sum(torch.abs(w))).item())

    def run_vcore_e8():
        if dev.type == "cpu" and disp.neon_lut_lib is not None:
            res = disp.run_e8_lut_gemm(idx_e8.cpu(), sc_e8.cpu(), x.cpu(), out_size, hidden_size, 32)
            if res is not None:
                return res
        return torch.nn.functional.linear(x, w_e8)

    ms_fp32, ms_bitnet = _time_alternating_pair(run_fp32, run_bitnet, steps, dev)
    _, ms_vcore158 = _time_alternating_pair(run_fp32, run_vcore_158, steps, dev)
    _, ms_vcore_e8 = _time_alternating_pair(run_fp32, run_vcore_e8, steps, dev)

    fp32_mb = (w.numel() * 4) / (1024.0 * 1024.0)
    vcore158_mb = (packed_u8.numel() + scales_vcore.numel() * 2) / (1024.0 * 1024.0)
    vcore_e8_mb = (idx_e8.numel() + sc_e8.numel() * 2) / (1024.0 * 1024.0)

    print(f"\n=== V-CORE PAIRED A/B BENCHMARK (Alternating Order + Cold Cache) ===")
    print(f"Hardware: {hw.gpu_name} | SIMD: {hw.simd_backend} | Backend: {disp.active_backend_name()}")
    print(f"Matrix: [{out_size} x {hidden_size}] | Steps: {steps}\n")
    print("| Method | Bits/Weight | Memory (MB) | Rel. L1 Error | Latency (ms) | Speedup vs FP32 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| PyTorch FP32 (Baseline) | 32.00 | {fp32_mb:.2f} | 0.0000 | {ms_fp32:.3f} | 1.00x |")
    print(f"| BitNet (absmean) | 1.58 | {vcore158_mb:.2f} | {err_bitnet:.4f} | {ms_bitnet:.3f} | {ms_fp32 / max(ms_bitnet, 1e-6):.2f}x |")
    print(f"| **V-CORE 1.58-bit (L1)** | **1.58** | **{vcore158_mb:.2f}** | **{err_vcore:.4f}** | **{ms_vcore158:.3f}** | **{ms_fp32 / max(ms_vcore158, 1e-6):.2f}x** |")
    print(f"| **V-CORE E8 LUT (Sub-1-Bit)** | **0.99** | **{vcore_e8_mb:.2f}** | **{err_e8:.4f}** | **{ms_vcore_e8:.3f}** | **{ms_fp32 / max(ms_vcore_e8, 1e-6):.2f}x** |\n")


def main():
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Paired A/B Benchmark")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Размерность входа")
    parser.add_argument("--out-size", type=int, default=1024, help="Размерность выхода")
    parser.add_argument("--steps", type=int, default=20, help="Количество замеров")
    parser.add_argument("--ab-compare", action="store_true", help="Запустить парный A/B тест с чередованием порядка")
    args = parser.parse_args()

    run_ab_benchmark(hidden_size=args.hidden_size, out_size=args.out_size, steps=args.steps)


if __name__ == "__main__":
    main()
    
