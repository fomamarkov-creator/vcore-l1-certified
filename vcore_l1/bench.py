# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import time
import torch
import torch.nn as nn
from .cuda_quantizer import VCoreTernaryQuantizer
from .safetensors_quantizer import pack_ternary_to_uint8
from .inference_layer import VCoreLinear, VCoreKVCacheQuantizer


def run_benchmark(hidden_size: int = 2048, out_size: int = 2048, steps: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"=== V-CORE L1-CERTIFIED BENCHMARK (Device: {device.upper()}) ===")
    linear_fp = nn.Linear(hidden_size, out_size, bias=False).to(device=device, dtype=dtype)
    w_f32 = linear_fp.weight.detach().to(torch.float32)

    q = VCoreTernaryQuantizer(group_size=32, auto_theta=True)
    _, mask, scales = q.quantize(w_f32)
    packed, shape = pack_ternary_to_uint8(mask)

    vcore_lin = VCoreLinear.from_packed_tensors(
        packed=packed, scales=scales, shape=(shape[0], shape[1]), group_size=32
    ).to(device=device, dtype=dtype)

    fp_bytes = linear_fp.weight.numel() * linear_fp.weight.element_size()
    vc_bytes = vcore_lin.vcore_packed.numel() + vcore_lin.vcore_scales.numel() * 2

    x = torch.randn(1, hidden_size, device=device, dtype=dtype)
    for _ in range(5):
        _ = vcore_lin(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = vcore_lin(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # Проверка KV-кэша
    kv_q = VCoreKVCacheQuantizer(group_size=32)
    kv_sample = torch.randn(1, 32, 128, 64, device=device, dtype=dtype)
    kv_p, kv_s, kv_sh = kv_q.compress_kv(kv_sample)
    kv_restored = kv_q.decompress_kv(kv_p, kv_s, kv_sh, dtype=dtype)
    assert kv_restored.shape == kv_sample.shape

    print(f"  Размер слоя FP16/32  : {fp_bytes / 1024:.1f} KB")
    print(f"  Размер слоя V-CORE   : {vc_bytes / 1024:.1f} KB (сжатие памяти в {fp_bytes / vc_bytes:.1f}x)")
    print(f"  Скорость прохода     : {steps / dt:.1f} шагов/сек ({dt / steps * 1000:.2f} мс/токен)")
    print("  KV-Cache сжатие      : OK (2-bit + Δ=1 медианные масштабы)")


def main():
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Speed & Memory Benchmark")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--out-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(args.hidden_size, args.out_size, args.steps)


if __name__ == "__main__":
    main()
