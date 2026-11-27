# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import torch


@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    cpu_model: str
    cpu_cores: int
    simd_backend: str          # "AVX-512", "AVX2", "ARM-NEON", "SCALAR"
    gpu_vendor: str            # "NVIDIA-CUDA", "APPLE-MPS", "AMD-ROCM", "INTEL-XPU", "NONE"
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_free_gb: float
    recommended_device: str
    recommended_vram_budget_gb: float
    recommended_spherical_emb: bool
    recommended_kv_mode: str   # "int4", "spherical", "ternary"
    recommended_integer_only: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_banner(self) -> str:
        gpu_part = (
            f"{self.gpu_name} ({self.vram_free_gb:.1f}/{self.vram_total_gb:.1f} GB VRAM)"
            if self.gpu_vendor != "NONE"
            else "No discrete GPU"
        )
        return (
            f"[{self.os_name} | {self.arch} | SIMD={self.simd_backend}] "
            f"GPU: {gpu_part} | RAM: {self.ram_free_gb:.1f}/{self.ram_total_gb:.1f} GB | "
            f"Auto-Profile: device={self.recommended_device}, budget={self.recommended_vram_budget_gb:.1f}GB, "
            f"spherical_emb={self.recommended_spherical_emb}, kv={self.recommended_kv_mode}"
        )


def _detect_cpu_simd(arch: str) -> str:
    arch_l = arch.lower()
    if any(k in arch_l for k in ("arm64", "aarch64", "armv8", "armv9")):
        return "ARM-NEON"
    if any(k in arch_l for k in ("x86_64", "amd64", "i686")):
        try:
            if os.path.isfile("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    flags = f.read().lower()
                if "avx512f" in flags:
                    return "AVX-512"
                if "avx2" in flags:
                    return "AVX2"
            elif platform.system() == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.features"], text=True).lower()
                if "avx2" in out:
                    return "AVX2"
        except Exception:
            pass
        return "AVX2"
    return "SCALAR"


def _detect_ram_gb() -> tuple[float, float]:
    try:
        if os.path.isfile("/proc/meminfo"):
            mem_tot, mem_avail = 0.0, 0.0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_tot = float(line.split()[1]) / (1024.0 * 1024.0)
                    elif line.startswith("MemAvailable:"):
                        mem_avail = float(line.split()[1]) / (1024.0 * 1024.0)
            if mem_tot > 0:
                return round(mem_tot, 2), round(mem_avail if mem_avail > 0 else mem_tot * 0.6, 2)
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            tot = (pages * page_size) / (1024.0 ** 3)
            return round(tot, 2), round(tot * 0.65, 2)
    except Exception:
        pass
    return 8.0, 4.0


def probe_hardware(preferred_device: Optional[str] = None) -> HardwareProfile:
    os_name = platform.system()
    arch = platform.machine() or "unknown"
    cpu_model = platform.processor() or arch
    cpu_cores = os.cpu_count() or 1
    simd_backend = _detect_cpu_simd(arch)
    ram_total_gb, ram_free_gb = _detect_ram_gb()

    gpu_vendor = "NONE"
    gpu_name = "CPU-Only"
    vram_total_gb = 0.0
    vram_free_gb = 0.0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        is_rocm = getattr(torch.version, "hip", None) is not None or "amd" in gpu_name.lower() or "radeon" in gpu_name.lower()
        gpu_vendor = "AMD-ROCM" if is_rocm else "NVIDIA-CUDA"
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_b / (1024.0 ** 3), 2)
            vram_total_gb = round(total_b / (1024.0 ** 3), 2)
        except Exception:
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(props.total_memory / (1024.0 ** 3), 2)
            vram_free_gb = round(vram_total_gb * 0.85, 2)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_vendor = "APPLE-MPS"
        gpu_name = "Apple Silicon Unified GPU"
        vram_total_gb = round(ram_total_gb * 0.70, 2)
        vram_free_gb = round(ram_free_gb * 0.75, 2)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        gpu_vendor = "INTEL-XPU"
        gpu_name = torch.xpu.get_device_name(0)
        vram_total_gb = 8.0
        vram_free_gb = 6.5

    if preferred_device:
        rec_dev = preferred_device
    elif gpu_vendor in ("NVIDIA-CUDA", "AMD-ROCM"):
        rec_dev = "cuda"
    elif gpu_vendor == "APPLE-MPS":
        rec_dev = "mps"
    elif gpu_vendor == "INTEL-XPU":
        rec_dev = "xpu"
    else:
        rec_dev = "cpu"

    avail_mem = vram_free_gb if rec_dev != "cpu" else ram_free_gb
    rec_budget = round(max(1.2, avail_mem * 0.82), 2)
    # Если памяти <= 8 ГБ, включаем сжатие эмбеддингов E8 и сферический KV-кэш
    low_mem = avail_mem <= 8.0
    rec_sph_emb = False
    rec_kv = "spherical" if low_mem else "int4"
    # На мобильных / слабых CPU без дискретной GPU рекомендуем Integer-Only Q14 режим
    rec_int_only = (rec_dev == "cpu" and simd_backend in ("ARM-NEON", "SCALAR"))

    return HardwareProfile(
        os_name=os_name,
        arch=arch,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        simd_backend=simd_backend,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        ram_total_gb=ram_total_gb,
        ram_free_gb=ram_free_gb,
        recommended_device=rec_dev,
        recommended_vram_budget_gb=rec_budget,
        recommended_spherical_emb=rec_sph_emb,
        recommended_kv_mode=rec_kv,
        recommended_integer_only=rec_int_only,
  )
  
