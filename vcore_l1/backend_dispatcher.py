# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import ctypes
import os
from typing import Optional
import torch
from .hardware_probe import probe_hardware


def _find_and_load_lib(lib_base: str) -> Optional[ctypes.CDLL]:
    exts = (".so", ".dylib", ".dll")
    dirs = [
        os.path.join(os.getcwd(), "build"),
        os.path.join(os.path.dirname(__file__), "..", "build"),
        os.path.dirname(__file__),
    ]
    for d in dirs:
        for ext in exts:
            path = os.path.join(d, f"{lib_base}{ext}")
            if os.path.isfile(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
    return None


class VCoreBackendDispatcher:
    """
    Единый C-ABI диспетчер вычислительных ядер:
    автоматически выбирает CUDA, AVX2, ARM NEON или PyTorch fallback по профилю железа.
    """

    def __init__(self):
        self.profile = probe_hardware()
        self.cuda_lib = _find_and_load_lib("libvcore_engine")
        self.cpu_lib = _find_and_load_lib("libvcore_cpu")
        self.neon_lut_lib = _find_and_load_lib("libvcore_lut_neon")
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        gemm_args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        if self.cuda_lib is not None and hasattr(self.cuda_lib, "run_vcore_fused_gemm"):
            self.cuda_lib.run_vcore_fused_gemm.argtypes = gemm_args
        if self.cpu_lib is not None and hasattr(self.cpu_lib, "vcore_cpu_fused_gemm"):
            self.cpu_lib.vcore_cpu_fused_gemm.argtypes = gemm_args
        if self.neon_lut_lib is not None:
            if hasattr(self.neon_lut_lib, "vcore_neon_fused_gemm"):
                self.neon_lut_lib.vcore_neon_fused_gemm.argtypes = gemm_args
            if hasattr(self.neon_lut_lib, "vcore_e8_lut_gemm"):
                self.neon_lut_lib.vcore_e8_lut_gemm.argtypes = gemm_args

    def active_backend_name(self) -> str:
        if self.profile.recommended_device == "cuda" and self.cuda_lib is not None:
            return f"CUDA-Fused-Kernel ({self.profile.gpu_name})"
        if self.profile.simd_backend == "ARM-NEON" and self.neon_lut_lib is not None:
            return "ARM-NEON-Native-SIMD"
        if self.cpu_lib is not None:
            return f"C++-{self.profile.simd_backend}-Native"
        return f"PyTorch-{self.profile.recommended_device.upper()}-Fallback"

    def run_e8_lut_gemm(
        self,
        e8_indices: torch.Tensor,
        scales: torch.Tensor,
        x_2d: torch.Tensor,
        out_features: int,
        in_features: int,
        group_size: int,
    ) -> Optional[torch.Tensor]:
        """Запускает нативный C++ / ARM NEON LUT-инференс для Sub-1-Bit E₈ весов."""
        if self.neon_lut_lib is None or x_2d.is_cuda or (in_features % 8 != 0) or (in_features % group_size != 0):
            return None
        batch_tokens = x_2d.size(0)
        x_f32 = x_2d.contiguous().to(torch.float32)
        sc_f32 = scales.contiguous().to(torch.float32)
        y_f32 = torch.empty((batch_tokens, out_features), dtype=torch.float32)
        self.neon_lut_lib.vcore_e8_lut_gemm(
            ctypes.c_void_p(e8_indices.contiguous().data_ptr()),
            ctypes.c_void_p(sc_f32.data_ptr()),
            ctypes.c_void_p(x_f32.data_ptr()),
            ctypes.c_void_p(y_f32.data_ptr()),
            batch_tokens,
            out_features,
            in_features,
            group_size,
        )
        return y_f32.to(x_2d.dtype)


_DISPATCHER: Optional[VCoreBackendDispatcher] = None


def get_dispatcher() -> VCoreBackendDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = VCoreBackendDispatcher()
    return _DISPATCHER
          
