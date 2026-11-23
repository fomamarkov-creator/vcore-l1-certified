# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import ctypes
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from .safetensors_quantizer import unpack_uint8_to_ternary


def _load_cuda_lib() -> Optional[ctypes.CDLL]:
    candidates = [
        os.path.join(os.getcwd(), "build", "libvcore_engine.so"),
        os.path.join(os.path.dirname(__file__), "..", "build", "libvcore_engine.so"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                lib = ctypes.CDLL(path)
                lib.run_vcore_fused_gemm.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                return lib
            except OSError:
                pass
    return None


_CUDA_LIB = _load_cuda_lib()


class VCoreLinear(nn.Module):
    """
    Drop-in замена для nn.Linear: хранит веса в сжатом 2-битном формате uint8
    и оптимальные Δ=1 медианные масштабы (float16).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        num_elems = out_features * in_features
        packed_len = (num_elems + 3) // 4
        num_blocks = (num_elems + group_size - 1) // group_size

        self.register_buffer("vcore_packed", torch.zeros(packed_len, dtype=torch.uint8))
        self.register_buffer("vcore_scales", torch.zeros(num_blocks, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_packed_tensors(
        cls,
        packed: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, int],
        group_size: int,
        bias: Optional[torch.Tensor] = None,
    ) -> "VCoreLinear":
        out_features, in_features = shape
        layer = cls(in_features, out_features, group_size=group_size, bias=(bias is not None))
        layer.vcore_packed.copy_(packed)
        layer.vcore_scales.copy_(scales.to(torch.float16))
        if bias is not None:
            layer.bias.copy_(bias.to(torch.float16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        batch_tokens = x_2d.shape[0]

        # Прямой Fused CUDA-инференс на GPU (без распаковки весов в VRAM!)
        if (
            _CUDA_LIB is not None
            and x_2d.is_cuda
            and (self.in_features % self.group_size == 0)
            and (self.in_features % 4 == 0)
        ):
            x_f32 = x_2d.contiguous().to(torch.float32)
            scales_f32 = self.vcore_scales.contiguous().to(torch.float32)
            y_f32 = torch.empty((batch_tokens, self.out_features), dtype=torch.float32, device=x.device)

            _CUDA_LIB.run_vcore_fused_gemm(
                ctypes.c_void_p(self.vcore_packed.data_ptr()),
                ctypes.c_void_p(scales_f32.data_ptr()),
                ctypes.c_void_p(x_f32.data_ptr()),
                ctypes.c_void_p(y_f32.data_ptr()),
                batch_tokens,
                self.out_features,
                self.in_features,
                self.group_size,
            )
            out = y_f32.to(x.dtype)
            if self.bias is not None:
                out = out + self.bias.to(x.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)

        # Быстрый векторный путь PyTorch (CPU или GPU без скомпилированного .so)
        mask = unpack_uint8_to_ternary(self.vcore_packed, (self.out_features, self.in_features))
        scales = self.vcore_scales.to(torch.float32)
        w_restored = (mask.reshape(-1, self.group_size) * scales.unsqueeze(1)).reshape(
            self.out_features, self.in_features
        ).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_restored, bias)


def inject_vcore_layers(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Автоматически заменяет все линейные слои nn.Linear в модели PyTorch / Hugging Face
    на сжатые слои VCoreLinear, загружая ключи *.vcore_packed и *.vcore_scales.
    """
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            w_key = f"{full_name}.weight"
            packed_key = f"{w_key}.vcore_packed"
            if packed_key in state_dict:
                packed = state_dict[packed_key]
                scales = state_dict[f"{w_key}.vcore_scales"]
                shape = tuple(state_dict[f"{w_key}.vcore_shape"].tolist())
                g_size = int(state_dict[f"{w_key}.vcore_group"].item())
                bias_tensor = state_dict.get(f"{full_name}.bias", None)

                vcore_layer = VCoreLinear.from_packed_tensors(
                    packed=packed,
                    scales=scales,
                    shape=(shape[0], shape[1]),
                    group_size=g_size,
                    bias=bias_tensor,
                )

                # Подменяем модуль в родительском контейнере
                parts = full_name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], vcore_layer)
    return model
