# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import ctypes
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from .cuda_quantizer import VCoreTernaryQuantizer
from .safetensors_quantizer import (
    pack_ternary_to_uint8,
    unpack_uint8_to_ternary,
    pack_int4_to_uint8,
    unpack_uint8_to_int4,
)


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


def _load_cpu_lib() -> Optional[ctypes.CDLL]:
    candidates = [
        os.path.join(os.getcwd(), "build", "libvcore_cpu.so"),
        os.path.join(os.path.dirname(__file__), "..", "build", "libvcore_cpu.so"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                lib = ctypes.CDLL(path)
                lib.vcore_cpu_fused_gemm.argtypes = [
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
_CPU_LIB = _load_cpu_lib()


class VCoreKVCacheQuantizer:
    """Сжатие KV-кэша на лету в 4-битный (или 2-битный) формат с L1-медианными масштабами."""
    def __init__(self, group_size: int = 32, theta: float = 1e-4, bits: int = 4):
        self.group_size = group_size
        self.bits = bits
        self.quantizer = VCoreTernaryQuantizer(group_size=group_size, theta=theta, auto_theta=True)

    @torch.no_grad()
    def compress_kv(self, kv_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        orig_shape = tuple(kv_tensor.shape)
        flat = kv_tensor.reshape(-1).to(torch.float32)
        rem = flat.numel() % self.group_size
        if rem != 0:
            flat = torch.cat([flat, torch.zeros(self.group_size - rem, dtype=flat.dtype, device=flat.device)])

        if self.bits == 2:
            _, mask, scales = self.quantizer.quantize(flat)
            packed, _ = pack_ternary_to_uint8(mask)
            return packed, scales.to(torch.float16), orig_shape

        blocks = flat.view(-1, self.group_size)
        abs_b = torch.abs(blocks)
        med_b = torch.median(abs_b, dim=1, keepdim=True).values
        max_b = torch.max(abs_b, dim=1, keepdim=True).values
        step = torch.clamp(torch.maximum(med_b / 2.5, max_b / 7.0), min=1e-8)
        q_int = torch.clamp(torch.round(blocks / step), -7.0, 7.0)
        packed, _ = pack_int4_to_uint8(q_int.reshape(-1))
        return packed, step.reshape(-1).to(torch.float16), orig_shape

    @torch.no_grad()
    def decompress_kv(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        total_elems = 1
        for d in orig_shape:
            total_elems *= d
        padded_len = ((total_elems + self.group_size - 1) // self.group_size) * self.group_size
        if self.bits == 2:
            grid = unpack_uint8_to_ternary(packed, (padded_len,)).to(packed.device)
        else:
            grid = unpack_uint8_to_int4(packed, (padded_len,)).to(packed.device)
        restored = (grid.reshape(-1, self.group_size) * scales.to(torch.float32).unsqueeze(1)).reshape(-1)
        return restored[:total_elems].reshape(orig_shape).to(dtype)


class VCoreDynamicCache:
    """
    Совместимый с Hugging Face transformers (v4 и v5+) кэш генерации,
    сжимающий историю ключей и значений на лету с L1-медианными масштабами.
    """
    is_compileable = False

    def __init__(self, group_size: int = 32, bits: int = 4):
        self.kv_q = VCoreKVCacheQuantizer(group_size=group_size, bits=bits)
        self.key_cache: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.dtype]] = []
        self.value_cache: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.dtype]] = []
        self.layers: List[Any] = []
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.size(-2)

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append((*self.kv_q.compress_kv(key_states), key_states.dtype))
            self.value_cache.append((*self.kv_q.compress_kv(value_states), value_states.dtype))
            return key_states, value_states

        k_prev = self.kv_q.decompress_kv(*self.key_cache[layer_idx])
        v_prev = self.kv_q.decompress_kv(*self.value_cache[layer_idx])
        k_full = torch.cat([k_prev, key_states], dim=-2)
        v_full = torch.cat([v_prev, value_states], dim=-2)

        self.key_cache[layer_idx] = (*self.kv_q.compress_kv(k_full), key_states.dtype)
        self.value_cache[layer_idx] = (*self.kv_q.compress_kv(v_full), value_states.dtype)
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> Tuple[int, int]:
        query_length = cache_position if isinstance(cache_position, int) else int(cache_position.shape)
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0


class VCoreLinear(nn.Module):
    """
    Drop-in замена для nn.Linear (2-bit / 4-bit + Δ=1 перестановка каналов с выбросами).
    Использует CUDA-ядра на GPU и нативный AVX2 C++ движок (libvcore_cpu.so) на CPU.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        bits: int = 2,
        bias: bool = False,
        has_perm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits

        num_elems = out_features * in_features
        packed_len = (num_elems + 3) // 4 if bits == 2 else (num_elems + 1) // 2
        num_blocks = (num_elems + group_size - 1) // group_size

        self.register_buffer("vcore_packed", torch.zeros(packed_len, dtype=torch.uint8))
        self.register_buffer("vcore_scales", torch.zeros(num_blocks, dtype=torch.float16))
        if has_perm:
            self.register_buffer("vcore_perm", torch.arange(in_features, dtype=torch.int64))
        else:
            self.vcore_perm = None

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
        bits: int = 2,
        bias: Optional[torch.Tensor] = None,
        perm: Optional[torch.Tensor] = None,
    ) -> "VCoreLinear":
        out_features, in_features = shape
        layer = cls(
            in_features,
            out_features,
            group_size=group_size,
            bits=bits,
            bias=(bias is not None),
            has_perm=(perm is not None),
        )
        layer.vcore_packed.copy_(packed)
        layer.vcore_scales.copy_(scales.to(torch.float16))
        if perm is not None:
            layer.vcore_perm.copy_(perm.to(torch.int64))
        if bias is not None:
            layer.bias.copy_(bias.to(torch.float16))
        return layer
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            x_2d = x.reshape(-1, self.in_features)
            if self.vcore_perm is not None:
                x_2d = x_2d[:, self.vcore_perm]
            batch_tokens = x_2d.size(0)

        # 1. Быстрый путь на GPU (CUDA Warp GEMV / Shared-Memory Tiled GEMM)
        if (
            self.bits == 2
            and _CUDA_LIB is not None
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

        # 2. Быстрый путь на CPU (нативный AVX2 SIMD без распаковки в float32)
        if (
            self.bits == 2
            and _CPU_LIB is not None
            and (not x_2d.is_cuda)
            and (self.in_features % self.group_size == 0)
            and (self.in_features % 4 == 0)
        ):
            x_f32 = x_2d.contiguous().to(torch.float32)
            scales_f32 = self.vcore_scales.contiguous().to(torch.float32)
            y_f32 = torch.empty((batch_tokens, self.out_features), dtype=torch.float32, device=x.device)

            _CPU_LIB.vcore_cpu_fused_gemm(
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

        # 3. Универсальный PyTorch фоллбэк (INT4 каскадные слои или нестандартный размер)
        num_elems = self.out_features * self.in_features
        padded_elems = ((num_elems + self.group_size - 1) // self.group_size) * self.group_size
        if self.bits == 2:
            grid = unpack_uint8_to_ternary(self.vcore_packed, (padded_elems,)).to(x.device)
        else:
            grid = unpack_uint8_to_int4(self.vcore_packed, (padded_elems,)).to(x.device)

        scales = self.vcore_scales.to(torch.float32)
        w_restored = (grid.reshape(-1, self.group_size) * scales.unsqueeze(1)).reshape(-1)[:num_elems].reshape(
            self.out_features, self.in_features
        )
        bias_f32 = self.bias.to(torch.float32) if self.bias is not None else None
        out = F.linear(x_2d.to(torch.float32), w_restored, bias_f32).to(x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)


def inject_vcore_layers(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            w_key = f"{full_name}.weight"
            packed_key = f"{w_key}.vcore_packed"
            if packed_key in state_dict:
                packed = state_dict[packed_key]
                scales = state_dict[f"{w_key}.vcore_scales"]
                out_f, in_f = map(int, state_dict[f"{w_key}.vcore_shape"].tolist())
                g_size = int(state_dict[f"{w_key}.vcore_group"].item())
                bits = int(state_dict[f"{w_key}.vcore_bits"].item()) if f"{w_key}.vcore_bits" in state_dict else 2
                perm = state_dict.get(f"{w_key}.vcore_perm", None)
                bias_tensor = state_dict.get(f"{full_name}.bias", None)

                vcore_layer = VCoreLinear.from_packed_tensors(
                    packed=packed,
                    scales=scales,
                    shape=(out_f, in_f),
                    group_size=g_size,
                    bits=bits,
                    bias=bias_tensor,
                    perm=perm,
                )
                parts = full_name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], vcore_layer)

    non_packed = {k: v for k, v in state_dict.items() if ".vcore_" not in k}
    model.load_state_dict(non_packed, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def load_vcore_causal_lm(model_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float16):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {model_dir} не найдены .safetensors файлы")

    merged_sd: Dict[str, torch.Tensor] = {}
    for sf in shard_files:
        merged_sd.update(load_file(sf))

    model = inject_vcore_layers(model, merged_sd)
    return model.to(device=device, dtype=dtype).eval()


class VCoreModelForCausalLM:
    @staticmethod
    def from_vcore_pretrained(
        model_dir: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        return load_vcore_causal_lm(model_dir=model_dir, device=device, dtype=dtype)
        
