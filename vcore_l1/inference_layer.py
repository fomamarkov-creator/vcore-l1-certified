# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import ctypes
import glob
import os
from contextlib import nullcontext
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
from .super_quantizer import MARKOV_LAMBDA_LIMIT, MARKOV_LE0_GAP
from .v5_engine import (
    integer_only_rmsnorm,
    pack_spherical_code,
    unpack_spherical_code,
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
    """
    Сжатие KV-кэша на лету: 4-bit (INT4), 2-bit (Ternary) или сферические коды E₈ ('spherical')
    с динамическим Электрическим Замком (Module 27: `kv_cache_electric_lock_sound`).
    """

    def __init__(
        self,
        group_size: int = 32,
        theta: float = 1e-4,
        bits: int = 4,
        mode: str = "auto",
        electric_lock: bool = True,
    ):
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.electric_lock = electric_lock
        self.quantizer = VCoreTernaryQuantizer(group_size=group_size, theta=theta, auto_theta=True)

    @staticmethod
    def _apply_electric_lock_stabilizer(kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Применяет прерыватель давления P_in(ρ) = (256 - ρ)/(ρ + 1):
        когда P_in <= P_ext (насыщение плотности активаций), зазор клиренса обнуляется,
        мягко демпфируя флуктуации на длинном контексте без потери ортогональности ключей GQA.
        """
        xf = kv_tensor.to(torch.float32)
        max_val = torch.clamp(torch.max(torch.abs(xf)), min=1e-6)
        rho = (torch.abs(xf) / max_val) * MARKOV_LAMBDA_LIMIT
        p_in = (MARKOV_LAMBDA_LIMIT - rho) / (rho + 1.0)
        p_ext = 1.0
        locked = p_in <= p_ext
        damped = xf * (1.0 - MARKOV_LE0_GAP)
        stabilized = torch.where(locked, damped, xf)
        return stabilized.to(kv_tensor.dtype)

    @torch.no_grad()
    def compress_kv(self, kv_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        orig_shape = tuple(kv_tensor.shape)
        work_kv = self._apply_electric_lock_stabilizer(kv_tensor) if self.electric_lock else kv_tensor

        if self.mode == "spherical":
            sp_idx, sp_rad, shape, _ = pack_spherical_code(
                work_kv, lattice="e8", hyperbolic_poincare=False
            )
            return sp_idx, sp_rad, shape

        flat = work_kv.reshape(-1).to(torch.float32)
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
        if self.mode == "spherical":
            return unpack_spherical_code(packed, scales, orig_shape, lattice="e8", dtype=dtype)

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
    поддерживающий INT4, 2-bit L1, сферические коды E₈ (`mode='spherical'`)
    и динамический Электрический Замок (`electric_lock=True`).
    """
    is_compileable = False

    def __init__(self, group_size: int = 32, bits: int = 4, mode: str = "auto", electric_lock: bool = True):
        self.kv_q = VCoreKVCacheQuantizer(group_size=group_size, bits=bits, mode=mode, electric_lock=electric_lock)
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
class VCoreSphericalEmbedding(nn.Module):
    """
    Сжатый слой nn.Embedding на гиперболических сферических кодах E₈:
    хранит словарь в формате (uint8 idx, fp16 radius) (~1.0 b/w) и распаковывает
    на лету только запрошенные токены `input_ids` без аллокации всей FP16 таблицы.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim % 8 != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) должно делиться на 8 для E₈")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.groups_per_row = embedding_dim // 8

        self.register_buffer("vcore_sph_idx", torch.zeros((num_embeddings, self.groups_per_row), dtype=torch.uint8))
        self.register_buffer("vcore_sph_rad", torch.zeros((num_embeddings, self.groups_per_row), dtype=torch.float16))

    @classmethod
    def from_packed(cls, idx: torch.Tensor, rad: torch.Tensor, shape: Tuple[int, int]) -> "VCoreSphericalEmbedding":
        num_emb, emb_dim = shape
        layer = cls(num_emb, emb_dim)
        layer.vcore_sph_idx.copy_(idx.view(num_emb, emb_dim // 8))
        layer.vcore_sph_rad.copy_(rad.view(num_emb, emb_dim // 8).to(torch.float16))
        return layer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sel_idx = self.vcore_sph_idx[input_ids]
        sel_rad = self.vcore_sph_rad[input_ids]
        out_shape = (*input_ids.shape, self.embedding_dim)
        return unpack_spherical_code(sel_idx, sel_rad, out_shape, lattice="e8", dtype=torch.float16)


class VCoreSphericalLinear(nn.Module):
    """Сжатый выходной слой lm_head на гиперболических сферических кодах E₈."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        num_blocks = (out_features * in_features + 7) // 8
        self.register_buffer("vcore_sph_idx", torch.zeros(num_blocks, dtype=torch.uint8))
        self.register_buffer("vcore_sph_rad", torch.zeros(num_blocks, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_packed(
        cls, idx: torch.Tensor, rad: torch.Tensor, shape: Tuple[int, int], bias: Optional[torch.Tensor] = None
    ) -> "VCoreSphericalLinear":
        out_f, in_f = shape
        layer = cls(in_f, out_f, bias=(bias is not None))
        layer.vcore_sph_idx.copy_(idx)
        layer.vcore_sph_rad.copy_(rad.to(torch.float16))
        if bias is not None:
            layer.bias.copy_(bias.to(torch.float16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = unpack_spherical_code(
            self.vcore_sph_idx, self.vcore_sph_rad, (self.out_features, self.in_features), lattice="e8", dtype=x.dtype
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class VCoreIntegerRMSNorm(nn.Module):
    """Бесплюючая (Integer-Only) замена RMSNorm на базе `fixedPointInvSqrtStepZ` (Q14)."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return integer_only_rmsnorm(x, self.weight, eps=self.eps, newton_iters=4, frac_bits=14)


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

        if not hasattr(self, "_w_cache") or self._w_cache is None or self._w_cache.device != x.device:
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
            self._w_cache = w_restored.to(x.dtype)

        bias_cast = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x_2d, self._w_cache, bias_cast)
        return out.reshape(*orig_shape[:-1], self.out_features)
def inject_vcore_layers(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    integer_only: bool = False,
) -> nn.Module:
    for full_name, module in list(model.named_modules()):
        # 1. Подмена сжатых линейных слоёв (1.58-bit / INT4 или Spherical E₈ lm_head)
        if isinstance(module, nn.Linear):
            w_key = f"{full_name}.weight"
            packed_key = f"{w_key}.vcore_packed"
            sph_key = f"{w_key}.vcore_sph_idx"

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

            elif sph_key in state_dict:
                sp_idx = state_dict[sph_key]
                sp_rad = state_dict[f"{w_key}.vcore_sph_rad"]
                out_f, in_f = map(int, state_dict[f"{w_key}.vcore_shape"].tolist())
                bias_tensor = state_dict.get(f"{full_name}.bias", None)

                sph_lin = VCoreSphericalLinear.from_packed(
                    idx=sp_idx, rad=sp_rad, shape=(out_f, in_f), bias=bias_tensor
                )
                parts = full_name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], sph_lin)

        # 2. Подмена сжатого словаря эмбеддингов (embed_tokens) на VCoreSphericalEmbedding
        elif isinstance(module, nn.Embedding):
            w_key = f"{full_name}.weight"
            sph_key = f"{w_key}.vcore_sph_idx"
            if sph_key in state_dict:
                sp_idx = state_dict[sph_key]
                sp_rad = state_dict[f"{w_key}.vcore_sph_rad"]
                num_emb, emb_dim = map(int, state_dict[f"{w_key}.vcore_shape"].tolist())

                sph_emb = VCoreSphericalEmbedding.from_packed(
                    idx=sp_idx, rad=sp_rad, shape=(num_emb, emb_dim)
                )
                parts = full_name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], sph_emb)

        # 3. В режиме integer_only подменяем RMSNorm на целочисленный VCoreIntegerRMSNorm
        elif integer_only and hasattr(module, "weight") and "RMSNorm" in type(module).__name__:
            eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            int_norm = VCoreIntegerRMSNorm(module.weight, eps=eps)
            parts = full_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], int_norm)

    non_packed = {k: v for k, v in state_dict.items() if ".vcore_" not in k}
    model.load_state_dict(non_packed, strict=False, assign=True)
    if hasattr(model, "tie_weights") and not any(
        isinstance(m, (VCoreSphericalEmbedding, VCoreSphericalLinear)) for m in model.modules()
    ):
        model.tie_weights()
    return model


def load_vcore_causal_lm(
    model_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    integer_only: bool = False,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    temp_dir = None
    if os.path.isfile(model_dir) and str(model_dir).endswith(".vcore"):
        from .vcore_bundle import unpack_vcore_bundle_to_temp
        temp_dir, *_ = unpack_vcore_bundle_to_temp(model_dir)
        model_dir = temp_dir.name

    try:
        from accelerate import init_empty_weights
        ctx = init_empty_weights()
    except ImportError:
        ctx = nullcontext()

    config = AutoConfig.from_pretrained(model_dir)
    with ctx:
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {model_dir} не найдены .safetensors файлы")

    merged_sd: Dict[str, torch.Tensor] = {}
    for sf in shard_files:
        merged_sd.update(load_file(sf))

    model = inject_vcore_layers(model, merged_sd, integer_only=integer_only)
    if any(p.is_meta for p in model.parameters()):
        model = model.to_empty(device=device)
        for mod in model.modules():
            if hasattr(mod, "rotary_emb") and hasattr(mod.rotary_emb, "inv_freq"):
                if mod.rotary_emb.inv_freq.is_meta:
                    mod.rotary_emb.__init__(config=config, device=device)
    else:
        model = model.to(device=device)

    model = model.to(dtype=dtype).eval()
    if temp_dir is not None:
        model._vcore_temp_dir = temp_dir
    return model


class VCoreModelForCausalLM:
    @staticmethod
    def from_vcore_pretrained(
        model_dir: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        integer_only: bool = False,
    ):
        return load_vcore_causal_lm(
            model_dir=model_dir, device=device, dtype=dtype, integer_only=integer_only
            )
                
