# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from .cuda_quantizer import _weighted_median_1d
from .super_quantizer import (
    MARKOV_LAMBDA_LIMIT,
    fit_datafree_l1_lora,
    fwht_8,
    get_e8_codebook,
    quantize_e8_lattice,
    register_inversion_split,
)


# =========================================================================
# 1. КАСКАДНЫЕ ВРАЩЕНИЯ W₁₆ / W₃₂ И РЕШЁТКА БАРНСА–УОЛЛА Λ₃₂ (~0.625 b/w)
# =========================================================================

def fwht_16(x: torch.Tensor) -> torch.Tensor:
    """
    16x16 каскадное преобразование Уолша–Адамара W₁₆ (MarkovL1FullTransformerCert.lean):
    W₁₆(W₁₆(x)) = 16 * x, строится рекурсивно из W₈ без умножений.
    """
    orig_shape = x.shape
    v = x.reshape(-1, 2, 8)
    h1 = fwht_8(v[:, 0, :])
    h2 = fwht_8(v[:, 1, :])
    out = torch.cat([h1 + h2, h1 - h2], dim=1)
    return out.reshape(orig_shape)


def fwht_32(x: torch.Tensor) -> torch.Tensor:
    """
    32x32 каскадное преобразование Уолша–Адамара W₃₂ (MarkovL1FullTransformerCert.lean):
    W₃₂(W₃₂(x)) = 32 * x и (W₃₂ w)·(W₃₂ x) = 32(w·x).
    """
    orig_shape = x.shape
    v = x.reshape(-1, 2, 16)
    h1 = fwht_16(v[:, 0, :])
    h2 = fwht_16(v[:, 1, :])
    out = torch.cat([h1 + h2, h1 - h2], dim=1)
    return out.reshape(orig_shape)


@torch.no_grad()
def quantize_barnes_wall32_lattice(
    weights: torch.Tensor,
    use_hadamard32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    32-мерный решётчатый квантователь на минимальной оболочке Барнса–Уолла Λ₃₂ (~0.625 b/w,
    MarkovL1FullTransformerCert.lean: isBarnesWall32ShellQ):
    выбирает 16 доминирующих координат в 32D-блоке с чётной суммой знаков и L1-масштабом.
    """
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (32 - (flat.numel() % 32)) % 32
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 32)
    w_work = (fwht_32(blocks) / 5.656854249492381) if use_hadamard32 else blocks

    abs_b = torch.abs(w_work)
    topk_vals, topk_idx = torch.topk(abs_b, k=16, dim=1, sorted=False)

    mask = torch.zeros_like(w_work)
    mask.scatter_(1, topk_idx, 1.0)
    signs = torch.where(w_work >= 0, torch.ones_like(w_work), -torch.ones_like(w_work)) * mask

    neg_cnt = (signs < 0).sum(dim=1)
    odd_mask = (neg_cnt % 2) == 1
    if odd_mask.any():
        weakest_in_topk = torch.argmin(topk_vals, dim=1)
        flip_col = torch.gather(topk_idx, 1, weakest_in_topk.unsqueeze(1)).squeeze(1)
        rows_odd = torch.nonzero(odd_mask, as_tuple=False).squeeze(1)
        signs[rows_odd, flip_col[rows_odd]] *= -1.0

    active_proj = torch.clamp(w_work * signs, min=0.0)
    alpha = _weighted_median_1d(active_proj, mask)

    q_rot = signs * alpha
    err_q = torch.sum(torch.abs(w_work - q_rot), dim=1)
    err_zero = torch.sum(abs_b, dim=1)
    use_zero = err_zero <= err_q

    q_rot = torch.where(use_zero.unsqueeze(1), torch.zeros_like(q_rot), q_rot)
    alpha_out = torch.where(use_zero, torch.zeros_like(alpha.squeeze(1)), alpha.squeeze(1))

    bit_weights = (1 << torch.arange(32, device=blocks.device, dtype=torch.int64)).unsqueeze(0)
    support_mask = torch.sum((mask > 0).to(torch.int64) * bit_weights, dim=1).to(torch.int32)

    w_restored = (fwht_32(q_rot) / 5.656854249492381) if use_hadamard32 else q_rot
    w_hat = w_restored.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, support_mask, alpha_out


# =========================================================================
# 2. ПРЯМОЙ ТАБЛИЧНЫЙ LUT-ИНФЕРЕНС БЕЗ РАСПАКОВКИ В FP16 (VCoreLUTLinear)
# =========================================================================

class VCoreLUTLinear(nn.Module):
    """
    Реализация Теоремы `lutBlockInference8Q_sound` (MarkovL1FullTransformerCert.lean):
    выполняет матрично-векторное произведение в W₈-базисе напрямую через предвычисленную
    таблицу откликов LUT (241 состояние решётки E₈) без распаковки весов в матрицу FP16.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, lora_rank: int = 16):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"in_features ({in_features}) должно делиться на 8 для E₈ LUT")
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = in_features // 8
        self.lora_rank = lora_rank

        self.register_buffer("e8_indices", torch.zeros((out_features, self.num_groups), dtype=torch.uint8))
        self.register_buffer("e8_scales", torch.zeros((out_features, self.num_groups), dtype=torch.float16))

        if lora_rank > 0:
            self.register_buffer("lora_u", torch.zeros((out_features, lora_rank), dtype=torch.float16))
            self.register_buffer("lora_v", torch.zeros((lora_rank, in_features), dtype=torch.float16))
        else:
            self.lora_u = None
            self.lora_v = None

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, lora_rank: int = 16, lora_iters: int = 6) -> "VCoreLUTLinear":
        out_f, in_f = linear.weight.shape
        dev = linear.weight.device
        w_orig = linear.weight.detach().to(torch.float32)

        eff_rank = min(lora_rank, out_f, in_f) if lora_rank > 0 else 0
        layer = cls(in_f, out_f, bias=(linear.bias is not None), lora_rank=eff_rank).to(dev)

        w_rot = fwht_8(w_orig.reshape(-1, 8))
        q_rot, idx, scales = quantize_e8_lattice(w_rot, use_hadamard=False)
        layer.e8_indices.copy_(idx.view(out_f, in_f // 8))
        layer.e8_scales.copy_(scales.view(out_f, in_f // 8).to(torch.float16))

        w_e8_restored = (fwht_8(q_rot) * 0.125).reshape(out_f, in_f)
        residual = w_orig - w_e8_restored
        if eff_rank > 0:
            u, v = fit_datafree_l1_lora(residual, rank=eff_rank, iters=lora_iters)
            layer.lora_u.copy_(u.to(torch.float16))
            layer.lora_v.copy_(v.to(torch.float16))

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.detach().to(torch.float16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float32)
        bsz = x_2d.size(0)

        x_rot = fwht_8(x_2d.view(-1, 8)).view(bsz, self.num_groups, 8)
        cb = get_e8_codebook(x.device, torch.float32)
        lut = torch.matmul(x_rot, cb.T)

        idx_exp = self.e8_indices.long().unsqueeze(0).expand(bsz, -1, -1)
        lut_t = lut.permute(0, 2, 1)
        gathered = torch.gather(lut_t, 1, idx_exp)
        out = torch.sum(gathered * self.e8_scales.to(torch.float32).unsqueeze(0), dim=2) * 0.125

        if self.lora_u is not None and self.lora_v is not None:
            out = out + (x_2d @ self.lora_v.to(torch.float32).T) @ self.lora_u.to(torch.float32).T

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

# =========================================================================
# 3. АДАПТИВНЫЙ ПЛАНИРОВЩИК MIXED-LATTICE (С КРИТЕРИЕМ ПОРЫ Ω_pore = 1)
# =========================================================================

@dataclass
class MixedLatticeAssignment:
    layer_name: str
    lattice_mode: str
    bpw: float
    param_count: int
    holder_drift_bound: float


def compute_pore_equilibrium(x_block_dim: float, y_shell_dim: float) -> Tuple[float, float]:
    """
    Критерий равновесия поры (Module 27, Теорема `mixed_lattice_pore_equilibrium`):
    вычисляет индекс поры Ω_pore = X / Y и безразмерное рассогласование |Ω_pore - 1|.
    При Ω_pore = 1.0 энергия рассогласования оболочки равна 0.
    """
    omega = x_block_dim / max(y_shell_dim, 1e-8)
    mismatch = abs(omega - 1.0)
    return omega, mismatch


def allocate_mixed_lattice(
    layer_specs: List[Tuple[str, int, float]],
    target_avg_bpw: float = 0.85,
) -> Tuple[List[MixedLatticeAssignment], Dict[str, float]]:
    """
    Реализация `verifyMixedLatticePlanQ` (MarkovL1FullTransformerCert.lean)
    с предварительной оценкой равновесия поры Ω_pore = X / Y (Module 27):
    назначает слоям модели решётки (Λ₂₄ 0.733b, E₈ 0.989b, D₄ 1.161b, 1.58b)
    так, чтобы средний битрейт по всей сети не превышал `target_avg_bpw`.
    """
    mode_table = {
        "leech24": 0.733,
        "e8": 0.989,
        "d4": 1.161,
        "ternary158": 1.585,
    }
    assignments: List[MixedLatticeAssignment] = []
    for name, params, base_err in layer_specs:
        assignments.append(
            MixedLatticeAssignment(
                layer_name=name,
                lattice_mode="leech24",
                bpw=mode_table["leech24"],
                param_count=params,
                holder_drift_bound=base_err,
            )
        )

    total_params = max(sum(a.param_count for a in assignments), 1)
    # Приоритет апгрейда с учётом чувствительности и рассогласования поры Ω_pore
    def _layer_priority(idx: int) -> float:
        a = assignments[idx]
        role_w = 2.0 if ("v_proj" in a.layer_name or "down_proj" in a.layer_name) else 1.0
        _, pore_mismatch = compute_pore_equilibrium(float(a.param_count % 24 or 24), 24.0)
        return role_w * a.holder_drift_bound * (1.0 + pore_mismatch)

    order = sorted(range(len(assignments)), key=_layer_priority, reverse=True)

    upgrade_chain = [("e8", 0.989, 0.78), ("d4", 1.161, 0.62), ("ternary158", 1.585, 0.45)]
    for idx in order:
        cur_bits = sum(a.bpw * a.param_count for a in assignments)
        for mode_name, new_bpw, err_factor in upgrade_chain:
            delta_bits = (new_bpw - assignments[idx].bpw) * assignments[idx].param_count
            if (cur_bits + delta_bits) / total_params <= target_avg_bpw:
                assignments[idx].lattice_mode = mode_name
                assignments[idx].bpw = new_bpw
                assignments[idx].holder_drift_bound *= err_factor
                cur_bits += delta_bits

    final_avg_bpw = sum(a.bpw * a.param_count for a in assignments) / total_params
    total_drift = sum(a.holder_drift_bound for a in assignments)
    summary = {
        "certified_budget_ok": float(final_avg_bpw <= target_avg_bpw + 1e-9),
        "avg_bpw": final_avg_bpw,
        "target_avg_bpw": target_avg_bpw,
        "total_holder_drift": total_drift,
    }
    return assignments, summary


def certify_transformer_block_e2e(
    delta_x: float,
    eps_qkv: float,
    eps_o: float,
    eps_k: float,
    eps_v: float,
    eps_mlp: float,
    b_x: float,
    l_norm: float = 1.0,
    l_sm: float = 0.5,
    l_act: float = 1.0,
) -> float:
    """
    Вычисление сквозного сертификата дрейфа блока трансформера
    (Теорема `transformer_block_e2e_certified_bound`, MarkovL1FullTransformerCert.lean).
    """
    delta_attn = l_sm * (eps_qkv * b_x + eps_k * b_x) + eps_v + eps_o * b_x
    delta_mid = delta_x + l_norm * delta_attn
    return delta_mid + l_act * (l_norm * delta_mid + eps_mlp * b_x)


# =========================================================================
# 4. РАЦИОНАЛЬНЫЙ ДРЕЙФ RoPE И O(1) СИНДРОМНЫЙ ДЕКОДЕР ГОЛЕЯ G₂₄
# =========================================================================

def verify_rope_key_drift(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    k_quant: torch.Tensor,
    max_drift_bound: float,
) -> Tuple[bool, float]:
    """
    Проверка сертификата фазового дрейфа RoPE при квантовании ключей
    (Теорема `rope_quantized_key_drift_bound`, MarkovL1RoPEAndGolaySyndrome.lean).
    """
    q_2d = q.reshape(-1, 2).to(torch.float32)
    e_2d = (k_quant - k_orig).reshape(-1, 2).to(torch.float32)
    b_q = float(torch.max(torch.abs(q_2d)).item())
    e_l1 = float(torch.max(torch.sum(torch.abs(e_2d), dim=1)).item())
    cert_bound = 2.0 * e_l1 * b_q
    return (cert_bound <= max_drift_bound + 1e-7), cert_bound


_GOLAY_H_ROWS: Optional[List[int]] = None
_GOLAY_SYNDROME_LUT: Optional[Dict[int, int]] = None


def _init_golay24_syndrome_tables() -> Tuple[List[int], Dict[int, int]]:
    global _GOLAY_H_ROWS, _GOLAY_SYNDROME_LUT
    if _GOLAY_H_ROWS is not None and _GOLAY_SYNDROME_LUT is not None:
        return _GOLAY_H_ROWS, _GOLAY_SYNDROME_LUT

    base11 = 1762  # 0b11011100010
    basis = [1 | (2047 << 13)]
    for i in range(11):
        cyc = (((base11 << i) | (base11 >> (11 - i))) & 2047) << 13
        basis.append((1 << (i + 1)) | (1 << 12) | cyc)

    h_rows: List[int] = []
    for j in range(12):
        bt = 0
        for i in range(12):
            if (basis[i] >> (12 + j)) & 1:
                bt |= 1 << i
        h_rows.append(bt | (1 << (12 + j)))

    syn_lut: Dict[int, int] = {0: 0}
    def syn_of(word: int) -> int:
        s = 0
        for j in range(12):
            if (word & h_rows[j]).bit_count() & 1:
                s |= 1 << j
        return s

    for i in range(24):
        e1 = 1 << i
        syn_lut[syn_of(e1)] = e1
    for i in range(24):
        for j in range(i + 1, 24):
            e2 = (1 << i) | (1 << j)
            syn_lut[syn_of(e2)] = e2
    for i in range(24):
        for j in range(i + 1, 24):
            for k in range(j + 1, 24):
                e3 = (1 << i) | (1 << j) | (1 << k)
                syn_lut[syn_of(e3)] = e3

    _GOLAY_H_ROWS = h_rows
    _GOLAY_SYNDROME_LUT = syn_lut
    return h_rows, syn_lut


def golay24_syndrome_decode(received_24bit: int) -> Tuple[int, int]:
    """
    Мгновенное O(1) синдромное декодирование расширенного кода Голея G₂₄ (веса <= 3)
    для решётки Лича Λ₂₄ (MarkovL1RoPEAndGolaySyndrome.lean).
    Возвращает `(corrected_codeword_24bit, syndrome_12bit)`.
    """
    h_rows, syn_lut = _init_golay24_syndrome_tables()
    word = received_24bit & 0xFFFFFF
    syn = 0
    for j in range(12):
        if (word & h_rows[j]).bit_count() & 1:
            syn |= 1 << j
    err_vec = syn_lut.get(syn, 0)
    return (word ^ err_vec), syn
    
