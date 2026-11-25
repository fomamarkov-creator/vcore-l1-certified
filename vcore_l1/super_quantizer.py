# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cuda_quantizer import _weighted_median_1d

# Фундаментальные рациональные инварианты Маркова (MarkovL1HCPAndBirkhoffQuantizer.lean)
MARKOV_LAMBDA_LIMIT = 256.0
MARKOV_N_BASE = 250.0
MARKOV_ZETA_RATIO = MARKOV_LAMBDA_LIMIT / MARKOV_N_BASE  # 1.024
MARKOV_LE0_GAP = MARKOV_ZETA_RATIO - 1.0                 # 0.024 (3/125)


# =========================================================================
# 1. ОРТОГОНАЛЬНОЕ ПРЕОБРАЗОВАНИЕ УОЛША–АДАМАРА H₄ И W₈ (БЕЗ УМНОЖЕНИЙ)
# =========================================================================

def fwht_4(x: torch.Tensor) -> torch.Tensor:
    """
    Инволютивное 4x4 преобразование Уолша–Адамара H₄ (MarkovL1NextGenQuantizer.lean):
    H₄(H₄(x)) = x, выравнивает выбросы по каналам без иррациональных коэффициентов.
    """
    orig_shape = x.shape
    v = x.reshape(-1, 4)
    x0, x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    y0 = (x0 + x1 + x2 + x3) * 0.5
    y1 = (x0 - x1 + x2 - x3) * 0.5
    y2 = (x0 + x1 - x2 - x3) * 0.5
    y3 = (x0 - x1 - x2 + x3) * 0.5
    return torch.stack([y0, y1, y2, y3], dim=1).reshape(orig_shape)


def fwht_8(x: torch.Tensor) -> torch.Tensor:
    """
    Ненормированное 8x8 преобразование Уолша–Адамара W₈ (MarkovL1UltimateE8Engine.lean):
    W₈(W₈(x)) = 8 * x, использует только сложения и вычитания.
    """
    orig_shape = x.shape
    v = x.reshape(-1, 8)
    a0 = v[:, 0] + v[:, 1]
    a1 = v[:, 0] - v[:, 1]
    a2 = v[:, 2] + v[:, 3]
    a3 = v[:, 2] - v[:, 3]
    a4 = v[:, 4] + v[:, 5]
    a5 = v[:, 4] - v[:, 5]
    a6 = v[:, 6] + v[:, 7]
    a7 = v[:, 6] - v[:, 7]

    b0 = a0 + a2
    b1 = a1 + a3
    b2 = a0 - a2
    b3 = a1 - a3
    b4 = a4 + a6
    b5 = a5 + a7
    b6 = a4 - a6
    b7 = a5 - a7

    out = torch.stack([
        b0 + b4, b1 + b5, b2 + b6, b3 + b7,
        b0 - b4, b1 - b5, b2 - b6, b3 - b7,
    ], dim=1)
    return out.reshape(orig_shape)


# =========================================================================
# 2. СЛОВАРИ ВЕКТОРНЫХ РЕШЁТОК: 3HCP (13-pt), D₄ (25-pt) И E₈ (241-pt)
# =========================================================================

_CACHE_3HCP: Dict[torch.device, torch.Tensor] = {}
_CACHE_D4: Dict[torch.device, torch.Tensor] = {}
_CACHE_E8: Dict[torch.device, torch.Tensor] = {}


def get_3hcp_codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """13-элементный словарь 3D-решётки 3HCP (~1.233 бита/вес, MarkovL1HCPAndBirkhoffQuantizer.lean)."""
    if device not in _CACHE_3HCP:
        pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0],
        ]
        _CACHE_3HCP[device] = torch.tensor(pts, device=device, dtype=torch.float32)
    return _CACHE_3HCP[device].to(dtype)


def get_d4_codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """25-элементный словарь 4D-решётки D₄ (~1.161 бита/вес, MarkovL1SuperQuantizer.lean)."""
    if device not in _CACHE_D4:
        pts = [[0.0, 0.0, 0.0, 0.0]]
        for i in range(4):
            for j in range(i + 1, 4):
                for s1 in (-1.0, 1.0):
                    for s2 in (-1.0, 1.0):
                        v = [0.0, 0.0, 0.0, 0.0]
                        v[i] = s1
                        v[j] = s2
                        pts.append(v)
        _CACHE_D4[device] = torch.tensor(pts, device=device, dtype=torch.float32)
    return _CACHE_D4[device].to(dtype)


def get_e8_codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """241-элементный словарь 8D-решётки E₈ (~0.989 бита/вес, MarkovL1UltimateE8Engine.lean)."""
    if device not in _CACHE_E8:
        pts = [[0.0] * 8]
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in (-1.0, 1.0):
                    for s2 in (-1.0, 1.0):
                        v = [0.0] * 8
                        v[i] = s1
                        v[j] = s2
                        pts.append(v)
        for mask in range(256):
            if bin(mask).count("1") % 2 == 0:
                v = [-0.5 if ((mask >> k) & 1) else 0.5 for k in range(8)]
                pts.append(v)
        _CACHE_E8[device] = torch.tensor(pts, device=device, dtype=torch.float32)
    return _CACHE_E8[device].to(dtype)
def _nonzero_block_median(abs_blocks: torch.Tensor) -> torch.Tensor:
    """Точная медиана ненулевых модулей в каждом блоке."""
    active = (abs_blocks > 1e-8).to(abs_blocks.dtype)
    scales = _weighted_median_1d(abs_blocks, active)
    return torch.clamp(scales, min=1e-8)


@torch.no_grad()
def quantize_3hcp_lattice(weights: torch.Tensor, chunk_size: int = 65536) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Векторное квантование троек весов в 13-точечную решётку 3HCP (~1.23 b/w)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (3 - (flat.numel() % 3)) % 3
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 3)
    scales = _nonzero_block_median(torch.abs(blocks))
    norm_blocks = blocks / scales
    cb = get_3hcp_codebook(blocks.device, torch.float32)

    n = blocks.size(0)
    best_idx = torch.empty((n,), device=blocks.device, dtype=torch.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(norm_blocks[start:end], cb, p=1)
        best_idx[start:end] = torch.argmin(dists, dim=1).to(torch.uint8)

    w_hat = (cb[best_idx.long()] * scales).reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales.reshape(-1)


@torch.no_grad()
def quantize_d4_lattice(weights: torch.Tensor, chunk_size: int = 65536) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Векторное квантование четвёрок весов в 25-точечную решётку D₄ (~1.16 b/w)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (4 - (flat.numel() % 4)) % 4
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 4)
    scales = _nonzero_block_median(torch.abs(blocks))
    norm_blocks = blocks / scales
    cb = get_d4_codebook(blocks.device, torch.float32)

    n = blocks.size(0)
    best_idx = torch.empty((n,), device=blocks.device, dtype=torch.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(norm_blocks[start:end], cb, p=1)
        best_idx[start:end] = torch.argmin(dists, dim=1).to(torch.uint8)

    w_hat = (cb[best_idx.long()] * scales).reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales.reshape(-1)


@torch.no_grad()
def quantize_e8_lattice(
    weights: torch.Tensor,
    use_hadamard: bool = True,
    chunk_size: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Суб-1-битное векторное квантование восьмёрок весов в 241-точечную решётку E₈ (~0.989 b/w).
    Каждый индекс E₈ упаковывается ровно в 1 байт uint8 (8 весов -> 8 бит = 1.0 b/w).
    """
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (8 - (flat.numel() % 8)) % 8
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 8)
    w_work = fwht_8(blocks) / 2.8284271247461903 if use_hadamard else blocks
    scales = _nonzero_block_median(torch.abs(w_work))
    norm_blocks = w_work / scales
    cb = get_e8_codebook(blocks.device, torch.float32)

    n = blocks.size(0)
    best_idx = torch.empty((n,), device=blocks.device, dtype=torch.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(norm_blocks[start:end], cb, p=1)
        best_idx[start:end] = torch.argmin(dists, dim=1).to(torch.uint8)

    q_rot = cb[best_idx.long()] * scales
    w_restored = fwht_8(q_rot) / 2.8284271247461903 if use_hadamard else q_rot
    w_hat = w_restored.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales.reshape(-1)


@torch.no_grad()
def quantize_asym_ternary(weights: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Асимметричный квантователь {-α_neg, 0, +α_pos} (MarkovL1NextGenQuantizer.lean)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, group_size)
    pos_mask = (blocks > 0).to(blocks.dtype)
    neg_mask = (blocks < 0).to(blocks.dtype)

    alpha_pos = _weighted_median_1d(torch.clamp(blocks, min=0.0), pos_mask)
    alpha_neg = _weighted_median_1d(torch.clamp(-blocks, min=0.0), neg_mask)

    q_pos = torch.where(blocks > (alpha_pos * 0.5), alpha_pos, torch.zeros_like(blocks))
    q_neg = torch.where((-blocks) > (alpha_neg * 0.5), -alpha_neg, torch.zeros_like(blocks))
    q = torch.where(blocks > 0, q_pos, torch.where(blocks < 0, q_neg, torch.zeros_like(blocks)))
    return q.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)


@torch.no_grad()
def quantize_residual_2stage(weights: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Двухступенчатый остаточный тернарный каскад (3.16 b/w без умножений)."""
    w1 = quantize_asym_ternary(weights, group_size=group_size)
    res = weights - w1
    w2 = quantize_asym_ternary(res, group_size=group_size)
    return (w1 + w2).to(weights.dtype)


class VCoreSuperQuantizer(nn.Module):
    """
    Адаптивный самосертифицирующийся суперквантователь (MarkovL1SuperQuantizer.lean):
    H₄ Адамар -> D₄ (1.16 b/w) -> Asym 1.58 b/w -> Residual 3.16 b/w.
    """

    def __init__(self, rel_error_budget: float = 0.25):
        super().__init__()
        self.rel_error_budget = rel_error_budget

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        orig_shape = weights.shape
        flat = weights.reshape(-1).to(torch.float32)
        pad = (4 - (flat.numel() % 4)) % 4
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

        blocks = flat.view(-1, 4)
        w_rot = fwht_4(blocks)
        block_l1 = torch.sum(torch.abs(w_rot), dim=1)
        eps = block_l1 * self.rel_error_budget

        q_d4, _, _ = quantize_d4_lattice(w_rot)
        err_d4 = torch.sum(torch.abs(w_rot - q_d4), dim=1)
        use_d4 = err_d4 <= eps

        q_asym = quantize_asym_ternary(w_rot, group_size=4)
        err_asym = torch.sum(torch.abs(w_rot - q_asym), dim=1)
        use_asym = (~use_d4) & (err_asym <= eps)

        q_res = quantize_residual_2stage(w_rot, group_size=4)
        use_res = (~use_d4) & (~use_asym)

        q_rot = torch.where(
            use_d4.unsqueeze(1),
            q_d4,
            torch.where(use_asym.unsqueeze(1), q_asym, q_res),
        )
        w_restored = fwht_4(q_rot).reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)

        n_blk = max(blocks.size(0), 1)
        frac_d4 = float(use_d4.sum().item()) / n_blk
        frac_asym = float(use_asym.sum().item()) / n_blk
        frac_res = float(use_res.sum().item()) / n_blk
        avg_bits = frac_d4 * 1.161 + frac_asym * 1.585 + frac_res * 3.170
        cert_max_err = float((2.0 * torch.sum(torch.abs(w_rot - q_rot))).item())

        stats = {
            "avg_bits_per_weight": avg_bits,
            "frac_d4_1_16b": frac_d4,
            "frac_asym_1_58b": frac_asym,
            "frac_residual_3_16b": frac_res,
            "certified_max_l1_err": cert_max_err,
            "actual_l1_err": float(torch.sum(torch.abs(weights.to(torch.float32) - w_restored.to(torch.float32))).item()),
        }
        return w_restored, stats
# =========================================================================
# 5. ИНФЕРЕНС В БАЗИСЕ УОЛША–АДАМАРА W₈ БЕЗ ОБРАТНОГО ПРЕОБРАЗОВАНИЯ ВЕСОВ
# =========================================================================

class VCoreInDomainHadamardLinear(nn.Module):
    """
    Реализация теоремы `in_domain_hadamard_exact_dot` (MarkovL1AttentionKVCache.lean):
    (W₈ w) · (W₈ x) = 8 · (w · x).
    Хранит веса E₈ непосредственно в повёрнутом пространстве Адамара W₈.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"in_features ({in_features}) должно делиться на 8 для W₈-базиса")
        self.in_features = in_features
        self.out_features = out_features
        num_blocks = (out_features * in_features) // 8
        self.register_buffer("e8_indices", torch.zeros(num_blocks, dtype=torch.uint8))
        self.register_buffer("e8_scales", torch.zeros(num_blocks, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "VCoreInDomainHadamardLinear":
        out_f, in_f = linear.weight.shape
        layer = cls(in_f, out_f, bias=(linear.bias is not None)).to(linear.weight.device)
        w_blocks = linear.weight.detach().to(torch.float32).reshape(-1, 8)
        w_rot = fwht_8(w_blocks)
        _, idx, scales = quantize_e8_lattice(w_rot, use_hadamard=False)
        layer.e8_indices.copy_(idx)
        layer.e8_scales.copy_(scales.to(torch.float16))
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.detach().to(torch.float16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float32)
        x_rot = fwht_8(x_2d.view(-1, 8)).reshape(-1, self.in_features)

        cb = get_e8_codebook(x.device, torch.float32)
        w_rot_hat = (cb[self.e8_indices.long()] * self.e8_scales.to(torch.float32).unsqueeze(1)).reshape(
            self.out_features, self.in_features
        )
        out = F.linear(x_rot, w_rot_hat) * 0.125
        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)


# =========================================================================
# 6. РЕГИСТРОВАЯ ИНВЕРСИЯ, 2:4 РАЗРЕЖЕННОСТЬ, DATA-FREE ℓ₁-LORA, MoE И FREIVALDS
# =========================================================================

@torch.no_grad()
def register_inversion_split(
    weights: torch.Tensor,
    limit: float = MARKOV_LAMBDA_LIMIT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Изолирует выбросы |w| >= limit в разреженный регистр, обнуляя основной канал."""
    is_outlier = torch.abs(weights) >= limit
    main_channel = torch.where(is_outlier, torch.zeros_like(weights), weights)
    outlier_register = torch.where(is_outlier, weights, torch.zeros_like(weights))
    return main_channel, outlier_register


@torch.no_grad()
def quantize_24_sparse_l1(weights: torch.Tensor) -> torch.Tensor:
    """Аппаратная 2:4 структурная разреженность в метрике L1 (MarkovL1EntropyAndSparsity.lean)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (4 - (flat.numel() % 4)) % 4
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 4)
    masks_list = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    masks = torch.tensor(masks_list, device=blocks.device, dtype=torch.float32)
    abs_b = torch.abs(blocks)
    signs = torch.sign(blocks)

    best_err = torch.full((blocks.size(0),), float("inf"), device=blocks.device, dtype=torch.float32)
    best_q = torch.zeros_like(blocks)

    for m_idx in range(6):
        m = masks[m_idx].unsqueeze(0)
        alpha = torch.sum(abs_b * m, dim=1, keepdim=True) * 0.5
        cand = signs * m * alpha
        err = torch.sum(torch.abs(blocks - cand), dim=1)
        imp = err < best_err
        best_err = torch.where(imp, err, best_err)
        best_q = torch.where(imp.unsqueeze(1), cand, best_q)

    return best_q.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)


@torch.no_grad()
def fit_datafree_l1_lora(
    residual: torch.Tensor,
    rank: int = 8,
    iters: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Data-Free малоранговая L1-компенсация остатка E ≈ U @ V^T (MarkovL1LeechAndLowRank.lean)."""
    E = residual.detach().to(torch.float32)
    out_f, in_f = E.shape
    eff_rank = min(rank, out_f, in_f)
    U = torch.zeros((out_f, eff_rank), device=E.device, dtype=torch.float32)
    V = torch.zeros((eff_rank, in_f), device=E.device, dtype=torch.float32)

    cur_res = E.clone()
    for r in range(eff_rank):
        row_norms = torch.sum(torch.abs(cur_res), dim=1)
        v_row = cur_res[int(torch.argmax(row_norms).item()), :].clone()
        if torch.max(torch.abs(v_row)) < 1e-8:
            break
        v_row = v_row / torch.max(torch.abs(v_row))

        u_r = torch.zeros((out_f,), device=E.device, dtype=torch.float32)
        for _ in range(iters):
            w_v = torch.abs(v_row).unsqueeze(0).expand(out_f, in_f)
            safe_v = torch.where(w_v > 1e-8, v_row.unsqueeze(0), torch.ones_like(w_v))
            ratios_u = torch.where(w_v > 1e-8, cur_res / safe_v, torch.zeros_like(cur_res))
            u_r = _weighted_median_1d(ratios_u, w_v).reshape(-1)
            if torch.max(torch.abs(u_r)) < 1e-8:
                break

            w_u = torch.abs(u_r).unsqueeze(0).expand(in_f, out_f)
            res_t = cur_res.T
            safe_u = torch.where(w_u > 1e-8, u_r.unsqueeze(0), torch.ones_like(w_u))
            ratios_v = torch.where(w_u > 1e-8, res_t / safe_u, torch.zeros_like(res_t))
            v_row = _weighted_median_1d(ratios_v, w_u).reshape(-1)
            if torch.max(torch.abs(v_row)) < 1e-8:
                break

        U[:, r] = u_r
        V[r, :] = v_row
        cur_res = cur_res - torch.outer(u_r, v_row)

    return U.to(residual.dtype), V.to(residual.dtype)


@torch.no_grad()
def verify_moe_routing_safety(
    gate_logits: torch.Tensor,
    logit_error_bounds: torch.Tensor,
    top_k: int = 2,
) -> bool:
    """Проверка инвариантности Top-K маршрутизации MoE (MarkovL1MoERoutingAndDynamicBits.lean)."""
    g = gate_logits.reshape(-1).to(torch.float32)
    d = logit_error_bounds.reshape(-1).to(torch.float32)
    if top_k <= 0 or top_k >= g.numel():
        return True

    sorted_idx = torch.argsort(g, descending=True)
    sel_idx = sorted_idx[:top_k]
    rej_idx = sorted_idx[top_k:]

    margins = g[sel_idx].unsqueeze(1) - g[rej_idx].unsqueeze(0)
    thresholds = d[sel_idx].unsqueeze(1) + d[rej_idx].unsqueeze(0)
    return bool(torch.all(margins > thresholds).item())


@torch.no_grad()
def verify_matmul_freivalds(
    A: torch.Tensor,
    B: torch.Tensor,
    C_claimed: torch.Tensor,
    num_trials: int = 4,
    atol: float = 1e-3,
) -> bool:
    """Быстрый O(n²) верификатор Фрейвалдса (MarkovL1ZeroKnowledgeInferenceCert.lean)."""
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C_claimed.to(torch.float32)
    n_cols = B_f.size(1)

    for _ in range(num_trials):
        r = torch.randint(0, 2, (n_cols, 1), device=A_f.device, dtype=torch.float32) * 2.0 - 1.0
        abr = A_f @ (B_f @ r)
        cr = C_f @ r
        if not torch.allclose(abr, cr, atol=atol, rtol=1e-3):
            return False
    return True
