# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Dict, List, Optional, Tuple
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
# 1. ОРТОГОНАЛЬНЫЕ ПРЕОБРАЗОВАНИЯ УОЛША–АДАМАРА H₄ И W₈ (БЕЗ УМНОЖЕНИЙ)
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
# 2. СЛОВАРИ ВЕКТОРНЫХ РЕШЁТОК: 3HCP (13-pt), D₄ (25-pt), E₈ (241-pt) И GOLAY G₂₄
# =========================================================================

_CACHE_3HCP: Dict[torch.device, torch.Tensor] = {}
_CACHE_D4: Dict[torch.device, torch.Tensor] = {}
_CACHE_E8: Dict[torch.device, torch.Tensor] = {}
_CACHE_GOLAY_SHELLS: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}


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


def get_golay24_shells(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует все 759 октад (вес 8) и 759 кооктад (вес 16) расширенного кода Голея G₂₄
    по порождающей матрице G = [I₁₂ | B₁₂] (MarkovL1LeechAndLowRank.lean).
    """
    if device not in _CACHE_GOLAY_SHELLS:
        qr_pal = 0b11011100010  # 1762_10
        basis: List[int] = []
        row0 = (1 << 23) | (qr_pal << 1)
        basis.append(row0)
        for shift in range(11):
            cyc = ((qr_pal << shift) | (qr_pal >> (11 - shift))) & 0x7FF
            row = (1 << (22 - shift)) | (cyc << 1) | 1
            basis.append(row)

        octads: List[List[int]] = []
        cooctads: List[List[int]] = []
        for m in range(1, 4096):
            cw = 0
            for bit in range(12):
                if (m >> bit) & 1:
                    cw ^= basis[bit]
            wt = cw.bit_count()
            if wt == 8:
                octads.append([23 - pos for pos in range(24) if (cw >> pos) & 1])
            elif wt == 16:
                cooctads.append([23 - pos for pos in range(24) if (cw >> pos) & 1])
        _CACHE_GOLAY_SHELLS[device] = (
            torch.tensor(octads, device=device, dtype=torch.long),
            torch.tensor(cooctads, device=device, dtype=torch.long),
        )
    return _CACHE_GOLAY_SHELLS[device]
    # =========================================================================
# 3. ТОЧНЫЙ L1-ПОИСК МАСШТАБА И ВЕКТОРНЫЕ КВАНТОВАТЕЛИ РЕШЁТОК (3HCP, D₄, E₈, Λ₂₄)
# =========================================================================

def _nonzero_block_median(abs_blocks: torch.Tensor) -> torch.Tensor:
    """Точная медиана ненулевых модулей в каждом блоке."""
    active = (abs_blocks > 1e-8).to(abs_blocks.dtype)
    scales = _weighted_median_1d(abs_blocks, active)
    return torch.clamp(scales, min=1e-8)


def _exact_l1_lattice_search(
    blocks: torch.Tensor,
    cb: torch.Tensor,
    scale_multipliers: Tuple[float, ...] = (1.0,),
    chunk_size: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Точный поиск точки решётки z* ∈ cb и оптимального L1-масштаба α* > 0,
    минимизирующих ||b - α * z||_1 (устраняет расхождение между спинорными ±0.5 и целыми ±1.0 корнями).
    """
    n, _ = blocks.shape
    base_scales = _nonzero_block_median(torch.abs(blocks))  # (n, 1)
    best_idx = torch.zeros((n,), device=blocks.device, dtype=torch.uint8)
    best_scale = torch.zeros((n, 1), device=blocks.device, dtype=torch.float32)
    w_hat = torch.zeros_like(blocks)

    cb_abs = torch.abs(cb)  # (M, d)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        b_chunk = blocks[start:end]      # (B, d)
        s_base = base_scales[start:end]  # (B, 1)

        min_err = torch.full((end - start,), float("inf"), device=blocks.device, dtype=torch.float32)
        chunk_idx = torch.zeros((end - start,), device=blocks.device, dtype=torch.long)

        for mult in scale_multipliers:
            norm_b = b_chunk / (s_base * mult)
            dists = torch.cdist(norm_b, cb, p=1) * (s_base * mult)
            cur_err, cur_idx = torch.min(dists, dim=1)
            imp = cur_err < min_err
            min_err = torch.where(imp, cur_err, min_err)
            chunk_idx = torch.where(imp, cur_idx, chunk_idx)

        # Точное аналитическое уточнение масштаба α*(z*) = weighted_median(|b_i| / |z*_i|, weights=|z*_i|)
        z_sel = cb[chunk_idx]            # (B, d)
        z_abs = cb_abs[chunk_idx]        # (B, d)
        active = z_abs > 1e-8
        safe_z = torch.where(active, z_abs, torch.ones_like(z_abs))
        ratios = torch.where(active, torch.abs(b_chunk) / safe_z, torch.zeros_like(b_chunk))
        weights = torch.where(active, z_abs, torch.zeros_like(z_abs))
        alpha_opt = _weighted_median_1d(ratios, weights)  # (B, 1)

        q_cand = z_sel * alpha_opt
        err_cand = torch.sum(torch.abs(b_chunk - q_cand), dim=1)
        err_zero = torch.sum(torch.abs(b_chunk), dim=1)
        use_zero = err_zero <= err_cand

        final_idx = torch.where(use_zero, torch.zeros_like(chunk_idx), chunk_idx)
        final_alpha = torch.where(use_zero.unsqueeze(1), torch.zeros_like(alpha_opt), alpha_opt)

        best_idx[start:end] = final_idx.to(torch.uint8)
        best_scale[start:end] = final_alpha
        w_hat[start:end] = cb[final_idx] * final_alpha

    return w_hat, best_idx, best_scale.reshape(-1)


@torch.no_grad()
def quantize_3hcp_lattice(weights: torch.Tensor, chunk_size: int = 65536) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Векторное квантование троек весов в 13-точечную решётку 3HCP (~1.233 b/w)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (3 - (flat.numel() % 3)) % 3
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 3)
    cb = get_3hcp_codebook(blocks.device, torch.float32)
    w_q, best_idx, scales = _exact_l1_lattice_search(blocks, cb, scale_multipliers=(1.0,), chunk_size=chunk_size)
    w_hat = w_q.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales


@torch.no_grad()
def quantize_d4_lattice(weights: torch.Tensor, chunk_size: int = 65536) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Векторное квантование четвёрок весов в 25-точечную решётку D₄ (~1.161 b/w)."""
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (4 - (flat.numel() % 4)) % 4
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 4)
    cb = get_d4_codebook(blocks.device, torch.float32)
    w_q, best_idx, scales = _exact_l1_lattice_search(blocks, cb, scale_multipliers=(1.0,), chunk_size=chunk_size)
    w_hat = w_q.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales


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
    cb = get_e8_codebook(blocks.device, torch.float32)

    q_rot, best_idx, scales = _exact_l1_lattice_search(
        w_work, cb, scale_multipliers=(1.0, 2.0), chunk_size=chunk_size
    )
    w_restored = fwht_8(q_rot) / 2.8284271247461903 if use_hadamard else q_rot
    w_hat = w_restored.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_idx, scales


def _eval_golay_shell(b_chunk: torch.Tensor, shell_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Оценивает проекцию 24D блока на заданную оболочку кода Голея (октады 8 или кооктады 16)."""
    gathered_abs = torch.abs(b_chunk)[:, shell_pos]           # (B, 759, k)
    sel_idx = torch.argmax(gathered_abs.sum(dim=-1), dim=1)   # (B,)
    sel_pos = shell_pos[sel_idx]                              # (B, k)

    vals = torch.gather(b_chunk, 1, sel_pos)                  # (B, k)
    signs = torch.where(vals >= 0, torch.ones_like(vals), -torch.ones_like(vals))

    # Чётность отрицательных знаков (isLeech24OctadShell)
    odd_mask = ((signs < 0).sum(dim=1) % 2) == 1
    if odd_mask.any():
        min_idx = torch.argmin(torch.abs(vals), dim=1)
        signs[odd_mask, min_idx[odd_mask]] *= -1.0

    signed_proj = vals * signs
    alpha = _weighted_median_1d(torch.clamp(signed_proj, min=0.0), torch.ones_like(vals))  # (B, 1)

    cand_block = torch.zeros_like(b_chunk)
    cand_block.scatter_(1, sel_pos, signs * alpha)
    err = torch.sum(torch.abs(b_chunk - cand_block), dim=1)
    return cand_block, err, alpha.squeeze(1)


@torch.no_grad()
def quantize_leech24_lattice(
    weights: torch.Tensor,
    use_hadamard: bool = True,
    chunk_size: int = 16384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Суб-1-битное векторное квантование 24-векторов в решётку Лича Λ₂₄ (~0.733 b/w)
    по двум оболочкам кода Голея G₂₄: 759 октад (вес 8) + 759 кооктад (вес 16).
    """
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (24 - (flat.numel() % 24)) % 24
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, 24)
    if use_hadamard:
        w_work = (fwht_8(blocks.view(-1, 8)) / 2.8284271247461903).view(-1, 24)
    else:
        w_work = blocks

    octads, cooctads = get_golay24_shells(blocks.device)
    n = blocks.size(0)
    best_shell = torch.zeros((n,), device=blocks.device, dtype=torch.int16)
    best_scale = torch.zeros((n,), device=blocks.device, dtype=torch.float32)
    q_rot = torch.zeros_like(w_work)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        b_chunk = w_work[start:end]

        cand_oct, err_oct, alpha_oct = _eval_golay_shell(b_chunk, octads)
        cand_coo, err_coo, alpha_coo = _eval_golay_shell(b_chunk, cooctads)

        use_coo = err_coo < err_oct
        cand_best = torch.where(use_coo.unsqueeze(1), cand_coo, cand_oct)
        err_best = torch.where(use_coo, err_coo, err_oct)
        alpha_best = torch.where(use_coo, alpha_coo, alpha_oct)

        err_zero = torch.sum(torch.abs(b_chunk), dim=1)
        use_zero = err_zero <= err_best

        q_rot[start:end] = torch.where(use_zero.unsqueeze(1), torch.zeros_like(cand_best), cand_best)
        best_shell[start:end] = torch.where(
            use_zero,
            torch.full_like(use_coo, -1, dtype=torch.int16),
            torch.where(use_coo, torch.ones_like(use_coo, dtype=torch.int16), torch.zeros_like(use_coo, dtype=torch.int16)),
        )
        best_scale[start:end] = torch.where(use_zero, torch.zeros_like(alpha_best), alpha_best)

    if use_hadamard:
        w_restored = (fwht_8(q_rot.view(-1, 8)) / 2.8284271247461903).view(-1, 24)
    else:
        w_restored = q_rot

    w_hat = w_restored.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)
    return w_hat, best_shell, best_scale
    # =========================================================================
# 4. ТОЧНЫЙ КРИТИЧЕСКИЙ ТЕРНАРНЫЙ СОЛВЕР, АСИММЕТРИЧНЫЙ И СУПЕРКВАНТОВАТЕЛЬ
# =========================================================================

@torch.no_grad()
def quantize_ternary_exact_blockwise(weights: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Точный O(g log g) тернарный L1-квантователь по критическим порогам θ* ∈ {0} ∪ {|w_i|}
    (MarkovL1TernaryQuantizer.lean, Corollary 3.3).
    """
    orig_shape = weights.shape
    flat = weights.reshape(-1).to(torch.float32)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, group_size)
    abs_b = torch.abs(blocks)
    signs = torch.sign(blocks)

    best_err = torch.sum(abs_b, dim=1)
    best_q = torch.zeros_like(blocks)

    sorted_abs, _ = torch.sort(abs_b, dim=1)
    thresholds = torch.cat([torch.zeros((blocks.size(0), 1), device=blocks.device), sorted_abs], dim=1)

    for k in range(group_size + 1):
        theta = thresholds[:, k : k + 1]
        active = (abs_b > theta).to(blocks.dtype)
        alpha = _weighted_median_1d(abs_b, active)
        cand = signs * active * alpha
        err = torch.sum(torch.abs(blocks - cand), dim=1)
        imp = err < best_err
        best_err = torch.where(imp, err, best_err)
        best_q = torch.where(imp.unsqueeze(1), cand, best_q)

    return best_q.reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)


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
    """Двухступенчатый остаточный тернарный каскад (3.170 b/w без умножений)."""
    w1 = quantize_ternary_exact_blockwise(weights, group_size=group_size)
    res = weights - w1
    w2 = quantize_ternary_exact_blockwise(res, group_size=group_size)
    return (w1 + w2).to(weights.dtype)


class VCoreSuperQuantizer(nn.Module):
    """
    Адаптивный самосертифицирующийся суперквантователь (MarkovL1SuperQuantizer.lean):
    H₄ Адамар -> D₄ (1.161 b/w) -> Exact Critical 1.585 b/w -> Residual 3.170 b/w.
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

        q_crit = quantize_ternary_exact_blockwise(w_rot, group_size=4)
        err_crit = torch.sum(torch.abs(w_rot - q_crit), dim=1)
        use_crit = (~use_d4) & (err_crit <= eps)

        q_res = quantize_residual_2stage(w_rot, group_size=4)
        use_res = (~use_d4) & (~use_crit)

        q_rot = torch.where(
            use_d4.unsqueeze(1),
            q_d4,
            torch.where(use_crit.unsqueeze(1), q_crit, q_res),
        )
        w_restored = fwht_4(q_rot).reshape(-1)[: weights.numel()].reshape(orig_shape).to(weights.dtype)

        n_blk = max(blocks.size(0), 1)
        frac_d4 = float(use_d4.sum().item()) / n_blk
        frac_crit = float(use_crit.sum().item()) / n_blk
        frac_res = float(use_res.sum().item()) / n_blk
        avg_bits = frac_d4 * 1.161 + frac_crit * 1.585 + frac_res * 3.170
        cert_max_err = float((2.0 * torch.sum(torch.abs(w_rot - q_rot))).item())

        stats = {
            "avg_bits_per_weight": avg_bits,
            "frac_d4_1_16b": frac_d4,
            "frac_critical_1_58b": frac_crit,
            "frac_residual_3_16b": frac_res,
            "certified_max_l1_err": cert_max_err,
            "actual_l1_err": float(torch.sum(torch.abs(weights.to(torch.float32) - w_restored.to(torch.float32))).item()),
        }
        return w_restored, stats
    # =========================================================================
# 5. ПОЛНЫЙ КОНВЕЙЕР ALGORITHM 1: IN-DOMAIN W₈ + E₈ + l1-LoRA + РЕГИСТР ВЫБРОСОВ
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


class VCoreInDomainHadamardLinear(nn.Module):
    """
    Реализация Algorithm 1 и Теоремы 5.1 (in_domain_hadamard_exact_dot):
    (W₈ w) · (W₈ x) = 8 · (w · x).
    Объединяет:
      1) Регистровую инверсию выбросов |w| >= limit;
      2) Хранение весов E₈ (uint8 + fp16 scale) непосредственно в W₈-базисе;
      3) Data-Free малоранговую L1-компенсацию остатка U @ V;
      4) Эмиссию сертификата максимальной строчной L1-ошибки ε_final.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lora_rank: int = 0,
    ):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"in_features ({in_features}) должно делиться на 8 для W₈-базиса")
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        num_blocks = (out_features * in_features) // 8

        self.register_buffer("e8_indices", torch.zeros(num_blocks, dtype=torch.uint8))
        self.register_buffer("e8_scales", torch.zeros(num_blocks, dtype=torch.float16))
        self.register_buffer("cert_row_l1_bound", torch.tensor(0.0, dtype=torch.float32))

        if lora_rank > 0:
            self.register_buffer("lora_u", torch.zeros((out_features, lora_rank), dtype=torch.float16))
            self.register_buffer("lora_v", torch.zeros((lora_rank, in_features), dtype=torch.float16))
        else:
            self.lora_u = None
            self.lora_v = None

        self.register_buffer("outlier_rows", torch.empty((0,), dtype=torch.long))
        self.register_buffer("outlier_cols", torch.empty((0,), dtype=torch.long))
        self.register_buffer("outlier_values", torch.empty((0,), dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        lora_rank: int = 16,
        lora_iters: int = 6,
        outlier_limit: float = MARKOV_LAMBDA_LIMIT,
    ) -> "VCoreInDomainHadamardLinear":
        out_f, in_f = linear.weight.shape
        dev = linear.weight.device
        w_orig = linear.weight.detach().to(torch.float32)

        # Шаг 1: Регистровая инверсия выбросов
        w_main, w_reg = register_inversion_split(w_orig, limit=outlier_limit)

        eff_rank = min(lora_rank, out_f, in_f) if lora_rank > 0 else 0
        layer = cls(in_f, out_f, bias=(linear.bias is not None), lora_rank=eff_rank).to(dev)

        # Сохраняем разреженный регистр выбросов для быстрого index_add_ без sparse_coo предупреждений
        nz = torch.nonzero(w_reg, as_tuple=False)
        if nz.numel() > 0:
            layer.outlier_rows = nz[:, 0].contiguous()
            layer.outlier_cols = nz[:, 1].contiguous()
            layer.outlier_values = w_reg[layer.outlier_rows, layer.outlier_cols].to(torch.float16)

        # Шаг 2-3: Поворот W₈ и проекция на решётку E₈ в W₈-базисе
        w_rot = fwht_8(w_main.reshape(-1, 8))
        q_rot, idx, scales = quantize_e8_lattice(w_rot, use_hadamard=False)
        layer.e8_indices.copy_(idx)
        layer.e8_scales.copy_(scales.to(torch.float16))

        # Шаг 4: Остаток E = W_main - (1/8) W₈(W_rot_hat) и Data-Free l1-LoRA
        w_e8_restored = (fwht_8(q_rot) * 0.125).reshape(out_f, in_f)
        residual = w_main - w_e8_restored

        if eff_rank > 0:
            u, v = fit_datafree_l1_lora(residual, rank=eff_rank, iters=lora_iters)
            layer.lora_u.copy_(u.to(torch.float16))
            layer.lora_v.copy_(v.to(torch.float16))
            final_res = residual - (u.to(torch.float32) @ v.to(torch.float32))
        else:
            final_res = residual

        # Шаг 5: Эмиссия сертификата максимальной строчной L1-ошибки ε_final
        row_l1 = torch.sum(torch.abs(final_res), dim=1)
        layer.cert_row_l1_bound.copy_(torch.max(row_l1))

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

        if self.lora_u is not None and self.lora_v is not None:
            out = out + (x_2d @ self.lora_v.to(torch.float32).T) @ self.lora_u.to(torch.float32).T

        if self.outlier_values.numel() > 0:
            contrib = x_2d[:, self.outlier_cols] * self.outlier_values.to(torch.float32).unsqueeze(0)
            out.index_add_(1, self.outlier_rows, contrib)

        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
        return out.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)
        # =========================================================================
# 6. 2:4 РАЗРЕЖЕННОСТЬ, DATA-FREE ℓ₁-LORA (С BACKFITTING), ℓ₁-GPTQ, MoE И FREIVALDS
# =========================================================================

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


def _update_rank1_l1(target: torch.Tensor, v_init: torch.Tensor, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Вспомогательный шаг чередующейся L1-оптимизации ранга 1 по Теореме 6.2."""
    out_f, in_f = target.shape
    v_row = v_init.clone()
    v_max = torch.max(torch.abs(v_row))
    if v_max < 1e-8:
        return torch.zeros((out_f,), device=target.device, dtype=torch.float32), torch.zeros_like(v_row)
    v_row = v_row / v_max

    u_r = torch.zeros((out_f,), device=target.device, dtype=torch.float32)
    res_t = target.T
    for _ in range(iters):
        w_v = torch.abs(v_row).unsqueeze(0).expand(out_f, in_f)
        safe_v = torch.where(w_v > 1e-8, v_row.unsqueeze(0), torch.ones_like(w_v))
        ratios_u = torch.where(w_v > 1e-8, target / safe_v, torch.zeros_like(target))
        u_r = _weighted_median_1d(ratios_u, w_v).reshape(-1)
        if torch.max(torch.abs(u_r)) < 1e-8:
            break

        w_u = torch.abs(u_r).unsqueeze(0).expand(in_f, out_f)
        safe_u = torch.where(w_u > 1e-8, u_r.unsqueeze(0), torch.ones_like(w_u))
        ratios_v = torch.where(w_u > 1e-8, res_t / safe_u, torch.zeros_like(res_t))
        v_row = _weighted_median_1d(ratios_v, w_u).reshape(-1)
        v_norm = torch.max(torch.abs(v_row))
        if v_norm < 1e-8:
            break
        v_row = v_row / v_norm

    return u_r, v_row


@torch.no_grad()
def fit_datafree_l1_lora(
    residual: torch.Tensor,
    rank: int = 8,
    iters: int = 6,
    backfit_sweeps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Data-Free малоранговая L1-компенсация остатка E ≈ U @ V (MarkovL1LeechAndLowRank.lean)
    с жадной инициализацией и совместным координатным спуском (backfitting) по всем r рангам.
    """
    E = residual.detach().to(torch.float32)
    out_f, in_f = E.shape
    eff_rank = min(rank, out_f, in_f)
    U = torch.zeros((out_f, eff_rank), device=E.device, dtype=torch.float32)
    V = torch.zeros((eff_rank, in_f), device=E.device, dtype=torch.float32)

    cur_res = E.clone()
    active_ranks = 0
    for r in range(eff_rank):
        row_norms = torch.sum(torch.abs(cur_res), dim=1)
        v_init = cur_res[int(torch.argmax(row_norms).item()), :]
        if torch.max(torch.abs(v_init)) < 1e-8:
            break
        u_r, v_row = _update_rank1_l1(cur_res, v_init, iters=iters)
        U[:, r] = u_r
        V[r, :] = v_row
        cur_res = cur_res - torch.outer(u_r, v_row)
        active_ranks += 1

    # Совместная доводка (backfitting) по уже найденным компонентам
    for _ in range(backfit_sweeps):
        for r in range(active_ranks):
            comp_r = torch.outer(U[:, r], V[r, :])
            target_r = cur_res + comp_r
            u_r, v_row = _update_rank1_l1(target_r, V[r, :], iters=max(2, iters // 2))
            U[:, r] = u_r
            V[r, :] = v_row
            cur_res = target_r - torch.outer(u_r, v_row)

    return U.to(residual.dtype), V.to(residual.dtype)


@torch.no_grad()
def quantize_triangular_l1_gptq(
    weights: torch.Tensor,
    chol_r: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Последовательное L1-квантование с верхнетреугольной обратной связью по корреляции
    активаций R (R^T R ≈ X^T X / N) по Теореме 2.2 (MarkovL1TriangularActivation.lean).
    """
    w = weights.detach().to(torch.float32).clone()
    r_mat = chol_r.detach().to(torch.float32)
    _, in_f = w.shape

    for start in range(0, in_f, block_size):
        end = min(start + block_size, in_f)
        w_blk = w[:, start:end].clone()
        err_blk = torch.zeros_like(w_blk)

        for j in range(end - start):
            col = w_blk[:, j : j + 1]
            q_col = quantize_ternary_exact_blockwise(col, group_size=min(32, col.numel()))
            e_j = col - q_col
            w_blk[:, j : j + 1] = q_col
            err_blk[:, j : j + 1] = e_j

            if j + 1 < (end - start):
                r_jj = torch.clamp(r_mat[start + j, start + j].abs(), min=1e-6)
                r_j_rest = r_mat[start + j, start + j + 1 : end].unsqueeze(0)
                w_blk[:, j + 1 :] += e_j @ (r_j_rest / r_jj)

        w[:, start:end] = w_blk
        if end < in_f:
            r_diag = torch.clamp(torch.diag(r_mat)[start:end].abs(), min=1e-6).unsqueeze(1)
            r_cross = r_mat[start:end, end:] / r_diag
            w[:, end:] += err_blk @ r_cross

    return w.to(weights.dtype)


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


@torch.no_grad()
def quantize_model_vcore(
    model: nn.Module,
    lora_rank: int = 16,
    lora_iters: int = 6,
    min_numel: int = 50000,
    skip_names: Tuple[str, ...] = ("lm_head",),
) -> Dict[str, float]:
    """
    Заменяет подходящие слои nn.Linear модели на сертифицированные слои
    VCoreInDomainHadamardLinear (Algorithm 1) с реальной упаковкой uint8 и l1-LoRA.
    """
    replaced = 0
    max_cert_err = 0.0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_names):
            continue
        if module.in_features % 8 != 0 or module.weight.numel() <= min_numel:
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        q_layer = VCoreInDomainHadamardLinear.from_linear(
            module, lora_rank=lora_rank, lora_iters=lora_iters
        )
        setattr(parent, child_name, q_layer)
        replaced += 1
        max_cert_err = max(max_cert_err, float(q_layer.cert_row_l1_bound.item()))

    return {
        "quantized_layers": float(replaced),
        "max_certified_row_l1_bound": max_cert_err,
}
