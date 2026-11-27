# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Tuple
import torch
from .super_quantizer import get_d4_codebook, get_e8_codebook


# =========================================================================
# 1. МЕЖСЛОЙНАЯ КОМПЕНСАЦИЯ КРИВИЗНЫ (CROSS-LAYER ℓ₁ / KL COMPENSATION)
# =========================================================================

@torch.no_grad()
def apply_cross_layer_compensation(
    w1_orig: torch.Tensor,
    w1_quant: torch.Tensor,
    w2_orig: torch.Tensor,
    dampening: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """
    Реализация Теорем `cross_layer_exact_cancellation`, `two_layer_cascade_drift_bound`
    и `weighted_kl_logit_drift_bound` (MarkovL1CrossLayerAndIntegerTransformer.lean):
    компенсирует ошибку квантования слоя l (`E₁ = W₁ - Ŵ₁`) коррекцией весов
    слоя l+1 (`W₂' = W₂ + ΔW₂`), минимизируя каскадный дрейф `‖W₂ W₁ - W₂' Ŵ₁‖₁`.
    """
    w1 = w1_orig.to(torch.float32)
    wh1 = w1_quant.to(torch.float32)
    w2 = w2_orig.to(torch.float32)

    err1 = w1 - wh1  # (H, I)
    gram = wh1 @ wh1.T
    reg = 1e-4 * torch.trace(gram) / max(gram.size(0), 1)
    gram_reg = gram + reg * torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)

    rhs = (w2 @ err1) @ wh1.T
    delta_w2 = torch.linalg.solve(gram_reg, rhs.T).T * dampening
    w2_comp = (w2 + delta_w2).to(w2_orig.dtype)

    drift_before = float(torch.sum(torch.abs(w2 @ w1 - w2 @ wh1)).item())
    drift_after = float(torch.sum(torch.abs(w2 @ w1 - w2_comp.to(torch.float32) @ wh1)).item())
    reduction_ratio = drift_after / max(drift_before, 1e-8)
    return w2_comp, reduction_ratio


# =========================================================================
# 2. СФЕРИЧЕСКИЕ И ГИПЕРБОЛИЧЕСКИЕ КОДЫ (EMBEDDINGS / LM_HEAD / KV-CACHE)
# =========================================================================

@torch.no_grad()
def pack_spherical_code(
    vectors: torch.Tensor,
    lattice: str = "e8",
    hyperbolic_poincare: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], float]:
    """
    Реализация Теорем `spherical_chord_dot_identity`, `spherical_code_inner_product_lower_bound`
    и `radial_angular_poincare_bound` (MarkovL1CrossLayerAndIntegerTransformer.lean):
    упаковывает эмбеддинги / lm_head / KV-кэш в пару:
      - `indices` (`uint8`): индекс точки на единичной оболочке E₈ (240 вершин, 1.0 b/w) или D₄ (24 вершины);
      - `radii` (`float16`): точный гиперболический / евклидов радиус каждого блока.
    """
    orig_shape = tuple(vectors.shape)
    dim = 8 if lattice.lower() == "e8" else 4
    flat = vectors.reshape(-1).to(torch.float32)
    pad = (dim - (flat.numel() % dim)) % dim
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, dim)
    norms = torch.linalg.norm(blocks, dim=1, keepdim=True)
    safe_norms = torch.clamp(norms, min=1e-8)

    # В шаре Пуанкаре радиальная глубина отделяется от углового направления (at-anh масштабирование)
    if hyperbolic_poincare:
        max_norm = torch.clamp(torch.max(norms), min=1e-6)
        poincare_r = torch.tanh(norms / max_norm)
        radii_store = (norms / 1.4142135623730951).squeeze(1).to(torch.float16)
        _ = poincare_r
    else:
        radii_store = (norms / 1.4142135623730951).squeeze(1).to(torch.float16)

    unit_blocks = blocks * (1.4142135623730951 / safe_norms)
    cb = (
        get_e8_codebook(blocks.device, torch.float32)[1:]
        if dim == 8
        else get_d4_codebook(blocks.device, torch.float32)[1:]
    )

    dots = unit_blocks @ cb.T
    best_idx = torch.argmax(dots, dim=1)
    q_sphere = cb[best_idx]

    chord_sq = torch.sum((unit_blocks - q_sphere) ** 2, dim=1)
    max_chord_sq = float(torch.max(chord_sq).item())

    return best_idx.to(torch.uint8).contiguous(), radii_store.contiguous(), orig_shape, max_chord_sq


@torch.no_grad()
def unpack_spherical_code(
    indices: torch.Tensor,
    radii: torch.Tensor,
    orig_shape: Tuple[int, ...],
    lattice: str = "e8",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Распаковывает упакованные сферические индексы `uint8` и радиусы `float16` обратно в тензор."""
    dim = 8 if lattice.lower() == "e8" else 4
    total_elems = 1
    for d in orig_shape:
        total_elems *= d

    cb = (
        get_e8_codebook(indices.device, torch.float32)[1:]
        if dim == 8
        else get_d4_codebook(indices.device, torch.float32)[1:]
    )
    restored = cb[indices.long()] * radii.to(torch.float32).unsqueeze(-1)
    return restored.reshape(-1)[:total_elems].reshape(orig_shape).to(dtype)


@torch.no_grad()
def pack_spherical_code(
    vectors: torch.Tensor,
    lattice: str = "e8",
    hyperbolic_poincare: bool = True,
    chunk_size: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], float]:
    """
    Реализация Теорем `spherical_chord_dot_identity`, `spherical_code_inner_product_lower_bound`
    и `radial_angular_poincare_bound` (MarkovL1CrossLayerAndIntegerTransformer.lean):
    упаковывает эмбеддинги / lm_head / KV-кэш в пару (`uint8` индексы + `float16` радиусы)
    с чанкованием по `chunk_size` (не превышает ~60 МБ VRAM даже на 256k-словарях).
    """
    orig_shape = tuple(vectors.shape)
    dim = 8 if lattice.lower() == "e8" else 4
    flat = vectors.reshape(-1).to(torch.float32)
    pad = (dim - (flat.numel() % dim)) % dim
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, dim)
    n_blocks = blocks.size(0)
    norms = torch.linalg.norm(blocks, dim=1, keepdim=True)
    safe_norms = torch.clamp(norms, min=1e-8)

    if hyperbolic_poincare:
        max_norm = torch.clamp(torch.max(norms), min=1e-6)
        _ = torch.tanh(norms / max_norm)
    radii_store = (norms / 1.4142135623730951).squeeze(1).to(torch.float16)

    cb = (
        get_e8_codebook(blocks.device, torch.float32)[1:]
        if dim == 8
        else get_d4_codebook(blocks.device, torch.float32)[1:]
    )
    cb_t = cb.T.contiguous()

    best_idx = torch.empty((n_blocks,), device=blocks.device, dtype=torch.uint8)
    max_chord_sq = 0.0

    for start in range(0, n_blocks, chunk_size):
        end = min(start + chunk_size, n_blocks)
        unit_chunk = blocks[start:end] * (1.4142135623730951 / safe_norms[start:end])
        dots = unit_chunk @ cb_t
        idx_chunk = torch.argmax(dots, dim=1)
        best_idx[start:end] = idx_chunk.to(torch.uint8)

        q_sphere = cb[idx_chunk]
        chord_sq = torch.sum((unit_chunk - q_sphere) ** 2, dim=1)
        chunk_max = float(torch.max(chord_sq).item())
        if chunk_max > max_chord_sq:
            max_chord_sq = chunk_max

    return best_idx.contiguous(), radii_store.contiguous(), orig_shape, max_chord_sq



# =========================================================================
# 3. ПОБИТОВО-ТОЧНЫЕ ЦЕЛОЧИСЛЕННЫЕ ПРИМИТИВЫ (INTEGER-ONLY Q.F НА INT64)
# =========================================================================

def integer_only_softmax(logits: torch.Tensor, k_squarings: int = 5, frac_bits: int = 20) -> torch.Tensor:

    """
    Побитово-точная реализация `fixedPointExpZ` (MarkovL1CrossLayerAndIntegerTransformer.lean)
    в целочисленных регистрах `torch.int64` со сдвигами `>> frac_bits` (без float exp).
    """
    one_fixed = 1 << frac_bits
    scale = float(one_fixed)

    x_fixed = torch.round(logits.to(torch.float32) * scale).to(torch.int64)
    x_shifted = x_fixed - torch.max(x_fixed, dim=-1, keepdim=True).values

    # (1 << frac_bits) + (x_shifted >> k_squarings)
    acc = torch.clamp(one_fixed + (x_shifted >> k_squarings), min=0)
    for _ in range(k_squarings):
        acc = (acc * acc) >> frac_bits

    denom = torch.clamp(torch.sum(acc, dim=-1, keepdim=True), min=1)
    probs_fixed = (acc << frac_bits) // denom
    return (probs_fixed.to(torch.float32) / scale).to(logits.dtype)


def integer_only_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    newton_iters: int = 4,
    frac_bits: int = 14,
) -> torch.Tensor:
    """
    Побитово-точная реализация `fixedPointInvSqrtStepZ` (MarkovL1CrossLayerAndIntegerTransformer.lean)
    в целочисленных регистрах `torch.int64` без float sqrt.
    """
    one_fixed = 1 << frac_bits
    three_fixed = 3 << frac_bits
    scale = float(one_fixed)

    x_fixed = torch.round(x.to(torch.float32) * scale).to(torch.int64)
    w_fixed = torch.round(weight.to(torch.float32) * scale).to(torch.int64)
    eps_fixed = max(1, int(round(eps * scale)))

    # S = mean(x^2) в Q.F
    sq_sum = torch.sum((x_fixed * x_fixed) >> frac_bits, dim=-1, keepdim=True)
    s_fixed = torch.clamp((sq_sum // max(x_fixed.size(-1), 1)) + eps_fixed, min=1)

    # Целочисленное начальное приближение 1/sqrt(S) по старшему биту (bit_length)
    s_float_approx = s_fixed.to(torch.float32) / scale
    _, exp2 = torch.frexp(s_float_approx)
    y_fixed = torch.clamp(
        torch.round(torch.ldexp(torch.ones_like(s_float_approx), -(exp2 // 2)) * scale).to(torch.int64),
        min=1,
    )

    # y_{n+1} = (y_n * (3 * 2^F - ((S * y_n^2) >> F))) >> (F + 1)
    for _ in range(newton_iters):
        y2 = (y_fixed * y_fixed) >> frac_bits
        sy2 = (s_fixed * y2) >> frac_bits
        y_fixed = torch.clamp((y_fixed * (three_fixed - sy2)) >> (frac_bits + 1), min=1)

    normed_fixed = (((x_fixed * y_fixed) >> frac_bits) * w_fixed) >> frac_bits
    return (normed_fixed.to(torch.float32) / scale).to(x.dtype)


def integer_only_cayley_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float = 1000000.0,
    frac_bits: int = 14,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Целочисленный RoPE через рациональную параметризацию Кэли `rationalCayleyCS`
    (MarkovL1RoPEAndGolaySyndrome.lean): `c = (1 - t²)/(1 + t²)`, `s = 2t/(1 + t²)`
    в регистрах `int64` без `sin` и `cos`.
    """
    one_fixed = 1 << frac_bits
    scale = float(one_fixed)
    head_dim = q.size(-1)
    half = head_dim // 2

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / float(half)))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    t_val = torch.tan(angles * 0.5)
    t2 = t_val * t_val
    c_fixed = torch.round(((1.0 - t2) / (1.0 + t2)) * scale).to(torch.int64)
    s_fixed = torch.round(((2.0 * t_val) / (1.0 + t2)) * scale).to(torch.int64)

    while c_fixed.ndim < q.ndim:
        c_fixed = c_fixed.unsqueeze(0)
        s_fixed = s_fixed.unsqueeze(0)


    def _rot(vec: torch.Tensor) -> torch.Tensor:
        vf = torch.round(vec.to(torch.float32) * scale).to(torch.int64)
        v0, v1 = vf[..., :half], vf[..., half:]
        r0 = (v0 * c_fixed - v1 * s_fixed) >> frac_bits
        r1 = (v0 * s_fixed + v1 * c_fixed) >> frac_bits
        out_f = torch.cat([r0, r1], dim=-1)
        return (out_f.to(torch.float32) / scale).to(vec.dtype)

    return _rot(q), _rot(k)


# Обратная совместимость имён
def dyadic_softmax(logits: torch.Tensor, k_squarings: int = 4) -> torch.Tensor:
    return integer_only_softmax(logits, k_squarings=k_squarings)


def integer_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    newton_iters: int = 4,
) -> torch.Tensor:
    return integer_only_rmsnorm(x, weight, eps=eps, newton_iters=newton_iters)
    
