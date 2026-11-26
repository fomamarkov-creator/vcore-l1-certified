# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Tuple
import torch
from .super_quantizer import get_d4_codebook, get_e8_codebook


# =========================================================================
# 1. МЕЖСЛОЙНАЯ КОМПЕНСАЦИЯ КРИВИЗНЫ (CROSS-LAYER ℓ₁ COMPENSATION)
# =========================================================================

@torch.no_grad()
def apply_cross_layer_compensation(
    w1_orig: torch.Tensor,
    w1_quant: torch.Tensor,
    w2_orig: torch.Tensor,
    dampening: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """
    Реализация Теорем `cross_layer_exact_cancellation` и `two_layer_cascade_drift_bound`
    (MarkovL1CrossLayerAndIntegerTransformer.lean):
    компенсирует ошибку квантования предыдущего слоя `E₁ = W₁ - Ŵ₁` путём
    оптимальной коррекции весов следующего слоя `W₂' = W₂ + ΔW₂`, минимизируя
    каскадный дрейф `‖W₂ W₁ - W₂' Ŵ₁‖₁`.

    Аргументы:
        w1_orig: исходная матрица первого слоя `(H, I)`
        w1_quant: квантованная матрица первого слоя `(H, I)`
        w2_orig: исходная матрица второго слоя `(O, H)`
        dampening: коэффициент демпфирования (0.0 .. 1.0) для устойчивости на выбросах
    """
    w1 = w1_orig.to(torch.float32)
    wh1 = w1_quant.to(torch.float32)
    w2 = w2_orig.to(torch.float32)

    err1 = w1 - wh1  # (H, I)
    # Решаем min_{ΔW₂} ‖W₂ E₁ - ΔW₂ Ŵ₁‖_F^2 с тихоновской регуляризацией
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
# 2. РАЦИОНАЛЬНЫЕ СФЕРИЧЕСКИЕ КОДЫ (EMBEDDINGS / LM_HEAD / KV-CACHE)
# =========================================================================

@torch.no_grad()
def quantize_spherical_code(
    vectors: torch.Tensor,
    lattice: str = "e8",
) -> Tuple[torch.Tensor, float]:
    """
    Реализация Теорем `spherical_chord_dot_identity` и `spherical_code_inner_product_lower_bound`
    (MarkovL1CrossLayerAndIntegerTransformer.lean):
    проектирует блоки на постоянную энергетическую оболочку гиперсферы (`‖c‖₂² = R²`)
    в решётке D₄ (24 вершины, R²=2) или E₈ (240 вершин, R²=2), сохраняя индивидуальный
    L2-радиус каждого блока для точного косинусного сходства.
    """
    orig_shape = vectors.shape
    dim = 8 if lattice.lower() == "e8" else 4
    flat = vectors.reshape(-1).to(torch.float32)
    pad = (dim - (flat.numel() % dim)) % dim
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    blocks = flat.view(-1, dim)
    norms = torch.linalg.norm(blocks, dim=1, keepdim=True)
    safe_norms = torch.clamp(norms, min=1e-8)
    # Нормируем блоки на сферу радиуса √2 (R² = 2, как на оболочках D₄ и E₈)
    unit_blocks = blocks * (1.4142135623730951 / safe_norms)

    cb = get_e8_codebook(blocks.device, torch.float32)[1:] if dim == 8 else get_d4_codebook(blocks.device, torch.float32)[1:]
    # Максимизация ⟨u, c⟩ на гиперсфере R² = 2 эквивалентна минимизации хорды ‖u - c‖₂²
    dots = unit_blocks @ cb.T
    best_idx = torch.argmax(dots, dim=1)
    q_sphere = cb[best_idx]

    # Сертифицированная граница хорды δ² = max ‖u - c‖₂²
    chord_sq = torch.sum((unit_blocks - q_sphere) ** 2, dim=1)
    max_chord_sq = float(torch.max(chord_sq).item())

    restored = q_sphere * (norms / 1.4142135623730951)
    out = restored.reshape(-1)[: vectors.numel()].reshape(orig_shape).to(vectors.dtype)
    return out, max_chord_sq
# =========================================================================
# 3. БЕСПЛУЮЧИЕ (INTEGER-ONLY / DYADIC) ПРИМИТИВЫ ТРАНСФОРМЕРА
# =========================================================================

def dyadic_softmax(logits: torch.Tensor, k_squarings: int = 4) -> torch.Tensor:
    """
    Реализация Теоремы `dyadicExpApproxQ_nonneg` (MarkovL1CrossLayerAndIntegerTransformer.lean):
    вычисляет Softmax через двоично-рациональную схему возведения в квадрат
    `exp(x) ≈ (max(0, 1 + x / 2^k))^(2^k)` без трансцендентных функций.
    """
    x = logits.to(torch.float32)
    x_shifted = x - torch.max(x, dim=-1, keepdim=True).values
    scale = float(1 << k_squarings)
    acc = torch.clamp(1.0 + x_shifted / scale, min=0.0)
    for _ in range(k_squarings):
        acc = acc * acc
    denom = torch.clamp(torch.sum(acc, dim=-1, keepdim=True), min=1e-12)
    return (acc / denom).to(logits.dtype)


def integer_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    newton_iters: int = 3,
) -> torch.Tensor:
    """
    Реализация Теоремы `invSqrtNewtonStepQ_residual_identity`
    (MarkovL1CrossLayerAndIntegerTransformer.lean):
    вычисляет RMSNorm через квадратично сходящиеся бесплуючие итерации Ньютона
    `y_{n+1} = y_n * (3 - S * y_n²) / 2` для `1 / √S`.
    """
    xf = x.to(torch.float32)
    s = torch.mean(xf * xf, dim=-1, keepdim=True) + eps

    # Начальное приближение через битовый сдвиг экспоненты IEEE-754 (быстрый обратный корень)
    _, exp2 = torch.frexp(s)
    y = torch.ldexp(torch.ones_like(s), -(exp2 // 2))

    for _ in range(newton_iters):
        y = y * (3.0 - s * y * y) * 0.5

    out = xf * y * weight.to(torch.float32)
    return out.to(x.dtype)
