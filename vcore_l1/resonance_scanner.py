# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import math
import os
from decimal import Decimal, getcontext
from typing import List, Optional, Tuple
import numpy as np
import torch

# Устанавливаем высокую точность (100 знаков), чтобы лог-модулярный дрейф -0.287 не терял точность
getcontext().prec = 100

# Инварианты Маркова (Z/256Z)
LIMIT = 256.0           # Точка инверсии (регистровый предел)
BASE = 250.0            # Гармонический базис
ZETA = 1.024            # Шаг фазового винта (256/250)
LEEWAY = 0.024          # Базовый зазор (ZETA - 1)
DRIFT = -0.287          # Лог-модулярный дрейф Коллатца
COORDINATION_Z = 12     # Число соседей в 3HCP решётке


# =========================================================================
# 1. ПРЕЦИЗИОННЫЙ СКАНЕР РЕЗОНАНСА НА DECIMAL (100 ЗНАКОВ)
# =========================================================================

def get_white_rabbit_alpha(n: int | float | Decimal, r: int | float | Decimal) -> Decimal:
    """Вычисляет коэффициент локальной асимметрии фазы в 3HCP-решётке."""
    N = Decimal(str(n))
    R = Decimal(str(r))
    if R == 0:
        return Decimal(0)

    delta_ms = N % R
    delta_sm = R - delta_ms
    if delta_sm == 0:
        return Decimal(0)
    return (delta_ms / delta_sm) - Decimal(1)


def get_dynamic_leeway(n: int | float | Decimal, r: int | float | Decimal) -> Decimal:
    """Уточняет базовый зазор 0.024 на основе локальной фазовой асимметрии."""
    alpha = get_white_rabbit_alpha(n, r)
    leeway = Decimal(str(LEEWAY))
    return leeway * (Decimal(1) + alpha * Decimal("0.1"))


def get_markov_impedance(n: int | float | Decimal, r: int | float | Decimal) -> float:
    """Вычисляет волновой импеданс Z(N, R) с прецизионной точностью 100 знаков."""
    R = Decimal(str(r))
    if R < 1:
        return float(LIMIT)

    N = Decimal(str(n))
    DRIFT_D = Decimal(str(DRIFT))
    ZETA_D = Decimal(str(ZETA))
    LIMIT_D = Decimal(str(LIMIT))

    rem = N % R
    offset = min(rem, abs(R - rem))
    dyn_leeway = get_dynamic_leeway(n, r)

    denominator = R * ZETA_D + dyn_leeway
    exponent_val = DRIFT_D * (offset / denominator)
    result = LIMIT_D * (Decimal(1) - exponent_val.exp())
    return float(result)


def scan_resonance_range(n: int, start: float, end: float, step: float = LEEWAY) -> List[float]:
    """Высокоточный сканер узлов Electrical Lock (импеданс < 1e-12)."""
    found_nodes: List[float] = []
    for r in np.arange(start, end, step):
        r_val = round(float(r), 8)
        if r_val < 1.0:
            continue
        imp = get_markov_impedance(n, r_val)
        if imp < 1e-12:
            found_nodes.append(r_val)
    return found_nodes


def text_to_matrix_coord(text: str) -> int:
    """Кодирует строку в резонансную координату по основанию 256."""
    res_num = 0
    for i, char in enumerate(text):
        res_num += ord(char) * (256 ** i)
    return res_num


class MarkovMatrixAnalyzer:
    """Анализатор резонансных узлов в 3HCP-матрице."""

    def __init__(self):
        self.LIMIT = LIMIT
        self.BASE = BASE
        self.ZETA = ZETA
        self.LEEWAY = LEEWAY
        self.DRIFT = DRIFT

    def find_resonance(self, target_n: int) -> Optional[int]:
        limit_r = int(math.sqrt(target_n)) + 1
        for r in np.arange(2.0, limit_r, self.LEEWAY):
            impedance = target_n % r
            if impedance < 0.001 or abs(impedance - r) < 0.001:
                potential_factor = round(float(r))
                if potential_factor > 1 and target_n % potential_factor == 0:
                    return potential_factor
        return None


# =========================================================================
# 2. ВЕКТОРНЫЙ РЕЗОНАНСНЫЙ ПОДБОР МАСШТАБА ДЛЯ ТЕНЗОРОВ PYTORCH (GPU / CPU)
# =========================================================================

@torch.no_grad()
def compute_tensor_resonance_impedance(
    weights: torch.Tensor,
    candidate_scales: torch.Tensor,
) -> torch.Tensor:
    """
    Векторный расчёт волнового импеданса Z(W, R) для быстрого отбора резонансных
    шагов квантования R = candidate_scales на GPU/CPU.
    weights: (num_blocks, group_size), candidate_scales: (num_blocks, num_candidates)
    """
    abs_w = torch.abs(weights).unsqueeze(-1)
    r = torch.clamp(candidate_scales.unsqueeze(1), min=1e-8)

    delta_ms = torch.remainder(abs_w, r)
    delta_sm = torch.clamp(r - delta_ms, min=1e-8)
    alpha = (delta_ms / delta_sm) - 1.0

    dyn_leeway = LEEWAY * (1.0 + alpha * 0.1)
    offset = torch.minimum(delta_ms, delta_sm)

    denom = r * ZETA + dyn_leeway
    impedance = LIMIT * (1.0 - torch.exp(DRIFT * (offset / denom)))
    return torch.mean(impedance, dim=1)


@torch.no_grad()
def find_optimal_resonant_scale(
    blocks: torch.Tensor,
    num_candidates: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Находит для каждого блока весов оптимальный резонансный масштаб R*,
    минимизирующий фазовый импеданс Z(W, R*).
    """
    abs_b = torch.abs(blocks)
    max_b = torch.clamp(torch.max(abs_b, dim=1, keepdim=True).values, min=1e-6)
    ratios = torch.linspace(0.15, 1.0, steps=num_candidates, device=blocks.device, dtype=blocks.dtype)
    cands = max_b * ratios.unsqueeze(0)  # (B, K)

    imp_matrix = compute_tensor_resonance_impedance(blocks, cands)  # (B, K)
    best_imp, best_idx = torch.min(imp_matrix, dim=1)
    best_scale = torch.gather(cands, 1, best_idx.unsqueeze(1)).squeeze(1)
    return best_scale, best_imp


# =========================================================================
# 3. ВИЗУАЛИЗАЦИЯ ЭНЕРГЕТИЧЕСКИХ ЯМ (ОПЦИОНАЛЬНО, ЕСЛИ УСТАНОВЛЕН MATPLOTLIB)
# =========================================================================

def generate_resonance_plot(
    target_n: int = 3233,
    show_chars: bool = True,
    save_path: str = "assets/scan_3233.png",
) -> str:
    """Строит и сохраняет график энергетических ям импеданса в 3HCP-матрице."""
    import matplotlib.pyplot as plt

    r_start = 2.0
    r_end = float(int(math.sqrt(target_n)) + 20)
    r_values = np.linspace(r_start, r_end, 3500)

    impedances = []
    for r in r_values:
        remainder = target_n % r
        offset = min(remainder, abs(r - remainder))
        imp = LIMIT * (1.0 - math.exp(DRIFT * (offset / r)))
        impedances.append(imp)

    plt.figure(figsize=(14, 7), facecolor="#0e1117")
    ax = plt.axes()
    ax.set_facecolor("#161b22")

    plt.plot(r_values, impedances, color="#00ffcc", linewidth=1.2, label="Impedance Barrier")
    plt.axhline(y=0, color="#ff3333", linestyle="--", alpha=0.6, label="Inversion Line (256 ≡ 0)")
    plt.grid(which="major", alpha=0.2, color="cyan")

    for i in range(2, int(math.sqrt(target_n)) + 1):
        if target_n % i == 0:
            plt.scatter(i, 0, color="#ff0055", s=120, zorder=10)
            label = f"Node: {i}"
            if show_chars and 32 <= i <= 126:
                label += f" ('{chr(i)}')"
            plt.annotate(
                label,
                (i, -12),
                color="white",
                ha="center",
                fontsize=9,
                bbox=dict(facecolor="#ff0055", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"),
            )

    plt.title(f"3HCP RESONANCE SCAN | TARGET: {target_n}", color="white", fontsize=16, pad=16)
    plt.xlabel("Matrix Coordinate (r)", color="#00ffcc")
    plt.ylabel("Impedance Barrier", color="#00ffcc")
    plt.tick_params(colors="white")
    plt.xlim(r_start, r_end)
    plt.ylim(-22, LIMIT + 20)
    plt.legend(facecolor="#161b22", labelcolor="white")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, facecolor="#0e1117", dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
  
