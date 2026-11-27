# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import math
import os
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple
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
        limit_r = int(math.sqrt(target_n)) + 2
        for r in np.arange(2.0, limit_r, self.LEEWAY):
            rem = target_n % r
            offset = min(rem, abs(r - rem))
            if offset <= self.LEEWAY:
                potential_factor = round(float(r))
                if potential_factor > 1 and target_n % potential_factor == 0:
                    return potential_factor
        return None

    # =========================================================================
# 2. ВЕКТОРНЫЙ РЕЗОНАНСНЫЙ ПОДБОР МАСШТАБА И МНОГОКАНАЛЬНЫЙ ЭМРТ-ТОМОГРАФ ВЕСОВ
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
    cands = max_b * ratios.unsqueeze(0)

    imp_matrix = compute_tensor_resonance_impedance(blocks, cands)
    best_imp, best_idx = torch.min(imp_matrix, dim=1)
    best_scale = torch.gather(cands, 1, best_idx.unsqueeze(1)).squeeze(1)
    return best_scale, best_imp


@torch.no_grad()
def scan_model_tomography(
    model_or_state_dict: Dict[str, torch.Tensor] | torch.nn.Module,
    num_candidates: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """
    Многоканальный резонансный томограф весов модели (Weight EM-Tomography):
    сканирует каждый 2D-слой по 4 каналам диагностики:
      1. Продольный срез по входным каналам (Z_in и пиковые выбросы);
      2. Поперечный срез по выходным нейронам (Z_out и когерентность строк);
      3. Резонансный импеданс блоков весов;
      4. Спектральный выигрыш от вращения Уолша–Адамара W8.
    Возвращает томографическую карту каждого слоя и рекомендуемый режим сжатия.
    """
    from .super_quantizer import fwht_8

    if isinstance(model_or_state_dict, torch.nn.Module):
        sd = {
            f"{n}.weight": m.weight.detach()
            for n, m in model_or_state_dict.named_modules()
            if isinstance(m, torch.nn.Linear)
        }
    else:
        sd = model_or_state_dict

    tomogram: Dict[str, Dict[str, Any]] = {}
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.ndim != 2:
            continue
        if tensor.numel() < 256:
            continue

        w = tensor.to(torch.float32)
        abs_w = torch.abs(w)

        # Канал 1: Продольный срез (входные каналы / столбцы)
        col_l1 = torch.mean(abs_w, dim=0)
        col_max = torch.max(abs_w, dim=0).values
        in_peak_ratio = float((torch.max(col_max) / torch.clamp(torch.median(col_l1), min=1e-6)).item())

        # Канал 2: Поперечный срез (выходные нейроны / строки)
        row_l1 = torch.mean(abs_w, dim=1)
        row_max = torch.max(abs_w, dim=1).values
        out_peak_ratio = float((torch.max(row_max) / torch.clamp(torch.median(row_l1), min=1e-6)).item())

        # Канал 3: Резонансный импеданс 32-элементных блоков
        pad = (32 - (w.numel() % 32)) % 32
        flat = torch.cat([w.reshape(-1), torch.zeros(pad, device=w.device)]) if pad > 0 else w.reshape(-1)
        blocks = flat.view(-1, 32)
        _, min_imp = find_optimal_resonant_scale(blocks, num_candidates=num_candidates)
        mean_impedance = float(torch.mean(min_imp).item())

        # Канал 4: Спектральный срез W8 (выравнивание выбросов после Адамара)
        pad8 = (8 - (w.numel() % 8)) % 8
        flat8 = torch.cat([w.reshape(-1), torch.zeros(pad8, device=w.device)]) if pad8 > 0 else w.reshape(-1)
        w_rot = fwht_8(flat8.view(-1, 8)) / 2.8284271247461903
        rot_peak_ratio = float(
            (torch.max(torch.abs(w_rot)) / torch.clamp(torch.median(torch.abs(w_rot)), min=1e-6)).item()
        )
        hadamard_gain = in_peak_ratio / max(rot_peak_ratio, 1e-6)

        # Вердикт томографа по оптимальному режиму сжатия слоя
        if "embed_tokens" in name or "lm_head" in name:
            rec_mode = "FP16_PROTECTED"
        elif in_peak_ratio > 45.0 or out_peak_ratio > 45.0 or mean_impedance > 190.0:
            rec_mode = "INT4_HIGH_PRECISION"
        elif hadamard_gain > 1.35 and mean_impedance < 145.0:
            rec_mode = "E8_LATTICE_SUB1BIT"
        else:
            rec_mode = "TERNARY_158BIT"

        tomogram[name] = {
            "shape": tuple(w.shape),
            "in_peak_ratio": round(in_peak_ratio, 2),
            "out_peak_ratio": round(out_peak_ratio, 2),
            "mean_impedance": round(mean_impedance, 2),
            "hadamard_gain": round(hadamard_gain, 2),
            "recommended_mode": rec_mode,
        }

    return tomogram


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
    
