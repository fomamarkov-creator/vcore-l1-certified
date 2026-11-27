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
            r_val = float(r)
            rem = target_n % r_val
            offset = min(rem, abs(r_val - rem))
            if offset <= self.LEEWAY * (target_n / max(r_val, 1.0)):
                potential_factor = round(r_val)
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
    activation_map: Optional[Dict[str, torch.Tensor]] = None,
    head_dim: int = 64,
    num_candidates: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """
    Многоканальный контрастный ЭМРТ-томограф весов модели (Weight EM-Tomography v2):
      1. Продольный контрастный срез по входным каналам (Z_in с весом активаций γ_j);
      2. Поперечный срез по выходным нейронам (Z_out);
      3. По-головой срез (Attention Head Tomography) для проекций внимания (q/k/v/o_proj);
      4. Резонансный импеданс 32D-блоков и спектральный выигрыш от вращения Адамара W8.
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

        # Канал 1: Функциональный контраст (если есть статистика активаций γ_j)
        has_contrast = False
        if activation_map is not None and name in activation_map:
            gamma = activation_map[name].to(device=w.device, dtype=torch.float32).reshape(-1)
            if gamma.numel() == w.size(1):
                gamma_norm = gamma / torch.clamp(torch.mean(gamma), min=1e-6)
                abs_w = abs_w * gamma_norm.unsqueeze(0)
                has_contrast = True

        # Продольный срез (входные каналы) и поперечный срез (выходные нейроны)
        col_l1 = torch.mean(abs_w, dim=0)
        col_max = torch.max(abs_w, dim=0).values
        in_peak_ratio = float((torch.max(col_max) / torch.clamp(torch.median(col_l1), min=1e-6)).item())

        row_l1 = torch.mean(abs_w, dim=1)
        row_max = torch.max(abs_w, dim=1).values
        out_peak_ratio = float((torch.max(row_max) / torch.clamp(torch.median(row_l1), min=1e-6)).item())

        # Канал 2: По-головой срез (Attention Head Tomography для q/k/v/o_proj)
        is_attn = any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
        hot_heads = 0
        total_heads = 0
        if is_attn and w.size(0) >= head_dim and (w.size(0) % head_dim == 0):
            total_heads = w.size(0) // head_dim
            heads_view = abs_w.view(total_heads, head_dim, w.size(1))
            head_energies = torch.mean(heads_view, dim=(1, 2))
            med_head = torch.clamp(torch.median(head_energies), min=1e-6)
            hot_heads = int((head_energies > (med_head * 1.45)).sum().item())

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
        elif in_peak_ratio > 45.0 or out_peak_ratio > 45.0 or mean_impedance > 190.0 or (total_heads > 0 and hot_heads >= max(1, total_heads // 3)):
            rec_mode = "INT4_HIGH_PRECISION"
        elif hadamard_gain > 1.35 and mean_impedance < 145.0:
            rec_mode = "E8_LATTICE_SUB1BIT"
        else:
            rec_mode = "TERNARY_158BIT"

        tomogram[name] = {
            "shape": tuple(w.shape),
            "contrast_enabled": has_contrast,
            "in_peak_ratio": round(in_peak_ratio, 2),
            "out_peak_ratio": round(out_peak_ratio, 2),
            "hot_heads": f"{hot_heads}/{total_heads}" if total_heads > 0 else "N/A",
            "mean_impedance": round(mean_impedance, 2),
            "hadamard_gain": round(hadamard_gain, 2),
            "recommended_mode": rec_mode,
        }

    return tomogram

@torch.no_grad()
def surgical_unlearn_concept(
    model: torch.nn.Module,
    tokenizer: Any,
    target_prompts: List[str],
    neutral_prompts: Optional[List[str]] = None,
    strength: float = 0.85,
    layer_filter: Tuple[str, ...] = ("down_proj", "o_proj"),
) -> Dict[str, Any]:
    """
    Хирургическое вырезание знаний / паттернов из модели (Differential EM-Tomography + Rank-1 Null-Space Projection):
      1. Снимает дифференциальный контраст активаций (target - neutral) на выходных проекциях;
      2. Вычисляет ортогональный вектор направления концепта u;
      3. Вычитает проекцию W_clean = W - strength * (W u) u^T с переупаковкой квантованных слоёв.
    """
    from .inference_layer import VCoreLinear
    from .safetensors_quantizer import pack_ternary_to_uint8, quantize_ternary_exact_critical, unpack_uint8_to_int4, unpack_uint8_to_ternary

    if not neutral_prompts:
        neutral_prompts = [
            "The weather today is calm and clear across the region.",
            "Mathematics studies numbers, structures, spaces, and change.",
            "Water boils at one hundred degrees Celsius at standard pressure.",
            "History records events and developments of human civilization.",
        ]

    device = next(model.parameters()).device
    model.eval()

    target_layers: Dict[str, torch.nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, VCoreLinear)) and any(k in name for k in layer_filter):
            target_layers[name] = mod

    def _collect_mean_inputs(prompts: List[str]) -> Dict[str, torch.Tensor]:
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        hooks = []

        def _make_hook(l_name: str):
            def _hook(_mod, inp, _out):
                if not inp or not isinstance(inp[0], torch.Tensor):
                    return
                x = inp[0].detach().to(torch.float32).reshape(-1, inp[0].size(-1))
                vec = torch.mean(x, dim=0).cpu()
                if l_name not in sums:
                    sums[l_name] = vec
                    counts[l_name] = 1
                else:
                    sums[l_name] += vec
                    counts[l_name] += 1
            return _hook

        for l_name, mod in target_layers.items():
            hooks.append(mod.register_forward_hook(_make_hook(l_name)))

        for p in prompts:
            raw_toks = tokenizer(p, return_tensors="pt", truncation=True, max_length=128)
            toks = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw_toks.items()}
            _ = model(**toks)

        for h in hooks:
            h.remove()

        return {k: (sums[k] / max(counts[k], 1)) for k in sums}
    

    base_acts = _collect_mean_inputs(neutral_prompts)
    targ_acts = _collect_mean_inputs(target_prompts)

    edited_layers = 0
    total_energy_removed = 0.0

    for l_name, mod in target_layers.items():
        if l_name not in base_acts or l_name not in targ_acts:
            continue

        diff = (targ_acts[l_name] - base_acts[l_name]).to(device=device, dtype=torch.float32)
        norm_diff = float(torch.linalg.norm(diff).item())
        if norm_diff < 1e-6:
            continue

        u = diff / norm_diff  # Единичный вектор направления концепта (in_features,)

        if isinstance(mod, VCoreLinear):
            # Восстанавливаем матрицу весов, выполняем Rank-1 вычитание и переупаковываем
            if hasattr(mod, "_w_cache") and mod._w_cache is not None:
                w = mod._w_cache.to(torch.float32)
            else:
                from .quantizer import unpack_uint8_to_int4, unpack_uint8_to_ternary
                num_elems = mod.out_features * mod.in_features
                padded = ((num_elems + mod.group_size - 1) // mod.group_size) * mod.group_size
                grid = (
                    unpack_uint8_to_ternary(mod.vcore_packed, (padded,)).to(device)
                    if mod.bits == 2
                    else unpack_uint8_to_int4(mod.vcore_packed, (padded,)).to(device)
                )
                w = (grid.reshape(-1, mod.group_size) * mod.vcore_scales.to(torch.float32).unsqueeze(1)).reshape(-1)[
                    :num_elems
                ].reshape(mod.out_features, mod.in_features)

            proj = torch.mv(w, u)  # (out_features,)
            delta_w = strength * torch.outer(proj, u)
            w_clean = w - delta_w
            total_energy_removed += float(torch.sum(torch.abs(delta_w)).item())

            if mod.bits == 2:
                _, q_mask, new_scales = quantize_ternary_exact_critical(w_clean.reshape(-1, mod.group_size))
                packed_u8, _ = pack_ternary_to_uint8(q_mask)
                mod.vcore_packed.copy_(packed_u8.to(mod.vcore_packed.device))
                mod.vcore_scales.copy_(new_scales.to(mod.vcore_scales.dtype))

            mod._w_cache = w_clean.to(mod.vcore_scales.dtype)
            edited_layers += 1

        elif isinstance(mod, torch.nn.Linear):
            w = mod.weight.data.to(torch.float32)
            proj = torch.mv(w, u)
            delta_w = strength * torch.outer(proj, u)
            mod.weight.data.copy_((w - delta_w).to(mod.weight.dtype))
            total_energy_removed += float(torch.sum(torch.abs(delta_w)).item())
            edited_layers += 1

    return {
        "edited_layers": edited_layers,
        "strength": strength,
        "total_l1_energy_ablated": round(total_energy_removed, 4),
    }
        

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
    
