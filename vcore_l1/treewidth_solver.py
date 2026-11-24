# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import List, Optional, Tuple
import torch


@torch.no_grad()
def solve_chain_overlap_fpt(
    w: torch.Tensor,
    grid: Optional[torch.Tensor] = None,
    num_grid: int = 17,
) -> Tuple[torch.Tensor, float]:
    """
    Полиномиальный FPT-решатель O(m * |G|^2) для Δ = 2 ленточных перекрытий
    по дискретной сетке G (MarkovL1TreewidthSolver.lean).
    """
    w_1d = w.reshape(-1).to(torch.float32)
    m = w_1d.numel()
    device = w_1d.device

    if m == 0:
        return torch.empty(0, dtype=torch.float32, device=device), 0.0

    if grid is None:
        max_abs = float(torch.max(torch.abs(w_1d)).item())
        bound = max(max_abs, 1e-4)
        grid_1d = torch.linspace(-bound, bound, steps=num_grid, device=device, dtype=torch.float32)
    else:
        grid_1d = grid.reshape(-1).to(device=device, dtype=torch.float32)

    g_size = grid_1d.numel()
    if m == 1:
        errs = torch.abs(w_1d[0] - grid_1d)
        best_idx = int(torch.argmin(errs).item())
        return grid_1d[best_idx : best_idx + 1], float(errs[best_idx].item())

    dp = torch.zeros(g_size, device=device, dtype=torch.float32)
    backpointers = torch.zeros((m - 1, g_size), device=device, dtype=torch.int64)

    for i in range(m - 1):
        node_cost = torch.abs(w_1d[i] - grid_1d)  # (G,)
        pair_sums = grid_1d.unsqueeze(1) + grid_1d.unsqueeze(0)  # (G, G)
        edge_cost = torch.abs(w_1d[i + 1] - pair_sums)  # (G, G)
        total = (dp + node_cost).unsqueeze(1) + edge_cost  # (G, G)
        dp, backpointers[i] = torch.min(total, dim=0)

    final_total = dp + torch.abs(w_1d[m - 1] - grid_1d)
    last_idx = int(torch.argmin(final_total).item())
    min_cost = float(final_total[last_idx].item())

    c_opt = torch.empty(m, device=device, dtype=torch.float32)
    curr_idx = last_idx
    c_opt[m - 1] = grid_1d[curr_idx]
    for i in range(m - 2, -1, -1):
        curr_idx = int(backpointers[i, curr_idx].item())
        c_opt[i] = grid_1d[curr_idx]

    return c_opt, min_cost


def solve_continuous_chain_dp(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Точный непрерывный решатель для цепочек перекрытий Δ = 2 (treewidth k = 1)
    без дискретной сетки (реализация `MarkovL1ContinuousChainDP.lean`):
    вычисляет глобальный минимум Φ(c) = sum_{i=1}^{m-1} (|w_i - c_i| + |w_{i+1} - (c_i + c_{i+1})|) + |w_m - c_m|
    по всему непрерывному пространству R^m через клиппирование наклонов PWL-функции.
    """
    ws: List[float] = [float(x) for x in w.detach().cpu().reshape(-1).tolist()]
    m = len(ws)
    if m == 0:
        return torch.empty(0, dtype=w.dtype, device=w.device), 0.0
    if m == 1:
        return torch.tensor([ws[0]], dtype=w.dtype, device=w.device), 0.0

    def pwl_eval(val0: float, left_s: float, breaks: List[Tuple[float, float]], z: float) -> float:
        def raw(t: float) -> float:
            acc = left_s * t
            for bx, ds in breaks:
                if bx < t:
                    acc += ds * (t - bx)
            return acc
        return val0 + (raw(z) - raw(0.0))

    def pwl_argmin(left_s: float, breaks: List[Tuple[float, float]]) -> float:
        curr = left_s
        for bx, ds in breaks:
            if curr >= -1e-12:
                return bx
            curr += ds
            if curr >= -1e-12:
                return bx
        return 0.0

    def pwl_add_abs(val0: float, left_s: float, breaks: List[Tuple[float, float]], shift: float):
        new_breaks: List[Tuple[float, float]] = []
        inserted = False
        for bx, ds in breaks:
            if not inserted and shift < bx - 1e-12:
                new_breaks.append((shift, 2.0))
                inserted = True
            if abs(shift - bx) <= 1e-12:
                merged = ds + 2.0
                if abs(merged) > 1e-12:
                    new_breaks.append((bx, merged))
                inserted = True
            else:
                new_breaks.append((bx, ds))
        if not inserted:
            new_breaks.append((shift, 2.0))
        return val0 + abs(shift), left_s - 1.0, new_breaks

    def clip_s(s: float) -> float:
        return max(-1.0, min(1.0, s))

    def pwl_inf_conv(val0: float, left_s: float, breaks: List[Tuple[float, float]]):
        x_star = pwl_argmin(left_s, breaks)
        min_v = pwl_eval(val0, left_s, breaks, x_star)
        curr_raw = left_s
        clipped_breaks: List[Tuple[float, float]] = []
        for bx, ds in breaks:
            next_raw = curr_raw + ds
            d_clip = clip_s(next_raw) - clip_s(curr_raw)
            if abs(d_clip) > 1e-12:
                clipped_breaks.append((bx, d_clip))
            curr_raw = next_raw
        new_left = clip_s(left_s)
        val_at_star = pwl_eval(0.0, new_left, clipped_breaks, x_star)
        return min_v - val_at_star, new_left, clipped_breaks

    def pwl_reflect_shift(val0: float, left_s: float, breaks: List[Tuple[float, float]], shift: float):
        total_s = left_s + sum(ds for _, ds in breaks)
        rev_breaks = [(shift - bx, ds) for bx, ds in reversed(breaks)]
        new_val0 = pwl_eval(val0, left_s, breaks, shift)
        return new_val0, -total_s, rev_breaks

    dp = (0.0, 0.0, [])
    history = []
    for i in range(m - 1):
        f_node = pwl_add_abs(*dp, ws[i])
        history.append((f_node, ws[i + 1]))
        h_conv = pwl_inf_conv(*f_node)
        dp = pwl_reflect_shift(*h_conv, ws[i + 1])

    final_dp = pwl_add_abs(*dp, ws[-1])
    c_curr = pwl_argmin(final_dp[1], final_dp[2])
    min_cost = pwl_eval(*final_dp, c_curr)

    cs = [0.0] * m
    cs[-1] = c_curr
    for i in range(m - 2, -1, -1):
        f_node, w_next = history[i]
        target = w_next - cs[i + 1]
        f_tot = pwl_add_abs(*f_node, target)
        cs[i] = pwl_argmin(f_tot[1], f_tot[2])

    return torch.tensor(cs, dtype=w.dtype, device=w.device), float(min_cost)
    @torch.no_grad()
def solve_triple_overlap_fpt(
    w: torch.Tensor,
    grid: Optional[torch.Tensor] = None,
    num_grid: int = 9,
) -> Tuple[torch.Tensor, float]:
    """
    Точный полиномиальный FPT-решатель O(m * |G|^3) для графов перекрытий степени Δ = 3
    (древесная ширина treewidth k = 2, модуль `MarkovL1GeneralTreewidthFPT.lean`),
    моделирующий совместное квантование троек связанных проекций (W_Q, W_K, W_V).
    """
    w_1d = w.reshape(-1).to(torch.float32)
    m = w_1d.numel()
    device = w_1d.device

    if m == 0:
        return torch.empty(0, dtype=torch.float32, device=device), 0.0

    if grid is None:
        max_abs = float(torch.max(torch.abs(w_1d)).item())
        bound = max(max_abs, 1e-4)
        grid_1d = torch.linspace(-bound, bound, steps=num_grid, device=device, dtype=torch.float32)
    else:
        grid_1d = grid.reshape(-1).to(device=device, dtype=torch.float32)

    g_size = grid_1d.numel()
    if m <= 2:
        return solve_chain_overlap_fpt(w_1d, grid=grid_1d)

    # dp[idx_prev2, idx_prev1] хранит минимальную стоимость префикса до пары (c_{i-2}, c_{i-1})
    dp = torch.zeros((g_size, g_size), device=device, dtype=torch.float32)
    backpointers = torch.zeros((m - 2, g_size, g_size), device=device, dtype=torch.int64)

    # Предвычисляем сетку сумм c_a + c_b + c_c формы (G, G, G)
    triple_sums = (
        grid_1d.view(g_size, 1, 1)
        + grid_1d.view(1, g_size, 1)
        + grid_1d.view(1, 1, g_size)
    )

    for i in range(m - 2):
        node_cost = torch.abs(w_1d[i] - grid_1d).view(g_size, 1, 1)  # (G, 1, 1)
        edge_cost = torch.abs(w_1d[i + 2] - triple_sums)  # (G, G, G)
        total = dp.unsqueeze(2) + node_cost + edge_cost  # (G, G, G)
        dp, backpointers[i] = torch.min(total, dim=0)

    # Добавляем краевые локальные стоимости для последних двух координат w_{m-2}, w_{m-1}
    end_cost = (
        dp
        + torch.abs(w_1d[m - 2] - grid_1d).unsqueeze(1)
        + torch.abs(w_1d[m - 1] - grid_1d).unsqueeze(0)
    )
    flat_idx = int(torch.argmin(end_cost).item())
    min_cost = float(end_cost.reshape(-1)[flat_idx].item())

    idx_prev1 = flat_idx // g_size
    idx_curr = flat_idx % g_size

    c_opt = torch.empty(m, device=device, dtype=torch.float32)
    c_opt[m - 1] = grid_1d[idx_curr]
    c_opt[m - 2] = grid_1d[idx_prev1]

    for i in range(m - 3, -1, -1):
        idx_prev2 = int(backpointers[i, idx_prev1, idx_curr].item())
        c_opt[i] = grid_1d[idx_prev2]
        idx_curr = idx_prev1
        idx_prev1 = idx_prev2

    return c_opt, min_cost
    
