# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Tuple
import torch


def solve_chain_overlap_fpt(weights: torch.Tensor, num_grid: int = 17) -> Tuple[torch.Tensor, float]:
    """
    Полиномиальный FPT-алгоритм динамического программирования O(m * |grid|^2)
    для ленточных перекрытий (Δ = 2, Treewidth = 1), верифицированный
    в модуле MarkovL1TreewidthSolver.lean.
    """
    w = weights.reshape(-1).to(torch.float32)
    m = w.numel()
    if m == 0:
        return w.clone(), 0.0

    max_val = float(torch.max(torch.abs(w)).item())
    if max_val == 0.0:
        return torch.zeros_like(w), 0.0

    grid = torch.linspace(-max_val, max_val, num_grid, device=w.device, dtype=torch.float32)
    if m == 1:
        errs = torch.abs(w - grid)
        idx = int(torch.argmin(errs).item())
        return grid[idx].view(1), float(errs[idx].item())

    # Динамическое программирование по цепочке зависимостей Δ = 2
    dp = torch.zeros(num_grid, device=w.device, dtype=torch.float32)
    backpointers = []

    for i in range(1, m):
        w_prev, w_curr = w[i - 1], w[i]
        # Матрица переходов (prev_state, curr_state)
        edge_cost = (
            dp.unsqueeze(1)
            + torch.abs(w_prev - grid).unsqueeze(1)
            + torch.abs(w_curr - (grid.unsqueeze(1) + grid.unsqueeze(0)))
        )
        dp, best_prev = torch.min(edge_cost, dim=0)
        backpointers.append(best_prev)

    best_last = int(torch.argmin(dp).item())
    min_cost = float(dp[best_last].item())

    # Обратный проход (восстановление оптимального вектора c*)
    states = [0] * m
    states[-1] = best_last
    for i in range(m - 2, -1, -1):
        states[i] = int(backpointers[i][states[i + 1]].item())

    c_opt = grid[torch.tensor(states, device=w.device)]
    return c_opt, min_cost
