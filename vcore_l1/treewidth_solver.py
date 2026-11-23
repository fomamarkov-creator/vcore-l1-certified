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
        return grid.narrow(0, idx, 1).clone(), float(errs.narrow(0, idx, 1).item())

    dp = torch.zeros(num_grid, device=w.device, dtype=torch.float32)
    backpointers = []

    for i in range(1, m):
        w_prev = w.narrow(0, i - 1, 1)
        w_curr = w.narrow(0, i, 1)
        edge_cost = (
            dp.unsqueeze(1)
            + torch.abs(w_prev - grid).unsqueeze(1)
            + torch.abs(w_curr - (grid.unsqueeze(1) + grid.unsqueeze(0)))
        )
        dp, best_prev = torch.min(edge_cost, dim=0)
        backpointers.append(best_prev)

    best_last = int(torch.argmin(dp).item())
    min_cost = float(dp.narrow(0, best_last, 1).item())

    states = [0 for _ in range(m)]
    states[m - 1] = best_last
    for i in range(m - 2, -1, -1):
        bp_row = backpointers[i]
        states[i] = int(bp_row.narrow(0, states[i + 1], 1).item())

    c_opt = grid.index_select(0, torch.tensor(states, device=w.device, dtype=torch.int64))
    return c_opt, min_cost
    
