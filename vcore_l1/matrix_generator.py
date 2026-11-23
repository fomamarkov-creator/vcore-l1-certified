# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import numpy as np
import torch

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def generate_markov_q(dim_n: int, dim_m: int = None, zeta: float = 1.024) -> np.ndarray:
    """
    Синтез ортогонального проектора Q = (I - P) * zeta размерности (dim_n x dim_n).
    Использует CuPy на GPU при наличии, либо PyTorch/NumPy на CPU.
    """
    if dim_m is None or dim_m >= dim_n:
        dim_m = max(1, dim_n // 2)

    if HAS_CUPY:
        basis = cp.zeros((dim_n, dim_m), dtype=cp.float32)
        t = cp.linspace(0, 2 * cp.pi, dim_n, dtype=cp.float32)
        for i in range(dim_m):
            basis[:, i] = cp.sin(t * (i + 1) * zeta)
        q, _ = cp.linalg.qr(basis)
        P = q @ q.T
        Q = (cp.eye(dim_n, dtype=cp.float32) - P) * zeta
        return cp.asnumpy(Q).astype(np.float32)

    # Фоллбэк на PyTorch (CPU или CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.linspace(0, 2 * torch.pi, dim_n, dtype=torch.float32, device=device)
    idx = torch.arange(1, dim_m + 1, dtype=torch.float32, device=device)
    basis = torch.sin(t.unsqueeze(1) * idx.unsqueeze(0) * zeta)
    q, _ = torch.linalg.qr(basis)
    P = q @ q.T
    Q = (torch.eye(dim_n, dtype=torch.float32, device=device) - P) * zeta
    return Q.cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    dim = 144
    matrix_q = generate_markov_q(dim)
    np.save("matrix_q_144.npy", matrix_q)
    print(f"[OK]: Матрица {dim}x{dim} синтезирована и сохранена в matrix_q_144.npy")
