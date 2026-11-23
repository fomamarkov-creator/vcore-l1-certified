[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v2.0.0`) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / CUDA (`CuPy` + `DLPack`) | **License:** AGPL-3.0

# V-CORE \(\ell_1\)-Certified: Hybrid Lean 4 + GPU 1.58-Bit Neural Network Quantizer

A hybrid symbolic-numeric quantization engine combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v2.0.0`) with a high-speed **GPU/CUDA execution backend** (`LinearSelector` + blockwise \(\Delta = 1\) median scaling) for 1-bit and 1.58-bit (`{-1, 0, +1}`) neural network compression.

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Architecture

This repository directly imports the formal verification library from `://github.com` (`v2.0.0`) via Lake and pairs it with GPU tensor kernels:

1. **Formal Certification Engine (`CertifierMain.lean` \(\to\) `vcore_certifier`):**
   * Links directly against `MarkovL1Complexity`, `MarkovL1Solver`, `MarkovL1WeightedSolver`, and `MarkovL1Quantizer`.
   * Executes exact-rational (\(\mathbb{Q}\)) \(\ell_1\)-optimality verification without `Classical.choice`, certifying that blockwise median scales \(\alpha_b^* = \mathrm{median}(\vert{}w_i\vert{}_{i \in S_b})\) achieve the global minimum \(\ell_1\) quantization error in the decoupled regime (\(\Delta = 1\)).
2. **Hybrid GPU / PyTorch Engine (`vcore_l1/cuda_quantizer.py`):**
   * **Stage 1 (`LinearSelector` CUDA Kernel via `CuPy` + `DLPack`):** Hard-threshold noise pruning (\(\vert{}w_i\vert{} \le \theta \to 0\)) directly on GPU tensors.
   * **Stage 2 (\(\Delta = 1\) Blockwise Median Scaling):** Decouples weight matrices into non-overlapping blocks of size \(g\) and computes optimal \(\ell_1\) median scales for ternary masks \(\{-1, 0, +1\}\), reducing \(\ell_1\) quantization error by **33.3%–80.2%** compared to standard global `Mean-Abs` scaling while suppressing heavy-tailed outliers.

---

## Quick Start

### 1. Build the Linked Lean 4 Certifier
```bash
lake update
lake exe cache get
lake build
lake exe vcore_certifier --verify-bench
```

### 2. Run Hybrid Quantization in Python / PyTorch
```python
import torch
from vcore_l1.cuda_quantizer import VCoreTernaryQuantizer, verify_with_lean4_engine

# 1. Verify mathematical optimality guarantees via the linked Lean 4 binary
print(verify_with_lean4_engine())

# 2. Quantize a neural network weight matrix into 1.58-bit {-1, 0, +1}
weights = torch.randn(64, 128)
quantizer = VCoreTernaryQuantizer(group_size=32, theta=1e-3)
w_quant, ternary_mask, scales = quantizer.quantize(weights)
```
