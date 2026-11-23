[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v2.0.0`) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / CUDA (`sm_70`–`sm_90`) | **License:** AGPL-3.0

# V-CORE \(\ell_1\)-Certified: Hybrid Lean 4 + GPU 1.58-Bit Neural Network Quantizer

A complete symbolic-numeric compression and inference engine combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v2.0.0`) with **blockwise adaptive \(\theta_b^*\) noise pruning**, **\(\Delta = 1\) optimal \(\ell_1\) median scaling**, **sharded `.safetensors` 2-bit repacking with Auto-Rollback**, and **Fused CUDA kernels** (`vcore_fused_ternary_gemv`).

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Key Capabilities

1. **Formal Lean 4 Verification (`CertifierMain.lean` \(\to\) `vcore_certifier`):**
   * Directly links to `MarkovL1Complexity`, `MarkovL1Solver`, `MarkovL1WeightedSolver`, and `MarkovL1Quantizer` (`v2.0.0`) to verify global \(\ell_1\)-optimality over exact rationals \(\mathbb{Q}\) without `Classical.choice`.
2. **Per-Block Adaptive Thresholding (\(\theta_b^*\)) & \(\Delta = 1\) Median Scaling (`vcore_l1/cuda_quantizer.py`):**
   * Automatically selects the optimal pruning threshold \(\theta_b^* \in \{\theta, 0.25 m_b, 0.50 m_b\}\) per block to minimize \(\ell_1\) quantization error (`{-1, 0, +1}`).
3. **Sharded Hugging Face Model Packer with Auto-Rollback (`vcore_l1/safetensors_quantizer.py` & `vcore-pack`):**
   * Compresses both single `.safetensors` checkpoints and multi-shard models (`model.safetensors.index.json`) into 2-bit `uint8` tensors (4 ternary weights per byte), automatically rolling back sensitive layers (`LayerNorm`, `bias`) to `FP16` if relative \(\ell_1\) error exceeds `max_rel_error`.
4. **Fused 2-Bit CUDA Kernel (`cuda/vcore_quantize_kernel.cu` \(\to\) `libvcore_engine.so`):**
   * Executes matrix-vector inference (`vcore_fused_ternary_gemv`) directly on 2-bit packed `uint8` weights in GPU registers without unpacking to `float32` in VRAM (compiled for NVIDIA architectures `sm_70` through `sm_90`).

---

## Quick Start & CLI Usage

### 1. Install & Build
```bash
git clone https://github.com
cd vcore-l1-certified
bash setup.sh
```

### 2. One-Line Model Compression via CLI (`vcore-pack`)
Compress a single `.safetensors` file or an entire sharded Hugging Face model directory:
```bash
vcore-pack --input ./path_to_hf_model --output ./model_vcore_158bit --group-size 32 --max-rel-error 0.40
```

### 3. Python API & Formal Lean 4 Check
```python
import torch
from vcore_l1 import VCoreTernaryQuantizer, verify_with_lean4_engine

# Verify formal Lean 4 theorems
print(verify_with_lean4_engine())

# Quantize tensor with per-block adaptive theta_b* and Delta=1 median
weights = torch.randn(128, 256)
quantizer = VCoreTernaryQuantizer(group_size=32, auto_theta=True)
w_quant, ternary_mask, scales = quantizer.quantize(weights)
```
