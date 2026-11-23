[![V-CORE L1-Certified CI & CUDA Build](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v2.0.0`) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / CUDA (`CuPy` + `DLPack`) | **License:** AGPL-3.0

# V-CORE \(\ell_1\)-Certified: Hybrid Lean 4 + GPU 1.58-Bit Neural Network Quantizer

A hybrid symbolic-numeric quantization engine combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v2.0.0`) with a high-speed **GPU/CUDA execution backend** (`LinearSelector` + blockwise \(\Delta = 1\) median scaling) and a **layerwise `.safetensors` 2-bit packer** for 1.58-bit (`{-1, 0, +1}`) neural network compression without retraining.

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Repository Architecture

1. **Formal Certification Engine (`CertifierMain.lean` \(\to\) `vcore_certifier`):**
   * Directly imports `MarkovL1Complexity`, `MarkovL1Solver`, `MarkovL1WeightedSolver`, and `MarkovL1Quantizer` (`v2.0.0`) via Lake.
   * Executes exact-rational (\(\mathbb{Q}\)) \(\ell_1\)-optimality proofs without `Classical.choice`, certifying that blockwise median scales \(\alpha_b^* = \mathrm{median}(\vert{}w_i\vert{}_{i \in S_b})\) achieve the global minimum \(\ell_1\) quantization error in the decoupled regime (\(\Delta = 1\)).
2. **GPU / PyTorch Quantizer (`vcore_l1/cuda_quantizer.py`):**
   * **Stage 1 (`LinearSelector` CUDA Kernel):** Hard-threshold noise pruning (\(\vert{}w_i\vert{} \le \theta \to 0\)).
   * **Stage 2 (\(\Delta = 1\) Blockwise Median Scaling):** Computes optimal per-block \(\ell_1\) median scales for ternary masks \(\{-1, 0, +1\}\), reducing quantization error by **33.3%–80.2%** compared to global `Mean-Abs` scaling.
3. **Layerwise `.safetensors` Repacker (`vcore_l1/safetensors_quantizer.py`):**
   * Iterates over full model checkpoints, preserves sensitive layers (`layernorm`, `bias`, `embed`, `lm_head`), and packs linear weight matrices into 2-bit `uint8` tensors (4 ternary weights per byte) + `float16` block scales (`~6–7x` disk compression).
4. **Native CUDA Library (`cuda/VCoreEngine.cpp`, `cuda/vcore_quantize_kernel.cu`, `CMakeLists.txt`):**
   * Compiles into `libvcore_engine.so` for direct low-level GPU execution.

---

## Installation & Quick Start

### 1. Automatic Setup (Linux / Google Colab)
```bash
git clone https://github.com
cd vcore-l1-certified
bash setup.sh
```

### 2. Run the Formal Lean 4 Certifier
```bash
lake update
lake exe cache get
lake build
lake exe vcore_certifier --verify-bench
```

---

## Usage Guide (Python / PyTorch)

### A. Quantize a Single Layer Tensor (`{-1, 0, +1}`)
```python
import torch
from vcore_l1 import VCoreTernaryQuantizer, verify_with_lean4_engine

# Verify formal mathematical guarantees via Lean 4
print(verify_with_lean4_engine())

# Quantize a 2D weight matrix into 1.58-bit ternary values
weights = torch.randn(128, 256)
quantizer = VCoreTernaryQuantizer(group_size=32, theta=1e-4)
w_quant, ternary_mask, scales = quantizer.quantize(weights)
```

### B. Adaptive Layerwise Compression of a Full `.safetensors` Model (with Auto-Rollback)
```python
from vcore_l1 import repack_safetensors_layerwise

stats = repack_safetensors_layerwise(
    input_path="model.safetensors",
    output_path="model_vcore_158bit.safetensors",
    group_size=32,
    theta=1e-4,
    max_rel_error=0.45,
    include_aux_layers=True,
)
print(stats)
```


### C. Unpack Weights for Inference
```python
from safetensors.torch import load_file
from vcore_l1 import unpack_uint8_to_ternary

data = load_file("model_vcore_158bit.safetensors")
mask = unpack_uint8_to_ternary(
    data["model.layers.0.self_attn.q_proj.weight.vcore_packed"],
    tuple(data["model.layers.0.self_attn.q_proj.weight.vcore_shape"].tolist()),
)
scales = data["model.layers.0.self_attn.q_proj.weight.vcore_scales"].to(mask.dtype)
w_restored = (mask.reshape(-1, 32) * scales.unsqueeze(1)).reshape(mask.shape)
```
