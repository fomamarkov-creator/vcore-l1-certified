[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v3.0.0` — 8 Verified Modules) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / Warp-CUDA (`sm_70`–`sm_90`) | **License:** AGPL-3.0

# V-CORE \(\ell_1\)-Certified: Hybrid Lean 4 + GPU 1.58-Bit & INT4 Quantization & Inference Engine

A complete symbolic-numeric compression and inference runtime combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v3.0.0`) with **activation-weighted \(\ell_1\) medians**, **channel-outlier permutation**, **multi-stage precision cascade (`1.58-bit -> INT4 -> FP16`) with Auto-Rollback**, **2-bit KV-cache compression**, and **warp-parallel CUDA kernels** (`libvcore_engine.so`).

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Full Stack Capabilities

1. **Formal Lean 4 Verification (`CertifierMain.lean` → `vcore_certifier`):**
   * Links directly against all 8 verified modules of `l1-projection-complexity-lean4` (`v3.0.0`)—including `MarkovL1TernaryQuantizer`, `MarkovL1WeightedTernary`, and `MarkovL1TreewidthSolver`—to certify exact-rational (\(\mathbb{Q}\)) optimality prior to GPU execution.
2. **Activation-Weighted \(\ell_1\) Medians & Channel Outlier Grouping (`vcore_l1/`):**
   * Sorts weight columns by \(\ell_1\)-norm (`vcore_perm`) to isolate heavy-tailed outliers into dedicated Δ = 1 blocks.
   * Supports activation-importance weights (`activation_scales`, \(\gamma_j = \mathbb{E}[\vert{}x_j\vert{}]\)) to protect salient input channels via exact weighted \(\ell_1\) medians.
3. **Multi-Stage Precision Cascade & VRAM Budgeter (`vcore_l1/safetensors_quantizer.py`, `vcore-pack`):**
   * Automatically steps down block sizes (`64 -> 32 -> 16` in 1.58-bit), falls back to **4-bit blockwise `INT4`** (`{-7..+7}`), and rolls back sensitive 1D layers (`LayerNorm`, `bias`) to `FP16` under `--max-rel-error` or `--target-vram-gb`.
   * Supports local `.safetensors`, sharded directories (`model.safetensors.index.json`), and **streaming Hugging Face Hub compression** (`--hf-repo`).
4. **One-Line Model Loader, 2-Bit KV-Cache & Warp-Parallel CUDA (`vcore_l1/inference_layer.py`, `cuda/`):**
   * Loads compressed models in one line via `VCoreModelForCausalLM.from_vcore_pretrained(...)`.
   * Compresses autoregressive context memory on the fly via `VCoreDynamicCache`.
   * Runs `vcore_fused_ternary_gemv_warp` (32-thread warp reduction with `__shfl_down_sync` and 32-bit `uint32` register loads) directly on 2-bit packed `uint8` weights without unpacking to `float32` in VRAM (`sm_70`–`sm_90`).

---

## Installation & Quick Start

### 1. Clone & Build Everything
```bash
git clone https://github.com/fomamarkov-creator/vcore-l1-certified
cd vcore-l1-certified
bash setup.sh
```

### 2. One-Line Model Compression (`vcore-pack`)
```bash
# Stream and compress a model directly from Hugging Face Hub under a 4 GB VRAM budget
vcore-pack --hf-repo Qwen/Qwen2.5-1.5B --output ./qwen25_vcore --target-vram-gb 4.0

# Or compress a local sharded model folder
vcore-pack --input ./hf_model_dir --output ./model_vcore --group-size 64 --max-rel-error 0.40
```

### 3. Run Inference & 2-Bit KV-Cache in PyTorch / Transformers
```python
from vcore_l1 import VCoreModelForCausalLM, VCoreDynamicCache, verify_with_lean4_engine

# 1. Run formal Lean 4 optimality verification
print(verify_with_lean4_engine())

# 2. Load compressed model in one line
model = VCoreModelForCausalLM.from_vcore_pretrained("./qwen25_vcore", device="cuda")

# 3. Generate with 2-bit compressed KV-cache
kv_cache = VCoreDynamicCache(group_size=32)
outputs = model.generate(**inputs, past_key_values=kv_cache, max_new_tokens=128)
```

### 4. Benchmark Speed & Memory (`vcore-bench`)
```bash
vcore-bench --hidden-size 2048 --out-size 2048 --steps 100
```
