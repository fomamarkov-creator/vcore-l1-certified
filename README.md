[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v2.0.0`) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / CUDA (`sm_70`–`sm_90`) | **License:** AGPL-3.0

# V-CORE \(\ell_1\)-Certified: Hybrid Lean 4 + GPU 1.58-Bit & INT4 Quantization & Inference Engine

A complete symbolic-numeric compression and inference runtime combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v2.0.0`) with **channel-outlier permutation**, **per-block adaptive \(\theta_b^*\) noise pruning**, **multi-stage precision cascade (`1.58-bit -> INT4 -> FP16`) with Auto-Rollback**, **2-bit KV-cache compression**, and **Fused `uint32` / `__shared__` CUDA kernels** (`libvcore_engine.so`).

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Full Stack Capabilities

1. **Formal Lean 4 Verification (`CertifierMain.lean` \(\to\) `vcore_certifier`):**
   * Links directly against `MarkovL1Complexity`, `MarkovL1Solver`, `MarkovL1WeightedSolver`, and `MarkovL1Quantizer` (`v2.0.0`) to certify global \(\ell_1\)-optimality over exact rationals \(\mathbb{Q}\) without `Classical.choice`.
2. **Channel Outlier Grouping & Adaptive \(\theta_b^*\) Pruning (`vcore_l1/cuda_quantizer.py`):**
   * Sorts weight columns by \(\ell_1\)-norm (`vcore_perm`) to isolate heavy-tailed outliers into dedicated \(\Delta = 1\) blocks, then selects the optimal pruning threshold \(\theta_b^*\) and median scale \(\alpha_b^*\) per block.
3. **Multi-Stage Precision Cascade & VRAM Budgeter (`vcore_l1/safetensors_quantizer.py`, `vcore-pack`):**
   * Automatically steps down block sizes (`64 -> 32 -> 16` in 1.58-bit), falls back to **4-bit INT4** (`{-7..+7}`), and rolls back sensitive 1D layers (`LayerNorm`, `bias`) to `FP16` under `--max-rel-error` or `--target-vram-gb`.
   * Supports local `.safetensors`, sharded directories (`model.safetensors.index.json`), and **streaming Hugging Face Hub compression** (`--hf-repo`, deleting raw shards on the fly to save up to 80% disk space).
4. **One-Line Model Loader, KV-Cache Compression & Fused CUDA (`vcore_l1/inference_layer.py`, `cuda/`):**
   * Loads compressed models in one line via `VCoreModelForCausalLM.from_vcore_pretrained(...)`.
   * Compresses autoregressive context memory on the fly via `VCoreDynamicCache`.
   * Runs `vcore_fused_ternary_gemv` / `vcore_fused_ternary_gemm` directly on 2-bit packed `uint8` weights using 32-bit `uint32` register loads and `__shared__` input caching (`sm_70`–`sm_90`).

---

## Installation & Quick Start

### 1. Clone & Build Everything
```bash
git clone https://github.com
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
