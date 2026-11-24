[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Lean Core:** `l1-projection-complexity-lean4` (`v3.0.0` — 8 Verified Modules) | **Toolchain:** Lean `v4.37.0-rc1` | **Backend:** PyTorch / Warp-CUDA (`sm_70`–`sm_90`) | **License:** AGPL-3.0

# V-CORE (ℓ₁)-Certified: Hybrid Lean 4 + GPU 1.58-Bit & INT4 Quantization & Inference Engine

A complete symbolic-numeric compression and inference runtime combining a **machine-checked Lean 4 formal verification core** (`l1-projection-complexity-lean4`, release `v3.0.0`) with **activation-weighted \(\ell_1\) medians**, **bounded-treewidth (Δ = 2) FPT dynamic programming**, **channel-outlier permutation**, **multi-stage precision cascade (`1.58-bit -> INT4 -> FP16`) with Auto-Rollback**, **2-bit KV-cache compression**, and **warp-parallel CUDA kernels** (`libvcore_engine.so`).

* **Author:** E. S. Markov (ORCID: `0009-0005-2235-5464`)
* **Affiliation:** Independent Researcher, Astrakhan, Russia
* **License:** GNU Affero General Public License v3.0 (AGPLv3)

---

## Full Stack Capabilities

1. **Formal Lean 4 Verification (`CertifierMain.lean` → `vcore_certifier`):**
   * Links directly against all 8 verified modules of `l1-projection-complexity-lean4` (`v3.0.0`)—including `MarkovL1TernaryQuantizer`, `MarkovL1WeightedTernary`, and `MarkovL1TreewidthSolver`—to certify exact-rational (\(\mathbb{Q}\)) optimality prior to GPU execution.
2. **Activation-Weighted \(\ell_1\) Medians, Treewidth FPT & Channel Outlier Grouping (`vcore_l1/`):**
   * Sorts weight columns by \(\ell_1\)-norm (`vcore_perm`) to isolate heavy-tailed outliers into dedicated Δ = 1 blocks.
   * Supports activation-importance weights (`--calib-samples`, \(\gamma_j = \mathbb{E}[\vert{}x_j\vert{}]\)) to protect salient input channels via exact weighted \(\ell_1\) medians, plus `solve_chain_overlap_fpt` for structured Δ = 2 chain overlaps.
3. **Multi-Stage Precision Cascade & VRAM Budgeter (`vcore_l1/safetensors_quantizer.py`, `vcore-pack`):**
   * Automatically steps down block sizes (`64 -> 32 -> 16` in 1.58-bit), falls back to **4-bit blockwise `INT4`** (`{-7..+7}`), and rolls back sensitive 1D layers (`LayerNorm`, `bias`) to `FP16` under `--max-rel-error` or `--target-vram-gb`.
   * Supports local `.safetensors`, sharded directories (`model.safetensors.index.json`), and **streaming Hugging Face Hub compression** (`--hf-repo`).
4. **One-Line Model Loader, 2-Bit KV-Cache & Warp-Parallel CUDA (`vcore_l1/inference_layer.py`, `cuda/`):**
   * Loads compressed models in one line via `VCoreModelForCausalLM.from_vcore_pretrained(...)`.
   * Compresses autoregressive context memory on the fly via `VCoreDynamicCache`.
   * Runs `vcore_fused_ternary_gemv_warp` (32-thread warp reduction with `__shfl_down_sync` and 32-bit `uint32` register loads) directly on 2-bit packed `uint8` weights without unpacking to `float32` in VRAM (`sm_70`–`sm_90`).

---



## Benchmarks (NVIDIA Tesla T4 GPU)

### 1. Warp-Parallel CUDA Kernel (`libvcore_engine.so`, `vcore-bench`)

| Layer Dimensions | FP16/32 Memory | V-CORE 1.58-Bit Memory | Memory Reduction | Throughput | Latency / Token |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1024 × 1024** | 2,048.0 KB | **320.0 KB** | **6.4×** | **2,424.6 steps/s** | **0.41 ms** |
| **2048 × 2048** | 8,192.0 KB | **1,280.0 KB** | **6.4×** | **1,486.7 steps/s** | **0.67 ms** |
| **4096 × 4096** | 32,768.0 KB | **5,120.0 KB** | **6.4×** | **364.4 steps/s** | **2.74 ms** |

### 2. Solver Accuracy vs. BitNet (`absmean`) & Bounded-Treewidth FPT

Evaluated on real projection weights (`Qwen2.5-0.5B-Instruct`, `layers.0.mlp.gate_proj`):

| Solver / Method | Total ℓ₁ Error | Relative Gain | Runtime (Tesla T4) |
| :--- | :---: | :---: | :---: |
| **BitNet `absmean` (1.58-bit)** | 36,105.3 | Baseline | — |
| **V-CORE Exact Critical-Threshold (1.58-bit)** | **31,813.6** | **-11.9% ℓ₁ error** | — |
| **Bounded-Treewidth (Δ=2) FPT (`m=512, |G|=17`)** | 7.695 (exact DP min) | Global optimum | **114.04 ms** |

### 3. End-to-End LLM Compression & Inference (`VCoreDynamicCache`)

| Model | Original Size | Packed Size | Compression | Rel. ℓ₁ Error | Inference Quality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen/Qwen2.5-0.5B-Instruct** | 942.3 MB | **453.7 MB** | **2.08×** | 10.76% | Coherent (8.4 tok/s) |
| **Qwen/Qwen2.5-1.5B-Instruct** | 2,944.4 MB | **1,152.3 MB** | **2.56×** | 10.63% | Coherent |

### 4. Mathematical Fidelity & GGUF Evaluation (`Qwen/Qwen2.5-1.5B-Instruct`)

Measured across all 151,936 vocabulary logits against the unquantized FP16 baseline (`2,944.4 MB` → `1,152.3 MB` V-CORE / `1,599.9 MB` GGUF):

| Metric | FP16 Baseline | V-CORE Compressed | Delta / Fidelity |
| :--- | :---: | :---: | :---: |
| **Perplexity (PPL)** | 14.646 | **15.254** | **+0.607 (+4.1%)** |
| **Logit Cosine Similarity (151,936 vocab)** | 1.0000 | **0.9617** | **96.2% alignment** |
| **Native `llama.cpp` GGUF Runtime** | — | **Verified (`Q4_0` + `F16`)** | 5.7 tok/s prompt · 4.6 tok/s gen |


## Installation & Quick Start

### 1. Clone & Build Everything
```bash
git clone https://github.com/fomamarkov-creator/vcore-l1-certified
cd vcore-l1-certified
bash setup.sh
```

### 2. One-Line Model Compression (`vcore-pack`)
```bash
# Data-free streaming compression from Hugging Face Hub under a 4 GB VRAM budget
vcore-pack --hf-repo Qwen/Qwen2.5-1.5B --output ./qwen25_vcore --target-vram-gb 4.0

# Activation-aware weighted 1.58-bit compression using 4 calibration samples
vcore-pack --hf-repo Qwen/Qwen2.5-1.5B --output ./qwen25_vcore_aw --calib-samples 4 --target-vram-gb 4.0

# Compress a local sharded model folder
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
### 5. Export Compressed Models to Standard GGUF (`vcore-export-gguf`)

Convert any V-CORE compressed model directory directly into a `.gguf` file (`TQ2_0` / `Q4_0` + `F16` with full tokenizer metadata) for local deployment in **llama.cpp**, **Ollama**, or **LM Studio**:

```bash
# Export an existing V-CORE compressed model directory to GGUF
vcore-export-gguf --input ./qwen25_vcore --output ./qwen25_vcore.gguf

# Or compress from Hugging Face Hub and export to GGUF in a single command
vcore-pack --hf-repo Qwen/Qwen2.5-1.5B-Instruct --output ./qwen15_vcore --max-rel-error 0.22 --export-gguf ./qwen15_vcore.gguf
```
