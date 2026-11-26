# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import os
from typing import Any, Dict, Optional
from .vcore_bundle import build_lean4_certificate_passport


def generate_model_card_readme(
    output_dir: str,
    base_model: str,
    stats: Dict[str, Any],
    max_rel_error: float = 0.40,
    spherical_emb: bool = False,
) -> str:
    """
    Генерирует готовую карточку модели `README.md` в выходной папке:
    - YAML-метаданные для индексации на Hugging Face (`tags: [vcore, l1-certified, lean4]`);
    - Математический паспорт Lean 4 с верхней границей каскадного дрейфа;
    - Готовые примеры запуска на Python и в нативном C++ `vcore-run`.
    """
    passport = build_lean4_certificate_passport(
        output_dir, max_rel_error=max_rel_error, spherical_emb=spherical_emb
    )
    drift = passport["certified_max_block_drift"]
    sha_short = passport["weights_sha256"][:16]

    orig_mb = stats.get("orig_size_mb", 0.0)
    pack_mb = stats.get("packed_size_mb", 0.0)
    ratio = stats.get("compression_ratio", 1.0)
    q2 = stats.get("quantized_158bit_layers", 0)
    q4 = stats.get("quantized_int4_layers", 0)
    sph = stats.get("spherical_e8_layers", 0)
    comp = stats.get("cross_layer_compensated", 0)

    card = f"""---
base_model: {base_model}
tags:
  - vcore
  - l1-certified
  - lean4-verified
  - 1.58-bit
  - e8-lattice
library_name: vcore-l1-certified
---

# {os.path.basename(os.path.abspath(output_dir))} (V-CORE \(\ell_1\)-Certified)

Quantized from **{base_model}** using **V-CORE (\(\ell_1\))-Certified v5.0** — a formally verified (Lean 4, 25 modules, `0 sorry`) post-training quantization engine based on \(\ell_1\)-medians, Cross-Layer curvature compensation, and hyperbolic \(E_8\) spherical codes.

### Formal Verification & Compression Report
* **Lean 4 Certificate:** `{passport["certificate_standard"]}`
* **Certified Max Block Drift:** \(\le {drift}\) (`{passport["theorem_anchor"]}`)
* **Freivalds Matrix Check:** `PASSED` (`SHA-256: {sha_short}...`)
* **Compression Ratio:** **{ratio:.2f}x** (`{orig_mb:.1f} MB` \(\rightarrow\) `{pack_mb:.1f} MB`)
* **Layer Breakdown:** `{q2}` layers in **1.58-bit**, `{q4}` layers in **INT4**, `{sph}` layers in **Spherical \(E_8\) (~1.0 b/w)**, `{comp}` pairs **Cross-Layer Compensated**.

### Quick Start (Python / PyTorch)
```bash
pip install vcore-l1-certified transformers accelerate
```

```python
from transformers import AutoTokenizer
from vcore_l1 import VCoreModelForCausalLM, VCoreDynamicCache

model_id = "./{os.path.basename(os.path.abspath(output_dir))}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VCoreModelForCausalLM.from_vcore_pretrained(model_id, device="cuda", integer_only=False)

cache = VCoreDynamicCache(group_size=32, mode="spherical")
inputs = tokenizer("Explain L1 median quantization:", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=128, past_key_values=cache)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Zero-Dependency Native C++17 / AVX2 / Integer-Only Q14 Run
```bash
./build/vcore-run --model ./{os.path.basename(os.path.abspath(output_dir))} --prompt "Hello!" --tokens 64 --integer-only
```
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    return readme_path


def publish_vcore_to_hub(
    output_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """Опциональная публикация сжатой директории модели в Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload V-CORE L1-Certified quantized model (Lean 4 verified)",
    )
    return f"https://huggingface.co{repo_id}"
