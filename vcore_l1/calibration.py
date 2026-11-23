# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Dict, List, Optional
import torch
import torch.nn as nn


DEFAULT_CALIB_PROMPTS = [
    "Explain the L1 metric projection theorem in non-reflexive Banach spaces.",
    "Write a Python function to compute the exact median of a rational vector.",
    "Large language models use transformer attention layers and feed-forward networks.",
    "Formal verification in Lean 4 guarantees mathematical correctness without floating-point drift.",
]


@torch.no_grad()
def collect_activation_importance(
    model_path_or_repo: str,
    num_samples: int = 4,
    seq_len: int = 64,
    token: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Прогоняет num_samples калибровочных текстов через модель Hugging Face и собирает
    среднюю амплитуду входных активаций E[|x_j|] для каждого линейного слоя nn.Linear.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo, torch_dtype=dtype, token=token
    ).to(device).eval()

    acc_norms: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook_fn(_module, inp, _out):
            if not inp or not isinstance(inp[0], torch.Tensor):
                return
            x = inp[0].detach().to(torch.float32)
            x_2d = x.reshape(-1, x.size(-1))
            mean_abs = torch.mean(torch.abs(x_2d), dim=0).cpu()
            if layer_name not in acc_norms:
                acc_norms[layer_name] = mean_abs
                counts[layer_name] = 1
            else:
                acc_norms[layer_name] += mean_abs
                counts[layer_name] += 1
        return hook_fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(f"{name}.weight")))

    prompts: List[str] = (DEFAULT_CALIB_PROMPTS * ((num_samples // len(DEFAULT_CALIB_PROMPTS)) + 1))[:num_samples]
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        _ = model(**enc)

    for h in hooks:
        h.remove()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {k: (acc_norms[k] / max(counts[k], 1)) for k in acc_norms}
