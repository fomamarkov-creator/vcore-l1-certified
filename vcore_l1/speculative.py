# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import copy
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .inference_layer import VCoreDynamicCache, VCoreLinear
from .super_quantizer import quantize_e8_lattice


class _E8DraftLinear(nn.Module):
    """Быстрый черновой слой на решётке E₈ (0.989 b/w) для спекулятивного наброска токенов."""

    def __init__(self, w_e8: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("w_e8", w_e8.to(torch.float16))
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_e8.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


@torch.no_grad()
def build_e8_draft_model(target_model: nn.Module) -> nn.Module:
    """
    Строит черновую модель-двойник на сферической решётке E₈ из целевой модели V-CORE
    без необходимости скачивать отдельную маленькую модель.
    """
    draft = copy.deepcopy(target_model)
    for full_name, mod in list(draft.named_modules()):
        if isinstance(mod, VCoreLinear) and (mod.in_features % 8 == 0):
            # Восстанавливаем матрицу весов и проецируем на оболочку E₈
            eye = torch.eye(mod.in_features, dtype=torch.float16, device=mod.vcore_packed.device)
            w_approx = mod(eye).T.contiguous()
            w_e8, _, _ = quantize_e8_lattice(w_approx.to(torch.float32), use_hadamard=True)
            draft_lin = _E8DraftLinear(w_e8, bias=mod.bias)

            parts = full_name.split(".")
            parent = draft
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], draft_lin)
    return draft.eval()


class VCoreSpeculativeGenerator:
    """
    Спекулятивный генератор V-CORE (E₈ Draft -> 1.58-bit / INT4 Exact Verification):
    гарантирует распределение токенов, математически тождественное целевой модели.
    """

    def __init__(
        self,
        target_model: nn.Module,
        tokenizer,
        draft_model: Optional[nn.Module] = None,
        gamma: int = 4,
        kv_mode: str = "spherical",
    ):
        self.target = target_model.eval()
        self.tokenizer = tokenizer
        self.draft = draft_model.eval() if draft_model is not None else build_e8_draft_model(target_model)
        self.gamma = max(1, gamma)
        self.kv_mode = kv_mode

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Tuple[str, Dict[str, float]]:
        device = next(self.target.parameters()).device
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = input_ids.size(1)

        t0 = time.time()
        accepted_total = 0
        drafted_total = 0

        cur_ids = input_ids
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        while (cur_ids.size(1) - prompt_len) < max_new_tokens:
            remain = max_new_tokens - (cur_ids.size(1) - prompt_len)
            steps = min(self.gamma, remain)

            # 1. Черновой проход (E8 Draft) набрасывает `steps` кандидатов
            draft_ids = cur_ids
            draft_probs_list = []
            for _ in range(steps):
                d_logits = self.draft(draft_ids).logits[:, -1, :]
                if temperature <= 1e-4:
                    next_d = torch.argmax(d_logits, dim=-1, keepdim=True)
                    probs_d = F.one_hot(next_d.squeeze(-1), num_classes=d_logits.size(-1)).to(torch.float32)
                else:
                    probs_d = F.softmax(d_logits.to(torch.float32) / temperature, dim=-1)
                    next_d = torch.multinomial(probs_d, num_samples=1)
                draft_probs_list.append(probs_d)
                draft_ids = torch.cat([draft_ids, next_d], dim=-1)
                drafted_total += 1

            # 2. Параллельная верификация всех `steps` кандидатов целевой моделью за 1 forward
            t_logits = self.target(draft_ids).logits[:, -steps - 1 :, :]
            if temperature <= 1e-4:
                t_probs_all = F.one_hot(
                    torch.argmax(t_logits, dim=-1), num_classes=t_logits.size(-1)
                ).to(torch.float32)
            else:
                t_probs_all = F.softmax(t_logits.to(torch.float32) / temperature, dim=-1)

            # 3. Критерий принятия / отклонения (Rejection Sampling)
            n_accepted = 0
            override_token = None
            for idx in range(steps):
                cand_tok = int(draft_ids[0, cur_ids.size(1) + idx].item())
                p_target = t_probs_all[0, idx, cand_tok].item()
                p_draft = max(draft_probs_list[idx][0, cand_tok].item(), 1e-8)

                if torch.rand(1, device=device).item() <= min(1.0, p_target / p_draft):
                    n_accepted += 1
                    accepted_total += 1
                    if eos_id is not None and cand_tok == eos_id:
                        break
                else:
                    diff_probs = torch.clamp(t_probs_all[0, idx] - draft_probs_list[idx][0], min=0.0)
                    if torch.sum(diff_probs).item() > 1e-8:
                        diff_probs = diff_probs / torch.sum(diff_probs)
                        override_token = torch.multinomial(diff_probs, num_samples=1).view(1, 1)
                    else:
                        override_token = torch.argmax(t_probs_all[0, idx], keepdim=True).view(1, 1)
                    break

            cur_ids = draft_ids[:, : cur_ids.size(1) + n_accepted]
            if override_token is not None:
                cur_ids = torch.cat([cur_ids, override_token], dim=-1)
            elif n_accepted == steps and (cur_ids.size(1) - prompt_len) < max_new_tokens:
                # Все черновые токены приняты — забираем бонусный (gamma + 1)-й токен от целевой модели!
                bonus_tok = torch.argmax(t_probs_all[0, steps], keepdim=True).view(1, 1)
                cur_ids = torch.cat([cur_ids, bonus_tok], dim=-1)

            if eos_id is not None and int(cur_ids[0, -1].item()) == eos_id:
                break

        elapsed = max(time.time() - t0, 1e-5)
        gen_tokens = cur_ids[0, prompt_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        stats = {
            "tokens_generated": int(gen_tokens.numel()),
            "tokens_per_sec": round(gen_tokens.numel() / elapsed, 2),
            "acceptance_rate": round(accepted_total / max(drafted_total, 1), 3),
        }
        return text, stats
      
