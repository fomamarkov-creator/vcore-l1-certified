# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Dict, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cuda_quantizer import VCoreTernaryQuantizer
from .inference_layer import VCoreLinear
from .super_quantizer import fwht_8, quantize_e8_lattice


class VCoreQATLinear(nn.Module):
    """
    Обучаемый линейный слой для тренировки с нуля (Pretraining) и QAT-дообучения
    напрямую в 1.58-bit (ℓ₁-медиана) или Sub-1-Bit решётке E₈ с вращением Адамара W₈.
    Использует Straight-Through Estimator (STE) с проксимальным нерастяжением.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 32,
        mode: str = "ternary",  # "ternary" (1.58-bit) или "e8" (0.989 b/w + FWHT)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.mode = mode

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.quantizer = VCoreTernaryQuantizer(group_size=group_size, theta=1e-4, auto_theta=True)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 32, mode: str = "ternary") -> "VCoreQATLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            group_size=group_size,
            mode=mode,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def _quantize_ste(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.mode == "e8" and (self.in_features % 8 == 0):
                w_q, _, _ = quantize_e8_lattice(w, use_hadamard=True)
            else:
                w_q, _, _ = self.quantizer.quantize(w)
        # Straight-Through Estimator: на forward идёт точный квантованный вес w_q,
        # на backward градиент проходит напрямую в master-веса self.weight
        return w + (w_q - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste = self._quantize_ste(self.weight)
        return F.linear(x, w_ste, self.bias)

    def to_fp_linear(self) -> nn.Linear:
        """Конвертирует обученный QAT-слой в обычный nn.Linear с квантованными весами для экспорта в vcore-pack."""
        out = nn.Linear(self.in_features, self.out_features, bias=(self.bias is not None))
        with torch.no_grad():
            if self.mode == "e8" and (self.in_features % 8 == 0):
                w_q, _, _ = quantize_e8_lattice(self.weight, use_hadamard=True)
            else:
                w_q, _, _ = self.quantizer.quantize(self.weight)
            out.weight.copy_(w_q)
            if self.bias is not None:
                out.bias.copy_(self.bias)
        return out


class VCoreQATLoRALinear(nn.Module):
    """
    Сверхлёгкий адаптер QAT-LoRA поверх замороженного сжатого слоя `VCoreLinear`:
    позволяет дообучать сжатую модель на 4–6 ГБ VRAM без распаковки весов.
    """

    def __init__(self, base_vcore: VCoreLinear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_vcore
        for p in self.base.parameters():
            p.requires_grad = False
        for b in self.base.buffers():
            b.requires_grad = False

        self.rank = rank
        self.scaling = alpha / float(rank)
        self.lora_a = nn.Parameter(torch.randn(rank, base_vcore.in_features) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(base_vcore.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def convert_model_to_qat(
    model: nn.Module,
    group_size: int = 32,
    mode: str = "ternary",
    lora_rank: int = 0,
    skip_names: Iterable[str] = ("lm_head",),
) -> nn.Module:
    """
    Переводит любую PyTorch / HuggingFace модель в режим обучения V-CORE QAT:
    - Если слой уже сжат в `VCoreLinear` и `lora_rank > 0` — вешает обучаемый `VCoreQATLoRALinear`;
    - Если слой обычный `nn.Linear` — заменяет на `VCoreQATLinear` для обучения с нуля / QAT.
    """
    skip_set = set(skip_names)
    for full_name, module in list(model.named_modules()):
        if any(s in full_name for s in skip_set):
            continue
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if isinstance(module, VCoreLinear) and lora_rank > 0:
            setattr(parent, parts[-1], VCoreQATLoRALinear(module, rank=lora_rank))
        elif isinstance(module, nn.Linear):
            setattr(parent, parts[-1], VCoreQATLinear.from_linear(module, group_size=group_size, mode=mode))
    return model


def train_vcore_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    l1_weight_decay: float = 1e-4,
) -> float:
    """Выполняет один шаг обучения QAT / Pretraining с проксимальным ℓ₁-регуляризатором."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss if hasattr(outputs, "loss") else F.cross_entropy(
        outputs.view(-1, outputs.size(-1)), labels.view(-1)
    )

    if l1_weight_decay > 0.0:
        l1_pen = torch.tensor(0.0, device=input_ids.device)
        for p in model.parameters():
            if p.requires_grad:
                l1_pen = l1_pen + torch.mean(torch.abs(p))
        loss = loss + l1_weight_decay * l1_pen

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())
