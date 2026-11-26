# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import json
import os
from typing import List
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .hardware_probe import probe_hardware
from .inference_layer import VCoreModelForCausalLM
from .qat_trainer import VCoreQATLinear, convert_model_to_qat, train_vcore_step


def _load_texts_from_file(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("prompt") or obj.get("content") or str(obj)
                texts.append(txt)
            else:
                texts.append(line)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE QAT & Pretraining Engine (vcore-train): обучение в 1.58-bit / E8"
    )
    parser.add_argument("--model", "-m", required=True, help="HF Repo, локальная папка модели или сжатая модель V-CORE")
    parser.add_argument("--data", "-d", required=True, help="Путь к обучающему файлу (.txt или .jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Куда сохранить обученную модель")
    parser.add_argument("--from-scratch", action="store_true", help="Инициализировать веса с нуля по config.json")
    parser.add_argument("--mode", choices=["ternary", "e8"], default="ternary", help="Решётка обучения: 1.58-bit или E8 (0.989 b/w)")
    parser.add_argument("--lora-rank", type=int, default=0, help="Ранг QAT-LoRA (если > 0, обучает адаптеры поверх сжатых весов)")
    parser.add_argument("--epochs", type=int, default=1, help="Число эпох обучения")
    parser.add_argument("--lr", type=float, default=2e-4, help="Скорость обучения (learning rate)")
    parser.add_argument("--max-len", type=int, default=256, help="Максимальная длина последовательности")
    args = parser.parse_args()

    hw = probe_hardware()
    device = hw.recommended_device
    print(f"[vcore-train] {hw.summary_banner()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.from_scratch:
        print(f"[vcore-train] Инициализация модели С НУЛЯ по конфигу '{args.model}'...")
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    elif os.path.isdir(args.model) and any(
        f.endswith(".safetensors") for f in os.listdir(args.model)
    ):
        try:
            model = VCoreModelForCausalLM.from_vcore_pretrained(args.model, device="cpu", dtype=torch.float32)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)

    model = convert_model_to_qat(model, group_size=32, mode=args.mode, lora_rank=args.lora_rank)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[vcore-train] Обучаемых параметров: {sum(p.numel() for p in trainable_params):,} (режим={args.mode}, lora={args.lora_rank})")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    texts = _load_texts_from_file(args.data)
    if not texts:
        raise ValueError(f"Файл {args.data} пуст")

    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, txt in enumerate(texts, 1):
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            loss_val = train_vcore_step(model, optimizer, enc["input_ids"], enc["input_ids"])
            total_loss += loss_val
            if step % 10 == 0 or step == len(texts):
                print(f"[Epoch {ep}/{args.epochs}] Step {step}/{len(texts)} | Loss = {total_loss / step:.4f}")

    # Конвертируем обученные QAT-слои в стандартный формат для упаковки через vcore-pack
    for full_name, mod in list(model.named_modules()):
        if isinstance(mod, VCoreQATLinear):
            parts = full_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], mod.to_fp_linear())

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[vcore-train] Модель сохранена в '{args.output}'. Теперь можно сжать её командой vcore-pack!")


if __name__ == "__main__":
    main()
  
