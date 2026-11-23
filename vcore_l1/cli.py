# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import os
from .safetensors_quantizer import (
    repack_safetensors_layerwise,
    repack_sharded_model,
    repack_hf_hub_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE L1-Certified: 1.58-bit (Delta=1) Safetensors, Sharded & HF Hub Model Packer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Локальный файл .safetensors или папка модели")
    group.add_argument("--hf-repo", help="ID репозитория Hugging Face (например, Qwen/Qwen2.5-1.5B)")

    parser.add_argument("--output", "-o", required=True, help="Путь к выходному файлу или папке")
    parser.add_argument("--group-size", "-g", type=int, default=64, help="Начальный размер блока каскада (64 -> 32 -> 16)")
    parser.add_argument("--theta", type=float, default=1e-4, help="Базовый порог шума Фильтра Маркова")
    parser.add_argument("--max-rel-error", type=float, default=0.40, help="Порог относительной L1-ошибки для Auto-Rollback")
    parser.add_argument("--token", default=None, help="Токен Hugging Face (для закрытых моделей)")

    args = parser.parse_args()

    if args.hf_repo:
        stats = repack_hf_hub_model(
            repo_id=args.hf_repo,
            output_dir=args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            token=args.token,
        )
    elif os.path.isdir(args.input):
        stats = repack_sharded_model(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
        )
    else:
        stats = repack_safetensors_layerwise(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
        )

    print("=== V-CORE L1-CERTIFIED COMPRESSION REPORT ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.3f}")
        else:
            print(f"  {k:<22}: {v}")


if __name__ == "__main__":
    main()
    
