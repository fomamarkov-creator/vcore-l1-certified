# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import os
from .safetensors_quantizer import repack_safetensors_layerwise, repack_sharded_model


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE L1-Certified: 1.58-bit (Delta=1) Safetensors & Sharded Model Packer"
    )
    parser.add_argument("--input", "-i", required=True, help="Путь к файлу .safetensors или папке модели Hugging Face")
    parser.add_argument("--output", "-o", required=True, help="Путь к выходному файлу или папке")
    parser.add_argument("--group-size", "-g", type=int, default=32, help="Размер развязанного блока Delta=1 (по умолчанию 32)")
    parser.add_argument("--theta", type=float, default=1e-4, help="Базовый порог шума Фильтра Маркова")
    parser.add_argument("--max-rel-error", type=float, default=0.40, help="Порог относительной L1-ошибки для Auto-Rollback")
    parser.add_argument("--no-auto-theta", action="store_true", help="Отключить поблочный автоподбор порога theta_b*")

    args = parser.parse_args()
    auto_theta = not args.no_auto_theta

    if os.path.isdir(args.input):
        stats = repack_sharded_model(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            auto_theta=auto_theta,
        )
    else:
        stats = repack_safetensors_layerwise(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            auto_theta=auto_theta,
        )

    print("=== V-CORE L1-CERTIFIED COMPRESSION REPORT ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.3f}")
        else:
            print(f"  {k:<22}: {v}")


if __name__ == "__main__":
    main()
