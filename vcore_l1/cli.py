# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import os
from .calibration import collect_activation_importance
from .gguf_exporter import export_vcore_to_gguf
from .safetensors_quantizer import (
    repack_safetensors_layerwise,
    repack_sharded_model,
    repack_hf_hub_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE L1-Certified: Activation-Weighted & Triangular-Compensated 1.58-bit & INT4 Cascade Packer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Локальный файл .safetensors или папка модели")
    group.add_argument("--hf-repo", help="ID репозитория Hugging Face (например, Qwen/Qwen2.5-1.5B)")

    parser.add_argument("--output", "-o", required=True, help="Путь к выходному файлу или папке")
    parser.add_argument("--group-size", "-g", type=int, default=64, help="Начальный размер блока каскада")
    parser.add_argument("--theta", type=float, default=1e-4, help="Базовый порог шума Фильтра Маркова")
    parser.add_argument("--max-rel-error", type=float, default=0.40, help="Порог относительной L1-ошибки")
    parser.add_argument("--target-vram-gb", type=float, default=None, help="Целевой бюджет видеопамяти в ГБ")
    parser.add_argument("--calib-samples", type=int, default=0, help="Число примеров для сбора активаций и треугольного фактора R (0 = Data-Free)")
    parser.add_argument("--triangular-feedback", action="store_true", help="Включить экспериментальную треугольную компенсацию корреляций Холецкого при калибровке")
    parser.add_argument("--export-gguf", default=None, help="Сразу экспортировать сжатую модель в .gguf (для llama.cpp / Ollama)")
    parser.add_argument("--token", default=None, help="Токен Hugging Face")

    args = parser.parse_args()

    act_map = None
    chol_map = None
    if args.calib_samples > 0:
        src_model = args.hf_repo if args.hf_repo else args.input
        use_cov = args.triangular_feedback
        print(f"[V-CORE] Сбор важности активаций ({args.calib_samples} примеров, triangular_feedback={use_cov})...")
        act_map, chol_map = collect_activation_importance(
            src_model,
            num_samples=args.calib_samples,
            token=args.token,
            collect_covariance=use_cov,
        )

    if args.hf_repo:
        stats = repack_hf_hub_model(
            repo_id=args.hf_repo,
            output_dir=args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            token=args.token,
            target_vram_gb=args.target_vram_gb,
            activation_map=act_map,
            chol_map=chol_map,
        )
    elif os.path.isdir(args.input):
        stats = repack_sharded_model(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            target_vram_gb=args.target_vram_gb,
            activation_map=act_map,
            chol_map=chol_map,
        )
    else:
        stats = repack_safetensors_layerwise(
            args.input,
            args.output,
            group_size=args.group_size,
            theta=args.theta,
            max_rel_error=args.max_rel_error,
            target_vram_gb=args.target_vram_gb,
            activation_map=act_map,
            chol_map=chol_map,
        )

    print("=== V-CORE L1-CERTIFIED COMPRESSION REPORT ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<24}: {v:.3f}")
        else:
            print(f"  {k:<24}: {v}")

    if args.export_gguf:
        if not os.path.isdir(args.output):
            raise ValueError("Для экспорта в GGUF укажите выходную директорию модели в --output")
        g_stats = export_vcore_to_gguf(args.output, args.export_gguf)
        size_mb = os.path.getsize(args.export_gguf) / (1024 * 1024)
        print("=== V-CORE GGUF EXPORT REPORT ===")
        print(f"  output_file   : {args.export_gguf} ({size_mb:.2f} MB)")
        for k, v in g_stats.items():
            print(f"  {k:<14}: {v}")


if __name__ == "__main__":
    main()
    
