# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import os
from .vcore_bundle import export_vcore_bundle
from .vlm_support import repack_vlm_model
from .calibration import collect_activation_importance
from .gguf_exporter import export_vcore_to_gguf
from .safetensors_quantizer import (
    repack_safetensors_layerwise,
    repack_sharded_model,
    repack_hf_hub_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE L1-Certified: Activation-Weighted, Cross-Layer & Spherical Quantizer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Локальный файл .safetensors или папка модели")
    group.add_argument("--hf-repo", help="ID репозитория Hugging Face (например, Qwen/Qwen2.5-1.5B)")

    parser.add_argument("--output", "-o", required=True, help="Путь к выходному файлу или папке")
    parser.add_argument("--group-size", "-g", type=int, default=64, help="Начальный размер блока каскада")
    parser.add_argument("--theta", type=float, default=1e-4, help="Базовый порог шума Фильтра Маркова")
    parser.add_argument("--max-rel-error", type=float, default=0.40, help="Порог относительной L1-ошибки")
    parser.add_argument("--target-vram-gb", type=float, default=None, help="Целевой бюджет видеопамяти в ГБ")
    parser.add_argument("--calib-samples", type=int, default=0, help="Число примеров для сбора активаций и фактора Холецкого R (0 = Data-Free)")
    parser.add_argument("--triangular-feedback", action="store_true", help="Включить треугольную компенсацию корреляций Холецкого при калибровке")
    parser.add_argument("--no-cross-layer", action="store_true", help="Отключить межслойную компенсацию кривизны (v_proj->o_proj, up_proj->down_proj)")
    parser.add_argument("--spherical-emb", action="store_true", help="Сжать embed_tokens и lm_head гиперболическими сферическими кодами E₈ (~1.0 b/w)")
    parser.add_argument("--export-gguf", default=None, help="Сразу экспортировать сжатую модель в .gguf (для llama.cpp / Ollama)")
    parser.add_argument("--token", default=None, help="Токен Hugging Face")
    parser.add_argument("--vlm", action="store_true", help="Режим мультимодальной модели (Qwen-VL / LLaVA: visual в E₈ + LLM каскад)")
    parser.add_argument("--bundle", default=None, help="Упаковать результат в единый файл .vcore с паспортом Lean 4")
    
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

    cross_layer = not args.no_cross_layer
    
    if args.vlm:
        if not args.input or not os.path.isdir(args.input):
            raise ValueError("Для --vlm укажите локальную папку модели в --input")
        stats = repack_vlm_model(
            input_dir=args.input,
            output_dir=args.output,
            group_size=args.group_size,
            max_rel_error=args.max_rel_error,
            spherical_emb=True,
        )
    elif args.hf_repo:
        
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
            cross_layer_comp=cross_layer,
            spherical_emb=args.spherical_emb,
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
            cross_layer_comp=cross_layer,
            spherical_emb=args.spherical_emb,
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
            cross_layer_comp=cross_layer,
            spherical_emb=args.spherical_emb,
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

    if args.bundle:
        if not os.path.isdir(args.output):
            raise ValueError("Для --bundle укажите выходную папку модели в --output")
        b_stats = export_vcore_bundle(
            model_dir=args.output,
            bundle_path=args.bundle,
            max_rel_error=args.max_rel_error,
            spherical_emb=args.spherical_emb,
        )
        print("=== V-CORE SINGLE-FILE BUNDLE (.vcore) ===")
        for k, v in b_stats.items():
            print(f"  {k:<26}: {v}")
            

if __name__ == "__main__":
    main()
            
