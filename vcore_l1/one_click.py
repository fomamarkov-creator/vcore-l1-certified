# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import os
import tempfile
from .backend_dispatcher import get_dispatcher
from .hub_publisher import generate_model_card_readme
from .safetensors_quantizer import repack_hf_hub_model, repack_sharded_model
from .tui import VCoreTerminalStudio
from .vcore_bundle import export_vcore_bundle


def main():
    parser = argparse.ArgumentParser(
        description="V-CORE One-Click Pull & Compress (vcore-pull): скачать, сжать под железо и запустить"
    )
    parser.add_argument("source", help="ID репозитория Hugging Face (например, Qwen/Qwen2.5-1.5B) или локальная папка")
    parser.add_argument("--out", "-o", default=None, help="Путь к выходному файлу .vcore (по умолчанию <имя_модели>.vcore)")
    parser.add_argument("--vram", type=float, default=None, help="Бюджет памяти в ГБ (по умолчанию авто по железу)")
    parser.add_argument("--chat", action="store_true", help="Сразу открыть терминальный чат после упаковки")
    parser.add_argument("--token", default=None, help="Токен Hugging Face (для закрытых моделей)")
    args = parser.parse_args()

    disp = get_dispatcher()
    hw = disp.profile
    print(f"[vcore-pull] {hw.summary_banner()}")

    budget = args.vram if args.vram is not None else hw.recommended_vram_budget_gb
    sph_emb = hw.recommended_spherical_emb
    clean_name = args.source.rstrip("/").split("/")[-1].lower()
    bundle_out = args.out or f"./{clean_name}.vcore"

    with tempfile.TemporaryDirectory(prefix="vcore_work_") as tmp_dir:
        print(f"[vcore-pull] Сжатие '{args.source}' (бюджет={budget} ГБ, spherical_emb={sph_emb})...")
        if os.path.isdir(args.source):
            stats = repack_sharded_model(
                args.source, tmp_dir, target_vram_gb=budget, spherical_emb=sph_emb
            )
        else:
            stats = repack_hf_hub_model(
                args.source, tmp_dir, target_vram_gb=budget, token=args.token, spherical_emb=sph_emb
            )

        generate_model_card_readme(tmp_dir, base_model=args.source, stats=stats, spherical_emb=sph_emb)
        b_info = export_vcore_bundle(tmp_dir, bundle_out, spherical_emb=sph_emb)

    print(f"\n[vcore-pull] Готово! Единый файл: {bundle_out} ({b_info['bundle_size_mb']} МБ)")
    print(f"[vcore-pull] Сертификат Lean 4: Max Block Drift <= {b_info['certified_max_block_drift']}")

    if args.chat:
        studio = VCoreTerminalStudio(initial_model=bundle_out)
        studio.run()
    else:
        print(f"[vcore-pull] Для запуска чата выполните: vcore-ui --model {bundle_out}")


if __name__ == "__main__":
    main()
