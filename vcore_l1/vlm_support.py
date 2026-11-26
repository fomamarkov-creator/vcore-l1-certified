# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import glob
import json
import os
import shutil
from typing import Dict, Optional
import torch
from safetensors.torch import load_file, save_file
from .inference_layer import inject_vcore_layers
from .safetensors_quantizer import repack_safetensors_layerwise
from .v5_engine import pack_spherical_code


def repack_vlm_model(
    input_dir: str,
    output_dir: str,
    group_size: int = 64,
    max_rel_error: float = 0.35,
    spherical_vision: bool = True,
    spherical_emb: bool = True,
) -> Dict[str, float]:
    """
    Специализированный упаковщик для Vision-Language моделей (Qwen2.5-VL / Qwen3-VL / LLaVA):
    - Визуальный энкодер (`visual.*`) сжимается гиперболическими сферическими кодами E₈ (~1.0 b/w)
      для сохранения геометрической точности патчей изображений;
    - Языковая модель сжимается каскадом 1.58-bit / INT4 с межслойной компенсацией кривизны.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {input_dir} не найдены .safetensors файлы")

    work_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_e8_layers = 0
    total_stats = {
        "vision_spherical_e8_layers": 0,
        "quantized_158bit_layers": 0,
        "quantized_int4_layers": 0,
        "cross_layer_compensated": 0,
        "orig_size_mb": 0.0,
        "packed_size_mb": 0.0,
    }
    new_weight_map: Dict[str, str] = {}

    for shard_path in shard_files:
        fname = os.path.basename(shard_path)
        out_shard = os.path.join(output_dir, fname)
        sd = load_file(shard_path)

        # Отделяем визуальный энкодер от языковой части
        llm_only: Dict[str, torch.Tensor] = {}
        vision_packed: Dict[str, torch.Tensor] = {}

        for k, t in sd.items():
            if ("visual." in k or "vision_tower." in k) and t.is_floating_point() and t.ndim == 2 and t.numel() >= 64 and spherical_vision:
                w_f32 = t.to(device=work_device, dtype=torch.float32)
                sp_idx, sp_rad, orig_shape, _ = pack_spherical_code(
                    w_f32, lattice="e8", hyperbolic_poincare=True
                )
                vision_packed[f"{k}.vcore_sph_idx"] = sp_idx.cpu()
                vision_packed[f"{k}.vcore_sph_rad"] = sp_rad.cpu()
                vision_packed[f"{k}.vcore_shape"] = torch.tensor(orig_shape, dtype=torch.int64)
                vision_e8_layers += 1
                del w_f32
            else:
                llm_only[k] = t

        tmp_llm_in = os.path.join(output_dir, f".tmp_in_{fname}")
        tmp_llm_out = os.path.join(output_dir, f".tmp_out_{fname}")
        save_file(llm_only, tmp_llm_in)
        st = repack_safetensors_layerwise(
            tmp_llm_in,
            tmp_llm_out,
            group_size=group_size,
            max_rel_error=max_rel_error,
            cross_layer_comp=True,
            spherical_emb=spherical_emb,
        )
        merged_out = load_file(tmp_llm_out)
        merged_out.update(vision_packed)
        save_file(merged_out, out_shard)

        for k in merged_out.keys():
            new_weight_map[k] = fname

        total_stats["orig_size_mb"] += os.path.getsize(shard_path) / (1024.0 * 1024.0)
        total_stats["packed_size_mb"] += os.path.getsize(out_shard) / (1024.0 * 1024.0)
        total_stats["quantized_158bit_layers"] += st["quantized_158bit_layers"]
        total_stats["quantized_int4_layers"] += st["quantized_int4_layers"]
        total_stats["cross_layer_compensated"] += st["cross_layer_compensated"]

        os.remove(tmp_llm_in)
        os.remove(tmp_llm_out)

    total_stats["vision_spherical_e8_layers"] = vision_e8_layers
    total_stats["compression_ratio"] = total_stats["orig_size_mb"] / max(total_stats["packed_size_mb"], 1e-6)

    for item in os.listdir(input_dir):
        src_item = os.path.join(input_dir, item)
        dst_item = os.path.join(output_dir, item)
        if item == "model.safetensors.index.json":
            with open(src_item, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            idx_data["weight_map"] = new_weight_map
            idx_data.setdefault("metadata", {})["vcore_vlm"] = "v5-spherical-vision"
            with open(dst_item, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)
        elif not item.endswith(".safetensors") and os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)

    return total_stats


class VCoreModelForVision2Seq:
    """Загрузчик сжатых мультимодальных моделей (Qwen2.5-VL / Qwen3-VL / LLaVA)."""

    @staticmethod
    def from_vcore_pretrained(
        model_dir: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        integer_only: bool = False,
    ):
        from transformers import AutoConfig, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_config(config, dtype=dtype, trust_remote_code=True)

        shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        merged_sd: Dict[str, torch.Tensor] = {}
        for sf in shard_files:
            merged_sd.update(load_file(sf))

        model = inject_vcore_layers(model, merged_sd, integer_only=integer_only)
        return model.to(device=device, dtype=dtype).eval()
          
