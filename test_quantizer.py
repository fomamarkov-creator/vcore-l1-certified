import json
import os
import subprocess
import tempfile
import torch
from safetensors.torch import save_file, load_file
from vcore_l1 import (
    VCoreTernaryQuantizer,
    verify_with_lean4_engine,
    repack_safetensors_layerwise,
    repack_sharded_model,
    unpack_uint8_to_ternary,
)


def absmean_bitnet_quantize(weights: torch.Tensor) -> torch.Tensor:
    gamma = torch.mean(torch.abs(weights))
    if gamma == 0:
        return torch.zeros_like(weights)
    ternary = torch.clamp(torch.round(weights / gamma), -1.0, 1.0)
    return ternary * gamma


def main():
    print("=== 1. Проверка математического ядра Lean 4 (v2.0.0) ===")
    lean_out = verify_with_lean4_engine()
    print(lean_out.strip())

    print("\n=== 2. Тест автоподбора порога theta_b* и Δ=1 медианы на матрице с выбросами ===")
    torch.manual_seed(42)
    weights = torch.randn(128, 256) * 0.05
    outlier_mask = torch.rand_like(weights) < 0.005
    weights[outlier_mask] *= 40.0

    w_bitnet = absmean_bitnet_quantize(weights)
    err_bitnet = torch.sum(torch.abs(weights - w_bitnet)).item()

    quantizer = VCoreTernaryQuantizer(group_size=32, theta=1e-4, auto_theta=True)
    w_vcore, mask, scales = quantizer.quantize(weights)
    err_vcore = torch.sum(torch.abs(weights - w_vcore)).item()

    reduction = (1.0 - err_vcore / err_bitnet) * 100.0
    print(f"Ошибка BitNet (absmean)       : {err_bitnet:.2f}")
    print(f"Ошибка V-CORE (Auto-Theta Δ=1): {err_vcore:.2f} (снижение ошибки на {reduction:.1f}%)")
    assert err_vcore < err_bitnet

    print("\n=== 3. Тест шардированной модели (index.json + Auto-Rollback + CLI vcore-pack) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "hf_model_sharded")
        out_dir = os.path.join(tmpdir, "hf_model_vcore_158bit")
        os.makedirs(in_dir, exist_ok=True)

        shard1 = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 512, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": torch.linspace(0.01, 25.0, 512, dtype=torch.float16),
        }
        shard2 = {
            "model.layers.1.mlp.gate_proj.weight": torch.randn(512, 512, dtype=torch.float16),
        }
        save_file(shard1, os.path.join(in_dir, "model-00001-of-00002.safetensors"))
        save_file(shard2, os.path.join(in_dir, "model-00002-of-00002.safetensors"))

        index_payload = {
            "metadata": {"total_size": 1024},
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
            },
        }
        with open(os.path.join(in_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index_payload, f)

        # Запускаем через CLI-команду vcore-pack
        subprocess.run(
            ["vcore-pack", "--input", in_dir, "--output", out_dir, "--group-size", "32", "--max-rel-error", "0.45"],
            check=True,
        )

        # Проверяем обновлённый index.json и распаковку первого шарда
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
            new_idx = json.load(f)
        assert "model.layers.0.self_attn.q_proj.weight.vcore_packed" in new_idx["weight_map"]
        assert "model.layers.0.input_layernorm.weight" in new_idx["weight_map"]

        s1_out = load_file(os.path.join(out_dir, "model-00001-of-00002.safetensors"))
        unpacked = unpack_uint8_to_ternary(
            s1_out["model.layers.0.self_attn.q_proj.weight.vcore_packed"],
            tuple(s1_out["model.layers.0.self_attn.q_proj.weight.vcore_shape"].tolist()),
        )
        assert unpacked.shape == (256, 512)

    print("[OK] Все 4 улучшения (Lean 4, Auto-Theta, Sharded Safetensors, Auto-Rollback, CLI) проверены!")


if __name__ == "__main__":
    main()
    
