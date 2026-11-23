import os
import tempfile
import torch
from safetensors.torch import save_file, load_file
from vcore_l1 import (
    VCoreTernaryQuantizer,
    verify_with_lean4_engine,
    repack_safetensors_layerwise,
    unpack_uint8_to_ternary,
)


def absmean_bitnet_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Классическое тернарное квантование BitNet (глобальный масштаб по среднему модулю)."""
    gamma = torch.mean(torch.abs(weights))
    if gamma == 0:
        return torch.zeros_like(weights)
    w_scaled = weights / gamma
    ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
    return ternary * gamma


def main():
    print("=== 1. Проверка математического ядра Lean 4 (v2.0.0) ===")
    lean_out = verify_with_lean4_engine()
    print(lean_out.strip())

    print("\n=== 2. Сравнительный бенчмарк на матрице (128 x 256) с выбросами ===")
    torch.manual_seed(42)
    weights = torch.randn(128, 256) * 0.05
    outlier_mask = torch.rand_like(weights) < 0.005
    weights[outlier_mask] *= 40.0

    orig_l1 = torch.sum(torch.abs(weights)).item()
    w_bitnet = absmean_bitnet_quantize(weights)
    err_bitnet = torch.sum(torch.abs(weights - w_bitnet)).item()

    quantizer = VCoreTernaryQuantizer(group_size=32, theta=1e-4)
    w_vcore, mask, scales = quantizer.quantize(weights)
    err_vcore = torch.sum(torch.abs(weights - w_vcore)).item()

    reduction = (1.0 - err_vcore / err_bitnet) * 100.0
    print(f"Исходная L1-норма весов       : {orig_l1:.2f}")
    print(f"Ошибка BitNet (absmean)       : {err_bitnet:.2f}")
    print(f"Ошибка V-CORE (Delta=1 Med)   : {err_vcore:.2f} (снижение ошибки на {reduction:.1f}%)")
    assert err_vcore < err_bitnet, "V-CORE должен давать меньшую L1-ошибку, чем absmean!"

    print("\n=== 3. Тест адаптивной послойной переупаковки (.safetensors + Auto-Rollback) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "model_fp16.safetensors")
        out_file = os.path.join(tmpdir, "model_vcore_158bit.safetensors")

        # Создаём чувствительный 1D-слой с сильным разбросом амплитуд (должен откатиться в FP16)
        sensitive_norm = torch.linspace(0.01, 25.0, 512, dtype=torch.float16)

        dummy_model = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 512, dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 512, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": sensitive_norm,
        }
        save_file(dummy_model, in_file)

        stats = repack_safetensors_layerwise(
            in_file,
            out_file,
            group_size=32,
            theta=1e-4,
            max_rel_error=0.45,
            include_aux_layers=True,
        )
        print(f"Квантовано слоёв  : {stats['quantized_layers']}")
        print(f"Откат в FP16      : {stats['rolled_back_layers']}")
        print(f"Размер до/после   : {stats['orig_size_mb']:.2f} MB -> {stats['packed_size_mb']:.2f} MB (сжатие в {stats['compression_ratio']:.1f}x)")

        packed_dict = load_file(out_file)
        assert stats["quantized_layers"] >= 2, "Рабочие матрицы должны быть сжаты"
        assert stats["rolled_back_layers"] >= 1, "Чувствительный слой должен автоматически откатиться в FP16"
        assert "model.layers.0.input_layernorm.weight" in packed_dict

        unpacked_mask = unpack_uint8_to_ternary(
            packed_dict["model.layers.0.self_attn.q_proj.weight.vcore_packed"],
            tuple(packed_dict["model.layers.0.self_attn.q_proj.weight.vcore_shape"].tolist()),
        )
        assert unpacked_mask.shape == (256, 512)

    print("[OK] Все тесты Lean 4, PyTorch, Safetensors и Auto-Rollback успешно пройдены!")


if __name__ == "__main__":
    main()
    
