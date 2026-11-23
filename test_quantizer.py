import torch
from vcore_l1 import VCoreTernaryQuantizer, verify_with_lean4_engine


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

    print("\n=== 2. Сравнительный бенчмарк на матрице слоя нейросети (128 x 256) с выбросами ===")
    torch.manual_seed(42)
    weights = torch.randn(128, 256) * 0.05
    # Добавляем 0.5% тяжелых выбросов (характерных для весов LLM)
    outlier_mask = torch.rand_like(weights) < 0.005
    weights[outlier_mask] *= 40.0

    orig_l1 = torch.sum(torch.abs(weights)).item()

    # Метод 1: Стандартный BitNet absmean
    w_bitnet = absmean_bitnet_quantize(weights)
    err_bitnet = torch.sum(torch.abs(weights - w_bitnet)).item()

    # Метод 2: Ваш гибридный V-CORE (LinearSelector + медиана блока Δ = 1)
    quantizer = VCoreTernaryQuantizer(group_size=32, theta=1e-4)
    w_vcore, mask, scales = quantizer.quantize(weights)
    err_vcore = torch.sum(torch.abs(weights - w_vcore)).item()

    reduction = (1.0 - err_vcore / err_bitnet) * 100.0
    print(f"Исходная L1-норма весов       : {orig_l1:.2f}")
    print(f"Ошибка BitNet (absmean)       : {err_bitnet:.2f}")
    print(f"Ошибка V-CORE (Delta=1 Med)   : {err_vcore:.2f} (снижение ошибки на {reduction:.1f}%)")
    assert err_vcore < err_bitnet, "V-CORE должен давать меньшую L1-ошибку, чем absmean!"
    print("[OK] Все проверки успешно пройдены.")


if __name__ == "__main__":
    main()
