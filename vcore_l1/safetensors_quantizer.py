import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .cuda_quantizer import VCoreTernaryQuantizer, verify_with_lean4_engine


def quantize_safetensors_model(
    input_path: str,
    output_path: str,
    group_size: int = 32,
    theta: float = 1e-4,
    verify_lean: bool = True,
):
    """
    Применяет сертифицированное 1.58-bit квантование (LinearSelector + Δ=1 медиана)
    ко всем линейным матрицам в файле весов Hugging Face (.safetensors),
    изолируя norm, embed, bias и lm_head.
    """
    if verify_lean:
        print("=== Проверка математического сертификата Lean 4 (v2.0.0) ===")
        print(verify_with_lean4_engine().strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantizer = VCoreTernaryQuantizer(group_size=group_size, theta=theta).to(device)

    weights = {}
    processed_layers, skipped_layers = 0, 0
    total_orig_l1, total_err_l1 = 0.0, 0.0

    print(f"\nОткрытие файла весов: {input_path} (Устройство: {device.upper()})")
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Топологический фильтр линейных матриц (исключаем norm, embed, bias)
            is_linear_weight = (
                any(p in key.lower() for p in ["proj", "weight"])
                and not any(n in key.lower() for n in ["norm", "embed", "bias"])
            )

            if is_linear_weight and tensor.ndim == 2 and tensor.numel() % group_size == 0:
                if "lm_head" in key.lower():
                    print(f"[Изоляция] Слой {key} оставлен без сжатия для сохранения точности логитов.")
                    weights[key] = tensor
                    skipped_layers += 1
                    continue

                t_dev = tensor.to(device=device, dtype=torch.float32)
                w_quant, _, _ = quantizer.quantize(t_dev)

                layer_orig = torch.sum(torch.abs(t_dev)).item()
                layer_err = torch.sum(torch.abs(t_dev - w_quant)).item()
                total_orig_l1 += layer_orig
                total_err_l1 += layer_err

                print(f"[Квантован Δ=1] {key} | Форма: {list(tensor.shape)} | L1-ошибка: {layer_err:.2f}")
                weights[key] = w_quant.to(device="cpu", dtype=tensor.dtype)
                processed_layers += 1
            else:
                weights[key] = tensor
                skipped_layers += 1

    save_file(weights, output_path)
    print(f"\n[УСПЕХ] Сжатая модель сохранена в: {output_path}")
    print(f"Итог: квантовано слоёв = {processed_layers}, сохранено без изменений = {skipped_layers}")
    if total_orig_l1 > 0:
        rel_err = (total_err_l1 / total_orig_l1) * 100.0
        print(f"Суммарная L1-норма: {total_orig_l1:.2f} | Ошибка квантования: {total_err_l1:.2f} ({rel_err:.2f}%)")


if __name__ == "__main__":
    in_file, out_file = "model.safetensors", "model_vcore_1_58bit.safetensors"
    if os.path.exists(in_file):
        quantize_safetensors_model(in_file, out_file, group_size=32, theta=1e-4)
    else:
        print(f"Файл {in_file} не найден. Поместите чекпоинт Hugging Face в текущую папку.")
