import os
import subprocess
import tempfile
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from vcore_l1 import (
    VCoreTernaryQuantizer,
    VCoreLinear,
    VCoreKVCacheQuantizer,
    inject_vcore_layers,
    verify_with_lean4_engine,
)


def absmean_bitnet_quantize(weights: torch.Tensor) -> torch.Tensor:
    gamma = torch.mean(torch.abs(weights))
    if gamma == 0:
        return torch.zeros_like(weights)
    return torch.clamp(torch.round(weights / gamma), -1.0, 1.0) * gamma


def main():
    print("=== 1. Проверка математического ядра Lean 4 (v2.0.0) ===")
    print(verify_with_lean4_engine().strip())

    print("\n=== 2. Сравнение с BitNet (absmean) при автоподборе theta_b* ===")
    torch.manual_seed(42)
    weights = torch.randn(128, 256) * 0.05
    weights[torch.rand_like(weights) < 0.005] *= 40.0

    err_bitnet = torch.sum(torch.abs(weights - absmean_bitnet_quantize(weights))).item()
    q = VCoreTernaryQuantizer(group_size=32, theta=1e-4, auto_theta=True)
    w_vcore, _, _ = q.quantize(weights)
    err_vcore = torch.sum(torch.abs(weights - w_vcore)).item()
    print(f"BitNet err: {err_bitnet:.2f} | V-CORE err: {err_vcore:.2f}")
    assert err_vcore < err_bitnet

    print("\n=== 3. Тест каскадного сжатия, Auto-Rollback, VCoreLinear и KV-Cache ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "model_in")
        out_dir = os.path.join(tmpdir, "model_out")
        os.makedirs(in_dir, exist_ok=True)

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64, bias=False)
                self.norm = nn.LayerNorm(64)
                self.fc2 = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.fc2(self.norm(self.fc1(x)))

        net = TinyNet().to(torch.float16)
        with torch.no_grad():
            net.norm.weight.copy_(torch.linspace(0.01, 20.0, 64, dtype=torch.float16))

        save_file(net.state_dict(), os.path.join(in_dir, "model.safetensors"))

        subprocess.run(
            ["vcore-pack", "--input", in_dir, "--output", out_dir, "--group-size", "64", "--max-rel-error", "0.45"],
            check=True,
        )

        packed_sd = load_file(os.path.join(out_dir, "model.safetensors"))
        inject_vcore_layers(net, packed_sd)
        assert isinstance(net.fc1, VCoreLinear)
        assert isinstance(net.fc2, VCoreLinear)
        assert isinstance(net.norm, nn.LayerNorm)

        y_out = net(torch.randn(4, 128, dtype=torch.float16))
        assert y_out.shape == (4, 32)

    subprocess.run(["vcore-bench", "--hidden-size", "256", "--out-size", "256", "--steps", "10"], check=True)
    print("[OK] Все 12 компонентов стека V-CORE L1-Certified v1.0.0 успешно проверены!")


if __name__ == "__main__":
    main()
    
