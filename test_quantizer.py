import os
import subprocess
import tempfile
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from vcore_l1 import (
    VCoreTernaryQuantizer,
    VCoreLinear,
    VCoreDynamicCache,
    inject_vcore_layers,
    verify_with_lean4_engine,
)


def absmean_bitnet_quantize(weights: torch.Tensor) -> torch.Tensor:
    gamma = torch.mean(torch.abs(weights))
    if gamma == 0:
        return torch.zeros_like(weights)
    return torch.clamp(torch.round(weights / gamma), -1.0, 1.0) * gamma


def main():
    print("=== 1. Проверка математического ядра Lean 4 (v3.0.0 — 8 модулей) ===")
    print(verify_with_lean4_engine().strip())

    print("\n=== 2. Сравнение с BitNet (absmean) и тест Activation-Weighted L1 ===")
    torch.manual_seed(42)
    weights = torch.randn(128, 256) * 0.05
    weights[torch.rand_like(weights) < 0.005] *= 40.0

    err_bitnet = torch.sum(torch.abs(weights - absmean_bitnet_quantize(weights))).item()
    q = VCoreTernaryQuantizer(group_size=32, theta=1e-4, auto_theta=True)
    w_vcore, _, _ = q.quantize(weights)
    err_vcore = torch.sum(torch.abs(weights - w_vcore)).item()
    assert err_vcore < err_bitnet

    act_imp = torch.ones(256)
    act_imp[:16] = 25.0
    w_weighted, _, _ = q.quantize(weights, activation_scales=act_imp)
    weighted_err_unweighted = torch.sum(act_imp.view(1, -1) * torch.abs(weights - w_vcore)).item()
    weighted_err_aware = torch.sum(act_imp.view(1, -1) * torch.abs(weights - w_weighted)).item()
    assert weighted_err_aware <= weighted_err_unweighted

    print("\n=== 3. Тест каскада (1.58-bit -> INT4 -> FP16), VCoreDynamicCache и VRAM Budget ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        in_dir = os.path.join(tmpdir, "model_in")
        out_dir = os.path.join(tmpdir, "model_out")
        os.makedirs(in_dir, exist_ok=True)

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64, bias=False)
                self.fc2 = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        net = TinyNet().to(torch.float16)
        with torch.no_grad():
            net.fc2.weight.copy_(torch.linspace(-7.0, 7.0, 64 * 32, dtype=torch.float16).view(32, 64))

        save_file(net.state_dict(), os.path.join(in_dir, "model.safetensors"))

        subprocess.run(
            ["vcore-pack", "--input", in_dir, "--output", out_dir, "--group-size", "64", "--max-rel-error", "0.25"],
            check=True,
        )

        packed_sd = load_file(os.path.join(out_dir, "model.safetensors"))
        inject_vcore_layers(net, packed_sd)
        assert isinstance(net.fc1, VCoreLinear)
        assert isinstance(net.fc2, VCoreLinear)
        assert {net.fc1.bits, net.fc2.bits} == {2, 4}

        y_out = net(torch.randn(4, 128, dtype=torch.float16))
        assert y_out.shape == (4, 32)

    cache = VCoreDynamicCache(group_size=32)
    k1, v1 = torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64)
    k2, v2 = torch.randn(1, 4, 1, 64), torch.randn(1, 4, 1, 64)
    cache.update(k1, v1, layer_idx=0)
    k_cat, v_cat = cache.update(k2, v2, layer_idx=0)
    assert k_cat.shape == (1, 4, 9, 64) and cache.get_seq_length() == 9

    from vcore_l1 import solve_chain_overlap_fpt
    w_chain = torch.tensor([4.0, 0.0, -4.0, 0.0])
    c_opt, fpt_cost = solve_chain_overlap_fpt(w_chain, num_grid=9)
    assert c_opt.shape == (4,) and fpt_cost == 0.0

    subprocess.run(["vcore-bench", "--hidden-size", "256", "--out-size", "256", "--steps", "10"], check=True)
    print("[OK] Все компоненты (Lean 4 v3.0.0, Activation-Weighted L1, Warp CUDA, Cascade, KV-Cache) работают!")


if __name__ == "__main__":
    main()
                
