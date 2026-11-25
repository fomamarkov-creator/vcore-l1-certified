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
    solve_chain_overlap_fpt,
    solve_continuous_chain_dp,
    solve_triple_overlap_fpt,
    quantize_ternary_triangular_feedback,
    MARKOV_LE0_GAP,
    fwht_4,
    fwht_8,
    fwht_16,
    fwht_32,
    quantize_3hcp_lattice,
    quantize_d4_lattice,
    quantize_e8_lattice,
    quantize_barnes_wall32_lattice,
    VCoreSuperQuantizer,
    VCoreInDomainHadamardLinear,
    VCoreLUTLinear,
    allocate_mixed_lattice,
    certify_transformer_block_e2e,
    verify_rope_key_drift,
    golay24_syndrome_decode,
    register_inversion_split,
    quantize_24_sparse_l1,
    fit_datafree_l1_lora,
    verify_moe_routing_safety,
    verify_matmul_freivalds,
)


def absmean_bitnet_quantize(weights: torch.Tensor) -> torch.Tensor:
    gamma = torch.mean(torch.abs(weights))
    if gamma == 0:
        return torch.zeros_like(weights)
    return torch.clamp(torch.round(weights / gamma), -1.0, 1.0) * gamma


def main():
    print("=== 1. Проверка математического ядра Lean 4 (v6.0.0 — 23 модуля) ===")
    print(verify_with_lean4_engine().strip())

    print("\n=== 2. Сравнение с BitNet (absmean), Activation-Weighted L1 и Triangular Feedback ===")
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

    X_calib = torch.randn(64, 256)
    X_calib[:, 1] = 0.85 * X_calib[:, 0] + 0.15 * X_calib[:, 1]
    cov = (X_calib.T @ X_calib) / 64.0 + 1e-3 * torch.eye(256)
    R_upper = torch.linalg.cholesky(cov).T
    w_tri, _, _ = quantize_ternary_triangular_feedback(weights, R_upper, group_size=32)
    assert w_tri.shape == weights.shape
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

    w_chain = torch.tensor([4.0, 0.0, -4.0, 0.0])
    c_opt, fpt_cost = solve_chain_overlap_fpt(w_chain, num_grid=9)
    assert c_opt.shape == (4,) and abs(fpt_cost - 8.0) < 1e-6

    w_cont = torch.tensor([4.0, 8.0, 4.0])
    c_cont, cont_cost = solve_continuous_chain_dp(w_cont)
    assert c_cont.shape == (3,) and abs(cont_cost - 8.0) < 1e-5

    w_triple = torch.tensor([2.0, 0.0, 6.0, 4.0])
    grid_tw2 = torch.tensor([-2.0, 0.0, 2.0, 4.0])
    c_tw2, tw2_cost = solve_triple_overlap_fpt(w_triple, grid=grid_tw2)
    assert c_tw2.shape == (4,) and abs(tw2_cost - 6.0) < 1e-5
        print("\n=== 4. Тест модулей 13–23: Решётки 3HCP / D₄ / E₈ / Λ₃₂, W₃₂, LUT, Mixed-Lattice, RoPE, Golay G₂₄ ===")
    v4 = torch.randn(16, 4)
    assert torch.allclose(fwht_4(fwht_4(v4)), v4, atol=1e-5)
    v8 = torch.randn(16, 8)
    assert torch.allclose(fwht_8(fwht_8(v8)), 8.0 * v8, atol=1e-4)
    v16 = torch.randn(8, 16)
    assert torch.allclose(fwht_16(fwht_16(v16)), 16.0 * v16, atol=1e-4)
    v32 = torch.randn(8, 32)
    assert torch.allclose(fwht_32(fwht_32(v32)), 32.0 * v32, atol=1e-3)

    w_test = torch.randn(64, 128)
    w_3hcp, _, _ = quantize_3hcp_lattice(w_test)
    w_d4, _, _ = quantize_d4_lattice(w_test)
    w_e8, idx_e8, _ = quantize_e8_lattice(w_test, use_hadamard=True)
    w_bw32, _, _ = quantize_barnes_wall32_lattice(w_test, use_hadamard32=True)
    assert w_3hcp.shape == w_test.shape and w_d4.shape == w_test.shape and w_e8.shape == w_test.shape
    assert w_bw32.shape == w_test.shape
    assert idx_e8.dtype == torch.uint8 and idx_e8.numel() == (64 * 128) // 8

    sq = VCoreSuperQuantizer(rel_error_budget=0.30)
    w_sq, sq_stats = sq.quantize(w_test)
    assert sq_stats["actual_l1_err"] <= sq_stats["certified_max_l1_err"] + 1e-4

    lin_fp = nn.Linear(128, 64, bias=True)
    lin_w8 = VCoreInDomainHadamardLinear.from_linear(lin_fp)
    lin_lut = VCoreLUTLinear.from_linear(lin_fp, lora_rank=16, lora_iters=6)
    x_in = torch.randn(8, 128)
    y_fp = lin_fp(x_in)
    y_w8 = lin_w8(x_in)
    y_lut = lin_lut(x_in)
    assert y_w8.shape == y_fp.shape
    assert torch.allclose(y_w8, y_lut, atol=1e-3)

    _, mix_sum = allocate_mixed_lattice(
        [("attn.v_proj", 10_000_000, 0.02), ("mlp.gate_proj", 50_000_000, 0.015), ("mlp.up_proj", 50_000_000, 0.015)],
        target_avg_bpw=0.82,
    )
    assert mix_sum["certified_budget_ok"] == 1.0
    assert certify_transformer_block_e2e(0.01, 0.005, 0.005, 0.01, 0.01, 0.01, 2.0) > 0.0

    ok_rope, _ = verify_rope_key_drift(torch.ones(4, 2), torch.zeros(4, 2), torch.full((4, 2), 0.05), max_drift_bound=0.25)
    assert ok_rope
    cw_clean, syn_after = golay24_syndrome_decode((1 << 5) ^ (1 << 11))
    assert syn_after != 0 and cw_clean == 0

    assert abs(MARKOV_LE0_GAP - 0.024) < 1e-7
    w_out = torch.tensor([10.0, 300.0, -5.0, -400.0])
    main_ch, reg_out = register_inversion_split(w_out, limit=256.0)
    assert torch.allclose(main_ch + reg_out, w_out)
    assert torch.max(torch.abs(main_ch)).item() < 256.0

    w_sp = quantize_24_sparse_l1(torch.randn(32, 64))
    nnz_per_block = torch.count_nonzero(w_sp.view(-1, 4), dim=1)
    assert int(torch.max(nnz_per_block).item()) <= 2

    residual = w_test - w_e8
    err_before = torch.sum(torch.abs(residual)).item()
    U_lora, V_lora = fit_datafree_l1_lora(residual, rank=8, iters=4)
    err_after = torch.sum(torch.abs(residual - U_lora @ V_lora)).item()
    assert err_after < err_before

    assert verify_moe_routing_safety(torch.tensor([3.0, 2.5, 1.0, 0.2]), torch.tensor([0.1, 0.1, 0.1, 0.1]), top_k=2)
    A_m = torch.randn(32, 64)
    B_m = torch.randn(64, 32)
    C_m = A_m @ B_m
    assert verify_matmul_freivalds(A_m, B_m, C_m)
    assert not verify_matmul_freivalds(A_m, B_m, C_m + 0.5)

    print(
        f"[OK] Все компоненты v4.0.0 (Lean 4 23-Module Core, E8/Λ32 Sub-1-Bit [{mix_sum['avg_bpw']:.2f} b/w], "
        f"LUT Kernel, Rational RoPE, Golay G24 Syndrome, ℓ₁-LoRA [{err_before:.1f} -> {err_after:.1f}]) работают!"
    )

    subprocess.run(["vcore-bench", "--hidden-size", "256", "--out-size", "256", "--steps", "10"], check=True)


if __name__ == "__main__":
    main()
