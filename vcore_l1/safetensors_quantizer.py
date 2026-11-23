# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import gc
import glob
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
import torch
from safetensors.torch import load_file, save_file
from .cuda_quantizer import VCoreTernaryQuantizer, _weighted_median_1d


def pack_ternary_to_uint8(ternary_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = tuple(ternary_mask.shape)
    flat = (ternary_mask.reshape(-1).to(torch.int8) + 1).to(torch.uint8)
    pad_len = (4 - (flat.numel() % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.ones(pad_len, dtype=torch.uint8, device=flat.device)])
    grouped = flat.view(-1, 4)
    b0 = grouped.select(1, 0)
    b1 = grouped.select(1, 1) << 2
    b2 = grouped.select(1, 2) << 4
    b3 = grouped.select(1, 3) << 6
    packed = b0 | b1 | b2 | b3
    return packed.contiguous(), orig_shape


def unpack_uint8_to_ternary(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    total_elems = 1
    for d in orig_shape:
        total_elems *= d
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    unpacked = torch.stack([b0, b1, b2, b3], dim=1).reshape(-1).narrow(0, 0, total_elems)
    return unpacked.to(torch.float32) - 1.0


def pack_int4_to_uint8(int4_vals: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    orig_shape = tuple(int4_vals.shape)
    flat = (torch.clamp(int4_vals.reshape(-1), -7, 7).to(torch.int8) + 7).to(torch.uint8)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.full((1,), 7, dtype=torch.uint8, device=flat.device)])
    grouped = flat.view(-1, 2)
    packed = (grouped.select(1, 0) & 0x0F) | ((grouped.select(1, 1) & 0x0F) << 4)
    return packed.contiguous(), orig_shape


def unpack_uint8_to_int4(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    total_elems = 1
    for d in orig_shape:
        total_elems *= d
    low = (packed & 0x0F).to(torch.float32) - 7.0
    high = ((packed >> 4) & 0x0F).to(torch.float32) - 7.0
    unpacked = torch.stack([low, high], dim=1).reshape(-1).narrow(0, 0, total_elems)
    return unpacked


def _pad_to_group(flat: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    orig_numel = flat.numel()
    rem = orig_numel % group_size
    if rem == 0:
        return flat, orig_numel
    pad = group_size - rem
    return torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)]), orig_numel
    @torch.no_grad()
def quantize_ternary_exact_critical(
    blocks: torch.Tensor,
    imp_blocks: Optional[torch.Tensor] = None,
    chunk_size: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = blocks.size(0)
    g = blocks.size(1)
    out_q = torch.empty_like(blocks)
    out_mask = torch.empty_like(blocks)
    out_scales = torch.empty((num_blocks,), device=blocks.device, dtype=blocks.dtype)

    for start in range(0, num_blocks, chunk_size):
        end = min(start + chunk_size, num_blocks)
        b_chunk = blocks.narrow(0, start, end - start)
        imp_chunk = (
            imp_blocks.narrow(0, start, end - start)
            if imp_blocks is not None
            else torch.ones_like(b_chunk)
        )
        c_len = b_chunk.size(0)
        abs_b = torch.abs(b_chunk)
        sorted_abs, _ = torch.sort(abs_b, dim=1)

        best_err = torch.full((c_len,), float("inf"), device=b_chunk.device, dtype=b_chunk.dtype)
        best_mask = torch.zeros_like(b_chunk)
        best_scales = torch.zeros((c_len, 1), device=b_chunk.device, dtype=b_chunk.dtype)

        for k in range(g):
            th = (
                torch.zeros((c_len, 1), device=b_chunk.device, dtype=b_chunk.dtype)
                if k == 0
                else sorted_abs.narrow(1, k - 1, 1)
            )
            active = abs_b > th
            mask = torch.where(active, torch.sign(b_chunk), torch.zeros_like(b_chunk))
            active_imp = torch.where(active, imp_chunk, torch.zeros_like(imp_chunk))
            scales = _weighted_median_1d(abs_b, active_imp)

            w_cand = mask * scales
            err = torch.sum(imp_chunk * torch.abs(b_chunk - w_cand), dim=1)
            improve = err < best_err
            best_err = torch.where(improve, err, best_err)
            best_mask = torch.where(improve.unsqueeze(1), mask, best_mask)
            best_scales = torch.where(improve.unsqueeze(1), scales, best_scales)

        out_q.narrow(0, start, c_len).copy_(best_mask * best_scales)
        out_mask.narrow(0, start, c_len).copy_(best_mask)
        out_scales.narrow(0, start, c_len).copy_(best_scales.reshape(-1))

    return out_q, out_mask, out_scales


@torch.no_grad()
def _quantize_int4_blockwise(w_f32: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, float]:
    flat = w_f32.reshape(-1)
    padded, orig_numel = _pad_to_group(flat, group_size)
    blocks = padded.view(-1, group_size)
    abs_b = torch.abs(blocks)
    med_b = torch.median(abs_b, dim=1, keepdim=True).values
    max_b = torch.max(abs_b, dim=1, keepdim=True).values
    step = torch.maximum(med_b / 2.5, max_b / 7.0)
    step = torch.clamp(step, min=1e-8)

    q_int = torch.clamp(torch.round(blocks / step), -7.0, 7.0)
    w_hat = (q_int * step).reshape(-1).narrow(0, 0, orig_numel).reshape(w_f32.shape)
    err = torch.sum(torch.abs(w_f32 - w_hat)).item()
    return q_int.reshape(-1).narrow(0, 0, orig_numel).reshape(w_f32.shape), step.reshape(-1), err


@torch.no_grad()
def _quantize_tensor_cascade(
    w_f32: torch.Tensor,
    cascade_groups: List[int],
    theta: float,
    max_rel_error: float,
    auto_theta: bool,
    allow_int4: bool = True,
    act_scales: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int, float]]:
    orig_l1 = torch.sum(torch.abs(w_f32)).item()
    flat = w_f32.reshape(-1)

    imp_flat = None
    if act_scales is not None and w_f32.ndim == 2 and act_scales.numel() == w_f32.size(1):
        imp_flat = torch.abs(act_scales).view(1, -1).expand_as(w_f32).reshape(-1).to(w_f32.device, dtype=w_f32.dtype)

    for g in cascade_groups:
        eff_g = min(g, flat.numel())
        if eff_g <= 0:
            continue
        padded_flat, orig_numel = _pad_to_group(flat, eff_g)
        imp_padded = None
        if imp_flat is not None:
            imp_padded, _ = _pad_to_group(imp_flat, eff_g)

        if auto_theta and eff_g <= 64:
            imp_blocks = imp_padded.view(-1, eff_g) if imp_padded is not None else None
            w_q_pad, mask_pad, scales = quantize_ternary_exact_critical(padded_flat.view(-1, eff_g), imp_blocks=imp_blocks)
            w_q_pad = w_q_pad.reshape(-1)
            mask_pad = mask_pad.reshape(-1)
        else:
            q = VCoreTernaryQuantizer(group_size=eff_g, theta=theta, auto_theta=auto_theta)
            w_q_pad, mask_pad, scales = q.quantize(padded_flat, activation_scales=imp_padded)

        w_q = w_q_pad.narrow(0, 0, orig_numel).reshape(w_f32.shape)
        mask = mask_pad.narrow(0, 0, orig_numel).reshape(w_f32.shape)
        err = torch.sum(torch.abs(w_f32 - w_q)).item()
        if (err / (orig_l1 + 1e-8)) <= max_rel_error:
            return mask, scales, eff_g, 2, err

    if allow_int4 and flat.numel() >= 16:
        eff_g4 = min(32, flat.numel())
        q_int4, scales4, err4 = _quantize_int4_blockwise(w_f32, group_size=eff_g4)
        if (err4 / (orig_l1 + 1e-8)) <= max_rel_error:
            return q_int4, scales4, eff_g4, 4, err4

    return None
    @torch.no_grad()
def _estimate_threshold_for_vram(
    state_dict: Dict[str, torch.Tensor],
    target_vram_gb: float,
    group_size: int,
    theta: float,
    auto_theta: bool,
) -> float:
    target_bytes = target_vram_gb * (1024 ** 3)
    total_fp16_bytes = sum(t.numel() * 2 for t in state_dict.values())
    if total_fp16_bytes <= target_bytes:
        return 0.35

    work_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_profiles = []
    fixed_bytes = 0
    for name, t in state_dict.items():
        if not t.is_floating_point() or t.ndim < 2 or t.numel() < 64:
            fixed_bytes += t.numel() * t.element_size()
            continue
        w_f32 = t.to(device=work_device, dtype=torch.float32)
        orig_l1 = torch.sum(torch.abs(w_f32)).item()
        eff_g = min(group_size, w_f32.numel())
        padded, orig_n = _pad_to_group(w_f32.reshape(-1), eff_g)
        if auto_theta and eff_g <= 64:
            w_q, _, _ = quantize_ternary_exact_critical(padded.view(-1, eff_g))
            w_q = w_q.reshape(-1)
        else:
            q = VCoreTernaryQuantizer(group_size=eff_g, theta=theta, auto_theta=auto_theta)
            w_q, _, _ = q.quantize(padded)
        rel_err = torch.sum(torch.abs(w_f32 - w_q.narrow(0, 0, orig_n).reshape(t.shape))).item() / (orig_l1 + 1e-8)
        fp_b = t.numel() * 2
        q2_b = (t.numel() + 3) // 4 + ((t.numel() + eff_g - 1) // eff_g) * 2
        layer_profiles.append((rel_err, fp_b, q2_b))
        del w_f32, padded, w_q

    if work_device.type == "cuda":
        torch.cuda.empty_cache()

    layer_profiles.sort(key=lambda x: x[0])
    cur_bytes = fixed_bytes + sum(x[1] for x in layer_profiles)
    chosen_thresh = 0.40
    for rel_err, fp_b, q2_b in layer_profiles:
        if cur_bytes <= target_bytes:
            break
        cur_bytes -= (fp_b - q2_b)
        chosen_thresh = max(chosen_thresh, rel_err + 1e-4)
    return min(chosen_thresh, 0.95)


@torch.no_grad()
def repack_safetensors_layerwise(
    input_path: str,
    output_path: str,
    group_size: int = 32,
    theta: float = 1e-4,
    max_rel_error: float = 0.40,
    include_aux_layers: bool = True,
    auto_theta: bool = True,
    cascade: bool = True,
    target_vram_gb: Optional[float] = None,
    channel_perm: bool = True,
    activation_map: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    state_dict = load_file(input_path)
    if target_vram_gb is not None and target_vram_gb > 0:
        max_rel_error = _estimate_threshold_for_vram(
            state_dict, target_vram_gb, group_size, theta, auto_theta
        )

    work_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_tensors: Dict[str, torch.Tensor] = {}
    cascade_groups = (
        sorted(list({g for g in [group_size, 32, 16] if g <= group_size}), reverse=True)
        if cascade else [group_size]
    )

    q2_layers = 0
    q4_layers = 0
    rolled_back_layers = 0
    total_l1_orig = 0.0
    total_l1_err = 0.0

    for name, tensor in state_dict.items():
        if not tensor.is_floating_point() or tensor.numel() == 0:
            out_tensors[name] = tensor.cpu()
            continue
        if not include_aux_layers and tensor.ndim < 2:
            out_tensors[name] = tensor.cpu()
            continue

        w_f32 = tensor.to(device=work_device, dtype=torch.float32)
        total_l1_orig += torch.sum(torch.abs(w_f32)).item()

        act_raw = activation_map.get(name, None) if activation_map is not None else None
        act_s = act_raw.to(device=work_device, dtype=torch.float32) if act_raw is not None else None
        perm = None
        if channel_perm and w_f32.ndim == 2 and w_f32.size(1) >= 16:
            col_l1 = torch.sum(torch.abs(w_f32), dim=0)
            if act_s is not None and act_s.numel() == w_f32.size(1):
                col_l1 = col_l1 * torch.abs(act_s)
            perm = torch.argsort(col_l1, descending=True)
            w_work = w_f32.index_select(1, perm)
            act_work = act_s.index_select(0, perm) if (act_s is not None and act_s.numel() == w_f32.size(1)) else None
        else:
            w_work = w_f32
            act_work = act_s

        res = _quantize_tensor_cascade(
            w_f32=w_work,
            cascade_groups=cascade_groups,
            theta=theta,
            max_rel_error=max_rel_error,
            auto_theta=auto_theta,
            allow_int4=cascade,
            act_scales=act_work,
        )

        if res is not None:
            grid, scales, chosen_g, bits, layer_err = res
            if bits == 2:
                packed_bytes, orig_shape = pack_ternary_to_uint8(grid)
                q2_layers += 1
            else:
                packed_bytes, orig_shape = pack_int4_to_uint8(grid)
                q4_layers += 1

            out_tensors[f"{name}.vcore_packed"] = packed_bytes.cpu()
            out_tensors[f"{name}.vcore_scales"] = scales.to(torch.float16).cpu()
            out_tensors[f"{name}.vcore_shape"] = torch.tensor(orig_shape, dtype=torch.int64)
            out_tensors[f"{name}.vcore_group"] = torch.tensor([chosen_g], dtype=torch.int64)
            out_tensors[f"{name}.vcore_bits"] = torch.tensor([bits], dtype=torch.int64)
            if perm is not None:
                out_tensors[f"{name}.vcore_perm"] = perm.to(torch.int64).cpu()
            total_l1_err += layer_err
        else:
            out_tensors[name] = tensor.cpu()
            rolled_back_layers += 1

        del w_f32, w_work
        if act_s is not None:
            del act_s

    del state_dict
    gc.collect()
    if work_device.type == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(out_tensors, output_path)

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "quantized_158bit_layers": q2_layers,
        "quantized_int4_layers": q4_layers,
        "rolled_back_fp16_layers": rolled_back_layers,
        "effective_max_rel_error": max_rel_error,
        "orig_size_mb": orig_mb,
        "packed_size_mb": new_mb,
        "compression_ratio": orig_mb / max(new_mb, 1e-6),
        "total_l1_orig": total_l1_orig,
        "total_l1_err": total_l1_err,
        }
def repack_sharded_model(
    input_dir: str,
    output_dir: str,
    group_size: int = 32,
    theta: float = 1e-4,
    max_rel_error: float = 0.40,
    include_aux_layers: bool = True,
    auto_theta: bool = True,
    cascade: bool = True,
    target_vram_gb: Optional[float] = None,
    activation_map: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"В папке {input_dir} не найдены файлы .safetensors")

    per_shard_vram = (target_vram_gb / len(shard_files)) if target_vram_gb else None
    agg = {
        "shards_processed": len(shard_files),
        "quantized_158bit_layers": 0,
        "quantized_int4_layers": 0,
        "rolled_back_fp16_layers": 0,
        "orig_size_mb": 0.0,
        "packed_size_mb": 0.0,
        "total_l1_orig": 0.0,
        "total_l1_err": 0.0,
    }
    new_weight_map: Dict[str, str] = {}

    for shard_path in shard_files:
        fname = os.path.basename(shard_path)
        out_shard = os.path.join(output_dir, fname)
        st = repack_safetensors_layerwise(
            shard_path,
            out_shard,
            group_size=group_size,
            theta=theta,
            max_rel_error=max_rel_error,
            include_aux_layers=include_aux_layers,
            auto_theta=auto_theta,
            cascade=cascade,
            target_vram_gb=per_shard_vram,
            activation_map=activation_map,
        )
        for k in (
            "quantized_158bit_layers",
            "quantized_int4_layers",
            "rolled_back_fp16_layers",
            "orig_size_mb",
            "packed_size_mb",
            "total_l1_orig",
            "total_l1_err",
        ):
            agg[k] += st[k]

        for k in load_file(out_shard).keys():
            new_weight_map[k] = fname

    for item in os.listdir(input_dir):
        src_item = os.path.join(input_dir, item)
        dst_item = os.path.join(output_dir, item)
        if item == "model.safetensors.index.json":
            with open(src_item, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            idx_data["weight_map"] = new_weight_map
            idx_data.setdefault("metadata", {})["vcore_quantized"] = "1.58bit-int4-cascade"
            with open(dst_item, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)
        elif not item.endswith(".safetensors") and os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)

    agg["compression_ratio"] = agg["orig_size_mb"] / max(agg["packed_size_mb"], 1e-6)
    return agg


def repack_hf_hub_model(
    repo_id: str,
    output_dir: str,
    group_size: int = 32,
    theta: float = 1e-4,
    max_rel_error: float = 0.40,
    token: Optional[str] = None,
    target_vram_gb: Optional[float] = None,
    activation_map: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(output_dir, exist_ok=True)
    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id)
    shard_names = sorted([f for f in repo_files if f.endswith(".safetensors")])
    if not shard_names:
        raise FileNotFoundError(f"В репозитории {repo_id} нет файлов .safetensors")

    per_shard_vram = (target_vram_gb / len(shard_names)) if target_vram_gb else None
    agg = {
        "shards_processed": len(shard_names),
        "quantized_158bit_layers": 0,
        "quantized_int4_layers": 0,
        "rolled_back_fp16_layers": 0,
        "orig_size_mb": 0.0,
        "packed_size_mb": 0.0,
        "total_l1_orig": 0.0,
        "total_l1_err": 0.0,
    }
    new_weight_map: Dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp_dl:
        for fname in repo_files:
            if not fname.endswith(".safetensors") and "/" not in fname:
                local_cfg = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=tmp_dl, token=token)
                if fname != "model.safetensors.index.json":
                    shutil.copy2(local_cfg, os.path.join(output_dir, fname))

        for s_name in shard_names:
            local_shard = hf_hub_download(repo_id=repo_id, filename=s_name, local_dir=tmp_dl, token=token)
            out_shard = os.path.join(output_dir, os.path.basename(s_name))
            st = repack_safetensors_layerwise(
                local_shard,
                out_shard,
                group_size=group_size,
                theta=theta,
                max_rel_error=max_rel_error,
                target_vram_gb=per_shard_vram,
                activation_map=activation_map,
            )
            for k in (
                "quantized_158bit_layers",
                "quantized_int4_layers",
                "rolled_back_fp16_layers",
                "orig_size_mb",
                "packed_size_mb",
                "total_l1_orig",
                "total_l1_err",
            ):
                agg[k] += st[k]

            for k in load_file(out_shard).keys():
                new_weight_map[k] = os.path.basename(s_name)

            if os.path.exists(local_shard):
                os.remove(local_shard)

        idx_src = os.path.join(tmp_dl, "model.safetensors.index.json")
        if os.path.exists(idx_src):
            with open(idx_src, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            idx_data["weight_map"] = new_weight_map
            idx_data.setdefault("metadata", {})["vcore_quantized"] = "1.58bit-int4-cascade"
            with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)

    agg["compression_ratio"] = agg["orig_size_mb"] / max(agg["packed_size_mb"], 1e-6)
    return agg
