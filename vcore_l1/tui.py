# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import json
import os
import shlex
import threading
import time
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from .backend_dispatcher import get_dispatcher
from .calibration import collect_activation_importance
from .gguf_exporter import export_vcore_to_gguf
from .inference_layer import VCoreDynamicCache, VCoreLinear, VCoreModelForCausalLM
from .resonance_scanner import (
    map_cognitive_domains,
    scan_model_tomography,
    surgical_unlearn_concept,
)
from .safetensors_quantizer import (
    repack_hf_hub_model,
    repack_safetensors_layerwise,
    repack_sharded_model,
)
from .vcore_bundle import unpack_vcore_bundle_to_temp


BANNER = r"""
====================================================================
  V-CORE L1-CERTIFIED v5.0 | NEURO-LAB TERMINAL STUDIO (CLI / TUI)
  1.58-bit / E8 Spherical / Integer-Only Q14 / Lean 4 Certified
====================================================================
"""


class VCoreTerminalStudio:
    def __init__(self, initial_model: str = None, device: str = None):
        self.dispatcher = get_dispatcher()
        self.hw = self.dispatcher.profile
        self.device = device or self.hw.recommended_device
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.certificate_info = None
        self._tmp_bundle = None
        self.history = []
        self.domain_map = {}
        self.last_thought_trace = []
        self.context_tokens_est = 0
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.kv_mode = self.hw.recommended_kv_mode
        self.integer_only = self.hw.recommended_integer_only

        if initial_model:
            self.load_model(initial_model)

    def print_help(self):
        print(
            "\nДоступные команды V-CORE Neuro-Lab Studio:\n"
            "  /pack <hf_repo_или_путь> <выходная_папка> [--vram ГБ] [--calib N] [--spherical-emb] [--gguf файл.gguf]\n"
            "                        — Сжать модель (авто-бюджет железа + ЭМРТ нейрогенез)\n"
            "  /load <путь_к_папке_или_.vcore> [--integer-only]\n"
            "                        — Загрузить модель и автоматически разметить домены знаний\n"
            "  /scan                 — ЭМРТ-томограмма весов + 4-зонная карта мозга\n"
            "  /domains              — Показать распределение слоёв по доменам (Математика/Физика/История/Код)\n"
            "  /xray <имя_слоя>      — Рентген конкретного слоя (домен, битность, % весов -1/0/+1)\n"
            "  /replay               — Замедленный повтор цепочки активаций последнего ответа\n"
            "  /export-brain <файл.brain.json>\n"
            "                        — Сохранить паспорт-карту мозга (совместим с vcore-web)\n"
            "  /unlearn <фраза>      — Точечно вырезать знание/тему из весов модели (ЭМРТ + Null-Space)\n"
            "  /hw                   — Показать профиль железа и активный C-ABI бэкенд\n"
            "  /clear                — Очистить историю диалога и сбросить давление KV-кэша\n"
            "  /settings             — Показать текущие параметры генерации и KV Lock\n"
            "  /tokens <N>           — Установить лимит новых токенов\n"
            "  /temp <0.0..1.5>      — Установить температуру генерации\n"
            "  /kv <int4|spherical|ternary>\n"
            "                        — Режим сжатия KV-кэша на лету\n"
            "  /help                 — Показать эту справку\n"
            "  /exit                 — Выйти из программы\n"
        )

    def _compute_kv_lock(self):
        rho = min(255.8, (self.context_tokens_est / 4096.0) * 256.0)
        p_in = (256.0 - rho) / (rho + 1.0)
        le0 = max(0.0, 0.024 * (1.0 - (1.0 - p_in) / (p_in + 0.001)))
        return round(p_in, 1), round(le0, 4)

    def render_brain_map(self, nodes_info: list, active_zones: set = None, title: str = "ЭМРТ-ТОМОГРАММА МОЗГА"):
        CYAN, MAGENTA, GOLD, GREEN, DIM, RESET = "\033[96m", "\033[95m", "\033[93m", "\033[92m", "\033[90m", "\033[0m"
        active_zones = active_zones or set()

        template = [
            "         .---======---.         ",
            "       /  A    A    A   \\       ",
            "      / A   A    M    M  \\      ",
            "     |  S  A   M   M   M  |     ",
            "     | S  S   M   M   M   |     ",
            "      \\  S   M   M   B   /      ",
            "       \\   '---'   B   /        ",
            "         '-----------'          ",
        ]

        zone_counts = {"S": 0, "A": 0, "M": 0, "B": 0}
        dom_counts = {"math": 0, "physics": 0, "history": 0, "code": 0, "general": 0}

        for item in nodes_info:
            n = str(item.get("name", "")).lower()
            if "embed" in n or "layers.0." in n:
                zone_counts["S"] += 1
            elif "lm_head" in n or "norm" in n:
                zone_counts["B"] += 1
            elif "attn" in n or ("proj" in n and any(k in n for k in ("q_", "k_", "v_", "o_"))):
                zone_counts["A"] += 1
            else:
                zone_counts["M"] += 1

            d = item.get("domain") or self.domain_map.get(item.get("name", ""), {}).get("domain", "general")
            if d in dom_counts:
                dom_counts[d] += 1

        p_in, le0 = self._compute_kv_lock()
        side_panel = [
            f"{CYAN}Функциональные доли:{RESET}",
            f"  {CYAN}◉ Ввод (Височная)  :{RESET} {zone_counts['S']}",
            f"  {MAGENTA}◉ Внимание (Темен.):{RESET} {zone_counts['A']}",
            f"  {GOLD}◉ Логика (Лобная)  :{RESET} {zone_counts['M']}",
            f"  {GREEN}◉ Выход (Брока)    :{RESET} {zone_counts['B']}",
            f"{GOLD}Когнитивные домены:{RESET}",
            f"  Матем: {dom_counts['math']} | Физ: {dom_counts['physics']} | Ист: {dom_counts['history']} | Код: {dom_counts['code']}",
            f"{GREEN}KV Electric Lock:{RESET} P_in={p_in} | Le₀={le0}",
        ]

        print(f"\n{CYAN}=== {title} (Узлов: {len(nodes_info)}) ==={RESET}")
        for idx, row in enumerate(template):
            out = []
            for ch in row:
                if ch in zone_counts:
                    has_nodes = zone_counts[ch] > 0
                    is_hot = ch in active_zones
                    sym = "◉" if (is_hot or has_nodes) else "·"
                    if ch == "S":
                        out.append(f"{CYAN}{sym}{RESET}")
                    elif ch == "A":
                        out.append(f"{MAGENTA}{sym}{RESET}")
                    elif ch == "M":
                        out.append(f"{GOLD}{sym}{RESET}")
                    elif ch == "B":
                        out.append(f"{GREEN}{sym}{RESET}")
                else:
                    out.append(f"{DIM}{ch}{RESET}")
            right_txt = side_panel[idx] if idx < len(side_panel) else ""
            print("  " + "".join(out) + "   " + right_txt)
        print()
            def load_model(self, model_dir: str, integer_only: bool = None):
        if integer_only is not None:
            self.integer_only = integer_only

        load_target = model_dir
        self.certificate_info = None
        if os.path.isfile(model_dir) and model_dir.endswith(".vcore"):
            print(f"[V-CORE] Распаковка единого бандла {model_dir}...")
            self._tmp_bundle, self.certificate_info = unpack_vcore_bundle_to_temp(model_dir)
            load_target = self._tmp_bundle.name

        print(f"[V-CORE] Загрузка модели из '{model_dir}' на {self.device.upper()} (integer_only={self.integer_only})...")
        t0 = time.time()
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=True)
        self.model = VCoreModelForCausalLM.from_vcore_pretrained(
            load_target, device=self.device, dtype=dtype, integer_only=self.integer_only
        )
        self.model_path = model_dir
        self.history = []
        self.context_tokens_est = 0
        self.last_thought_trace = []
        print("[V-CORE ЭМРТ] Картирование доменов знаний (Математика / Физика / История / Код)...")
        self.domain_map = map_cognitive_domains(self.model, self.tokenizer)
        if self.certificate_info:
            drift = self.certificate_info.get("certified_max_block_drift", "N/A")
            print(f"[Lean 4 Passport] {self.certificate_info.get('certificate_standard')} | Max Block Drift <= {drift}")
        print(f"[V-CORE] Модель и нейро-карта готовы за {time.time() - t0:.2f} сек!\n")

    def cmd_pack(self, args_list):
        if len(args_list) < 2:
            print("Использование: /pack <hf_repo_или_путь> <выходная_папка> [--vram 4.0] [--calib 16] [--spherical-emb] [--gguf out.gguf]")
            return

        src = args_list[0]
        dst = args_list[1]
        target_vram = self.hw.recommended_vram_budget_gb
        calib_samples = 0
        gguf_out = None
        spherical_emb = self.hw.recommended_spherical_emb

        i = 2
        while i < len(args_list):
            if args_list[i] == "--vram" and i + 1 < len(args_list):
                target_vram = float(args_list[i + 1])
                i += 2
            elif args_list[i] == "--calib" and i + 1 < len(args_list):
                calib_samples = int(args_list[i + 1])
                i += 2
            elif args_list[i] == "--gguf" and i + 1 < len(args_list):
                gguf_out = args_list[i + 1]
                i += 2
            elif args_list[i] == "--spherical-emb":
                spherical_emb = True
                i += 1
            else:
                i += 1

        print(f"[V-CORE] Авто-профиль железа: бюджет={target_vram} ГБ, spherical_emb={spherical_emb}")
        act_map, chol_map = None, None
        if calib_samples > 0:
            print(f"[V-CORE] Калибровка активаций и фактора Холецкого ({calib_samples} примеров)...")
            act_map, chol_map = collect_activation_importance(src, num_samples=calib_samples, collect_covariance=True)

        packed_nodes = []

        def on_tui_layer(evt):
            packed_nodes.append(evt)
            pct = int(evt.get("progress", 0) * 100)
            print(
                f"\r[ЭМРТ-Нейрогенез {pct:>3}%] Слой #{len(packed_nodes)}: {evt['name']:<40} [{evt['mode']}]",
                end="",
                flush=True,
            )

        if os.path.isfile(src):
            stats = repack_safetensors_layerwise(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map,
                spherical_emb=spherical_emb, use_tomography=True, progress_callback=on_tui_layer
            )
        elif os.path.isdir(src):
            stats = repack_sharded_model(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map,
                spherical_emb=spherical_emb, use_tomography=True, progress_callback=on_tui_layer
            )
        else:
            stats = repack_hf_hub_model(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map,
                spherical_emb=spherical_emb, use_tomography=True, progress_callback=on_tui_layer
            )

        print()
        self.render_brain_map(packed_nodes, active_zones={"S", "A", "M", "B"}, title="КАРТА МОЗГА ПОСЛЕ СЖАТИЯ")

        print("=== ОТЧЁТ О СЖАТИИ V-CORE ===")
        for k, v in stats.items():
            print(f"  {k:<25}: {v:.3f}" if isinstance(v, float) else f"  {k:<25}: {v}")

        if gguf_out and os.path.isdir(dst):
            g_stats = export_vcore_to_gguf(dst, gguf_out)
            print(f"=== GGUF ЭКСПОРТ ({gguf_out}) === {g_stats}")

    def cmd_scan(self):
        if self.model is None:
            print("[!] Сначала загрузите модель командой /load <папка_модели_или_.vcore>\n")
            return
        tomo = scan_model_tomography(self.model)
        print(f"\n=== ЭМРТ-ТОМОГРАММА МОДЕЛИ ({len(tomo)} слоёв) ===")
        for l_name, info in list(tomo.items())[:15]:
            dom = self.domain_map.get(l_name, {}).get("domain", "general")
            print(
                f"  {l_name:<36} | Dom={dom:<7} | Peak={info['in_peak_ratio']:>5.1f} | "
                f"Imp={info['mean_impedance']:>5.1f} -> {info['recommended_mode']}"
            )
        if len(tomo) > 15:
            print(f"  ... и ещё {len(tomo) - 15} слоёв.")

        nodes_list = [
            {"name": k, "mode": v["recommended_mode"], "domain": self.domain_map.get(k, {}).get("domain", "general")}
            for k, v in tomo.items()
        ]
        self.render_brain_map(nodes_list, active_zones={"S", "A", "M", "B"}, title="ЭМРТ-СКАН ЗАГРУЖЕННОЙ МОДЕЛИ")

    def cmd_domains(self):
        if not self.domain_map:
            print("[!] Карта доменов пуста. Загрузите модель командой /load\n")
            return
        grouped = {"math": [], "physics": [], "history": [], "code": [], "general": []}
        for name, info in self.domain_map.items():
            d = info.get("domain", "general")
            sc = info.get("scores", {}).get(d, 0.0)
            grouped.setdefault(d, []).append((name, sc))

        print("\n=== КОГНИТИВНАЯ КАРТА ДОМЕНОВ МОДЕЛИ ===")
        titles = {
            "math": "МАТЕМАТИКА",
            "physics": "ФИЗИКА",
            "history": "ИСТОРИЯ",
            "code": "КОД / АЛГОРИТМЫ",
            "general": "ОБЩАЯ СЕМАНТИКА",
        }
        for key in ("math", "physics", "history", "code", "general"):
            items = sorted(grouped[key], key=lambda x: x[1], reverse=True)
            top_str = ", ".join([f"{n} (+{s*100:.1f}%)" for n, s in items[:3]]) if items else "нет"
            print(f"  {titles[key]:<18} [{len(items):>3} слоёв] | Топ-узлы: {top_str}")
        print()

    def cmd_xray(self, query: str):
        if self.model is None:
            print("[!] Сначала загрузите модель командой /load\n")
            return
        q = query.strip()
        if not q:
            print("Использование: /xray <часть_имени_слоя, например layers.12.mlp.down_proj>\n")
            return
        for name, mod in self.model.named_modules():
            if q in name and isinstance(mod, (torch.nn.Linear, VCoreLinear)):
                w_name = f"{name}.weight"
                dom_info = self.domain_map.get(w_name, {})
                print(f"\n=== РЕНТГЕН НЕЙРОНА: {w_name} ===")
                print(f"  Домен знаний : {dom_info.get('domain', 'general')} {dom_info.get('scores', {})}")
                if isinstance(mod, VCoreLinear):
                    from .safetensors_quantizer import unpack_uint8_to_ternary
                    n_el = mod.out_features * mod.in_features
                    print(f"  Форма        : [{mod.out_features} x {mod.in_features}] | Битность: {mod.bits}-bit | Group: {mod.group_size}")
                    if mod.bits == 2:
                        vals = unpack_uint8_to_ternary(mod.vcore_packed, (n_el,))
                        neg = float((vals < -0.5).float().mean().item()) * 100
                        zero = float((vals.abs() < 0.5).float().mean().item()) * 100
                        pos = float((vals > 0.5).float().mean().item()) * 100
                        print(f"  Веса (-1/0/+1): -1: {neg:.1f}% | 0: {zero:.1f}% | +1: {pos:.1f}%\n")
                else:
                    w = mod.weight.detach()
                    print(f"  Форма        : {list(w.shape)} | Точность: FP16/FP32\n")
                return
        print(f"[!] Слой по фильтру '{q}' не найден.\n")

    def cmd_export_brain(self, out_path: str):
        if self.model is None:
            print("[!] Сначала загрузите модель командой /load\n")
            return
        path = out_path.strip() or "vcore_model_map.brain.json"
        tomo = scan_model_tomography(self.model)
        nodes = []
        for idx, (name, info) in enumerate(tomo.items()):
            d_entry = self.domain_map.get(name, {})
            nodes.append({
                "id": idx,
                "name": name,
                "shape": list(info.get("shape", [])),
                "mode": info.get("recommended_mode", "TERNARY_158BIT"),
                "impedance": info.get("mean_impedance", 1.0),
                "domain": d_entry.get("domain", "general"),
                "domainScores": d_entry.get("scores", {}),
            })
        card = {
            "format": "VCORE-BRAIN-PASSPORT-V2",
            "model_path": self.model_path,
            "total_nodes": len(nodes),
            "nodes": nodes,
            "last_thought_trace": self.last_thought_trace,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        print(f"[V-CORE] Паспорт мозга ({len(nodes)} узлов) сохранён в '{path}'.\n")
                def cmd_replay(self):
        if not self.last_thought_trace:
            print("[!] Сначала отправьте запрос в чат, чтобы записать траекторию импульсов.\n")
            return
        print(f"\n=== ЗАМЕДЛЕННЫЙ ПОВТОР МЫСЛИ ({len(self.last_thought_trace)} шагов) ===")
        for idx, step in enumerate(self.last_thought_trace, 1):
            hot = step["hot_layers"][0] if step["hot_layers"] else "MLP"
            dom = step.get("domain", "general").upper()
            print(f"  [{idx:>2}] Токен: {step['text']!r:<14} ➔ Узел: {hot:<34} | Домен: {dom}")
            time.sleep(0.12)
        print()

    def cmd_unlearn(self, concept_text: str):
        if self.model is None or self.tokenizer is None:
            print("[!] Сначала загрузите модель командой /load <папка_модели_или_.vcore>\n")
            return
        if not concept_text.strip():
            print("Использование: /unlearn <фраза или тема для вырезания>\n")
            return
        print(f"[V-CORE ЭМРТ] Поиск и ортогональное вырезание паттерна: '{concept_text}'...")
        res = surgical_unlearn_concept(self.model, self.tokenizer, target_prompts=[concept_text], strength=0.85)
        self.history = []
        self.context_tokens_est = 0
        print(
            f"[V-CORE ЭМРТ] Вырезано из {res['edited_layers']} слоёв "
            f"(удалено L1-энергии: {res['total_l1_energy_ablated']}). История чата сброшена.\n"
        )

    def chat_turn(self, user_text: str):
        if self.model is None or self.tokenizer is None:
            print("[!] Сначала загрузите модель командой /load <папка_модели_или_.vcore>\n")
            return

        self.history.append({"role": "user", "content": user_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in self.history]) + "\nassistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.context_tokens_est += int(inputs["input_ids"].shape[1])
        kv_cache = VCoreDynamicCache(group_size=32, mode=self.kv_mode)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        live_energies = {}
        hooks = []

        def _make_hook(l_name):
            def _hook(_mod, _inp, out):
                if isinstance(out, torch.Tensor):
                    live_energies[l_name] = float(out.detach().abs().mean().item())
            return _hook

        for m_name, mod in self.model.named_modules():
            if isinstance(mod, (torch.nn.Linear, VCoreLinear)) and (
                m_name.endswith("down_proj") or m_name.endswith("o_proj")
            ):
                hooks.append(mod.register_forward_hook(_make_hook(f"{m_name}.weight")))

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            past_key_values=kv_cache,
            streamer=streamer,
            do_sample=(self.temperature > 0.0),
            temperature=max(self.temperature, 1e-4) if self.temperature > 0.0 else None,
        )

        print("\n[Импульс ➔ Ввод (S) ➔ Внимание (A) ➔ Логика MLP (M) ➔ Выход Брока (B)]")
        t0 = time.time()
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        print("Assistant > ", end="", flush=True)
        full_reply = []
        token_count = 0
        self.last_thought_trace = []
        domain_hits = {"math": 0, "physics": 0, "history": 0, "code": 0, "general": 0}

        try:
            for piece in streamer:
                print(piece, end="", flush=True)
                full_reply.append(piece)
                token_count += 1
                self.context_tokens_est += 1

                sorted_hot = sorted(live_energies.items(), key=lambda x: x[1], reverse=True)
                hot_names = [k for k, _ in sorted_hot[:3]]
                top_layer = hot_names[0] if hot_names else ""
                step_dom = self.domain_map.get(top_layer, {}).get("domain", "general")
                if step_dom in domain_hits:
                    domain_hits[step_dom] += 1

                self.last_thought_trace.append({
                    "text": piece,
                    "hot_layers": hot_names,
                    "domain": step_dom,
                    "energy": round(sorted_hot[0][1], 3) if sorted_hot else 0.5,
                })
        finally:
            thread.join()
            for h in hooks:
                h.remove()

        elapsed = max(time.time() - t0, 1e-5)
        reply_text = "".join(full_reply)
        self.history.append({"role": "assistant", "content": reply_text})

        hot_layer = max(live_energies.items(), key=lambda x: x[1])[0] if live_energies else "N/A"
        dominant_dom = max(domain_hits.items(), key=lambda x: x[1])[0].upper()
        p_in, le0 = self._compute_kv_lock()
        print(
            f"\n\n[Скорость: {token_count / elapsed:.1f} ток/с | Домен: {dominant_dom} | "
            f"Пик: {hot_layer} | KV P_in={p_in} (Le₀={le0})]\n"
        )

    def run(self):
        print(BANNER)
        print(f"[HW-DISPATCH] {self.hw.summary_banner()}")
        print(f"[ABI-BACKEND] Active native kernel: {self.dispatcher.active_backend_name()}")
        self.print_help()
        while True:
            try:
                raw = input("vcore> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход из V-CORE Studio.")
                break

            if not raw:
                continue
            if raw.startswith("/"):
                parts = shlex.split(raw)
                cmd = parts[0].lower()
                if cmd in ("/exit", "/quit"):
                    break
                elif cmd == "/help":
                    self.print_help()
                elif cmd == "/hw":
                    print(f"[HW-DISPATCH] {self.hw.summary_banner()}")
                    print(f"[ABI-BACKEND] Active native kernel: {self.dispatcher.active_backend_name()}")
                elif cmd == "/scan":
                    self.cmd_scan()
                elif cmd == "/domains":
                    self.cmd_domains()
                elif cmd == "/xray":
                    self.cmd_xray(raw[len("/xray"):].strip())
                elif cmd == "/replay":
                    self.cmd_replay()
                elif cmd == "/export-brain":
                    self.cmd_export_brain(raw[len("/export-brain"):].strip())
                elif cmd == "/unlearn":
                    self.cmd_unlearn(raw[len("/unlearn"):].strip())
                elif cmd == "/clear":
                    self.history = []
                    self.context_tokens_est = 0
                    print("[V-CORE] История диалога и давление KV-кэша сброшены.")
                elif cmd == "/settings":
                    p_in, le0 = self._compute_kv_lock()
                    print(
                        f"Модель: {self.model_path} | Устройство: {self.device} | "
                        f"Max tokens: {self.max_new_tokens} | Temp: {self.temperature} | "
                        f"KV-кэш: {self.kv_mode} (P_in={p_in}, Le₀={le0}) | Integer-Only: {self.integer_only}"
                    )
                elif cmd == "/tokens" and len(parts) > 1:
                    self.max_new_tokens = int(parts[1])
                    print(f"[V-CORE] max_new_tokens = {self.max_new_tokens}")
                elif cmd == "/temp" and len(parts) > 1:
                    self.temperature = float(parts[1])
                    print(f"[V-CORE] temperature = {self.temperature}")
                elif cmd == "/kv" and len(parts) > 1 and parts[1] in ("int4", "spherical", "ternary"):
                    self.kv_mode = parts[1]
                    print(f"[V-CORE] Режим KV-кэша = {self.kv_mode}")
                elif cmd == "/load" and len(parts) > 1:
                    int_only = ("--integer-only" in parts)
                    self.load_model(parts[1], integer_only=int_only if int_only else None)
                elif cmd == "/pack":
                    self.cmd_pack(parts[1:])
                else:
                    print("[!] Неизвестная команда. Введите /help для списка команд.")
            else:
                self.chat_turn(raw)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Neuro-Lab Terminal Studio")
    parser.add_argument("--model", "-m", default=None, help="Путь к сжатой модели или файлу .vcore")
    parser.add_argument("--device", "-d", default=None, help="cuda, mps или cpu (по умолчанию автовыбор)")
    args = parser.parse_args()

    studio = VCoreTerminalStudio(initial_model=args.model, device=args.device)
    studio.run()


if __name__ == "__main__":
    main()
            
