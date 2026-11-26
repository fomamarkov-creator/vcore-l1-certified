# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import os
import shlex
import threading
import time
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from .backend_dispatcher import get_dispatcher
from .calibration import collect_activation_importance
from .gguf_exporter import export_vcore_to_gguf
from .inference_layer import VCoreDynamicCache, VCoreModelForCausalLM
from .safetensors_quantizer import (
    repack_safetensors_layerwise,
    repack_sharded_model,
    repack_hf_hub_model,
)
from .vcore_bundle import unpack_vcore_bundle_to_temp


BANNER = r"""
====================================================================
  V-CORE L1-CERTIFIED v5.0 | TERMINAL STUDIO (CLI / TUI)
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
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.kv_mode = self.hw.recommended_kv_mode
        self.integer_only = self.hw.recommended_integer_only

        if initial_model:
            self.load_model(initial_model)

    def print_help(self):
        print(
            "\nДоступные команды V-CORE Studio:\n"
            "  /pack <hf_repo_или_путь> <выходная_папка> [--vram ГБ] [--calib N] [--spherical-emb] [--gguf файл.gguf]\n"
            "                        — Сжать модель (авто-бюджет железа + Cross-Layer компенсация)\n"
            "  /load <путь_к_папке_или_.vcore> [--integer-only]\n"
            "                        — Загрузить сжатую модель или единый бандл .vcore с паспортом Lean 4\n"
            "  /hw                   — Показать профиль железа и активный C-ABI бэкенд\n"
            "  /clear                — Очистить историю диалога\n"
            "  /settings             — Показать текущие параметры генерации\n"
            "  /tokens <N>           — Установить лимит новых токенов\n"
            "  /temp <0.0..1.5>      — Установить температуру генерации\n"
            "  /kv <int4|spherical|ternary>\n"
            "                        — Режим сжатия KV-кэша на лету\n"
            "  /help                 — Показать эту справку\n"
            "  /exit                 — Выйти из программы\n"
        )

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
        if self.certificate_info:
            drift = self.certificate_info.get("certified_max_block_drift", "N/A")
            print(f"[Lean 4 Passport] {self.certificate_info.get('certificate_standard')} | Max Block Drift <= {drift}")
        print(f"[V-CORE] Модель готова за {time.time() - t0:.2f} сек!\n")
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

        if os.path.isfile(src):
            stats = repack_safetensors_layerwise(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map, spherical_emb=spherical_emb
            )
        elif os.path.isdir(src):
            stats = repack_sharded_model(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map, spherical_emb=spherical_emb
            )
        else:
            stats = repack_hf_hub_model(
                src, dst, target_vram_gb=target_vram, activation_map=act_map, chol_map=chol_map, spherical_emb=spherical_emb
            )

        print("\n=== ОТЧЁТ О СЖАТИИ V-CORE ===")
        for k, v in stats.items():
            print(f"  {k:<25}: {v:.3f}" if isinstance(v, float) else f"  {k:<25}: {v}")

        if gguf_out and os.path.isdir(dst):
            g_stats = export_vcore_to_gguf(dst, gguf_out)
            print(f"=== GGUF ЭКСПОРТ ({gguf_out}) === {g_stats}")

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
        kv_cache = VCoreDynamicCache(group_size=32, mode=self.kv_mode)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            past_key_values=kv_cache,
            streamer=streamer,
            do_sample=(self.temperature > 0.0),
            temperature=max(self.temperature, 1e-4) if self.temperature > 0.0 else None,
        )

        t0 = time.time()
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        print("\nAssistant > ", end="", flush=True)
        full_reply = []
        token_count = 0
        for piece in streamer:
            print(piece, end="", flush=True)
            full_reply.append(piece)
            token_count += 1
        thread.join()

        elapsed = max(time.time() - t0, 1e-5)
        reply_text = "".join(full_reply)
        self.history.append({"role": "assistant", "content": reply_text})
        print(f"\n\n[Скорость: {token_count / elapsed:.1f} ток/с | KV-кэш: {self.kv_mode} | Integer-Only: {self.integer_only}]\n")

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
                elif cmd == "/clear":
                    self.history = []
                    print("[V-CORE] История диалога очищена.")
                elif cmd == "/settings":
                    print(
                        f"Модель: {self.model_path} | Устройство: {self.device} | "
                        f"Max tokens: {self.max_new_tokens} | Temp: {self.temperature} | "
                        f"KV-кэш: {self.kv_mode} | Integer-Only: {self.integer_only}"
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
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Interactive Terminal Studio")
    parser.add_argument("--model", "-m", default=None, help="Путь к сжатой модели или файлу .vcore")
    parser.add_argument("--device", "-d", default=None, help="cuda, mps или cpu (по умолчанию автовыбор)")
    args = parser.parse_args()

    studio = VCoreTerminalStudio(initial_model=args.model, device=args.device)
    studio.run()


if __name__ == "__main__":
    main()
            
