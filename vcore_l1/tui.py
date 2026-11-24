# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import os
import sys
import time
from threading import Thread
from typing import Dict, List, Optional

import torch
from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .cuda_quantizer import verify_with_lean4_engine
from .inference_layer import (
    VCoreDynamicCache,
    VCoreModelForCausalLM,
    _CPU_LIB,
    _CUDA_LIB,
)
from .safetensors_quantizer import repack_hf_hub_model, repack_sharded_model

console = Console()


def _detect_backend_badge() -> str:
    if torch.cuda.is_available() and _CUDA_LIB is not None:
        gpu_name = torch.cuda.get_device_name(0)
        return f"[bold #3FB950]● CUDA WARP/TILED ({gpu_name})[/]"
    if _CPU_LIB is not None:
        return "[bold #58A6FF]● NATIVE AVX2 SIMD (CPU)[/]"
    return "[bold #D29922]● PYTORCH FALLBACK (CPU)[/]"


def _get_dir_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


class VCoreTerminalStudio:
    def __init__(self, initial_model: Optional[str] = None):
        self.model_dir: Optional[str] = initial_model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.history: List[Dict[str, str]] = []
        self.stats = {
            "model_name": "Не загружена",
            "size_mb": 0.0,
            "last_tps": 0.0,
            "last_tokens": 0,
            "last_sec": 0.0,
            "kv_bits": 4,
            "status": "Ожидание команды (/load или /pack)",
        }

    def build_header(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        title = Text("V-CORE (ℓ₁)-CERTIFIED v2.0 • NEURAL STUDIO", style="bold #E6EDF3")
        badge = Text.from_markup(f"{_detect_backend_badge()}  [dim]|[/]  [bold #D2A8FF]Lean 4 v4.0.0 (12 Mods)[/]")
        grid.add_row(title, badge)
        return Panel(grid, style="on #11161F", border_style="#30363D", box=ROUNDED)

    def build_sidebar(self) -> Panel:
        tbl = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        tbl.add_column("K", style="#8B949E")
        tbl.add_column("V", style="bold #E6EDF3", justify="right")

        tbl.add_row("Модель:", self.stats["model_name"][:22])
        tbl.add_row("Размер:", f"{self.stats['size_mb']:.1f} MB")
        tbl.add_row("Устройство:", self.device.upper())
        tbl.add_row("KV-Cache:", f"{self.stats['kv_bits']}-bit L1")
        tbl.add_row("Скорость:", f"[#3FB950]{self.stats['last_tps']:.1f} tok/s[/]")
        tbl.add_row("Токенов:", f"{self.stats['last_tokens']} ({self.stats['last_sec']:.2f}s)")

        cmds = Text(
            "\nКоманды:\n"
            " /load <dir>   Загрузить\n"
            " /pack <id>    Сжать HF\n"
            " /verify       Lean 4\n"
            " /clear        Сброс чата\n"
            " /exit         Выход",
            style="#8B949E",
        )
        status_box = Panel(
            Text(self.stats["status"], style="#58A6FF"),
            title="[dim]Статус ядра[/]",
            border_style="#21262D",
            box=ROUNDED,
        )

        wrap = Table.grid(expand=True)
        wrap.add_row(tbl)
        wrap.add_row(cmds)
        wrap.add_row(status_box)
        return Panel(wrap, title="[bold #58A6FF]Телеметрия V-CORE[/]", style="on #0D1117", border_style="#30363D", box=ROUNDED)

    def build_chat_panel(self, streaming_reply: str = "") -> Panel:
        chat_text = Text()
        if not self.history and not streaming_reply:
            chat_text.append(
                "Добро пожаловать в V-CORE Neural Studio!\n"
                "• Загрузите сжатую модель: /load ./qwen15_vcore\n"
                "• Или сожмите модель с Hugging Face: /pack Qwen/Qwen2.5-0.5B-Instruct\n"
                "• После загрузки просто введите ваш вопрос.\n",
                style="#8B949E",
            )
        else:
            for msg in self.history[-8:]:
                if msg["role"] == "user":
                    chat_text.append("\nВы ➤ ", style="bold #58A6FF")
                    chat_text.append(msg["content"] + "\n", style="#E6EDF3")
                else:
                    chat_text.append("\nV-CORE ● ", style="bold #3FB950")
                    chat_text.append(msg["content"] + "\n", style="#C9D1D9")

            if streaming_reply:
                chat_text.append("\nV-CORE ● ", style="bold #3FB950")
                chat_text.append(streaming_reply + "▌\n", style="#FFFFFF")

        return Panel(
            chat_text,
            title="[bold #3FB950]Диалог с моделью[/]",
            style="on #090C10",
            border_style="#30363D",
            box=ROUNDED,
        )

    def render_layout(self, streaming_reply: str = "") -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self.build_header(), size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(self.build_sidebar(), size=32),
            Layout(self.build_chat_panel(streaming_reply)),
        )
        return layout
    def load_model(self, path: str) -> None:
        from transformers import AutoTokenizer

        if not os.path.exists(path):
            self.stats["status"] = f"Ошибка: путь {path} не найден"
            return

        self.stats["status"] = "Загрузка весов и токенизатора..."
        console.print(self.render_layout())
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = VCoreModelForCausalLM.from_vcore_pretrained(
            path, device=self.device, dtype=self.dtype
        )
        dt = time.perf_counter() - t0
        self.model_dir = path
        self.stats["model_name"] = os.path.basename(os.path.abspath(path))
        self.stats["size_mb"] = _get_dir_size_mb(path)
        self.stats["status"] = f"Модель готова ({dt:.2f}s)"

    def generate_stream(self, user_prompt: str, max_new_tokens: int = 192) -> None:
        if self.model is None or self.tokenizer is None:
            self.history.append({"role": "assistant", "content": "Сначала загрузите модель командой /load <путь>."})
            return

        from transformers import TextIteratorStreamer

        self.history.append({"role": "user", "content": user_prompt})
        messages = [{"role": "system", "content": "You are V-CORE, a helpful AI assistant."}] + self.history[-6:]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = user_prompt

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        kv_cache = VCoreDynamicCache(group_size=32, bits=self.stats["kv_bits"])
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            past_key_values=kv_cache,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        self.stats["status"] = "Генерация ответа..."
        t0 = time.perf_counter()
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        acc_text = ""
        tok_count = 0
        with Live(self.render_layout(""), console=console, refresh_per_second=15) as live:
            for chunk in streamer:
                acc_text += chunk
                tok_count += 1
                live.update(self.render_layout(acc_text))

        thread.join()
        dt = max(time.perf_counter() - t0, 1e-6)
        self.history.append({"role": "assistant", "content": acc_text.strip()})
        self.stats["last_tokens"] = tok_count
        self.stats["last_sec"] = dt
        self.stats["last_tps"] = tok_count / dt
        self.stats["status"] = f"Готово ({self.stats['last_tps']:.1f} tok/s)"

    def run(self) -> None:
        if self.model_dir:
            self.load_model(self.model_dir)

        while True:
            console.clear()
            console.print(self.render_layout())
            try:
                user_in = console.input("[bold #58A6FF]Введите запрос или команду ➤ [/]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_in:
                continue
            if user_in in ("/exit", "/quit"):
                break
            if user_in == "/clear":
                self.history.clear()
                self.stats["status"] = "История диалога очищена"
                continue
            if user_in == "/verify":
                res = verify_with_lean4_engine().strip()
                self.history.append({"role": "assistant", "content": res})
                self.stats["status"] = "Сертификат Lean 4 проверен"
                continue
            if user_in.startswith("/load "):
                target = user_in[6:].strip()
                self.load_model(target)
                continue
            if user_in.startswith("/pack "):
                repo = user_in[6:].strip()
                out_dir = "./" + repo.split("/")[-1] + "-vcore"
                self.stats["status"] = f"Сжатие {repo}..."
                console.clear()
                console.print(self.render_layout())
                if os.path.isdir(repo):
                    st = repack_sharded_model(repo, out_dir, group_size=64)
                else:
                    st = repack_hf_hub_model(repo, out_dir, group_size=64)
                self.history.append({
                    "role": "assistant",
                    "content": f"Модель сжата в {out_dir} (сжатие {st['compression_ratio']:.2f}x, {st['packed_size_mb']:.1f} MB). Загружаю..."
                })
                self.load_model(out_dir)
                continue

            self.generate_stream(user_in)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V-CORE Terminal Neural Studio (TUI)")
    parser.add_argument("--model", "-m", default=None, help="Путь к сжатой модели V-CORE")
    args = parser.parse_args()

    studio = VCoreTerminalStudio(initial_model=args.model)
    studio.run()


if __name__ == "__main__":
    main()
