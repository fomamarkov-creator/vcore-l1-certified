# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from .hardware_probe import probe_hardware
from .inference_layer import VCoreDynamicCache, VCoreModelForCausalLM
from .vcore_bundle import unpack_vcore_bundle_to_temp


class VCoreOpenAIServerState:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        kv_mode: Optional[str] = None,
        integer_only: Optional[bool] = None,
    ):
        self.hw = probe_hardware(preferred_device=device)
        self.device = self.hw.recommended_device
        self.kv_mode = kv_mode or self.hw.recommended_kv_mode
        self.integer_only = self.hw.recommended_integer_only if integer_only is None else integer_only
        self.model_id = os.path.basename(os.path.abspath(model_path))
        self.certificate: Dict[str, Any] = {"certificate_standard": "Lean4-MarkovL1-v5.0"}
        self._tmp_bundle_dir = None

        load_dir = model_path
        if os.path.isfile(model_path) and model_path.endswith(".vcore"):
            self._tmp_bundle_dir, self.certificate = unpack_vcore_bundle_to_temp(model_path)
            load_dir = self._tmp_bundle_dir.name

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
        self.model = VCoreModelForCausalLM.from_vcore_pretrained(
            load_dir, device=self.device, dtype=dtype, integer_only=self.integer_only
        )

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        parts = [f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}" for m in messages]
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def generate_text(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
        prompt = self.format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        cache = VCoreDynamicCache(group_size=32, mode=self.kv_mode)
        do_sample = temperature > 1e-4
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                past_key_values=cache,
                do_sample=do_sample,
                temperature=max(temperature, 1e-4) if do_sample else None,
            )
        gen_only = out_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_only, skip_special_tokens=True)

    def stream_text(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7):
        prompt = self.format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        cache = VCoreDynamicCache(group_size=32, mode=self.kv_mode)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        do_sample = temperature > 1e-4

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            past_key_values=cache,
            streamer=streamer,
            do_sample=do_sample,
            temperature=max(temperature, 1e-4) if do_sample else None,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece


def create_openai_handler(state: VCoreOpenAIServerState):
    class OpenAICompatHandler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [{
                        "id": state.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "vcore-l1-certified",
                        "lean4_certificate": state.certificate,
                    }],
                })
            elif self.path.rstrip("/") == "/v1/vcore/status":
                self._send_json(200, {
                    "model": state.model_id,
                    "hardware": state.hw.to_dict(),
                    "kv_mode": state.kv_mode,
                    "integer_only": state.integer_only,
                    "lean4_certificate": state.certificate,
                })
            else:
                self._send_json(404, {"error": "Endpoint not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": "Endpoint not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages", [])
            max_tokens = int(body.get("max_tokens", 256))
            temperature = float(body.get("temperature", 0.7))
            stream = bool(body.get("stream", False))
            comp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if not stream:
                answer = state.generate_text(messages, max_tokens=max_tokens, temperature=temperature)
                self._send_json(200, {
                    "id": comp_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": state.model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }],
                })
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for token_str in state.stream_text(messages, max_tokens=max_tokens, temperature=temperature):
                chunk = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_id,
                    "choices": [{"index": 0, "delta": {"content": token_str}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            end_chunk = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(end_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
            self.wfile.flush()

        def log_message(self, fmt, *args):
            pass

    return OpenAICompatHandler


def main():
    parser = argparse.ArgumentParser(description="V-CORE OpenAI-Compatible API Server (vcore-serve)")
    parser.add_argument("--model", "-m", required=True, help="Путь к папке модели V-CORE или файлу .vcore")
    parser.add_argument("--host", default="127.0.0.1", help="Хост сервера")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Порт сервера")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (по умолчанию автовыбор)")
    parser.add_argument("--kv-mode", default=None, help="int4 / spherical / ternary")
    parser.add_argument("--integer-only", action="store_true", help="Включить целочисленный Q14 режим")
    args = parser.parse_args()

    state = VCoreOpenAIServerState(
        model_path=args.model,
        device=args.device,
        kv_mode=args.kv_mode,
        integer_only=True if args.integer_only else None,
    )
    print(f"[vcore-serve] {state.hw.summary_banner()}")
    print(f"[vcore-serve] OpenAI API запущен на http://{args.host}:{args.port}/v1 (model={state.model_id})")
    server = ThreadingHTTPServer((args.host, args.port), create_openai_handler(state))
    server.serve_forever()


if __name__ == "__main__":
    main()
