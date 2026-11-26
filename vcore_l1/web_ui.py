# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
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


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V-CORE L1-Certified v5.0 | Wave Web Studio</title>
<style>
  :root {
    --bg: #07090f; --panel: #0f1422; --border: #1e293b;
    --accent: #38bdf8; --accent2: #818cf8; --text: #e2e8f0; --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 16px; letter-spacing: 0.4px; }
  .badge { background: rgba(56,189,248,0.15); color: var(--accent); padding: 3px 10px; border-radius: 99px; font-size: 12px; border: 1px solid rgba(56,189,248,0.35); }
  .hw-pill { background: rgba(129,140,248,0.15); color: #c7d2fe; padding: 3px 10px; border-radius: 99px; font-size: 12px; border: 1px solid rgba(129,140,248,0.35); }
  #wave-canvas { width: 100%; height: 64px; background: linear-gradient(180deg, #0b101d 0%, #07090f 100%); border-bottom: 1px solid var(--border); }
  .main { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 320px; background: var(--panel); border-right: 1px solid var(--border); padding: 16px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }
  .card { background: rgba(15,23,42,0.7); border: 1px solid var(--border); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); }
  input, select {
    width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border);
    background: #090d16; color: var(--text); font-size: 13px;
  }
  button {
    padding: 9px 12px; border-radius: 6px; border: none; font-weight: 600; font-size: 13px; cursor: pointer;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #050811; transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button.secondary { background: #1e293b; color: var(--text); }
  .chat-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 82%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #1d4ed8; color: #fff; border-bottom-right-radius: 2px; }
  .msg.bot { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); border-bottom-left-radius: 2px; }
  .input-bar { padding: 14px 20px; background: var(--panel); border-top: 1px solid var(--border); display: flex; gap: 10px; }
  .input-bar input { flex: 1; font-size: 14px; padding: 10px 14px; }
  .status-line { font-size: 12px; color: var(--muted); }
  @media (max-width: 768px) { .main { flex-direction: column; } .sidebar { width: 100%; max-height: 44vh; border-right: none; border-bottom: 1px solid var(--border); } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>V-CORE L1-CERTIFIED</span>
    <span class="badge">v5.0 | E8 &amp; Integer-Only Q14</span>
    <span class="hw-pill" id="hw-banner">Сканирование железа...</span>
  </div>
  <div class="status-line" id="status-text">Модель не загружена</div>
</header>
<canvas id="wave-canvas"></canvas>
<div class="main">
  <div class="sidebar">
    <div class="card">
      <h3>1. Сжатие модели (vcore-pack)</h3>
      <input id="pack-src" placeholder="HF Repo (Qwen/Qwen2.5-1.5B) или путь">
      <input id="pack-dst" placeholder="Выходная папка (./qwen_vcore)" value="./qwen_vcore">
      <div style="display:flex; gap:8px;">
        <input id="pack-vram" type="number" step="0.5" placeholder="Авто-бюджет ГБ">
        <input id="pack-calib" type="number" value="0" placeholder="Калибр.">
      </div>
      <select id="pack-sph">
        <option value="auto">Сферические E8 эмбеддинги: Авто (по памяти)</option>
        <option value="on">Сферические E8 эмбеддинги: Вкл (~1.0 b/w)</option>
        <option value="off">Сферические E8 эмбеддинги: Выкл (FP16)</option>
      </select>
      <button onclick="runPack()">Сжать модель (Cross-Layer + L1)</button>
      <div class="status-line" id="pack-log">Готов к упаковке</div>
    </div>
    <div class="card">
      <h3>2. Загрузка модели / .vcore</h3>
      <input id="load-dir" placeholder="Путь к папке модели или model.vcore" value="./qwen_vcore">
      <div style="display:flex; gap:8px;">
        <select id="load-dev">
          <option value="auto">Device: Авто</option>
          <option value="cuda">CUDA (GPU)</option>
          <option value="mps">Apple MPS</option>
          <option value="cpu">CPU (AVX2 / NEON)</option>
        </select>
        <select id="load-int">
          <option value="false">Float / AVX2</option>
          <option value="true">Integer-Only Q14</option>
        </select>
      </div>
      <button onclick="loadModel()">Загрузить модель</button>
    </div>
    <div class="card">
      <h3>3. Параметры диалога</h3>
      <div style="display:flex; gap:8px;">
        <input id="gen-tokens" type="number" value="256" title="Max tokens">
        <input id="gen-temp" type="number" step="0.1" value="0.7" title="Temperature">
      </div>
      <select id="kv-mode">
        <option value="spherical">KV-кэш: Spherical E8 (1.0 b/w)</option>
        <option value="int4">KV-кэш: INT4</option>
        <option value="ternary">KV-кэш: 2-bit Ternary</option>
      </select>
      <button class="secondary" onclick="clearChat()">Очистить чат</button>
    </div>
  </div>
  <div class="chat-wrap">
    <div id="messages">
      <div class="msg bot">Привет! Я V-CORE Wave Studio v5.0. Железо определено автоматически — выбери или сожми модель слева и пиши свой запрос.</div>
    </div>
    <div class="input-bar">
      <input id="prompt-input" placeholder="Введите сообщение..." onkeydown="if(event.key==='Enter') sendMsg()">
      <button onclick="sendMsg()">Отправить</button>
    </div>
  </div>
</div>
<script>
const canvas = document.getElementById('wave-canvas');
const ctx = canvas.getContext('2d');
let waveAmp = 8, targetAmp = 8, phase = 0;

function resizeCanvas() {
  canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  waveAmp += (targetAmp - waveAmp) * 0.08;
  phase += 0.045;
  const colors = ['rgba(56,189,248,0.85)', 'rgba(129,140,248,0.65)', 'rgba(16,185,129,0.45)'];
  colors.forEach((col, idx) => {
    ctx.beginPath();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.2 * window.devicePixelRatio;
    for (let x = 0; x <= w; x += 6) {
      const freq = 0.008 + idx * 0.003;
      const y = h / 2 + Math.sin(x * freq + phase + idx * 1.3) * waveAmp * window.devicePixelRatio * Math.sin((x / w) * Math.PI);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  requestAnimationFrame(drawWave);
}
drawWave();

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.hw_banner) document.getElementById('hw-banner').textContent = d.hw_banner;
    if (d.recommended_budget && !document.getElementById('pack-vram').value) {
      document.getElementById('pack-vram').placeholder = `Авто: ${d.recommended_budget} ГБ`;
    }
    if (d.loaded) {
      let certTxt = d.cert_drift ? ` | Lean4 Drift <= ${d.cert_drift}` : '';
      document.getElementById('status-text').textContent = `Активна: ${d.model_path} (${d.device.toUpperCase()}${certTxt})`;
    }
  } catch (e) {}
}
refreshStatus();

async function runPack() {
  const src = document.getElementById('pack-src').value.trim();
  const dst = document.getElementById('pack-dst').value.trim();
  const vram = document.getElementById('pack-vram').value;
  const calib = document.getElementById('pack-calib').value;
  const sph = document.getElementById('pack-sph').value;
  if (!src || !dst) return alert('Укажите источник и выходную папку!');
  document.getElementById('pack-log').textContent = 'Сжатие модели...';
  targetAmp = 24;
  const r = await fetch('/api/pack', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({src, dst, vram, calib, sph})
  });
  const d = await r.json();
  targetAmp = 8;
  document.getElementById('pack-log').textContent = d.message || d.error;
  if (!d.error) document.getElementById('load-dir').value = dst;
}

async function loadModel() {
  const model_dir = document.getElementById('load-dir').value.trim();
  const device = document.getElementById('load-dev').value;
  const integer_only = document.getElementById('load-int').value === 'true';
  document.getElementById('status-text').textContent = 'Загрузка модели...';
  targetAmp = 20;
  const r = await fetch('/api/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model_dir, device, integer_only})
  });
  const d = await r.json();
  targetAmp = 8;
  if (d.error) {
    document.getElementById('status-text').textContent = 'Ошибка: ' + d.error;
  } else {
    refreshStatus();
  }
}

async function clearChat() {
  await fetch('/api/clear', {method: 'POST'});
  document.getElementById('messages').innerHTML = '';
}

async function sendMsg() {
  const inp = document.getElementById('prompt-input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  const box = document.getElementById('messages');
  const uDiv = document.createElement('div'); uDiv.className = 'msg user'; uDiv.textContent = text; box.appendChild(uDiv);
  const bDiv = document.createElement('div'); bDiv.className = 'msg bot'; bDiv.textContent = '...'; box.appendChild(bDiv);
  box.scrollTop = box.scrollHeight;
  targetAmp = 26;

  const max_tokens = parseInt(document.getElementById('gen-tokens').value || '256');
  const temperature = parseFloat(document.getElementById('gen-temp').value || '0.7');
  const kv_mode = document.getElementById('kv-mode').value;

  const r = await fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, max_tokens, temperature, kv_mode})
  });
  const d = await r.json();
  targetAmp = 8;
  bDiv.textContent = d.reply || ('Ошибка: ' + d.error);
  if (d.speed) document.getElementById('status-text').textContent = `Скорость: ${d.speed} | KV: ${kv_mode}`;
  box.scrollTop = box.scrollHeight;
}
</script>
</body>
</html>
"""


class WebStudioState:
    def __init__(self):
        self.dispatcher = get_dispatcher()
        self.hw = self.dispatcher.profile
        self.model = None
        self.tokenizer = None
        self.model_path = ""
        self.device = self.hw.recommended_device
        self.certificate_info: Optional[Dict[str, Any]] = None
        self._tmp_bundle = None
        self.history: List[Dict[str, str]] = []
        self.lock = threading.Lock()


def create_handler(state: WebStudioState):
    class VCoreWebHandler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                raw = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/api/status":
                self._send_json(200, {
                    "loaded": state.model is not None,
                    "model_path": state.model_path,
                    "device": state.device,
                    "hw_banner": f"{state.hw.gpu_name} | {state.hw.simd_backend} | RAM {state.hw.ram_free_gb:.1f}GB",
                    "recommended_budget": state.hw.recommended_vram_budget_gb,
                    "cert_drift": state.certificate_info.get("certified_max_block_drift") if state.certificate_info else None,
                })
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            try:
                if self.path == "/api/pack":
                    src = body["src"]
                    dst = body["dst"]
                    vram = float(body["vram"]) if body.get("vram") else state.hw.recommended_vram_budget_gb
                    calib = int(body.get("calib", 0))
                    sph_mode = body.get("sph", "auto")
                    sph_emb = state.hw.recommended_spherical_emb if sph_mode == "auto" else (sph_mode == "on")

                    act_map, chol_map = None, None
                    if calib > 0:
                        act_map, chol_map = collect_activation_importance(src, num_samples=calib, collect_covariance=True)

                    if os.path.isfile(src):
                        st = repack_safetensors_layerwise(src, dst, target_vram_gb=vram, activation_map=act_map, chol_map=chol_map, spherical_emb=sph_emb)
                    elif os.path.isdir(src):
                        st = repack_sharded_model(src, dst, target_vram_gb=vram, activation_map=act_map, chol_map=chol_map, spherical_emb=sph_emb)
                    else:
                        st = repack_hf_hub_model(src, dst, target_vram_gb=vram, activation_map=act_map, chol_map=chol_map, spherical_emb=sph_emb)
                    self._send_json(200, {"message": f"Готово! Сжатие {st['compression_ratio']:.2f}x ({st['packed_size_mb']:.1f} МБ)"})

                elif self.path == "/api/load":
                    with state.lock:
                        model_dir = body["model_dir"]
                        req_dev = body.get("device", "auto")
                        dev = state.hw.recommended_device if req_dev == "auto" else req_dev
                        int_only = bool(body.get("integer_only", False))

                        load_target = model_dir
                        state.certificate_info = None
                        if os.path.isfile(model_dir) and model_dir.endswith(".vcore"):
                            state._tmp_bundle, state.certificate_info = unpack_vcore_bundle_to_temp(model_dir)
                            load_target = state._tmp_bundle.name

                        dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
                        state.tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=True)
                        state.model = VCoreModelForCausalLM.from_vcore_pretrained(
                            load_target, device=dev, dtype=dtype, integer_only=int_only
                        )
                        state.model_path = model_dir
                        state.device = dev
                        state.history = []
                    self._send_json(200, {"ok": True})

                elif self.path == "/api/clear":
                    with state.lock:
                        state.history = []
                    self._send_json(200, {"ok": True})

                elif self.path == "/api/chat":
                    with state.lock:
                        if state.model is None:
                            self._send_json(400, {"error": "Сначала загрузите модель!"})
                            return
                        state.history.append({"role": "user", "content": body["message"]})
                        if hasattr(state.tokenizer, "apply_chat_template"):
                            prompt = state.tokenizer.apply_chat_template(state.history, tokenize=False, add_generation_prompt=True)
                        else:
                            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in state.history]) + "\nassistant: "

                        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)
                        kv_mode = body.get("kv_mode", "spherical")
                        cache = VCoreDynamicCache(group_size=32, mode=kv_mode)
                        max_tok = int(body.get("max_tokens", 256))
                        temp = float(body.get("temperature", 0.7))

                        t0 = time.time()
                        with torch.no_grad():
                            out_ids = state.model.generate(
                                **inputs,
                                max_new_tokens=max_tok,
                                past_key_values=cache,
                                do_sample=(temp > 0.0),
                                temperature=max(temp, 1e-4) if temp > 0.0 else None,
                            )
                        elapsed = max(time.time() - t0, 1e-5)
                        gen_only = out_ids[0, inputs["input_ids"].shape:]
                        reply = state.tokenizer.decode(gen_only, skip_special_tokens=True)
                        state.history.append({"role": "assistant", "content": reply})
                        speed = f"{len(gen_only) / elapsed:.1f} ток/с"
                    self._send_json(200, {"reply": reply, "speed": speed})
                else:
                    self._send_json(404, {"error": "Not found"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def log_message(self, format, *args):
            pass

    return VCoreWebHandler


def run_web_server(host: str = "127.0.0.1", port: int = 7860, initial_model: Optional[str] = None):
    state = WebStudioState()
    print(f"[HW-DISPATCH] {state.hw.summary_banner()}")
    if initial_model:
        load_target = initial_model
        if os.path.isfile(initial_model) and initial_model.endswith(".vcore"):
            state._tmp_bundle, state.certificate_info = unpack_vcore_bundle_to_temp(initial_model)
            load_target = state._tmp_bundle.name
        dtype = torch.float16 if state.device in ("cuda", "mps") else torch.float32
        state.tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=True)
        state.model = VCoreModelForCausalLM.from_vcore_pretrained(load_target, device=state.device, dtype=dtype)
        state.model_path = initial_model

    server = ThreadingHTTPServer((host, port), create_handler(state))
    print(f"[V-CORE Web Studio] Сервер запущен: http://{host}:{port}")
    server.serve_forever()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Interactive Wave Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Хост (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Порт (по умолчанию 7860)")
    parser.add_argument("--model", "-m", default=None, help="Путь к модели или .vcore для автозагрузки")
    args = parser.parse_args()
    run_web_server(host=args.host, port=args.port, initial_model=args.model)


if __name__ == "__main__":
    main()
                            
