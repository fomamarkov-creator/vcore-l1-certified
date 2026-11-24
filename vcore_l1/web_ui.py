# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import torch
from .inference_layer import (
    VCoreDynamicCache,
    VCoreModelForCausalLM,
    _CPU_LIB,
    _CUDA_LIB,
)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>V-CORE (ℓ₁)-Certified • Neural Wave UI</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: #0B0D17; color: #F0F6FC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  canvas#waveCanvas {
    position: fixed; inset: 0; width: 100%; height: 100%;
    z-index: 0; touch-action: none;
  }
  .app {
    position: relative; z-index: 1;
    display: flex; flex-direction: column;
    max-width: 860px; height: 100%; margin: 0 auto;
    padding: 12px; pointer-events: none;
  }
  .topbar, .chat-card, .input-bar {
    pointer-events: auto;
    background: rgba(13, 17, 23, 0.62);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px; margin-bottom: 10px; gap: 8px; flex-wrap: wrap;
  }
  .brand { font-weight: 700; font-size: 15px; letter-spacing: 0.3px; }
  .badge {
    font-size: 12px; padding: 4px 10px; border-radius: 999px;
    background: rgba(63, 185, 80, 0.18); color: #3FB950;
    border: 1px solid rgba(63, 185, 80, 0.35);
  }
  .model-bar { display: flex; gap: 6px; align-items: center; }
  .model-bar input {
    background: rgba(9, 12, 16, 0.75); color: #fff;
    border: 1px solid rgba(255,255,255,0.16); border-radius: 10px;
    padding: 6px 10px; font-size: 13px; width: 190px;
  }
  .btn {
    background: rgba(88, 166, 255, 0.22); color: #79C0FF;
    border: 1px solid rgba(88, 166, 255, 0.4); border-radius: 10px;
    padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer;
  }
  .btn:hover { background: rgba(88, 166, 255, 0.35); }
  .chat-card {
    flex: 1; overflow-y: auto; padding: 16px; margin-bottom: 10px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 82%; padding: 12px 16px; border-radius: 16px;
    line-height: 1.5; font-size: 14px; white-space: pre-wrap;
  }
  .msg.user {
    align-self: flex-end;
    background: rgba(31, 111, 235, 0.45);
    border: 1px solid rgba(88, 166, 255, 0.35);
  }
  .msg.bot {
    align-self: flex-start;
    background: rgba(22, 27, 34, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  .meta { font-size: 11px; color: #8B949E; margin-top: 6px; }
  .input-bar {
    display: flex; gap: 8px; padding: 10px; align-items: center;
  }
  .input-bar input {
    flex: 1; background: transparent; border: none; outline: none;
    color: #fff; font-size: 15px; padding: 6px 8px;
  }
  .send-btn {
    background: #238636; color: #fff; border: none; border-radius: 12px;
    padding: 10px 18px; font-weight: 700; font-size: 14px; cursor: pointer;
  }
  .send-btn:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<canvas id="waveCanvas"></canvas>
<div class="app">
  <div class="topbar">
    <div class="brand">V-CORE (ℓ₁) • Wave Studio</div>
    <div class="badge" id="hwBadge">Загрузка статуса...</div>
    <div class="model-bar">
      <input id="modelPath" placeholder="./qwen15_vcore" />
      <button class="btn" onclick="loadModel()">Загрузить</button>
    </div>
  </div>
  <div class="chat-card" id="chatBox">
    <div class="msg bot">Коснитесь экрана в любом месте — по частицам пойдёт волновая интерференция. Укажите путь к сжатой папке модели сверху и нажмите «Загрузить».</div>
  </div>
  <div class="input-bar">
    <input id="promptInput" placeholder="Задайте вопрос нейросети..." onkeydown="if(event.key==='Enter')sendMsg()" />
    <button class="send-btn" id="sendBtn" onclick="sendMsg()">Пуск ➤</button>
  </div>
</div>

<script>
const cv = document.getElementById('waveCanvas');
const ctx = cv.getContext('2d');
let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
const ripples = [];
let tGlobal = 0;

function resize() {
  W = window.innerWidth; H = window.innerHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);
resize();

function spawnRipple(x, y, amp = 1.0) {
  ripples.push({ x, y, t: tGlobal, amp });
  if (ripples.length > 14) ripples.shift();
}

let lastMove = 0;
window.addEventListener('pointerdown', e => spawnRipple(e.clientX, e.clientY, 1.35));
window.addEventListener('pointermove', e => {
  if (e.buttons > 0 || e.pointerType === 'touch') {
    const now = performance.now();
    if (now - lastMove > 55) { spawnRipple(e.clientX, e.clientY, 0.75); lastMove = now; }
  }
});

setInterval(() => {
  if (ripples.length < 3) spawnRipple(W * (0.2 + 0.6 * Math.random()), H * (0.2 + 0.6 * Math.random()), 0.6);
}, 3800);

function drawBackground(t) {
  const base = ctx.createLinearGradient(0, 0, W, H);
  base.addColorStop(0, '#161338');
  base.addColorStop(0.5, '#3B185F');
  base.addColorStop(1, '#1A103C');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, H);

  const blobs = [
    { x: W * (0.25 + 0.15 * Math.sin(t * 0.4)), y: H * (0.25 + 0.12 * Math.cos(t * 0.3)), r: Math.max(W, H) * 0.55, c: 'rgba(56, 189, 248, 0.28)' },
    { x: W * (0.78 + 0.14 * Math.cos(t * 0.35)), y: H * (0.35 + 0.16 * Math.sin(t * 0.45)), r: Math.max(W, H) * 0.52, c: 'rgba(236, 72, 153, 0.32)' },
    { x: W * (0.55 + 0.18 * Math.sin(t * 0.28)), y: H * (0.78 + 0.12 * Math.cos(t * 0.38)), r: Math.max(W, H) * 0.58, c: 'rgba(251, 146, 60, 0.30)' }
  ];
  ctx.globalCompositeOperation = 'screen';
  for (const b of blobs) {
    const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
    g.addColorStop(0, b.c); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  }
  ctx.globalCompositeOperation = 'source-over';
}

function frame() {
  tGlobal += 0.032;
  drawBackground(tGlobal);
  for (let i = ripples.length - 1; i >= 0; i--) {
    if (tGlobal - ripples[i].t > 5.5) ripples.splice(i, 1);
  }
  const spacing = 26;
  const cols = Math.ceil(W / spacing) + 1;
  const rows = Math.ceil(H / spacing) + 1;

  for (let r = 0; r < rows; r++) {
    const y0 = r * spacing;
    for (let c = 0; c < cols; c++) {
      const x0 = c * spacing;
      let hSum = 0, dx = 0, dy = 0;
      for (let k = 0; k < ripples.length; k++) {
        const rp = ripples[k];
        const age = tGlobal - rp.t;
        const vx = x0 - rp.x, vy = y0 - rp.y;
        const dist = Math.hypot(vx, vy) + 0.001;
        const band = dist - age * 145.0;
        if (Math.abs(band) < 130.0) {
          const env = Math.exp(-(band * band) / 2600.0) * Math.exp(-age * 0.55) * rp.amp;
          const wave = Math.sin(band * 0.14) * env;
          hSum += wave;
          dx += (vx / dist) * wave * 5.5;
          dy += (vy / dist) * wave * 5.5;
        }
      }
      const energy = Math.min(1.5, Math.max(-0.6, hSum));
      const radius = Math.max(0.9, 1.45 + energy * 1.45);
      const alpha = Math.min(0.95, Math.max(0.14, 0.24 + energy * 0.45));
      ctx.beginPath();
      ctx.arc(x0 + dx, y0 + dy, radius, 0, Math.PI * 2);
      ctx.fillStyle = energy > 0.25 ? `rgba(255, 245, 230, ${alpha})` : `rgba(190, 225, 255, ${alpha})`;
      ctx.fill();
    }
  }
  requestAnimationFrame(frame);
}
frame();

function addMsg(text, who, meta = '') {
  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta'; m.textContent = meta;
    div.appendChild(m);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function refreshStatus() {
  const r = await fetch('/api/status'); const d = await r.json();
  document.getElementById('hwBadge').textContent = d.backend + (d.model ? ' • ' + d.model : '');
  if (d.model_path) document.getElementById('modelPath').value = d.model_path;
}
refreshStatus();

async function loadModel() {
  const p = document.getElementById('modelPath').value.trim();
  if (!p) return;
  const note = addMsg('Загрузка модели ' + p + '...', 'bot');
  spawnRipple(W * 0.5, H * 0.3, 1.4);
  const r = await fetch('/api/load', { method: 'POST', body: JSON.stringify({ path: p }) });
  const d = await r.json();
  note.textContent = d.ok ? 'Модель подключена: ' + d.model + ' (' + d.load_sec + ' с)' : 'Ошибка: ' + d.error;
  refreshStatus();
}

async function sendMsg() {
  const inp = document.getElementById('promptInput');
  const btn = document.getElementById('sendBtn');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = ''; btn.disabled = true;
  addMsg(txt, 'user');
  spawnRipple(W * 0.5, H * 0.5, 1.2);
  const botDiv = addMsg('Думаю...', 'bot');
  try {
    const r = await fetch('/api/chat', { method: 'POST', body: JSON.stringify({ prompt: txt }) });
    const d = await r.json();
    spawnRipple(W * 0.5, H * 0.5, 1.5);
    botDiv.textContent = d.reply || d.error;
    if (d.meta) {
      const m = document.createElement('div'); m.className = 'meta'; m.textContent = d.meta;
      botDiv.appendChild(m);
    }
  } catch (e) { botDiv.textContent = 'Ошибка соединения: ' + e; }
  btn.disabled = false;
}
</script>
</body>
</html>
"""
class VCoreWebState:
    def __init__(self, initial_model: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.model_path = initial_model or ""
        self.model_name = ""
        self.history: List[Dict[str, str]] = []

        if torch.cuda.is_available() and _CUDA_LIB is not None:
            self.backend = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif _CPU_LIB is not None:
            self.backend = "Native AVX2 CPU"
        else:
            self.backend = f"PyTorch ({self.device.upper()})"

        if self.model_path and os.path.exists(self.model_path):
            self.load(self.model_path)

    def load(self, path: str) -> Dict[str, Any]:
        from transformers import AutoTokenizer

        if not os.path.exists(path):
            return {"ok": False, "error": f"Путь не найден: {path}"}
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = VCoreModelForCausalLM.from_vcore_pretrained(
            path, device=self.device, dtype=self.dtype
        )
        dt = time.perf_counter() - t0
        self.model_path = path
        self.model_name = os.path.basename(os.path.abspath(path))
        self.history.clear()
        return {"ok": True, "model": self.model_name, "load_sec": round(dt, 2)}

    def chat(self, prompt: str, max_new_tokens: int = 192) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            return {"error": "Сначала загрузите сжатую модель сверху."}

        self.history.append({"role": "user", "content": prompt})
        msgs = [{"role": "system", "content": "You are V-CORE, a helpful AI assistant."}] + self.history[-6:]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text_in = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text_in = prompt

        inputs = self.tokenizer(text_in, return_tensors="pt").to(self.device)
        in_len = inputs["input_ids"].shape[1]
        kv = VCoreDynamicCache(group_size=32, bits=4)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                past_key_values=kv,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        dt = max(time.perf_counter() - t0, 1e-6)
        gen_ids = out_ids[0][in_len:]
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": reply})

        n_tok = int(gen_ids.numel())
        return {
            "reply": reply,
            "meta": f"{n_tok} токенов • {n_tok / dt:.1f} tok/s • {dt:.2f} с ({self.backend})",
        }


def run_web_server(host: str = "0.0.0.0", port: int = 8080, initial_model: Optional[str] = None):
    state = VCoreWebState(initial_model=initial_model)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            pass

        def _send_json(self, payload: Dict[str, Any], code: int = 200):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/api/status":
                self._send_json({
                    "backend": state.backend,
                    "model": state.model_name,
                    "model_path": state.model_path,
                })
            else:
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            if self.path == "/api/load":
                self._send_json(state.load(data.get("path", "")))
            elif self.path == "/api/chat":
                self._send_json(state.chat(data.get("prompt", "")))
            else:
                self._send_json({"error": "Not found"}, code=404)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[V-CORE Wave UI] Сервер запущен: http://localhost:{port} ({state.backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[V-CORE Wave UI] Остановлен.")


def main():
    parser = argparse.ArgumentParser(description="V-CORE Wave Particle Gradient Web UI")
    parser.add_argument("--model", "-m", default=None, help="Путь к сжатой модели V-CORE")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_web_server(host=args.host, port=args.port, initial_model=args.model)


if __name__ == "__main__":
    main()
      
