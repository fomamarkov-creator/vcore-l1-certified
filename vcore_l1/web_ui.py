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
from .inference_layer import VCoreDynamicCache, VCoreLinear, VCoreModelForCausalLM
from .resonance_scanner import scan_model_tomography, surgical_unlearn_concept
from .safetensors_quantizer import (
    repack_hf_hub_model,
    repack_safetensors_layerwise,
    repack_sharded_model,
)
from .vcore_bundle import unpack_vcore_bundle_to_temp


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V-CORE L1-Certified | Functional Brain Tomography Studio</title>
<style>
  :root {
    --bg: #04060b; --panel: #0a0f1a; --border: #1e293b;
    --accent: #38bdf8; --accent2: #a855f7; --gold: #fbbf24; --text: #e2e8f0; --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    padding: 9px 18px; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 15px; }
  .badge { background: rgba(251,191,36,0.14); color: #fde68a; padding: 3px 10px; border-radius: 99px; font-size: 11px; border: 1px solid rgba(251,191,36,0.35); }
  .hw-pill { background: rgba(56,189,248,0.14); color: #7dd3fc; padding: 3px 10px; border-radius: 99px; font-size: 11px; border: 1px solid rgba(56,189,248,0.35); }
  .tomo-stage {
    position: relative; width: 100%; height: 235px;
    background: radial-gradient(ellipse at 50% 52%, #0e1326 0%, #04060b 78%);
    border-bottom: 1px solid var(--border); overflow: hidden;
  }
  #brain-canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
  .hud-top {
    position: absolute; top: 10px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center;
    pointer-events: none; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; color: #cbd5e1;
  }
  .hud-controls {
    position: absolute; top: 34px; left: 16px; display: flex; gap: 6px; z-index: 5;
  }
  .mode-btn {
    padding: 4px 10px; font-size: 11px; border-radius: 6px; background: rgba(15,23,42,0.85);
    color: var(--muted); border: 1px solid var(--border); cursor: pointer; font-weight: 600;
  }
  .mode-btn.active { background: rgba(56,189,248,0.2); color: #e0f2fe; border-color: var(--accent); }
  .hud-legend {
    position: absolute; bottom: 8px; left: 16px; right: 16px; display: flex; justify-content: space-between; flex-wrap: wrap;
    pointer-events: none; font-size: 11px; color: var(--muted);
  }
  .legend-items { display: flex; gap: 14px; background: rgba(4,6,11,0.72); padding: 3px 10px; border-radius: 6px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  #node-tooltip {
    position: absolute; display: none; pointer-events: none; z-index: 10;
    background: rgba(10,15,26,0.95); border: 1px solid var(--accent); border-radius: 8px;
    padding: 7px 10px; font-size: 11px; font-family: monospace; color: #f8fafc; box-shadow: 0 4px 18px rgba(0,0,0,0.8);
  }
  .main { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 320px; background: var(--panel); border-right: 1px solid var(--border); padding: 12px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
  .card { background: rgba(15,23,42,0.75); border: 1px solid var(--border); border-radius: 9px; padding: 10px; display: flex; flex-direction: column; gap: 7px; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px; color: var(--muted); }
  input, select {
    width: 100%; padding: 7px 9px; border-radius: 6px; border: 1px solid var(--border);
    background: #060912; color: var(--text); font-size: 12.5px;
  }
  button.primary {
    padding: 8px 12px; border-radius: 6px; border: none; font-weight: 600; font-size: 12.5px; cursor: pointer;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #04060b;
  }
  button.primary:disabled { opacity: 0.45; cursor: not-allowed; }
  button.danger { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); padding: 7px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }
  button.secondary { background: #1e293b; color: var(--text); padding: 7px 10px; border-radius: 6px; border: none; font-size: 12px; cursor: pointer; }
  .chat-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #messages { flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 82%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: 13.5px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #1d4ed8; color: #fff; border-bottom-right-radius: 2px; }
  .msg.bot { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); border-bottom-left-radius: 2px; }
  .input-bar { padding: 11px 16px; background: var(--panel); border-top: 1px solid var(--border); display: flex; gap: 10px; }
  .input-bar input { flex: 1; font-size: 13.5px; padding: 9px 13px; }
  .status-line { font-size: 11.5px; color: var(--muted); word-break: break-word; }
  @media (max-width: 768px) { .main { flex-direction: column; } .sidebar { width: 100%; max-height: 44vh; border-right: none; border-bottom: 1px solid var(--border); } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>V-CORE L1-CERTIFIED</span>
    <span class="badge">fMRI / ЭМРТ Нейро-Томограф</span>
    <span class="hw-pill" id="hw-banner">Сканирование железа...</span>
  </div>
  <div class="status-line" id="status-text">Модель не загружена</div>
</header>

<div class="tomo-stage">
  <canvas id="brain-canvas"></canvas>
  <div id="node-tooltip"></div>
  <div class="hud-top">
    <span id="hud-left">ЭМРТ-МОНИТОР: ОЖИДАНИЕ СЖАТИЯ ИЛИ ЗАГРУЗКИ МОДЕЛИ</span>
    <span id="hud-right">НЕЙРОНОВ: 0 | ФАЗА: ПОКОЙ</span>
  </div>
  <div class="hud-controls">
    <button class="mode-btn active" id="btn-view-brain" onclick="setViewMode('brain')">🧠 Проекция мозга (fMRI)</button>
    <button class="mode-btn" id="btn-view-synapse" onclick="setViewMode('synapse')">🧬 Дендриты (Микро)</button>
  </div>
  <div class="hud-legend">
    <div class="legend-items">
      <span><i class="dot" style="background:#38bdf8"></i>Ввод (Височная)</span>
      <span><i class="dot" style="background:#c084fc"></i>Внимание (Теменная)</span>
      <span><i class="dot" style="background:#fbbf24"></i>Логика MLP (Лобная)</span>
      <span><i class="dot" style="background:#34d399"></i>Выход (Зона Брока)</span>
    </div>
    <div class="legend-items">
      <span>Сигнал входа ➔ Обработка ➔ Разряд ответа ⚡</span>
    </div>
  </div>
</div>

<div class="main">
  <div class="sidebar">
    <div class="card">
      <h3>1. Сжатие и нейрогенез (SSE)</h3>
      <input id="pack-src" placeholder="HF Repo (Qwen/Qwen2.5-1.5B) или путь">
      <input id="pack-dst" placeholder="Выходная папка (./qwen_vcore)" value="./qwen_vcore">
      <div style="display:flex; gap:6px;">
        <input id="pack-vram" type="number" step="0.5" placeholder="Бюджет ГБ">
        <input id="pack-calib" type="number" value="0" placeholder="Калибр.">
      </div>
      <label style="font-size:11.5px; color:var(--muted); display:flex; align-items:center; gap:6px;">
        <input type="checkbox" id="pack-tomo" checked style="width:auto;"> Авто-режим по ЭМРТ-томографии
      </label>
      <button class="primary" id="btn-pack" onclick="runPackSSE()">Сжать и вырастить карту мозга</button>
      <div class="status-line" id="pack-log">Готов к упаковке</div>
    </div>

    <div class="card">
      <h3>2. Загрузка модели / .vcore</h3>
      <input id="load-dir" placeholder="Путь к папке или model.vcore" value="./qwen_vcore">
      <div style="display:flex; gap:6px;">
        <select id="load-dev">
          <option value="auto">Device: Авто</option>
          <option value="cuda">CUDA (GPU)</option>
          <option value="mps">Apple MPS</option>
          <option value="cpu">CPU (AVX2)</option>
        </select>
        <select id="load-int">
          <option value="false">Float / AVX2</option>
          <option value="true">Integer-Only Q14</option>
        </select>
      </div>
      <button class="primary" onclick="loadModel()">Загрузить и просканировать</button>
    </div>

    <div class="card">
      <h3>3. Хирургия памяти &amp; Чат</h3>
      <div style="display:flex; gap:6px;">
        <input id="unlearn-txt" placeholder="Тема для вырезания (/unlearn)">
        <button class="danger" onclick="runUnlearn()">Вырезать</button>
      </div>
      <div style="display:flex; gap:6px;">
        <input id="gen-tokens" type="number" value="256" title="Max tokens">
        <input id="gen-temp" type="number" step="0.1" value="0.7" title="Temperature">
        <select id="kv-mode" style="width:110px;">
          <option value="spherical">KV: E8</option>
          <option value="int4">KV: INT4</option>
          <option value="ternary">KV: 2-bit</option>
        </select>
      </div>
      <button class="secondary" onclick="clearChat()">Очистить историю диалога</button>
    </div>
  </div>

  <div class="chat-wrap">
    <div id="messages">
      <div class="msg bot">ЭМРТ-томограф готов. Загрузи или сожми модель слева, затем отправь сообщение: ты увидишь, как одиночный импульс промпта входит в сенсорную зону мозга, зажигает активные слои внимания и MLP, а ответ модели выходит потоком разрядов из зоны Брока.</div>
    </div>
    <div class="input-bar">
      <input id="prompt-input" placeholder="Введите промпт для запуска импульса в мозг модели..." onkeydown="if(event.key==='Enter') sendMsg()">
      <button class="primary" id="btn-send" onclick="sendMsg()">Отправить</button>
    </div>
  </div>
</div>
<script>
const canvas = document.getElementById('brain-canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('node-tooltip');
let dpr = window.devicePixelRatio || 1;
let W = 0, H = 0, tick = 0;
let viewMode = 'brain'; // 'brain' | 'synapse'

const ZONES = {
  input_temporal: { cx: -0.22, cy: 0.14, rx: 0.11, ry: 0.12, hue: 198, title: 'Сенсорная зона (Ввод)' },
  attn_parietal:  { cx: -0.02, cy: -0.20, rx: 0.18, ry: 0.11, hue: 282, title: 'Теменная доля (Attention)' },
  mlp_frontal:    { cx: 0.21,  cy: -0.05, rx: 0.15, ry: 0.16, hue: 38,  title: 'Лобная доля (MLP / Логика)' },
  output_broca:   { cx: 0.20,  cy: 0.22,  rx: 0.10, ry: 0.09, hue: 156, title: 'Зона Брока (Выход токенов)' }
};

let brainNodes = [];
let tracts = [];
let signals = [];

function setViewMode(mode) {
  viewMode = mode;
  document.getElementById('btn-view-brain').classList.toggle('active', mode === 'brain');
  document.getElementById('btn-view-synapse').classList.toggle('active', mode === 'synapse');
}

function resizeCanvas() {
  dpr = window.devicePixelRatio || 1;
  W = canvas.width = canvas.offsetWidth * dpr;
  H = canvas.height = canvas.offsetHeight * dpr;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function classifyLayerZone(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('embed_tokens') || n.includes('layers.0.') || n.includes('layers.1.')) return 'input_temporal';
  if (n.includes('lm_head') || n.includes('model.norm')) return 'output_broca';
  if (n.includes('self_attn') || n.includes('q_proj') || n.includes('k_proj') || n.includes('v_proj') || n.includes('o_proj')) {
    return 'attn_parietal';
  }
  return 'mlp_frontal';
}

function extractLayerDepth(name) {
  const m = (name || '').match(/layers\.(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

function addBrainNode(layerInfo) {
  const name = layerInfo.name || `layer_${brainNodes.length}`;
  const existing = brainNodes.find(n => n.name === name);
  if (existing) {
    existing.mode = layerInfo.mode || existing.mode;
    existing.impedance = layerInfo.impedance || existing.impedance;
    existing.activity = 1.0;
    return existing;
  }

  const zoneKey = classifyLayerZone(name);
  const z = ZONES[zoneKey];
  const idx = brainNodes.length;
  const layerDepth = extractLayerDepth(name);
  const shape = Array.isArray(layerInfo.shape) ? layerInfo.shape : [];
  const aspect = shape.length >= 2 ? (shape[0] / Math.max(shape[1], 1)) : 1.0;
  const peakRatio = layerInfo.peak || layerInfo.peak_ratio || 1.0;
  const impVal = layerInfo.impedance || 1.0;

  // Уникальная угловая и радиальная проекция на основе реальных параметров слоя модели
  const angle = (layerDepth * 0.6180339 * Math.PI * 2) + (aspect * 1.37) + (idx * 0.29);
  const rNorm = Math.min(0.92, 0.22 + ((Math.log1p(peakRatio) + (impVal % 17) / 17.0) * 0.24));
  const bx = z.cx + Math.cos(angle) * z.rx * rNorm;
  const by = z.cy + Math.sin(angle) * z.ry * rNorm;

  // Координаты для микро-режима дендритов
  const zoneNodes = brainNodes.filter(n => n.zone === zoneKey);
  const parent = zoneNodes.length > 0 ? zoneNodes[Math.floor((zoneNodes.length - 1) / 2)] : null;
  let sx = 0, sy = 0, parentId = -1;
  if (!parent) {
    const zOrder = ['input_temporal', 'attn_parietal', 'mlp_frontal', 'output_broca'].indexOf(zoneKey);
    sx = -0.32 + zOrder * 0.21;
    sy = 0.34;
  } else {
    parentId = parent.id;
    const branchAngle = -Math.PI / 2 + ((layerDepth % 7) - 3) * 0.26 + (aspect - 1.0) * 0.2;
    const branchLen = 0.13 * Math.pow(0.87, Math.min(4, Math.floor(zoneNodes.length / 4)));
    sx = Math.max(-0.44, Math.min(0.44, parent.sx + Math.cos(branchAngle) * branchLen));
    sy = Math.max(-0.38, Math.min(0.36, parent.sy + Math.sin(branchAngle) * branchLen));
  }

  let hue = z.hue;
  const m = (layerInfo.mode || '').toLowerCase();
  if (m.includes('e8')) hue = 174;
  else if (m.includes('fp16') || impVal > 150) hue = 42;

  const node = {
    id: idx,
    name: name,
    zone: zoneKey,
    mode: layerInfo.mode || '1.58b',
    impedance: impVal,
    shape: shape,
    activity: 1.0,
    growth: 0.05,
    bx: bx,
    by: by,
    sx: sx,
    sy: sy,
    parentId: parentId,
    hue: hue
  };
  brainNodes.push(node);

  if (brainNodes.length > 1) {
    const prevSame = zoneNodes[zoneNodes.length - 1];
    tracts.push({ from: prevSame ? prevSame.id : brainNodes.length - 2, to: node.id });
  }

  document.getElementById('hud-right').textContent = `НЕЙРОНОВ: ${brainNodes.length} | ФАЗА: НЕЙРОГЕНЕЗ`;
  return node;
}

function getNodeScreenPos(node) {
  const cx = W * 0.5;
  const cy = H * 0.52;
  const scale = Math.min(W * 0.85, H * 1.15);
  if (viewMode === 'brain') {
    return { x: cx + node.bx * scale, y: cy + node.by * scale };
  }
  return { x: cx + node.sx * W * 0.9, y: cy + node.sy * H * 0.85 };
}

function firePromptSignal() {
  const inputOrigin = { x: W * 0.08, y: H * 0.85 };
  const targetNodes = brainNodes.filter(n => n.zone === 'input_temporal');
  const target = targetNodes.length > 0 ? targetNodes[0] : brainNodes[0];

  if (!target) {
    const dummyPos = { x: W * 0.5 - Math.min(W * 0.85, H * 1.15) * 0.22, y: H * 0.52 + Math.min(W * 0.85, H * 1.15) * 0.14 };
    signals.push({
      x1: inputOrigin.x, y1: inputOrigin.y, x2: dummyPos.x, y2: dummyPos.y,
      progress: 0, speed: 0.045, hue: 198, size: 5.5 * dpr, chainToZone: 'attn_parietal'
    });
    return;
  }

  const pos = getNodeScreenPos(target);
  target.activity = 1.2;
  signals.push({
    x1: inputOrigin.x, y1: inputOrigin.y, x2: pos.x, y2: pos.y,
    progress: 0, speed: 0.055, hue: 198, size: 5.5 * dpr, chainToZone: 'attn_parietal'
  });
}

function fireResponsePulse(hotLayerNames, energy = 0.8) {
  const outputExit = { x: W * 0.92, y: H * 0.85 };
  if (hotLayerNames && hotLayerNames.length > 0) {
    hotLayerNames.forEach(hName => {
      const n = brainNodes.find(item => item.name.includes(hName) || hName.includes(item.name));
      if (n) {
        n.activity = Math.min(1.5, 0.75 + energy);
        const brocaNodes = brainNodes.filter(b => b.zone === 'output_broca');
        const broca = brocaNodes[0] || n;
        const p1 = getNodeScreenPos(n);
        const p2 = getNodeScreenPos(broca);
        signals.push({
          x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y,
          progress: 0, speed: 0.085, hue: 42, size: 3.6 * dpr, chainToZone: 'exit'
        });
      }
    });
  } else if (brainNodes.length > 0) {
    const frontal = brainNodes.filter(n => n.zone === 'mlp_frontal');
    const src = frontal[Math.floor(Math.random() * frontal.length)] || brainNodes[0];
    src.activity = 1.1;
    const p1 = getNodeScreenPos(src);
    signals.push({
      x1: p1.x, y1: p1.y, x2: outputExit.x, y2: outputExit.y,
      progress: 0, speed: 0.075, hue: 156, size: 4.0 * dpr, chainToZone: null
    });
  }
}
function drawBrainSilhouette() {
  const cx = W * 0.5;
  const cy = H * 0.52;
  const s = Math.min(W * 0.85, H * 1.15);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(s, s);

  ctx.beginPath();
  ctx.moveTo(-0.34, 0.05);
  ctx.bezierCurveTo(-0.38, -0.18, -0.18, -0.34, 0.06, -0.32);
  ctx.bezierCurveTo(0.28, -0.30, 0.40, -0.14, 0.37, 0.06);
  ctx.bezierCurveTo(0.35, 0.20, 0.22, 0.26, 0.08, 0.22);
  ctx.bezierCurveTo(-0.02, 0.28, -0.15, 0.32, -0.25, 0.26);
  ctx.bezierCurveTo(-0.33, 0.22, -0.36, 0.14, -0.34, 0.05);
  ctx.closePath();

  const grad = ctx.createRadialGradient(0.05, -0.04, 0.04, 0.0, 0.0, 0.42);
  grad.addColorStop(0, 'rgba(56, 189, 248, 0.09)');
  grad.addColorStop(0.6, 'rgba(139, 92, 246, 0.05)');
  grad.addColorStop(1, 'rgba(15, 23, 42, 0.02)');
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.lineWidth = 1.6 / s * dpr;
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(-0.22, -0.08);
  ctx.bezierCurveTo(-0.10, -0.22, 0.12, -0.20, 0.24, -0.06);
  ctx.moveTo(-0.15, 0.06);
  ctx.bezierCurveTo(0.0, -0.04, 0.16, 0.04, 0.26, 0.12);
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
  ctx.lineWidth = 1.1 / s * dpr;
  ctx.stroke();

  ctx.restore();
}

function renderFrame() {
  tick += 0.035;
  ctx.fillStyle = 'rgba(4, 6, 11, 0.36)';
  ctx.fillRect(0, 0, W, H);

  if (viewMode === 'brain') drawBrainSilhouette();

  for (let i = 0; i < tracts.length; i++) {
    const tr = tracts[i];
    const n1 = brainNodes[tr.from];
    const n2 = brainNodes[tr.to];
    if (!n1 || !n2) continue;

    const p1 = getNodeScreenPos(n1);
    const p2 = getNodeScreenPos(n2);
    const curX = p1.x + (p2.x - p1.x) * n2.growth;
    const curY = p1.y + (p2.y - p1.y) * n2.growth;

    const actBoost = Math.max(n1.activity, n2.activity);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    const midX = (p1.x + curX) * 0.5 + Math.sin(i + tick * 0.5) * 4 * dpr;
    const midY = (p1.y + curY) * 0.5 - 5 * dpr;
    ctx.quadraticCurveTo(midX, midY, curX, curY);

    ctx.strokeStyle = `hsla(${n2.hue}, 88%, 62%, ${0.14 + actBoost * 0.38})`;
    ctx.lineWidth = (viewMode === 'synapse' ? 1.8 : 1.0) * dpr;
    ctx.stroke();
  }

  for (let i = 0; i < brainNodes.length; i++) {
    const n = brainNodes[i];
    if (n.growth < 1.0) n.growth = Math.min(1.0, n.growth + 0.05);
    if (n.activity > 0.12) n.activity = Math.max(0.12, n.activity - 0.018);

    const pos = getNodeScreenPos(n);
    const baseR = (viewMode === 'synapse' ? 3.6 : 2.8) * dpr;
    const r = baseR * (0.85 + n.activity * 0.85);

    if (n.activity > 0.25) {
      const glowR = r * (3.2 + n.activity * 2.2);
      const g = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowR);
      g.addColorStop(0, `hsla(${n.hue}, 100%, 68%, ${Math.min(0.65, n.activity * 0.45)})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, glowR, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fillStyle = n.activity > 0.75 ? '#ffffff' : `hsla(${n.hue}, 95%, 72%, ${0.55 + n.activity * 0.45})`;
    ctx.fill();
  }

  for (let i = signals.length - 1; i >= 0; i--) {
    const s = signals[i];
    s.progress += s.speed;

    if (s.progress >= 1.0) {
      if (s.chainToZone === 'attn_parietal') {
        const attnNodes = brainNodes.filter(n => n.zone === 'attn_parietal');
        attnNodes.slice(0, 4).forEach(an => {
          an.activity = 1.2;
          const pA = getNodeScreenPos(an);
          const frontal = brainNodes.filter(n => n.zone === 'mlp_frontal');
          const fn = frontal[Math.floor(Math.random() * frontal.length)];
          if (fn) {
            const pF = getNodeScreenPos(fn);
            signals.push({
              x1: pA.x, y1: pA.y, x2: pF.x, y2: pF.y,
              progress: 0, speed: 0.065, hue: 282, size: 3.8 * dpr, chainToZone: null
            });
          }
        });
      } else if (s.chainToZone === 'exit') {
        const outputExit = { x: W * 0.92, y: H * 0.85 };
        signals.push({
          x1: s.x2, y1: s.y2, x2: outputExit.x, y2: outputExit.y,
          progress: 0, speed: 0.09, hue: 156, size: 4.2 * dpr, chainToZone: null
        });
      }
      signals.splice(i, 1);
      continue;
    }

    const t = s.progress;
    const ix = s.x1 + (s.x2 - s.x1) * t;
    const iy = s.y1 + (s.y2 - s.y1) * t - Math.sin(t * Math.PI) * 14 * dpr;

    ctx.beginPath();
    ctx.arc(ix, iy, s.size, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = `hsl(${s.hue}, 100%, 65%)`;
    ctx.shadowBlur = 14 * dpr;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  requestAnimationFrame(renderFrame);
}
renderFrame();

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * dpr;
  const my = (e.clientY - rect.top) * dpr;
  let found = null;
  let minDist = 14 * dpr;

  for (const n of brainNodes) {
    const pos = getNodeScreenPos(n);
    const dist = Math.hypot(pos.x - mx, pos.y - my);
    if (dist < minDist) {
      minDist = dist;
      found = n;
    }
  }

  if (found) {
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(rect.width - 250, e.clientX - rect.left + 12) + 'px';
    tooltip.style.top = Math.max(8, e.clientY - rect.top - 52) + 'px';
    const shapeStr = found.shape && found.shape.length ? found.shape.join('×') : 'N/A';
    tooltip.innerHTML = `<b>${found.name}</b><br>Зона: ${ZONES[found.zone].title}<br>Форма: [${shapeStr}] | ${found.mode} | Imp: ${found.impedance}`;
  } else {
    tooltip.style.display = 'none';
  }
});
async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.hw_banner) document.getElementById('hw-banner').textContent = d.hw_banner;
    if (d.recommended_budget && !document.getElementById('pack-vram').value) {
      document.getElementById('pack-vram').placeholder = `Бюджет: ${d.recommended_budget} ГБ`;
    }
    if (d.loaded) {
      let certTxt = d.cert_drift ? ` | Lean4 Drift <= ${d.cert_drift}` : '';
      document.getElementById('status-text').textContent = `Активна: ${d.model_path} (${d.device.toUpperCase()}${certTxt})`;
      if (brainNodes.length === 0) fetchTomographyMap();
    }
  } catch (e) {}
}
refreshStatus();

async function fetchTomographyMap() {
  try {
    const r = await fetch('/api/tomography');
    const d = await r.json();
    if (d.ready && d.layers && d.layers.length > 0) {
      brainNodes = [];
      tracts = [];
      d.layers.forEach((layer, idx) => {
        setTimeout(() => addBrainNode(layer), idx * 16);
      });
      document.getElementById('hud-left').textContent = `ЭМРТ-ТОМОГРАММА ПОСТРОЕНА | АКТИВНЫХ УЗЛОВ: ${d.layers.length}`;
      document.getElementById('hud-right').textContent = `НЕЙРОНОВ: ${d.layers.length} | ФАЗА: МОНИТОРИНГ`;
    }
  } catch (e) {}
}

async function runPackSSE() {
  const src = document.getElementById('pack-src').value.trim();
  const dst = document.getElementById('pack-dst').value.trim();
  const vram = document.getElementById('pack-vram').value;
  const calib = document.getElementById('pack-calib').value;
  const use_tomo = document.getElementById('pack-tomo').checked;
  if (!src || !dst) return alert('Укажите источник и выходную папку!');

  const btn = document.getElementById('btn-pack');
  btn.disabled = true;
  brainNodes = [];
  tracts = [];
  document.getElementById('pack-log').textContent = 'Запуск ЭМРТ-сканера и потокового сжатия...';

  try {
    const resp = await fetch('/api/pack_sse', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src, dst, vram, calib, use_tomo })
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();

      for (const p of parts) {
        const line = p.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === 'layer') {
          addBrainNode(evt.layer);
          const pct = Math.round((evt.layer.progress || 0) * 100);
          document.getElementById('hud-left').textContent = `НЕЙРОГЕНЕЗ [${pct}%]: ${evt.layer.name} (${evt.layer.mode})`;
          document.getElementById('pack-log').textContent = `[${pct}%] Сжат: ${evt.layer.name}`;
        } else if (evt.type === 'done') {
          document.getElementById('pack-log').textContent = evt.message;
          document.getElementById('hud-left').textContent = `КАРТА МОЗГА СФОРМИРОВАНА | ${evt.message}`;
          document.getElementById('hud-right').textContent = `НЕЙРОНОВ: ${brainNodes.length} | ФАЗА: ГОТОВ`;
          document.getElementById('load-dir').value = dst;
        } else if (evt.type === 'error') {
          document.getElementById('pack-log').textContent = 'Ошибка: ' + evt.error;
        }
      }
    }
  } catch (e) {
    document.getElementById('pack-log').textContent = 'Ошибка сети: ' + e;
  } finally {
    btn.disabled = false;
  }
}

async function loadModel() {
  const model_dir = document.getElementById('load-dir').value.trim();
  const device = document.getElementById('load-dev').value;
  const integer_only = document.getElementById('load-int').value === 'true';
  document.getElementById('status-text').textContent = 'Загрузка модели...';
  const r = await fetch('/api/load', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_dir, device, integer_only })
  });
  const d = await r.json();
  if (d.error) {
    document.getElementById('status-text').textContent = 'Ошибка: ' + d.error;
  } else {
    await refreshStatus();
    await fetchTomographyMap();
  }
}

async function runUnlearn() {
  const concept = document.getElementById('unlearn-txt').value.trim();
  if (!concept) return alert('Введите тему или фразу для вырезания!');
  document.getElementById('hud-left').textContent = `ЭМРТ-АБЛЯЦИЯ ПАТТЕРНА: "${concept}"...`;

  const r = await fetch('/api/unlearn', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ concept })
  });
  const d = await r.json();
  if (d.error) {
    alert('Ошибка: ' + d.error);
  } else {
    brainNodes.forEach(n => {
      if (n.name.includes('down_proj') || n.name.includes('o_proj')) {
        n.hue = 0;
        n.activity = 1.4;
      }
    });
    document.getElementById('hud-left').textContent =
      `ВЫРЕЗАНО ИЗ ${d.edited_layers} СЛОЁВ (Удалено L1-энергии: ${d.total_l1_energy_ablated})`;
  }
}

async function clearChat() {
  await fetch('/api/clear', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
}

async function sendMsg() {
  const inp = document.getElementById('prompt-input');
  const btn = document.getElementById('btn-send');
  const text = inp.value.trim();
  if (!text) return;

  inp.value = '';
  btn.disabled = true;

  const box = document.getElementById('messages');
  const uDiv = document.createElement('div'); uDiv.className = 'msg user'; uDiv.textContent = text; box.appendChild(uDiv);
  const bDiv = document.createElement('div'); bDiv.className = 'msg bot'; bDiv.textContent = ''; box.appendChild(bDiv);
  box.scrollTop = box.scrollHeight;

  firePromptSignal();
  document.getElementById('hud-left').textContent = 'ВХОДНОЙ ИМПУЛЬС ➔ СЕНСОРНАЯ КОРА ➔ ОБРАБОТКА ВНИМАНИЯ...';
  document.getElementById('hud-right').textContent = `НЕЙРОНОВ: ${brainNodes.length} | ФАЗА: ПРЯМОЙ ПРОХОД ⚡`;

  const max_tokens = parseInt(document.getElementById('gen-tokens').value || '256');
  const temperature = parseFloat(document.getElementById('gen-temp').value || '0.7');
  const kv_mode = document.getElementById('kv-mode').value;

  try {
    const resp = await fetch('/api/chat_sse', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, max_tokens, temperature, kv_mode })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();

      for (const p of parts) {
        const line = p.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));

        if (evt.type === 'token') {
          bDiv.textContent += evt.text;
          box.scrollTop = box.scrollHeight;
          fireResponsePulse(evt.hot_layers, evt.energy || 0.8);
          document.getElementById('hud-left').textContent =
            `ГЕНЕРАЦИЯ ОТВЕТА | АКТИВНЫЕ УЗЛЫ: ${(evt.hot_layers || []).slice(0, 2).join(', ') || 'MLP/FFN'}`;
        } else if (evt.type === 'done') {
          if (evt.speed) document.getElementById('status-text').textContent = `Скорость: ${evt.speed} | KV: ${kv_mode}`;
        } else if (evt.type === 'error') {
          bDiv.textContent = 'Ошибка: ' + evt.error;
        }
      }
    }
  } catch (e) {
    bDiv.textContent = 'Ошибка соединения: ' + e;
  } finally {
    btn.disabled = false;
    document.getElementById('hud-right').textContent = `НЕЙРОНОВ: ${brainNodes.length} | ФАЗА: ПОКОЙ`;
    box.scrollTop = box.scrollHeight;
  }
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

        def _send_sse(self, payload: Dict[str, Any]):
            msg = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(msg)
            self.wfile.flush()

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
            elif self.path == "/api/tomography":
                with state.lock:
                    if state.model is None:
                        self._send_json(200, {"ready": False, "layers": []})
                        return
                    tomo = scan_model_tomography(state.model)
                    layers_data = [
                        {
                            "name": name,
                            "shape": list(info.get("shape",)),
                            "peak": float(info.get("in_peak_ratio", 1.0)),
                            "impedance": float(info.get("mean_impedance", 1.0)),
                            "gain": float(info.get("hadamard_gain", 1.0)),
                            "mode": str(info.get("recommended_mode", "TERNARY_158BIT")),
                        }
                        for name, info in tomo.items()
                    ]
                self._send_json(200, {"ready": True, "layers": layers_data})
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            if self.path == "/api/pack_sse":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                src = body["src"]
                dst = body["dst"]
                vram = float(body["vram"]) if body.get("vram") else state.hw.recommended_vram_budget_gb
                calib = int(body.get("calib", 0))
                use_tomo = bool(body.get("use_tomo", True))
                sph_emb = state.hw.recommended_spherical_emb
                sse_lock = threading.Lock()

                def on_layer_packed(evt: Dict[str, Any]):
                    with sse_lock:
                        try:
                            self._send_sse({"type": "layer", "layer": evt})
                        except Exception:
                            pass

                try:
                    act_map, chol_map = None, None
                    if calib > 0:
                        act_map, chol_map = collect_activation_importance(src, num_samples=calib, collect_covariance=True)

                    kwargs = dict(
                        target_vram_gb=vram,
                        activation_map=act_map,
                        chol_map=chol_map,
                        spherical_emb=sph_emb,
                        use_tomography=use_tomo,
                        progress_callback=on_layer_packed,
                    )
                    if os.path.isfile(src):
                        st = repack_safetensors_layerwise(src, dst, **kwargs)
                    elif os.path.isdir(src):
                        st = repack_sharded_model(src, dst, **kwargs)
                    else:
                        st = repack_hf_hub_model(src, dst, **kwargs)

                    msg = f"Сжатие {st['compression_ratio']:.2f}x ({st['packed_size_mb']:.1f} МБ)"
                    with sse_lock:
                        self._send_sse({"type": "done", "message": msg})
                except Exception as e:
                    with sse_lock:
                        self._send_sse({"type": "error", "error": str(e)})
                return

            if self.path == "/api/chat_sse":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                with state.lock:
                    if state.model is None:
                        self._send_sse({"type": "error", "error": "Сначала загрузите модель!"})
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
                    streamer = TextIteratorStreamer(state.tokenizer, skip_prompt=True, skip_special_tokens=True)

                    live_energies: Dict[str, float] = {}
                    hooks = []

                    def _make_fmri_hook(l_name: str):
                        def _hook(_mod, _inp, out):
                            if isinstance(out, torch.Tensor):
                                live_energies[l_name] = float(out.detach().abs().mean().item())
                        return _hook

                    for m_name, mod in state.model.named_modules():
                        if isinstance(mod, (torch.nn.Linear, VCoreLinear)) and (
                            m_name.endswith("down_proj") or m_name.endswith("o_proj")
                        ):
                            hooks.append(mod.register_forward_hook(_make_fmri_hook(f"{m_name}.weight")))

                    gen_kwargs = dict(
                        **inputs,
                        max_new_tokens=max_tok,
                        past_key_values=cache,
                        streamer=streamer,
                        do_sample=(temp > 0.0),
                        temperature=max(temp, 1e-4) if temp > 0.0 else None,
                    )

                    t0 = time.time()
                    gen_thread = threading.Thread(target=state.model.generate, kwargs=gen_kwargs, daemon=True)
                    gen_thread.start()

                    full_reply = []
                    tok_count = 0
                    try:
                        for piece in streamer:
                            full_reply.append(piece)
                            tok_count += 1
                            sorted_hot = sorted(live_energies.items(), key=lambda x: x, reverse=True)
                            hot_names = [k for k, _ in sorted_hot[:3]]
                            max_e = sorted_hot if sorted_hot else 0.5
                            self._send_sse({
                                "type": "token",
                                "text": piece,
                                "hot_layers": hot_names,
                                "energy": min(1.5, round(max_e, 3)),
                            })
                    finally:
                        gen_thread.join()
                        for h in hooks:
                            h.remove()

                    elapsed = max(time.time() - t0, 1e-5)
                    state.history.append({"role": "assistant", "content": "".join(full_reply)})
                    self._send_sse({"type": "done", "speed": f"{tok_count / elapsed:.1f} ток/с"})
                return

            try:
                if self.path == "/api/load":
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

                elif self.path == "/api/unlearn":
                    with state.lock:
                        if state.model is None or state.tokenizer is None:
                            self._send_json(400, {"error": "Сначала загрузите модель!"})
                            return
                        res = surgical_unlearn_concept(
                            state.model, state.tokenizer, target_prompts=[body["concept"]], strength=0.85
                        )
                        state.history = []
                    self._send_json(200, res)

                elif self.path == "/api/clear":
                    with state.lock:
                        state.history = []
                    self._send_json(200, {"ok": True})
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
    parser = argparse.ArgumentParser(description="V-CORE L1-Certified Functional Brain Tomography Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Хост (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Порт (по умолчанию 7860)")
    parser.add_argument("--model", "-m", default=None, help="Путь к модели или .vcore для автозагрузки")
    args = parser.parse_args()
    run_web_server(host=args.host, port=args.port, initial_model=args.model)


if __name__ == "__main__":
    main()
                        
