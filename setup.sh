#!/bin/bash
# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

set -e
echo "--- V-CORE L1-CERTIFIED: НАЧАЛО СИНХРОНИЗАЦИИ ---"

# 1. Установка Python-зависимостей и пакета vcore_l1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps

# 2. Сборка связанного верифицированного ядра Lean 4 (если установлен lake)
if command -v lake &> /dev/null; then
    echo "[LEAN 4]: Сборка сертификатора vcore_certifier (v4.0.0)..."
    lake update
    lake exe cache get
    lake build
    lake exe vcore_certifier --verify-bench
else
    echo "[INFO]: lake не найден в PATH, пропускаем пересборку Lean 4."
fi

# 3. Сборка нативного C++ AVX2 CPU-движка и (при наличии nvcc) CUDA GPU-ядра
if command -v cmake &> /dev/null; then
    if command -v nvcc &> /dev/null; then
        echo "[NATIVE C++/CUDA]: Компиляция libvcore_cpu (AVX2) и libvcore_engine (CUDA)..."
    else
        echo "[NATIVE C++]: Компиляция нативного AVX2 CPU-движка libvcore_cpu (nvcc не найден)..."
    fi
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
else
    echo "[INFO]: cmake не найден, используется режим PyTorch CPU/GPU."
fi

echo "--- СИСТЕМА V-CORE L1-CERTIFIED ГОТОВА К РАБОТЕ ---"
