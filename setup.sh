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
    echo "[LEAN 4]: Сборка сертификатора vcore_certifier (v2.0.0)..."
    lake update
    lake exe cache get
    lake build
    lake exe vcore_certifier --verify-bench
else
    echo "[INFO]: lake не найден в PATH, пропускаем пересборку Lean 4."
fi

# 3. Сборка C++/CUDA ядра (если доступен компилятор nvcc)
if command -v nvcc &> /dev/null; then
    echo "[CUDA]: Компиляция нативного ядра libvcore_engine..."
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    cd ..
else
    echo "[INFO]: nvcc не найден, используется режим PyTorch CPU/GPU."
fi

echo "--- СИСТЕМА V-CORE L1-CERTIFIED ГОТОВА К РАБОТЕ ---"
