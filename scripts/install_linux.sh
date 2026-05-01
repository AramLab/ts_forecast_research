#!/usr/bin/env bash
# =============================================================================
# install_linux.sh — Установка на Linux / Google Colab
# =============================================================================
# Использование:
#   bash scripts/install_linux.sh
#
# В Google Colab скопируйте содержимое в ячейку или:
#   !bash scripts/install_linux.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

echo "============================================================"
echo " Установка ts_forecast на Linux / Google Colab"
echo "============================================================"

# Определяем, работаем ли в Colab
IN_COLAB=false
python3 -c "import google.colab" 2>/dev/null && IN_COLAB=true
$IN_COLAB && log "Обнаружена среда Google Colab" || log "Стандартный Linux"

# --- Системные зависимости (компилятор и т.д.) ---
log "Установка системных зависимостей..."
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq \
        build-essential \
        gcc g++ \
        python3-dev \
        libhdf5-dev \
        pkg-config \
        2>/dev/null
    log "Системные пакеты установлены (apt)"
elif command -v yum &>/dev/null; then
    yum install -y -q gcc gcc-c++ python3-devel
    log "Системные пакеты установлены (yum)"
fi

# --- pip обновление ---
log "Обновление pip..."
pip install --upgrade pip setuptools wheel -q

# --- EMD-signal / PyEMD (CEEMDAN) ---
log "Установка EMD-signal (CEEMDAN)..."
pip install EMD-signal -q && log "EMD-signal установлен" || warn "Ошибка EMD-signal"

# --- Основные пакеты ---
log "Установка основных пакетов..."
pip install -q \
    datasetsforecast \
    pmdarima \
    "statsforecast>=1.6.0" \
    statsmodels \
    prophet \
    tensorflow \
    scikit-learn \
    matplotlib \
    seaborn \
    optuna \
    tqdm \
    rich \
    jupyter \
    ipykernel

log "Все пакеты установлены"

# --- Проверка ---
echo ""
echo "============================================================"
echo " Проверка установки"
echo "============================================================"
python3 - <<'EOF'
packages = {
    'numpy': 'import numpy as np; print(f"  numpy {np.__version__}")',
    'pandas': 'import pandas as pd; print(f"  pandas {pd.__version__}")',
    'pmdarima': 'import pmdarima as pm; print(f"  pmdarima {pm.__version__}")',
    'statsforecast': 'import statsforecast; print(f"  statsforecast OK")',
    'prophet': 'from prophet import Prophet; print(f"  prophet OK")',
    'tensorflow': 'import tensorflow as tf; print(f"  tensorflow {tf.__version__}")',
    'datasetsforecast': 'import datasetsforecast; print(f"  datasetsforecast OK")',
    'PyEMD (CEEMDAN)': 'from PyEMD import CEEMDAN; print(f"  CEEMDAN OK")',
}

ok, fail = 0, 0
for name, cmd in packages.items():
    try:
        exec(cmd)
        ok += 1
    except Exception as e:
        print(f'  ❌ {name}: {e}')
        fail += 1

print(f'\n✅ Установлено: {ok}/{ok+fail}')
if fail:
    print('⚠️  Запустите main.py с USE_CEEMDAN=False если CEEMDAN недоступен')
EOF

echo ""
echo "============================================================"
echo " Готово! Запуск:"
echo ""
echo "   python main.py --mode demo --datasets m4 m3"
echo "============================================================"
