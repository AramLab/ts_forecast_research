#!/usr/bin/env bash
# =============================================================================
# install_mac.sh — Установка на Mac Apple Silicon (M1 / M2 / M3)
# =============================================================================
# Использование:
#   bash scripts/install_mac.sh
# =============================================================================

set -e  # остановиться при любой ошибке

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo "============================================================"
echo " Установка окружения для ts_forecast на Mac Apple Silicon"
echo "============================================================"
echo ""

# --- Шаг 1: Проверка архитектуры ---
ARCH=$(uname -m)
log "Архитектура: $ARCH"

if [[ "$ARCH" != "arm64" ]]; then
    warn "Вы не на Apple Silicon ($ARCH). Используйте install_linux.sh"
    warn "Продолжаем всё равно..."
fi

# --- Шаг 2: Xcode Command Line Tools ---
echo ""
log "Шаг 1/6: Проверка Xcode Command Line Tools..."
if xcode-select -p &>/dev/null; then
    log "Xcode CLT уже установлены: $(xcode-select -p)"
else
    warn "Устанавливаем Xcode Command Line Tools..."
    warn "Появится диалоговое окно — нажмите Install и дождитесь завершения"
    xcode-select --install
    # Ждём завершения
    while ! xcode-select -p &>/dev/null; do
        sleep 5
    done
    log "Xcode CLT установлены"
fi

# --- Шаг 3: Homebrew ---
echo ""
log "Шаг 2/6: Проверка Homebrew..."
if command -v brew &>/dev/null; then
    log "Homebrew уже установлен: $(brew --version | head -1)"
else
    warn "Устанавливаем Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Добавляем в PATH для M1
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
    log "Homebrew установлен"
fi

# --- Шаг 4: Miniforge (conda для arm64) ---
echo ""
log "Шаг 3/6: Проверка Miniforge (conda arm64)..."
if command -v conda &>/dev/null; then
    log "conda уже доступна: $(conda --version)"
else
    warn "Устанавливаем Miniforge через Homebrew..."
    brew install miniforge
    conda init zsh 2>/dev/null || true
    conda init bash 2>/dev/null || true
    
    # Источник конфигурации
    source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
    log "Miniforge установлен"
    warn "ВАЖНО: Закройте и откройте терминал заново, затем повторно запустите этот скрипт"
    warn "Или выполните: source ~/.zshrc"
    exit 0
fi

# --- Шаг 5: Conda окружение ---
echo ""
log "Шаг 4/6: Создание conda окружения 'ts_forecast'..."
ENV_NAME="ts_forecast"

# Источник conda для использования conda activate внутри скрипта
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Окружение '${ENV_NAME}' уже существует. Переиспользуем."
else
    conda create -n "$ENV_NAME" python=3.11 -y
    log "Окружение создано"
fi

conda activate "$ENV_NAME"
log "Окружение активировано: $(which python)"

# --- Шаг 6: Установка C-зависимостей через conda в нужное окружение ---
echo ""
log "Шаг 5/6: Установка C/Cython зависимостей через conda-forge..."
conda install -n "$ENV_NAME" -c conda-forge numpy scipy cython gcc -y
log "C-зависимости установлены"

# --- Шаг 7: Установка Python пакетов ---
echo ""
log "Шаг 6/6: Установка Python пакетов..."

# Используем python/pip из окружения напрямую
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# Сначала EMD-signal (проблемный пакет)
echo "  Устанавливаем EMD-signal..."
if "$PIP" install EMD-signal 2>&1 | grep -qE "Successfully installed|already satisfied"; then
    log "EMD-signal установлен успешно"
else
    warn "Стандартная установка не прошла, пробуем через Rosetta 2..."
    if arch -x86_64 "$PIP" install EMD-signal 2>&1 | grep -qE "Successfully installed|already satisfied"; then
        log "EMD-signal установлен через Rosetta 2"
    else
        warn "EMD-signal НЕ установлен. CEEMDAN-модели будут недоступны."
        warn "Попробуйте вручную: arch -x86_64 pip install EMD-signal"
    fi
fi

# Остальные пакеты
echo "# PyWavelets (вейвлет-гибриды)
echo "  Устанавливаем PyWavelets..."
"$PIP" install PyWavelets -q && log "PyWavelets установлен" || warn "PyWavelets не установлен: pip install PyWavelets"

# Остальные пакеты
echo "  Устанавливаем остальные пакеты из requirements.txt...""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

"$PIP" install -r "${PROJ_DIR}/requirements.txt"

log "Все пакеты установлены"

# --- Проверка ---
echo ""
echo "============================================================"
echo " Проверка установки"
echo "============================================================"
"$PYTHON" - <<'EOF'
checks = {
    'numpy': 'import numpy',
    'pandas': 'import pandas',
    'pmdarima': 'import pmdarima',
    'statsforecast': 'import statsforecast',
    'prophet': 'from prophet import Prophet',
    'tensorflow': 'import tensorflow',
    'datasetsforecast': 'import datasetsforecast',
    'PyEMD (CEEMDAN)': 'from PyEMD import CEEMDAN',
    'sklearn': 'from sklearn.preprocessing import MinMaxScaler',
}

all_ok = True
for name, imp in checks.items():
    try:
        exec(imp)
        print(f'  ✅ {name}')
    except ImportError as e:
        print(f'  ❌ {name}: {e}')
        if 'CEEMDAN' not in name:
            all_ok = False

if all_ok:
    print('\n✅ Все основные пакеты установлены!')
else:
    print('\n⚠️  Некоторые пакеты отсутствуют. Смотри README.md')
EOF

echo ""
echo "============================================================"
echo " Готово!"
echo ""
echo " Окружение: ${CONDA_BASE}/envs/${ENV_NAME}"
echo ""
echo " Для работы в новом терминале:"
echo "   source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "   conda activate ts_forecast"
echo "   cd $(pwd)"
echo "   python main.py --mode demo --datasets m4 m3"
echo ""
echo " Или без conda activate (используем python напрямую):"
echo "   ${PYTHON} main.py --mode demo --datasets m4 m3"
echo ""
echo " Jupyter:"
echo "   ${CONDA_BASE}/envs/${ENV_NAME}/bin/jupyter notebook notebooks/main_analysis.ipynb"
echo "============================================================"
