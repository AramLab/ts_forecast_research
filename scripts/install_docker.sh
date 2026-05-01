#!/usr/bin/env bash
# =============================================================================
# install_docker.sh — Сборка и запуск Docker контейнера
# =============================================================================
# Использование:
#   bash scripts/install_docker.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# Проверяем Docker
if ! command -v docker &>/dev/null; then
    echo "Docker не найден. Установите Docker Desktop:"
    echo "  Mac: https://docs.docker.com/desktop/install/mac-install/"
    echo "  Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

log "Docker найден: $(docker --version)"

# Создаём Dockerfile если не существует
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cat > "$PROJECT_DIR/Dockerfile" <<'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (компилятор для EMD-signal)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc g++ \
    python3-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pip обновление
RUN pip install --upgrade pip setuptools wheel

# Сначала тяжёлые пакеты (кешируются Docker слоями)
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    numpy scipy

# CEEMDAN
RUN pip install --no-cache-dir EMD-signal

# Остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаём папку результатов
RUN mkdir -p results/plots

# Порт Jupyter
EXPOSE 8888

# По умолчанию запускаем Jupyter
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/app/notebooks"]
DOCKERFILE

log "Dockerfile создан"

# Сборка образа
echo ""
log "Сборка Docker образа ts_forecast..."
warn "Это займёт 5-15 минут при первом запуске..."
docker build -t ts_forecast "$PROJECT_DIR"
log "Образ собран"

# Создаём папку результатов на хосте
mkdir -p "$PROJECT_DIR/results"

# Запуск контейнера
echo ""
log "Запуск контейнера..."
docker run -d \
    --name ts_forecast_app \
    -p 8888:8888 \
    -v "$PROJECT_DIR/results:/app/results" \
    ts_forecast

sleep 3

echo ""
echo "============================================================"
echo " ✅ Контейнер запущен!"
echo ""
echo "   Jupyter Notebook: http://localhost:8888"
echo "   Результаты сохраняются в: $(pwd)/results/"
echo ""
echo "   Для CLI внутри контейнера:"
echo "   docker exec -it ts_forecast_app python main.py --mode demo --datasets m4 m3"
echo ""
echo "   Остановить:"
echo "   docker stop ts_forecast_app && docker rm ts_forecast_app"
echo "============================================================"
