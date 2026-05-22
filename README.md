# ts_forecast_final

Гибридные методы прогнозирования временных рядов на основе CEEMDAN.

## Единая точка входа: run_experiment.py

### Запуск

```bash
# Синтетика (4 длинных ряда, N=1440: Trend Break × 2, Complex Season × 2)
python run_experiment.py --dataset synthetic --no_plots

# M3 Monthly (локальный TSF-файл, папка m3/datasets/)
python run_experiment.py --dataset m3 --m3_group Monthly --n_series 50 --workers 5 --no_plots

# M4 Monthly (длинные ряды N≥500, скачивается автоматически)
python run_experiment.py --dataset m4 --m4_group Monthly --n_series 50 --workers 5 --no_plots

# M3 Quarterly / Yearly / Other
python run_experiment.py --dataset m3 --m3_group Quarterly --n_series 50 --workers 5 --no_plots

# M4 Quarterly / Yearly / Daily / Weekly / Hourly
python run_experiment.py --dataset m4 --m4_group Quarterly --n_series 50 --workers 5 --no_plots

# Два датасета одновременно
python run_experiment.py --dataset m3 --workers 4 --out results/m3 &
python run_experiment.py --dataset m4 --workers 4 --out results/m4 &
wait
```

### Флаги

| Флаг | Default | Описание |
|------|---------|----------|
| `--dataset` | m4 | `synthetic`, `m3`, `m4` |
| `--n_series` | 50 | Рядов (для synthetic игнорируется) |
| `--min_length` | авто | Мин. длина (M4: 500, M3: 100) |
| `--m3_group` | Monthly | Группа M3 (`Monthly`, `Quarterly`, `Yearly`, `Other`) |
| `--m4_group` | Monthly | Группа M4 (`Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, `Yearly`) |
| `--horizon` | 18 | Горизонт h |
| `--workers` | 4 | Параллельных процессов (M3 Pro: 5) |
| `--timeout` | 600 | Таймаут на ряд (сек) |
| `--no_ceemdan` | — | Пропустить CEEMDAN-гибриды |
| `--no_wavelet` | — | Пропустить вейвлет-гибриды |
| `--no_lstm` | — | Пропустить LSTM и CEEMDAN+LSTM |
| `--no_prophet` | — | Пропустить Prophet |
| `--no_transformer` | — | Пропустить Transformer |
| `--no_plots` | — | HTML без графиков |
| `--ceemdan_trials` | 30 | Ансамблей CEEMDAN |
| `--n_wavelet_modes` | 3 | Макс. мод вейвлета (1–3) |
| `--wavelet` | db4 | Тип вейвлета |
| `--out` | results/\<dataset\> | Папка результатов |

## Все модели (24+)

| Группа | Модели |
|--------|--------|
| Базовые | ARIMA, ETS, Prophet, LSTM |
| CEEMDAN | CEEMDAN+ARIMA, CEEMDAN+ETS, CEEMDAN+Prophet, CEEMDAN+LSTM |
| Wavelet A | Wavelet(1/2/3)+ARIMA, Wavelet(1/2/3)+ETS |
| Wavelet B | ARIMA+Wavelet(1/2), ETS+Wavelet(1/2) |
| STL-Base | STL+ARIMA(trend)+Naive, STL+ETS(trend)+Naive |
| STL-A | STL+ARIMA(trend)+Wavelet, STL+ETS(trend)+Wavelet |
| STL-B | STL+Wavelet(trend)+ARIMA, STL+Wavelet(trend)+ETS |
| Transformer | Transformer(Wavelet,4), Transformer(CEEMDAN,4) |

## Синтетические ряды (N=1440)

По указанию научного руководителя — длинные ряды (1440 точек = 120 лет × 12 мес):
- **Trend Break (скачок)** — скачок тренда +30, смена наклона
- **Trend Break (спад)** — скачок -20, крутой спад
- **Complex Season (3 цикла)** — периоды 12, 7, 3.7 (некратные)
- **Complex Season (4 цикла)** — периоды 12, 5, 3.3, 2.1 (некратные)

## Ошибка M4 (numpy._core.numeric)

Стартовый кеш M4 несовместим с текущей версией numpy. 
Скрипт автоматически удаляет устаревший pickle и скачивает заново.

## Установка (macOS)

```bash
bash scripts/install_mac.sh
```

## Структура

```
run_experiment.py        ← единая точка входа (параллельный)
runner.py                ← движок с forecast_series() для всех 24+ моделей
analysis/metrics.py      ← sMAPE, MASE, RMSE, MAE
ceemdan/ceemdan_hybrid.py
data/loaders.py          ← M3, M4 + авто-очистка stale cache
data/synthetic.py        ← длинные синтетические ряды N=1440
models/arima_model.py
models/ets_model.py
models/lstm_model.py
models/prophet_model.py
models/stl_hybrid.py
models/wavelet/wavelet_hybrid.py
models/transformer/mode_transformer.py
m3/datasets/             ← TSF-файлы M3
scripts/                 ← установочные скрипты
```
