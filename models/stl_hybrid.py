"""
models/stl_hybrid.py

STL-гибридные модели (по заданию научного руководителя).

Три варианта (Шаг 3 задания):
─────────────────────────────────────────────────────────────
Вариант STL-A: STL декомпозиция → ARIMA/ETS на тренде, вейвлет на сезонности
Вариант STL-B: STL декомпозиция → вейвлет на тренде, ARIMA/ETS на сезонности
  (модели поменяны местами)
Вариант STL-Base: STL декомпозиция → ARIMA/ETS на тренде, наивный на сезонности
  (baseline без вейвлетов)

Схема декомпозиции STL:
  Ряд = Тренд + Сезонность + Остаток

  STL-A:
    Тренд     → ARIMA/ETS
    Сезонность → вейвлет-прогноз (реконструкция + наивное продолжение)
    Остаток   → наивный (0 или последнее значение)
    Итог      = сумма трёх прогнозов

  STL-B (модели поменяны местами):
    Тренд     → вейвлет-прогноз
    Сезонность → ARIMA/ETS
    Остаток   → наивный
    Итог      = сумма

Зависимости:
  pip install statsmodels PyWavelets
"""
import warnings
import numpy as np
import pandas as pd
from typing import Callable, Literal, Optional

from analysis.metrics import calculate_metrics, infer_period


# ── STL-декомпозиция ──────────────────────────────────────────────────────────

def stl_decompose(
    series: pd.Series,
    period: Optional[int] = None,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STL (Seasonal and Trend decomposition using Loess).

    Возвращает (trend, seasonal, residual) — три массива длины len(series).

    Параметры
    ----------
    series : входной ряд (pd.Series с DatetimeIndex)
    period : период сезонности (None = определяется автоматически)
    robust : устойчивость к выбросам (рекомендуется True)
    """
    from statsmodels.tsa.seasonal import STL

    m = period if period is not None else infer_period(series)
    if m <= 1:
        m = 12  # fallback

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl = STL(series, period=m, robust=robust)
        result = stl.fit()

    return result.trend, result.seasonal, result.resid


# ── Вспомогательные прогнозы ──────────────────────────────────────────────────

def _stat_forecast(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    test_size: int,
    model_fn: Callable,
) -> np.ndarray:
    """Прогнозирует компоненту статистической моделью (ARIMA/ETS)."""
    s = pd.Series(values, index=index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, _ = model_fn(s, test_size=test_size)
        return np.array(pred.values, dtype=float)
    except Exception:
        return np.full(test_size, float(values[-1]))


def _wavelet_forecast(
    values: np.ndarray,
    test_size: int,
    wavelet: str = "db4",
    n_trend_modes: int = 2,
) -> np.ndarray:
    """
    Вейвлет-прогноз компоненты:
    DWT → реконструкция первых n мод → наивное сезонное продолжение.
    """
    import pywt

    if len(values) < 8:
        return np.full(test_size, float(values[-1]))

    level = min(
        pywt.dwt_max_level(len(values), wavelet),
        int(np.log2(len(values))) - 1,
    )
    level = max(level, 1)
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Берём первые n_trend_modes мод
    n = min(n_trend_modes, len(coeffs))
    mask = [np.zeros_like(c) for c in coeffs]
    for i in range(n):
        mask[i] = coeffs[i]
    reconstructed = pywt.waverec(mask, wavelet)[:len(values)]

    # Наивное продолжение: повтор последнего цикла
    cycle_len = max(4, len(reconstructed) // 6)
    if len(reconstructed) >= cycle_len:
        last_cycle = reconstructed[-cycle_len:]
        reps = (test_size // cycle_len) + 2
        return np.tile(last_cycle, reps)[:test_size]
    return np.full(test_size, float(reconstructed[-1]))


def _season_naive_forecast(
    season_values: np.ndarray,
    test_size: int,
    m: int,
) -> np.ndarray:
    """Наивный прогноз сезонности: повтор последнего цикла длиной m."""
    if m > 1 and len(season_values) >= m:
        last = season_values[-m:]
        reps = (test_size // m) + 2
        return np.tile(last, reps)[:test_size]
    return np.full(test_size, float(season_values[-1]))


def _resid_forecast(resid_values: np.ndarray, test_size: int) -> np.ndarray:
    """Прогноз остатка: нули (E[resid] = 0 по построению STL)."""
    return np.zeros(test_size)


# ── STL-Base: ARIMA/ETS на тренде, наивный на сезонности ─────────────────────

def stl_base(
    series: pd.Series,
    model_fn: Callable,
    model_label: str,
    test_size: int = 18,
    robust: bool = True,
) -> tuple[pd.Series, dict]:
    """
    STL-Base:
      Тренд     → ARIMA/ETS
      Сезонность → наивный (повтор)
      Остаток   → 0
    Служит baseline для сравнения с гибридами.
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)

    trend, seasonal, resid = stl_decompose(train, period=m, robust=robust)

    trend_fc = _stat_forecast(trend, train.index, test_size, model_fn)
    season_fc = _season_naive_forecast(seasonal, test_size, m)
    resid_fc = _resid_forecast(resid, test_size)

    forecast = trend_fc + season_fc + resid_fc
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"STL+{model_label}(trend)+Naive(season)"

    return pd.Series(forecast, index=test.index), metrics


# ── STL-A: ARIMA/ETS на тренде, вейвлет на сезонности ───────────────────────

def stl_stat_trend_wavelet_season(
    series: pd.Series,
    model_fn: Callable,
    model_label: str,
    test_size: int = 18,
    wavelet: str = "db4",
    n_wavelet_modes: int = 2,
    robust: bool = True,
) -> tuple[pd.Series, dict]:
    """
    STL-A:
      1. STL → тренд + сезонность + остаток
      2. Тренд → ARIMA/ETS
      3. Сезонность → вейвлет-прогноз
      4. Остаток → 0
      5. Итог = сумма
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)

    trend, seasonal, resid = stl_decompose(train, period=m, robust=robust)

    trend_fc = _stat_forecast(trend, train.index, test_size, model_fn)
    season_fc = _wavelet_forecast(seasonal, test_size, wavelet, n_wavelet_modes)
    resid_fc = _resid_forecast(resid, test_size)

    forecast = trend_fc + season_fc + resid_fc
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"STL+{model_label}(trend)+Wavelet(season)"

    return pd.Series(forecast, index=test.index), metrics


# ── STL-B: вейвлет на тренде, ARIMA/ETS на сезонности ───────────────────────

def stl_wavelet_trend_stat_season(
    series: pd.Series,
    model_fn: Callable,
    model_label: str,
    test_size: int = 18,
    wavelet: str = "db4",
    n_wavelet_modes: int = 2,
    robust: bool = True,
) -> tuple[pd.Series, dict]:
    """
    STL-B (модели поменяны местами):
      1. STL → тренд + сезонность + остаток
      2. Тренд → вейвлет-прогноз
      3. Сезонность → ARIMA/ETS
      4. Остаток → 0
      5. Итог = сумма
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)

    trend, seasonal, resid = stl_decompose(train, period=m, robust=robust)

    trend_fc = _wavelet_forecast(trend, test_size, wavelet, n_wavelet_modes)
    season_fc = _stat_forecast(seasonal, train.index, test_size, model_fn)
    resid_fc = _resid_forecast(resid, test_size)

    forecast = trend_fc + season_fc + resid_fc
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"STL+Wavelet(trend)+{model_label}(season)"

    return pd.Series(forecast, index=test.index), metrics


# ── Готовые комбинации ────────────────────────────────────────────────────────

def stl_arima_base(series, test_size=18, **kw):
    from models.arima_model import arima_forecast
    fn = lambda s, test_size: arima_forecast(s, test_size=test_size)
    return stl_base(series, fn, "ARIMA", test_size, **kw)

def stl_ets_base(series, test_size=18, **kw):
    from models.ets_model import ets_forecast
    fn = lambda s, test_size: ets_forecast(s, test_size=test_size)
    return stl_base(series, fn, "ETS", test_size, **kw)

def stl_arima_wavelet_season(series, test_size=18, wavelet="db4", n_wavelet_modes=2, **kw):
    """STL-A: ARIMA на тренде, вейвлет на сезонности."""
    from models.arima_model import arima_forecast
    fn = lambda s, test_size: arima_forecast(s, test_size=test_size)
    return stl_stat_trend_wavelet_season(
        series, fn, "ARIMA", test_size, wavelet, n_wavelet_modes, **kw
    )

def stl_ets_wavelet_season(series, test_size=18, wavelet="db4", n_wavelet_modes=2, **kw):
    """STL-A: ETS на тренде, вейвлет на сезонности."""
    from models.ets_model import ets_forecast
    fn = lambda s, test_size: ets_forecast(s, test_size=test_size)
    return stl_stat_trend_wavelet_season(
        series, fn, "ETS", test_size, wavelet, n_wavelet_modes, **kw
    )

def stl_wavelet_arima_season(series, test_size=18, wavelet="db4", n_wavelet_modes=2, **kw):
    """STL-B: вейвлет на тренде, ARIMA на сезонности."""
    from models.arima_model import arima_forecast
    fn = lambda s, test_size: arima_forecast(s, test_size=test_size)
    return stl_wavelet_trend_stat_season(
        series, fn, "ARIMA", test_size, wavelet, n_wavelet_modes, **kw
    )

def stl_wavelet_ets_season(series, test_size=18, wavelet="db4", n_wavelet_modes=2, **kw):
    """STL-B: вейвлет на тренде, ETS на сезонности."""
    from models.ets_model import ets_forecast
    fn = lambda s, test_size: ets_forecast(s, test_size=test_size)
    return stl_wavelet_trend_stat_season(
        series, fn, "ETS", test_size, wavelet, n_wavelet_modes, **kw
    )
