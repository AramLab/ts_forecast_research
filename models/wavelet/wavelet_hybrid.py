import warnings
import numpy as np
import pandas as pd
import pywt
from typing import Callable, Optional
from analysis.metrics import calculate_metrics, infer_period

def wavelet_decompose(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    n_trend_modes: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(max_level, 4, int(np.log2(len(signal))) - 1)
        level = max(level, 1)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    modes = []
    for i in range(len(coeffs)):
        mask = [np.zeros_like(c) for c in coeffs]
        mask[i] = coeffs[i]
        rec = pywt.waverec(mask, wavelet)[:len(signal)]
        if np.any(~np.isfinite(rec)):
            rec = np.zeros_like(rec)
        modes.append(rec)
    n = min(n_trend_modes, len(modes))
    trend_modes = modes[:n]
    season_modes = modes[n:]
    print(f"[Wavelet] decompose: level={level}, total_modes={len(modes)}, trend_modes={n}, season_modes={len(season_modes)}")
    return trend_modes, season_modes

def _reconstruct_season_naive(season_modes, test_size, m):
    total = np.sum(season_modes, axis=0) if season_modes else np.array([])
    if len(total) == 0:
        return np.zeros(test_size)
    if np.any(np.isnan(total)) or np.any(np.isinf(total)):
        return np.full(test_size, total[0] if len(total) else 0.0)
    period = m if m > 1 else max(2, len(total) // 10)
    period = min(period, len(total))
    if period < 1:
        period = 1
    if len(total) >= period:
        last = total[-period:]
        reps = (test_size // period) + 2
        fc = np.tile(last, reps)[:test_size]
        print(f"  [SeasNaive] period={period}, first={fc[:3]}")
        return fc
    fc = np.full(test_size, total[-1])
    print(f"  [SeasNaive] fallback last value, first={fc[:3]}")
    return fc

def _stat_forecast_mode(mode_values: np.ndarray, train_index: pd.DatetimeIndex, test_size: int, base_model_fn: Callable) -> np.ndarray:
    s = pd.Series(mode_values[:len(train_index)], index=train_index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, _ = base_model_fn(s, test_size=test_size)
        return np.array(pred.values, dtype=float)
    except Exception:
        vals = mode_values
        m_local = max(1, len(vals) // 10)
        if len(vals) >= m_local:
            return np.tile(vals[-m_local:], (test_size // m_local) + 2)[:test_size]
        return np.full(test_size, vals[-1])

def wavelet_trend_stat_season(
    series: pd.Series,
    base_model_fn: Callable,
    model_label: str,
    test_size: int = 18,
    wavelet: str = "db4",
    n_trend_modes: int = 2,
) -> tuple[pd.Series, dict]:
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)
    print(f"[WaveletA] {model_label}, n_trend_modes={n_trend_modes}, m={m}")

    trend_modes, season_modes = wavelet_decompose(train.values, wavelet=wavelet, n_trend_modes=n_trend_modes)
    trend_total = np.sum(trend_modes, axis=0) if trend_modes else train.values.copy()
    print(f"  trend_total: min={trend_total.min():.3f}, max={trend_total.max():.3f}, last={trend_total[-1]:.3f}")

    # Тренд – статистическая модель
    trend_series = pd.Series(trend_total[:len(train)], index=train.index)
    try:
        trend_pred, _ = base_model_fn(trend_series, test_size)
        trend_forecast = trend_pred.values.astype(float)
        print(f"  trend forecast by {model_label} OK, first={trend_forecast[:3]}")
    except Exception as e:
        print(f"  trend forecast FALLBACK ({type(e).__name__}: {e})")
        trend_forecast = np.full(test_size, trend_total[-1])

    # Сезонность – наивный сезонный прогноз
    season_forecast = _reconstruct_season_naive(season_modes, test_size, m)
    print(f"  season naive forecast, first={season_forecast[:3]}")

    forecast = trend_forecast + season_forecast
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"Wavelet({n_trend_modes})+{model_label}"
    print(f"  final sMAPE={metrics['sMAPE (%)']:.2f}%")
    return pd.Series(forecast, index=test.index), metrics

def stat_trend_wavelet_season(
    series: pd.Series,
    base_model_fn: Callable,
    model_label: str,
    test_size: int = 18,
    wavelet: str = "db4",
    n_trend_modes: int = 2,
) -> tuple[pd.Series, dict]:
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)
    print(f"[WaveletB] {model_label}, n_trend_modes={n_trend_modes}, m={m}")

    trend_modes, season_modes = wavelet_decompose(train.values, wavelet=wavelet, n_trend_modes=n_trend_modes)
    trend_total = np.sum(trend_modes, axis=0) if trend_modes else train.values.copy()
    print(f"  trend_total: min={trend_total.min():.3f}, max={trend_total.max():.3f}, last={trend_total[-1]:.3f}")

    # Тренд – статистическая модель
    trend_series = pd.Series(trend_total[:len(train)], index=train.index)
    try:
        trend_pred, _ = base_model_fn(trend_series, test_size)
        trend_forecast = trend_pred.values.astype(float)
        print(f"  trend forecast by {model_label} OK, first={trend_forecast[:3]}")
    except Exception as e:
        print(f"  trend forecast FALLBACK ({type(e).__name__}: {e})")
        trend_forecast = np.full(test_size, trend_total[-1])

    # Сезонность – наивный сезонный прогноз
    season_forecast = _reconstruct_season_naive(season_modes, test_size, m)
    print(f"  season naive forecast, first={season_forecast[:3]}")

    forecast = trend_forecast + season_forecast
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"{model_label}+Wavelet({n_trend_modes})"
    print(f"  final sMAPE={metrics['sMAPE (%)']:.2f}%")
    return pd.Series(forecast, index=test.index), metrics

def wavelet_arima(series, test_size=18, mode="A", n_modes=2, wavelet="db4"):
    from models.arima_model import arima_forecast
    fn = lambda s, ts: arima_forecast(s, test_size=ts)
    if mode == "A":
        return wavelet_trend_stat_season(series, fn, "ARIMA", test_size, wavelet, n_modes)
    else:
        return stat_trend_wavelet_season(series, fn, "ARIMA", test_size, wavelet, n_modes)

def wavelet_ets(series, test_size=18, mode="A", n_modes=2, wavelet="db4"):
    from models.ets_model import ets_forecast
    fn = lambda s, ts: ets_forecast(s, test_size=ts)
    if mode == "A":
        return wavelet_trend_stat_season(series, fn, "ETS", test_size, wavelet, n_modes)
    else:
        return stat_trend_wavelet_season(series, fn, "ETS", test_size, wavelet, n_modes)