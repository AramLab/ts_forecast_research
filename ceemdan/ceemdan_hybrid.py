import os
import warnings
import hashlib
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from analysis.metrics import calculate_metrics, infer_period

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def check_ceemdan() -> bool:
    try:
        from PyEMD import CEEMDAN
        _ = CEEMDAN(trials=1, noise_width=0.01)
        return True
    except (ImportError, TypeError, AttributeError) as e:
        print(f"[CEEMDAN check] Failed: {type(e).__name__}: {e}")
        return False

CEEMDAN_AVAILABLE = check_ceemdan()
if CEEMDAN_AVAILABLE:
    from PyEMD import CEEMDAN

_IMF_CACHE = {}
_IMF_CACHE_MAX = 3

def _get_imfs(
    train_values: np.ndarray,
    trials: int = 50,
    noise_width: float = 0.05,
    random_state: Optional[int] = None
) -> np.ndarray:
    params_str = f"{trials}_{noise_width}_{random_state}"
    key = f"{hashlib.md5(train_values.tobytes()).hexdigest()}_{params_str}"
    if key in _IMF_CACHE:
        print(f"[CEEMDAN] using cached IMFs (key={key[:16]}...)")
        return _IMF_CACHE[key]

    if not CEEMDAN_AVAILABLE:
        raise ImportError("PyEMD не установлен. pip install EMD-signal")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_kwargs = {"trials": trials, "noise_width": noise_width}
        if random_state is not None:
            init_kwargs["random_state"] = random_state
        cem = CEEMDAN(**init_kwargs)

        if hasattr(cem, 'decompose'):
            imfs = cem.decompose(train_values.astype(float))
        else:
            imfs = cem(train_values.astype(float))

    if len(_IMF_CACHE) >= _IMF_CACHE_MAX:
        _IMF_CACHE.pop(next(iter(_IMF_CACHE)))
    _IMF_CACHE[key] = imfs
    print(f"[CEEMDAN] decomposed into {len(imfs)} IMFs (cached)")
    return imfs

def _imf_fallback(imf: np.ndarray, test_size: int, m: int) -> np.ndarray:
    if np.std(imf) < 1e-10:
        return np.zeros(test_size)
    if m > 1 and len(imf) >= m:
        return np.tile(imf[-m:], (test_size // m) + 2)[:test_size]
    return np.full(test_size, imf[-1])

# ========== Модели для прогнозирования отдельных IMF ==========
def _arima_imf(s: pd.Series, test_size: int, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    import pmdarima as pm
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                s, seasonal=False,
                max_p=2, max_q=2, d=None, max_d=1,
                stepwise=True, error_action="ignore", suppress_warnings=True
            )
        fc = model.predict(n_periods=test_size)
        if verbose:
            print(f"    [ARIMA_IMF] success, first={fc[:3]}")
        return np.array(fc, dtype=float), {}
    except Exception as e:
        if verbose:
            print(f"    [ARIMA_IMF] ERROR: {type(e).__name__}: {e} -> fallback")
        m = infer_period(s)
        return _imf_fallback(s.values, test_size, m), {}

def _ets_imf(s: pd.Series, test_size: int, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS
        from models.ets_model import _detect_freq_str
        m = infer_period(s)
        freq_str = _detect_freq_str(s)
        df = pd.DataFrame({
            "unique_id": "imf",
            "ds": s.index,
            "y": s.values.astype(float)
        })
        sf = StatsForecast(
            models=[AutoETS(season_length=m if m > 1 else 1, model="ZZZ")],
            freq=freq_str, n_jobs=1, verbose=False
        )
        sf.fit(df)
        fc = sf.predict(h=test_size)["AutoETS"].values
        if verbose:
            print(f"    [ETS_IMF] success, first={fc[:3]}")
        return fc, {}
    except Exception as e:
        if verbose:
            print(f"    [ETS_IMF] ERROR: {type(e).__name__}: {e} -> fallback")
        m = infer_period(s)
        return _imf_fallback(s.values, test_size, m), {}

def _prophet_imf(s: pd.Series, test_size: int, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        if isinstance(s.index, pd.DatetimeIndex):
            train_dates = s.index
            freq = pd.infer_freq(s.index) or "MS"
        else:
            start_date = pd.Timestamp("2000-01-01")
            freq = "MS"
            train_dates = pd.date_range(start=start_date, periods=len(s), freq=freq)
        df = pd.DataFrame({"ds": train_dates, "y": s.values.astype(float)})
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            uncertainty_samples=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        last_date = train_dates[-1]
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                        periods=test_size, freq=freq)
        else:
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                        periods=test_size, freq="MS")
        future = pd.DataFrame({"ds": future_dates})
        fc = model.predict(future)["yhat"].values
        if verbose:
            print(f"    [Prophet_IMF] success, first={fc[:3]}")
        return fc, {}
    except Exception as e:
        if verbose:
            print(f"    [Prophet_IMF] ERROR: {type(e).__name__}: {e} -> fallback")
        m = infer_period(s)
        return _imf_fallback(s.values, test_size, m), {}

def _es_imf(s: pd.Series, test_size: int, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = infer_period(s)
        vals = s.values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if m > 1 and len(vals) >= 2 * m:
                fit = ExponentialSmoothing(
                    vals, trend="add", seasonal="add", seasonal_periods=m
                ).fit(optimized=True)
            else:
                fit = ExponentialSmoothing(
                    vals, trend="add", seasonal=None
                ).fit(optimized=True)
            fc = fit.forecast(test_size)
        if verbose:
            print(f"    [ES_IMF] success, first={fc[:3]}")
        return np.array(fc, dtype=float), {}
    except Exception as e:
        if verbose:
            print(f"    [ES_IMF] ERROR: {type(e).__name__}: {e} -> fallback")
        m = infer_period(s)
        return _imf_fallback(s.values, test_size, m), {}

def _lstm_imf(s: pd.Series, test_size: int, verbose: bool = False, **kwargs) -> Tuple[np.ndarray, dict]:
    """Прогнозирование IMF с помощью LSTM (fast mode)."""
    try:
        from models.lstm_model import lstm_fast
        fc, _ = lstm_fast(s, test_size=test_size)  # lstm_fast возвращает (series, dict)
        if verbose:
            print(f"    [LSTM_IMF] success, first={fc[:3]}")
        return fc.values, {}
    except Exception as e:
        if verbose:
            print(f"    [LSTM_IMF] ERROR: {type(e).__name__}: {e} -> fallback")
        m = infer_period(s)
        return _imf_fallback(s.values, test_size, m), {}

# ========== Основная функция гибридного прогноза ==========
def ceemdan_hybrid_forecast(
    series: pd.Series,
    imf_model_name: str,
    model_name: str,
    test_size: int = 18,
    ceemdan_trials: int = 50,
    noise_width: float = 0.05,
    verbose: bool = False,
    random_state: Optional[int] = 42,
) -> Tuple[pd.Series, dict]:
    if test_size >= len(series):
        raise ValueError(f"test_size ({test_size}) must be < len(series) ({len(series)})")

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    m = infer_period(series)

    if not CEEMDAN_AVAILABLE:
        raise ImportError("PyEMD не установлен. pip install EMD-signal")

    imfs = _get_imfs(
        train.values,
        trials=ceemdan_trials,
        noise_width=noise_width,
        random_state=random_state
    )
    if verbose:
        print(f"[CEEMDAN] {model_name}: {len(imfs)} IMFs, using {imf_model_name}")

    model_dispatch = {
        "ARIMA": _arima_imf,
        "ETS": _ets_imf,
        "Prophet": _prophet_imf,
        "ES": _es_imf,
        "LSTM": _lstm_imf,      # <- добавлена настоящая LSTM
    }
    if imf_model_name not in model_dispatch:
        raise ValueError(f"Unknown imf_model_name: {imf_model_name}. Available: {list(model_dispatch.keys())}")
    imf_fn = model_dispatch[imf_model_name]

    imf_forecasts = []
    fallback_count = 0
    MIN_IMF_LENGTH = test_size + 12

    for i, imf in enumerate(imfs):
        imf_series = pd.Series(imf.astype(float), index=train.index[:len(imf)])
        if len(imf_series) < MIN_IMF_LENGTH:
            if verbose:
                print(f"  IMF {i+1}: too short ({len(imf_series)} < {MIN_IMF_LENGTH}), fallback")
            imf_forecasts.append(_imf_fallback(imf, test_size, m))
            fallback_count += 1
            continue

        try:
            fc, _ = imf_fn(imf_series, test_size, verbose=verbose)
            imf_forecasts.append(np.array(fc, dtype=float))
            if verbose:
                print(f"  IMF {i+1}: success, forecast[:3]={fc[:3]}")
        except Exception as e:
            if verbose:
                print(f"  IMF {i+1}: EXCEPTION {type(e).__name__}: {e}, fallback")
            imf_forecasts.append(_imf_fallback(imf, test_size, m))
            fallback_count += 1

    if fallback_count == len(imfs):
        raise RuntimeError(
            f"Все {len(imfs)} IMF упали в fallback — модель {model_name} неработоспособна."
        )

    forecast = np.sum(imf_forecasts, axis=0).astype(float)
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = model_name
    metrics["fallback_ratio"] = fallback_count / len(imfs)

    if verbose:
        print(f"  {model_name} final sMAPE={metrics['sMAPE (%)']:.2f}%")
    return pd.Series(forecast, index=test.index), metrics

# ========== Публичные обёртки ==========
def ceemdan_arima(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    return ceemdan_hybrid_forecast(
        series, "ARIMA", "CEEMDAN+ARIMA",
        test_size, ceemdan_trials,
        random_state=random_state,
        verbose=kw.get("verbose", False)
    )

def ceemdan_ets(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    return ceemdan_hybrid_forecast(
        series, "ETS", "CEEMDAN+ETS",
        test_size, ceemdan_trials,
        random_state=random_state,
        verbose=kw.get("verbose", False)
    )

def ceemdan_prophet(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    return ceemdan_hybrid_forecast(
        series, "Prophet", "CEEMDAN+Prophet",
        test_size, ceemdan_trials,
        random_state=random_state,
        verbose=kw.get("verbose", False)
    )

def ceemdan_es(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    """Гибрид CEEMDAN + Exponential Smoothing (честное имя)."""
    return ceemdan_hybrid_forecast(
        series, "ES", "CEEMDAN+ES",
        test_size, ceemdan_trials,
        random_state=random_state,
        verbose=kw.get("verbose", False)
    )

def ceemdan_lstm_true(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    """Гибрид CEEMDAN + LSTM (настоящий LSTM для каждой IMF)."""
    return ceemdan_hybrid_forecast(
        series, "LSTM", "CEEMDAN+LSTM",
        test_size, ceemdan_trials,
        random_state=random_state,
        verbose=kw.get("verbose", False)
    )

# Старая ceemdan_lstm (которая использовала ES) – оставляем для обратной совместимости,
# но теперь она вызывает ceemdan_es.
def ceemdan_lstm(series, test_size=18, ceemdan_trials=50, random_state=42, **kw):
    import warnings
    warnings.warn(
        "ceemdan_lstm() теперь использует Exponential Smoothing, а не LSTM. "
        "Для настоящего LSTM используйте ceemdan_lstm_true().",
        UserWarning, stacklevel=2
    )
    return ceemdan_es(series, test_size, ceemdan_trials, random_state, **kw)