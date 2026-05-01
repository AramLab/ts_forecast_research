"""
models/arima_model.py
Auto ARIMA прогнозирование через pmdarima.

Два режима скорости (параметр fast):
  fast=False (default) - полный SARIMA поиск, точнее, ~5-30с на ряд
  fast=True            - упрощённый ARIMA без сезонности, ~1-3с на ряд
                         используется внутри CEEMDAN для IMF-компонент
"""
import numpy as np
import pandas as pd
from analysis.metrics import calculate_metrics, infer_period


def arima_forecast(
    series: pd.Series,
    test_size: int = 18,
    fast: bool = False,
    verbose: bool = False,
) -> tuple[pd.Series, dict]:
    import pmdarima as pm

    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    m = infer_period(series)

    if fast:
        # Быстрый режим: нет сезонной части, ограниченный поиск
        model = pm.auto_arima(
            train,
            seasonal=False,
            max_p=2, max_q=2,
            d=None, max_d=1,
            stepwise=True,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
            trace=verbose,
        )
    else:
        model = pm.auto_arima(
            train,
            seasonal=(m > 1),
            m=m if m > 1 else 1,
            D=1 if m > 1 else 0,
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            stepwise=True,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
            trace=verbose,
        )

    forecast = np.array(model.predict(n_periods=test_size), dtype=float)
    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = f"ARIMA{model.order}"
    return pd.Series(forecast, index=test.index), metrics


def arima_fast(series: pd.Series, test_size: int = 18):
    """Быстрая ARIMA без сезонности - для использования внутри CEEMDAN."""
    return arima_forecast(series, test_size=test_size, fast=True)
