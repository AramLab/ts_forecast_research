"""
models/ets_model.py
Auto ETS прогнозирование через statsforecast.

Исправление: statsforecast требует явную freq строку.
Когда индекс без freq (как у M4 без asfreq), определяем её
по медиане разниц между датами.
"""
import numpy as np
import pandas as pd

from analysis.metrics import calculate_metrics, infer_period


def _detect_freq_str(series: pd.Series) -> str:
    """
    Определяет строку частоты для statsforecast по данным ряда.
    statsforecast принимает: 'MS', 'QS', 'YS', 'W', 'D', 'h' и т.д.
    """
    # Сначала пробуем атрибут freq
    idx = series.index
    if hasattr(idx, "freq") and idx.freq is not None:
        fname = getattr(idx.freq, "name", str(idx.freq))
        return fname

    # Определяем по медиане разниц между датами
    if len(idx) < 2:
        return "MS"
    diffs = pd.Series(idx).diff().dropna()
    median_days = diffs.dt.days.median()

    if median_days <= 1.5:
        return "D"
    elif median_days <= 8:
        return "W"
    elif median_days <= 16:
        return "2W"
    elif median_days <= 35:
        return "MS"       # monthly
    elif median_days <= 100:
        return "QS"       # quarterly
    else:
        return "YS"       # yearly


def ets_forecast(
    series: pd.Series,
    test_size: int = 18,
) -> tuple[pd.Series, dict]:
    """
    Автоматический подбор ETS через statsforecast.AutoETS.
    Выбирает лучшую (Error, Trend, Seasonality) комбинацию по AIC.

    Returns
    -------
    (forecast_series, metrics_dict)
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    m     = infer_period(series)
    freq_str = _detect_freq_str(series)

    df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": train.index,
        "y":  train.values.astype(float),
    })

    sf = StatsForecast(
        models=[AutoETS(season_length=m if m > 1 else 1, model="ZZZ")],
        freq=freq_str,
        n_jobs=1,
    )
    sf.fit(df)
    forecast_df = sf.predict(h=test_size)
    forecast = forecast_df["AutoETS"].values.astype(float)

    metrics = calculate_metrics(test.values, forecast, train.values, m)
    metrics["Model"] = "AutoETS"

    return pd.Series(forecast, index=test.index), metrics
