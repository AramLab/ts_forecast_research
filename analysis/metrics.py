"""
analysis/metrics.py
Функции расчёта метрик качества прогнозирования.
"""
import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error. Диапазон [0, 200%]."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-8
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, m: int = 12) -> float:
    """Mean Absolute Scaled Error. MASE < 1 — лучше наивной модели."""
    y_true  = np.array(y_true,  dtype=float)
    y_pred  = np.array(y_pred,  dtype=float)
    y_train = np.array(y_train, dtype=float)
    if len(y_train) <= m:
        return float('nan')
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale < 1e-8:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def calculate_metrics(y_true, y_pred, y_train, m: int = 12) -> dict:
    """RMSE, MAE, sMAPE (%), MASE."""
    y_true  = np.array(y_true,  dtype=float)
    y_pred  = np.array(y_pred,  dtype=float)
    y_train = np.array(y_train, dtype=float)
    # Используем NumPy-реализации, чтобы избежать зависимостей на sklearn/scipy
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "RMSE":      rmse,
        "MAE":       mae,
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE":      mase(y_true, y_pred, y_train, m),
    }


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Агрегирует список результатов в итоговый DataFrame."""
    df = pd.DataFrame(results)
    if df.empty:
        return df
    agg = (
        df.groupby("Model")
        .agg(
            sMAPE_mean=("sMAPE (%)", "mean"),
            sMAPE_median=("sMAPE (%)", "median"),
            sMAPE_std=("sMAPE (%)", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAE_mean=("MAE", "mean"),
            MASE_mean=("MASE", "mean"),
            n_series=("Series_ID", "count"),
        )
        .reset_index()
        .sort_values("sMAPE_mean")
    )
    return agg


def infer_period(series: pd.Series) -> int:
    """
    Определяет период сезонности.
    Сначала по freq индекса, затем по медиане разниц дат.
    """
    idx = series.index
    # По атрибуту freq
    if hasattr(idx, "freq") and idx.freq is not None:
        fname = getattr(idx.freq, "name", str(idx.freq))
        if "M" in fname:  return 12
        if "Q" in fname:  return 4
        if "W" in fname:  return 52
        if "D" in fname:  return 7
        if "H" in fname:  return 24

    # По медиане разниц
    if len(idx) >= 2:
        diffs = pd.Series(idx).diff().dropna()
        median_days = diffs.dt.days.median()
        if median_days <= 1.5:   return 7    # daily -> weekly seasonality
        if median_days <= 8:     return 52   # weekly
        if median_days <= 35:    return 12   # monthly
        if median_days <= 100:   return 4    # quarterly
        return 1                             # yearly

    return 12  # default
