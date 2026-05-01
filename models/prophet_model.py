"""
models/prophet_model.py
Prophet прогнозирование (Meta).
"""
import numpy as np
import pandas as pd

from analysis.metrics import calculate_metrics, infer_period


def prophet_forecast(
    series: pd.Series,
    test_size: int = 18,
    tune: bool = False,
    n_trials: int = 20,
) -> tuple[pd.Series, dict]:
    """
    Прогнозирование с Prophet.

    Prophet требует DatetimeIndex — если реальных дат нет (как в M4),
    генерируем синтетические месячные даты.

    Parameters
    ----------
    series : pd.Series
    test_size : горизонт прогноза
    tune : запустить оптимизацию через Optuna (медленно)
    n_trials : количество проб Optuna (при tune=True)

    Returns
    -------
    (forecast_series, metrics_dict)
    """
    from prophet import Prophet

    train_vals = series.values[:-test_size].astype(float)
    test_vals  = series.values[-test_size:].astype(float)
    m = infer_period(series)

    # Синтетические месячные даты (Prophet требует Timestamp)
    start = pd.Timestamp("2000-01-01")
    train_dates = pd.date_range(start=start, periods=len(train_vals), freq="MS")
    test_dates  = pd.date_range(
        start=train_dates[-1] + pd.DateOffset(months=1),
        periods=test_size, freq="MS",
    )

    train_df = pd.DataFrame({"ds": train_dates, "y": train_vals})

    # Параметры модели
    if tune:
        params = _tune_prophet(train_df, m, n_trials)
    else:
        params = dict(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode="additive",
        )

    model = Prophet(
        growth="linear",
        yearly_seasonality=(m == 12),
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        **params,
    )
    model.fit(train_df)

    future_df = pd.DataFrame({"ds": test_dates})
    forecast = model.predict(future_df)["yhat"].values.astype(float)

    metrics = calculate_metrics(test_vals, forecast, train_vals, m)
    metrics["Model"] = "Prophet"

    train_s = pd.Series(train_vals, index=series.index[:-test_size])
    test_s  = pd.Series(test_vals,  index=series.index[-test_size:])
    del train_s, test_s  # не используем, нужны только для type hints

    return pd.Series(forecast, index=series.index[-test_size:]), metrics


def _tune_prophet(train_df: pd.DataFrame, m: int, n_trials: int) -> dict:
    """Оптимизация гиперпараметров Prophet через Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics

        def objective(trial):
            params = {
                "changepoint_prior_scale": trial.suggest_float(
                    "changepoint_prior_scale", 0.001, 0.5, log=True
                ),
                "seasonality_prior_scale": trial.suggest_float(
                    "seasonality_prior_scale", 0.01, 10.0, log=True
                ),
                "seasonality_mode": trial.suggest_categorical(
                    "seasonality_mode", ["additive", "multiplicative"]
                ),
            }
            try:
                mdl = Prophet(
                    yearly_seasonality=(m == 12),
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    **params,
                )
                mdl.fit(train_df)
                horizon_days = max(m * 30, 90)
                df_cv = cross_validation(
                    mdl,
                    initial=f"{max(3 * m * 30, 180)} days",
                    period=f"{max(m * 30, 60)} days",
                    horizon=f"{horizon_days} days",
                    parallel=None,
                )
                return performance_metrics(df_cv, rolling_window=1)["rmse"].values[0]
            except Exception:
                return float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        print(f"  Prophet tuned: RMSE={study.best_value:.4f}")
        return study.best_params
    except ImportError:
        print("  Optuna не установлена — используем параметры по умолчанию")
        return {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}
    except Exception as e:
        print(f"  Ошибка оптимизации Prophet: {e}")
        return {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}
