# data/synthetic.py
import numpy as np
import pandas as pd

def make_trend_break(
    n=600, break_ratio=0.6, trend_before=0.08, trend_after=-0.1,
    jump=30.0, season_amp=8.0, season_period=12, noise_std=2.5,
    seed=42, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    break_idx = int(n * break_ratio)
    trend = np.where(
        t < break_idx,
        t * trend_before,
        break_idx * trend_before + jump + (t - break_idx) * trend_after
    )
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = np.random.normal(0, noise_std, n)
    values = 100 + trend + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="trend_break")

def make_double_season(
    n=600, periods=(12, 7, 3.5), amplitudes=(12.0, 6.0, 3.0),
    trend_slope=0.02, noise_std=4.0, seed=99, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    trend = trend_slope * t
    season = sum(a * np.sin(2 * np.pi * t / p) for p, a in zip(periods, amplitudes))
    noise = np.random.normal(0, noise_std, n)
    values = 100 + trend + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="double_season")

def make_nonlinear_trend(
    n=600, exp_factor=0.008, season_amp=10.0, season_period=12,
    noise_std=3.5, seed=7, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    trend = 50 * np.exp(exp_factor * t)
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = np.random.normal(0, noise_std, n)
    values = trend + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="nonlinear_trend")

def make_high_noise(
    n=600, trend_slope=0.05, season_amp=8.0, season_period=12,
    noise_std=15.0, seed=17, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    trend = trend_slope * t
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = np.random.normal(0, noise_std, n)
    values = 100 + trend + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="high_noise")

def make_changing_frequency(
    n=600, start_period=12, end_period=6,
    amp_start=8.0, amp_end=12.0, trend_slope=0.02,
    noise_std=4.0, seed=55, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    period = start_period + (end_period - start_period) * (t / n)
    phase = 2 * np.pi * np.cumsum(1 / period)
    amp = amp_start + (amp_end - amp_start) * (t / n)
    season = amp * np.sin(phase)
    trend = trend_slope * t
    noise = np.random.normal(0, noise_std, n)
    values = 100 + trend + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="changing_frequency")

def make_outliers(
    n=600, season_amp=10.0, season_period=12, trend_slope=0.03,
    outlier_count=5, outlier_magnitude=10, noise_std=2.5,
    seed=88, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    trend = trend_slope * t
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = np.random.normal(0, noise_std, n)
    values = 100 + trend + season + noise
    for _ in range(outlier_count):
        idx = np.random.randint(10, n - 20)
        values[idx:idx+3] += outlier_magnitude * np.random.choice([-1, 1])
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="outliers")

def make_pure_seasonal(
    n=600, season_amp=10.0, season_period=12, noise_std=1.5,
    seed=42, start="1900-01"
) -> pd.Series:
    np.random.seed(seed)
    t = np.arange(n)
    season = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = np.random.normal(0, noise_std, n)
    values = 100 + season + noise
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(values, index=idx, name="pure_seasonal")

def get_all_synthetic(n: int = 600) -> dict:
    return {
        "1. Линейный тренд": make_trend_break(
            n=n, break_ratio=1.0, jump=0, trend_after=0.05, noise_std=2.0, seed=1
        ),
        "2. Тренд с изломом": make_trend_break(
            n=n, break_ratio=0.6, trend_before=0.1, trend_after=-0.12,
            jump=25, noise_std=2.5, seed=2
        ),
        "3. Двойная сезонность": make_double_season(
            n=n, periods=(12, 7, 3.5), amplitudes=(12, 7, 3.5),
            noise_std=4.5, seed=3
        ),
        "4. Нелинейный тренд": make_nonlinear_trend(
            n=n, exp_factor=0.008, season_amp=10, noise_std=3.5, seed=4
        ),
        "5. Высокий шум": make_high_noise(
            n=n, noise_std=15.0, season_amp=8, seed=5
        ),
        "6. Плавная смена частоты": make_changing_frequency(
            n=n, start_period=12, end_period=6, noise_std=4.5, seed=6
        ),
        "7. Выбросы": make_outliers(
            n=n, outlier_count=5, outlier_magnitude=12, noise_std=3.0, seed=7
        ),
        "8. Чистая сезонность [контроль]": make_pure_seasonal(
            n=n, season_amp=10, noise_std=1.5, seed=8
        ),
    }