"""
data/tsf_reader.py
Парсер TSF (Time Series Forecasting) файлов — формат Monash Time Series Repository.

Используется для M3, M4 и других датасетов в .tsf формате.
Источник датасетов: https://forecastingdata.org/

M3 TSF файлы можно скачать с:
  https://zenodo.org/record/4656083  (M3 Monthly, Quarterly, Yearly, Other)
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


# Маппинг частот TSF → pandas resample freq
FREQ_MAP = {
    "daily":     ("D",  7),
    "weekly":    ("W",  52),
    "monthly":   ("MS", 12),
    "quarterly": ("QS", 4),
    "yearly":    ("YS", 1),
    "hourly":    ("h",  24),
    "minutely":  ("min",60),
    "seconds":   ("s",  60),
}


def parse_tsf(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Парсит .tsf файл в стандартный long-format DataFrame.

    Parameters
    ----------
    filepath : str — путь к .tsf файлу

    Returns
    -------
    df : pd.DataFrame с колонками unique_id, ds, y
    metadata : dict с ключами frequency, horizon, missing, equal_length и т.д.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TSF файл не найден: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    metadata = {}
    data_lines = []
    in_data = False

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("@data"):
            in_data = True
            continue

        if not in_data:
            # Парсим заголовок
            if line.lower().startswith("@relation"):
                metadata["relation"] = line.split(maxsplit=1)[-1]
            elif line.lower().startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 3:
                    metadata[f"attr_{parts[1]}"] = parts[2]
            elif line.lower().startswith("@frequency"):
                metadata["frequency"] = line.split()[-1].lower()
            elif line.lower().startswith("@horizon"):
                metadata["horizon"] = int(line.split()[-1])
            elif line.lower().startswith("@missing"):
                metadata["missing"] = line.split()[-1].lower() == "true"
            elif line.lower().startswith("@equallength"):
                metadata["equal_length"] = line.split()[-1].lower() == "true"
        else:
            if line:
                data_lines.append(line)

    freq_str = metadata.get("frequency", "monthly")
    pandas_freq, season_length = FREQ_MAP.get(freq_str, ("MS", 12))
    metadata["pandas_freq"] = pandas_freq
    metadata["season_length"] = season_length

    records = _parse_data_lines(data_lines, pandas_freq)
    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"TSF файл пустой или не распознан: {filepath}")

    print(f"  TSF: {df['unique_id'].nunique()} рядов, {len(df)} наблюдений "
          f"[{freq_str}, горизонт={metadata.get('horizon', '?')}]")
    return df, metadata


def _parse_data_lines(lines: list[str], pandas_freq: str) -> list[dict]:
    """
    Парсит строки данных TSF.

    Форматы:
      T1:1990-01-01 00-00-00:1.0,2.0,3.0,...
      T1:1990-01-01:1.0,2.0,...
      T1::1.0,2.0,...   (без даты)
    """
    records = []
    freq_delta_map = {
        "D": timedelta(days=1),
        "W": timedelta(weeks=1),
        "MS": None,   # обрабатываем через pd.date_range
        "QS": None,
        "YS": None,
        "h": timedelta(hours=1),
    }

    for line in lines:
        # Разбиваем по двоеточию — но не более 3 частей
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue

        series_id = parts[0].strip()

        if len(parts) == 3:
            timestamp_str = parts[1].strip()
            values_str = parts[2].strip()
        else:
            timestamp_str = ""
            values_str = parts[1].strip()

        # Парсим значения
        values = []
        for v in values_str.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                values.append(float(v))
            except ValueError:
                values.append(np.nan)

        if not values:
            continue

        # Парсим начальную дату
        start_date = _parse_timestamp(timestamp_str)

        # Генерируем даты
        try:
            dates = pd.date_range(
                start=start_date,
                periods=len(values),
                freq=pandas_freq,
            )
        except Exception:
            # Фолбэк: просто целочисленный индекс, начиная с даты
            dates = pd.date_range(
                start=start_date,
                periods=len(values),
                freq="MS",
            )

        for dt, val in zip(dates, values):
            records.append({
                "unique_id": series_id,
                "ds": dt,
                "y": val,
            })

    return records


def _parse_timestamp(ts: str) -> datetime:
    """Парсит строку даты из TSF. Возвращает datetime."""
    if not ts:
        return datetime(1900, 1, 1)

    # Убираем время вида "00-00-00" или "00:00:00"
    ts_clean = re.sub(r"\s+\d{2}[-:]\d{2}[-:]\d{2}$", "", ts).strip()

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y-%m",
        "%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue

    # Последний шанс
    try:
        return pd.to_datetime(ts_clean).to_pydatetime()
    except Exception:
        return datetime(1900, 1, 1)
