"""
data/loaders.py
Загрузка эталонных датасетов M3, M4, M5 из пакета datasetsforecast.
Все данные скачиваются автоматически при первом вызове и кешируются в CACHE_DIR.

Реальная структура данных datasetsforecast v1.0:
  M4.load()  -> (train_df, test_df)   колонки: unique_id, ds, y
  M3.load()  -> (train_df, test_df)   колонки: unique_id, ds, y
  M5.load()  -> (train_df, future_df, S_df)
               train_df колонки: unique_id, ds, y  (уже long-format!)
               unique_id формат: "FOODS_1_001_CA_1_validation"
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional

CACHE_DIR = os.environ.get("TS_CACHE_DIR", "/tmp/ts_datasets")


# ── Вспомогательные ──────────────────────────────────────────────────────────

def _ensure_datasetsforecast():
    try:
        import datasetsforecast  # noqa
    except ImportError:
        import subprocess, sys
        print("  Устанавливаем datasetsforecast...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasetsforecast", "-q"],
            check=True,
        )


def _normalize_long_df(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    """
    Приводит датафрейм к стандартному виду: unique_id, ds, y.
    Обрабатывает разные варианты именования колонок от datasetsforecast.
    """
    df = df.copy()

    # unique_id
    for c in ["unique_id", "id", "series_id", "item_id"]:
        if c in df.columns:
            if c != "unique_id":
                df = df.rename(columns={c: "unique_id"})
            break
    else:
        print(f"  ❌ {source_name}: нет колонки ID. Есть: {df.columns.tolist()}")
        return None

    # ds
    for c in ["ds", "date", "timestamp", "time", "period"]:
        if c in df.columns:
            if c != "ds":
                df = df.rename(columns={c: "ds"})
            break
    else:
        print(f"  ❌ {source_name}: нет колонки дат. Есть: {df.columns.tolist()}")
        return None

    # y
    for c in ["y", "value", "sales", "target"]:
        if c in df.columns:
            if c != "y":
                df = df.rename(columns={c: "y"})
            break
    else:
        print(f"  ❌ {source_name}: нет колонки значений. Есть: {df.columns.tolist()}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.dropna(subset=["ds", "y"])

    return df[["unique_id", "ds", "y"]].reset_index(drop=True)


def _to_series(df: pd.DataFrame, series_id: str, dataset_name: str) -> Optional[pd.Series]:
    """
    Извлекает один ряд из long-format датафрейма.
    НЕ вызывает asfreq() — он добавляет NaN за пропущенные периоды
    и ломает модели на рядах M4 где даты нерегулярны.
    """
    data = df[df["unique_id"] == series_id].sort_values("ds").copy()
    if len(data) == 0:
        return None

    values = data["y"].values.astype(float)
    idx = pd.to_datetime(data["ds"])

    # Убираем NaN
    mask = ~np.isnan(values)
    values, idx = values[mask], idx[mask]
    if len(values) < 4:
        return None

    s = pd.Series(values, index=idx, name=f"{dataset_name}_{series_id}")
    if s.index.duplicated().any():
        s = s.groupby(level=0).mean()
    return s.sort_index()


# ── M4 ───────────────────────────────────────────────────────────────────────

def _clear_stale_m4_cache() -> None:
    """
    Удаляет весь кеш M4 если он несовместим с текущей numpy.
    datasetsforecast хранит кеш в CACHE_DIR/m4/ как .pkl файлы.
    При обновлении numpy (>=2.0) старые pickle несовместимы.
    """
    import shutil, glob

    # Ищем все pkl-файлы в любом подкаталоге кеша M4
    m4_cache_dir = os.path.join(CACHE_DIR, "m4")
    pkl_files = glob.glob(
        os.path.join(m4_cache_dir, "**", "*.pkl"), recursive=True
    ) + glob.glob(
        os.path.join(m4_cache_dir, "**", "*.pickle"), recursive=True
    )

    for pkl in pkl_files:
        try:
            import pickle
            with open(pkl, "rb") as f:
                pickle.load(f)
            # файл читается — он совместим, не трогаем
        except Exception:
            print(f"  ⚠ Несовместимый кеш numpy, удаляем: {pkl}")
            try:
                os.remove(pkl)
            except OSError:
                pass


def load_m4(group: str = "Monthly") -> Optional[pd.DataFrame]:
    """
    Загрузка M4.
    Группы: Hourly, Daily, Weekly, Monthly, Quarterly, Yearly
    Returns: pd.DataFrame -> unique_id, ds, y

    При ошибке numpy._core.numeric автоматически сбрасывает стale-кеш
    и скачивает датасет заново.
    """
    _ensure_datasetsforecast()
    _clear_stale_m4_cache()

    def _do_load():
        from datasetsforecast.m4 import M4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return M4.load(directory=CACHE_DIR, group=group)

    try:
        data = _do_load()
    except Exception as e_first:
        # Если загрузка упала — скорее всего из-за нового stale pickle
        # который datasetsforecast создал в процессе попытки.
        # Удаляем ВСЕ pkl M4 и пробуем ещё раз.
        import shutil, glob
        m4_dir = os.path.join(CACHE_DIR, "m4")
        if os.path.isdir(m4_dir):
            print(f"  ⚠ Первая попытка упала ({e_first.__class__.__name__}).")
            print(f"  ↳ Удаляем весь кеш M4: {m4_dir}")
            shutil.rmtree(m4_dir, ignore_errors=True)
            print(f"  ↳ Скачиваем M4 заново...")
        try:
            data = _do_load()
        except Exception as e:
            print(f"❌ Ошибка загрузки M4 ({group}): {e}")
            import traceback; traceback.print_exc()
            return None

    try:
        raw = data[0] if isinstance(data, (list, tuple)) else data
        df = _normalize_long_df(raw, f"M4/{group}")
        if df is None:
            return None
        print(f"✅ M4 ({group}): {df['unique_id'].nunique()} рядов, {len(df)} наблюдений")
        return df
    except Exception as e:
        print(f"❌ Ошибка обработки M4 ({group}): {e}")
        return None


def load_m3(group: str = "Monthly") -> Optional[pd.DataFrame]:
    """
    Загрузка M3.
    Группы: Monthly, Quarterly, Yearly, Other
    M3 использует .xls файлы — требует xlrd.
    Returns: pd.DataFrame -> unique_id, ds, y
    """
    _ensure_datasetsforecast()
    try:
        from datasetsforecast.m3 import M3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                data = M3.load(directory=CACHE_DIR, group=group)
            except Exception as e_inner:
                if "xlrd" in str(e_inner).lower() or "excel" in str(e_inner).lower() or "engine" in str(e_inner).lower():
                    import subprocess, sys
                    print("  Устанавливаем xlrd для чтения .xls файлов M3...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "xlrd==2.0.1", "-q"],
                        check=False,
                    )
                    data = M3.load(directory=CACHE_DIR, group=group)
                else:
                    raise
        raw = data[0] if isinstance(data, (list, tuple)) else data
        df = _normalize_long_df(raw, f"M3/{group}")
        if df is None:
            return None
        print(f"✅ M3 ({group}): {df['unique_id'].nunique()} рядов, {len(df)} наблюдений")
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки M3 ({group}): {e}")
        import traceback; traceback.print_exc()
        return None


# ── M5 ───────────────────────────────────────────────────────────────────────

def load_m5(
    aggregation: str = "monthly_category",
    category: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Загрузка M5 с агрегацией.

    datasetsforecast M5.load() возвращает (train_df, future_df, S_df).
    train_df — уже long-format: unique_id (item×store), ds, y.
    unique_id формат: "FOODS_1_001_CA_1_validation"
    Части: category_deptnum_itemnum_state_storenum_split

    aggregation:
      'monthly_category' — по категории (FOODS / HOBBIES / HOUSEHOLD) → 3 ряда
      'monthly_dept'     — по отделу (FOODS_1, FOODS_2, ...) → 7 рядов
      'monthly_store'    — по магазину (CA_1, CA_2, ...) → 10 рядов
      'monthly_state'    — по штату (CA, TX, WI, ...) → 4 ряда
      'daily_item'       — исходный item×store → 30490 рядов (медленно!)

    category: 'FOODS' | 'HOBBIES' | 'HOUSEHOLD'  (фильтр, только для item/dept уровня)
    """
    _ensure_datasetsforecast()
    try:
        from datasetsforecast.m5 import M5
        print(f"  Загрузка M5 (aggregation={aggregation})...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = M5.load(directory=CACHE_DIR)

        # Распаковка
        if isinstance(data, (list, tuple)):
            train_raw = data[0]
        else:
            train_raw = data

        train_df = _normalize_long_df(train_raw, "M5/train")
        if train_df is None:
            return None

        n_raw = train_df["unique_id"].nunique()
        print(f"  M5 raw: {n_raw} рядов, {len(train_df)} строк")
        print(f"  M5 пример ID: {list(train_df['unique_id'].unique()[:2])}")

        # Агрегация
        df = _m5_aggregate(train_df, aggregation, category)
        if df is not None and not df.empty:
            print(f"✅ M5 ({aggregation}): {df['unique_id'].nunique()} рядов, {len(df)} наблюдений")
            print(f"   Ряды: {list(df['unique_id'].unique())}")
        return df

    except Exception as e:
        print(f"❌ Ошибка загрузки M5: {e}")
        import traceback; traceback.print_exc()
        return None


def _parse_m5_id(uid: str):
    """
    Парсит unique_id M5: "FOODS_1_001_CA_1_validation"
    Возвращает dict с cat, dept, state, store.
    """
    parts = uid.split("_")
    result = {"cat": "UNK", "dept": "UNK", "state": "UNK", "store": "UNK"}
    try:
        if len(parts) >= 1:
            result["cat"] = parts[0]                          # FOODS
        if len(parts) >= 2:
            result["dept"] = f"{parts[0]}_{parts[1]}"        # FOODS_1
        if len(parts) >= 4:
            result["state"] = parts[3]                        # CA
        if len(parts) >= 5:
            result["store"] = f"{parts[3]}_{parts[4]}"       # CA_1
    except Exception:
        pass
    return result


def _m5_aggregate(
    train_df: pd.DataFrame,
    aggregation: str,
    category: Optional[str],
) -> Optional[pd.DataFrame]:
    """Агрегирует M5 long-format до нужного уровня."""
    try:
        df = train_df.copy()

        # Парсим структуру ID
        parsed = df["unique_id"].apply(_parse_m5_id).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

        # Фильтр по категории
        if category:
            cat_upper = category.upper()
            df = df[df["cat"] == cat_upper].copy()
            if len(df) == 0:
                avail = sorted(train_df["unique_id"].apply(lambda x: x.split("_")[0]).unique())
                print(f"  ⚠ Категория '{category}' не найдена. Доступные: {avail}")
                return None

        # Выбор колонки группировки
        group_map = {
            "monthly_category": "cat",
            "daily_category":   "cat",
            "monthly_dept":     "dept",
            "monthly_store":    "store",
            "monthly_state":    "state",
            "daily_item":       "unique_id",
            "weekly_item":      "unique_id",
        }
        group_col = group_map.get(aggregation, "cat")

        # Суммируем по группе и дате
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg_df = (
                df.groupby([group_col, "ds"], observed=True)["y"]
                .sum()
                .reset_index()
                .rename(columns={group_col: "unique_id"})
            )

        # Ресемплируем до нужной частоты
        resample_map = {
            "monthly_category": "MS",
            "monthly_dept":     "MS",
            "monthly_store":    "MS",
            "monthly_state":    "MS",
            "weekly_item":      "W",
            "daily_item":       None,
            "daily_category":   None,
        }
        resample_freq = resample_map.get(aggregation, "MS")

        if resample_freq:
            records = []
            for uid, grp in agg_df.groupby("unique_id", observed=True):
                s = grp.set_index("ds")["y"].sort_index()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = s.resample(resample_freq).sum()
                for dt, val in s.items():
                    records.append({"unique_id": str(uid), "ds": dt, "y": float(val)})
            result = pd.DataFrame(records)
        else:
            agg_df["unique_id"] = agg_df["unique_id"].astype(str)
            result = agg_df

        return result

    except Exception as e:
        print(f"  ❌ Ошибка агрегации M5: {e}")
        import traceback; traceback.print_exc()
        return None


# ── Загрузка всех датасетов ──────────────────────────────────────────────────

def load_all_datasets(
    use_m4: bool = True,
    use_m3: bool = True,
    use_m5: bool = True,
    m4_group: str = "Monthly",
    m3_group: str = "Monthly",
    m5_aggregation: str = "monthly_category",
    m5_category: Optional[str] = None,
) -> dict:
    result = {}
    print("\n" + "=" * 60)
    print("  ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    if use_m4:
        result["M4"] = load_m4(group=m4_group)
    if use_m3:
        result["M3"] = load_m3(group=m3_group)
    if use_m5:
        result["M5"] = load_m5(aggregation=m5_aggregation, category=m5_category)
    available = [k for k, v in result.items() if v is not None]
    print(f"\n  Загружено: {available}")
    return result


# ── Подготовка одного ряда ────────────────────────────────────────────────────

def prepare_series(
    df: pd.DataFrame,
    series_id: str,
    dataset_name: str,
    min_length: int = 24,
) -> Optional[pd.Series]:
    """
    Возвращает pd.Series или None если ряд слишком короткий.
    min_length=24 — минимум 2 года для месячных данных.
    """
    s = _to_series(df, series_id, dataset_name)
    if s is None:
        return None
    s = s.dropna()
    if len(s) < min_length:
        print(f"    (длина {len(s)} < {min_length}, пропускаем)")
        return None
    return s


def get_series_ids(
    df: pd.DataFrame,
    mode: str = "demo",
    max_series: int = 10,
    category_col: Optional[str] = None,
    category_val: Optional[str] = None,
) -> list:
    """
    Список ID рядов по режиму.
    mode: 'demo' | 'sample' | 'category' | 'full'
    """
    if category_col and category_val and category_col in df.columns:
        filtered = df[df[category_col] == category_val]
    else:
        filtered = df

    all_ids = filtered["unique_id"].unique()
    n = len(all_ids)

    if mode == "demo":
        indices = sorted(set([0, n // 2, n - 1]))
        ids = [all_ids[i] for i in indices]
    elif mode == "sample":
        ids = all_ids[:max_series]
    elif mode == "category":
        ids = all_ids[:max_series] if max_series > 0 else all_ids
    elif mode == "full":
        ids = all_ids
    else:
        ids = all_ids[:3]

    return list(ids)


# ── M3 из TSF-файла (один файл) ──────────────────────────────────────────────

def load_m3_tsf(
    filepath: str,
    group: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Загрузка M3 из одного .tsf файла (Monash format).
    Добавляет колонку 'freq' для корректной стратификации.

    Скачать файлы:
      Monthly:    https://zenodo.org/record/4656083  → m3_monthly_dataset.tsf
      Quarterly:  https://zenodo.org/record/4656073  → m3_quarterly_dataset.tsf
      Yearly:     https://zenodo.org/record/4656078  → m3_yearly_dataset.tsf
      Other:      https://zenodo.org/record/4656088  → m3_other_dataset.tsf

    Для выборки с разными частотами используйте load_m3_multigroup().
    """
    from data.tsf_reader import parse_tsf

    # Маппинг частот TSF → метки
    _FREQ_LABEL = {
        "monthly": "Monthly", "quarterly": "Quarterly",
        "yearly": "Yearly",   "other": "Other",
        "weekly": "Weekly",   "daily": "Daily",
    }

    try:
        print(f"  Загрузка M3 из TSF: {filepath}")
        df, metadata = parse_tsf(filepath)
        df = _normalize_long_df(df, "M3/TSF")
        if df is None:
            return None

        # Проставляем freq из метаданных файла
        raw_freq = metadata.get("frequency", "").lower()
        freq_label = _FREQ_LABEL.get(raw_freq, raw_freq.capitalize() or "Unknown")
        df["freq"] = freq_label

        n_series = df["unique_id"].nunique()
        horizon = metadata.get("horizon", "?")
        print(f"✅ M3 (TSF/{freq_label}): {n_series} рядов, "
              f"{len(df)} наблюдений, горизонт={horizon}")
        return df

    except Exception as e:
        print(f"❌ Ошибка загрузки M3 TSF: {e}")
        import traceback; traceback.print_exc()
        return None


# ── M3 из нескольких TSF-файлов (разные частоты) ─────────────────────────────

def load_m3_multigroup(
    monthly_tsf:    Optional[str] = None,
    quarterly_tsf:  Optional[str] = None,
    yearly_tsf:     Optional[str] = None,
    other_tsf:      Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Загружает M3 из нескольких .tsf файлов разных частот и объединяет.
    Каждый ряд получает колонку 'freq' → используется для стратификации.

    Для выборки с разными частотами нужно передать хотя бы 2 файла.

    Параметры:
        monthly_tsf   — путь к m3_monthly_dataset.tsf
        quarterly_tsf — путь к m3_quarterly_dataset.tsf
        yearly_tsf    — путь к m3_yearly_dataset.tsf
        other_tsf     — путь к m3_other_dataset.tsf   (прочие, нерегулярные)

    Пример:
        df = load_m3_multigroup(
            monthly_tsf   = "data/m3_monthly_dataset.tsf",
            quarterly_tsf = "data/m3_quarterly_dataset.tsf",
            yearly_tsf    = "data/m3_yearly_dataset.tsf",
        )
        # df содержит 1428 + 756 + 645 = 2829 рядов
        # колонка freq: Monthly | Quarterly | Yearly

    Размеры групп M3:
        Monthly:    1428 рядов, горизонт 18
        Quarterly:   756 рядов, горизонт  8
        Yearly:      645 рядов, горизонт  6
        Other:       174 ряда,  горизонт  8
        ИТОГО:      3003 рядов
    """
    files = {
        "Monthly":   monthly_tsf,
        "Quarterly": quarterly_tsf,
        "Yearly":    yearly_tsf,
        "Other":     other_tsf,
    }

    # Оставляем только переданные файлы
    files = {k: v for k, v in files.items() if v is not None}
    if not files:
        print("❌ load_m3_multigroup: не передан ни один TSF-файл.")
        return None

    parts = []
    total_series = 0

    for freq_label, filepath in files.items():
        df_part = load_m3_tsf(filepath)
        if df_part is None:
            print(f"  ⚠ Пропускаем {freq_label} (файл не загрузился)")
            continue
        # Перезаписываем freq меткой группы (надёжнее чем из метаданных)
        df_part["freq"] = freq_label
        # Добавляем суффикс к ID чтобы избежать коллизий (N0001_M, N0001_Q...)
        suffix = freq_label[0]  # M, Q, Y, O
        df_part["unique_id"] = df_part["unique_id"].astype(str) + f"_{suffix}"
        parts.append(df_part)
        n = df_part["unique_id"].nunique()
        total_series += n

    if not parts:
        print("❌ Ни один файл M3 не загрузился.")
        return None

    combined = pd.concat(parts, ignore_index=True)
    n_all = combined["unique_id"].nunique()

    print(f"\n✅ M3 multi-group: {n_all} рядов, {len(combined)} наблюдений")
    freq_dist = combined.groupby("freq")["unique_id"].nunique()
    for freq, cnt in freq_dist.items():
        print(f"   {freq:<12}: {cnt} рядов")

    return combined
