"""
data/sampling.py

Стратифицированная выборка временных рядов для режима "research".

Логика:
  1. Если рядов ≤ max_total → берём все
  2. Если рядов > max_total → стратифицированная выборка:
       - Стратификация по частоте (freq) и длине ряда (length_bucket)
       - Пропорциональная квота из каждой страты
       - Гарантированный минимум из каждой страты (min_per_stratum)

Определение частоты:
  - M4: по префиксу ID (M→Monthly, Q→Quarterly, Y→Yearly, W→Weekly, D→Daily, H→Hourly)
  - M3: по колонке 'freq' если есть (добавляется при multi-group загрузке),
        иначе по медиане разниц дат (векторизованно, быстро)
  - M5: по медиане разниц дат (ежедневные данные)
"""
import numpy as np
import pandas as pd
from typing import Optional


# ── Определение частоты ───────────────────────────────────────────────────────

# Префиксы ID для M4
_M4_PREFIX = {
    "M": "Monthly", "Q": "Quarterly", "Y": "Yearly",
    "W": "Weekly",  "D": "Daily",     "H": "Hourly",
}

# Медиана дней → метка частоты
_DAYS_TO_FREQ = [
    (1.5,   "Hourly"),
    (2.5,   "Daily"),
    (9,     "Weekly"),
    (35,    "Monthly"),
    (100,   "Quarterly"),
    (1e9,   "Yearly"),
]


def _days_to_freq_label(median_days: float) -> str:
    for threshold, label in _DAYS_TO_FREQ:
        if median_days <= threshold:
            return label
    return "Yearly"


def build_meta(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Строит таблицу мета-данных по каждому ряду:
      unique_id | freq | n_obs | stratum

    Определение частоты (в порядке приоритета):
      1. Колонка 'freq' в df (если загружено через load_m3_multigroup)
      2. Префикс unique_id (M4-стиль: M1234 → Monthly)
      3. Медиана разниц дат (векторизованно — быстро)
    """
    if verbose:
        print("  Вычисление мета-данных (частота, длина)...", end=" ", flush=True)

    all_ids = df["unique_id"].unique()
    lengths = df.groupby("unique_id", sort=False).size().rename("n_obs")

    # 1. Колонка freq уже есть?
    if "freq" in df.columns:
        freq_series = (
            df.groupby("unique_id", sort=False)["freq"]
            .first()
            .rename("freq")
        )
    else:
        # 2. Пробуем по префиксу ID
        prefix_freq = pd.Series(
            {uid: _M4_PREFIX.get(str(uid)[0].upper()) for uid in all_ids},
            name="freq",
        )
        has_prefix = prefix_freq.notna()

        if has_prefix.all():
            freq_series = prefix_freq
        else:
            # 3. Векторизованный расчёт по датам
            # Сортируем один раз
            df_sorted = df.sort_values(["unique_id", "ds"])
            ds = pd.to_datetime(df_sorted["ds"])
            uid_col = df_sorted["unique_id"]

            # diff внутри каждой группы
            same_group = uid_col == uid_col.shift(1)
            day_diffs = ds.diff().dt.days.where(same_group)

            medians = (
                day_diffs.groupby(uid_col, sort=False)
                .median()
                .rename("median_days")
            )
            freq_from_dates = medians.apply(_days_to_freq_label).rename("freq")

            # Объединяем: где есть префикс — берём его, иначе по датам
            freq_series = prefix_freq.where(has_prefix, freq_from_dates)

    meta = pd.DataFrame({"freq": freq_series, "n_obs": lengths}).reset_index()
    meta.columns = ["unique_id", "freq", "n_obs"]
    meta["freq"] = meta["freq"].fillna("Unknown")
    meta["length_bucket"] = pd.cut(
        meta["n_obs"],
        bins=[0, 50, 150, 500, np.inf],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str)
    meta["stratum"] = meta["freq"] + "__" + meta["length_bucket"]

    if verbose:
        freq_counts = meta["freq"].value_counts()
        parts = ", ".join(f"{f}={n}" for f, n in freq_counts.items())
        print(f"OK  ({parts})")

    return meta


# ── Стратифицированная выборка ────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    max_total: int = 1000,
    seed: int = 42,
    min_per_stratum: int = 5,
    verbose: bool = True,
) -> list:
    """
    Стратифицированная выборка unique_id из long-format датафрейма.

    Стратифицирует по: частоте × длине ряда.
    Квота из каждой страты пропорциональна её размеру,
    но не менее min_per_stratum (если в страте есть столько рядов).

    Parameters
    ----------
    df          : long format DataFrame (unique_id, ds, y)
    max_total   : максимальное число рядов в выборке
    seed        : random seed
    min_per_stratum : гарантированный минимум из каждой страты
    verbose     : выводить ли таблицу статистики

    Returns
    -------
    list of unique_id (len ≤ max_total)
    """
    rng = np.random.default_rng(seed)
    all_ids = df["unique_id"].unique()
    n_total = len(all_ids)

    if verbose:
        print(f"  Всего рядов в датасете: {n_total}")

    # Если всё влезает — берём все
    if n_total <= max_total:
        if verbose:
            print(f"  ✅ Берём все {n_total} рядов (≤ порога {max_total})")
        return list(all_ids)

    if verbose:
        print(f"  📊 {n_total} > {max_total} → стратифицированная выборка")

    meta = build_meta(df, verbose=verbose)

    stratum_counts = meta["stratum"].value_counts()
    selected_ids = []
    remaining = max_total

    # Обходим страты от наименьшей к наибольшей
    # (чтобы маленькие страты гарантированно получили min_per_stratum)
    for stratum, count in stratum_counts.sort_values().items():
        if remaining <= 0:
            break
        # Пропорциональная квота
        proportional = int(max_total * count / n_total)
        quota = max(min(min_per_stratum, count), proportional)
        quota = min(quota, count, remaining)

        candidates = meta.loc[meta["stratum"] == stratum, "unique_id"].values
        chosen = rng.choice(candidates, size=quota, replace=False)
        selected_ids.extend(chosen.tolist())
        remaining -= quota

    # Добираем остаток из самых длинных рядов (ещё не выбранных)
    if remaining > 0:
        already = set(selected_ids)
        pool = (
            meta[~meta["unique_id"].isin(already)]
            .sort_values("n_obs", ascending=False)["unique_id"]
            .values
        )
        selected_ids.extend(pool[:remaining].tolist())

    # Перемешиваем (чтобы порядок обработки был случайным)
    rng.shuffle(selected_ids)

    if verbose:
        _print_sample_stats(meta, selected_ids, n_total)

    return selected_ids


def _print_sample_stats(meta: pd.DataFrame, selected_ids: list, n_total: int):
    """Выводит таблицу распределения частот в популяции и выборке."""
    n_sample = len(selected_ids)
    sample_meta = meta[meta["unique_id"].isin(selected_ids)]

    pop_freq   = meta["freq"].value_counts()
    samp_freq  = sample_meta["freq"].value_counts()
    all_freqs  = sorted(pop_freq.index)

    print(f"\n  Выборка {n_sample} из {n_total} рядов:")
    print(f"  {'Частота':<13} {'Популяция':>10}  {'Выборка':>8}  {'Δ%':>6}")
    print(f"  {'─'*42}")
    for freq in all_freqs:
        p_n  = pop_freq.get(freq, 0)
        s_n  = samp_freq.get(freq, 0)
        p_pct = p_n / n_total * 100
        s_pct = s_n / n_sample * 100 if n_sample else 0
        delta = s_pct - p_pct
        sign  = "+" if delta >= 0 else ""
        print(f"  {freq:<13} {p_n:>6} ({p_pct:4.1f}%)  "
              f"{s_n:>4} ({s_pct:4.1f}%)  {sign}{delta:+.1f}%")
    print(f"  {'─'*42}")
    print(f"  {'ИТОГО':<13} {n_total:>6} (100%)  {n_sample:>4} (100%)")

    len_dist = sample_meta["n_obs"].describe()
    print(f"\n  Длина рядов в выборке: "
          f"min={len_dist['min']:.0f}, "
          f"median={len_dist['50%']:.0f}, "
          f"max={len_dist['max']:.0f}")

    # Страты
    n_strata = meta["stratum"].nunique()
    n_covered = sample_meta["stratum"].nunique()
    print(f"  Страт покрыто: {n_covered}/{n_strata}")


def research_series_ids(
    df: pd.DataFrame,
    max_total: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """Точка входа для --mode research."""
    return stratified_sample(df, max_total=max_total, seed=seed, verbose=verbose)
