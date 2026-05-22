"""
analysis/plots.py
Функции визуализации прогнозов и сравнения моделей.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Optional

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11


# ── Палитра моделей ──────────────────────────────────────────────────────────
MODEL_COLORS = {
    "ARIMA":           "#4C72B0",
    "ETS":             "#55A868",
    "Prophet":         "#C44E52",
    "LSTM":            "#8172B2",
    "CEEMDAN+ARIMA":   "#CCB974",
    "CEEMDAN+ETS":     "#64B5CD",
    "CEEMDAN+Prophet": "#DD8452",
    "CEEMDAN+LSTM":    "#937860",
}


def _model_color(model_name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if key in str(model_name):
            return color
    return "#888888"


# ── Одиночный прогноз ────────────────────────────────────────────────────────
def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: np.ndarray,
    title: str,
    model_name: str,
    smape_val: float,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """Визуализация одного прогноза."""
    fig, ax = plt.subplots(figsize=(13, 5))

    train_idx = train.index
    test_idx = test.index

    # Рисуем фактические значения как одну непрерывную линию (train + test),
    # затем накладываем подсветку обучающей части.
    try:
        if len(test_idx) > 0:
            full_idx = np.concatenate([train_idx, test_idx])
            full_vals = np.concatenate([train.values, test.values])
        else:
            full_idx = train_idx
            full_vals = train.values
        ax.plot(full_idx, full_vals, "k-", lw=2, label="Факт", alpha=0.9)
        ax.plot(train_idx, train.values, "b-", lw=2, label="Обучение", alpha=0.7)
    except Exception:
        ax.plot(train_idx, train.values, "b-", lw=2, label="Обучение", alpha=0.7)
        ax.plot(test_idx, test.values, "g-", lw=2, label="Факт", alpha=0.85)
    # Рисуем прогноз как непрерывную линию, начиная с последней точки обучения
    try:
        if len(test_idx) > 0:
            last_train_val = train.values[-1]
            last_train_idx = train_idx[-1]
            if last_train_idx == test_idx[0]:
                ax.plot(
                    test_idx, forecast,
                    linestyle="--", lw=2.5,
                    color=_model_color(model_name),
                    label=f"{model_name}  sMAPE={smape_val:.1f}%",
                )
            else:
                ax.plot(
                    [last_train_idx] + list(test_idx),
                    np.concatenate([[last_train_val], forecast]),
                    linestyle="--", lw=2.5,
                    color=_model_color(model_name),
                    label=f"{model_name}  sMAPE={smape_val:.1f}%",
                )
        else:
            ax.plot(
                test_idx, forecast,
                linestyle="--", lw=2.5,
                color=_model_color(model_name),
                label=f"{model_name}  sMAPE={smape_val:.1f}%",
            )
    except Exception:
        ax.plot(
            test_idx, forecast,
            linestyle="--", lw=2.5,
            color=_model_color(model_name),
            label=f"{model_name}  sMAPE={smape_val:.1f}%",
        )

    # Вертикальная граница обучение/тест
    if len(test) > 0:
        ax.axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.6)
    ax.set_title(f"{model_name} — {title}", fontsize=13)
    ax.set_xlabel("Период")
    ax.set_ylabel("Значение")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{model_name}_{title}".replace(" ", "_").replace("/", "_")[:80]
        plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


# ── Лучший базовый vs лучший гибрид (для диплома) ────────────────────────────
_BASE_MODELS = {"ARIMA", "ETS", "Prophet", "LSTM"}


def plot_best_pair(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict,          # {model_name: np.ndarray}
    metrics_df: pd.DataFrame,
    title: str,
    train_tail: int = 60,     # сколько последних точек обучения показывать
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Рисунок 2×1: верхний подграфик — лучшая базовая модель по sMAPE,
    нижний — лучший гибрид. Создаёт ОДИН файл best_pair_<title>.png.

    Удобно вставлять в диплом: один рисунок с номером, две панели,
    всё на одной странице.
    """
    if metrics_df.empty or not forecasts:
        return

    available = [m for m in metrics_df.index if m in forecasts]
    base_models   = [m for m in available if m in _BASE_MODELS]
    hybrid_models = [m for m in available if m not in _BASE_MODELS]

    if not base_models or not hybrid_models:
        # Если нет пары — рисуем только то, что есть
        candidates = base_models or hybrid_models
        if not candidates:
            return
        selected = candidates[:1]
        labels   = ["Базовая" if candidates is base_models else "Гибрид"]
    else:
        selected = [base_models[0], hybrid_models[0]]
        labels   = ["Лучшая базовая модель", "Лучший гибрид"]

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    # Хвост обучающей выборки — чтобы не рисовать 500+ точек
    tail = min(train_tail, len(train))
    train_tail_data = train.iloc[-tail:]
    train_idx = train_tail_data.index
    test_idx = test.index

    panel_colors = {"Лучшая базовая модель": "#c0392b",
                    "Лучший гибрид":         "#2980b9"}

    for ax, mname, label in zip(axes, selected, labels):
        smape_val = metrics_df.loc[mname, "sMAPE (%)"]
        header_color = panel_colors.get(label, "#555555")

        # Показываем фактическую кривую как непрерывную линию (хвост обучения + тест),
        # и накладываем хвост обучения серым для контраста.
        try:
            if len(test_idx) > 0:
                full_idx = np.concatenate([train_idx, test_idx])
                full_vals = np.concatenate([train_tail_data.values, test.values])
            else:
                full_idx = train_idx
                full_vals = train_tail_data.values
            ax.plot(full_idx, full_vals,
                    color="#2c3e50", lw=2.5, label="Факт", alpha=0.95, zorder=10)
            ax.plot(train_idx, train_tail_data.values,
                    color="#7f8c8d", lw=1.8, label="Обучение (хвост)", alpha=0.7)
        except Exception:
            ax.plot(train_idx, train_tail_data.values,
                    color="#7f8c8d", lw=1.8, label="Обучение (хвост)", alpha=0.7)
            ax.plot(test_idx, test.values,
                    color="#2c3e50", lw=2.5, label="Факт", alpha=0.95, zorder=10)
        # Прогноз — рисуем непрерывно от последней точки обучения
        try:
            last_train = train_tail_data.values[-1]
            last_idx = train_tail_data.index[-1]
            fc = forecasts[mname]
            if len(test_idx) > 0 and last_idx != test_idx[0]:
                ax.plot(
                    [last_idx] + list(test_idx),
                    np.concatenate([[last_train], fc]),
                    linestyle="--", lw=2.5,
                    color=_model_color(mname),
                    label=f"Прогноз  sMAPE = {smape_val:.2f}%",
                    zorder=9,
                )
            else:
                ax.plot(test_idx, fc,
                        linestyle="--", lw=2.5,
                        color=_model_color(mname),
                        label=f"Прогноз  sMAPE = {smape_val:.2f}%",
                        zorder=9)
        except Exception:
            ax.plot(test_idx, forecasts[mname],
                    linestyle="--", lw=2.5,
                    color=_model_color(mname),
                    label=f"Прогноз  sMAPE = {smape_val:.2f}%",
                    zorder=9)

        if len(test_idx) > 0:
            ax.axvline(x=test_idx[0], color="gray", linestyle=":", alpha=0.5, lw=1.2)

        ax.set_title(f"{label}: {mname}  (sMAPE = {smape_val:.2f}%)",
                     fontsize=11, fontweight="bold", color=header_color, loc="left")
        ax.set_xlabel("Период")
        ax.set_ylabel("Значение")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"best_pair_{title}".replace(" ", "_").replace("/", "_")[:90]
        plt.savefig(os.path.join(save_dir, f"{fname}.png"),
                    dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


# ── Сравнение нескольких прогнозов ───────────────────────────────────────────
def plot_all_forecasts(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict,         # {model_name: np.ndarray}
    metrics_df: pd.DataFrame,
    title: str,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """Все прогнозы на одном графике + bar-chart sMAPE."""
    n_models = len(forecasts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Левый: все прогнозы ---
    ax = axes[0]
    train_idx = train.index
    test_idx = test.index

    # Рисуем фактические как непрерывную линию (train + test), затем подсветка train
    try:
        if len(test_idx) > 0:
            full_idx = np.concatenate([train_idx, test_idx])
            full_vals = np.concatenate([train.values, test.values])
        else:
            full_idx = train_idx
            full_vals = train.values
        ax.plot(full_idx, full_vals, "k-", lw=2.5, label="Факт", alpha=0.9, zorder=10)
        ax.plot(train_idx, train.values, "b-", lw=2, label="Обучение", alpha=0.6)
    except Exception:
        ax.plot(train_idx, train.values, "b-", lw=2, label="Обучение", alpha=0.6)
        ax.plot(test_idx, test.values, "k-", lw=2.5, label="Факт", alpha=0.9, zorder=10)

    for mname, fcast in forecasts.items():
        smape_val = metrics_df.loc[mname, "sMAPE (%)"] if mname in metrics_df.index else float("nan")
        try:
            last_train = train.values[-1]
            last_idx = train_idx[-1]
            if len(test_idx) > 0 and last_idx != test_idx[0]:
                ax.plot(
                    [last_idx] + list(test_idx),
                    np.concatenate([[last_train], fcast]),
                    linestyle="--", lw=1.8, alpha=0.8,
                    color=_model_color(mname),
                    label=f"{mname} ({smape_val:.1f}%)",
                )
            else:
                ax.plot(
                    test_idx, fcast,
                    linestyle="--", lw=1.8, alpha=0.8,
                    color=_model_color(mname),
                    label=f"{mname} ({smape_val:.1f}%)",
                )
        except Exception:
            ax.plot(
                test_idx, fcast,
                linestyle="--", lw=1.8, alpha=0.8,
                color=_model_color(mname),
                label=f"{mname} ({smape_val:.1f}%)",
            )

    if len(test_idx) > 0:
        ax.axvline(x=test_idx[0], color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Все прогнозы — {title}", fontsize=12)
    ax.set_xlabel("Период")
    ax.set_ylabel("Значение")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Правый: bar sMAPE ---
    ax2 = axes[1]
    if not metrics_df.empty:
        sorted_df = metrics_df.sort_values("sMAPE (%)")
        colors = [_model_color(m) for m in sorted_df.index]
        bars = ax2.bar(range(len(sorted_df)), sorted_df["sMAPE (%)"], color=colors, alpha=0.85)
        ax2.set_xticks(range(len(sorted_df)))
        ax2.set_xticklabels(sorted_df.index, rotation=40, ha="right", fontsize=9)
        ax2.set_ylabel("sMAPE (%)")
        ax2.set_title(f"Сравнение по sMAPE — {title}", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)
        for b in bars:
            ax2.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.15,
                f"{b.get_height():.1f}",
                ha="center", fontsize=8,
            )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"comparison_{title}".replace(" ", "_").replace("/", "_")[:80]
        plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


# ── Сводный анализ датасета ──────────────────────────────────────────────────
def plot_dataset_summary(
    summary_df: pd.DataFrame,
    dataset_name: str,
    save_dir: Optional[str] = None,
    show: bool = True,
    top_n_models: Optional[int] = None,
) -> None:
    """Средний sMAPE + boxplot по всем рядам датасета."""
    if summary_df is None or summary_df.empty:
        return

    model_means = summary_df.groupby("Model")["sMAPE (%)"].mean().sort_values()
    if top_n_models is not None and top_n_models > 0:
        model_means = model_means.iloc[:top_n_models]
    colors = [_model_color(m) for m in model_means.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart среднего sMAPE
    ax = axes[0]
    bars = ax.bar(range(len(model_means)), model_means.values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(model_means)))
    ax.set_xticklabels(model_means.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Средний sMAPE (%)")
    ax.set_title(
        f"{dataset_name}: Средний sMAPE по моделям"
        + (f" (топ-{len(model_means)})" if top_n_models is not None and top_n_models > 0 else "")
    )
    ax.grid(axis="y", alpha=0.3)
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.1,
            f"{b.get_height():.1f}",
            ha="center", fontsize=8,
        )

    # Boxplot
    ax2 = axes[1]
    order = model_means.index.tolist()
    data_bp = [
        summary_df[summary_df["Model"] == m]["sMAPE (%)"].dropna().values
        for m in order
    ]
    bp = ax2.boxplot(data_bp, labels=order, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("sMAPE (%)")
    ax2.set_title(
        f"{dataset_name}: Распределение sMAPE"
        + (f" (топ-{len(order)})" if top_n_models is not None and top_n_models > 0 else "")
    )
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Анализ {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"summary_{dataset_name}.png"),
            dpi=120, bbox_inches="tight",
        )

    if show:
        plt.show()
    plt.close()


# ── Сравнение нескольких датасетов ──────────────────────────────────────────
def plot_datasets_comparison(
    summaries: dict,  # {dataset_name: summary_df}
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """Сравнение среднего sMAPE между несколькими датасетами."""
    summaries = {k: v for k, v in summaries.items() if v is not None and not v.empty}
    if len(summaries) < 2:
        return

    all_models = sorted(
        set(m for df in summaries.values() for m in df["Model"].unique())
    )

    x = np.arange(len(all_models))
    width = 0.8 / len(summaries)
    palette = plt.cm.Set2(np.linspace(0, 1, len(summaries)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (ds_name, df) in enumerate(summaries.items()):
        means = df.groupby("Model")["sMAPE (%)"].mean()
        vals  = [means.get(m, np.nan) for m in all_models]
        offset = (i - len(summaries) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds_name, alpha=0.85, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Средний sMAPE (%)")
    ax.set_title("Сравнение датасетов: средний sMAPE по моделям", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, "datasets_comparison.png"),
            dpi=120, bbox_inches="tight",
        )

    if show:
        plt.show()
    plt.close()


# ── Тепловая карта метрик ────────────────────────────────────────────────────
def plot_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "sMAPE (%)",
    dataset_name: str = "",
    save_dir: Optional[str] = None,
    show: bool = True,
    top_n_models: Optional[int] = None,
) -> None:
    """Тепловая карта метрики: строки = ряды, столбцы = модели."""
    if summary_df is None or summary_df.empty:
        return

    pivot = summary_df.pivot_table(
        index="Series_ID", columns="Model", values=metric
    )
    if pivot.empty:
        return

    # Сортировка по среднему
    col_order = pivot.mean().sort_values().index
    if top_n_models is not None and top_n_models > 0:
        col_order = col_order[:top_n_models]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), min(20, len(pivot) * 0.4 + 2)))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r",
        annot=(len(pivot) <= 20),
        fmt=".1f", linewidths=0.3,
        cbar_kws={"label": metric},
    )
    ax.set_title(f"{dataset_name}: {metric} по рядам и моделям", fontsize=12)
    ax.set_xlabel("Модель")
    ax.set_ylabel("Ряд")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"heatmap_{dataset_name}_{metric.replace(' ', '_')}.png"),
            dpi=100, bbox_inches="tight",
        )

    if show:
        plt.show()
    plt.close()
