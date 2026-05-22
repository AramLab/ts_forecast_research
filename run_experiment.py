"""
run_experiment.py — параллельный запуск всех экспериментов.

Модели (24+):
  Базовые:       ARIMA, ETS, Prophet, LSTM
  CEEMDAN:       CEEMDAN+ARIMA, CEEMDAN+ETS, CEEMDAN+Prophet, CEEMDAN+LSTM
  Wavelet A:     Wavelet(1/2/3)+ARIMA, Wavelet(1/2/3)+ETS
  Wavelet B:     ARIMA+Wavelet(1/2), ETS+Wavelet(1/2)
  STL-Base:      STL+ARIMA(trend)+Naive, STL+ETS(trend)+Naive
  STL-A:         STL+ARIMA(trend)+Wavelet, STL+ETS(trend)+Wavelet
  STL-B:         STL+Wavelet(trend)+ARIMA, STL+Wavelet(trend)+ETS
  Transformer:   Transformer(Wavelet,4), Transformer(CEEMDAN,4)

Датасеты:
  synthetic — 4 длинных ряда (N=1440), Trend Break × 2 + Complex Season × 2
  m3        — M3 Monthly (TSF-файл)
  m4        — M4 Monthly (скачивается автоматически)

Использование:
  python run_experiment.py --dataset synthetic --no_plots
  python run_experiment.py --dataset m3 --n_series 50 --workers 5 --no_plots
  python run_experiment.py --dataset m4 --n_series 50 --workers 5 --no_plots

  # Два датасета одновременно:
  python run_experiment.py --dataset m3 --workers 4 --out results/m3 &
  python run_experiment.py --dataset m4 --workers 4 --out results/m4 &
  wait

Параллелизм (macOS M3 Pro):
  spawn-метод: дочерние процессы стартуют чисто, без fork.
  Segfault в PyEMD изолирован в дочернем процессе.
  OMP_NUM_THREADS=1 внутри каждого процесса — нет конкуренции за ядра.
  Рекомендуется --workers 5 (N_perf_cores - 1).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd

# ── Подавление шума ДО всех импортов ─────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",   "3")
for _lg in ["cmdstanpy", "prophet", "tensorflow", "keras",
            "statsforecast", "numba", "PyEMD"]:
    logging.getLogger(_lg).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# АРГУМЕНТЫ
# ══════════════════════════════════════════════════════════════════════════════

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--dataset",   default="m4",
                   choices=["synthetic", "m3", "m4"])
    p.add_argument("--m3_tsf",    default="m3/datasets/m3_monthly_dataset.tsf")
    p.add_argument("--m3_group",  default="Monthly",
                   choices=["Monthly", "Quarterly", "Yearly", "Other"],
                   help="Группа M3")
    p.add_argument("--m4_group",  default="Monthly",
                   choices=["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                   help="Группа M4")
    p.add_argument("--n_series",  type=int, default=50,
                   help="Число рядов (для synthetic игнорируется)")
    p.add_argument("--min_length", type=int, default=0,
                   help="Мин. длина ряда (0 = авто)")
    p.add_argument("--horizon",   type=int, default=0,
                   help="Горизонт h (0 = авто: 18)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--workers",   type=int, default=4,
                   help="Параллельных процессов (рекомендуется 5 для M3 Pro)")
    p.add_argument("--timeout",   type=int, default=600,
                   help="Таймаут на один ряд в секундах")
    p.add_argument("--no_ceemdan",  action="store_true")
    p.add_argument("--no_wavelet",  action="store_true")
    p.add_argument("--no_lstm",     action="store_true",
                   help="Пропустить LSTM и CEEMDAN+LSTM")
    p.add_argument("--no_prophet",  action="store_true",
                   help="Пропустить Prophet и CEEMDAN+Prophet")
    p.add_argument("--no_transformer", action="store_true",
                   help="Пропустить Transformer-модели")
    p.add_argument("--no_plots",  action="store_true",
                   help="HTML-отчёт без графиков")
    p.add_argument("--top3_plots", action="store_true",
                   help="Сохранять лучший базовый vs лучший гибрид (один рисунок 2×1 на ряд)")
    p.add_argument("--ceemdan_trials", type=int, default=30)
    p.add_argument("--n_wavelet_modes", type=int, default=3,
                   help="Макс. число вейвлет-мод (1-3, default 3)")
    p.add_argument("--wavelet",   default="db4")
    p.add_argument("--out",       default="",
                   help="Папка для результатов (default: results/<dataset>)")
    p.add_argument("--length", type=int, default=600,
               help="Длина синтетических рядов (для --dataset synthetic)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# РАБОЧАЯ ФУНКЦИЯ (отдельный процесс на каждый ряд)
# ══════════════════════════════════════════════════════════════════════════════

def _worker(task: tuple) -> dict:
    import signal
    signal.signal(signal.SIGSEGV, lambda s, f: sys.exit(1))
    """
    Прогнозирует один ряд всеми моделями через runner.forecast_series.
    Запускается в отдельном spawn-процессе.
    Любой краш (включая segfault PyEMD) изолирован и не убивает соседей.
    """
    (sid, values, h, m, cfg) = task

    # Заглушки внутри процесса
    import os, warnings, logging
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    for _lg in ["cmdstanpy", "prophet", "tensorflow", "keras",
                "statsforecast", "numba", "PyEMD"]:
        logging.getLogger(_lg).setLevel(logging.ERROR)

    import numpy as np
    import pandas as pd
    import sys
    sys.path.insert(0, cfg["project_dir"])

    # Строим pd.Series с DatetimeIndex и правильной частотой
    idx = pd.date_range(start="1900-01", periods=len(values), freq="MS")
    series = pd.Series(values.astype(float), index=idx, name=sid)

    result: dict = {"Series_ID": sid, "N": len(values)}

    try:
        from runner import forecast_series

        metrics_df, _ = forecast_series(
            series,
            title=sid,
            test_size=h,
            use_ceemdan=cfg["use_ceemdan"],
            use_wavelet=cfg["use_wavelet"],
            use_transformer=cfg["use_transformer"],
            use_ceemdan_lstm=cfg["use_ceemdan_lstm"],
            wavelet=cfg["wavelet"],
            n_modes=cfg["n_wavelet_modes"],
            ceemdan_trials=cfg["ceemdan_trials"],
            plots_dir=None,
            show_plots=False,
            verbose=False,
        )

        # Разворачиваем metrics_df в плоский словарь
        for model_name, row in metrics_df.iterrows():
            # убираем суффикс [fallback] для чистоты колонок
            clean = model_name.replace("[fallback]", "").strip()
            result[clean]            = round(float(row["sMAPE (%)"]), 4)
            result[clean + "_MASE"]  = round(float(row["MASE"]),     4)

        # Подавляем дополнительно Prophet/LSTM если флаги выставлены
        if cfg["no_prophet"]:
            for k in list(result.keys()):
                if "Prophet" in k and k != "Series_ID":
                    del result[k]
        if cfg["no_lstm"]:
            for k in list(result.keys()):
                if "LSTM" in k and k != "Series_ID":
                    del result[k]

    except Exception as e:
        import traceback
        result["fatal_error"] = traceback.format_exc()[:500]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def _load_tasks(args, h: int, project_dir: str) -> list[tuple]:
    cfg = {
        "project_dir":    project_dir,
        "use_ceemdan":    not args.no_ceemdan,
        "use_wavelet":    not args.no_wavelet,
        "use_transformer": not args.no_transformer,
        "use_ceemdan_lstm": not args.no_lstm,
        "no_prophet":     args.no_prophet,
        "no_lstm":        args.no_lstm,
        "wavelet":        args.wavelet,
        "n_wavelet_modes": args.n_wavelet_modes,
        "ceemdan_trials": args.ceemdan_trials,
    }

    if args.dataset == "synthetic":
        from data.synthetic import get_all_synthetic
        series_dict = get_all_synthetic(n=args.length)
        tasks = []
        # Важно: test_size = args.horizon (уже 24)
        for name, s in series_dict.items():
            tasks.append((name, s.values.astype(float), args.horizon, 12, cfg))
        print(f"  Синтетических рядов: {len(tasks)}, каждый N={len(tasks[0][1])}")
        return tasks

    elif args.dataset == "m3":
        from data.loaders import load_m3, load_m3_tsf, prepare_series
        if args.m3_group == "Monthly" and args.m3_tsf:
            df = load_m3_tsf(args.m3_tsf)
            if df is None:
                print("  ⚠ fallback: загружаем M3 Monthly через datasetsforecast")
                df = load_m3(group="Monthly")
        else:
            df = load_m3(group=args.m3_group)
        if df is None:
            sys.exit("❌ M3 не загружен. Проверьте интернет или путь к файлу")
        min_len = args.min_length or 100
        counts  = df.groupby("unique_id")["y"].count()
        valid   = counts[counts >= min_len + h].index.tolist()
        valid.sort(key=lambda sid: -counts[sid])
        rng      = np.random.default_rng(args.seed)
        selected = rng.choice(valid, size=min(args.n_series, len(valid)),
                              replace=False).tolist()
        print(f"  M3: выбрано {len(selected)} рядов (N={counts[selected].min()}–{counts[selected].max()})")
        tasks = []
        for sid in selected:
            s = prepare_series(df, sid, "M3", min_length=h * 2)
            if s is not None:
                tasks.append((sid, s.values.astype(float), h, 12, cfg))
        return tasks

    else:  # m4
        from data.loaders import load_m4, prepare_series
        df = load_m4(group=args.m4_group)
        if df is None:
            sys.exit("❌ M4 не загружен. Проверьте интернет или запустите --dataset m3")
        min_len = args.min_length or 500
        counts  = df.groupby("unique_id")["y"].count()
        valid   = counts[counts >= min_len + h].index.tolist()
        valid.sort(key=lambda sid: -counts[sid])
        # Берём срез из первых 2000 длинных рядов для разнообразия
        pool = valid[:min(2000, len(valid))]
        rng  = np.random.default_rng(args.seed)
        selected = rng.choice(pool, size=min(args.n_series, len(pool)),
                              replace=False).tolist()
        print(f"  M4: выбрано {len(selected)} рядов (N={counts[selected].min()}–{counts[selected].max()})")
        tasks = []
        for sid in selected:
            s = prepare_series(df, sid, "M4", min_length=h * 2)
            if s is not None:
                tasks.append((sid, s.values.astype(float), h, 12, cfg))
        return tasks


# ══════════════════════════════════════════════════════════════════════════════
# HTML-ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

# Все модели в порядке диплома
_ALL_MODELS = [
    "ARIMA", "ETS", "Prophet", "LSTM",
    "CEEMDAN+ARIMA", "CEEMDAN+ETS", "CEEMDAN+Prophet", "CEEMDAN+LSTM",
    "Wavelet(1)+ARIMA", "Wavelet(1)+ETS",
    "Wavelet(2)+ARIMA", "Wavelet(2)+ETS",
    "Wavelet(3)+ARIMA", "Wavelet(3)+ETS",
    "ARIMA+Wavelet(1)", "ETS+Wavelet(1)",
    "ARIMA+Wavelet(2)", "ETS+Wavelet(2)",
    "STL+ARIMA(trend)+Naive(season)", "STL+ETS(trend)+Naive(season)",
    "STL+ARIMA(trend)+Wavelet(season)", "STL+ETS(trend)+Wavelet(season)",
    "STL+Wavelet(trend)+ARIMA(season)", "STL+Wavelet(trend)+ETS(season)",
    "Transformer(Wavelet,4)", "Transformer(CEEMDAN,4)",
]
_BASE = {"ARIMA", "ETS", "Prophet", "LSTM"}


def _save_html(df: pd.DataFrame, out_dir: str, dataset: str,
               no_plots: bool, h: int):
    cols = [c for c in _ALL_MODELS if c in df.columns]
    means = {c: df[c].dropna().mean() for c in cols}
    means = dict(sorted(means.items(), key=lambda x: x[1]))

    best_base   = min((v for k, v in means.items() if k in _BASE),
                      default=float("inf"))
    best_hybrid = min((v for k, v in means.items() if k not in _BASE),
                      default=float("inf"))

    # ── сводная таблица ───────────────────────────────────────────────────────
    def _row_bg(i):
        return ["#fff9e6", "#f0f9f0", "#f0f4fb",
                "#fff", "#f8f8f8"][i % 5]

    summary_html = ""
    for i, (model, val) in enumerate(means.items()):
        is_base = model in _BASE
        medal   = ["🥇 ", "🥈 ", "🥉 "][i] if i < 3 else ""
        tag     = "[base]" if is_base else "<b>[hybrid]</b>"
        gain    = ""
        if not is_base and val < best_base:
            gain = f' <span style="color:#27ae60">✅ лучше на {best_base-val:.2f} п.п.</span>'
        n_ok    = int(df[model].dropna().count())
        summary_html += (
            f'<tr style="background:{_row_bg(i)}">'
            f"<td>{medal}{model} {tag}</td>"
            f"<td><b>{val:.2f}%</b>{gain}</td>"
            f"<td>{df[model].dropna().std():.2f}%</td>"
            f"<td>{n_ok}</td></tr>"
        )

    # ── детали по рядам ───────────────────────────────────────────────────────
    det_head = "".join(
        f'<th style="background:{"#922b21" if m in _BASE else "#1a5276"}'
        f';white-space:nowrap">{m}</th>'
        for m in means
    )
    det_rows = ""
    for _, row in df.sort_values("N", ascending=False).iterrows():
        valid = {m: row[m] for m in means if pd.notna(row.get(m))}
        if not valid:
            continue
        best_m = min(valid, key=valid.get)
        cells  = (f'<td><b>{row["Series_ID"]}</b></td>'
                  f'<td>{int(row["N"])}</td>')
        for m in means:
            v = row.get(m)
            if pd.isna(v) or v is None:
                cells += "<td>—</td>"
            else:
                clr = ("#27ae60" if m == best_m and m not in _BASE else
                       "#c0392b" if m == best_m else "")
                st = f'style="color:{clr};font-weight:bold"' if clr else ""
                cells += f"<td {st}>{v:.1f}%</td>"
        det_rows += f"<tr>{cells}</tr>"

    # ── вердикт ───────────────────────────────────────────────────────────────
    hybrid_wins = sum(
        min((row.get(k, np.inf) for k in means if k not in _BASE),
            default=np.inf)
        < min((row.get(k, np.inf) for k in means if k in _BASE),
              default=np.inf)
        for _, row in df.iterrows()
    )
    win_pct = 100 * hybrid_wins / max(len(df), 1)
    if best_hybrid < best_base:
        vcol    = "#27ae60"
        best_hname = min((k for k in means if k not in _BASE), key=means.get)
        best_bname = min((k for k in means if k in _BASE),     key=means.get)
        verdict = (f"✅ Гибрид <b>{best_hname}</b> ({best_hybrid:.2f}%) "
                   f"лучше базовой <b>{best_bname}</b> ({best_base:.2f}%)")
    else:
        vcol    = "#e67e22"
        verdict = (f"ℹ Базовые модели лучше ({best_base:.2f}% vs "
                   f"гибрид {best_hybrid:.2f}%)")

    # ── опциональный chart ────────────────────────────────────────────────────
    chart_html = ""
    if not no_plots and len(means) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt, base64
            from io import BytesIO

            fig, ax = plt.subplots(figsize=(12, max(4, len(means) * 0.35)))
            colors = ["#e74c3c" if m in _BASE else "#2980b9"
                      for m in means.keys()]
            bars = ax.barh(list(means.keys()), list(means.values()),
                           color=colors, alpha=0.85)
            ax.set_xlabel("Средний sMAPE (%)")
            ax.set_title(
                f"{dataset.upper()}: {len(df)} рядов, h={h}\n"
                "Красный = базовые, Синий = гибриды", fontsize=10)
            ax.invert_yaxis()
            for bar, val in zip(bars, means.values()):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}%", va="center", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            chart_html = (
                f'<img src="data:image/png;base64,{b64}"'
                ' style="width:100%;max-width:960px">'
            )
        except Exception:
            pass

    n_min = int(df["N"].min()) if len(df) else "?"
    n_max = int(df["N"].max()) if len(df) else "?"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{dataset.upper()} — все модели</title>
<style>
  body  {{ font-family:'Segoe UI',Arial,sans-serif; max-width:1400px;
          margin:0 auto; padding:20px; background:#f4f6f9; color:#2c3e50; }}
  h1   {{ border-bottom:3px solid #2980b9; padding-bottom:8px; }}
  h2   {{ color:#2980b9; margin-top:26px; }}
  .box  {{ background:white; border-radius:8px; padding:18px;
          margin-bottom:20px; box-shadow:0 2px 6px rgba(0,0,0,.07); }}
  .note {{ background:#eaf4fb; border-left:4px solid #2980b9;
          padding:10px 14px; border-radius:4px; font-size:.93em; }}
  .verdict {{ font-size:1.1em; padding:12px 16px; border-radius:6px;
             border-left:5px solid {vcol}; background:#fafafa; }}
  table {{ border-collapse:collapse; width:100%; font-size:12px; }}
  th    {{ background:#2c3e50; color:white; padding:6px 8px; }}
  td    {{ padding:5px 8px; border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#f0f4f8; }}
  .scroll {{ overflow-x:auto; }}
</style>
</head>
<body>
<h1>📊 {dataset.upper()} — гибридные vs базовые (все модели)</h1>

<div class="note">
  Рядов: <b>{len(df)}</b> | N: {n_min}–{n_max} | h={h} |
  Без графиков: {"да" if no_plots else "нет"}
</div>

<div class="box">
  <h2>🏆 Вердикт</h2>
  <div class="verdict">{verdict}</div>
  <p style="margin-top:10px;font-size:.9em">
    Гибрид лучше базового: <b>{hybrid_wins}/{len(df)} рядов ({win_pct:.0f}%)</b>
  </p>
</div>

<div class="box">
  <h2>📈 Средний sMAPE по моделям</h2>
  {chart_html}
  <div style="overflow-x:auto;margin-top:14px">
  <table>
    <thead>
      <tr><th>Модель</th><th>sMAPE (ср.)</th>
          <th>sMAPE (ст. откл.)</th><th>N рядов</th></tr>
    </thead>
    <tbody>{summary_html}</tbody>
  </table>
  </div>
</div>

<div class="box">
  <h2>📋 Детали по каждому ряду</h2>
  <p style="font-size:.85em;margin-bottom:8px">
    🟢 зелёный = гибрид победил | 🔴 красный = базовая победила
  </p>
  <div class="scroll">
  <table>
    <thead>
      <tr><th>Series ID</th><th>N</th>{det_head}</tr>
    </thead>
    <tbody>{det_rows}</tbody>
  </table>
  </div>
</div>
</body></html>"""

    path = os.path.join(out_dir, f"report_{dataset}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📄 HTML → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ПЕЧАТЬ ПРОГРЕССА
# ══════════════════════════════════════════════════════════════════════════════

def _print_row(row: dict, done: int, total: int, elapsed: float):
    sid  = row.get("Series_ID", "?")
    N    = row.get("N", "?")
    err  = row.get("fatal_error")

    if err:
        print(f"  [{done:>3}/{total}] {sid:<16} N={N}  ❌ {err[:80]}")
        return

    vals = {m: row[m] for m in _ALL_MODELS if pd.notna(row.get(m))}
    if not vals:
        print(f"  [{done:>3}/{total}] {sid:<16} N={N}  — все модели упали")
        return

    best  = min(vals, key=vals.get)
    tag   = "[base]" if best in _BASE else "[hybrid]"
    print(f"  [{done:>3}/{total}] {sid:<16} N={N:<5} "
          f"→ {best} {tag}  sMAPE={vals[best]:.2f}%  "
          f"(+{elapsed:.0f}s)")

    for m in _ALL_MODELS:
        v = row.get(m)
        if pd.notna(v):
            mase = row.get(m + "_MASE")
            ms   = f"  MASE={mase:.3f}" if pd.notna(mase) else ""
            print(f"    {m:<45} sMAPE={v:7.2f}%{ms}")


def _print_summary(df: pd.DataFrame, dataset: str):
    cols  = [c for c in _ALL_MODELS if c in df.columns]
    means = {c: df[c].dropna().mean() for c in cols}
    means = dict(sorted(means.items(), key=lambda x: x[1]))

    print(f"\n{'='*65}")
    print(f"  ИТОГ {dataset.upper()}: {len(df)} рядов")
    print(f"{'─'*65}")

    best_base   = min((v for k, v in means.items() if k in _BASE),
                      default=float("inf"))
    best_hybrid = min((v for k, v in means.items() if k not in _BASE),
                      default=float("inf"))

    for model, val in means.items():
        tag   = "[base]  " if model in _BASE else "[hybrid]"
        star  = " ← лучший гибрид" if model not in _BASE and val == best_hybrid else ""
        arrow = ">>>" if val == min(means.values()) else "   "
        print(f"  {arrow} {tag} {model:<45} {val:.2f}%{star}")

    wins = sum(
        min((row.get(k, np.inf) for k in means if k not in _BASE), default=np.inf)
        < min((row.get(k, np.inf) for k in means if k in _BASE),   default=np.inf)
        for _, row in df.iterrows()
    )
    print(f"\n  Гибрид лучше базовой: {wins}/{len(df)} рядов "
          f"({100*wins/max(len(df),1):.0f}%)")
    if best_hybrid < best_base:
        print(f"  ✅ Лучший гибрид выигрывает на {best_base-best_hybrid:.2f} п.п.")
    else:
        print(f"  ℹ Базовые лучше в среднем на {best_hybrid-best_base:.2f} п.п.")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # spawn обязателен на macOS: безопасно для C-расширений PyEMD, pywt
    mp.set_start_method("spawn", force=True)

    args = _parse()

    # Авто-параметры
    if args.horizon == 0:
        args.horizon = 24

    out_dir = args.out or f"results/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*65}")
    print(f"  Датасет: {args.dataset.upper()} | h={args.horizon} | "
          f"workers={args.workers}")
    print(f"{'='*65}")

    # Загрузка
    tasks = _load_tasks(args, args.horizon, project_dir)
    if not tasks:
        sys.exit("Нет рядов. Проверьте параметры.")

    total  = len(tasks)
    rows   = []
    done   = 0
    t_glob = time.time()

    print(f"\n  Запуск {total} рядов на {args.workers} процессах...\n")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        fmap = {pool.submit(_worker, t): t for t in tasks}

        for future in as_completed(fmap):
            task = fmap[future]
            sid  = task[0]
            t0   = time.time()
            try:
                row = future.result(timeout=args.timeout)
            except FuturesTimeout:
                row = {"Series_ID": sid, "N": len(task[1]),
                       "fatal_error": f"TIMEOUT {args.timeout}s"}
                print(f"  [TIMEOUT] {sid}")
            except Exception as e:
                row = {"Series_ID": sid, "N": len(task[1]),
                       "fatal_error": str(e)[:200]}
                print(f"  [CRASH]   {sid}: {e}")

            rows.append(row)
            done += 1
            _print_row(row, done, total, time.time() - t_glob)

    df = pd.DataFrame(rows)
    _print_summary(df, args.dataset)

    # CSV
    csv_path = os.path.join(out_dir, f"metrics_{args.dataset}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ CSV → {csv_path}")

    # HTML
    _save_html(df, out_dir, args.dataset, args.no_plots, args.horizon)

    print(f"\n  Общее время: {time.time()-t_glob:.1f}s\n")


if __name__ == "__main__":
    main()
