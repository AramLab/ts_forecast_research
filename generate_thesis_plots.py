"""
generate_thesis_plots.py
────────────────────────
Генерирует иллюстрации «лучший базовый vs лучший гибрид» для диплома.

Использование
─────────────
# Синтетика — все 8 рядов при N=600, h=24:
python generate_thesis_plots.py --dataset synthetic --horizon 24

# Синтетика с конкретной длиной ряда:
python generate_thesis_plots.py --dataset synthetic --horizon 12 --length 300
python generate_thesis_plots.py --dataset synthetic --horizon 24 --length 600
python generate_thesis_plots.py --dataset synthetic --horizon 36 --length 1200

# M3/M4 — конкретные ряды (как в таблицах диплома):
python generate_thesis_plots.py --dataset m3 --series_ids M3-M1,M3-M10,M3-M100 --horizon 18
python generate_thesis_plots.py --dataset m4 --series_ids M4-M1,M4-M10,M4-M100 --horizon 18

# M3/M4 — представительные ряды из готового CSV (после массового прогона):
python generate_thesis_plots.py --dataset m3 --csv results/m3/metrics_m3.csv --n_series 3
python generate_thesis_plots.py --dataset m4 --csv results/m4/metrics_m4.csv --n_series 3

# Только сводные графики (bar/boxplot/heatmap) из CSV, без прогона:
python generate_thesis_plots.py --dataset m3 --csv results/m3/metrics_m3.csv --summary_only

Результат кладётся в папку results/thesis_plots/<dataset>/
"""

import os
import sys
import argparse
import warnings
from typing import Optional
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_BASE = {"ARIMA", "ETS", "Prophet", "LSTM"}


def _pick_representative(df: pd.DataFrame, n: int = 3) -> list[str]:
    """
    Выбирает n рядов из CSV метрик:
      - 1 где гибрид выигрывает максимально
      - 1 медианный по разнице
      - 1 где базовая лучше (если есть)
    """
    model_cols = [c for c in df.columns
                  if c not in ("Series_ID", "N", "fatal_error")
                  and not c.endswith("_MASE")]
    base_cols   = [c for c in model_cols if c in _BASE]
    hybrid_cols = [c for c in model_cols if c not in _BASE]

    if not base_cols or not hybrid_cols:
        return df["Series_ID"].tolist()[:n]

    def best_base(row):
        vals = [row[c] for c in base_cols if pd.notna(row.get(c))]
        return min(vals) if vals else np.nan

    def best_hybrid(row):
        vals = [row[c] for c in hybrid_cols if pd.notna(row.get(c))]
        return min(vals) if vals else np.nan

    df = df.copy()
    df["_best_base"]   = df.apply(best_base,   axis=1)
    df["_best_hybrid"] = df.apply(best_hybrid, axis=1)
    df["_gap"]         = df["_best_base"] - df["_best_hybrid"]
    df = df.dropna(subset=["_best_base", "_best_hybrid"])
    if df.empty:
        return []

    chosen = []
    chosen.append(df.loc[df["_gap"].idxmax(), "Series_ID"])

    base_wins = df[df["_gap"] < 0]
    if not base_wins.empty:
        sid = base_wins.loc[base_wins["_gap"].idxmin(), "Series_ID"]
        if sid not in chosen:
            chosen.append(sid)

    remaining = df[~df["Series_ID"].isin(chosen)]
    if not remaining.empty:
        median_gap = remaining["_gap"].median()
        sid = remaining.iloc[(remaining["_gap"] - median_gap).abs().argsort()[:1]]["Series_ID"].values[0]
        if sid not in chosen:
            chosen.append(sid)

    return chosen[:n]


def _load_series(sid: str, dataset: str, horizon: int, project_dir: str):
    """Загружает один ряд из M3 или M4."""
    from data.loaders import prepare_series

    if dataset == "m3":
        # Используем новую функцию load_m3_tsf и передаем аргумент filepath
        from data.loaders import load_m3_tsf
        df = load_m3_tsf(
            filepath=os.path.join(project_dir, "m3", "datasets", "m3_monthly_dataset.tsf")
        )
    elif dataset == "m4":
        # Используем новую функцию load_m4 с указанием группы
        from data.loaders import load_m4
        df = load_m4(group="Monthly")
    else:
        return None

    # Если df загрузился с ошибкой (None), прерываем выполнение
    if df is None:
        return None

    # prepare_series осталась прежней и корректно отфильтрует ряд
    return prepare_series(df, sid, dataset, min_length=horizon * 2)


def _generate_summary_plots(csv_path: str, dataset: str, out_dir: str, top_models: Optional[int] = None):
    """Сводные bar/boxplot/heatmap из готового CSV."""
    from analysis.plots import plot_dataset_summary, plot_heatmap

    df = pd.read_csv(csv_path)

    if "Model" in df.columns and "sMAPE (%)" in df.columns:
        summary = df[["Series_ID", "Model", "sMAPE (%)"]].copy()
    else:
        model_cols = [c for c in df.columns
                      if c not in ("Series_ID", "N", "fatal_error", "length", "horizon")
                      and not c.endswith("_MASE")]
        rows = []
        for _, row in df.iterrows():
            for m in model_cols:
                v = row.get(m)
                if pd.notna(v):
                    rows.append({"Series_ID": row["Series_ID"], "Model": m, "sMAPE (%)": float(v)})
        summary = pd.DataFrame(rows)

    if summary.empty:
        print("  ⚠ Нет данных для сводных графиков")
        return

    os.makedirs(out_dir, exist_ok=True)
    plot_dataset_summary(summary, dataset, save_dir=out_dir, show=False, top_n_models=top_models)
    plot_heatmap(summary, "sMAPE (%)", dataset, save_dir=out_dir, show=False, top_n_models=top_models)
    print(f"  ✅ Сводные графики → {out_dir}/")


def _generate_summary_plots_batch(batch_dir: str, base_out: str, top_models: Optional[int] = None):
    """Сгенерировать summary/heatmap для каждого metrics_synthetic.csv в batch-дереве."""
    if not os.path.isdir(batch_dir):
        print(f"  ⚠ Batch каталог не найден: {batch_dir}")
        return

    entries = sorted(os.listdir(batch_dir))
    found = False
    for entry in entries:
        subdir = os.path.join(batch_dir, entry)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, "metrics_synthetic.csv")
        if not os.path.exists(csv_path):
            continue
        found = True
        out_dir = os.path.join(base_out, entry)
        print(f"\n  Обработка {entry}: {csv_path}")
        top_models_arg = top_models if top_models and top_models > 0 else None
        _generate_summary_plots(csv_path, "synthetic", out_dir, top_models=top_models_arg)
    if not found:
        print(f"  ⚠ В каталоге {batch_dir} не найдено ни одного metrics_synthetic.csv")


def _run_series_plot(sid: str, dataset: str, horizon: int, out_dir: str, project_dir: str, args):
    """Прогоняет один ряд M3/M4 и сохраняет best_pair график."""
    from runner import forecast_series

    print(f"  🔄 {sid} ...")
    series = _load_series(sid, dataset, horizon, project_dir)
    if series is None:
        print(f"  ⚠ Ряд {sid} не найден, пропускаем")
        return

    ser_out = os.path.join(out_dir, sid.replace("/", "_").replace(" ", "_"))
    os.makedirs(ser_out, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_df, forecasts = forecast_series(
            series, title=sid, test_size=horizon,
            use_ceemdan=not args.no_ceemdan,
            use_wavelet=not args.no_wavelet,
            use_transformer=False,
            use_ceemdan_lstm=False,
            ceemdan_trials=args.ceemdan_trials,
            plots_dir=ser_out,
            show_plots=False,
            top3_plots=True,
            verbose=True,
        )

    # Сохраняем метрики и прогнозы в CSV рядом с графиками
    try:
        import pandas as _pd
        metrics_path = os.path.join(ser_out, f"metrics_{sid}.csv")
        metrics_df.reset_index().to_csv(metrics_path, index=False, encoding="utf-8-sig")
        # forecasts -> wide table: ds, actual, <model1>, <model2>, ...
        test_idx = series.index[-horizon:]
        rows = {"ds": list(test_idx), "actual": list(series.iloc[-horizon:].values)}
        for m, arr in forecasts.items():
            rows[m] = list(arr)
        _pd.DataFrame(rows).to_csv(os.path.join(ser_out, f"forecasts_{sid}.csv"), index=False, encoding='utf-8-sig')
        print(f"  ✅ Metrics CSV → {metrics_path}")
        print(f"  ✅ Forecasts CSV → {os.path.join(ser_out, f'forecasts_{sid}.csv')}")
    except Exception as _e:
        print(f"  ⚠ Не удалось сохранить CSV: {_e}")
    print(f"  ✅ best_pair_{sid}.png → {ser_out}/")



def _run_synthetic_combo(length: int, horizon: int, base_out: str, args) -> None:
    """Один прогон синтетики для заданной пары (length, horizon)."""
    from data.synthetic import get_all_synthetic
    from runner import forecast_series

    combo_dir = os.path.join(base_out, f"syn_L{length}_H{horizon}", "plots")
    os.makedirs(combo_dir, exist_ok=True)

    print(f"\n  {'='*55}")
    print(f"  N={length}, h={horizon}  ->  {combo_dir}/")
    print(f"  {'='*55}")

    series_dict = get_all_synthetic(n=length)
    for name, series in series_dict.items():
        plot_title = f"{name}  [N={length}, h={horizon}]"
        ser_out = os.path.join(combo_dir, name.replace(" ", "_").replace(".", ""))
        os.makedirs(ser_out, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_series(
                series, title=plot_title, test_size=horizon,
                use_ceemdan=not args.no_ceemdan,
                use_wavelet=not args.no_wavelet,
                use_transformer=False, use_ceemdan_lstm=False,
                ceemdan_trials=args.ceemdan_trials,
                plots_dir=ser_out, show_plots=False,
                top3_plots=True, verbose=True,
            )
    print(f"  N={length}, h={horizon} gotovo.")


def main():
    p = argparse.ArgumentParser(description="Generator illyustracij dlya diploma")
    p.add_argument("--dataset",        choices=["synthetic", "m3", "m4"], required=True)
    p.add_argument("--csv",            default="",  help="Put k metrics_<dataset>.csv")
    p.add_argument("--series_ids",     default="",  help="Konkretnye ryady: M3-M1,M3-M10")
    p.add_argument("--n_series",       type=int, default=3)
    p.add_argument("--horizon",        type=int, default=18)
    p.add_argument("--length",         type=int, default=600,
                   help="Dlina sinteticheskih ryadov (300/600/1200). Ignoriruetsya pri --all")
    p.add_argument("--all",            action="store_true",
                   help="Sintetika: vse 9 kombinacij (N=300/600/1200 x h=12/24/36)")
    p.add_argument("--batch_dir",      default="",  help="Каталог batch-результатов для синтетики")
    p.add_argument("--out",            default="",  help="Papka vyvoda")
    p.add_argument("--summary_only",   action="store_true")
    p.add_argument("--top_models",      type=int, default=5,
                   help="Количество моделей для summary/heatmap (0 = все)")
    p.add_argument("--ceemdan_trials", type=int, default=30)
    p.add_argument("--no_ceemdan",     action="store_true")
    p.add_argument("--no_wavelet",     action="store_true")
    args = p.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    base_out = args.out or "results/thesis_plots"
    os.makedirs(base_out, exist_ok=True)

    # Sintetika
    if args.dataset == "synthetic":
        top_models = args.top_models if args.top_models and args.top_models > 0 else None
        if args.batch_dir:
            print(f"\n{'='*65}")
            print(f"  Sintetika: batch-каталог {args.batch_dir}")
            print(f"{'='*65}")
            _generate_summary_plots_batch(args.batch_dir, base_out, top_models=top_models)
            return
        if args.csv and os.path.exists(args.csv):
            batch_name = os.path.basename(os.path.dirname(args.csv)) or os.path.splitext(os.path.basename(args.csv))[0]
            out_dir = os.path.join(base_out, batch_name)
            print(f"\n{'='*65}")
            print(f"  Sintetika: summary из {args.csv}")
            print(f"  Вывод -> {out_dir}/")
            print(f"{'='*65}")
            _generate_summary_plots(args.csv, "synthetic", out_dir, top_models=top_models)
            return
        if args.all:
            combos = [(N, h) for N in [300, 600, 1200] for h in [12, 24, 36]]
            print(f"\n{'='*65}")
            print(f"  Sintetika: vse {len(combos)} kombinacij")
            print(f"  Vyvod -> {base_out}/syn_L<N>_H<h>/")
            print(f"{'='*65}")
            for length, horizon in combos:
                _run_synthetic_combo(length, horizon, base_out, args)
        else:
            print(f"\n{'='*65}")
            print(f"  Sintetika: N={args.length}, h={args.horizon}")
            print(f"{'='*65}")
            _run_synthetic_combo(args.length, args.horizon, base_out, args)
        print(f"\n  Vse sinteticheskie grafiki gotovy -> {base_out}/\n")
        return

    # M3 / M4
    out_dir = args.out or os.path.join(base_out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.csv and os.path.exists(args.csv):
        print("  Svodnye grafiki (bar / heatmap)...")
        top_models = args.top_models if args.top_models and args.top_models > 0 else None
        print(f"  Отрисовываются топ-{top_models if top_models else 'all'} моделей")
        _generate_summary_plots(args.csv, args.dataset, out_dir, top_models=top_models)

    if args.summary_only:
        print("\n  Rezhim --summary_only zavershyon.")
        return

    if args.series_ids:
        series_list = [s.strip() for s in args.series_ids.split(",") if s.strip()]
    elif args.csv and os.path.exists(args.csv):
        df_csv = pd.read_csv(args.csv)
        series_list = _pick_representative(df_csv, n=args.n_series)
    else:
        print("  Ukazhite --series_ids ili --csv")
        sys.exit(1)

    print(f"  Ryady dlya illyustracij: {series_list}")
    for sid in series_list:
        _run_series_plot(sid, args.dataset, args.horizon, out_dir, project_dir, args)

    print(f"\n  Gotovo -> {out_dir}/\n")


if __name__ == "__main__":
    main()
