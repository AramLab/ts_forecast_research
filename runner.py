"""
runner.py — Центральный движок анализа.
"""
import os
import logging
# Ограничение потоков — предотвращает OOM на Mac при параллельных CEEMDAN-вызовах
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional

from analysis.metrics import infer_period
from analysis.plots import (
    plot_forecast, plot_all_forecasts, plot_best_pair,
    plot_dataset_summary, plot_datasets_comparison, plot_heatmap,
)
from analysis.report import generate_dataset_report, generate_summary_report


def _check_components() -> dict:
    status = {}
    import os
    skip_lstm = os.environ.get("NO_LSTM") == "1"
    for pkg, key in [("PyEMD", "ceemdan"), ("tensorflow", "lstm"), ("pywt", "wavelet")]:
        if key == "lstm" and skip_lstm:
            status[key] = False
            continue
        try:
            __import__(pkg)
            status[key] = True
        except ImportError:
            status[key] = False
    return status


def _naive_fallback(series, test_size, model_name):
    from analysis.metrics import calculate_metrics
    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    m = infer_period(series)
    if m > 1 and len(train) >= m:
        last = train.values[-m:]
        fc = np.tile(last, (test_size // m) + 2)[:test_size]
    else:
        fc = np.full(test_size, float(train.iloc[-1]))
    metrics = calculate_metrics(test.values, fc, train.values, m)
    metrics["Model"] = f"{model_name}[fallback]"
    return pd.Series(fc, index=test.index), metrics


def forecast_series(
    series: pd.Series,
    title: str,
    test_size: int = 18,
    use_ceemdan: bool = True,
    use_wavelet: bool = True,
    use_transformer: bool = True,
    use_ceemdan_lstm: bool = True,
    wavelet: str = "db4",
    n_modes: int = 4,
    ceemdan_trials: int = 50,
    plots_dir: Optional[str] = None,
    show_plots: bool = False,
    top3_plots: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Прогнозирует ряд всеми доступными моделями.
    Возвращает (metrics_df, forecasts_dict).
    """
    status = _check_components()
    if verbose:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"  Длина: {len(series)}, тест: {test_size}, m={infer_period(series)}")
        print(f"{'='*65}")

    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]
    metrics_list, forecasts = [], {}

    def run(name, fn, *a, **kw):
        t0 = time.time()
        if verbose:
            print(f"  ▶ {name}...", end=" ", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred, met = fn(*a, **kw)
            elapsed = time.time() - t0
            if verbose:
                print(f"sMAPE={met['sMAPE (%)']:.1f}%  ({elapsed:.0f}s)")
            if (plots_dir or show_plots) and not top3_plots:
                plot_forecast(
                    train, test, pred.values, title=title,
                    model_name=name, smape_val=met["sMAPE (%)"],
                    save_dir=plots_dir, show=show_plots,
                )
            forecasts[name] = pred.values
            met["Model"] = name
            metrics_list.append(met)
        except Exception as e:
            elapsed = time.time() - t0
            # Специальная обработка для CEEMDAN-гибридов, у которых все IMF упали в fallback
            if isinstance(e, RuntimeError) and "все IMF упали в fallback" in str(e).lower():
                if verbose:
                    print(f"❌ {name} НЕРАБОТОСПОСОБНА (все IMF в fallback): {e}  ({elapsed:.0f}s)")
                # Не добавляем модель в результаты – просто выходим
                return
            # Для всех остальных ошибок – стандартный наивный fallback
            if verbose:
                print(f"❌ {e}  ({elapsed:.0f}s)")
            pred, met = _naive_fallback(series, test_size, name)
            forecasts[name] = pred.values
            metrics_list.append(met)

    # ── 1-4: Базовые ────────────────────────────────────────────────────────
    from models.arima_model   import arima_forecast
    from models.ets_model     import ets_forecast
    from models.prophet_model import prophet_forecast

    run("ARIMA",   arima_forecast,   series, test_size=test_size)
    run("ETS",     ets_forecast,     series, test_size=test_size)
    run("Prophet", prophet_forecast, series, test_size=test_size)

    if status["lstm"]:
        from models.lstm_model import lstm_forecast
        run("LSTM", lstm_forecast, series, test_size=test_size)

    # ── 5-8: CEEMDAN-гибриды ────────────────────────────────────────────────
    if use_ceemdan and status["ceemdan"]:
        from ceemdan.ceemdan_hybrid import (
            ceemdan_arima, ceemdan_ets, ceemdan_prophet,
            ceemdan_es, ceemdan_lstm_true,   # добавляем настоящую LSTM
            _get_imfs, _IMF_CACHE,
        )
        # Прогреваем кеш: декомпозиция ОДИН раз для всех гибридов
        try:
            if verbose:
                print(f"  ▶ CEEMDAN декомпозиция...", end=" ", flush=True)
            import time as _t; _t0 = _t.time()
            _get_imfs(series.values[:-test_size], trials=ceemdan_trials, noise_width=0.05)
            if verbose:
                print(f"{_t.time()-_t0:.0f}s (кеш готов для 4 гибридов)")
        except Exception as _e:
            if verbose:
                print(f"❌ {_e}")

        _ckw = {"ceemdan_trials": ceemdan_trials}
        run("CEEMDAN+ARIMA",   ceemdan_arima,   series, test_size=test_size, **_ckw)
        run("CEEMDAN+ETS",     ceemdan_ets,     series, test_size=test_size, **_ckw)
        run("CEEMDAN+Prophet", ceemdan_prophet, series, test_size=test_size, **_ckw)
        run("CEEMDAN+ES",      ceemdan_es,      series, test_size=test_size, **_ckw)   # чистая ES
        if status["lstm"] and use_ceemdan_lstm:
            # НАСТОЯЩИЙ CEEMDAN+LSTM (LSTM для каждой IMF)
            run("CEEMDAN+LSTM", ceemdan_lstm_true, series, test_size=test_size, **_ckw)
        elif not use_ceemdan_lstm and verbose:
            print("  ⚡ CEEMDAN+LSTM пропущен (--low_memory)")
        from ceemdan.ceemdan_hybrid import _IMF_CACHE
        _IMF_CACHE.clear()
        import gc
        gc.collect()

    # ── 9-16: Вейвлет-гибриды ───────────────────────────────────────────────
    if use_wavelet and status["wavelet"]:
        from models.wavelet.wavelet_hybrid import wavelet_arima, wavelet_ets

        # Режим A: вейвлет = тренд (1, 2, 3 моды)
        for n in [1, 2, 3]:
            run(f"Wavelet({n})+ARIMA", wavelet_arima,
                series, test_size=test_size, mode="A", n_modes=n, wavelet=wavelet)
            run(f"Wavelet({n})+ETS",   wavelet_ets,
                series, test_size=test_size, mode="A", n_modes=n, wavelet=wavelet)

        # Режим B: ARIMA/ETS = тренд, вейвлет = сезонность
        for n in [1, 2]:
            run(f"ARIMA+Wavelet({n})", wavelet_arima,
                series, test_size=test_size, mode="B", n_modes=n, wavelet=wavelet)
            run(f"ETS+Wavelet({n})",   wavelet_ets,
                series, test_size=test_size, mode="B", n_modes=n, wavelet=wavelet)

    # ── STL-гибриды ─────────────────────────────────────────────────────────
    if use_wavelet and status["wavelet"]:
        from models.stl_hybrid import (
            stl_arima_base, stl_ets_base,
            stl_arima_wavelet_season, stl_ets_wavelet_season,
            stl_wavelet_arima_season, stl_wavelet_ets_season,
        )
        _wkw = {"wavelet": wavelet, "n_wavelet_modes": 2}

        # STL-Base: ARIMA/ETS на тренде, наивный на сезонности (baseline)
        run("STL+ARIMA(trend)+Naive(season)", stl_arima_base,
            series, test_size=test_size)
        run("STL+ETS(trend)+Naive(season)",   stl_ets_base,
            series, test_size=test_size)

        # STL-A: ARIMA/ETS на тренде, вейвлет на сезонности
        run("STL+ARIMA(trend)+Wavelet(season)", stl_arima_wavelet_season,
            series, test_size=test_size, **_wkw)
        run("STL+ETS(trend)+Wavelet(season)",   stl_ets_wavelet_season,
            series, test_size=test_size, **_wkw)

        # STL-B: вейвлет на тренде, ARIMA/ETS на сезонности (модели поменяны)
        run("STL+Wavelet(trend)+ARIMA(season)", stl_wavelet_arima_season,
            series, test_size=test_size, **_wkw)
        run("STL+Wavelet(trend)+ETS(season)",   stl_wavelet_ets_season,
            series, test_size=test_size, **_wkw)

    # ── 17-18: Трансформер на модах ─────────────────────────────────────────
    if use_transformer and status["lstm"]:
        from models.transformer.mode_transformer import mode_transformer_forecast

        run("Transformer(Wavelet,4)",  mode_transformer_forecast,
            series, test_size=test_size, source="wavelet",
            n_modes=n_modes, wavelet=wavelet)
        if use_ceemdan and status["ceemdan"]:
            run("Transformer(CEEMDAN,4)", mode_transformer_forecast,
                series, test_size=test_size, source="ceemdan", n_modes=n_modes)

    # ── Итоговый датафрейм ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    metrics_df = metrics_df[["RMSE", "MAE", "sMAPE (%)", "MASE"]].sort_values("sMAPE (%)")

    if len(forecasts) > 1 and (plots_dir or show_plots):
        if top3_plots:
            plot_best_pair(
                train, test, forecasts, metrics_df, title=title,
                save_dir=plots_dir, show=show_plots,
            )
        else:
            plot_all_forecasts(
                train, test, forecasts, metrics_df, title=title,
                save_dir=plots_dir, show=show_plots,
            )

    if verbose:
        print(f"  {'─'*50}")
        print(f"  Лучшая: {metrics_df.index[0]}  sMAPE={metrics_df['sMAPE (%)'].iloc[0]:.1f}%")

    return metrics_df, forecasts


def analyze_dataset(
    df,
    dataset_name,
    series_ids,
    test_size=18,
    use_ceemdan=True,
    use_wavelet=True,
    use_transformer=True,
    use_ceemdan_lstm=True,
    wavelet="db4",
    n_modes=4,
    ceemdan_trials=50,
    results_dir="results",
    show_plots=False,
    save_plots=True,
    top3_plots=False,
    checkpoint_every=50,
) -> Optional[pd.DataFrame]:
    """
    Запускает анализ по списку рядов одного датасета.
    Поддерживает: прогресс-бар, ETA, автосохранение промежуточных результатов.
    """
    from data.loaders import prepare_series
    from data.progress import ProgressTracker

    ds_plots_dir = os.path.join(results_dir, "plots", dataset_name) if save_plots else None
    if ds_plots_dir:
        os.makedirs(ds_plots_dir, exist_ok=True)

    all_results = []
    total = len(series_ids)
    csv_path = os.path.join(results_dir, f"metrics_{dataset_name.lower()}.csv")

    print(f"\n{'='*65}")
    print(f"  АНАЛИЗ {dataset_name}: {total} рядов, горизонт={test_size}")
    print(f"  Чекпоинт каждые {checkpoint_every} рядов → {csv_path}")
    print(f"{'='*65}\n")

    progress = ProgressTracker(total=total, dataset_name=dataset_name)

    for i, sid in enumerate(series_ids, 1):
        t_row = time.time()
        print(f"\n[{i}/{total}] {dataset_name} — {sid}")

        series = prepare_series(df, sid, dataset_name, min_length=test_size * 2)
        if series is None:
            print("  ⚠ Слишком короткий ряд, пропускаем")
            progress.update(elapsed_sec=0, skipped=True)
            continue

        try:
            metrics_df, _ = forecast_series(
                series,
                title=f"{dataset_name}_{sid}",
                test_size=test_size,
                use_ceemdan=use_ceemdan,
                use_wavelet=use_wavelet,
                use_transformer=use_transformer,
                use_ceemdan_lstm=use_ceemdan_lstm,
                wavelet=wavelet,
                n_modes=n_modes,
                ceemdan_trials=ceemdan_trials,
                plots_dir=ds_plots_dir,
                show_plots=show_plots,
                top3_plots=top3_plots,
                verbose=True,
            )
            elapsed = time.time() - t_row
            for model_name, row in metrics_df.iterrows():
                all_results.append({
                    "Dataset":    dataset_name,
                    "Series_ID":  sid,
                    "Length":     len(series),
                    "Model":      model_name,
                    "RMSE":       row["RMSE"],
                    "MAE":        row["MAE"],
                    "sMAPE (%)":  row["sMAPE (%)"],
                    "MASE":       row["MASE"],
                })
            progress.update(elapsed_sec=elapsed, success=True)

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            progress.update(elapsed_sec=time.time() - t_row, success=False)

        # Промежуточное сохранение
        if all_results and (i % checkpoint_every == 0 or i == total):
            _checkpoint_save(all_results, csv_path, i, total)

    progress.summary()

    if not all_results:
        print(f"  ⚠ Нет результатов для {dataset_name}")
        return None

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Сохранено: {csv_path}  ({len(result_df)} строк)")

    # Сводка по моделям
    print(f"\n  Средний sMAPE ({dataset_name}):")
    means = result_df.groupby("Model")["sMAPE (%)"].mean().sort_values()
    for model, val in means.items():
        trophy = "🏆" if model == means.index[0] else "  "
        print(f"  {trophy} {model:<40} {val:.2f}%")

    return result_df


def _checkpoint_save(results: list, path: str, done: int, total: int):
    """Сохраняет промежуточный CSV."""
    try:
        pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  💾 Чекпоинт сохранён: {done}/{total} → {path}")
    except Exception as e:
        print(f"\n  ⚠ Не удалось сохранить чекпоинт: {e}")


def run_full_analysis(
    datasets,
    series_ids_map,
    test_size=18,
    use_ceemdan=True,
    use_wavelet=True,
    use_transformer=True,
    use_ceemdan_lstm=True,
    wavelet="db4",
    n_modes=4,
    ceemdan_trials=50,
    results_dir="results",
    show_plots=False,
    save_plots=True,
    top3_plots=False,
    checkpoint_every=50,
) -> dict:
    """
    Запускает анализ по всем датасетам, строит графики и HTML-отчёты.
    Возвращает {dataset_name: summary_df}.
    """
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summaries = {}

    for ds_name, df in datasets.items():
        if df is None:
            continue
        ids = series_ids_map.get(ds_name, [])
        if not ids:
            continue

        ds_plots = os.path.join(plots_dir, ds_name)
        os.makedirs(ds_plots, exist_ok=True)

        summary = analyze_dataset(
            df, ds_name, ids,
            test_size=test_size,
            use_ceemdan=use_ceemdan,
            use_wavelet=use_wavelet,
            use_transformer=use_transformer,
            use_ceemdan_lstm=use_ceemdan_lstm,
            wavelet=wavelet,
            n_modes=n_modes,
            ceemdan_trials=ceemdan_trials,
            results_dir=results_dir,
            show_plots=show_plots,
            save_plots=save_plots,
            top3_plots=top3_plots,
            checkpoint_every=checkpoint_every,
        )
        summaries[ds_name] = summary

        if summary is not None:
            plot_dataset_summary(summary, ds_name, save_dir=ds_plots, show=False)
            plot_heatmap(summary, "sMAPE (%)", ds_name, save_dir=ds_plots, show=False)
            generate_dataset_report(
                dataset_name=ds_name,
                summary_df=summary,
                plots_dir=ds_plots,
                output_path=os.path.join(results_dir, f"report_{ds_name}.html"),
            )

    non_empty = {k: v for k, v in summaries.items() if v is not None}

    if len(non_empty) > 1:
        plot_datasets_comparison(non_empty, save_dir=plots_dir, show=False)

    if non_empty:
        all_df = pd.concat(non_empty.values(), ignore_index=True)
        all_df.to_csv(os.path.join(results_dir, "summary_all.csv"),
                      index=False, encoding="utf-8-sig")
        generate_summary_report(
            summaries=non_empty,
            plots_dir=plots_dir,
            output_path=os.path.join(results_dir, "report_SUMMARY.html"),
        )

    print(f"\n{'='*65}")
    print(f"  ГОТОВО → {results_dir}/")
    try:
        html_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".html"))
        for h in html_files:
            print(f"  📄 {h}")
    except Exception:
        pass
    print(f"{'='*65}")

    return summaries


def run_synthetic(
    test_size: int = 24,
    use_ceemdan: bool = True,
    use_wavelet: bool = True,
    use_transformer: bool = False,
    wavelet: str = "db4",
    n_modes: int = 4,
    ceemdan_trials: int = 30,
    results_dir: str = "results/synthetic",
    show_plots: bool = False,
    save_plots: bool = True,
    top3_plots: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Запускает все модели на синтетических рядах (задание научного руководителя).

    Два типа рядов:
      1. Trend Break  — ряд со скачком тренда
      2. Complex Season — ряд со сложной сезонностью (несколько некратных циклов)

    Возвращает сводный DataFrame с метриками по всем рядам и моделям.
    """
    from data.synthetic import get_all_synthetic

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots") if save_plots else None
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    all_rows = []
    summary_forecasts = {}

    series_dict = get_all_synthetic()

    for name, series in series_dict.items():
        print(f"\n{'='*65}")
        print(f"  Синтетический ряд: {name}")
        print(f"  Длина: {len(series)}, тест: {test_size}")
        print(f"{'='*65}")

        ser_plots = os.path.join(plots_dir, name.replace(" ", "_").replace("(", "").replace(")", "")) if plots_dir else None
        if ser_plots:
            os.makedirs(ser_plots, exist_ok=True)

        metrics_df, forecasts = forecast_series(
            series=series,
            title=name,
            test_size=test_size,
            use_ceemdan=use_ceemdan,
            use_wavelet=use_wavelet,
            use_transformer=use_transformer,
            use_ceemdan_lstm=False,
            wavelet=wavelet,
            n_modes=n_modes,
            ceemdan_trials=ceemdan_trials,
            plots_dir=ser_plots,
            show_plots=show_plots,
            top3_plots=top3_plots,
            verbose=verbose,
        )

        for model_name, row in metrics_df.iterrows():
            all_rows.append({
                "Series": name,
                "Model": model_name,
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "sMAPE (%)": row["sMAPE (%)"],
                "MASE": row["MASE"],
            })

        summary_forecasts[name] = (series, metrics_df, forecasts)

    # Сводная таблица
    summary_df = pd.DataFrame(all_rows)

    # Сохраняем CSV
    csv_path = os.path.join(results_dir, "metrics_synthetic.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Сводная таблица: средний sMAPE по всем рядам
    pivot = summary_df.groupby("Model")["sMAPE (%)"].mean().sort_values()

    print(f"\n{'='*65}")
    print("  ИТОГИ по синтетическим рядам (средний sMAPE):")
    print(f"{'─'*65}")
    for model, val in pivot.items():
        bar = "█" * int(val / 2)
        print(f"  {model:<42} {val:6.1f}%  {bar}")
    print(f"{'='*65}")
    print(f"  Результаты: {csv_path}")

    # HTML-отчёт
    _write_synthetic_html(summary_df, summary_forecasts, results_dir, test_size)

    return summary_df


def _write_synthetic_html(
    summary_df: pd.DataFrame,
    summary_forecasts: dict,
    results_dir: str,
    test_size: int,
) -> None:
    """Генерирует HTML-отчёт по синтетическим рядам."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    def fig_to_b64(fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    blocks = []

    for name, (series, metrics_df, forecasts) in summary_forecasts.items():
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # График: топ-5 моделей
        top5 = metrics_df.head(5).index.tolist()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index, train.values, color="black", lw=1.5, label="Train")
        ax.plot(test.index, test.values, color="gray", lw=2, ls="--", label="Test (факт)")
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        for model, color in zip(top5, colors):
            if model in forecasts:
                ax.plot(test.index, forecasts[model], color=color, lw=1.5,
                        label=f"{model} ({metrics_df.loc[model, 'sMAPE (%)']:.1f}%)")
        ax.set_title(name, fontsize=13)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        img_b64 = fig_to_b64(fig)

        # Таблица метрик
        table_rows = ""
        for i, (model, row) in enumerate(metrics_df.iterrows()):
            bg = "#fef9e7" if i == 0 else ("white" if i % 2 == 0 else "#f8f9fa")
            medal = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else ""))
            table_rows += (
                f'<tr style="background:{bg}">'
                f'<td>{medal}{model}</td>'
                f'<td>{row["RMSE"]:.2f}</td>'
                f'<td>{row["MAE"]:.2f}</td>'
                f'<td><b>{row["sMAPE (%)"]:.1f}%</b></td>'
                f'<td>{row["MASE"]:.3f}</td>'
                f'</tr>'
            )

        blocks.append(f"""
        <div class="series-block">
          <h2>{name}</h2>
          <img src="data:image/png;base64,{img_b64}" style="width:100%;max-width:900px">
          <table>
            <thead>
              <tr><th>Модель</th><th>RMSE</th><th>MAE</th><th>sMAPE</th><th>MASE</th></tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
        """)

    # Итоговая сводка
    pivot = summary_df.groupby("Model")["sMAPE (%)"].mean().sort_values()
    pivot_rows = ""
    for i, (model, val) in enumerate(pivot.items()):
        bg = "#fef9e7" if i == 0 else ("white" if i % 2 == 0 else "#f8f9fa")
        medal = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else ""))
        pivot_rows += (
            f'<tr style="background:{bg}">'
            f'<td>{medal}{model}</td>'
            f'<td><b>{val:.1f}%</b></td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Синтетические ряды — результаты</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px;
         margin: 0 auto; padding: 20px; background: #f5f6fa; color: #2c3e50; }}
  h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
  h2   {{ color: #2980b9; margin-top: 30px; }}
  .series-block {{ background: white; border-radius: 10px; padding: 20px;
                   margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
  th    {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  .summary {{ background: white; border-radius: 10px; padding: 20px;
              margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
</style>
</head>
<body>
<h1>🧪 Синтетические ряды — сравнение моделей</h1>
<div class="summary">
  <h2>📊 Сводная таблица (средний sMAPE по всем рядам)</h2>
  <table>
    <thead><tr><th>Модель</th><th>Средний sMAPE (%)</th></tr></thead>
    <tbody>{pivot_rows}</tbody>
  </table>
</div>
{''.join(blocks)}
</body>
</html>"""

    out = os.path.join(results_dir, "report_synthetic.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📄 HTML-отчёт: {out}")
