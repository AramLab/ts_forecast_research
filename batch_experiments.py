#!/usr/bin/env python3
"""
batch_experiments.py

Запускает серию экспериментов на синтетических рядах с разными длинами и горизонтами,
собирает метрики в единый CSV/HTML для анализа.
"""

import subprocess
import pandas as pd
import os
import time
from datetime import datetime

# Параметры для перебора
LENGTHS = [300, 600, 1200]          # длина ряда
HORIZONS = [12, 24, 36]             # горизонт прогноза (месяцы)
WORKERS = 4
DATASET = "synthetic"

# Корневая папка для результатов
BASE_RESULTS_DIR = "results/batch"

def run_single_experiment(length, horizon):
    """Запускает один эксперимент через run_experiment.py"""
    out_dir = os.path.join(BASE_RESULTS_DIR, f"L{length}_H{horizon}")
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", "run_experiment.py",
        "--dataset", DATASET,
        "--length", str(length),
        "--horizon", str(horizon),
        "--workers", str(WORKERS),
        "--no_plots",           # отключаем графики для скорости
        "--out", out_dir,
    ]
    
    print(f"\n🚀 Запуск: length={length}, horizon={horizon} → {out_dir}")
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"✅ Завершено за {elapsed:.1f}с")
    
    # Читаем итоговый CSV с метриками
    csv_path = os.path.join(out_dir, f"metrics_{DATASET}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["length"] = length
        df["horizon"] = horizon
        return df
    else:
        print(f"⚠️ Файл {csv_path} не найден")
        return None
    
def plot_results(pivot_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Для лучшей базовой и лучшего гибрида
    baseline_models = ["ARIMA", "ETS", "Prophet", "LSTM"]
    hybrid_models = [m for m in pivot_df["Model"].unique() if m not in baseline_models]
    
    best_base = pivot_df[pivot_df["Model"].isin(baseline_models)].groupby(["length", "horizon"])["sMAPE (%)"].min().reset_index()
    best_hybrid = pivot_df[pivot_df["Model"].isin(hybrid_models)].groupby(["length", "horizon"])["sMAPE (%)"].min().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in [("Базовая", best_base), ("Гибрид", best_hybrid)]:
        ax.plot(data["length"], data["sMAPE (%)"], marker='o', label=label)
    ax.set_xlabel("Длина ряда")
    ax.set_ylabel("Лучший sMAPE (%)")
    ax.set_title("Влияние длины ряда на ошибку прогноза (горизонт=24)")
    ax.legend()
    plt.savefig(os.path.join(BASE_RESULTS_DIR, "length_impact.png"))

def main():
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    all_results = []
    
    for length in LENGTHS:
        for horizon in HORIZONS:
            df = run_single_experiment(length, horizon)
            if df is not None:
                all_results.append(df)
    
    # Объединяем все результаты
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(BASE_RESULTS_DIR, "all_experiments.csv"), index=False)
        
        # Создаём сводную таблицу: средний sMAPE по (length, horizon, Model)
        pivot = combined.groupby(["length", "horizon", "Model"])["sMAPE (%)"].mean().reset_index()
        pivot.to_csv(os.path.join(BASE_RESULTS_DIR, "summary_pivot.csv"), index=False)
        
        # Выводим топ-5 моделей для каждой комбинации
        print("\n" + "="*80)
        print("СВОДНАЯ ТАБЛИЦА (средний sMAPE, %):")
        for (L, H), group in pivot.groupby(["length", "horizon"]):
            print(f"\n📌 length={L}, horizon={H}")
            top = group.nsmallest(5, "sMAPE (%)")
            for _, row in top.iterrows():
                print(f"   {row['Model']:<40} {row['sMAPE (%)']:.2f}%")
        print("="*80)
    else:
        print("❌ Нет результатов для объединения")

if __name__ == "__main__":
    main()