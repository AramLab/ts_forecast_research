"""
analysis/report.py
Генерация HTML-отчётов:
  generate_dataset_report()  — отдельный HTML на каждый датасет
  generate_summary_report()  — сводный HTML по всем датасетам
"""
import os
import base64
from datetime import datetime
from pathlib import Path
import pandas as pd


CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #2c3e50; }
.header { background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
          color: white; padding: 30px 40px; margin-bottom: 0; }
.header h1 { font-size: 1.8em; font-weight: 600; }
.header .meta { font-size: 0.9em; opacity: 0.8; margin-top: 6px; }
.tabs { display: flex; background: #2c3e50; padding: 0 40px; gap: 4px; flex-wrap: wrap; }
.tab { padding: 12px 20px; cursor: pointer; color: rgba(255,255,255,0.7);
       font-size: 0.9em; border-bottom: 3px solid transparent; transition: all 0.2s; }
.tab:hover { color: white; }
.tab.active { color: white; border-bottom-color: #3498db; font-weight: 600; }
.tab-content { display: none; padding: 30px 40px; }
.tab-content.active { display: block; }
.container { max-width: 1400px; margin: 0 auto; }
.card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 24px; overflow: hidden; }
.card-header { padding: 16px 20px; background: #f8f9fa; border-bottom: 1px solid #e9ecef;
               font-weight: 600; font-size: 1em; color: #34495e; }
.card-body { padding: 20px; }
table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
th { background: #3498db; color: white; padding: 10px 14px; text-align: left; font-weight: 500; }
td { padding: 9px 14px; border-bottom: 1px solid #f0f0f0; }
tr:hover td { background: #f8fbff; }
tr.best td { background: #e8f5e9; font-weight: 600; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; }
.badge-blue { background: #e3f2fd; color: #1565c0; }
.badge-green { background: #e8f5e9; color: #2e7d32; }
.badge-orange { background: #fff3e0; color: #e65100; }
.img-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; }
.img-card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; overflow: hidden; }
.img-card img { width: 100%; display: block; }
.img-cap { padding: 8px 12px; font-size: 0.8em; color: #666; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }
.stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             color: white; border-radius: 8px; padding: 16px; }
.stat-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.stat-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.stat-card.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.stat-val { font-size: 1.8em; font-weight: 700; }
.stat-lbl { font-size: 0.8em; opacity: 0.85; margin-top: 4px; }
.warn { background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px;
        border-radius: 4px; font-size: 0.9em; color: #5d4037; }
</style>
<script>
function showTab(groupId, tabId) {
  document.querySelectorAll('[data-group="'+groupId+'"]').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('[data-tab="'+groupId+'-'+tabId+'"]').forEach(el => el.classList.add('active'));
  document.querySelectorAll('#tab-'+groupId+'-'+tabId).forEach(el => el.classList.add('active'));
}
</script>
"""


def _b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _table(df: pd.DataFrame, best_row: str = None) -> str:
    """DataFrame → HTML таблица с подсветкой лучшей строки."""
    rows = ""
    for idx, row in df.iterrows():
        cls = ' class="best"' if str(idx) == str(best_row) else ""
        cells = "".join(f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                        for v in row)
        rows += f"<tr{cls}><td><b>{idx}</b></td>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr><th>Модель</th>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _images_html(plots_dir: str, prefix_filter: str = "") -> str:
    if not os.path.isdir(plots_dir):
        return "<p class='warn'>Графики не найдены</p>"
    pngs = sorted(Path(plots_dir).glob("*.png"))
    if prefix_filter:
        pngs = [p for p in pngs if prefix_filter.lower() in p.stem.lower()]
    if not pngs:
        return "<p class='warn'>Нет PNG файлов в папке</p>"
    cards = ""
    for png in pngs:
        b64 = _b64(str(png))
        if b64:
            cards += f"""<div class="img-card">
              <img src="data:image/png;base64,{b64}" loading="lazy">
              <div class="img-cap">{png.stem}</div>
            </div>"""
    return f'<div class="img-grid">{cards}</div>'


def _model_metrics_table(df: pd.DataFrame) -> str:
    """Агрегированная таблица метрик по моделям."""
    agg = (
        df.groupby("Model")
        .agg(
            sMAPE_mean=("sMAPE (%)", "mean"),
            sMAPE_std=("sMAPE (%)", "std"),
            sMAPE_min=("sMAPE (%)", "min"),
            RMSE_mean=("RMSE", "mean"),
            MAE_mean=("MAE", "mean"),
            MASE_mean=("MASE", "mean"),
            N=("Series_ID", "count"),
        )
        .round(3)
        .sort_values("sMAPE_mean")
    )
    best = agg.index[0]
    return _table(agg, best_row=best)


# ── Отдельный HTML на один датасет ───────────────────────────────────────────

def generate_dataset_report(
    dataset_name: str,
    summary_df: pd.DataFrame,
    plots_dir: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    n_series   = summary_df["Series_ID"].nunique()
    n_models   = summary_df["Model"].nunique()
    best_model = summary_df.groupby("Model")["sMAPE (%)"].mean().idxmin()
    best_smape = summary_df.groupby("Model")["sMAPE (%)"].mean().min()

    # Группы моделей для вкладок
    model_groups = {
        "Базовые":          ["ARIMA", "ETS", "Prophet", "LSTM"],
        "CEEMDAN":          ["CEEMDAN+"],
        "Вейвлет (A)":      ["Wavelet("],
        "Вейвлет (B)":      ["+Wavelet("],
        "Трансформер":      ["Transformer("],
    }

    def in_group(model_name, keys):
        return any(k in str(model_name) for k in keys)

    tabs_nav = ""
    tabs_content = ""
    for gi, (gname, keys) in enumerate(model_groups.items()):
        group_df = summary_df[summary_df["Model"].apply(lambda m: in_group(m, keys))]
        if group_df.empty:
            continue
        active = "active" if gi == 0 else ""
        gid = f"ds_{dataset_name}_g{gi}"
        tabs_nav += f'<div class="tab {active}" data-group="{gid}" data-tab="{gid}-0" onclick="showTab(\'{gid}\',0)">{gname}</div>'
        tbl = _model_metrics_table(group_df)
        pngs = _images_html(plots_dir)
        tabs_content += f"""
        <div class="tab-content {active}" id="tab-{gid}-0">
          <div class="card"><div class="card-header">Средние метрики — {gname}</div>
            <div class="card-body">{tbl}</div></div>
          <div class="card"><div class="card-header">Графики</div>
            <div class="card-body">{pngs}</div></div>
        </div>"""

    # Вкладка "Все модели"
    gid_all = f"ds_{dataset_name}_all"
    tabs_nav = f'<div class="tab active" data-group="{gid_all}" data-tab="{gid_all}-0" onclick="showTab(\'{gid_all}\',0)">Все модели</div>' + tabs_nav
    all_tbl = _model_metrics_table(summary_df)
    tabs_content = f"""
    <div class="tab-content active" id="tab-{gid_all}-0">
      <div class="stat-grid" style="margin-bottom:20px">
        <div class="stat-card blue"><div class="stat-val">{n_series}</div><div class="stat-lbl">Рядов</div></div>
        <div class="stat-card green"><div class="stat-val">{n_models}</div><div class="stat-lbl">Моделей</div></div>
        <div class="stat-card orange"><div class="stat-val">{best_smape:.1f}%</div><div class="stat-lbl">Лучший sMAPE</div></div>
        <div class="stat-card"><div class="stat-val" style="font-size:1em">{best_model}</div><div class="stat-lbl">Лучшая модель</div></div>
      </div>
      <div class="card"><div class="card-header">Все модели — средние метрики по {n_series} рядам</div>
        <div class="card-body">{all_tbl}</div></div>
      <div class="card"><div class="card-header">Графики</div>
        <div class="card-body">{_images_html(plots_dir)}</div></div>
    </div>""" + tabs_content

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<title>{dataset_name} — Анализ временных рядов</title>
{CSS}
</head><body>
<div class="header">
  <h1>📊 {dataset_name} — Сравнение моделей прогнозирования</h1>
  <div class="meta">Сгенерировано: {now} · Рядов: {n_series} · Моделей: {n_models}</div>
</div>
<div class="tabs">{tabs_nav}</div>
<div class="container">{tabs_content}</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ {output_path}")


# ── Сводный HTML по всем датасетам ───────────────────────────────────────────

def generate_summary_report(
    summaries: dict,
    plots_dir: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    non_empty = {k: v for k, v in summaries.items() if v is not None and not v.empty}

    tabs_nav, tabs_content = "", ""
    for gi, (ds_name, df) in enumerate(non_empty.items()):
        active = "active" if gi == 0 else ""
        gid = f"sum_g{gi}"
        best_model = df.groupby("Model")["sMAPE (%)"].mean().idxmin()
        best_smape = df.groupby("Model")["sMAPE (%)"].mean().min()
        n_series   = df["Series_ID"].nunique()
        tabs_nav += f'<div class="tab {active}" data-group="sum" data-tab="sum-{gi}" onclick="showTab(\'sum\',{gi})">{ds_name}</div>'
        tbl = _model_metrics_table(df)
        ds_plots = os.path.join(plots_dir, ds_name)
        pngs = _images_html(ds_plots if os.path.isdir(ds_plots) else plots_dir)
        tabs_content += f"""
        <div class="tab-content {active}" id="tab-sum-{gi}">
          <div class="stat-grid" style="margin-bottom:20px">
            <div class="stat-card blue"><div class="stat-val">{n_series}</div><div class="stat-lbl">Рядов</div></div>
            <div class="stat-card green"><div class="stat-val">{best_smape:.1f}%</div><div class="stat-lbl">Лучший sMAPE</div></div>
            <div class="stat-card"><div class="stat-val" style="font-size:1em">{best_model}</div><div class="stat-lbl">Лучшая модель</div></div>
          </div>
          <div class="card"><div class="card-header">{ds_name} — метрики</div>
            <div class="card-body">{tbl}</div></div>
          <div class="card"><div class="card-header">Графики {ds_name}</div>
            <div class="card-body">{pngs}</div></div>
        </div>"""

    # Общее сравнение
    if len(non_empty) > 1:
        all_df = pd.concat(non_empty.values(), ignore_index=True)
        global_best = all_df.groupby("Model")["sMAPE (%)"].mean().idxmin()
        global_smape = all_df.groupby("Model")["sMAPE (%)"].mean().min()
        tabs_nav = f'<div class="tab" data-group="sum" data-tab="sum-cmp" onclick="showTab(\'sum\',\'cmp\')">📊 Сравнение</div>' + tabs_nav
        comp_tbl = _model_metrics_table(all_df)
        root_pngs = _images_html(plots_dir)
        tabs_content = f"""
        <div class="tab-content" id="tab-sum-cmp">
          <div class="stat-grid" style="margin-bottom:20px">
            <div class="stat-card blue"><div class="stat-val">{all_df['Series_ID'].nunique()}</div><div class="stat-lbl">Всего рядов</div></div>
            <div class="stat-card green"><div class="stat-val">{global_smape:.1f}%</div><div class="stat-lbl">Лучший sMAPE</div></div>
            <div class="stat-card"><div class="stat-val" style="font-size:0.9em">{global_best}</div><div class="stat-lbl">Лучшая модель (всё)</div></div>
          </div>
          <div class="card"><div class="card-header">Сравнение всех датасетов</div>
            <div class="card-body">{comp_tbl}</div></div>
          <div class="card"><div class="card-header">Сводные графики</div>
            <div class="card-body">{root_pngs}</div></div>
        </div>""" + tabs_content

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<title>Сводный отчёт — M3/M4/M5</title>
{CSS}
</head><body>
<div class="header">
  <h1>📊 Сводный отчёт: M3 / M4 / M5</h1>
  <div class="meta">Сгенерировано: {now} · Датасетов: {len(non_empty)}</div>
</div>
<div class="tabs">{tabs_nav}</div>
<div class="container">{tabs_content}</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ {output_path}")


# Обратная совместимость со старым кодом
def generate_html_report(summaries, plots_dir="results/plots", output_path="results/report.html"):
    generate_summary_report(summaries, plots_dir, output_path)
