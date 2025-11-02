# -*- coding: utf-8 -*-
"""
Year-type analysis (Q25/Q75 on ANNUAL precipitation) with publication-grade plots
- Classification basis: ANNUAL precipitation (sum of Pr in a calendar year)
- Include temperature metrics (added Tx/Tn for annual and growing season)
- Add annual ET0 (supplemented)
- Color scheme: color-blind friendly for papers
- Outputs:
    - year_type_analysis_Q2575.png  (2x2 panel figure with temperature)
    - year_type_results_Q2575.csv   (table with metrics & year-type including temperature/annual ET0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========================= User settings =========================
INPUT_CSV = r"C:/Users/Hymn/Desktop/aquacrop/output40/Meteo_calc_data_TestProject.csv"
OUTPUT_FIG = "year_type_analysis_Q2575.png"
OUTPUT_CSV = "year_type_results_Q2575.csv"

# Paper-friendly color palette (color-blind safe)
# Orange, Blue, Green  —— 干旱/平水/湿润
COLORS = {
    "Dry Year":   "#D55E00",  # orange
    "Normal Year":"#0072B2",  # blue
    "Wet Year":   "#009E73",  # green
}
MARKERS = {
    "Dry Year":   "o",
    "Normal Year":"s",
    "Wet Year":   "^",
}
YEAR_TYPES = ["Dry Year", "Normal Year", "Wet Year"]

# Seaborn style for papers
sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)

# ================================================================

def convert_csv_to_tab_txt(input_df: pd.DataFrame, output_file: str):
    """（可选）把CSV导出为制表符TXT；不影响分析"""
    header = "Day\tMonth\tYear\tTmin(C)\tTmax(C)\tPrcp(mm)\tEt0(mm)\n"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        for _, row in input_df.iterrows():
            line = (f"{int(row['Day'])}\t"
                    f"{int(row['Month'])}\t"
                    f"{int(row['Year'])}\t"
                    f"{row['Tn']:.1f}\t"
                    f"{row['Tx']:.1f}\t"
                    f"{row['Pr']:.1f}\t"
                    f"{row['ET0']:.1f}\n")
            f.write(line)

def classify_year_types_by_annual_precip(annual_precip: pd.Series):
    """基于全年年降水分位数 Q25/Q75 分三类，返回：分类Series、阈值、各类典型年"""
    q25 = float(annual_precip.quantile(0.25))
    q75 = float(annual_precip.quantile(0.75))

    def _lab(p):
        if p < q25:
            return "Dry Year"
        elif p > q75:
            return "Wet Year"
        else:
            return "Normal Year"

    labels = annual_precip.apply(_lab)

    # 典型年：各类内距该类中位年降水最接近者
    typical_years = {}
    for cat in YEAR_TYPES:
        idx = labels[labels == cat].index
        if len(idx) == 0:
            typical_years[cat] = None
            continue
        target = annual_precip.loc[idx].median()
        sel = (annual_precip.loc[idx] - target).abs().sort_values().index[0]
        typical_years[cat] = int(sel)

    return labels, q25, q75, typical_years

def compute_metrics(df: pd.DataFrame):
    """计算指标（含全年ET0，明确区分全年/生长季）：
       1) 全年：年降水+年ET0+日最高/最低温均值
       2) 生长季(4-10月)：Pr, ET0, Pr/ET0 + 日最高/最低温均值
    """
    # 年份、日期提取
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # ---- (1) 全年指标（新增Annual_ET0）----
    annual_all = df.groupby('Year', as_index=True).agg(
        AnnualPr=('Pr', 'sum'),               # 全年降水量
        Annual_ET0=('ET0', 'sum'),            # 新增：全年总ET0
        AnnualTxMean=('Tx', 'mean'),          # 全年日最高温均值(℃)
        AnnualTnMean=('Tn', 'mean')           # 全年日最低温均值(℃)
    ).round(1)

    # ---- (2) 生长季指标（GS_前缀标注）----
    df_apr_oct = df[(df['Month'] >= 4) & (df['Month'] <= 10)]  # 4-10月为生长季
    grow = df_apr_oct.groupby('Year').agg(
        GS_Pr=('Pr', 'sum'),                     # 生长季降水量
        GS_ET0=('ET0', 'sum'),                   # 生长季ET0
        GS_TxMean=('Tx', 'mean'),                # 生长季日最高温均值(℃)
        GS_TnMean=('Tn', 'mean')                 # 生长季日最低温均值(℃)
    ).round(1)

    # 防零处理和水分平衡指数计算
    grow['GS_ET0'] = grow['GS_ET0'].replace(0, 0.1)
    grow['GS_Rain_ET0_Ratio'] = (grow['GS_Pr'] / grow['GS_ET0']).astype(float).round(2)

    return annual_all, grow

def main():
    # ---------------- Read ----------------
    df = pd.read_csv(INPUT_CSV)
    # ---------------- Metrics -------------
    annual_metrics, grow_metrics = compute_metrics(df)
    annual_precip_all = annual_metrics['AnnualPr']  # 提取全年降水用于分类

    # 以“全年年降水”做分类
    labels, q25, q75, typical_years = classify_year_types_by_annual_precip(annual_precip_all)

    # 合并表（包含气温、全年ET0指标）
    out = grow_metrics.copy()
    # 加入全年指标（含新增的Annual_ET0）
    out['AnnualPr'] = annual_metrics['AnnualPr']
    out['Annual_ET0'] = annual_metrics['Annual_ET0']  # 新增全年ET0列
    out['AnnualTxMean'] = annual_metrics['AnnualTxMean']
    out['AnnualTnMean'] = annual_metrics['AnnualTnMean']
    # 加入年型分类
    out['YearType'] = labels

    # 保存CSV
    out.reset_index().to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # ==================== 绘图（2×2，含气温图表） ====================
    fig = plt.figure(figsize=(15, 11), dpi=300)
    fig.suptitle("Climatological year-type analysis (Q25/Q75 based on ANNUAL precipitation)",
                 fontsize=16, fontweight="bold", y=0.98)

    # ---------- (1) Scatter: Pr (GS) vs ET0 (GS) ----------
    ax1 = plt.subplot(2, 2, 1)
    for yt in YEAR_TYPES:
        sub = out[out['YearType'] == yt]
        ax1.scatter(sub['GS_Pr'], sub['GS_ET0'],
                    s=55, linewidth=0.5, edgecolors='white',
                    c=COLORS[yt], marker=MARKERS[yt], alpha=0.9, label=yt)
    # 典型年标注
    for yt, y in typical_years.items():
        if y is None or y not in out.index:
            continue
        ax1.scatter(out.loc[y, 'GS_Pr'], out.loc[y, 'GS_ET0'],
                    s=160, facecolors='none', edgecolors='black', linewidth=1.8, marker='o', zorder=5)
        ax1.annotate(f"{y}", (out.loc[y, 'GS_Pr'], out.loc[y, 'GS_ET0']),
                     textcoords="offset points", xytext=(6, 6), ha='left', fontsize=9)
    ax1.set_xlabel("Total rainfall in growing season (mm)")
    ax1.set_ylabel("Total ET0 in growing season (mm)")
    ax1.set_title("GS rainfall vs ET0 by year-type")
    ax1.legend(frameon=True, title="Year type")
    ax1.grid(True, alpha=0.25)

    # ---------- (2) Box: Water balance index (GS Pr/ET0) ----------
    ax2 = plt.subplot(2, 2, 2)
    palette = [COLORS[yt] for yt in YEAR_TYPES]
    sns.boxplot(data=out.reset_index(), x='YearType', y='GS_Rain_ET0_Ratio',
                order=YEAR_TYPES, palette=palette, width=0.6, ax=ax2)
    sns.stripplot(data=out.reset_index(), x='YearType', y='GS_Rain_ET0_Ratio',
                  order=YEAR_TYPES, color='black', alpha=0.45, size=3, jitter=True, ax=ax2)
    ax2.axhline(1.0, linestyle='--', color='grey', linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Year type")
    ax2.set_ylabel("Water-balance index (GS Pr / GS ET0)")
    ax2.set_title("Water-balance by year-type")
    ax2.grid(True, axis='y', alpha=0.25)

    # ---------- (3) Bar: Annual precipitation by year ----------
    ax3 = plt.subplot(2, 2, 3)
    ann_df = annual_precip_all.to_frame().reset_index().rename(columns={'index': 'Year'})
    ann_df['YearType'] = labels.values
    ann_df = ann_df.sort_values('Year')
    # 着色柱状
    for yt in YEAR_TYPES:
        sub = ann_df[ann_df['YearType'] == yt]
        ax3.bar(sub['Year'].astype(int), sub['AnnualPr'], color=COLORS[yt], width=0.8, label=yt)
    # 阈值线
    ax3.axhline(q25, linestyle='--', color='grey', linewidth=1.2, alpha=0.9, label='Q25 (Annual Pr)')
    ax3.axhline(q75, linestyle='--', color='grey', linewidth=1.2, alpha=0.9, label='Q75 (Annual Pr)')
    # 典型年标注
    for yt, y in typical_years.items():
        if y is None:
            continue
        yv = annual_precip_all.loc[y]
        ax3.scatter([y], [yv], s=110, facecolors='none', edgecolors='black', linewidth=1.6, zorder=4)
        ax3.annotate(f"{y}", (y, yv), textcoords="offset points", xytext=(6, 5), ha='left', fontsize=9)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Annual precipitation (mm)")
    ax3.set_title("Annual precipitation (type by Q25/Q75)")
    ax3.legend(ncol=2, frameon=True)
    ax3.grid(True, axis='y', alpha=0.25)
    for lbl in ax3.get_xticklabels():
        lbl.set_rotation(45)

    # ---------- (4) Box: Growing season mean maximum temperature ----------
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(data=out.reset_index(), x='YearType', y='GS_TxMean',
                order=YEAR_TYPES, palette=palette, width=0.6, ax=ax4)
    sns.stripplot(data=out.reset_index(), x='YearType', y='GS_TxMean',
                  order=YEAR_TYPES, color='black', alpha=0.45, size=3, jitter=True, ax=ax4)
    ax4.set_xlabel("Year type")
    ax4.set_ylabel("Mean daily max temperature in GS (℃)")
    ax4.set_title("Growing season temperature by year-type")
    ax4.grid(True, axis='y', alpha=0.25)

    # Layout & save
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    # ---------- Console summary ----------
    print(f"\nSaved figure: {OUTPUT_FIG}")
    print(f"Saved table:  {OUTPUT_CSV}")
    print(f"Q25 (Annual Pr) = {q25:.2f} mm,  Q75 (Annual Pr) = {q75:.2f} mm")
    for yt in YEAR_TYPES:
        years = out[out['YearType'] == yt].index.tolist()
        years.sort()
        print(f"{yt} ({len(years)} years): {years}")
    print("\nTypical years (by AnnualPr median-proximity):")
    for yt, y in typical_years.items():
        if y is None:
            print(f"  {yt}: None")
        else:
            print(f"  {yt}: {y}  (AnnualPr={annual_precip_all.loc[y]:.1f} mm, AnnualET0={annual_metrics.loc[y, 'Annual_ET0']:.1f} mm)")

if __name__ == "__main__":
    main()