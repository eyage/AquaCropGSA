#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquaCrop MaizeGDD — Daily eFAST Sensitivity (Canopy Cover)
----------------------------------------------------------
Time-resolved (daily) eFAST sensitivity of the Canopy Cover (CC) output
over a unified 130-day window. eFAST is computed for each day (0..129).

Outputs (for Canopy Cover):
  - efast_S1_heatmap_daily_{year}.png
  - efast_ST_heatmap_daily_{year}.png
  - efast_timeseries_comparison_all_years_daily.png
  - efast_S1_timeseries_daily_{year}.csv
  - efast_ST_timeseries_daily_{year}.csv
  - efast_timeseries_report_daily.txt
"""

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import fast_sampler
from SALib.analyze import fast

# Weather & seed
path = get_filepath(r'C:\Users\Hymn\Desktop\Aquacrop-SA\meteorological_data_tab.txt')
wdf = prepare_weather(path)
np.random.seed(42)

# Parameter space (for bounds lookup)
original_problem = {
    'num_vars': 51,
    'names': [
        'CN', 'th_s', 'th_fc', 'th_wp', 'Ksat', 'REW',
        'Emergence', 'HIstart', 'Senescence', 'Maturity', 'YldForm',
        'Flowering', 'SeedSize', 'PlantPop', 'CGC', 'CCx', 'CDC', 'Tbase', 'Tupp',
        'MaxRooting', 'Zmax', 'Zmin', 'fshape_r', 'SxTopQ', 'SxBotQ',
        'Kcb', 'fage', 'fsink',
        'WP', 'HI0', 'exc', 'WPy',
        'Aer', 'fshape_w4',
        'p_up1', 'p_lo1', 'fshape_w1',
        'p_up2', 'fshape_w2',
        'p_up3', 'fshape_w3',
        'p_up4',
        'dHI_pre', 'a_HI', 'b_HI', 'dHI0',
        'Tmin_up', 'Tmax_up', 'Tmin_lo', 'Tmax_lo', 'GDD_up',
    ],
    'bounds': [
        [46, 77], [0.38, 0.48], [0.18, 0.28], [0.06, 0.12], [200, 1000], [7, 12],
        [55, 105], [615, 1145], [980, 1500], [1600, 1850], [525, 975],
        [125, 235], [4.5, 8.5], [50000, 95000], [0.010, 0.015], [0.70, 0.99],
        [0.008, 0.012], [7.5, 8.5], [29.5, 30.5],
        [1260, 1540], [1.5, 3.0], [0.3, 0.5], [1.00, 1.5], [0.02, 0.055], [0.01, 0.013],
        [0.85, 1.1], [0.20, 0.5], [0.4, 0.6],
        [27, 45], [0.35, 0.55], [35, 65], [75, 125],
        [4, 6], [0.8, 1.2],
        [0.09, 0.19], [0.58, 0.86], [1.9, 3.9],
        [0.55, 0.83], [4.0, 8.0],
        [0.55, 0.83], [1.8, 3.6],
        [0.64, 0.96],
        [0, 0.1], [3.5, 10], [1.5, 4.5], [8, 23],
        [8, 12], [35, 45], [4, 6], [40, 50], [9, 15],
    ]
}
param_bounds_dict = dict(zip(original_problem['names'], original_problem['bounds']))

# Year-specific parameter subsets (from Morris)
year_params = {
    '1985': ['th_fc', 'th_wp', 'HIstart', 'Senescence', 'Maturity', 'YldForm',
             'SeedSize', 'PlantPop', 'CGC', 'CCx', 'CDC', 'Tbase',
             'MaxRooting', 'Zmax', 'fshape_r', 'Kcb', 'fage', 'WP',
             'HI0', 'WPy', 'p_lo1', 'fshape_w1', 'p_up2', 'p_up3'],
    '2006': ['th_fc', 'th_wp', 'REW', 'HIstart', 'Senescence', 'Maturity', 'YldForm',
             'SeedSize', 'PlantPop', 'CGC', 'CCx', 'CDC', 'Tbase',
             'MaxRooting', 'Zmax', 'fshape_r', 'Kcb', 'fage', 'WP',
             'HI0', 'WPy', 'p_lo1', 'fshape_w1', 'p_up3'],
    '2012': ['th_fc', 'th_wp', 'Emergence', 'HIstart', 'Senescence', 'Maturity', 'YldForm',
             'SeedSize', 'PlantPop', 'CGC', 'CCx', 'CDC', 'Tbase',
             'MaxRooting', 'Zmax', 'Zmin', 'fshape_r', 'Kcb', 'fage', 'WP',
             'HI0', 'WPy', 'p_lo1', 'fshape_w1']
}
def create_problem_for_year(year: str) -> dict:
    params = year_params[year]
    return {'num_vars': len(params), 'names': params, 'bounds': [param_bounds_dict[p] for p in params]}

# Constraints for phenology and soil water content
def enforce_parameter_constraints(param_dict: dict) -> dict:
    min_gap = 50
    stages = ['Emergence', 'MaxRooting', 'Senescence', 'Maturity']
    vals = sorted([param_dict[s] for s in stages])
    for i in range(1, 4):
        if vals[i] - vals[i - 1] < min_gap:
            vals[i] = vals[i - 1] + min_gap
    param_dict['Emergence'], param_dict['MaxRooting'], param_dict['Senescence'], param_dict['Maturity'] = vals
    if param_dict['Maturity'] - param_dict['Emergence'] < 100:
        param_dict['Maturity'] = param_dict['Emergence'] + 100
        param_dict['MaxRooting'] = param_dict['Emergence'] + 30
        param_dict['Senescence'] = param_dict['Maturity'] - 30
    E = param_dict['Emergence']
    Fw = max(param_dict['Flowering'], E + 80)
    HI = max(param_dict['HIstart'], Fw + 10)
    YF = max(param_dict['YldForm'], 100)
    if HI + YF > param_dict['Maturity'] - 10:
        YF = max(100, param_dict['Maturity'] - 10 - HI)
        if HI + YF > param_dict['Maturity'] - 10:
            HI = max(Fw + 10, param_dict['Maturity'] - 10 - YF)
    param_dict['Flowering'] = Fw
    param_dict['HIstart'] = HI
    param_dict['YldForm'] = YF
    if not (param_dict['th_wp'] < param_dict['th_fc'] < param_dict['th_s']):
        ths = sorted([param_dict['th_wp'], param_dict['th_fc'], param_dict['th_s']])
        param_dict['th_wp'], param_dict['th_fc'], param_dict['th_s'] = ths
    return param_dict

# Run model and return Canopy Cover time series: (dap, canopy_cover)
def run_model_with_params_timeseries(params, param_names, year: str):
    try:
        full_param_dict = {
            'CN': 61, 'th_s': 0.43, 'th_fc': 0.23, 'th_wp': 0.09, 'Ksat': 600, 'REW': 9,
            'Emergence': 80, 'HIstart': 880, 'Senescence': 1240, 'Maturity': 1725, 'YldForm': 750,
            'Flowering': 180, 'SeedSize': 6.5, 'PlantPop': 72500, 'CGC': 0.0125, 'CCx': 0.845,
            'CDC': 0.010, 'Tbase': 8.0, 'Tupp': 30.0,
            'MaxRooting': 1400, 'Zmax': 2.25, 'Zmin': 0.4, 'fshape_r': 1.25, 'SxTopQ': 0.0375, 'SxBotQ': 0.0115,
            'Kcb': 0.975, 'fage': 0.35, 'fsink': 0.5,
            'WP': 36, 'HI0': 0.45, 'exc': 50, 'WPy': 100,
            'Aer': 5, 'fshape_w4': 1.0,
            'p_up1': 0.14, 'p_lo1': 0.72, 'fshape_w1': 2.9,
            'p_up2': 0.69, 'fshape_w2': 6.0,
            'p_up3': 0.69, 'fshape_w3': 2.7,
            'p_up4': 0.80,
            'dHI_pre': 0.05, 'a_HI': 6.75, 'b_HI': 3.0, 'dHI0': 15.5,
            'Tmin_up': 10, 'Tmax_up': 40, 'Tmin_lo': 5, 'Tmax_lo': 45, 'GDD_up': 12,
        }
        for i, name in enumerate(param_names):
            full_param_dict[name] = params[i]
        full_param_dict = enforce_parameter_constraints(full_param_dict)
        full_param_dict.update({
            'Determinant': 1, 'CalendarType': 2, 'CropType': 3, 'PlantMethod': 1,
            'SwitchGDD': 0, 'GDDmethod': 3, 'PolHeatStress': 1, 'PolColdStress': 1, 'TrColdStress': 1,
            'GDD_lo': 0, 'ETadj': 1, 'p_lo2': 1.0, 'p_lo3': 1.0, 'p_lo4': 1.0
        })

        crop_params = {k: v for k, v in full_param_dict.items()
                       if k not in ['CN', 'th_s', 'th_fc', 'th_wp', 'Ksat', 'REW']}
        maize = Crop('MaizeGDD', planting_date='05/01', **crop_params)
        soil = Soil('custom', cn=full_param_dict['CN'], rew=full_param_dict['REW'])
        soil.add_layer(thickness=soil.zSoil,
                       thWP=full_param_dict['th_wp'],
                       thFC=full_param_dict['th_fc'],
                       thS=full_param_dict['th_s'],
                       Ksat=full_param_dict['Ksat'],
                       penetrability=100)
        init_wc = InitialWaterContent(wc_type='Pct', value=[70])

        model = AquaCropModel(
            sim_start_time=f'{year}/05/01',
            sim_end_time=f'{year}/12/30',
            weather_df=wdf,
            soil=soil,
            crop=maize,
            initial_water_content=init_wc
        )
        model.run_model(till_termination=True)
        crop_growth = model.get_crop_growth()  # expects 'dap' and 'canopy_cover'
        return crop_growth[['dap', 'canopy_cover']].values
    except Exception as e:
        print(f"[{year}] Model run error: {e}")
        return None

# Main loop (per year)
years = {'1985': 'Dry', '2006': 'Normal', '2012': 'Wet'}
N = 100
MAX_DAYS = 130
VAR_EPS = 1e-12
POS_RATIO = 0.70

all_timeseries_Si = {}

for y in years:
    print("\n" + "=" * 60)
    print(f"Daily eFAST (Canopy Cover) for {y} ({years[y]})")
    print("=" * 60)

    problem = create_problem_for_year(y)
    print(f"Num parameters: {problem['num_vars']}")
    print(f"Parameters: {problem['names']}")

    print("\nGenerating eFAST samples...")
    X = fast_sampler.sample(problem, N)
    print(f"Generated {len(X)} samples")

    print("\nRunning model to collect daily Canopy Cover...")
    all_timeseries = []
    valid_runs = 0
    for i, params in enumerate(X):
        if i % 100 == 0:
            print(f"[{y}] Progress: {i}/{len(X)}")
        ts_data = run_model_with_params_timeseries(params, problem['names'], year=y)
        if ts_data is not None and len(ts_data) > 10:
            all_timeseries.append(ts_data)
            valid_runs += 1
    print(f"[{y}] Valid simulations: {valid_runs}/{len(X)}")
    if valid_runs < max(10, int(0.3 * len(X))):
        print(f"[{y}] Warning: too few valid simulations — skipping this year")
        continue

    print(f"[{y}] Aligning CC time series to {MAX_DAYS} days...")
    Y_timeseries = np.zeros((len(all_timeseries), MAX_DAYS))
    for i, ts in enumerate(all_timeseries):
        days = len(ts)
        if days >= MAX_DAYS:
            Y_timeseries[i, :] = ts[:MAX_DAYS, 1]
        else:
            Y_timeseries[i, :days] = ts[:, 1]
            Y_timeseries[i, days:] = ts[-1, 1]

    print(f"[{y}] Running daily eFAST (CC)...")
    time_steps = list(range(MAX_DAYS))
    num_p = len(problem['names'])
    S1_timeseries = np.full((num_p, MAX_DAYS), np.nan)
    ST_timeseries = np.full((num_p, MAX_DAYS), np.nan)

    for t in time_steps:
        if t % 10 == 0:
            print(f"  Day {t}/{MAX_DAYS-1}")
        Y_t = Y_timeseries[:, t]
        if (np.var(Y_t) <= VAR_EPS) or ((Y_t > 0).sum() < POS_RATIO * len(Y_t)):
            continue
        try:
            Si_t = fast.analyze(problem, Y_t, print_to_console=False)
            S1_timeseries[:, t] = Si_t['S1']
            ST_timeseries[:, t] = Si_t['ST']
        except Exception as e:
            print(f"  Day {t} eFAST error: {e}")
            continue

    all_timeseries_Si[y] = {
        'S1': S1_timeseries,
        'ST': ST_timeseries,
        'time_steps': time_steps,
        'param_names': problem['names']
    }
    print(f"[{y}] Daily CC sensitivity finished.")

print("\nAll years processed (daily CC).")

# Heatmaps per year (Top-15)
print("\nRendering daily CC heatmaps (Top 15)...")
for y in years:
    if y not in all_timeseries_Si:
        continue
    S1_data = all_timeseries_Si[y]['S1']
    ST_data = all_timeseries_Si[y]['ST']
    time_steps = all_timeseries_Si[y]['time_steps']
    param_names = all_timeseries_Si[y]['param_names']

    mean_S1 = np.nanmean(S1_data, axis=1)
    top_idx_S1 = np.argsort(mean_S1)[::-1][:15]
    S1_plot = S1_data[top_idx_S1, :]
    labels_S1 = [param_names[i] for i in top_idx_S1]
    fig, ax = plt.subplots(figsize=(18, 9))
    mask = np.isnan(S1_plot)
    sns.heatmap(
        S1_plot, mask=mask, yticklabels=labels_S1, xticklabels=False,
        cmap='YlOrRd', cbar_kws={'label': 'S1 (first-order, CC)'},
        ax=ax, vmin=0, vmax=np.nanmax(S1_plot)
    )
    tick_every = 10
    xticks = np.arange(0, len(time_steps), tick_every)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([f'Day {t}' for t in time_steps[::tick_every]], rotation=0)
    ax.set_xlabel('Days after planting (DAP)')
    ax.set_ylabel('Parameters')
    ax.set_title(f'{y} ({years[y]}) — Daily S1 for Canopy Cover (Top 15)')
    plt.tight_layout()
    plt.savefig(f'efast_S1_heatmap_daily_{y}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: efast_S1_heatmap_daily_{y}.png")
    plt.close()

    mean_ST = np.nanmean(ST_data, axis=1)
    top_idx_ST = np.argsort(mean_ST)[::-1][:15]
    ST_plot = ST_data[top_idx_ST, :]
    labels_ST = [param_names[i] for i in top_idx_ST]
    fig, ax = plt.subplots(figsize=(18, 9))
    mask = np.isnan(ST_plot)
    sns.heatmap(
        ST_plot, mask=mask, yticklabels=labels_ST, xticklabels=False,
        cmap='RdYlBu_r', cbar_kws={'label': 'ST (total, CC)'},
        ax=ax, vmin=0, vmax=np.nanmax(ST_plot)
    )
    xticks = np.arange(0, len(time_steps), tick_every)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([f'Day {t}' for t in time_steps[::tick_every]], rotation=0)
    ax.set_xlabel('Days after planting (DAP)')
    ax.set_ylabel('Parameters')
    ax.set_title(f'{y} ({years[y]}) — Daily ST for Canopy Cover (Top 15)')
    plt.tight_layout()
    plt.savefig(f'efast_ST_heatmap_daily_{y}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: efast_ST_heatmap_daily_{y}.png")
    plt.close()

# Cross-year daily heatmaps for common parameters
print("\nRendering cross-year daily CC heatmaps for common parameters...")
common_params = sorted(set(year_params['1985']) & set(year_params['2006']) & set(year_params['2012']))
if common_params:
    fig, axes = plt.subplots(3, 2, figsize=(22, 18))
    for idx, y in enumerate(years):
        if y not in all_timeseries_Si:
            continue
        S1_data = all_timeseries_Si[y]['S1']
        ST_data = all_timeseries_Si[y]['ST']
        time_steps = all_timeseries_Si[y]['time_steps']
        param_names = all_timeseries_Si[y]['param_names']

        common_idx = [param_names.index(p) for p in common_params if p in param_names]
        common_labels = [p for p in common_params if p in param_names]

        ax_s1 = axes[idx, 0]
        S1_common = S1_data[common_idx, :]
        mask = np.isnan(S1_common)
        sns.heatmap(
            S1_common, mask=mask, xticklabels=False, yticklabels=common_labels,
            cmap='YlOrRd', cbar_kws={'label': 'S1 (CC)'}, ax=ax_s1,
            vmin=0, vmax=np.nanmax(S1_common)
        )
        xticks = np.arange(0, len(time_steps), 10)
        ax_s1.set_xticks(xticks + 0.5)
        ax_s1.set_xticklabels([f'{t}' for t in time_steps[::10]], rotation=0)
        ax_s1.set_xlabel('DAP'); ax_s1.set_ylabel('Parameters')
        ax_s1.set_title(f'{y} — S1 (common parameters, daily CC)')

        ax_st = axes[idx, 1]
        ST_common = ST_data[common_idx, :]
        mask = np.isnan(ST_common)
        sns.heatmap(
            ST_common, mask=mask, xticklabels=False, yticklabels=common_labels,
            cmap='RdYlBu_r', cbar_kws={'label': 'ST (CC)'}, ax=ax_st,
            vmin=0, vmax=np.nanmax(ST_common)
        )
        xticks = np.arange(0, len(time_steps), 10)
        ax_st.set_xticks(xticks + 0.5)
        ax_st.set_xticklabels([f'{t}' for t in time_steps[::10]], rotation=0)
        ax_st.set_xlabel('DAP'); ax_st.set_ylabel('Parameters')
        ax_st.set_title(f'{y} — ST (common parameters, daily CC)')

    plt.suptitle('Cross-year daily sensitivity for Canopy Cover (common parameters)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('efast_timeseries_comparison_all_years_daily.png', dpi=300, bbox_inches='tight')
    print("Saved: efast_timeseries_comparison_all_years_daily.png")
    plt.close()

# Save daily time-series sensitivity to CSV
print("\nSaving daily Canopy Cover sensitivity to CSV...")
for y in years:
    if y not in all_timeseries_Si:
        continue
    S1_data = all_timeseries_Si[y]['S1']
    ST_data = all_timeseries_Si[y]['ST']
    time_steps = all_timeseries_Si[y]['time_steps']
    param_names = all_timeseries_Si[y]['param_names']

    df_s1 = pd.DataFrame(S1_data.T, columns=param_names)
    df_s1.insert(0, 'Day', time_steps)
    df_s1.to_csv(f'efast_S1_timeseries_daily_{y}.csv', index=False, encoding='utf-8')

    df_st = pd.DataFrame(ST_data.T, columns=param_names)
    df_st.insert(0, 'Day', time_steps)
    df_st.to_csv(f'efast_ST_timeseries_daily_{y}.csv', index=False, encoding='utf-8')

    print(f"[{y}] Saved daily CC CSVs")

# Text report (daily CC)
print("\nWriting daily Canopy Cover sensitivity report...")
with open('efast_timeseries_report_daily.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("Daily eFAST Sensitivity Report — Canopy Cover (CC)\n")
    f.write("=" * 60 + "\n\n")
    for y in years:
        if y not in all_timeseries_Si:
            continue
        f.write(f"\n{y} ({years[y]}) — Canopy Cover\n")
        f.write("-" * 60 + "\n")
        S1_data = all_timeseries_Si[y]['S1']
        ST_data = all_timeseries_Si[y]['ST']
        param_names = all_timeseries_Si[y]['param_names']

        mean_S1 = np.nanmean(S1_data, axis=1)
        mean_ST = np.nanmean(ST_data, axis=1)
        cv_S1 = np.nanstd(S1_data, axis=1) / (mean_S1 + 1e-10)
        cv_ST = np.nanstd(ST_data, axis=1) / (mean_ST + 1e-10)

        top_idx = np.argsort(mean_ST)[::-1][:10]
        f.write("Top 10 parameters by mean ST (CC):\n")
        for rank, idx in enumerate(top_idx, 1):
            f.write(
                f"  {rank}. {param_names[idx]:15s} - mean ST: {mean_ST[idx]:.4f}, "
                f"mean S1: {mean_S1[idx]:.4f}, ST CV: {cv_ST[idx]:.4f}\n"
            )

        top_cv_idx = np.argsort(cv_ST)[::-1][:5]
        f.write("\nTop 5 parameters by ST variability (CC, CV):\n")
        for rank, idx in enumerate(top_cv_idx, 1):
            f.write(
                f"  {rank}. {param_names[idx]:15s} - ST CV: {cv_ST[idx]:.4f}, "
                f"mean ST: {mean_ST[idx]:.4f}\n"
            )

print("Saved: efast_timeseries_report_daily.txt")

print("\n" + "=" * 60)
print("All tasks completed.")
print("=" * 60)
print("\nGenerated files (Canopy Cover):")
print("  - efast_S1_heatmap_daily_1985.png")
print("  - efast_ST_heatmap_daily_1985.png")
print("  - efast_S1_heatmap_daily_2006.png")
print("  - efast_ST_heatmap_daily_2006.png")
print("  - efast_S1_heatmap_daily_2012.png")
print("  - efast_ST_heatmap_daily_2012.png")
print("  - efast_timeseries_comparison_all_years_daily.png")
print("  - efast_S1_timeseries_daily_1985.csv")
print("  - efast_ST_timeseries_daily_1985.csv")
print("  - efast_S1_timeseries_daily_2006.csv")
print("  - efast_ST_timeseries_daily_2006.csv")
print("  - efast_S1_timeseries_daily_2012.csv")
print("  - efast_ST_timeseries_daily_2012.csv")
print("  - efast_timeseries_report_daily.txt")
