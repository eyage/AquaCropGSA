#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquaCrop MaizeGDD — eFAST Sensitivity Analysis
-----------------------------------------------
Runs eFAST sensitivity analysis using Morris-filtered parameter subsets
for three representative years (1985: dry, 2006: normal, 2012: wet).

Outputs:
  - efast_results_1985.csv / efast_results_2006.csv / efast_results_2012.csv
  - efast_results_all_years.csv (combined table)
  - efast_ST_S1_comparison_by_year.png
  - efast_ST_common_params_comparison.png
  - efast_interaction_effects.png
  - efast_analysis_report.txt
"""

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import fast_sampler
from SALib.analyze import fast

# Weather data and RNG seed
path = get_filepath(r'C:\Users\Hymn\Desktop\Aquacrop-SA\meteorological_data_tab.txt')
wdf = prepare_weather(path)
np.random.seed(42)

# Full parameter space (for bounds lookup)
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

# Year-specific parameter subsets (from Morris screening)
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

def enforce_parameter_constraints(param_dict: dict) -> dict:
    min_gap = 50
    stages = ['Emergence', 'MaxRooting', 'Senescence', 'Maturity']
    vals = sorted([param_dict[s] for s in stages])
    for i in range(1, 4):
        if vals[i] - vals[i - 1] < min_gap:
            vals[i] = vals[i - 1] + min_gap
    param_dict['Emergence'], param_dict['MaxRooting'], param_dict['Senescence'], param_dict['Maturity'] = vals

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

def run_model_with_params(params, param_names, year: str) -> float:
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
            'SwitchGDD': 0, 'GDDmethod': 3, 'PolHeatStress': 1, 'PolColdStress': 1,
            'TrColdStress': 1, 'GDD_lo': 0, 'ETadj': 1,
            'p_lo2': 1.0, 'p_lo3': 1.0, 'p_lo4': 1.0
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
        results = model.get_simulation_results()
        return results['Dry yield (tonne/ha)'].mean()
    except Exception as e:
        print(f"[{year}] Model run error: {e}")
        return 0.0

years = {'1985': 'Dry', '2006': 'Normal', '2012': 'Wet'}
all_Si, all_Y = {}, {}
N = 100  # eFAST sample size per parameter

for y in years:
    print("\n" + "=" * 60)
    print(f"Starting eFAST for {y} ({years[y]})")
    print("=" * 60)

    problem = create_problem_for_year(y)
    print(f"Num parameters: {problem['num_vars']}")
    print(f"Parameters: {problem['names']}")

    print("\nGenerating eFAST samples...")
    X = fast_sampler.sample(problem, N)
    print(f"Generated {len(X)} samples (~ {problem['num_vars']} × {N} = {problem['num_vars'] * N})")

    print("\nRunning model evaluations...")
    Y = []
    for i, params in enumerate(X):
        if i % 1000 == 0:
            print(f"[{y}] Progress: {i}/{len(X)}")
        Y.append(run_model_with_params(params, problem['names'], year=y))

    Y = np.nan_to_num(np.array(Y), nan=0.0)
    all_Y[y] = Y
    print(f"[{y}] Valid runs: {(Y > 0).sum()}/{len(Y)}")

    print(f"[{y}] Running eFAST analysis...")
    Si = fast.analyze(problem, Y, print_to_console=False)
    all_Si[y] = Si

    result_df = pd.DataFrame({
        'Parameter': problem['names'],
        'S1': Si['S1'],
        'ST': Si['ST'],
        'S1_conf': Si['S1_conf'],
        'ST_conf': Si['ST_conf']
    }).sort_values('ST', ascending=False)
    result_df.to_csv(f'efast_results_{y}.csv', index=False, encoding='utf-8')
    print(f"[{y}] Saved: efast_results_{y}.csv")

print("\nAll years completed.")

print("\nBuilding combined table...")
rows, all_params = [], set()
for y in years:
    all_params.update(year_params[y])

for param in sorted(all_params):
    row = {'Parameter': param}
    for y in years:
        if param in year_params[y]:
            idx = year_params[y].index(param)
            row[f'S1_{y}'] = all_Si[y]['S1'][idx]
            row[f'ST_{y}'] = all_Si[y]['ST'][idx]
            row[f'S1_conf_{y}'] = all_Si[y]['S1_conf'][idx]
            row[f'ST_conf_{y}'] = all_Si[y]['ST_conf'][idx]
        else:
            row[f'S1_{y}'] = np.nan
            row[f'ST_{y}'] = np.nan
            row[f'S1_conf_{y}'] = np.nan
            row[f'ST_conf_{y}'] = np.nan
    rows.append(row)

df_all = pd.DataFrame(rows)
st_cols = [f'ST_{y}' for y in years]
df_all['max_ST'] = df_all[st_cols].max(axis=1)
df_all['dominant_year'] = df_all[st_cols].idxmax(axis=1).str[-4:]
df_all = df_all.sort_values('max_ST', ascending=False)
df_all.to_csv('efast_results_all_years.csv', index=False, encoding='utf-8')
print("Saved: efast_results_all_years.csv")

print("\nRendering figures...")

# Figure 1: Top-15 S1 vs ST per year
top_n = 15
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for idx, y in enumerate(years):
    ax = axes[idx]
    problem_y = create_problem_for_year(y)
    params_y = problem_y['names']
    st = all_Si[y]['ST']
    s1 = all_Si[y]['S1']
    order = np.argsort(st)[::-1][:top_n]
    top_params = [params_y[i] for i in order]
    top_st = [st[i] for i in order]
    top_s1 = [s1[i] for i in order]
    x = np.arange(len(top_params))
    w = 0.35
    ax.barh(x - w/2, top_s1, w, label='S1 (first-order)', alpha=0.85)
    ax.barh(x + w/2, top_st, w, label='ST (total)', alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top_params, fontsize=9)
    ax.set_xlabel('Sensitivity index')
    ax.set_title(f'{y} ({years[y]}) — Top {top_n}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig('efast_ST_S1_comparison_by_year.png', dpi=300, bbox_inches='tight')
print("Saved: efast_ST_S1_comparison_by_year.png")

# Figure 2: ST comparison for common parameters (Top-20 by mean ST)
common_params = sorted(set(year_params['1985']) & set(year_params['2006']) & set(year_params['2012']))
if common_params:
    fig, ax = plt.subplots(figsize=(14, 10))
    st_data = {p: [all_Si[y]['ST'][year_params[y].index(p)] for y in years] for p in common_params}
    mean_st = {p: float(np.mean(st_data[p])) for p in common_params}
    top_params = sorted(common_params, key=lambda p: mean_st[p], reverse=True)[:20]
    x = np.arange(len(top_params))
    w = 0.25
    for j, y in enumerate(years):
        vals = [st_data[p][j] for p in top_params]
        ax.barh(x + j*w - w, vals, w, label=f'{y} ({years[y]})', alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top_params, fontsize=10)
    ax.set_xlabel('ST (total effect)')
    ax.set_title('Common parameters across years — ST comparison (Top 20)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('efast_ST_common_params_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: efast_ST_common_params_comparison.png")

# Figure 3: Interaction measure (ST - S1) per year (Top-15)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for idx, y in enumerate(years):
    ax = axes[idx]
    problem_y = create_problem_for_year(y)
    params_y = problem_y['names']
    s1 = all_Si[y]['S1']
    st = all_Si[y]['ST']
    inter = st - s1
    order = np.argsort(inter)[::-1][:15]
    top_params = [params_y[i] for i in order]
    top_inter = [inter[i] for i in order]
    ax.barh(range(len(top_params)), top_inter, alpha=0.8)
    ax.set_yticks(range(len(top_params)))
    ax.set_yticklabels(top_params, fontsize=9)
    ax.set_xlabel('Interaction (ST - S1)')
    ax.set_title(f'{y} ({years[y]}) — Interaction Top 15')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    ax.axvline(x=0.10, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Strong (>0.10)')
    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Moderate (>0.05)')
    ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('efast_interaction_effects.png', dpi=300, bbox_inches='tight')
print("Saved: efast_interaction_effects.png")

# Text report
print("\nWriting analysis report...")
with open('efast_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("eFAST Sensitivity Analysis Report\n")
    f.write("=" * 60 + "\n\n")
    for y in years:
        f.write(f"{y} ({years[y]})\n")
        f.write("-" * 60 + "\n")
        problem_y = create_problem_for_year(y)
        params_y = problem_y['names']
        s1 = all_Si[y]['S1']
        st = all_Si[y]['ST']
        order = np.argsort(st)[::-1][:10]
        f.write("Top 10 parameters by ST:\n")
        for rank, idx in enumerate(order, 1):
            f.write(f"  {rank:>2}. {params_y[idx]:15s}  ST: {st[idx]:.4f}  S1: {s1[idx]:.4f}  Interaction: {st[idx]-s1[idx]:.4f}\n")
        f.write("\nSummary:\n")
        f.write(f"  Num parameters: {len(params_y)}\n")
        f.write(f"  Num simulations: {len(all_Y[y])}\n")
        f.write(f"  Num valid outputs: {(all_Y[y] > 0).sum()}\n")
        f.write(f"  Mean yield: {all_Y[y].mean():.2f} tonne/ha\n")
        f.write(f"  Std yield:  {all_Y[y].std():.2f} tonne/ha\n")
        high = int(np.sum(st > 0.10))
        mid = int(np.sum((st > 0.05) & (st <= 0.10)))
        low = int(np.sum(st <= 0.05))
        f.write("\nSensitivity tiers:\n")
        f.write(f"  High (ST > 0.10): {high}\n")
        f.write(f"  Medium (0.05 < ST ≤ 0.10): {mid}\n")
        f.write(f"  Low (ST ≤ 0.05): {low}\n")
        f.write("\n")

print("Saved: efast_analysis_report.txt")

print("\n" + "=" * 60)
print("All tasks completed.")
print("=" * 60)
print("\nGenerated files:")
print("  1. efast_results_1985.csv")
print("  2. efast_results_2006.csv")
print("  3. efast_results_2012.csv")
print("  4. efast_results_all_years.csv")
print("  5. efast_ST_S1_comparison_by_year.png")
print("  6. efast_ST_common_params_comparison.png")
print("  7. efast_interaction_effects.png")
print("  8. efast_analysis_report.txt")
