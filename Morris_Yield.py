#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AquaCrop Morris Global Sensitivity Analysis (GSA)
-------------------------------------------------
Performs full Morris sensitivity analysis for the AquaCrop MaizeGDD model
across three representative climatic years (1985: dry, 2006: normal, 2012: wet).
Outputs:
    - morris_results_all_years.csv
    - morris_mu_star_topN_all_years.png
"""

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# Weather data and reproducibility
path = get_filepath(r'C:\Users\Hymn\Desktop\Aquacrop-SA\meteorological_data_tab.txt')
wdf = prepare_weather(path)
np.random.seed(42)

# Parameter space definition
problem = {
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

# Morris sampling
print("Generating Morris samples...")
X = morris_sample.sample(problem, N=30, num_levels=4)
print(f"Generated {len(X)} samples")

# Enforce physical and phenological constraints
def enforce_parameter_constraints(param_dict):
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
    param_dict['Flowering'], param_dict['HIstart'], param_dict['YldForm'] = Fw, HI, YF

    if not (param_dict['th_wp'] < param_dict['th_fc'] < param_dict['th_s']):
        ths = sorted([param_dict['th_wp'], param_dict['th_fc'], param_dict['th_s']])
        param_dict['th_wp'], param_dict['th_fc'], param_dict['th_s'] = ths
    return param_dict

# Run AquaCrop model for a given parameter set and year
def run_model_with_params(params, year):
    try:
        param_dict = dict(zip(problem['names'], params))
        param_dict = enforce_parameter_constraints(param_dict)
        param_dict.update({
            'Determinant': 1, 'CalendarType': 2, 'CropType': 3, 'PlantMethod': 1,
            'SwitchGDD': 0, 'GDDmethod': 3, 'PolHeatStress': 1, 'PolColdStress': 1,
            'TrColdStress': 1, 'GDD_lo': 0, 'ETadj': 1,
            'p_lo2': 1.0, 'p_lo3': 1.0, 'p_lo4': 1.0
        })

        crop_params = {k: v for k, v in param_dict.items()
                       if k not in ['CN', 'th_s', 'th_fc', 'th_wp', 'Ksat', 'REW']}
        maize = Crop('MaizeGDD', planting_date='05/01', **crop_params)
        soil = Soil('custom', cn=param_dict['CN'], rew=param_dict['REW'])
        soil.add_layer(thickness=soil.zSoil,
                       thWP=param_dict['th_wp'], thFC=param_dict['th_fc'],
                       thS=param_dict['th_s'], Ksat=param_dict['Ksat'],
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

# Years configuration
years = {'1985': 'Dry', '2006': 'Normal', '2012': 'Wet'}
all_Si, all_Y = {}, {}

# Run Morris analysis for each year
for y in years:
    print(f"\n==== Running Morris analysis for {y} ({years[y]}) ====")
    Y = []
    for i, params in enumerate(X):
        if i % 500 == 0:
            print(f"[{y}] Progress: {i}/{len(X)}")
        Y.append(run_model_with_params(params, year=y))
    Y = np.nan_to_num(np.array(Y), nan=0.0)
    all_Y[y] = Y
    print(f"[{y}] Valid simulations: {(Y > 0).sum()}/{len(Y)}")
    Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=4)
    all_Si[y] = Si

print("\nAll years analyzed successfully.")

# Combine results into one summary table
rows, names = [], problem['names']
for i, name in enumerate(names):
    row = {'Parameter': name}
    for y in years:
        row[f'mu_star_{y}'] = all_Si[y]['mu_star'][i]
        row[f'sigma_{y}'] = all_Si[y]['sigma'][i]
        row[f'mu_star_conf_{y}'] = all_Si[y]['mu_star_conf'][i]
    rows.append(row)

df_all = pd.DataFrame(rows)
df_all['max_mu_star'] = df_all[[f'mu_star_{y}' for y in years]].max(axis=1)
df_all['dominant_year'] = df_all[[f'mu_star_{y}' for y in years]].idxmax(axis=1).str[-4:]
df_all.to_csv('morris_results_all_years.csv', index=False, encoding='utf-8-sig')
print("Saved: morris_results_all_years.csv")

# Export each year's results
for y in years:
    out = pd.DataFrame({
        'Parameter': names,
        'mu': all_Si[y]['mu'],
        'mu_star': all_Si[y]['mu_star'],
        'sigma': all_Si[y]['sigma'],
        'mu_star_conf': all_Si[y]['mu_star_conf']
    }).sort_values('mu_star', ascending=False)
    out.to_csv(f'morris_results_{y}.csv', index=False, encoding='utf-8-sig')
    print(f"Saved: morris_results_{y}.csv")

# Plot top-N parameters comparison
top_n = 20
top_params = df_all.nlargest(top_n, 'max_mu_star')['Parameter'].tolist()
fig, ax = plt.subplots(figsize=(12, 10))
bar_height = 0.6
indices = np.arange(top_n)
width = bar_height / 3.0
offsets = [-width, 0, width]
labels = [f"{y} ({years[y]})" for y in years]

for j, y in enumerate(years):
    mu_vals = df_all.set_index('Parameter').loc[top_params, f'mu_star_{y}'].values
    ax.barh(indices + offsets[j], mu_vals, height=width, label=labels[j])

ax.set_yticks(indices)
ax.set_yticklabels(top_params, fontsize=10)
ax.set_xlabel('μ* (Mean Absolute Effect)', fontsize=12)
ax.set_title(f'Top {top_n} Most Influential Parameters (Morris μ*)', fontsize=14)
ax.grid(True, axis='x', alpha=0.3)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('morris_mu_star_topN_all_years.png', dpi=300, bbox_inches='tight')
print("Saved: morris_mu_star_topN_all_years.png")
print("Process completed.")
