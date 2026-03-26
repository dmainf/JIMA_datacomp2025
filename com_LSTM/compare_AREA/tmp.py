import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10

base_path = '.'
QUANTILE_COL = 'q0.5'
TARGET_BOOK = '52ヘルツのクジラたち_original'

csv_files = sorted(glob.glob(f'{base_path}/*/predictions_all.csv'))
model_names = [os.path.basename(os.path.dirname(p)) for p in csv_files]
dfs = {name: pd.read_csv(p).rename(columns={'書名': 'book_name'})
       for name, p in zip(model_names, csv_files)}

def area_metrics(actual, pred, days):
    diff = pred - actual
    over  = np.trapezoid(np.where(diff > 0, diff, 0), days)
    under = np.trapezoid(np.where(diff < 0, -diff, 0), days)
    return over, under

n_models = len(model_names)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)

for ax, model in zip(axes, model_names):
    grp = dfs[model][dfs[model]['book_name'] == TARGET_BOOK].sort_values('day')
    days   = grp['day'].values
    actual = grp['actual'].values
    pred   = grp[QUANTILE_COL].values

    over, under = area_metrics(actual, pred, days)

    ax.plot(days, actual, color='black', lw=1.5, label='actual')
    ax.plot(days, pred, color='#d6336c', lw=1.5, linestyle='--', label='pred (q0.5)')
    ax.fill_between(days, actual, pred, where=(pred >= actual),
                    interpolate=True, alpha=0.4, color='#e03131', label=f'over ({over:.1f})')
    ax.fill_between(days, actual, pred, where=(pred <= actual),
                    interpolate=True, alpha=0.4, color='#1971c2', label=f'under ({under:.1f})')

    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xlabel('day', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)

axes[0].set_ylabel('sales', fontsize=10)
fig.suptitle(f'Area comparison: {TARGET_BOOK}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('tmp_single_book.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: tmp_single_book.png")
