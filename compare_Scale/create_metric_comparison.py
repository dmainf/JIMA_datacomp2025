import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11

base_path = '.'

csv_files = sorted(glob.glob(f'{base_path}/*/predictions_all.csv'))

QUANTILES = ['q0.1', 'q0.5', 'q0.9']
ZSCORE_THRESHOLD = 2.0

models = []
pred_dfs = []
for csv_path in csv_files:
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)

def compute_series_total(df):
    df = df.copy()
    total_per_book = df.groupby('book_name')['actual'].sum()
    df['series_total'] = df['book_name'].map(total_per_book)
    return df

def plot_scale_diff(df, quantile_col, model_name, output_dir):
    df = compute_series_total(df.copy())
    df['diff'] = df[quantile_col] - df['actual']
    df = df[np.isfinite(df['diff'])]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['series_total'], df['diff'], alpha=0.3, s=10, color='#d6336c')
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, label='error = 0')
    ax.set_xlabel('total actual (per series)', fontsize=13)
    ax.set_ylabel('error = predicted - actual', fontsize=13)
    ax.set_title(f'{model_name}  [{quantile_col}]', fontsize=14, fontweight='bold')
    stats = df['diff'].describe()
    textstr = (f"n={int(stats['count'])}  mean={stats['mean']:.3f}  "
               f"std={stats['std']:.3f}  median={stats['50%']:.3f}")
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.set_yscale('symlog')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_scale_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{model_name}_scale_diff.png")

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_scale_diff_all(dfs, quantile_col, models, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        df = compute_series_total(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df = df[np.isfinite(df['diff'])]
        ax.scatter(df['series_total'], df['diff'], alpha=0.2, s=5,
                   color=color_cycle[i % len(color_cycle)], label=model_name)
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, label='error = 0')
    ax.set_yscale('symlog')
    ax.set_xlabel('total actual (per series)', fontsize=13)
    ax.set_ylabel('error = predicted - actual', fontsize=13)
    ax.set_title(f'All Models  [{quantile_col}]', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_models_scale_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/all_models_scale_diff.png")

def plot_scale_diff_grid(dfs, quantile_col, models, output_dir):
    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        ax = axes[i]
        df = compute_series_total(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df = df[np.isfinite(df['diff'])]
        ax.scatter(df['series_total'], df['diff'], alpha=0.3, s=5, color='#d6336c')
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
        ax.set_yscale('symlog')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('total actual (per series)', fontsize=10)
        ax.set_ylabel('predicted - actual', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
    for j in range(len(models), len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'All Models  [{quantile_col}]', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/grid_scale_diff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/grid_scale_diff.png")

for quantile in QUANTILES:
    scale_dir = f'metric_comparison/{quantile}'
    os.makedirs(scale_dir, exist_ok=True)
    for model_name, df in zip(models, pred_dfs):
        plot_scale_diff(df, quantile, model_name, scale_dir)
    plot_scale_diff_all(pred_dfs, quantile, models, scale_dir)
    plot_scale_diff_grid(pred_dfs, quantile, models, scale_dir)
    print()

print("Done.")
