import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11

base_path = '../compare_SPIKE'

csv_files = sorted(glob.glob(f'{base_path}/*/predictions_all.csv'))

QUANTILES = {'q0.1': 0.1, 'q0.5': 0.5, 'q0.9': 0.9}
ZSCORE_THRESHOLD = 2.0

models = []
pred_dfs = []
for csv_path in csv_files:
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)

def compute_global_zscore(df):
    mean = df['actual'].mean()
    std = df['actual'].std()
    df = df.copy()
    df['zscore'] = (df['actual'] - mean) / std if std > 0 else 0.0
    return df

def pinball_loss(actual, predicted, tau):
    diff = actual - predicted
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)

def plot_crps(df, model_name, output_dir):
    df = compute_global_zscore(df.copy())
    # CRPS ≈ 2 * mean of pinball losses across quantiles
    crps = sum(
        pinball_loss(df['actual'].values, df[col].values, tau)
        for col, tau in QUANTILES.items()
    ) * 2 / len(QUANTILES)
    df = df.copy()
    df['crps'] = crps
    df = df[np.isfinite(df['crps'])]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['zscore'], df['crps'], alpha=0.3, s=10, color='#d6336c')
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, label='CRPS = 0')
    stats = df['crps'].describe()
    textstr = (f"n={int(stats['count'])}  mean={stats['mean']:.3f}  "
               f"std={stats['std']:.3f}  median={stats['50%']:.3f}")
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.set_xlabel('z-score of actual', fontsize=13)
    ax.set_ylabel('CRPS (approx. from quantiles)', fontsize=13)
    ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_crps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{model_name}_crps.png")

out_dir = 'crps_plots'
os.makedirs(out_dir, exist_ok=True)
for model_name, df in zip(models, pred_dfs):
    plot_crps(df, model_name, out_dir)
print()
print("Done.")

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_crps_all(dfs, models, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        df = compute_global_zscore(df.copy())
        crps = sum(
            pinball_loss(df['actual'].values, df[col].values, tau)
            for col, tau in QUANTILES.items()
        ) * 2 / len(QUANTILES)
        df = df.copy()
        df['crps'] = crps
        df = df[np.isfinite(df['crps'])]
        ax.scatter(df['zscore'], df['crps'], alpha=0.2, s=5,
                   color=color_cycle[i % len(color_cycle)], label=model_name)
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, label='CRPS = 0')
    ax.set_xlabel('z-score of actual', fontsize=13)
    ax.set_ylabel('CRPS (approx. from quantiles)', fontsize=13)
    ax.set_title('All Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_models_crps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/all_models_crps.png")

plot_crps_all(pred_dfs, models, out_dir)

def plot_crps_grid(dfs, models, output_dir):
    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        ax = axes[i]
        df = compute_global_zscore(df.copy())
        crps = sum(
            pinball_loss(df['actual'].values, df[col].values, tau)
            for col, tau in QUANTILES.items()
        ) * 2 / len(QUANTILES)
        df = df.copy()
        df['crps'] = crps
        df = df[np.isfinite(df['crps'])]
        ax.scatter(df['zscore'], df['crps'], alpha=0.3, s=5, color='#d6336c')
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('z-score of actual', fontsize=10)
        ax.set_ylabel('CRPS', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
    for j in range(len(models), len(axes)):
        axes[j].axis('off')
    plt.suptitle('All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/grid_crps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/grid_crps.png")

plot_crps_grid(pred_dfs, models, out_dir)
