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
STD_BINS = [0, 20, 40, 60, 80, 100, 120, 140, np.inf]
STD_BIN_LABELS = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140+']

models = []
pred_dfs = []
for csv_path in csv_files:
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)

def compute_series_std(df):
    df = df.copy()
    std_per_book = df.groupby('book_name')['actual'].std()
    df['series_std'] = df['book_name'].map(std_per_book)
    return df

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def build_binned_stats(dfs, models, quantile_col):
    records = []
    for df, model_name in zip(dfs, models):
        df = compute_series_std(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df['abs_diff'] = df['diff'].abs()
        df = df[np.isfinite(df['diff'])]
        df['std_bin'] = pd.cut(df['series_std'], bins=STD_BINS, labels=STD_BIN_LABELS, right=False)
        for bin_label in STD_BIN_LABELS:
            subset = df[df['std_bin'] == bin_label]['diff']
            if len(subset) == 0:
                continue
            records.append({
                'model': model_name,
                'std_bin': bin_label,
                'mae': subset.abs().mean(),
                'rmse': np.sqrt((subset ** 2).mean()),
                'median_err': subset.median(),
                'n': len(subset),
                'diff_values': subset.values,
            })
    return pd.DataFrame(records)

def plot_binned_line(stats_df, metric, quantile_col, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(stats_df['model'].unique()):
        sub = stats_df[stats_df['model'] == model_name]
        ax.plot(sub['std_bin'], sub[metric], marker='o', linewidth=1.5,
                color=color_cycle[i % len(color_cycle)], label=model_name)
    ax.set_xlabel('std of actual (per series, binned)', fontsize=13)
    ylabel = {'mae': 'MAE', 'rmse': 'RMSE', 'median_err': 'Median Error'}[metric]
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{ylabel} per std bin  [{quantile_col}]', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.tight_layout()
    fname = f'{output_dir}/binned_line_{metric}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

def plot_binned_boxplot(dfs, models, quantile_col, output_dir):
    bin_order = STD_BIN_LABELS
    n_bins = len(bin_order)
    n_models = len(models)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        df = compute_series_std(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df = df[np.isfinite(df['diff'])]
        df['std_bin'] = pd.cut(df['series_std'], bins=STD_BINS, labels=bin_order, right=False)
        positions = [j + (i - n_models / 2 + 0.5) * width for j in range(n_bins)]
        data_by_bin = [df[df['std_bin'] == b]['diff'].values for b in bin_order]
        bp = ax.boxplot(data_by_bin, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color_cycle[i % len(color_cycle)], alpha=0.6),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0))
        bp['boxes'][0].set_label(model_name)
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_yscale('symlog')
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_order, rotation=30)
    ax.set_xlabel('std of actual (per series, binned)', fontsize=13)
    ax.set_ylabel('predicted - actual', fontsize=13)
    ax.set_title(f'Error distribution per std bin  [{quantile_col}]', fontsize=14, fontweight='bold')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_cycle[i % len(color_cycle)], alpha=0.6)
               for i in range(n_models)]
    ax.legend(handles, models, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    fname = f'{output_dir}/binned_boxplot.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

for quantile in QUANTILES:
    std_dir = f'metric_comparison/{quantile}'
    os.makedirs(std_dir, exist_ok=True)
    stats_df = build_binned_stats(pred_dfs, models, quantile)
    for metric in ['mae', 'rmse', 'median_err']:
        plot_binned_line(stats_df, metric, quantile, std_dir)
    plot_binned_boxplot(pred_dfs, models, quantile, std_dir)
    print()

print("Done.")
