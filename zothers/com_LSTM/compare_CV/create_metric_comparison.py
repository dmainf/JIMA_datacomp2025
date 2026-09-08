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
CV_BINS = [i / 10 for i in range(0, 31)] + [np.inf]
CV_BIN_LABELS = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(0, 30)] + ['3.0+']

models = []
pred_dfs = []
for csv_path in csv_files:
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)

def compute_series_cv(df):
    df = df.copy()
    stats = df.groupby('book_name')['actual'].agg(['std', 'mean'])
    cv_per_book = stats['std'] / stats['mean']
    df['series_cv'] = df['book_name'].map(cv_per_book)
    return df

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def build_binned_stats(dfs, models, quantile_col):
    records = []
    for df, model_name in zip(dfs, models):
        df = compute_series_cv(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df = df[np.isfinite(df['diff']) & np.isfinite(df['series_cv'])]
        df['cv_bin'] = pd.cut(df['series_cv'], bins=CV_BINS, labels=CV_BIN_LABELS, right=False)
        for bin_label in CV_BIN_LABELS:
            subset = df[df['cv_bin'] == bin_label]['diff']
            if len(subset) == 0:
                continue
            subset_full = df[df['cv_bin'] == bin_label]
            mape_vals = subset.abs() / subset_full['actual'].replace(0, np.nan).abs() * 100
            records.append({
                'model': model_name,
                'cv_bin': bin_label,
                'mae': subset.abs().mean(),
                'rmse': np.sqrt((subset ** 2).mean()),
                'median_err': subset.median(),
                'mape': mape_vals[np.isfinite(mape_vals)].mean(),
                'n': len(subset),
                'n_books': subset_full['book_name'].nunique(),
                'diff_values': subset.values,
            })
    return pd.DataFrame(records)

def plot_binned_line(stats_df, metric, quantile_col, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(stats_df['model'].unique()):
        sub = stats_df[stats_df['model'] == model_name]
        ax.plot(sub['cv_bin'], sub[metric], marker='o', linewidth=1.5,
                color=color_cycle[i % len(color_cycle)], label=model_name)
    ax.set_xlabel('CV of actual (per series, binned)', fontsize=13)
    ylabel = {'mae': 'MAE', 'rmse': 'RMSE', 'median_err': 'Median Error', 'mape': 'MAPE (%)'}[metric]
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{ylabel} per CV bin  [{quantile_col}]', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.xticks(rotation=30)
    plt.tight_layout()
    fname = f'{output_dir}/binned_line_{metric}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

def plot_binned_boxplot(dfs, models, quantile_col, output_dir):
    bin_order = CV_BIN_LABELS
    n_bins = len(bin_order)
    n_models = len(models)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (df, model_name) in enumerate(zip(dfs, models)):
        df = compute_series_cv(df.copy())
        df['diff'] = df[quantile_col] - df['actual']
        df = df[np.isfinite(df['diff']) & np.isfinite(df['series_cv'])]
        df['cv_bin'] = pd.cut(df['series_cv'], bins=CV_BINS, labels=bin_order, right=False)
        positions = [j + (i - n_models / 2 + 0.5) * width for j in range(n_bins)]
        df['mape'] = (df['diff'].abs() / df['actual'].replace(0, np.nan).abs() * 100)
        data_by_bin = [df.loc[df['cv_bin'] == b, 'mape'].dropna().values for b in bin_order]
        bp = ax.boxplot(data_by_bin, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color_cycle[i % len(color_cycle)], alpha=0.6),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0))
        bp['boxes'][0].set_label(model_name)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_order, rotation=30)
    ax.set_xlabel('CV of actual (per series, binned)', fontsize=13)
    ax.set_ylabel('MAPE (%)', fontsize=13)
    ax.set_title(f'MAPE distribution per CV bin  [{quantile_col}]', fontsize=14, fontweight='bold')
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
    cv_dir = f'metric_comparison/{quantile}'
    os.makedirs(cv_dir, exist_ok=True)
    stats_df = build_binned_stats(pred_dfs, models, quantile)
    print(f"[{quantile}] data count per bin:")
    first_model = stats_df[stats_df['model'] == models[0]]
    for _, row in first_model.iterrows():
        print(f"  {row['cv_bin']}: {row['n']} rows, {row['n_books']} books")
    for metric in ['mae', 'rmse', 'median_err', 'mape']:
        plot_binned_line(stats_df, metric, quantile, cv_dir)
    plot_binned_boxplot(pred_dfs, models, quantile, cv_dir)
    print()

print("Done.")
