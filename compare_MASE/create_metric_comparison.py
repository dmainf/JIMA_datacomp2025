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

ANOMALY_THRESHOLD = 1e6
QUANTILES = ['q0.1', 'q0.5', 'q0.9']

marker_pool = ['p', 'D', 'o', 's', '^', 'v', 'X', 'P']
raw_metrics = ['MASE_q0.5', 'MASE_q0.1', 'MASE_q0.9']
label_map = {
    'MASE_q0.5': r'MASE (median, $q_{0.5}$)',
    'MASE_q0.1': r'MASE (lower, $q_{0.1}$)',
    'MASE_q0.9': r'MASE (upper, $q_{0.9}$)',
}
metric_direction = {m: 'lower' for m in raw_metrics}

models = []
colors = []
markers = []
pred_dfs = []
for i, csv_path in enumerate(csv_files):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    colors.append('#d6336c')
    markers.append(marker_pool[i % len(marker_pool)])
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)


def compute_metrics(df):
    df = df.copy()
    ae_ref = (df['actual'] - df['q0.5']).abs()
    bad_books = set(df.loc[~np.isfinite(ae_ref) | (ae_ref > ANOMALY_THRESHOLD), 'book_name'])
    df = df[~df['book_name'].isin(bad_books)]

    mase_q05, mase_q01, mase_q09 = [], [], []
    for _, grp in df.groupby('book_name'):
        grp = grp.sort_values('day')
        actual = grp['actual'].values
        naive_mae = np.abs(np.diff(actual)).mean()
        if naive_mae == 0 or not np.isfinite(naive_mae):
            continue
        for col, store in [('q0.5', mase_q05), ('q0.1', mase_q01), ('q0.9', mase_q09)]:
            mae = np.abs(actual - grp[col].values).mean()
            store.append(mae / naive_mae)

    return {
        'MASE_q0.5': np.nanmean(mase_q05),
        'MASE_q0.1': np.nanmean(mase_q01),
        'MASE_q0.9': np.nanmean(mase_q09),
    }


def plot_single(metric, values, models, colors, markers, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
    for xj, val, marker, color in zip(x, values, markers, colors):
        ax.plot(xj, val, marker=marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)
    best_idx = np.argmin(values)
    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Naïve baseline (MASE=1)')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + y_range * 0.05,
                f'{value:.4f}', ha='center', va='bottom', fontsize=22)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.25)
    ax.set_xlabel('Model', fontsize=26, fontweight='bold')
    ax.set_ylabel(label_map[metric], fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{metric}.png")


def plot_combined(all_means, raw_metrics, models, colors, markers, output_dir):
    ncols = 2
    nrows = (len(raw_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i, metric in enumerate(raw_metrics):
        ax = axes[i]
        values = [means[i] for means in all_means]
        x = np.arange(len(models))
        bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
        for xj, val, marker, color in zip(x, values, markers, colors):
            ax.plot(xj, val, marker=marker, color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
        best_idx = np.argmin(values)
        y_range = max(values) - min(values) if max(values) != min(values) else max(values)
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label='Naïve (MASE=1)')
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + y_range * 0.05,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + y_range * 0.15)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(label_map[metric], fontsize=10)
        ax.set_title(label_map[metric], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx == best_idx:
                label.set_fontweight('bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    for i in range(len(raw_metrics), len(axes)):
        axes[i].axis('off')
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i],
                   markersize=10, label=models[i], markeredgecolor='black', markeredgewidth=1.5)
        for i in range(len(models))
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10,
               frameon=True, edgecolor='black', ncol=4, bbox_to_anchor=(0.98, 0.02))
    plt.suptitle('MASE Comparison Across Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: {output_dir}/all_metrics_combined.png")


output_dir = 'metric_comparison'
os.makedirs(output_dir, exist_ok=True)
all_metrics = [compute_metrics(df) for df in pred_dfs]
all_means = [[m[metric] for metric in raw_metrics] for m in all_metrics]
for i, metric in enumerate(raw_metrics):
    values = [means[i] for means in all_means]
    plot_single(metric, values, models, colors, markers, output_dir)
plot_combined(all_means, raw_metrics, models, colors, markers, output_dir)

print("\nSummary table:")
print(f"{'Model':<20}", end='')
for m in raw_metrics:
    print(f"{m:>14}", end='')
print()
for model_name, means in zip(models, all_means):
    print(f"{model_name:<20}", end='')
    for v in means:
        print(f"{v:>14.4f}", end='')
    print()

print("\nNote: MASE denominator = mean |y_t - y_{t-1}| within test period (lag-1 naive baseline).")
print("MASE < 1 means the model outperforms naive random walk.")
print("\nDone.")
