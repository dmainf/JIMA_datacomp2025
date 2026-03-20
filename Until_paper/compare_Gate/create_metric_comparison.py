import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import glob

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11

base_path = '.'

highlight_model = 'Gate-RAF'
color_highlight = '#d6336c'
color_default = '#999999'
marker_pool = ['p', 'D', 'o', 's', '^', 'v', 'X', 'P']

csv_files = sorted(glob.glob(f'{base_path}/*/evaluation_all.csv'))

models = []
colors = []
markers = []
dfs = []
for i, csv_path in enumerate(csv_files):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    colors.append(color_highlight if dir_name == highlight_model else color_default)
    markers.append(marker_pool[i % len(marker_pool)])
    dfs.append(pd.read_csv(csv_path))

output_dir = 'metric_comparison'
os.makedirs(output_dir, exist_ok=True)

raw_metrics = ['MAE', 'RMSE', 'wQL_0.1', 'wQL_0.9']

label_map = {
    'MAE': r'MAE',
    'RMSE': r'RMSE',
    'wQL_0.1': r'$QL_{0.1}$',
    'wQL_0.9': r'$QL_{0.9}$',
}

metric_direction = {
    'MAE': 'lower',
    'RMSE': 'lower',
    'wQL_0.1': 'lower',
    'wQL_0.9': 'lower',
}

ANOMALY_THRESHOLD = 1e6
metrics_cols = ['MAE', 'RMSE', 'wQL_0.1', 'wQL_0.5', 'wQL_0.9', 'wQL_Mean']

exclude_books = set()
for df in dfs:
    existing = [c for c in metrics_cols if c in df.columns]
    bad = ~np.isfinite(df[existing]).all(axis=1) | (df[existing] > ANOMALY_THRESHOLD).any(axis=1)
    exclude_books |= set(df.loc[bad, 'book_name'])
dfs = [df[~df['book_name'].isin(exclude_books)].copy() for df in dfs]
print(f"Excluded {len(exclude_books)} books across all models:")
for b in sorted(exclude_books):
    print(f"  - {b}")

def calculate_global_metrics(df):
    results = {}
    total_sales = df['Total_Sales'].sum()

    for metric in ['wQL_0.1', 'wQL_0.5', 'wQL_0.9', 'wQL_Mean']:
        if metric in df.columns:
            results[metric] = df[metric].sum() / total_sales

    for metric in ['MAE', 'RMSE']:
        if metric in df.columns:
            results[metric] = df[metric].mean()

    return results

all_metrics = [calculate_global_metrics(df) for df in dfs]
all_means = [[m[metric] for metric in raw_metrics] for m in all_metrics]

for i, metric in enumerate(raw_metrics):
    fig, ax = plt.subplots(figsize=(7, 6))

    values = [means[i] for means in all_means]
    x = np.arange(len(models))

    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
    for j, (xj, val, marker, color) in enumerate(zip(x, values, markers, colors)):
        ax.plot(xj, val, marker=marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)

    direction = metric_direction[metric]
    if direction == 'lower':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)

    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_offset = height + y_range * 0.05
        text_str = f'{value:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                text_str, ha='center', va='bottom', fontsize=22)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.25)

    ax.set_xlabel('Model', fontsize=26, fontweight='bold')
    ax.set_ylabel(label_map[metric], fontsize=26, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    filename = f'{metric}'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/{filename}.png")

print(f"\nAll plots saved in '{output_dir}/' directory")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, metric in enumerate(raw_metrics):
    ax = axes[i]

    values = [means[i] for means in all_means]
    x = np.arange(len(models))

    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)

    for j, (xj, val, marker, color) in enumerate(zip(x, values, markers, colors)):
        ax.plot(xj, val, marker=marker, color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)

    direction = metric_direction[metric]
    if direction == 'lower':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)

    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_offset = height + y_range * 0.05
        text_str = f'{value:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                text_str, ha='center', va='bottom', fontsize=8)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel(label_map[metric], fontsize=11)
    ax.set_title(f'{label_map[metric]}', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

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

plt.suptitle('Comparison of All Metrics Across Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.03, 1, 0.99])
plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCombined plot saved: {output_dir}/all_metrics_combined.png")