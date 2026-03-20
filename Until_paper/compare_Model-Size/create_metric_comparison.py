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

model_info = {
    'tiny(DoRA)+RAF':  {'params': 9e6,   'label': 'Tiny (9M)'},
    'mini(DoRA)+RAF':  {'params': 21e6,  'label': 'Mini (21M)'},
    'small(DoRA)+RAF': {'params': 48e6,  'label': 'Small (48M)'},
    'base(DoRA)+RAF':  {'params': 205e6, 'label': 'Base (205M)'},
}

color_pool = ['#9467bd', '#ff9896', '#e377c2', '#17becf', '#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
marker_pool = ['p', 'D', 'o', 's', '^', 'v', 'X', 'P']

csv_files = sorted(glob.glob(f'{base_path}/*/evaluation_all.csv'))

models = []
params = []
colors = []
markers = []
dfs = []
for i, csv_path in enumerate(csv_files):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    params.append(model_info[dir_name]['params'])
    colors.append(color_pool[i % len(color_pool)])
    markers.append(marker_pool[i % len(marker_pool)])
    dfs.append(pd.read_csv(csv_path))

sort_idx = np.argsort(params)
models = [models[i] for i in sort_idx]
params = [params[i] for i in sort_idx]
colors = [colors[i] for i in sort_idx]
markers = [markers[i] for i in sort_idx]
dfs = [dfs[i] for i in sort_idx]

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
metrics_cols = ['MAE', 'RMSE', 'wQL_0.1', 'wQL_0.9']

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
    for metric in ['wQL_0.1', 'wQL_0.9']:
        if metric in df.columns:
            results[metric] = df[metric].sum() / total_sales
    for metric in ['MAE', 'RMSE']:
        if metric in df.columns:
            results[metric] = df[metric].mean()
    return results

all_metrics = [calculate_global_metrics(df) for df in dfs]
all_means = [[m[metric] for metric in raw_metrics] for m in all_metrics]

x_params = np.array(params)

for i, metric in enumerate(raw_metrics):
    fig, ax = plt.subplots(figsize=(7, 6))
    values = [means[i] for means in all_means]

    ax.plot(x_params, values, color='#1f77b4', linewidth=2, alpha=0.7, zorder=1)
    for j, (xj, val) in enumerate(zip(x_params, values)):
        ax.plot(xj, val, marker=markers[j], color=colors[j], markersize=12,
                markeredgecolor='black', markeredgewidth=1.5, zorder=2)

    direction = metric_direction[metric]
    if direction == 'lower':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)

    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for j, (xj, val) in enumerate(zip(x_params, values)):
        text_str = f'{val:.4f}'
        ax.annotate(text_str, (xj, val), textcoords="offset points",
                    xytext=(0, 22), ha='center', fontsize=22)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.25)

    ax.set_xscale('log')
    ax.set_xlim(left=5.5e6, right=3.2e8)
    ax.set_xlabel('Parameters', fontsize=26, fontweight='bold')
    ax.set_ylabel(label_map[metric], fontsize=26, fontweight='bold')

    ax.set_xticks(x_params)
    ax.set_xticklabels([model_info[m]['label'] for m in models], fontsize=22)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='y', labelsize=22)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', alpha=0.2, linestyle=':')

    legend_elements = [
        plt.Line2D([0], [0], marker=markers[j], color='w', markerfacecolor=colors[j],
                   markersize=16, label=model_info[models[j]]['label'],
                   markeredgecolor='black', markeredgewidth=1.5)
        for j in range(len(models))
    ]
    ax.legend(handles=legend_elements, fontsize=20, frameon=True, edgecolor='black')

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

    ax.plot(x_params, values, color='#1f77b4', linewidth=2, alpha=0.7, zorder=1)
    for j, (xj, val) in enumerate(zip(x_params, values)):
        ax.plot(xj, val, marker=markers[j], color=colors[j], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=2)

    direction = metric_direction[metric]
    if direction == 'lower':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)

    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for j, (xj, val) in enumerate(zip(x_params, values)):
        text_str = f'{val:.4f}'
        ax.annotate(text_str, (xj, val), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters', fontsize=11)
    ax.set_ylabel(label_map[metric], fontsize=11)
    ax.set_title(f'{label_map[metric]}', fontsize=12)

    ax.set_xticks(x_params)
    ax.set_xticklabels([model_info[m]['label'] for m in models], fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', alpha=0.2, linestyle=':')

for i in range(len(raw_metrics), len(axes)):
    axes[i].axis('off')

legend_elements = [
    plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i],
               markersize=10, label=model_info[models[i]]['label'], markeredgecolor='black', markeredgewidth=1.5)
    for i in range(len(models))
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=10,
           frameon=True, edgecolor='black', ncol=4, bbox_to_anchor=(0.98, 0.02))

plt.suptitle('Comparison of All Metrics vs Model Size', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.03, 1, 0.99])
plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCombined plot saved: {output_dir}/all_metrics_combined.png")
