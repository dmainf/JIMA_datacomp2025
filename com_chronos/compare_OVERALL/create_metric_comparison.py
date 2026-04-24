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

marker_pool = ['p', 'D', 'o', 's', '^', 'v', 'X', 'P']

def get_color(name):
    return '#d6336c'

csv_files = sorted(glob.glob(f'{base_path}/*/predictions_all.csv'))

ANOMALY_THRESHOLD = 1e6
QUANTILES = ['q0.1', 'q0.5', 'q0.9']
raw_metrics = ['MAE', 'RMSE']

label_map = {'MAE': r'MAE', 'RMSE': r'RMSE'}
metric_direction = {m: 'lower' for m in raw_metrics}

models = []
colors = []
markers = []
pred_dfs = []
for i, csv_path in enumerate(csv_files):
    dir_name = os.path.basename(os.path.dirname(csv_path))
    models.append(dir_name)
    colors.append(get_color(dir_name))
    markers.append(marker_pool[i % len(marker_pool)])
    df = pd.read_csv(csv_path).rename(columns={'書名': 'book_name'})
    pred_dfs.append(df)

def compute_metrics(df, quantile_col):
    df = df.copy()
    df['ae'] = (df['actual'] - df[quantile_col]).abs()
    df['se'] = (df['actual'] - df[quantile_col]) ** 2
    bad_books = set(df.loc[~np.isfinite(df['ae']) | (df['ae'] > ANOMALY_THRESHOLD), 'book_name'])
    df = df[~df['book_name'].isin(bad_books)]
    return {'MAE': df['ae'].mean(), 'RMSE': np.sqrt(df['se'].mean())}

def plot_single(metric, values, models, colors, markers, output_dir):
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 6))
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
    for xj, val, marker, color in zip(x, values, markers, colors):
        ax.plot(xj, val, marker=marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)
    if metric_direction[metric] == 'lower':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)
    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for bar, value in zip(bars, values):
        y_offset = bar.get_height() + y_range * 0.05
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{value:.4f}', ha='center', va='bottom')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.25)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel(label_map[metric], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{metric}.png")

def plot_combined(all_means, raw_metrics, models, colors, markers, output_dir):
    n = len(raw_metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
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
        if metric_direction[metric] == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        y_range = max(values) - min(values) if max(values) != min(values) else max(values)
        for bar, value in zip(bars, values):
            y_offset = bar.get_height() + y_range * 0.05
            ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
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
    print(f"Combined plot saved: {output_dir}/all_metrics_combined.png")

all_quantile_metrics = {}
for quantile in QUANTILES:
    output_dir = f'metric_comparison/{quantile}'
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = [compute_metrics(df, quantile) for df in pred_dfs]
    all_quantile_metrics[quantile] = all_metrics
    all_means = [[m[metric] for metric in raw_metrics] for m in all_metrics]
    for i, metric in enumerate(raw_metrics):
        values = [means[i] for means in all_means]
        plot_single(metric, values, models, colors, markers, output_dir)
    plot_combined(all_means, raw_metrics, models, colors, markers, output_dir)
    print()

def generate_latex_table(all_quantile_metrics, models, quantiles, metrics):
    quantile_label = {'q0.1': r'$\tau=0.1$', 'q0.5': r'$\tau=0.5$', 'q0.9': r'$\tau=0.9$'}
    model_label = {
        'GateRAF': r'\textbf{GateRAF}',
        'Multi-RAF': r'\textbf{Multi-RAF}',
        'ProtoRAF': r'\textbf{ProtoRAF}',
        'baseline': r'\textbf{Baseline}',
    }

    best_vals = {}
    for q in quantiles:
        for m in metrics:
            vals = [all_quantile_metrics[q][i][m] for i in range(len(models))]
            best_vals[(q, m)] = min(vals)

    ncols = 1 + len(quantiles) * len(metrics)
    col_spec = 'l' + ''.join(['rr'] * len(quantiles))

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'  \caption{Comparison of MAE and RMSE for each model and quantile level.}')
    lines.append(r'  \label{tab:metric_comparison}')
    lines.append(r'  \centering')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \hline')

    # Spanning header row: empty cell + "Quantile" spanning all metric columns
    n_metric_cols = len(quantiles) * len(metrics)
    lines.append(f'    & \\multicolumn{{{n_metric_cols}}}{{c}}{{Quantile}} \\\\')

    # Quantile sub-headers with cmidrule
    cmidrule = '    '
    col_start = 2
    for q in quantiles:
        cmidrule += f'\\cline{{{col_start}-{col_start + len(metrics) - 1}}} '
        col_start += len(metrics)
    lines.append(cmidrule)

    header_q = r'    \multicolumn{1}{c}{Model}'
    for q in quantiles:
        header_q += f' & \\multicolumn{{{len(metrics)}}}{{c}}{{{quantile_label[q]}}}'
    header_q += r' \\'
    lines.append(header_q)

    cmidrule2 = '    '
    col_start = 2
    for q in quantiles:
        cmidrule2 += f'\\cline{{{col_start}-{col_start + len(metrics) - 1}}} '
        col_start += len(metrics)
    lines.append(cmidrule2)

    header_m = '    '
    for q in quantiles:
        for m in metrics:
            header_m += f' & \\multicolumn{{1}}{{c}}{{{m}}}'
    header_m += r' \\'
    lines.append(header_m)
    lines.append(r'    \hline\hline')

    for i, model in enumerate(models):
        row = f'    {model_label.get(model, model)}'
        for q in quantiles:
            for m in metrics:
                val = all_quantile_metrics[q][i][m]
                cell = f'{val:.4f}'
                if abs(val - best_vals[(q, m)]) < 1e-9:
                    cell = f'\\textbf{{{cell}}}'
                row += f' & {cell}'
        row += r' \\'
        lines.append(row)

    lines.append(r'    \hline')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)

latex_str = generate_latex_table(all_quantile_metrics, models, QUANTILES, raw_metrics)
latex_path = 'metric_comparison/table_metric_comparison.tex'
with open(latex_path, 'w') as f:
    f.write(latex_str + '\n')
print(f"LaTeX table saved: {latex_path}")
print("Done.")
