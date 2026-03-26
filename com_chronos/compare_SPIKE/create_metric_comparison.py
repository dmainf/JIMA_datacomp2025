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

ANOMALY_THRESHOLD = 1e6
SPIKE_ZSCORES = [1.5, 2.0, 2.5, 3.0]
QUANTILES = ['q0.1', 'q0.5', 'q0.9']

marker_pool = ['p', 'D', 'o', 's', '^', 'v', 'X', 'P']

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


def label_spikes(df, zscore_threshold=2.0):
    """Per-series z-score spike labeling."""
    df = df.copy()
    stats = df.groupby('book_name')['actual'].agg(['mean', 'std'])
    df = df.join(stats, on='book_name')
    df['zscore'] = np.where(df['std'] > 0, (df['actual'] - df['mean']) / df['std'], 0.0)
    df['is_spike'] = df['zscore'] > zscore_threshold
    return df


def compute_metrics(df, quantile_col, spike_zscore=2.0):
    df = df.copy()
    ae = (df['actual'] - df[quantile_col]).abs()
    bad_books = set(df.loc[~np.isfinite(ae) | (ae > ANOMALY_THRESHOLD), 'book_name'])
    df = df[~df['book_name'].isin(bad_books)]
    df = label_spikes(df, zscore_threshold=spike_zscore)

    spike = df[df['is_spike']]
    normal = df[~df['is_spike']]

    mae_spike  = (spike['actual'] - spike[quantile_col]).abs().mean()
    mae_normal = (normal['actual'] - normal[quantile_col]).abs().mean()
    mae_all    = (df['actual'] - df[quantile_col]).abs().mean()

    # Spike detection: predict spike when prediction > per-series (mean + spike_zscore * std)
    df['spike_threshold'] = df['mean'] + spike_zscore * df['std']
    df['pred_spike'] = df[quantile_col] > df['spike_threshold']

    tp = (df['is_spike'] & df['pred_spike']).sum()
    fp = (~df['is_spike'] & df['pred_spike']).sum()
    fn = (df['is_spike'] & ~df['pred_spike']).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    pred_normal = ~df['pred_spike']
    is_normal   = ~df['is_spike']
    tp_n = (is_normal & pred_normal).sum()
    fp_n = (df['is_spike'] & pred_normal).sum()
    fn_n = (is_normal & df['pred_spike']).sum()
    precision_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    recall_n    = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1_n        = 2 * precision_n * recall_n / (precision_n + recall_n) if (precision_n + recall_n) > 0 else 0.0

    return {
        'MAE_spike':    mae_spike,
        'MAE_normal':   mae_normal,
        'MAE_all':      mae_all,
        'Precision':    precision,
        'Recall':       recall,
        'F1':           f1,
        'Precision_n':  precision_n,
        'Recall_n':     recall_n,
        'F1_n':         f1_n,
        'n_spike':      int(df['is_spike'].sum()),
        'n_total':      len(df),
    }


def plot_conditional_mae(all_metrics, models, colors, markers, quantile_col, output_dir):
    mae_spike  = [m['MAE_spike']  for m in all_metrics]
    mae_normal = [m['MAE_normal'] for m in all_metrics]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    bars1 = ax.bar(x - width/2, mae_spike,  width, label='Spike (z>2)',  color='#e63946', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_normal, width, label='Normal',       color='#457b9d', alpha=0.8)
    for xj, val, marker, c in zip(x - width/2, mae_spike, markers, colors):
        ax.plot(xj, val, marker=marker, color='#e63946', markersize=10,
                markeredgecolor='black', markeredgewidth=1.2)
    for xj, val, marker, c in zip(x + width/2, mae_normal, markers, colors):
        ax.plot(xj, val, marker=marker, color='#457b9d', markersize=10,
                markeredgecolor='black', markeredgewidth=1.2)
    for bar, val in zip(bars1, mae_spike):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, mae_normal):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
    ax.set_title(f'Conditional MAE: Spike vs Normal  [{quantile_col}]', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/conditional_MAE.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/conditional_MAE.png")


def plot_improvement(all_metrics, models, colors, markers, quantile_col, output_dir, mode='spike'):
    assert mode in ('spike', 'normal')
    key = 'MAE_spike' if mode == 'spike' else 'MAE_normal'
    label_sup = 'spike' if mode == 'spike' else 'normal'
    fname = f'{mode}_improvement.png'
    baseline = next(m[key] for m, name in zip(all_metrics, models) if name == 'noRAF')
    improvements = []
    for m, name in zip(all_metrics, models):
        improvements.append((baseline - m[key]) / baseline * 100)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 6))
    bar_colors = ['#2a9d8f' if v >= 0 else '#e63946' for v in improvements]
    bars = ax.bar(x, improvements, color=bar_colors, alpha=0.8, width=0.6)
    for xj, val, marker in zip(x, improvements, markers):
        ax.plot(xj, val, marker=marker, color='black', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)
    y_range = max(improvements) - min(improvements) if max(improvements) != min(improvements) else abs(max(improvements))
    for bar, val in zip(bars, improvements):
        offset = y_range * 0.05 if val >= 0 else -y_range * 0.12
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{val:+.1f}%', ha='center', va='bottom')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel(f'MAE {label_sup} Improvement over baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{"Spike" if mode == "spike" else "Normal"} MAE Improvement  [{quantile_col}]\n'
        r'(vs noRAF)',
        fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{fname}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{fname}")


def plot_f1_normal(all_metrics, models, colors, markers, quantile_col, output_dir):
    precision = [m['Precision_n'] for m in all_metrics]
    recall    = [m['Recall_n']    for m in all_metrics]
    f1        = [m['F1_n']        for m in all_metrics]

    n = len(models)
    width = min(0.22, 0.7 / 3)
    figw = max(8, n * 2.2)
    label_fs = max(7, min(10, int(80 / n)))

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(figw, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2a9d8f', alpha=0.8)
    ax.bar(x,         recall,    width, label='Recall',    color='#e9c46a', alpha=0.8)
    ax.bar(x + width, f1,        width, label='F1',        color='#e76f51', alpha=0.8)

    all_vals = precision + recall + f1
    ymax = max(all_vals) if all_vals else 1.0
    ax.set_ylim(0, ymax * 1.2)

    for vals, offset in [(precision, -width), (recall, 0), (f1, width)]:
        for xj, val in zip(x + offset, vals):
            ax.text(xj, val + ymax * 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=label_fs, rotation=45 if n > 5 else 0)

    best_f1 = np.argmax(f1)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Normal Detection: Precision / Recall / F1  [{quantile_col}]',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=max(9, min(13, int(100 / n))),
                       rotation=30 if n > 5 else 0, ha='right' if n > 5 else 'center')
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_f1:
            label.set_fontweight('bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normal_detection_F1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/normal_detection_F1.png")


def plot_spike_improvement(all_metrics, models, colors, markers, quantile_col, output_dir):
    plot_improvement(all_metrics, models, colors, markers, quantile_col, output_dir, mode='spike')


def plot_f1(all_metrics, models, colors, markers, quantile_col, output_dir):
    precision = [m['Precision'] for m in all_metrics]
    recall    = [m['Recall']    for m in all_metrics]
    f1        = [m['F1']        for m in all_metrics]

    n = len(models)
    width = min(0.22, 0.7 / 3)
    figw = max(8, n * 2.2)
    label_fs = max(7, min(10, int(80 / n)))

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(figw, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2a9d8f', alpha=0.8)
    ax.bar(x,         recall,    width, label='Recall',    color='#e9c46a', alpha=0.8)
    ax.bar(x + width, f1,        width, label='F1',        color='#e76f51', alpha=0.8)

    all_vals = precision + recall + f1
    ymax = max(all_vals) if all_vals else 1.0
    ax.set_ylim(0, ymax * 1.2)

    for vals, offset in [(precision, -width), (recall, 0), (f1, width)]:
        for xj, val in zip(x + offset, vals):
            ax.text(xj, val + ymax * 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=label_fs, rotation=45 if n > 5 else 0)

    best_f1 = np.argmax(f1)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Spike Detection: Precision / Recall / F1  [{quantile_col}]',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=max(9, min(13, int(100 / n))),
                       rotation=30 if n > 5 else 0, ha='right' if n > 5 else 'center')
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_f1:
            label.set_fontweight('bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spike_detection_F1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/spike_detection_F1.png")


for spike_z in SPIKE_ZSCORES:
    for quantile in QUANTILES:
        out_dir = f'metric_comparison/z{spike_z}/{quantile}'
        os.makedirs(out_dir, exist_ok=True)
        all_metrics = [compute_metrics(df, quantile, spike_zscore=spike_z) for df in pred_dfs]

        plot_conditional_mae(all_metrics, models, colors, markers, quantile, out_dir)
        plot_spike_improvement(all_metrics, models, colors, markers, quantile, out_dir)
        plot_improvement(all_metrics, models, colors, markers, quantile, out_dir, mode='normal')
        plot_f1(all_metrics, models, colors, markers, quantile, out_dir)
        plot_f1_normal(all_metrics, models, colors, markers, quantile, out_dir)

        print(f"\n[z>{spike_z}  {quantile}] Summary:")
        n_spike = all_metrics[0]['n_spike']
        n_total = all_metrics[0]['n_total']
        print(f"  spike timesteps: {n_spike}/{n_total} ({100*n_spike/n_total:.1f}%)")
        baseline_spike = next(m['MAE_spike'] for m, name in zip(all_metrics, models) if name == 'noRAF')
        print(f"  {'Model':<20} {'MAE_spike':>10} {'MAE_normal':>10} {'Improv%':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        for model, m in zip(models, all_metrics):
            improv = (baseline_spike - m['MAE_spike']) / baseline_spike * 100
            print(f"  {model:<20} {m['MAE_spike']:>10.4f} {m['MAE_normal']:>10.4f} "
                  f"{improv:>+8.1f}% {m['Precision']:>7.3f} {m['Recall']:>7.3f} {m['F1']:>7.3f}")
    print()

print("Done.")
