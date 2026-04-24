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


PAPER_GRAY_MAP = {
    'GateRAF':  '#000000',
    'Multi-RAF':'#444444',
    'ProtoRAF': '#888888',
    'noRAF':    '#aaaaaa',
}
PAPER_MARKER_MAP = {
    'GateRAF':  'o',
    'Multi-RAF':'s',
    'ProtoRAF': '^',
    'noRAF':    'D',
}

COLOR_MAP = {
    'GateRAF':  '#e63946',
    'Multi-RAF':'#2a9d8f',
    'ProtoRAF': '#f4a261',
    'noRAF':    '#457b9d',
}
LS_MAP = {
    'GateRAF':  '-',
    'Multi-RAF':'--',
    'ProtoRAF': '-.',
    'noRAF':    ':',
}


def prepare_df(df, quantile_col, spike_z):
    df = df.copy()
    ae = (df['actual'] - df[quantile_col]).abs()
    bad_books = set(df.loc[~np.isfinite(ae) | (ae > ANOMALY_THRESHOLD), 'book_name'])
    df = df[~df['book_name'].isin(bad_books)].copy()
    stats = df.groupby('book_name')['actual'].agg(['mean', 'std'])
    df = df.join(stats, on='book_name')
    df['zscore'] = np.where(df['std'] > 0, (df['actual'] - df['mean']) / df['std'], 0.0)
    df['is_spike'] = df['zscore'] > spike_z
    df['interval_width'] = df['q0.9'] - df['q0.1']
    df['spike_threshold'] = df['mean'] + spike_z * df['std']
    df['pred_spike'] = df[quantile_col] > df['spike_threshold']
    df['series_cv'] = df['std'] / df['mean'].replace(0, np.nan)
    return df


def plot_interval_calibration(pred_dfs, models, quantile_col, spike_z, output_dir):
    N_BINS = 10
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, quantile_col, spike_z)
        dfw = dfw[np.isfinite(dfw['interval_width']) & (dfw['interval_width'] >= 0)]
        dfw['iw_bin'] = pd.qcut(dfw['interval_width'], q=N_BINS, labels=False, duplicates='drop')
        grp = dfw.groupby('iw_bin').agg(
            bin_center=('interval_width', 'median'),
            mean_zscore=('zscore', 'mean'),
        ).reset_index()
        ax.plot(grp['bin_center'], grp['mean_zscore'],
                marker='o', linewidth=1.8,
                linestyle=LS_MAP.get(model_name, '-'),
                color=COLOR_MAP.get(model_name, None),
                label=model_name)
    ax.set_xlabel('Prediction interval width (q0.9 − q0.1)', fontsize=12)
    ax.set_ylabel(r'Mean actual $z$-score', fontsize=12)
    ax.set_title(f'Mean $z$-score by interval width  [{quantile_col}]',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    fname = f'{output_dir}/interval_calibration.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")
    print(f"  [corr interval_width vs zscore]")
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, quantile_col, spike_z)
        dfw = dfw[np.isfinite(dfw['interval_width']) & (dfw['interval_width'] >= 0)]
        r = np.corrcoef(dfw['interval_width'], dfw['zscore'])[0, 1]
        print(f"    {model_name:<12} corr(iw,zscore)={r:.4f}")


def plot_temporal_proximity(pred_dfs, models, quantile_col, spike_z, output_dir):
    K = 10
    bins = np.arange(-K - 0.5, K + 1.5, 1)
    bin_centers = np.arange(-K, K + 1)
    fig_fp, ax_fp = plt.subplots(figsize=(10, 5))
    fig_fn, ax_fn = plt.subplots(figsize=(10, 5))
    print(f"  [temporal proximity] z>{spike_z} {quantile_col}")
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, quantile_col, spike_z)
        fp_dists, fn_dists = [], []
        for book, grp in dfw.groupby('book_name'):
            grp = grp.sort_values('day').reset_index(drop=True)
            actual_arr = np.where(grp['is_spike'].values)[0]
            pred_arr   = np.where(grp['pred_spike'].values)[0]
            if len(actual_arr) == 0 or len(pred_arr) == 0:
                continue
            for pi in pred_arr:
                dists = actual_arr - pi
                nearest = dists[np.argmin(np.abs(dists))]
                if not grp.loc[pi, 'is_spike']:
                    fp_dists.append(nearest)
            for ai in actual_arr:
                if grp.loc[ai, 'pred_spike']:
                    continue
                dists = ai - pred_arr
                nearest = dists[np.argmin(np.abs(dists))]
                fn_dists.append(nearest)
        color = COLOR_MAP.get(model_name, None)
        ls = LS_MAP.get(model_name, '-')
        for ax, arr in [(ax_fp, fp_dists), (ax_fn, fn_dists)]:
            arr = np.array(arr)
            if len(arr) == 0:
                continue
            hist, _ = np.histogram(np.clip(arr, -K, K), bins=bins)
            norm = hist / hist.sum()
            ax.plot(bin_centers, norm, marker='o', linewidth=1.6, markersize=5,
                    linestyle=ls, color=color, label=f'{model_name} (n={len(arr)})')
        near3_fp = (np.abs(np.array(fp_dists)) <= 3).mean() if fp_dists else float('nan')
        near3_fn = (np.abs(np.array(fn_dists)) <= 3).mean() if fn_dists else float('nan')
        print(f"    {model_name:<12} FP±3={near3_fp:.3f}  FN±3={near3_fn:.3f}"
              f"  (nFP={len(fp_dists)}, nFN={len(fn_dists)})")
    for ax, xlabel in [
        (ax_fp, 'Δdays (nearest actual spike day − predicted spike day)'),
        (ax_fn, 'Δdays (actual spike day − nearest predicted spike day)'),
    ]:
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Fraction of predictions', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
    fig_fp.tight_layout(); fig_fn.tight_layout()
    for fig, tag in [(fig_fp, 'fp'), (fig_fn, 'fn')]:
        fname = f'{output_dir}/proximity_{tag}.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.close('all')


def plot_cv_vs_f1(pred_dfs, models, quantile_col, spike_z, output_dir):
    CV_BINS   = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    CV_LABELS = ['<0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3+']
    records = []
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, quantile_col, spike_z)
        for book, grp in dfw.groupby('book_name'):
            tp = (grp['is_spike'] & grp['pred_spike']).sum()
            fp = (~grp['is_spike'] & grp['pred_spike']).sum()
            fn = (grp['is_spike'] & ~grp['pred_spike']).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            cv   = grp['series_cv'].iloc[0]
            records.append({'model': model_name, 'cv': cv, 'f1': f1,
                            'precision': prec, 'recall': rec,
                            'n_spike': int(grp['is_spike'].sum())})
    ps_df = pd.DataFrame(records)
    ps_df = ps_df[np.isfinite(ps_df['cv'])]
    ps_df['cv_bin'] = pd.cut(ps_df['cv'], bins=CV_BINS, labels=CV_LABELS, right=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, ylabel in zip(axes,
                                  ['f1', 'precision', 'recall'],
                                  ['F1', 'Precision', 'Recall']):
        for model_name in models:
            sub = ps_df[(ps_df['model'] == model_name) & (ps_df['n_spike'] >= 1)]
            grp = sub.groupby('cv_bin', observed=True)[metric].mean().reset_index()
            ax.plot(grp['cv_bin'], grp[metric],
                    marker='o', linewidth=1.8,
                    linestyle=LS_MAP.get(model_name, '-'),
                    color=COLOR_MAP.get(model_name, None),
                    label=model_name)
        ax.set_xlabel('Per-series CV (std / mean)', fontsize=12)
        ax.set_ylabel(f'Mean {ylabel}', fontsize=12)
        ax.set_title(f'{ylabel} vs series CV  [z>{spike_z}, {quantile_col}]',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        plt.setp(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    fname = f'{output_dir}/cv_vs_f1.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")
    pivot = ps_df[ps_df['n_spike'] >= 1].groupby(
        ['model', 'cv_bin'], observed=True)['f1'].mean().unstack()
    print(f"  [cv_vs_f1] Mean F1 per CV bin (z>{spike_z}, {quantile_col}, series with >=1 spike):")
    print(pivot.to_string())


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

CHAR_Z = 2.0
print("\n" + "=" * 60)
print("Characterization analyses")
print("=" * 60)
for quantile in QUANTILES:
    char_dir = f'metric_comparison/z{CHAR_Z}/{quantile}/characterization'
    os.makedirs(char_dir, exist_ok=True)
    print(f"\n--- {quantile} ---")
    plot_interval_calibration(pred_dfs, models, quantile, CHAR_Z, char_dir)
    plot_temporal_proximity(pred_dfs, models, quantile, CHAR_Z, char_dir)
    plot_cv_vs_f1(pred_dfs, models, quantile, CHAR_Z, char_dir)

Z_BIN_EDGES  = [-np.inf, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
Z_BIN_LABELS = ['<-1', '-1–0', '0–0.5', '0.5–1', '1–1.5', '1.5–2', '2–3', '>3']


def plot_mae_vs_zscore(pred_dfs, models, colors, markers, quantiles, output_dir):
    ncols = len(quantiles)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    axes = np.array(axes).flatten()

    mae_table = {}
    rmse_table = {}
    for qi, quantile in enumerate(quantiles):
        ax = axes[qi]
        mae_table[quantile] = {}
        rmse_table[quantile] = {}
        for mi, (model_name, df) in enumerate(zip(models, pred_dfs)):
            dfw = prepare_df(df, quantile, spike_z=CHAR_Z)
            ae = (dfw['actual'] - dfw[quantile]).abs()
            bad_books = set(dfw.loc[~np.isfinite(ae) | (ae > ANOMALY_THRESHOLD), 'book_name'])
            dfw = dfw[~dfw['book_name'].isin(bad_books)].copy()
            dfw['ae'] = (dfw['actual'] - dfw[quantile]).abs()
            dfw['se'] = (dfw['actual'] - dfw[quantile]) ** 2
            dfw['z_bin'] = pd.cut(dfw['zscore'], bins=Z_BIN_EDGES, labels=Z_BIN_LABELS)
            agg = dfw.groupby('z_bin', observed=False).agg(
                mae=('ae', 'mean'),
                mse=('se', 'mean'),
                zmid=('zscore', 'median'),
            ).reset_index()
            agg['rmse'] = np.sqrt(agg['mse'])
            valid = np.isfinite(agg['mae'].values)
            ax.plot(np.arange(len(Z_BIN_LABELS))[valid], agg['mae'].values[valid],
                    marker=markers[mi % len(markers)],
                    color=COLOR_MAP.get(model_name, f'C{mi}'),
                    linestyle=LS_MAP.get(model_name, '-'),
                    linewidth=1.8, markersize=7, label=model_name)
            mae_table[quantile][model_name] = {
                str(bl): float(v) for bl, v in zip(agg['z_bin'], agg['mae'])
            }
            rmse_table[quantile][model_name] = {
                str(bl): float(v) for bl, v in zip(agg['z_bin'], agg['rmse'])
            }
        ax.set_xticks(np.arange(len(Z_BIN_LABELS)))
        ax.set_xticklabels(Z_BIN_LABELS, rotation=30, ha='right', fontsize=9)
        ax.set_xlabel('Within-series z-score of actual value', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(r'$\tau=' + quantile[1:] + r'$', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('MAE by Within-series Z-score of Actual Value',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f'{output_dir}/mae_vs_zscore.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")
    return mae_table, rmse_table


def generate_latex_metric_vs_zscore(all_table, models, quantiles, metric):
    model_label = {
        'GateRAF':  r'\textbf{GateRAF}',
        'Multi-RAF': r'\textbf{Multi-RAF}',
        'ProtoRAF': r'\textbf{ProtoRAF}',
        'noRAF':    r'\textbf{Baseline}',
    }
    quantile_label = {'q0.1': r'$\tau=0.1$', 'q0.5': r'$\tau=0.5$', 'q0.9': r'$\tau=0.9$'}
    bin_labels = Z_BIN_LABELS
    n_bins = len(bin_labels)

    def tex_label(bl):
        return bl.replace('–', '--').replace('<', '$<$').replace('>', '$>$')

    col_spec = 'l' + 'r' * n_bins
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(f'    \\caption{{{metric} stratified by within-series z-score of actual sales.}}')
    lines.append(f'    \\label{{tab:{metric.lower()}_vs_zscore}}')
    lines.append(r'    \centering')
    lines.append(f'    \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \hline')
    lines.append(f'    & \\multicolumn{{{n_bins}}}{{c}}{{Within-series z-score of actual value}} \\\\')
    lines.append(f'    \\cline{{2-{1 + n_bins}}}')

    header = r'    \multicolumn{1}{c}{Model}'
    for bl in bin_labels:
        header += ' & \\multicolumn{1}{c}{' + tex_label(bl) + '}'
    header += r' \\'
    lines.append(header)
    lines.append(r'    \hline\hline')

    for qi, quantile in enumerate(quantiles):
        if qi > 0:
            lines.append(r'    \hline')
        lines.append(f'    \\multicolumn{{{1 + n_bins}}}{{l}}{{{quantile_label[quantile]}}} \\\\')

        best_per_bin = {}
        for bl in bin_labels:
            vals = [(all_table[quantile].get(m, {}).get(bl, float('nan')), m) for m in models]
            vals = [(v, m) for v, m in vals if np.isfinite(v)]
            if vals:
                best_per_bin[bl] = min(vals, key=lambda x: x[0])[0]

        for model_name in models:
            row = f'    \\quad {model_label.get(model_name, model_name)}'
            for bl in bin_labels:
                v = all_table[quantile].get(model_name, {}).get(bl, float('nan'))
                if np.isfinite(v):
                    cell = f'{v:.2f}'
                    if bl in best_per_bin and abs(v - best_per_bin[bl]) < 1e-9:
                        cell = f'\\textbf{{{cell}}}'
                else:
                    cell = '--'
                row += f' & {cell}'
            row += r' \\'
            lines.append(row)

    lines.append(r'    \hline')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


print("\n" + "=" * 60)
print("MAE/RMSE vs within-series z-score analysis")
print("=" * 60)
zscore_out_dir = 'metric_comparison'
mae_table_z, rmse_table_z = plot_mae_vs_zscore(
    pred_dfs, models, colors, markers, QUANTILES, zscore_out_dir
)
for metric, table in [('MAE', mae_table_z), ('RMSE', rmse_table_z)]:
    latex_str = generate_latex_metric_vs_zscore(table, models, QUANTILES, metric)
    tex_path = f'{zscore_out_dir}/table_{metric.lower()}_vs_zscore.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_str + '\n')
    print(f"LaTeX table saved: {tex_path}")

def plot_temporal_proximity_combined(pred_dfs, models, quantile_col, spike_z, output_dir):
    K = 10
    bins = np.arange(-K - 0.5, K + 1.5, 1)
    bin_centers = np.arange(-K, K + 1)

    table_data = {m: {'fp': np.zeros(len(bin_centers)), 'fn': np.zeros(len(bin_centers))}
                  for m in models}
    plot_data = {m: {'fp': [], 'fn': []} for m in models}
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, quantile_col, spike_z)
        fp_dists, fn_dists = [], []
        for book, grp in dfw.groupby('book_name'):
            grp = grp.sort_values('day').reset_index(drop=True)
            actual_arr = np.where(grp['is_spike'].values)[0]
            pred_arr   = np.where(grp['pred_spike'].values)[0]
            if len(actual_arr) == 0 or len(pred_arr) == 0:
                continue
            for pi in pred_arr:
                dists = actual_arr - pi
                nearest = dists[np.argmin(np.abs(dists))]
                if not grp.loc[pi, 'is_spike']:
                    fp_dists.append(nearest)
            for ai in actual_arr:
                if grp.loc[ai, 'pred_spike']:
                    continue
                dists = ai - pred_arr
                nearest = dists[np.argmin(np.abs(dists))]
                fn_dists.append(nearest)

        color  = COLOR_MAP.get(model_name, '#333333')
        ls     = LS_MAP.get(model_name, '-')
        marker = PAPER_MARKER_MAP.get(model_name, 'o')
        for key, arr_list in [('fp', fp_dists), ('fn', fn_dists)]:
            arr = np.array(arr_list)
            if len(arr) == 0:
                continue
            hist, _ = np.histogram(np.clip(arr, -K, K), bins=bins)
            norm = hist / hist.sum()
            table_data[model_name][key] = norm
            plot_data[model_name][key] = (norm, marker, ls, color, len(arr))

    xlabels = {
        'fp': r'$\Delta$days (nearest actual spike day $-$ predicted spike day)',
        'fn': r'$\Delta$days (actual spike day $-$ nearest predicted spike day)',
    }
    os.makedirs(output_dir, exist_ok=True)
    for key in ('fp', 'fn'):
        fig, ax = plt.subplots(figsize=(6, 4))
        for model_name in models:
            entry = plot_data[model_name].get(key)
            if not entry:
                continue
            norm, marker, ls, color, n = entry
            ax.plot(bin_centers, norm, marker=marker, linewidth=1.6, markersize=4,
                    linestyle=ls, color=color, markevery=2,
                    label=f'{model_name} ($n={n}$)')
        ax.axvline(0, color='black', linestyle='--', linewidth=1.0)
        ax.set_xlabel(xlabels[key], fontsize=11)
        ax.set_ylabel('Fraction', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        if quantile_col == 'q0.5':
            fname_eps = f'{output_dir}/temporal_proximity_{key}_{quantile_col}.eps'
            plt.savefig(fname_eps, bbox_inches='tight', format='eps')
            print(f"Saved: {fname_eps}")
        fname_png = f'{output_dir}/temporal_proximity_{key}_{quantile_col}.png'
        plt.savefig(fname_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname_png}")
        plt.close()
    return bin_centers, table_data


def generate_latex_temporal_proximity_table(bin_centers, table_data_per_q, models, quantiles):
    model_label = {
        'GateRAF':   r'\textbf{GateRAF}',
        'Multi-RAF': r'\textbf{Multi-RAF}',
        'ProtoRAF':  r'\textbf{ProtoRAF}',
        'noRAF':     r'\textbf{Baseline}',
    }
    quantile_label = {'q0.1': r'$\tau=0.1$', 'q0.5': r'$\tau=0.5$', 'q0.9': r'$\tau=0.9$'}
    n_models = len(models)
    col_spec = 'r' + 'r' * n_models
    lines = []
    ref_q = 'q0.5'
    lines.append(r'\begin{table*}[t]')
    lines.append(
        r'    \caption{Fraction of events by $\Delta$days between predicted and actual spike days'
        r' ($\tau=0.5$).}'
    )
    lines.append(r'    \label{tab:temporal_proximity}')
    lines.append(r'    \centering')
    lines.append(f'    \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \hline')
    header = r'    \multicolumn{1}{c}{$\Delta$days}'
    for m in models:
        header += f' & \\multicolumn{{1}}{{c}}{{{model_label.get(m, m)}}}'
    header += r' \\'
    lines.append(header)
    lines.append(r'    \hline\hline')
    table_data = table_data_per_q[ref_q]
    for section_key, section_label in [
        ('fp', r'Predicted spike, no concurrent actual spike'),
        ('fn', r'Actual spike, no concurrent predicted spike'),
    ]:
        lines.append(f'    \\multicolumn{{{1 + n_models}}}{{l}}{{{section_label}}} \\\\')
        for bi, day in enumerate(bin_centers):
            row = f'    {int(day):+d}'
            for m in models:
                v = table_data[m][section_key][bi]
                row += f' & {v:.3f}'
            row += r' \\'
            lines.append(row)
        lines.append(r'    \hline')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


def plot_interval_calibration_paper(pred_dfs, models, spike_z, output_dir):
    N_BINS = 10
    REF_Q = 'q0.5'
    pool = pd.concat([
        prepare_df(df, REF_Q, spike_z).pipe(
            lambda d: d.loc[np.isfinite(d['interval_width']) & (d['interval_width'] >= 0),
                             'interval_width']
        )
        for df in pred_dfs
    ])
    _, bin_edges = pd.qcut(pool, q=N_BINS, retbins=True, duplicates='drop')
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    fig, ax = plt.subplots(figsize=(7, 5))
    model_zscore = {}
    for model_name, df in zip(models, pred_dfs):
        dfw = prepare_df(df, REF_Q, spike_z)
        dfw = dfw[np.isfinite(dfw['interval_width']) & (dfw['interval_width'] >= 0)].copy()
        dfw['iw_bin'] = pd.cut(dfw['interval_width'], bins=bin_edges,
                               labels=False, include_lowest=True)
        grp = dfw.groupby('iw_bin', observed=False).agg(
            plot_center=('interval_width', 'median'),
            mean_zscore=('zscore', 'mean'),
        ).reset_index()
        model_zscore[model_name] = grp['mean_zscore'].tolist()
        ax.plot(grp['plot_center'], grp['mean_zscore'],
                marker=PAPER_MARKER_MAP.get(model_name, 'o'),
                linewidth=1.8, markersize=6,
                linestyle=LS_MAP.get(model_name, '-'),
                color=COLOR_MAP.get(model_name, '#333333'),
                label=model_name)
    ax.set_xlabel(r'Interval width ($q_{0.9} - q_{0.1}$)', fontsize=12)
    ax.set_ylabel(r'Mean actual $z$-score', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext, kw in [('eps', {'format': 'eps'}), ('png', {'dpi': 300})]:
        fname = f'{output_dir}/interval_calibration.{ext}'
        plt.savefig(fname, bbox_inches='tight', **kw)
        print(f"Saved: {fname}")
    plt.close()
    return {'bin_centers': bin_centers, 'model_zscore': model_zscore}


def generate_latex_interval_calibration_table(bin_data, models):
    model_label = {
        'GateRAF':   r'\textbf{GateRAF}',
        'Multi-RAF': r'\textbf{Multi-RAF}',
        'ProtoRAF':  r'\textbf{ProtoRAF}',
        'noRAF':     r'\textbf{Baseline}',
    }
    bin_centers = bin_data['bin_centers']
    n_bins = len(bin_centers)
    col_spec = 'l' + 'r' * n_bins
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(
        r'    \caption{Mean actual $z$-score by prediction interval width bin '
        r'($q_{0.9}-q_{0.1}$, equal-frequency bins).}'
    )
    lines.append(r'    \label{tab:interval_calibration}')
    lines.append(r'    \centering')
    lines.append(f'    \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \hline')
    header = r'    \multicolumn{1}{c}{Model}'
    for bc in bin_centers:
        header += f' & \\multicolumn{{1}}{{c}}{{{bc:.1f}}}'
    header += r' \\'
    lines.append(header)
    lines.append(r'    \hline\hline')
    for model_name in models:
        zscores = bin_data['model_zscore'].get(model_name, [float('nan')] * n_bins)
        row = f'    \\quad {model_label.get(model_name, model_name)}'
        for v in zscores:
            row += f' & {v:.2f}' if np.isfinite(v) else ' & --'
        row += r' \\'
        lines.append(row)
    lines.append(r'    \hline')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


print("\n" + "=" * 60)
print("Temporal proximity paper figure and LaTeX table")
print("=" * 60)
paper_dir = 'metric_comparison'
proximity_table_data = {}
proximity_bin_centers = None
for q in QUANTILES:
    bc, td = plot_temporal_proximity_combined(pred_dfs, models, q, CHAR_Z, paper_dir)
    proximity_bin_centers = bc
    proximity_table_data[q] = td
latex_prox = generate_latex_temporal_proximity_table(
    proximity_bin_centers, proximity_table_data, models, QUANTILES
)
tex_path = f'{paper_dir}/table_temporal_proximity.tex'
with open(tex_path, 'w') as f:
    f.write(latex_prox + '\n')
print(f"LaTeX table saved: {tex_path}")
with open(f'{paper_dir}/figure_temporal_proximity.tex', 'w') as f:
    f.write(
        '\\begin{figure}[t]\n'
        '\\centering\n'
        '\\includegraphics[width=\\linewidth]{temporal_proximity_fp_q0.5.eps}\n'
        '\\caption{キャプション（FP）．}\n'
        '\\label{Fig:temporal_proximity_fp}\n'
        '\\end{figure}\n'
        '\n'
        '\\begin{figure}[t]\n'
        '\\centering\n'
        '\\includegraphics[width=\\linewidth]{temporal_proximity_fn_q0.5.eps}\n'
        '\\caption{キャプション（FN）．}\n'
        '\\label{Fig:temporal_proximity_fn}\n'
        '\\end{figure}\n'
    )
print(f"LaTeX figure snippet saved: {paper_dir}/figure_temporal_proximity.tex")

print("\n" + "=" * 60)
print("Interval calibration paper figure and LaTeX table")
print("=" * 60)
a1_bin_data = plot_interval_calibration_paper(pred_dfs, models, CHAR_Z, paper_dir)
latex_a1 = generate_latex_interval_calibration_table(a1_bin_data, models)
tex_path = f'{paper_dir}/table_interval_calibration.tex'
with open(tex_path, 'w') as f:
    f.write(latex_a1 + '\n')
print(f"LaTeX table saved: {tex_path}")
with open(f'{paper_dir}/figure_interval_calibration.tex', 'w') as f:
    f.write(
        '\\begin{figure}[t]\n'
        '\\centering\n'
        '\\includegraphics[width=\\linewidth]{interval_calibration.eps}\n'
        '\\caption{キャプション．}\n'
        '\\label{Fig:interval_calibration}\n'
        '\\end{figure}\n'
    )
print(f"LaTeX figure snippet saved: {paper_dir}/figure_interval_calibration.tex")

print("\nDone.")
