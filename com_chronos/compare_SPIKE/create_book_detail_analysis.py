import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

fm.fontManager.addfont('/usr/local/texlive/2026/texmf-dist/fonts/truetype/public/ipaex/ipaexm.ttf')
import matplotlib as mpl
mpl.rc('font', family='IPAexMincho')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_PATH, 'metric_comparison')

BOOKS = [
    ('ぼっち・ざ・ろっく！_original',   'ぼっち・ざ・ろっく！',   'figure_book_bocchi_detail'),
    ('デラックスクロスワード_original', 'デラックスクロスワード', 'figure_book_crossword_detail'),
    ('母の待つ里_original',            '母の待つ里',            'figure_book_haha_detail'),
]

MODELS = [
    ('GateRAF',   'GateRAF',   '#d6336c', '-',  'o', 1.6),
    ('Multi-RAF', 'Multi-RAF', '#1971c2', '--', 's', 1.2),
    ('ProtoRAF',  'ProtoRAF',  '#2f9e44', ':',  '^', 1.2),
    ('noRAF',     'Baseline',  '#e67700', '-.', 'D', 1.0),
]

ZBINS = [
    (-np.inf, 0,      '$z<0$'),
    (0,       1,      '$0 \\leq z<1$'),
    (1,       2,      '$1 \\leq z<2$'),
    (2,       np.inf, '$z\\geq 2$'),
]
N_CALIB_BINS = 5
BAR_WIDTH = 0.18

model_dfs = {}
for model_dir, *_ in MODELS:
    df = pd.read_csv(os.path.join(BASE_PATH, model_dir, 'predictions_all.csv'))
    model_dfs[model_dir] = df

for book_key, book_display, stem in BOOKS:
    actual = model_dfs['GateRAF'][model_dfs['GateRAF']['書名'] == book_key]['actual'].values
    mean_a, std_a = actual.mean(), actual.std()
    z = (actual - mean_a) / (std_a + 1e-6)

    # ── 4.3: スパイク強度別 MAE ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(ZBINS))
    for i, (model_dir, label, color, ls, marker, lw) in enumerate(MODELS):
        sub = model_dfs[model_dir][model_dfs[model_dir]['書名'] == book_key].reset_index(drop=True)
        q05 = sub['q0.5'].values
        maes = []
        for lo, hi, _ in ZBINS:
            mask = (z > lo) & (z <= hi)
            maes.append(np.abs(actual[mask] - q05[mask]).mean() if mask.sum() > 0 else np.nan)
        offset = (i - 1.5) * BAR_WIDTH
        ax.bar(x + offset, maes, BAR_WIDTH, color=color, label=label)

    xtick_labels = []
    for lo, hi, lab in ZBINS:
        cnt = int(((z > lo) & (z <= hi)).sum())
        xtick_labels.append(f'{lab}\n(n={cnt})')
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel('MAE')
    ax.set_title(f'{book_display}　スパイク強度別予測精度 ($\\tau=0.5$)')
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    for ext, dpi in [('eps', 300), ('png', 200)]:
        path = os.path.join(OUTPUT_DIR, f'{stem}_zscore.{ext}')
        kw = {'format': 'eps'} if ext == 'eps' else {}
        plt.savefig(path, dpi=dpi, bbox_inches='tight', **kw)
    plt.close()
    print(f'Saved: {stem}_zscore')

    # ── 4.5: 予測区間幅の校正 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for model_dir, label, color, ls, marker, lw in MODELS:
        sub = model_dfs[model_dir][model_dfs[model_dir]['書名'] == book_key].reset_index(drop=True)
        widths = (sub['q0.9'] - sub['q0.1']).values

        sorted_idx = np.argsort(widths)
        bins = np.array_split(sorted_idx, N_CALIB_BINS)
        centers, zscores = [], []
        for b in bins:
            if len(b) > 0:
                centers.append(widths[b].mean())
                zscores.append(z[b].mean())
        ax.plot(centers, zscores, color=color, linestyle=ls, marker=marker,
                linewidth=lw, markersize=5, label=label)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('予測区間幅 ($q_{0.9}-q_{0.1}$)')
    ax.set_ylabel('実績値の$z$スコア平均')
    ax.set_title(f'{book_display}　予測区間幅の校正')
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    for ext, dpi in [('eps', 300), ('png', 200)]:
        path = os.path.join(OUTPUT_DIR, f'{stem}_calib.{ext}')
        kw = {'format': 'eps'} if ext == 'eps' else {}
        plt.savefig(path, dpi=dpi, bbox_inches='tight', **kw)
    plt.close()
    print(f'Saved: {stem}_calib')
