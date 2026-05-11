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
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOOKS = [
    ('ぼっち・ざ・ろっく！_original', 'ぼっち・ざ・ろっく！'),
    ('デラックスクロスワード_original', 'デラックスクロスワード'),
    ('母の待つ里_original', '母の待つ里'),
]

MODELS = [
    ('GateRAF',   'GateRAF',   '#d6336c', '-',  1.6),
    ('Multi-RAF', 'Multi-RAF', '#1971c2', '--', 1.2),
    ('ProtoRAF',  'ProtoRAF',  '#2f9e44', ':',  1.2),
    ('noRAF',     'Baseline',  '#e67700', '-.', 1.0),
]

model_dfs = {}
for model_dir, *_ in MODELS:
    df = pd.read_csv(os.path.join(BASE_PATH, model_dir, 'predictions_all.csv'))
    model_dfs[model_dir] = df

BOOK_FILENAMES = {
    'ぼっち・ざ・ろっく！_original':   'figure_book_bocchi',
    'デラックスクロスワード_original': 'figure_book_crossword',
    '母の待つ里_original':            'figure_book_haha',
}

for book_key, book_display in BOOKS:
    fig, ax = plt.subplots(figsize=(8, 3.2))

    ref_df = model_dfs['GateRAF']
    sub_ref = ref_df[ref_df['書名'] == book_key].reset_index(drop=True)
    actual = sub_ref['actual'].values
    days = np.arange(len(actual))

    for model_dir, label, color, ls, lw in MODELS:
        sub = model_dfs[model_dir][model_dfs[model_dir]['書名'] == book_key].reset_index(drop=True)
        q05 = sub['q0.5'].values
        mae = np.abs(actual - q05).mean()
        ax.plot(days, q05, color=color, linestyle=ls, linewidth=lw,
                label=f'{label} (MAE={mae:.2f})', zorder=3)

    ax.plot(days, actual, color='black', linewidth=2.0, label='実績', zorder=4)

    ax.set_title(book_display, fontsize=11, pad=4)
    ax.set_xlabel('予測日', fontsize=9)
    ax.set_ylabel('販売冊数', fontsize=9)
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(actual) - 1)

    plt.tight_layout()

    stem = BOOK_FILENAMES[book_key]
    out_eps = os.path.join(OUTPUT_DIR, f'{stem}.eps')
    out_png = os.path.join(OUTPUT_DIR, f'{stem}.png')
    plt.savefig(out_eps, format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_eps}")
