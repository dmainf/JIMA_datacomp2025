import pandas as pd
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

BOOKS = [
    ('ぼっち・ざ・ろっく！_original',   'ぼっち・ざ・ろっく！',   'bocchi'),
    ('デラックスクロスワード_original', 'デラックスクロスワード', 'crossword'),
    ('母の待つ里_original',            '母の待つ里',            'haha'),
]

MODELS = [
    ('GateRAF',   'GateRAF'),
    ('Multi-RAF', 'Multi-RAF'),
    ('ProtoRAF',  'ProtoRAF'),
    ('noRAF',     'Baseline'),
]

ZBINS = [
    (-np.inf, 0,      '$z<0$'),
    (0,       1,      '$0\\leq z<1$'),
    (1,       2,      '$1\\leq z<2$'),
    (2,       np.inf, '$z\\geq2$'),
]

model_dfs = {}
for model_dir, _ in MODELS:
    df = pd.read_csv(os.path.join(BASE_PATH, model_dir, 'predictions_all.csv'))
    model_dfs[model_dir] = df

for book_key, book_display, stem in BOOKS:
    actual = model_dfs['GateRAF'][model_dfs['GateRAF']['書名'] == book_key]['actual'].values
    mean_a, std_a = actual.mean(), actual.std()
    z = (actual - mean_a) / (std_a + 1e-6)

    counts = []
    for lo, hi, _ in ZBINS:
        counts.append(int(((z > lo) & (z <= hi)).sum()))

    rows = {}
    for model_dir, label in MODELS:
        sub = model_dfs[model_dir][model_dfs[model_dir]['書名'] == book_key].reset_index(drop=True)
        q05 = sub['q0.5'].values
        maes = []
        for lo, hi, _ in ZBINS:
            mask = (z > lo) & (z <= hi)
            maes.append(np.abs(actual[mask] - q05[mask]).mean() if mask.sum() > 0 else float('nan'))
        rows[label] = maes

    print(f'\n% === {book_display} ===')
    print(r'\begin{table}[tb]')
    print(f'\\caption{{{book_display}における$z$スコアビン別MAE（$\\tau=0.5$）．括弧内はサンプル数．}}')
    print(f'\\label{{tab:book_{stem}_zscore}}')
    print(r'\centering')
    print(r'\begin{tabular}{lrrrr}')
    print(r'\hline')

    header_parts = ['モデル']
    for (_, _, lab), cnt in zip(ZBINS, counts):
        header_parts.append(f'{lab} (n={cnt})')
    print(' & '.join(header_parts) + r' \\')
    print(r'\hline\hline')

    for label, maes in rows.items():
        best = []
        for j in range(len(ZBINS)):
            col_vals = [rows[l][j] for l in [m[1] for m in MODELS] if not np.isnan(rows[l][j])]
            if col_vals:
                best.append(min(col_vals))
            else:
                best.append(None)

        cells = [f'\\textbf{{{label}}}']
        for j, v in enumerate(maes):
            if np.isnan(v):
                cells.append('---')
            elif best[j] is not None and abs(v - best[j]) < 1e-9:
                cells.append(f'\\textbf{{{v:.2f}}}')
            else:
                cells.append(f'{v:.2f}')
        print(' & '.join(cells) + r' \\')

    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')
