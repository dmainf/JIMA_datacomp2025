"""Top-K の感度を集計し，LaTeX表を出力する．
run_topk_sweep.py が保存した K*/predictions_all.csv を読む．
査読者#A コメント2（Kの感度）への対応．
"""
import os
import glob
import numpy as np
import pandas as pd

QUANTILES = ['q0.1', 'q0.5', 'q0.9']
QUANTILE_LABEL = {'q0.1': r'$\tau=0.1$', 'q0.5': r'$\tau=0.5$', 'q0.9': r'$\tau=0.9$'}
BASE_K = 3          # 学習時に用いたK

dirs = sorted(glob.glob('K*'), key=lambda d: int(d[1:]))
rows = []
for d in dirs:
    df = pd.read_csv(f'{d}/predictions_all.csv')
    rec = {'K': int(d[1:]), 'n_books': df['書名'].nunique(), 'n_rows': len(df)}
    for q in QUANTILES:
        err = df['actual'] - df[q]
        rec[f'MAE_{q}'] = err.abs().mean()
        rec[f'RMSE_{q}'] = np.sqrt((err ** 2).mean())
    rows.append(rec)
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

base = t[t['K'] == BASE_K].iloc[0]
print(f'\nK={BASE_K} を基準とした変化率［%］')
for _, r in t.iterrows():
    cells = '  '.join(f'{c}={((r[c] / base[c]) - 1) * 100:+.1f}%'
                      for c in t.columns if c.startswith(('MAE', 'RMSE')))
    print(f'  K={int(r["K"]):<2} {cells}')
span = {c: (t[c].max() / t[c].min() - 1) * 100 for c in t.columns if c.startswith(('MAE', 'RMSE'))}
print('\nK=1〜8 での最大変動幅［%］: ' + '  '.join(f'{k}={v:.1f}%' for k, v in span.items()))

best = {c: t.loc[t[c].idxmin(), 'K'] for c in t.columns if c.startswith(('MAE', 'RMSE'))}

def fmt(r, c):
    s = f'{r[c]:.4f}'
    return rf'\textbf{{{s}}}' if best[c] == r['K'] else s

lines = [r'\begin{table}[t]', r'    \revAcolor',
         r'    \caption{参照件数$K$を変えた場合の予測精度（推論時のみ$K$を変更）}',
         r'    \label{tab:topk_sensitivity}', r'    \centering',
         r'    \setlength{\tabcolsep}{4pt}%',
         r'    \begin{tabular}{lrrrrrr}', r'        \hline',
         '        & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{QUANTILE_LABEL[q]}}}' for q in QUANTILES) + r' \\',
         r'        \cline{2-3} \cline{4-5} \cline{6-7}',
         r'        \multicolumn{1}{c}{$K$} & MAE & RMSE & MAE & RMSE & MAE & RMSE \\',
         r'        \hline\hline']
for _, r in t.iterrows():
    mark = r'$^{*}$' if r['K'] == BASE_K else ''
    cells = [fmt(r, f'{m}_{q}') for q in QUANTILES for m in ('MAE', 'RMSE')]
    lines.append(f'        {int(r["K"])}{mark} & ' + ' & '.join(cells) + r' \\')
lines += [r'        \hline', r'    \end{tabular}',
          r'    \\[2pt] \footnotesize $^{*}$学習時に用いた設定．',
          r'\end{table}']
os.makedirs('metric_comparison', exist_ok=True)
with open('metric_comparison/table_topk_sensitivity.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
t.to_csv('metric_comparison/topk_sensitivity.csv', index=False)
print('\nSaved: metric_comparison/table_topk_sensitivity.tex, topk_sensitivity.csv')
