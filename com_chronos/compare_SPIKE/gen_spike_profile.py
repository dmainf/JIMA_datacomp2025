"""系列単位のスパイク発生頻度で層別したMAE/RMSEを算出し，LaTeX表を出力する．
査読者#A コメント4（スパイクの頻度・規模ごとの予測性能）への対応．
評価期間中に一度もスパイク(z>2)が生じなかった系列は，層別の対象外として除外する．
最大規模（系列内zスコアの最大値）による層別は，時点単位の評価（本文の表2）と重複するため
表には載せず，参考値として標準出力にのみ表示する．
"""
import os
import numpy as np
import pandas as pd

SPIKE_Z = 2.0
QUANTILES = ['q0.1', 'q0.5', 'q0.9']
QUANTILE_LABEL = {'q0.1': r'$\tau=0.1$', 'q0.5': r'$\tau=0.5$', 'q0.9': r'$\tau=0.9$'}
MODELS = ['GateRAF', 'Multi-RAF', 'ProtoRAF', 'noRAF']
MODEL_LABEL = {'GateRAF': r'\textbf{GateRAF}', 'Multi-RAF': r'\textbf{Multi-RAF}',
               'ProtoRAF': r'\textbf{ProtoRAF}', 'noRAF': r'\textbf{Baseline}'}
FREQ_BINS = [0, 1, 2, 3, 4, 5, np.inf]
FREQ_LABELS = ['1', '2', '3', '4', '5', '6以上']
MAG_BINS = [SPIKE_Z, 3, 5, np.inf]
MAG_LABELS = ['2--3', '3--5', '5以上']


def load(model):
    df = pd.read_csv(f'{model}/predictions_all.csv').rename(columns={'書名': 'book'})
    stats = df.groupby('book')['actual'].agg(['mean', 'std'])
    df = df.join(stats, on='book')
    df['zscore'] = np.where(df['std'] > 0, (df['actual'] - df['mean']) / df['std'], 0.0)
    df['is_spike'] = df['zscore'] > SPIKE_Z
    return df


def metrics(df, books, q):
    sub = df[df['book'].isin(books)]
    err = sub['actual'] - sub[q]
    return err.abs().mean(), np.sqrt((err ** 2).mean()), len(sub)


dfs = {m: load(m) for m in MODELS}
per = dfs[MODELS[0]].groupby('book').agg(n_spike=('is_spike', 'sum'), max_z=('zscore', 'max'))
n_all = len(per)
per = per[per['n_spike'] >= 1]
print(f'全書名 {n_all} 中，スパイクを含む {len(per)} 書名を対象（除外 {n_all - len(per)} 書名）')

per['freq_bin'] = pd.cut(per['n_spike'], FREQ_BINS, labels=FREQ_LABELS)
per['mag_bin'] = pd.cut(per['max_z'], MAG_BINS, labels=MAG_LABELS)
groups = [(l, set(per.index[per['freq_bin'] == l])) for l in FREQ_LABELS]
mag_groups = [(l, set(per.index[per['mag_bin'] == l])) for l in MAG_LABELS]

res = {q: {m: [metrics(dfs[m], b, q) for _, b in groups] for m in MODELS} for q in QUANTILES}
n_books = [len(b) for _, b in groups]
n_points = [res['q0.5'][MODELS[0]][i][2] for i in range(len(groups))]

for q in QUANTILES:
    print(f"\n── {q} ──  {'区分':<6}{'書名':>6}{'時点':>8}  " + '  '.join(f'{m:>17}' for m in MODELS))
    for i, (lab, _) in enumerate(groups):
        cells = '  '.join(f'{res[q][m][i][0]:6.3f} /{res[q][m][i][1]:8.3f}' for m in MODELS)
        print(f'{"":<12}{lab:<6}{n_books[i]:>6}{n_points[i]:>8}  {cells}')
    print('              GateRAF改善率 vs ProtoRAF:',
          '  '.join(f'{groups[i][0]}:{(1-res[q]["GateRAF"][i][0]/res[q]["ProtoRAF"][i][0])*100:+.1f}%'
                    for i in range(len(groups))))

print('\n[参考] 最大規模別（表には載せない）')
for lab, books in mag_groups:
    vals = {m: metrics(dfs[m], books, 'q0.5') for m in MODELS}
    cells = '  '.join(f'{m}={vals[m][0]:.3f}' for m in MODELS)
    imp = (1 - vals['GateRAF'][0] / vals['ProtoRAF'][0]) * 100
    print(f'  最大z {lab:<5} 書名{len(books):>4}  {cells}  改善率{imp:+.1f}%'
          f"  最小={min(MODELS, key=lambda m: vals[m][0])}")

best = {(q, k, i): min(MODELS, key=lambda m: res[q][m][i][0 if k == 'mae' else 1])
        for q in QUANTILES for i in range(len(groups)) for k in ('mae', 'rmse')}

def fmt(q, m, i, kind):
    s = f'{res[q][m][i][0 if kind == "mae" else 1]:.2f}'
    return rf'\textbf{{{s}}}' if best[(q, kind, i)] == m else s

lines = [r'\begin{table*}[t]',
         r'    \revAcolor',
         r'    \caption{スパイクの発生頻度により層別したMAEおよびRMSE}',
         r'    \label{tab:spike_profile}', r'    \centering',
         r'    \setlength{\tabcolsep}{3pt}%',
         r'    \begin{tabular}{lrrrrrrrrrrrr}', r'        \hline',
         r'        & \multicolumn{12}{c}{発生頻度［回］} \\',
         r'        \cline{2-13}',
         '        ' + ' & '.join([r'\multicolumn{1}{c}{モデル}'] +
                                 [rf'\multicolumn{{2}}{{c}}{{{l}}}' for l in FREQ_LABELS]) + r' \\',
         r'        \cline{2-3} \cline{4-5} \cline{6-7} \cline{8-9} \cline{10-11} \cline{12-13}',
         '        ' + ' & '.join([''] + ['MAE', 'RMSE'] * 6) + r' \\',
         r'        \hline',
         '        ' + ' & '.join([r'\multicolumn{1}{c}{対象書名数}'] +
                                 [rf'\multicolumn{{2}}{{c}}{{{n}}}' for n in n_books]) + r' \\',
         r'        \hline\hline']
for q in QUANTILES:
    lines.append(f'        \\multicolumn{{{2 * len(groups) + 1}}}{{l}}{{{QUANTILE_LABEL[q]}}} \\\\')
    for m in MODELS:
        row = [r'\quad ' + MODEL_LABEL[m]]
        for i in range(len(groups)):
            row += [fmt(q, m, i, 'mae'), fmt(q, m, i, 'rmse')]
        lines.append('        ' + ' & '.join(row) + r' \\')
    lines.append(r'        \hline')
lines += [r'    \end{tabular}', r'\end{table*}']

os.makedirs('metric_comparison', exist_ok=True)
out = 'metric_comparison/table_spike_profile.tex'
with open(out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nLaTeX table saved: {out}')
