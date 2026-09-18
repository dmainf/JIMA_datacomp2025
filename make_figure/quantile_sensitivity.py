import os
import pandas as pd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PRICE_PARQUET = os.path.join(BASE_PATH, '..', 'codes', 'chronos_predict+GateRAF', 'data', 'df.parquet')

MODELS = [
    ('GateRAF', 'GateRAF'),
    ('baseline', 'Baseline'),
    ('SARIMA', 'SARIMA'),
    ('Prophet', 'Prophet'),
]
QUANTILES = ['q0.1', 'q0.5', 'q0.9']

price = (pd.read_parquet(PRICE_PARQUET, columns=['書名', '本体価格'])
           .groupby('書名', observed=True)['本体価格']
           .agg(lambda x: x.mode().iloc[0]))

rows = []
for fname, label in MODELS:
    df = pd.read_csv(os.path.join(BASE_PATH, f'{fname}.csv'))
    p = df['書名'].map(price)
    sales = df['actual'].sum()
    for q in QUANTILES:
        err = df[q] - df['actual']
        over = err.clip(lower=0)          # 過剰入荷（返本）
        short = (-err).clip(lower=0)      # 機会損失
        rows.append({
            'model': label,
            'tau': float(q[1:]),
            'over': over.sum(),
            'return_rate': over.sum() / (sales + over.sum()) * 100,
            'short': short.sum(),
            'over_yen': (over * p).sum() / 1e4,
            'short_yen': (short * p).sum() / 1e4,
        })

out = pd.DataFrame(rows)
out['total_yen'] = out['over_yen'] + out['short_yen']
out.to_csv(os.path.join(BASE_PATH, 'quantile_sensitivity.csv'), index=False)

pd.set_option('display.width', 200)
print(out.round(2).to_string(index=False))
