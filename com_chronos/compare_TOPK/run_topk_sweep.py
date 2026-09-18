"""学習済みGateRAF（K=3で学習）に対し，推論時のTop-K件数を変えて予測を行い，
他のcompare_*と同じ形式（K別ディレクトリ/predictions_all.csv）で保存する．
査読者#A コメント2（Kの感度）への対応．再学習は行わない．
"""
import os
import sys
import numpy as np
import pandas as pd

CODE_DIR = '/Users/dmainf/lab/JIMA_datacomp2025/codes/chronos_predict+GateRAF'
sys.path.insert(0, CODE_DIR)
from func_gate_raf import (CONFIG, TimeSeriesRetriever, ChronosBoltFiDDataset,
                           load_model, run_inference)

KS = [1, 2, 3, 5, 8]
Q_INDICES = [0, 4, 8]          # chronos-bolt の分位点 [0.1,...,0.9] における 0.1 / 0.5 / 0.9
ADAPTER = os.path.join(CODE_DIR, 'gate_raf_checkpoints/final_adapter')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_parquet(os.path.join(CODE_DIR, 'data/df_for.parquet'),
                     columns=['書名', '日付', 'POS販売冊数'])
retriever = TimeSeriesRetriever(CONFIG["context_length"], CONFIG["retrieval_length"])
retriever.build_index(df, step=CONFIG["index_step"])
model = load_model(ADAPTER)

for k in KS:
    print(f'\n===== Top-K = {k} =====', flush=True)
    ds = ChronosBoltFiDDataset(df=df, prediction_length=CONFIG["prediction_length"],
                               mode="inference", retriever=retriever,
                               context_length=CONFIG["context_length"],
                               top_k=k, decile_books=None, all_predict=True)
    forecasts = run_inference(model, ds)
    rows = []
    for meta, fc in zip(ds.metadata, forecasts):
        pred = fc.numpy()
        for day, y in enumerate(meta['target']):
            rows.append({'書名': meta['id'], 'day': day, 'actual': y,
                         'q0.1': pred[Q_INDICES[0], day],
                         'q0.5': pred[Q_INDICES[1], day],
                         'q0.9': pred[Q_INDICES[2], day]})
    out = os.path.join(OUT_DIR, f'K{k}')
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'predictions_all.csv'), index=False)
    print(f'Saved → {out}/predictions_all.csv  ({len(rows)} rows)', flush=True)
