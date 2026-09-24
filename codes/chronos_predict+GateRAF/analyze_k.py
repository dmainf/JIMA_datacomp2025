import numpy as np
import pandas as pd
import torch
import faiss
from func_gate_raf import (CONFIG, TimeSeriesRetriever, ChronosBoltFiDDataset,
                           extract_decile_books, load_model, fid_collate_fn)

K = 7
CONFIG["top_k"] = K
df = pd.read_parquet('data/df_for.parquet', columns=['書名', '日付', 'POS販売冊数'])

ret = TimeSeriesRetriever(CONFIG["context_length"], CONFIG["retrieval_length"])
ret.build_index(df, step=CONFIG["index_step"])

# ---- 1) 検索順位ごとの距離 ----
queries, qdates = [], []
for book, g in df.groupby('書名', observed=True):
    s = g.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
    if len(s) <= CONFIG["prediction_length"] + CONFIG["context_length"]:
        continue
    q = s[:-CONFIG["prediction_length"]]
    use = min(len(q), CONFIG["retrieval_length"])
    sl = q[-use:]
    if use < CONFIG["retrieval_length"]:
        sl = np.pad(sl, (CONFIG["retrieval_length"] - use, 0), 'constant')
    queries.append(sl)
    qdates.append(g.sort_values('日付')['日付'].values[len(q) - 1])

Q = np.stack(queries)
mu = Q.mean(axis=1, keepdims=True); sd = Q.std(axis=1, keepdims=True) + 1e-5
Qn = np.ascontiguousarray(((Q - mu) / sd).astype('float32'))
faiss.omp_set_num_threads(1)
D, I = ret.index.search(Qn, K * 5 + 10)
fd = ret.timestamps[I]; qd = pd.to_datetime(np.array(qdates)).values.reshape(-1, 1)
ok = fd < qd

dist = np.full((len(Q), K), np.nan)
for i in range(len(Q)):
    d = D[i][ok[i]][:K]
    dist[i, :len(d)] = d

print("\n=== 検索順位ごとの距離（二乗ユークリッド，標準化後）===")
for j in range(K):
    v = dist[:, j]; v = v[~np.isnan(v)]
    print(f"  第{j+1}位  中央値={np.median(v):8.2f}  平均={v.mean():8.2f}  1位比={np.median(v)/np.median(dist[:,0][~np.isnan(dist[:,0])]):.2f}")

# ---- 2) 順位ごとのゲート出力 ----
decile = extract_decile_books(df)
ds = ChronosBoltFiDDataset(df=df, prediction_length=CONFIG["prediction_length"],
                           mode="inference", retriever=ret,
                           context_length=CONFIG["context_length"],
                           top_k=K, decile_books=decile, all_predict=True)
model = load_model("gate_raf_k7_checkpoints/final_adapter")
model.eval()

gm = [m for n, m in model.named_modules() if n.endswith("gate_mlp")][0]
buf = []
gm.register_forward_hook(lambda mod, inp, out: buf.append(out.detach().float().cpu()))

from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=16, collate_fn=fid_collate_fn)
dev = next(model.parameters()).device
scores = []
with torch.no_grad():
    for i, b in enumerate(dl):
        buf.clear()
        model(**{k: v.to(dev) for k, v in b.items()})
        g = buf[0]                                   # [B, K*L_patches, 1]
        B = g.shape[0]; L = g.shape[1] // K
        scores.append(g.view(B, K, L).mean(dim=2).squeeze(-1))
        if i >= 20:
            break
S = torch.cat(scores).numpy()
print(f"\n=== 順位ごとのゲート出力（{S.shape[0]}サンプル，1に近いほど参照を通す）===")
for j in range(K):
    print(f"  第{j+1}位  平均={S[:,j].mean():.4f}  中央値={np.median(S[:,j]):.4f}")
print(f"\n  第1位に対する第7位の比 = {S[:,6].mean()/S[:,0].mean():.3f}")
