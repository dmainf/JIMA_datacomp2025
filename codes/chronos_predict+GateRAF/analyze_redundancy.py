import numpy as np, pandas as pd, torch, faiss
from torch.utils.data import DataLoader
from func_gate_raf import (CONFIG, TimeSeriesRetriever, ChronosBoltFiDDataset,
                           extract_decile_books, load_model, fid_collate_fn)

df = pd.read_parquet('data/df_for.parquet', columns=['書名','日付','POS販売冊数'])
ret = TimeSeriesRetriever(CONFIG["context_length"], CONFIG["retrieval_length"])
ret.build_index(df, step=CONFIG["index_step"])

# ---- (a) 参照どうしの距離（冗長性）----
K = 7
CONFIG["top_k"] = K
ds7 = ChronosBoltFiDDataset(df=df, prediction_length=CONFIG["prediction_length"],
        mode="inference", retriever=ret, context_length=CONFIG["context_length"],
        top_k=K, decile_books=extract_decile_books(df), all_predict=True)

q2r, r2r = [], []
for s in ds7.samples:
    R = s["raf_context"].numpy()
    if not s["raf_mask"].numpy().any(axis=1).all():
        continue
    Z = (R - R.mean(axis=1, keepdims=True)) / (R.std(axis=1, keepdims=True) + 1e-5)
    c = s["context"].numpy()[-CONFIG["retrieval_length"]:]
    cz = (c - c.mean()) / (c.std() + 1e-5)
    q2r.append([((Z[j]-cz)**2).sum() for j in range(K)])
    r2r.append([((Z[j]-Z[0])**2).sum() for j in range(1, K)])
q2r = np.array(q2r); r2r = np.array(r2r)
print(f"\n=== 参照の冗長性（{len(q2r)}系列，標準化後の二乗ユークリッド）===")
print("  予測対象と各参照の距離（中央値）: " + "  ".join(f"{np.median(q2r[:,j]):.1f}" for j in range(K)))
print("  第1参照と第2〜7参照の距離（中央値）: " + "  ".join(f"{np.median(r2r[:,j]):.1f}" for j in range(K-1)))
print(f"  → 参照どうしの距離は予測対象との距離の {np.median(r2r)/np.median(q2r):.2f} 倍")

# ---- (b) K=1 と K=7 でゲート出力は変わるか ----
def gate_mean(ckpt, k):
    CONFIG["top_k"] = k
    d = ChronosBoltFiDDataset(df=df, prediction_length=CONFIG["prediction_length"],
            mode="inference", retriever=ret, context_length=CONFIG["context_length"],
            top_k=k, decile_books=extract_decile_books(df), all_predict=True)
    m = load_model(ckpt); m.eval()
    gm = [mm for n,mm in m.named_modules() if n.endswith("gate_mlp")][0]
    buf=[]; h=gm.register_forward_hook(lambda a,b,o: buf.append(o.detach().float().cpu()))
    dev = next(m.parameters()).device; acc=[]
    with torch.no_grad():
        for i,b in enumerate(DataLoader(d, batch_size=16, collate_fn=fid_collate_fn)):
            buf.clear(); m(**{kk:v.to(dev) for kk,v in b.items()})
            acc.append(buf[0].view(-1))
            if i>=25: break
    h.remove()
    g = torch.cat(acc).numpy()
    return g.mean(), np.median(g), len(g)

print("\n=== ゲート出力（参照1件あたり）===")
for ck,k in [("gate_raf_k1_checkpoints/final_adapter",1),
             ("gate_raf_checkpoints/final_adapter",3),
             ("gate_raf_k7_checkpoints/final_adapter",7)]:
    mu, med, n = gate_mean(ck, k)
    print(f"  K={k}  平均={mu:.4f}  中央値={med:.4f}   → 通過する総量の目安 K×平均 = {k*mu:.3f}")
