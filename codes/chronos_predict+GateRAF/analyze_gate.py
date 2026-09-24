import numpy as np, pandas as pd, torch, faiss
from torch.utils.data import DataLoader
from func_gate_raf import (CONFIG, TimeSeriesRetriever, ChronosBoltFiDDataset,
                           extract_decile_books, load_model, fid_collate_fn)

K = 7
CONFIG["top_k"] = K
df = pd.read_parquet('data/df_for.parquet', columns=['書名','日付','POS販売冊数'])
ret = TimeSeriesRetriever(CONFIG["context_length"], CONFIG["retrieval_length"])
ret.build_index(df, step=CONFIG["index_step"])
ds = ChronosBoltFiDDataset(df=df, prediction_length=CONFIG["prediction_length"],
        mode="inference", retriever=ret, context_length=CONFIG["context_length"],
        top_k=K, decile_books=extract_decile_books(df), all_predict=True)
model = load_model("gate_raf_k7_checkpoints/final_adapter"); model.eval()
gm = [m for n,m in model.named_modules() if n.endswith("gate_mlp")][0]
buf=[]; gm.register_forward_hook(lambda m,i,o: buf.append(o.detach().float().cpu()))

dev = next(model.parameters()).device
allg=[]
with torch.no_grad():
    for i,b in enumerate(DataLoader(ds,batch_size=16,collate_fn=fid_collate_fn)):
        buf.clear(); model(**{k:v.to(dev) for k,v in b.items()})
        g=buf[0]; B=g.shape[0]; L=g.shape[1]//K
        allg.append(g.view(B,K,L).squeeze(-1))
        if i>=40: break
G=torch.cat(allg).numpy()                     # [N, K, L_patches]
print(f"サンプル数={G.shape[0]}  参照数K={G.shape[1]}  パッチ数={G.shape[2]}")

print("\n=== ゲート出力の全体分布 ===")
f=G.ravel()
print(f"  平均={f.mean():.4f} 中央値={np.median(f):.4f} 標準偏差={f.std():.4f}")
print(f"  最小={f.min():.4f} 5%={np.percentile(f,5):.4f} 95%={np.percentile(f,95):.4f} 最大={f.max():.4f}")

R = G.mean(axis=2)                            # [N, K] 参照ごとの平均ゲート
print("\n=== ばらつきの内訳（参照ごとの平均ゲート）===")
print(f"  同一サンプル内の順位間ばらつき（標準偏差の平均） = {R.std(axis=1).mean():.4f}")
print(f"  サンプル間のばらつき（標準偏差）                 = {R.mean(axis=1).std():.4f}")
print(f"  パッチ間のばらつき（標準偏差の平均）             = {G.std(axis=2).mean():.4f}")

print("\n=== 第1位 vs 第7位（同一サンプル内で対応づけ）===")
d = R[:,0]-R[:,6]
print(f"  差の平均={d.mean():+.4f}  第1位が大きいサンプルの割合={np.mean(d>0)*100:.1f}%")
from scipy import stats
print(f"  順位とゲートの相関（Spearman, サンプル平均）= "
      f"{np.mean([stats.spearmanr(np.arange(K),R[i]).statistic for i in range(len(R))]):+.3f}")
