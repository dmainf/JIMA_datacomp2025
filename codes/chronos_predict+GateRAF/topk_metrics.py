import numpy as np
import pandas as pd

SRC = {1: "GateRAF-k1/predictions_all.csv",
       3: "../../make_figure/GateRAF.csv",
       5: "GateRAF-k5/predictions_all.csv",
       7: "GateRAF-k7/predictions_all.csv"}

rows = []
for k, path in sorted(SRC.items()):
    df = pd.read_csv(path)
    a = df["actual"].values
    rec = {"K": k, "n": len(df)}
    for q in ["q0.1", "q0.5", "q0.9"]:
        e = df[q].values - a
        rec[f"{q}_MAE"] = np.abs(e).mean()
        rec[f"{q}_RMSE"] = np.sqrt((e ** 2).mean())
    rows.append(rec)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out.to_csv("topk_metrics.csv", index=False)
print("\n→ topk_metrics.csv")
