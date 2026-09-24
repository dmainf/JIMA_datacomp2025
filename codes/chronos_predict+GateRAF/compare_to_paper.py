import sys
import numpy as np
import pandas as pd

NEW = sys.argv[1] if len(sys.argv) > 1 else "GateRAF-k3verify/predictions_all.csv"
REF = "../../make_figure/GateRAF.csv"

new = pd.read_csv(NEW).sort_values(["書名", "day"]).reset_index(drop=True)
ref = pd.read_csv(REF).sort_values(["書名", "day"]).reset_index(drop=True)

print(f"new: {NEW}  {new.shape}  books={new['書名'].nunique()}")
print(f"ref: {REF}  {ref.shape}  books={ref['書名'].nunique()}")

if new.shape != ref.shape or not new["書名"].equals(ref["書名"]):
    print("!! 行数または書名の並びが一致しません")
    sys.exit(1)

for c in ["actual", "q0.1", "q0.5", "q0.9"]:
    a, b = new[c].values, ref[c].values
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-9)
    print(f"{c:6s} max|Δ|={diff.max():.6g}  mean|Δ|={diff.mean():.6g}  "
          f"max rel={np.max(diff/denom):.6g}  完全一致={np.array_equal(a, b)}")

mae_new = np.abs(new["q0.5"] - new["actual"]).mean()
mae_ref = np.abs(ref["q0.5"] - ref["actual"]).mean()
print(f"\nMAE(q0.5)  new={mae_new:.6f}  ref={mae_ref:.6f}  Δ={mae_new-mae_ref:+.6f}")
