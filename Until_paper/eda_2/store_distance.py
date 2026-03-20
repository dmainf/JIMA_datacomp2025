import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

EXCLUDE_CODES = [2, 24, 26, 27]  # 除外したい書店コード
DATA_DIR = Path("data")
STORE_INFO_FILE = "../data/Store.csv"
OUTPUT_DIR = Path("figure_2")
OUTPUT_DIR.mkdir(exist_ok=True)

store_info = pd.read_csv(STORE_INFO_FILE)
print("Loaded store info:", len(store_info), "rows")

sales_summary = []
for path in DATA_DIR.glob("df_*.parquet"):
    df = pd.read_parquet(path)
    store_code = int(path.stem.split("_")[1])

    df["売上"] = df["本体価格"] * df["POS販売冊数"]
    total_sales = df["売上"].sum()
    sales_summary.append({"書店コード": store_code, "売上合計": total_sales})

sales_df = pd.DataFrame(sales_summary)
print("Aggregated sales for", len(sales_df), "stores")

merged = pd.merge(store_info, sales_df, on="書店コード", how="inner")
print("Merged dataframe:", merged.shape)

corr_all = merged["売上合計"].corr(merged["駅距離"])
print(f"\n全店舗の売上と駅距離の相関係数: {corr_all:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(merged["駅距離"], merged["売上合計"],
            alpha=0.7, color="royalblue", edgecolor="k")
plt.xlabel("店舗～駅間距離 (m)")
plt.ylabel("累計売上 (億円)")
plt.title(f"全店舗：駅距離と売上の関係\n相関係数 = {corr_all:.3f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_vs_distance_all.png", dpi=300)
plt.close()
print(f" {OUTPUT_DIR / 'sales_vs_distance_all.png'}")

# === 特定店舗を除外した場合 ===
filtered = merged[~merged["書店コード"].isin(EXCLUDE_CODES)]
corr_filtered = filtered["売上合計"].corr(filtered["駅距離"])
print(f"除外後の相関係数: {corr_filtered:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(filtered["駅距離"], filtered["売上合計"],
            alpha=0.7, color="royalblue", label="対象店舗", edgecolor="k")
plt.scatter(merged[merged["書店コード"].isin(EXCLUDE_CODES)]["駅距離"],
            merged[merged["書店コード"].isin(EXCLUDE_CODES)]["売上合計"],
            alpha=0.7, color="red", label="除外店舗", edgecolor="k")
plt.xlabel("店舗～駅間距離(m)")
plt.ylabel("累計売上 (億円)")
plt.title(f"除外後：駅距離と売上の関係\n相関係数 = {corr_filtered:.3f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_vs_distance_filtered.png", dpi=300)
plt.close()
print(f" {OUTPUT_DIR / 'sales_vs_distance_filtered.png'}")

print("\nComplete!")
