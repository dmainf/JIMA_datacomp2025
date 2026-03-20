import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

EXCLUDE_CODES = [2, 24, 26, 27]  # 除外したい書店コードを指定
STORE_INFO_FILE = "../data/Store.csv"
DATA_DIR = Path("data")
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
merged = pd.merge(store_info, sales_df, on="書店コード", how="inner")
print("Merged dataframe:", merged.shape)

store_type_cols = ["駅構内", "複合施設", "独立店舗"]
for col in store_type_cols:
    if col not in merged.columns:
        raise ValueError(f"CSVに「{col}」列が見つかりません。")

def summarize_store_types(df):
    """店舗タイプ別の平均売上を集計"""
    summaries = []
    for col in store_type_cols:
        subset = df[df[col] == 1]
        summaries.append({
            "店舗型": col,
            "店舗数": len(subset),
            "平均売上": subset["売上合計"].mean() if len(subset) > 0 else 0
        })
    return pd.DataFrame(summaries)

summary_all = summarize_store_types(merged)
filtered = merged[~merged["書店コード"].isin(EXCLUDE_CODES)]
summary_filtered = summarize_store_types(filtered)

# === 5. 可視化関数 ===
def plot_bar(summary_df, title, filename, ylim=None):
    plt.figure(figsize=(7,5))
    plt.bar(summary_df["店舗型"], summary_df["平均売上"],
            color="cornflowerblue", edgecolor="k", alpha=0.8)
    plt.xlabel("店舗型")
    plt.ylabel("1店舗あたり年間平均売上（億円）")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
    print(f" {OUTPUT_DIR / filename}")

ymax = max(summary_all["平均売上"].max(), summary_filtered["平均売上"].max()) * 1.1
ymin = 0
ylim = (ymin, ymax)

plot_bar(summary_all, "全店舗：店舗型ごとの平均売上", "sales_by_storetype_all.png", ylim)
plot_bar(summary_filtered,
          f"除外後：店舗型ごとの平均売上",
          "sales_by_storetype_filtered.png",
          ylim)

# === 7. 結果を出力 ===
print("\n平均売上（全体）:")
print(summary_all)
print("\n平均売上（除外後）:")
print(summary_filtered)
