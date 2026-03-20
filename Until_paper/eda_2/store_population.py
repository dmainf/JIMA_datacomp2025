import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

EXCLUDE_CODES = [2, 24, 26 , 27]  # 除外したい書店コード
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
print("Aggregated sales for", len(sales_df), "stores")

merged = pd.merge(store_info, sales_df, on="書店コード", how="inner")
print("Merged dataframe:", merged.shape)

for col in ["人口", "合算人口"]:
    if col not in merged.columns:
        raise ValueError(f"CSVに「{col}」列が見つかりません。列名を確認してください。")

def plot_and_save(x_col, y_col, df_all, exclude_codes, output_prefix, x_label):
    """全体と除外後の2種類を自動出力"""
    # --- 全体 ---
    corr_all = df_all[y_col].corr(df_all[x_col])
    plt.figure(figsize=(7, 5))
    plt.scatter(df_all[x_col], df_all[y_col],
                alpha=0.7, color="royalblue", edgecolor="k")
    plt.xlabel(x_label)
    plt.ylabel("累計売上（億円）")
    plt.title(f"全店舗：{x_label}と売上の関係\n相関係数 = {corr_all:.3f}")
    plt.grid(True)
    plt.tight_layout()
    path_all = OUTPUT_DIR / f"{output_prefix}_all.png"
    plt.savefig(path_all, dpi=300)
    plt.close()
    print(f" {path_all}")

    # --- 除外後 ---
    filtered = df_all[~df_all["書店コード"].isin(exclude_codes)]
    corr_filtered = filtered[y_col].corr(filtered[x_col])
    plt.figure(figsize=(7, 5))
    plt.scatter(filtered[x_col], filtered[y_col],
                alpha=0.7, color="royalblue", edgecolor="k", label="対象店舗")
    plt.scatter(df_all[df_all["書店コード"].isin(exclude_codes)][x_col],
                df_all[df_all["書店コード"].isin(exclude_codes)][y_col],
                alpha=0.7, color="red", edgecolor="k", label="除外店舗")
    plt.xlabel(x_label)
    plt.ylabel("累計売上（億円）")
    plt.title(f"除外後：{x_label}と売上の関係\n相関係数 = {corr_filtered:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path_filtered = OUTPUT_DIR / f"{output_prefix}_filtered.png"
    plt.savefig(path_filtered, dpi=300)
    plt.close()
    print(f" {path_filtered}")

    # --- 結果を返す ---
    return corr_all, corr_filtered


# === 6. 人口 × 売上 ===
corr_pop_all, corr_pop_filtered = plot_and_save(
    "人口", "売上合計", merged, EXCLUDE_CODES, "sales_vs_population", "人口（店舗メッシュ内）"
)

# === 7. 合算人口 × 売上 ===
corr_total_all, corr_total_filtered = plot_and_save(
    "合算人口", "売上合計", merged, EXCLUDE_CODES, "sales_vs_total_population", "合算人口（中心＋周囲）"
)

# === 8. 結果表示 ===
print("\n相関結果")
print(f"・人口 × 売上（全体）      : {corr_pop_all:.3f}")
print(f"・人口 × 売上（除外後）    : {corr_pop_filtered:.3f}")
print(f"・合算人口 × 売上（全体）  : {corr_total_all:.3f}")
print(f"・合算人口 × 売上（除外後）: {corr_total_filtered:.3f}")

print("\nComplete!")
