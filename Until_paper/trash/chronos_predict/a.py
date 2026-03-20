import pandas as pd

# ファイルパスは状況に合わせて変更する（make_df.pyの出力先など）
file_path = 'data/df_for.parquet' 

try:
    df = pd.read_parquet(file_path)
except FileNotFoundError:
    # ファイルが見つからない場合はmake_df.pyの出力先を確認
    df = pd.read_parquet('df.parquet')

target_col = 'POS販売冊数'
threshold = 500

total_records = len(df)
over_threshold = df[df[target_col] > threshold]
count_over = len(over_threshold)
ratio = (count_over / total_records) * 100

print(f"=== Check for threshold: {threshold} ===")
print(f"Total records: {total_records:,}")
print(f"Records > {threshold}: {count_over:,}")
print(f"Ratio: {ratio:.5f}%")

print(f"\nTop 20 sales values:")
print(df[target_col].nlargest(20).values)

# 分布の確認（パーセンタイル）
print("\nPercentiles:")
print(df[target_col].quantile([0.9, 0.99, 0.999, 0.9999, 1.0]))