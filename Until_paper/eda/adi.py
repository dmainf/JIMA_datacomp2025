import pandas as pd
import numpy as np

print("loading data...")
df = pd.read_parquet('data/sales_df.parquet')
print("complete!")

total_records = len(df)
zero_demand_records = (df['POS販売冊数'] == 0).sum()
non_zero_records = (df['POS販売冊数'] > 0).sum()

zero_demand_rate = zero_demand_records / total_records
adi = total_records / non_zero_records if non_zero_records > 0 else np.nan

print("\n" + "="*80)
print("全書店・全書名を1つのプールとして扱った間欠需要指標")
print("="*80)
print(f"総レコード数: {total_records:,}")
print(f"需要あり（>0）: {non_zero_records:,} ({non_zero_records/total_records*100:.2f}%)")
print(f"需要なし（=0）: {zero_demand_records:,} ({zero_demand_rate*100:.2f}%)")
print()
print(f"ゼロ需要率: {zero_demand_rate:.4f} ({zero_demand_rate*100:.2f}%)")
print(f"ADI (平均需要間隔): {adi:.4f}")
print()
print("解釈:")
print(f"  - 全体の{zero_demand_rate*100:.1f}%の販売機会で需要がゼロ")
print(f"  - 平均して{adi:.2f}レコードに1回販売が発生")
print("="*80)

