import pandas as pd
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from lib.prepro import *

print("loading data...")
df = pd.read_parquet('df.parquet')
print(f"complete!")

df = remove_volume_number(df)

unique_titles = df['書名'].nunique()
print(f"書名のユニーク数: {unique_titles:,}")

df['売上'] = df['POS販売冊数'] * df['本体価格']
sales_by_title = df.groupby('書名')['売上'].sum().reset_index()
sales_by_title = sales_by_title.sort_values('売上', ascending=False).reset_index(drop=True)

total_sales = sales_by_title['売上'].sum()
sales_by_title['累積売上'] = sales_by_title['売上'].cumsum()
sales_by_title['累積売上率'] = sales_by_title['累積売上'] / total_sales

threshold_10 = sales_by_title[sales_by_title['累積売上率'] >= 0.1].iloc[0]
threshold_30 = sales_by_title[sales_by_title['累積売上率'] >= 0.3].iloc[0]
threshold_80 = sales_by_title[sales_by_title['累積売上率'] >= 0.8].iloc[0]
threshold_index_10 = sales_by_title[sales_by_title['累積売上率'] >= 0.1].index[0]
threshold_index_30 = sales_by_title[sales_by_title['累積売上率'] >= 0.3].index[0]
threshold_index_80 = sales_by_title[sales_by_title['累積売上率'] >= 0.8].index[0]

print(f"\n全体売上: {total_sales:,.0f}円")
print(f"10%売上に到達: {threshold_index_10 + 1}番目の書名")
print(f"30%売上に到達: {threshold_index_30 + 1}番目の書名")
print(f"80%売上に到達: {threshold_index_80 + 1}番目の書名")
print(f"書名: {threshold_80['書名']}")
print(f"累積売上率: {threshold_80['累積売上率']:.2%}")
print(f"\nTop 10書名:")
print(sales_by_title.head(10)[['書名', '売上', '累積売上率']])

sales_by_title['順位'] = range(1, len(sales_by_title) + 1)
sales_by_title['累積売上率_pct'] = sales_by_title['累積売上率'] * 100

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.figure(figsize=(10, 6))
plt.plot(sales_by_title['順位'], sales_by_title['累積売上率_pct'])
plt.xlabel('書名の順位（売上降順）')
plt.ylabel('累積売上 (%)')
plt.title('累積売上と書名順位の関係')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('longtale_graph.png', dpi=300)
print(f"\nグラフを保存しました: longtale_graph.png")