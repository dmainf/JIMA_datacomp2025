import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
sys.path.append('../lib')
from prepro import clean_df

matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

print("loading data...")
df = pd.read_csv('../data/data.txt', sep='\t')

print(f"総行数: {len(df)}")
print(f"ユニークな日付数: {df['日付'].nunique()}")
print(f"欠損値の数: {df['日付'].isnull().sum()}")
print(f"欠損値の割合: {(df['日付'].isnull().sum()/len(df['日付'])*100).round(2)}")

df = clean_df(df)

date_counts = df.groupby(['年', '月', '日']).size().reset_index(name='count')
date_counts = date_counts.sort_values(['年', '月', '日'])

date_counts['日付ラベル'] = date_counts['年'].astype(str) + '-' + date_counts['月'].astype(str).str.zfill(2) + '-' + date_counts['日'].astype(str).str.zfill(2)

plt.figure(figsize=(14, 9))
plt.plot(range(len(date_counts)), date_counts['count'].values, marker='o', linestyle='-', markersize=3)

for i in range(0, len(date_counts), 7):
    plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xlabel('日付', fontsize=11)
plt.ylabel('出現回数', fontsize=11)
plt.title('日付の時系列分布', fontsize=13)

step = max(1, len(date_counts) // 20)
plt.xticks(range(0, len(date_counts), step), date_counts['日付ラベル'].iloc[::step], rotation=45, ha='right')

plt.tight_layout()
print(f"\n日付範囲: {date_counts['日付ラベル'].iloc[0]} ~ {date_counts['日付ラベル'].iloc[-1]}")
os.makedirs('figure', exist_ok=True)
plt.savefig('figure/日付_distribution.png', dpi=150, bbox_inches='tight')
print("\nグラフを figure/日付_distribution.png に保存しました")
