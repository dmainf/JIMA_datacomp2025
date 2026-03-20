import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import numpy as np
sys.path.append('../lib')
from prepro import clean_df

matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

print("loading data...")
df = pd.read_csv('../data/data.txt', sep='\t')

df = clean_df(df)

# 本体価格ごとの出現回数を集計
price_counts = df.groupby('本体価格').size().reset_index(name='count')
price_counts = price_counts.sort_values('本体価格')

# 4000円以下と4000円超に分ける
below_4000 = price_counts[price_counts['本体価格'] <= 4000].copy()
above_4000 = price_counts[price_counts['本体価格'] > 4000].copy()

# 新しいx座標を作成
# 4000円以下はそのまま
below_4000['x'] = below_4000['本体価格']

# 4000円超は4000から5000の範囲に圧縮
if len(above_4000) > 0:
    min_above = above_4000['本体価格'].min()
    max_above = above_4000['本体価格'].max()
    above_4000['x'] = 4000 + 1000 * (above_4000['本体価格'] - min_above) / (max_above - min_above)

# 結合
all_data = pd.concat([below_4000, above_4000])

plt.figure(figsize=(20, 9))
plt.plot(all_data['x'].values, all_data['count'].values, marker='o', linestyle='-', markersize=3)

# 4000円の境界線を引く
plt.axvline(x=4000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='4,000円境界（以降圧縮）')

plt.xlabel('本体価格（円）※4,000円以降は圧縮スケール', fontsize=11)
plt.ylabel('出現回数', fontsize=11)
plt.title('本体価格の分布（4,000円以降圧縮）', fontsize=13)

# x軸のラベルを調整
xticks = list(range(0, 4001, 100))
xticklabels = [f'{x:,}' for x in xticks]

# 4000円以降のラベルを追加
if len(above_4000) > 0:
    above_ticks = [4000 + 250, 4000 + 500, 4000 + 750, 4000 + 1000]
    above_labels = ['10,000', '20,000', '40,000', '60,000']
    xticks.extend(above_ticks)
    xticklabels.extend(above_labels)

plt.xticks(xticks, xticklabels, rotation=45, ha='right')
plt.yticks(range(0, int(all_data['count'].max()) + 10000, 10000))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

print(f"\n価格範囲: {price_counts['本体価格'].min():.0f}円 ~ {price_counts['本体価格'].max():.0f}円")
print(f"\n上位10件の価格:")
top10 = df.groupby('本体価格').size().sort_values(ascending=False).head(10)
print(top10)

os.makedirs('figure', exist_ok=True)
plt.savefig('figure/本体価格_dist.png', dpi=150, bbox_inches='tight')
print("\nグラフを figure/本体価格_dist.png に保存しました")
plt.close()

# 生データのプロット（スケーリングなし）
plt.figure(figsize=(12, 9))
plt.plot(price_counts['本体価格'].values, price_counts['count'].values, marker='o', linestyle='-', markersize=3)

plt.xlabel('本体価格（円）', fontsize=11)
plt.ylabel('出現回数', fontsize=11)
plt.title('本体価格の分布（スケーリングなし）', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('figure/trush/本体価格_raw_dist.png', dpi=150, bbox_inches='tight')
print("グラフを figure/trush/本体価格_raw_dist.png に保存しました")
plt.close()
