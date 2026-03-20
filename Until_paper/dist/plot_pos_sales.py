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
df = clean_df(df)

# POS販売冊数ごとの出現回数を集計
pos_counts = df.groupby('POS販売冊数').size().reset_index(name='count')
pos_counts = pos_counts.sort_values('POS販売冊数')

# 20冊以下と20冊超に分ける
below_20 = pos_counts[pos_counts['POS販売冊数'] <= 20].copy()
above_20 = pos_counts[pos_counts['POS販売冊数'] > 20].copy()

# 新しいx座標を作成
below_20['x'] = below_20['POS販売冊数']

# 20冊超は20から30の範囲に圧縮
if len(above_20) > 0:
    min_above = above_20['POS販売冊数'].min()
    max_above = above_20['POS販売冊数'].max()
    above_20['x'] = 20 + 10 * (above_20['POS販売冊数'] - min_above) / (max_above - min_above)

# 結合
all_data = pd.concat([below_20, above_20])

plt.figure(figsize=(20, 9))
plt.plot(all_data['x'].values, all_data['count'].values, marker='o', linestyle='-', markersize=3)

# 20冊の境界線を引く
plt.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='20冊境界（以降圧縮）')

plt.xlabel('POS販売冊数 ※20冊以降は圧縮スケール', fontsize=11)
plt.ylabel('出現回数', fontsize=11)
plt.title('POS販売冊数の分布（20冊以降圧縮）', fontsize=13)

# x軸のラベルを調整
xticks = list(range(-20, 21, 2))
xticklabels = [f'{x}' for x in xticks]

# 20冊以降のラベルを追加
if len(above_20) > 0:
    above_ticks = [20 + 2.5, 20 + 5, 20 + 7.5, 20 + 10]
    above_labels = ['50', '100', '200', '309']
    xticks.extend(above_ticks)
    xticklabels.extend(above_labels)

plt.xticks(xticks, xticklabels, rotation=45, ha='right')
plt.yticks(range(0, int(all_data['count'].max()) + 500000, 500000))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

print(f"\nPOS販売冊数範囲: {pos_counts['POS販売冊数'].min():.0f}冊 ~ {pos_counts['POS販売冊数'].max():.0f}冊")
print(f"\n上位10件の冊数:")
top10 = df.groupby('POS販売冊数').size().sort_values(ascending=False).head(10)
print(top10)

os.makedirs('figure', exist_ok=True)
plt.savefig('figure/POS販売冊数_dist.png', dpi=150, bbox_inches='tight')
print("\nグラフを figure/POS販売冊数_dist.png に保存しました")
plt.close()
