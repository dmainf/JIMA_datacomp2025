import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import sys
sys.path.append('../lib')
from prepro import delete_space, normalize_author, normalize_title

print("loading data...")
df_raw = pd.read_csv('../data/data.txt', sep='\t')

# 前処理
space_columns = ['書名', '著者名', '出版社']
df = df_raw.copy()
for column in space_columns:
    df = delete_space(df, column)

df = normalize_author(df, '著者名')
df = normalize_title(df, '書名')

# 著者名の出現回数ごとの頻度を計算
author_value_counts = df['著者名'].value_counts()
author_counts_of_counts = author_value_counts.value_counts().sort_index()
author_null_count = df['著者名'].isnull().sum()

# 書名の出現回数ごとの頻度を計算
title_value_counts = df['書名'].value_counts()
title_counts_of_counts = title_value_counts.value_counts().sort_index()
title_null_count = df['書名'].isnull().sum()

# 総データ数
total_count = len(df)

# グラフ作成
plt.figure(figsize=(20, 9))

# 著者名のプロット
plt.bar(author_counts_of_counts.index, author_counts_of_counts.values,
        alpha=0.6, label='著者名', color='blue', width=0.8)

# 書名のプロット
plt.bar(title_counts_of_counts.index, title_counts_of_counts.values,
        alpha=0.6, label='書名', color='red', width=0.8)

plt.xlabel('出現回数（対数スケール）', fontsize=11)
plt.ylabel('該当する種類数（対数スケール）', fontsize=11)
plt.title('著者名と書名の出現回数ごとの頻度分布の比較', fontsize=13)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=12)

# 欠損値情報をグラフに表示
author_null_percentage = (author_null_count / total_count * 100).round(2)
title_null_percentage = (title_null_count / total_count * 100).round(2)

info_text = f'著者名 欠損値: {author_null_count:,}件 ({author_null_percentage}%)\n書名 欠損値: {title_null_count:,}件 ({title_null_percentage}%)'
plt.text(0.98, 0.98, info_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

import os
os.makedirs('figure', exist_ok=True)
output_file = 'figure/著者名_書名_counts_comparison_loglog.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nグラフを {output_file} に保存しました")
plt.show()

print("\n=== 統計情報 ===")
print(f"著者名:")
print(f"  ユニーク数: {df['著者名'].nunique():,}")
print(f"  欠損値: {author_null_count:,}件 ({author_null_percentage}%)")
print(f"  最大出現回数: {author_value_counts.max():,}")
print(f"\n書名:")
print(f"  ユニーク数: {df['書名'].nunique():,}")
print(f"  欠損値: {title_null_count:,}件 ({title_null_percentage}%)")
print(f"  最大出現回数: {title_value_counts.max():,}")
