import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from lib.prepro import *
from lib.fueture_eng import *

print("loading data...")
df = pd.read_parquet('../data/df.parquet')
print("complete!")

# 著者名関連のカラムとテキストカラムを除いた数値カラムを取得
exclude_cols = ['月', '日', 'ISBN', '出版社', '書名', '著者名', '大分類', '中分類', '小分類']
# 著者名_pca列を除外
feature_cols = [col for col in df.columns
                if col not in exclude_cols
                and not col.startswith('著者名_pca_')
                and not col.startswith('著者名_enc')]

print(f"Found {len(feature_cols)} feature columns for clustering")
print(f"Feature columns: {feature_cols[:10]}... (showing first 10)")

# 全データでPCAを再実行して分散寄与率を確認（書名_pcaのみ）
print("\n=== Analyzing 書名 PCA components (all data) ===")
from sklearn.decomposition import PCA
pca_cols = [col for col in df.columns if col.startswith('書名_pca_')]
pca_analyzer = PCA(n_components=min(10, len(pca_cols)))
pca_analyzer.fit(df[pca_cols].dropna())
print("\nExplained variance ratio for top components:")
for i, var_ratio in enumerate(pca_analyzer.explained_variance_ratio_[:10]):
    print(f"  書名_pca_{i}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
print(f"\nCumulative variance (first 3 components): {pca_analyzer.explained_variance_ratio_[:3].sum():.4f} ({pca_analyzer.explained_variance_ratio_[:3].sum()*100:.2f}%)")

# 各月ごとにクラスタリングを実行
months = sorted(df['月'].unique())
n_clusters = 5

for month in months:
    print(f"\n{'='*60}")
    print(f"Processing month {month}...")
    print(f"{'='*60}")

    # 月ごとのデータを抽出
    month_df = df[df['月'] == month].copy()
    month_data = month_df[feature_cols].copy()

    if len(month_data) == 0:
        print(f"  No valid data for month {month}, skipping...")
        continue

    print(f"  Data shape: {month_data.shape}")

    # k=5でクラスタリング（全特徴量を使用）
    print(f"  Computing k-means with k={n_clusters} using {len(feature_cols)} features...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(month_data)

    # クラスタラベルを元のデータフレームに追加
    month_df['cluster'] = labels

    # 各クラスタの書名サンプルを表示
    print(f"\n  === Cluster samples for Month {month} ===")
    for i in range(n_clusters):
        cluster_books = month_df[month_df['cluster'] == i]['書名'].head(10).tolist()
        print(f"\n  Cluster {i} (n={len(month_df[month_df['cluster'] == i])}):")
        for j, book in enumerate(cluster_books, 1):
            print(f"    {j}. {book}")

    # 可視化用に書名_pca_0〜2を取得
    vis_data = month_df[['書名_pca_0', '書名_pca_1', '書名_pca_2']].copy()

    # 最初の3つのPCA成分を使って3次元プロット
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # クラスタごとに色を変えてプロット
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(vis_data.iloc[mask, 0],
                   vis_data.iloc[mask, 1],
                   vis_data.iloc[mask, 2],
                   c=[colors[i]],
                   label=f'Cluster {i}',
                   alpha=0.6,
                   s=20)

    # 軸ラベルに分散寄与率を追加
    var_ratios = pca_analyzer.explained_variance_ratio_
    ax.set_xlabel(f'syomei_pca_0\n(Variance: {var_ratios[0]*100:.2f}%)', fontsize=11)
    ax.set_ylabel(f'syomei_pca_1\n(Variance: {var_ratios[1]*100:.2f}%)', fontsize=11)
    ax.set_zlabel(f'syomei_pca_2\n(Variance: {var_ratios[2]*100:.2f}%)', fontsize=11)
    ax.set_title(f'K-means Clustering (k={n_clusters}) - Month {month}\nClustered with {len(feature_cols)} features, visualized with top 3 PCA components\nTotal Variance: {var_ratios[:3].sum()*100:.2f}%', fontsize=12)
    ax.legend(loc='upper right')

    # グラフを保存
    output_path = f'figure/cluster_3d_month_{month}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")

print(f"\nAll plots saved to figure/")
