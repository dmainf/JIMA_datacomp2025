import pandas as pd

df = pd.read_csv('../data/data.txt', sep='\t')

# パターン1: 著者名・出版社・書名・本体価格が全て欠損
pattern1 = df[df[['著者名', '出版社', '書名', '本体価格']].isna().all(axis=1)]

# パターン2: 著者名のみが欠損（他の3つのうち少なくとも1つは存在）
pattern2 = df[df['著者名'].isna() & ~df[['出版社', '書名', '本体価格']].isna().all(axis=1)]

print(f"パターン1（全て欠損）: {len(pattern1)}行")
print(f"パターン2（著者名のみ欠損）: {len(pattern2)}行")

print("\n=== パターン2のサンプル（最初の20件） ===")
print(pattern2[['書名', '出版社', '大分類', '中分類', '小分類']].head(20))

print("\n=== パターン2の大分類の分布 ===")
print(pattern2['大分類'].value_counts())

# パターン2をCSVに保存
pattern2.to_csv('pattern2_no_author.csv', index=False, encoding='utf-8-sig')
print(f"\nパターン2のデータを 'pattern2_no_author.csv' に保存しました")
