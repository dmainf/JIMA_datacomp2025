import pandas as pd
from lib.prepro import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("loading data...")
df_raw = pd.read_csv('data/data.txt', sep='\t')
print("complete!")

print("cleaning data...")
df = clean_df(df_raw)
print("cleaning complete!")

# 著者名だけが欠損していて、出版社・書名・本体価格は全て揃っているレコードを抽出
df_missing_author = df[
    df['著者名'].isnull() &
    df['出版社'].notnull() &
    df['書名'].notnull() &
    df['本体価格'].notnull()
].copy()

print("\n" + "="*50)
print("=== 著者名欠損データの基本情報 ===")
print("="*50)
print(f"欠損レコード数: {len(df_missing_author):,} / {len(df):,} ({len(df_missing_author)/len(df)*100:.2f}%)")
print(f"\nデータの形状: {df_missing_author.shape}")

print("\n" + "="*50)
print("=== 他の項目の欠損状況 ===")
print("="*50)
print("\n著者名欠損レコード内の他項目欠損数:")
print(df_missing_author.isnull().sum())

print("\n" + "="*50)
print("=== 出版社別の分析 ===")
print("="*50)
print("\n著者名欠損が多い出版社 TOP 30:")
publisher_counts = df_missing_author['出版社'].value_counts().head(30)
for i, (publisher, count) in enumerate(publisher_counts.items(), 1):
    total_by_publisher = df[df['出版社'] == publisher].shape[0]
    ratio = count / total_by_publisher * 100 if total_by_publisher > 0 else 0
    print(f"{i:2d}. {publisher}: {count:,}件 (この出版社の{ratio:.2f}%)")

print("\n" + "="*50)
print("=== 書名の分析 ===")
print("="*50)
print("\n著者名欠損が多い書名 TOP 30:")
title_counts = df_missing_author['書名'].value_counts().head(30)
for i, (title, count) in enumerate(title_counts.items(), 1):
    print(f"{i:2d}. {title}: {count:,}件")

print("\n" + "="*50)
print("=== 分類の分析 ===")
print("="*50)
print("\n大分類別の分布:")
large_cat_counts = df_missing_author['大分類'].value_counts()
for category, count in large_cat_counts.items():
    print(f"{category}: {count:,}件 ({count/len(df_missing_author)*100:.2f}%)")

print("\n中分類別の分布 TOP 20:")
mid_cat_counts = df_missing_author['中分類'].value_counts().head(20)
for i, (category, count) in enumerate(mid_cat_counts.items(), 1):
    print(f"{i:2d}. {category}: {count:,}件")

print("\n小分類別の分布 TOP 20:")
small_cat_counts = df_missing_author['小分類'].value_counts().head(20)
for i, (category, count) in enumerate(small_cat_counts.items(), 1):
    print(f"{i:2d}. {category}: {count:,}件")

print("\n" + "="*50)
print("=== 価格帯の分析 ===")
print("="*50)
print("\n価格の統計情報:")
print(df_missing_author['本体価格'].describe())

print("\n価格帯別の分布:")
bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, float('inf')]
labels = ['~500円', '501-1000円', '1001-1500円', '1501-2000円', '2001-3000円', '3001-5000円', '5001-10000円', '10000円~']
df_missing_author['価格帯'] = pd.cut(df_missing_author['本体価格'], bins=bins, labels=labels)
price_dist = df_missing_author['価格帯'].value_counts().sort_index()
for price_range, count in price_dist.items():
    print(f"{price_range}: {count:,}件 ({count/len(df_missing_author)*100:.2f}%)")

print("\n" + "="*50)
print("=== 販売数の分析 ===")
print("="*50)
print("\n販売数の統計情報:")
print(df_missing_author['POS販売冊数'].describe())
print(f"\n総販売冊数: {df_missing_author['POS販売冊数'].sum():,}冊")
print(f"平均販売冊数: {df_missing_author['POS販売冊数'].mean():.2f}冊")
print(f"中央値: {df_missing_author['POS販売冊数'].median():.0f}冊")

print("\n販売冊数の分布:")
print(df_missing_author['POS販売冊数'].value_counts().sort_index().head(20))

print("\n" + "="*50)
print("=== 時系列分析 ===")
print("="*50)
df_missing_author['年月'] = df_missing_author['年'].astype(str) + '-' + df_missing_author['月'].astype(str).str.zfill(2)
monthly_counts = df_missing_author['年月'].value_counts().sort_index()
print("\n月別の著者名欠損レコード数:")
for period, count in monthly_counts.items():
    print(f"{period}: {count:,}件")

print("\n年別の統計:")
yearly_counts = df_missing_author['年'].value_counts().sort_index()
for year, count in yearly_counts.items():
    print(f"{year}年: {count:,}件")

print("\n" + "="*50)
print("=== ISBNパターンの分析 ===")
print("="*50)
print("\nISBNプレフィックス (最初の13文字) TOP 30:")
df_missing_author['ISBN_prefix'] = df_missing_author['ISBN'].astype(str).str[:13]
isbn_prefix_counts = df_missing_author['ISBN_prefix'].value_counts().head(30)
for i, (prefix, count) in enumerate(isbn_prefix_counts.items(), 1):
    print(f"{i:2d}. {prefix}: {count:,}件")

print("\n出版社別のISBNプレフィックス分布:")
print("\nISBNプレフィックス (最初の10文字) TOP 20:")
df_missing_author['ISBN_short'] = df_missing_author['ISBN'].astype(str).str[:10]
isbn_short_counts = df_missing_author['ISBN_short'].value_counts().head(20)
for i, (prefix, count) in enumerate(isbn_short_counts.items(), 1):
    print(f"{i:2d}. {prefix}: {count:,}件")

print("\n" + "="*50)
print("=== 書店別の分析 ===")
print("="*50)
print("\n著者名欠損が多い書店 TOP 20:")
store_counts = df_missing_author['書店コード'].value_counts().head(20)
for i, (store, count) in enumerate(store_counts.items(), 1):
    print(f"{i:2d}. 書店コード {store}: {count:,}件")

print("\n" + "="*50)
print("=== サンプルデータ ===")
print("="*50)
print("\n著者名欠損データのサンプル (最初の20件):")
display_cols = ['年', '月', '日', '出版社', '書名', '著者名', '本体価格', 'POS販売冊数', '大分類', '中分類', '小分類']
print(df_missing_author[display_cols].head(20))

print("\n各大分類からのサンプル:")
for cat in df_missing_author['大分類'].dropna().unique()[:5]:
    print(f"\n--- {cat} ---")
    print(df_missing_author[df_missing_author['大分類'] == cat][display_cols].head(3))

# CSVに出力
output_file = 'data/missing_author_analysis.csv'
df_missing_author.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n著者名欠損データを {output_file} に出力しました。")
