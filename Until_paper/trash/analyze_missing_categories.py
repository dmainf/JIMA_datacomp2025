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

# 大分類・中分類・小分類が欠損しているレコードを抽出
df_missing_cat = df[
    df['大分類'].isnull() |
    df['中分類'].isnull() |
    df['小分類'].isnull()
].copy()

print("\n" + "="*50)
print("=== 分類欠損データの基本情報 ===")
print("="*50)
print(f"欠損レコード数: {len(df_missing_cat):,} / {len(df):,} ({len(df_missing_cat)/len(df)*100:.2f}%)")
print(f"\nデータの形状: {df_missing_cat.shape}")

print("\n" + "="*50)
print("=== 欠損パターンの分析 ===")
print("="*50)
print("\n各分類の欠損数:")
print(f"大分類のみ欠損: {df_missing_cat['大分類'].isnull().sum():,}")
print(f"中分類のみ欠損: {df_missing_cat['中分類'].isnull().sum():,}")
print(f"小分類のみ欠損: {df_missing_cat['小分類'].isnull().sum():,}")
print(f"全て欠損: {df_missing_cat[['大分類', '中分類', '小分類']].isnull().all(axis=1).sum():,}")

print("\n" + "="*50)
print("=== 出版社別の分析 ===")
print("="*50)
print("\n分類欠損が多い出版社 TOP 20:")
publisher_counts = df_missing_cat['出版社'].value_counts().head(20)
for i, (publisher, count) in enumerate(publisher_counts.items(), 1):
    print(f"{i:2d}. {publisher}: {count:,}件")

print("\n" + "="*50)
print("=== 書名の分析 ===")
print("="*50)
print("\n分類欠損が多い書名 TOP 20:")
title_counts = df_missing_cat['書名'].value_counts().head(20)
for i, (title, count) in enumerate(title_counts.items(), 1):
    print(f"{i:2d}. {title}: {count:,}件")

print("\n" + "="*50)
print("=== 著者名の分析 ===")
print("="*50)
print("\n分類欠損が多い著者 TOP 20:")
author_counts = df_missing_cat['著者名'].value_counts().head(20)
for i, (author, count) in enumerate(author_counts.items(), 1):
    print(f"{i:2d}. {author}: {count:,}件")

print("\n" + "="*50)
print("=== 価格帯の分析 ===")
print("="*50)
print("\n価格の統計情報:")
print(df_missing_cat['本体価格'].describe())

print("\n価格帯別の分布:")
bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, float('inf')]
labels = ['~500円', '501-1000円', '1001-1500円', '1501-2000円', '2001-3000円', '3001-5000円', '5001-10000円', '10000円~']
df_missing_cat['価格帯'] = pd.cut(df_missing_cat['本体価格'], bins=bins, labels=labels)
price_dist = df_missing_cat['価格帯'].value_counts().sort_index()
for price_range, count in price_dist.items():
    print(f"{price_range}: {count:,}件 ({count/len(df_missing_cat)*100:.2f}%)")

print("\n" + "="*50)
print("=== 販売数の分析 ===")
print("="*50)
print("\n販売数の統計情報:")
print(df_missing_cat['POS販売冊数'].describe())
print(f"\n総販売冊数: {df_missing_cat['POS販売冊数'].sum():,}冊")
print(f"平均販売冊数: {df_missing_cat['POS販売冊数'].mean():.2f}冊")
print(f"中央値: {df_missing_cat['POS販売冊数'].median():.0f}冊")

print("\n" + "="*50)
print("=== 時系列分析 ===")
print("="*50)
df_missing_cat['年月'] = df_missing_cat['年'].astype(str) + '-' + df_missing_cat['月'].astype(str).str.zfill(2)
monthly_counts = df_missing_cat['年月'].value_counts().sort_index()
print("\n月別の分類欠損レコード数:")
for period, count in monthly_counts.items():
    print(f"{period}: {count:,}件")

print("\n" + "="*50)
print("=== ISBNパターンの分析 ===")
print("="*50)
print("\nISBNプレフィックス (最初の13文字) TOP 20:")
df_missing_cat['ISBN_prefix'] = df_missing_cat['ISBN'].astype(str).str[:13]
isbn_prefix_counts = df_missing_cat['ISBN_prefix'].value_counts().head(20)
for i, (prefix, count) in enumerate(isbn_prefix_counts.items(), 1):
    print(f"{i:2d}. {prefix}: {count:,}件")

print("\n" + "="*50)
print("=== サンプルデータ ===")
print("="*50)
print("\n分類欠損データのサンプル (最初の10件):")
display_cols = ['年', '月', '日', '出版社', '書名', '著者名', '本体価格', 'POS販売冊数', '大分類', '中分類', '小分類']
print(df_missing_cat[display_cols].head(10))

# CSVに出力
output_file = 'data/missing_categories_analysis.csv'
df_missing_cat.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n分類欠損データを {output_file} に出力しました。")
