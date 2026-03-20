import pandas as pd
from lib.prepro import *

print("loading data...")
df_raw = pd.read_parquet('data/store_detail.parquet')
print("complete!")

print("=== データの形状 ===")
print(df_raw.shape)
print("\n=== 欠損値の数 ===")
print(df_raw.isnull().sum())
print("\n=== 欠損値の割合 (%) ===")
print((df_raw.isnull().sum() / len(df_raw) * 100).round(2))
print("\n=== データ型 ===")
print(df_raw.dtypes)
print()

"""
print("###after cleaning###\n")
df = df_raw.copy()
df = clean_df(df)
print("=== データの形状 ===")
print(df.shape)
print("\n=== 欠損値の数 ===")
print(df.isnull().sum())
print("\n=== 欠損値の割合 (%) ===")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\n=== データ型 ===")
print(df.dtypes)

print()

print("saving dataflame to data/df.parquet...")
df.to_parquet('data/df.parquet')
print("complete!")
"""