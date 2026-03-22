import pandas as pd
from lib import *

print("loading data...")
df_raw = pd.read_parquet('data.parquet')
normalized_title_list = pd.read_csv('normalized_title_list.csv')
print("Complete!")

print_df(df_raw)

drop_stores = [2, 24, 26, 27]
spacious_cols=['出版社', '大分類', '中分類', '小分類']

df = df_raw.copy()
df = drop_unsure(df)
df = drop_unstore(df, drop_stores)
df = drop_negative_sales(df)

df = filter_by_total_sales(df, min_sales=366)
df = normalize_titles(df, normalized_title_list)
df = remove_volume_number(df)
df = normalize_author(df)
df = fill_unknown_author(df)
df = delete_space(df, spacious_cols)

df = convert_to_datetime(df)
df = convert_to_category(df)

print_df(df)

print("saving dataflame to data/df.parquet...")
df.to_parquet('df.parquet')
print("Complete!")