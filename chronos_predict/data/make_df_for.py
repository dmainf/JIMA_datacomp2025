import pandas as pd
import gc
from lib import *

print("loading data...")
df = pd.read_parquet('df.parquet')
print("Complete!")

keys = ['日付', '書名']
attrs = ['出版社','著者名','大分類','中分類','小分類','本体価格']

book_attrs = calc_book_attrs(df, attrs)
sales = calc_sales(df, keys)
del df
gc.collect()

full_index = make_full_index(book_attrs)
df = add_sales(full_index, sales, keys)
del full_index
gc.collect()

df['POS販売冊数'] = df['POS販売冊数'].clip(upper=500)

df = add_book_attrs(df, book_attrs)
df = convert_types(df)
df = convert_to_category(df)
df = clean_categories(df)

print_df(df)

df.to_parquet('df_for.parquet', index=False)
print("Saved to 'df_for.parquet'")
