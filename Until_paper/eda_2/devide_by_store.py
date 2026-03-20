import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.prepro import *

print("loading data...")
df = pd.read_parquet('../data/df.parquet')
print(f"complete!")

output_dir = Path('data')
output_dir.mkdir(exist_ok=True)

store_code_column = '書店コード'
if store_code_column not in df.columns:
    print(f"Available columns: {df.columns.tolist()}")
    raise ValueError(f"Column '{store_code_column}' not found in dataframe")
unique_stores = df[store_code_column].unique()
print(f"Found {len(unique_stores)} unique stores")

for store_code in unique_stores:
    print(f"Processing store: {store_code}")
    df_store = df[df[store_code_column] == store_code]
    output_path = output_dir / f'df_{store_code}.parquet'
    df_store.to_parquet(output_path, index=False)
    print(f"  Saved {len(df_store)} rows to {output_path}")

print("All stores processed successfully!")