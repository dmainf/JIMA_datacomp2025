import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent
data_dir = base_dir / 'data'

print("loading data...")
df = pd.read_parquet(data_dir / 'data.parquet')
print("complete!")

df = df.dropna(subset=['出版社', '書名', '著者名', '本体価格'], how='all').copy()

print("\nextracting records with missing classifications...")
missing_all_classifications = df[
    df['大分類'].isna() &
    df['中分類'].isna() &
    df['小分類'].isna()
]

print(f"found {len(missing_all_classifications)} records with all classifications missing")
print(f"shape: {missing_all_classifications.shape}")

print("\nsaving to data/missing_classifications.parquet...")
missing_all_classifications.to_parquet(data_dir / 'missing_classifications.parquet')
print("complete!")
