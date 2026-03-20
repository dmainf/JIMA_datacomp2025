import pandas as pd
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
import time
from collections import Counter
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[2]))
from lib.prepro import *

def fast_mode(x):
    if len(x) <= 1:
        return x.iloc[0] if len(x) == 1 else None
    x_clean = x.dropna()
    if len(x_clean) == 0:
        return x.iloc[0]
    if len(x_clean) == 1:
        return x_clean.iloc[0]
    return Counter(x_clean).most_common(1)[0][0]

def process_store(args):
    store_code, parquet_path, output_dir = args
    try:
        df_store = pd.read_parquet(parquet_path, filters=[('書店コード', '==', store_code)], engine='pyarrow')
        if df_store.empty:
            return store_code, 0, 0, False
        df_store = remove_volume(df_store)
        df_store = df_store.drop(columns=['月', '日', '累積日数'], errors='ignore')
        df_store['売上'] = df_store['POS販売冊数'] * df_store['本体価格']
        other_cols = [c for c in df_store.columns if c not in ['書店コード', '日付', '書名', 'POS販売冊数', '本体価格', '売上']]
        dup_mask = df_store.duplicated(subset=['日付', '書名'], keep=False)
        has_duplicates = dup_mask.any()
        if not has_duplicates:
            df_result = df_store.copy()
        else:
            agg_dict = {'POS販売冊数': 'sum', '本体価格': 'mean', '売上': 'sum'}
            for col in other_cols:
                agg_dict[col] = fast_mode
            df_no_dup = df_store[~dup_mask]
            df_dup = df_store[dup_mask]
            df_agg = df_dup.groupby(['日付', '書名'], as_index=False, sort=False).agg(agg_dict)
            df_result = pd.concat([df_no_dup, df_agg], ignore_index=True)
        df_result['書店コード'] = store_code
        cols = ['書店コード', '日付', '書名', 'POS販売冊数', '本体価格', '売上'] + other_cols
        df_result = df_result[[c for c in cols if c in df_result.columns]]
        output_path = output_dir / f'df_{store_code}.parquet'
        df_result.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        return store_code, len(df_result), len(df_store), has_duplicates
    except Exception as e:
        print(f"Error processing store {store_code}: {e}")
        return store_code, 0, 0, False

if __name__ == '__main__':
    start_time = time.time()
    parquet_path = Path('../../data/df.parquet')
    output_dir = Path('./by_store')
    output_dir.mkdir(parents=True, exist_ok=True)

    store_codes = pq.read_table(parquet_path, columns=['書店コード']).to_pandas()['書店コード'].unique()
    args_list = [(code, parquet_path, output_dir) for code in store_codes]
    n_workers = max(1, cpu_count() - 1)
    print(f"Processing {len(args_list)} stores with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(process_store, args_list)

    results = [r for r in results if r is not None]
    total_grouped = sum(r[1] for r in results)
    total_original = sum(r[2] for r in results)
    stores_with_dups = sum(1 for r in results if r[3])

    print("\n=== Results ===")
    for store_code, grouped_rows, original_rows, has_dup in sorted(results):
        dup_str = " (had duplicates)" if has_dup else ""
        print(f"Store {store_code}: {grouped_rows:,} rows (from {original_rows:,}){dup_str}")
    print(f"\nTotal: {total_grouped:,} rows (from {total_original:,} rows)")
    print(f"Stores with duplicates: {stores_with_dups}/{len(results)}")

    print(f"Total time: {time.time() - start_time:.2f}s")

