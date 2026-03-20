import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder, StandardScaler
from functools import partial
import pickle

STATIC_CAT_COLS = [
    '書店コード',
    '書名',
    '出版社',
    '著者名',
    '大分類',
    '中分類',
    '小分類',
    '書店名',
    '住所',
    '昼間フラグ',
    'メッシュ5'
]

STATIC_REAL_COLS = [
    '本体価格',
    '駅距離',
    '営業時間(平)',
    '営業時間(特)',
    '駅構内',
    '複合施設',
    '独立店舗',
    '市区別_夜間人口',
    '市区別_昼間人口',
    '周辺合計人口',
    '合計人口',
    '昼間/夜間割合',
    'mesh_pop_center',
    'mesh_pop_neighbors_avg',
    'mesh_pop_neighbors_sum',
    'mesh_pop_gradient_NS',
    'mesh_pop_gradient_EW',
    'mesh_pop_center_vs_neighbors'
]

TEMPORAL_COLS = [
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_month_sin', 'day_of_month_cos',
    'week_of_year_sin', 'week_of_year_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'quarter',
    'days_since_release',
    'is_new_release_7d',
    'is_new_release_30d'
]

TIME_COLS = ['開店時間(平)', '閉店時間(平)', '開店時間(特)', '閉店時間(特)']

def prepare_dataset(df, context_length, prediction_length, max_window):
    dataset = df.copy()
    print("Loading and processing release dates...")
    release_date_path = Path() / 'data' / 'release_dates.parquet'
    if release_date_path.exists():
        release_df = pd.read_parquet(release_date_path)
        release_map_df = release_df.explode(['巻数', '発売日'])
        release_map_df['発売日'] = pd.to_datetime(release_map_df['発売日'])
        release_map_df['書名'] = release_map_df['作品_シリーズ'] + '_' + release_map_df['巻数'].astype(str)
        release_map_df = release_map_df[['書名', '発売日']].drop_duplicates()
        dataset = pd.merge(dataset, release_map_df, on='書名', how='left')
    else:
        print("Warning: release_dates.parquet not found. Skipping release date features.")
        dataset['発売日'] = pd.NaT
    print("Calculating 'first_sale_date' for books without release date...")
    dataset['first_sale_date'] = dataset.groupby(['書店コード', '書名'])['日付'].transform('min')
    dataset['発売日'] = dataset['発売日'].fillna(dataset['first_sale_date'])
    print("Generating future temporal features...")
    dataset['days_since_release'] = (dataset['日付'] - dataset['発売日']).dt.days.astype(np.int16)
    dataset['is_new_release_7d'] = ((dataset['days_since_release'] >= 0) & (dataset['days_since_release'] <= 7)).astype(np.int8)
    dataset['is_new_release_30d'] = ((dataset['days_since_release'] >= 0) & (dataset['days_since_release'] <= 30)).astype(np.int8)
    dataset = dataset.drop(columns=['発売日', 'first_sale_date'])
    dataset['day_of_week_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.dayofweek / 7).astype(np.float32)
    dataset['day_of_week_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.dayofweek / 7).astype(np.float32)
    dataset['day_of_month_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.day / 30).astype(np.float32)
    dataset['day_of_month_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.day / 30).astype(np.float32)
    week_num = dataset['日付'].dt.isocalendar().week.astype(float)
    dataset['week_of_year_sin'] = np.sin(2 * np.pi * week_num / 52).astype(np.float32)
    dataset['week_of_year_cos'] = np.cos(2 * np.pi * week_num / 52).astype(np.float32)
    dataset['month_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.month / 12).astype(np.float32)
    dataset['month_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.month / 12).astype(np.float32)
    dataset['is_weekend'] = (dataset['日付'].dt.dayofweek >= 5).astype(np.int8)
    dataset['quarter'] = dataset['日付'].dt.quarter.astype(np.int8)
    print("Applying log1p transform to '売上' target...")
    dataset['売上'] = np.log1p(dataset['売上'].values.astype(np.float32))
    min_length = context_length + prediction_length
    dataset = dataset.groupby(['書店コード', '書名']).filter(lambda g: g['日付'].nunique() >= min_length)
    dataset['n_slice'] = (dataset.groupby(['書店コード', '書名'])['日付'].transform(lambda x: x.nunique() // min_length))
    print("Engineering spatial mesh features...")
    pop_cols = [f'メッシュ{i}_人口' for i in range(1, 10)]
    for col in pop_cols:
        if col not in dataset.columns:
            dataset[col] = 0
    mesh_pops = dataset[pop_cols].values.astype(np.float32)
    dataset['mesh_pop_center'] = mesh_pops[:, 4]
    neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    neighbor_pops = mesh_pops[:, neighbor_indices]
    dataset['mesh_pop_neighbors_sum'] = np.sum(neighbor_pops, axis=1)
    dataset['mesh_pop_neighbors_avg'] = np.mean(neighbor_pops, axis=1)
    dataset['mesh_pop_center_vs_neighbors'] = dataset['mesh_pop_center'] - dataset['mesh_pop_neighbors_avg']
    pop_north_avg = np.mean(mesh_pops[:, [0, 1, 2]], axis=1)
    pop_south_avg = np.mean(mesh_pops[:, [6, 7, 8]], axis=1)
    dataset['mesh_pop_gradient_NS'] = pop_north_avg - pop_south_avg
    pop_east_avg = np.mean(mesh_pops[:, [2, 5, 8]], axis=1)
    pop_west_avg = np.mean(mesh_pops[:, [0, 3, 6]], axis=1)
    dataset['mesh_pop_gradient_EW'] = pop_east_avg - pop_west_avg
    drop_mesh_cols = pop_cols + [f'メッシュ{i}' for i in range(1, 10) if i != 5]
    drop_mesh_cols = [c for c in drop_mesh_cols if c in dataset.columns]
    dataset = dataset.drop(columns=drop_mesh_cols)
    dataset, encoders = label_enc(dataset)
    LAG_WINDOWS, ROLL_WINDOWS = get_feature_windows(
        context_length,
        prediction_length,
        max_window
    )
    print(f"LAG_WINDOWS: {LAG_WINDOWS}")
    print(f"ROLL_WINDOWS: {ROLL_WINDOWS}")
    dataset = dataset.sort_values(['書店コード', '書名', '日付']).reset_index(drop=True)
    for lag in LAG_WINDOWS:
        dataset[f'売上_lag{lag}'] = dataset.groupby(['書店コード', '書名'])['売上'].shift(lag)
    for window in ROLL_WINDOWS:
        dataset[f'売上_roll_mean{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        dataset[f'売上_roll_std{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)
        dataset[f'売上_roll_max{window}'] = dataset.groupby(['書店コード', '書名'])['売上'].rolling(window, min_periods=1).max().reset_index(level=[0,1], drop=True)
    dataset = dataset.fillna(0)
    dataset = downcast_integers(dataset)
    print("Scaling STATIC_REAL_COLS...")
    real_cols_to_scale = [col for col in STATIC_REAL_COLS if col in dataset.columns]
    scaler = StandardScaler()
    dataset[real_cols_to_scale] = scaler.fit_transform(dataset[real_cols_to_scale])
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to scaler.pkl")
    return dataset, encoders

def label_enc(df):
    df_encoded = df.copy()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_int_cols = [c for c in ['メッシュ5'] if c in df.columns and pd.api.types.is_integer_dtype(df[c])]
    cat_cols = object_cols + categorical_int_cols
    encoders = {col: LabelEncoder() for col in cat_cols}
    for col, le in encoders.items():
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded, encoders

def downcast_integers(df):
    df_downcasted = df.copy()
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32', 'int16']:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= 0:
                if col_max <= 255:
                    new_dtype = np.uint8
                elif col_max <= 65535:
                    new_dtype = np.uint16
                elif col_max <= 4294967295:
                    new_dtype = np.uint32
                else:
                    continue
            else:
                if col_min >= -128 and col_max <= 127:
                    new_dtype = np.int8
                elif col_min >= -32768 and col_max <= 32767:
                    new_dtype = np.int16
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    new_dtype = np.int32
                else:
                    continue
            original_values = df[col].copy()
            df_downcasted[col] = df[col].astype(new_dtype)
            if not (df_downcasted[col] == original_values).all():
                df_downcasted[col] = original_values
                print(f"Warning: {col} downcast verification failed, keeping original dtype")
    return df_downcasted

def get_feature_windows(context_length, prediction_length, max_window=None):
    daily_lags = list(range(1, 8))
    weekly_lags = list(range(14, context_length + 1, 7))
    lag_windows = sorted(list(set(daily_lags + weekly_lags)))
    lag_windows = [lag for lag in lag_windows if lag <= context_length]
    roll_windows = [3, 7, 14, 28, 56]
    roll_windows = sorted(list(set([w for w in roll_windows if w <= context_length])))
    return lag_windows, roll_windows

def create_slices(dataset, context_length, prediction_length, region_extend, max_window,
                  augmentation_factor=3, use_bidirectional=False):
    slices = []
    min_length = context_length + prediction_length
    for key, idx in dataset.groupby(['書店コード','書名']).indices.items():
        store, book = key
        group = dataset.iloc[idx]
        n_slices = int(group['n_slice'].iloc[0])
        group_sorted = group.sort_values('日付').reset_index(drop=True)
        date_first_idxs = group_sorted['日付'].values.searchsorted(np.unique(group_sorted['日付'].values))
        n_unique = len(date_first_idxs)
        region_size = n_unique / n_slices
        for i in range(n_slices):
            start_base = int(i * region_size)
            start_min = max(0, start_base - region_extend // 2)
            start_max = min(n_unique, start_base + region_extend // 2)
            start = np.random.randint(start_min, start_max + 1) if start_max > start_min else start_min
            end_base = int(start + region_size)
            end_min = end_base
            end_max = min(n_unique, end_base + region_extend)
            end = np.random.randint(end_min, end_max + 1) if end_max > end_min else end_min
            max_shift = min(min_length, n_unique - end)
            for aug_idx in range(augmentation_factor):
                if max_shift > 0 and augmentation_factor > 1:
                    shift_range = max_shift // augmentation_factor
                    shift_base = aug_idx * shift_range
                    shift_jitter = np.random.randint(-shift_range // 4, shift_range // 4 + 1) if shift_range >= 4 else 0
                    shift = max(0, min(max_shift, shift_base + shift_jitter))
                elif aug_idx == 0:
                    shift = 0
                else:
                    continue
                shifted_start = start + shift
                shifted_end = end + shift
                if shifted_end >= n_unique:
                    continue
                start_idx = int(date_first_idxs[shifted_start])
                end_idx = int(date_first_idxs[shifted_end]) - 1 if shifted_end < n_unique else len(group_sorted) - 1
                slice_data = group_sorted.iloc[start_idx:end_idx + 1].copy()
                slices.append(((store, book, i, aug_idx), slice_data))
    process_fn = process_group
    num_cores = max(1, cpu_count() - 1)
    with Pool(num_cores) as pool:
        print(f"Processing {len(slices)} samples (with {augmentation_factor}x augmentation)...")
        processed_list = pool.map(process_fn, slices)
    return processed_list

def process_group(args):
    key, group = args
    static_cat_cols = [c for c in STATIC_CAT_COLS if c in group.columns]
    static_cat = [int(group[col].iloc[0]) for col in static_cat_cols]
    static_real_cols = [c for c in STATIC_REAL_COLS if c in group.columns]
    static_real = [float(group[col].iloc[0]) for col in static_real_cols]
    for time_col in [c for c in TIME_COLS if c in group.columns]:
        time_val = group[time_col].iloc[0]
        static_real.append(float(time_val.hour + time_val.minute / 60.0))
    start_date = group['日付'].iloc[0]
    feat_dynamic_real = np.stack([
        group[col].values for col in TEMPORAL_COLS
    ], axis=0).astype(np.float32)
    lag_cols = sorted([c for c in group.columns if c.startswith('売上_lag')], key=lambda x: int(x.split('lag')[1]))
    roll_mean_cols = sorted([c for c in group.columns if 'roll_mean' in c], key=lambda x: int(x.split('mean')[1]))
    roll_std_cols = sorted([c for c in group.columns if 'roll_std' in c], key=lambda x: int(x.split('std')[1]))
    roll_max_cols = sorted([c for c in group.columns if 'roll_max' in c], key=lambda x: int(x.split('max')[1]))
    past_dynamic_cols = lag_cols + roll_mean_cols + roll_std_cols + roll_max_cols
    feat_dynamic_past = group[past_dynamic_cols].values.T.astype(np.float32)
    target_values = group['売上'].values.astype(np.float32)
    return {
        "start": pd.Period(start_date, freq="D"),
        "target": target_values,
        "feat_static_cat": static_cat,
        "feat_static_real": static_real,
        "feat_dynamic_real": feat_dynamic_real,
        "past_feat_dynamic_real": feat_dynamic_past
    }
