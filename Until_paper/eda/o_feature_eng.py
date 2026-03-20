import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import calendar
from pathlib import Path
from functools import partial
import pickle
from multiprocessing import Pool, cpu_count

STATIC_CAT_COLS = ['書店コード', '書名', '出版社', '著者名', '大分類', '中分類', '小分類']
STATIC_REAL_COLS = ['本体価格', '営業時間(平)', '営業時間(特)', '駅構内', '複合施設', '独立店舗']
TIME_COLS = ['開店時間(平)', '閉店時間(平)', '開店時間(特)', '閉店時間(特)']
TEMPORAL_COLS = [
    'month_in_year_sin', 'month_in_year_cos', 'week_in_year_sin', 'week_in_year_cos',
    'day_in_year_sin', 'day_in_year_cos', 'week_in_month_sin', 'week_in_month_cos',
    'day_in_month_sin', 'day_in_month_cos', 'day_in_week_sin', 'day_in_week_cos',
    'is_weekend', 'quarter', 'mesh_pop_center', 'mesh_pop_neighbors_avg',
    'mesh_pop_gradient_NS', 'mesh_pop_gradient_EW'
]


def prepare_dataset(df, context_length, prediction_length):
    dataset = df.copy()

    dataset = filtering_dataset(df, context_length, prediction_length)
    dataset = add_temporal(dataset)
    dataset = add_mesh(dataset)
    dataset, encoders = label_enc(dataset)
    dataset = add_lag(dataset, context_length, prediction_length)
    dataset = scalling_float(dataset)

    return dataset, encoders

def filtering_dataset(dataset, context_length, prediction_length):
    print("Filtering dataset by minimum length...")
    min_length = context_length + prediction_length
    dataset = dataset.groupby(['書店コード', '書名']).filter(lambda g: g['日付'].nunique() >= min_length)
    dataset['n_slice'] = (dataset.groupby(['書店コード', '書名'])['日付'].transform(lambda x: x.nunique() // min_length))
    print("Complete!")
    return dataset

def add_temporal(dataset):
    """
    year
    month
    week
    day
    """
    print("Generating temporal features...")
    months_in_year = 12
    dataset['month_in_year_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.month / months_in_year).astype(np.float32)
    dataset['month_in_year_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.month / months_in_year).astype(np.float32)
    weeks_in_year = 53
    dataset['week_in_year_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.isocalendar().week / weeks_in_year).astype(np.float32)
    dataset['week_in_year_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.isocalendar().week / weeks_in_year).astype(np.float32)
    days_in_year = 366
    dataset['day_in_year_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.dayofyear / days_in_year).astype(np.float32)
    dataset['day_in_year_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.dayofyear / days_in_year).astype(np.float32)
    dataset['day_in_month'] = dataset['日付'].dt.day
    week_in_month = dataset['day_in_month'].apply(lambda d: (d - 1) // 7 + 1)
    dataset['week_in_month_sin'] = np.sin(2 * np.pi * week_in_month / week_in_month.max()).astype(np.float32)
    dataset['week_in_month_cos'] = np.cos(2 * np.pi * week_in_month / week_in_month.max()).astype(np.float32)
    days_in_month = dataset['日付'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    dataset['day_in_month_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.day / days_in_month).astype(np.float32)
    dataset['day_in_month_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.day / days_in_month).astype(np.float32)
    days_in_week = 7
    dataset['day_in_week_sin'] = np.sin(2 * np.pi * dataset['日付'].dt.dayofweek / days_in_week).astype(np.float32)
    dataset['day_in_week_cos'] = np.cos(2 * np.pi * dataset['日付'].dt.dayofweek / days_in_week).astype(np.float32)
    dataset['is_weekend'] = (dataset['日付'].dt.dayofweek >= 5).astype(np.int32)
    dataset['quarter'] = dataset['日付'].dt.quarter.astype(np.int32)
    print("Complete!")
    return dataset

def add_mesh(dataset):
    """
    0 1 2
    3 4 5
    6 7 8
    """
    print("Engineering spatial mesh features...")
    pop_cols = [f'メッシュ{i}_人口' for i in range(1, 10)]
    mesh_pops = dataset[pop_cols].values.astype(np.float32)
    dataset['mesh_pop_center'] = mesh_pops[:, 4]
    neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    neighbor_pops = mesh_pops[:, neighbor_indices]
    dataset['mesh_pop_neighbors_avg'] = np.mean(neighbor_pops, axis=1)
    pop_north_avg = np.mean(mesh_pops[:, [0, 1, 2]], axis=1)
    pop_south_avg = np.mean(mesh_pops[:, [6, 7, 8]], axis=1)
    dataset['mesh_pop_gradient_NS'] = pop_north_avg - pop_south_avg
    pop_east_avg = np.mean(mesh_pops[:, [2, 5, 8]], axis=1)
    pop_west_avg = np.mean(mesh_pops[:, [0, 3, 6]], axis=1)
    dataset['mesh_pop_gradient_EW'] = pop_east_avg - pop_west_avg
    dataset = dataset.drop(columns=[f'メッシュ{i}_人口' for i in range(1, 10)])
    print("Complete!")
    return dataset

def label_enc(df):
    print("Label encoding categorical features...")
    df_encoded = df.copy()
    object_cols = df.columns[df.dtypes == object]
    encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        values = df_encoded[col]
        df_encoded[col] = le.fit_transform(values).astype(np.int32)
        encoders[col] = le
    print("Complete!")
    return df_encoded, encoders

def add_lag(dataset, context_length, prediction_length):
    print("Creating lag and rolling features...")
    ROLL_WINDOWS = get_feature_windows(context_length, prediction_length)
    dataset = dataset.sort_values(['書店コード', '書名', '日付']).reset_index(drop=True)
    for window in ROLL_WINDOWS:
        dataset[f'POS販売冊数_roll_mean{window}'] = dataset.groupby(['書店コード', '書名'])['POS販売冊数'].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True).astype(np.float32)
        dataset[f'POS販売冊数_roll_std{window}'] = dataset.groupby(['書店コード', '書名'])['POS販売冊数'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True).astype(np.float32)
        dataset[f'POS販売冊数_roll_max{window}'] = dataset.groupby(['書店コード', '書名'])['POS販売冊数'].rolling(window, min_periods=1).max().reset_index(level=[0,1], drop=True).astype(np.float32)
    dataset = dataset.fillna(0)
    print("Complete!")
    return dataset

def get_feature_windows(context_length, prediction_length):
    print("Getting feature windows...")
    window = [3, 5, 7]
    min_length = context_length + prediction_length
    window += [2**k for k in range(3, 10) if 2**k <= min_length]
    print("Complete!")
    return sorted(set(window))

def scalling_float(dataset):
    print("Scaling STATIC_REAL_COLS...")
    real_cols_to_scale = dataset.select_dtypes(include=['float32']).columns.tolist()
    scaler = StandardScaler()
    dataset[real_cols_to_scale] = scaler.fit_transform(dataset[real_cols_to_scale])
    print("Complete!")
    return dataset


def create_slices(dataset, context_length, prediction_length, region_extend, augmentation_factor):
    slices = []
    min_length = context_length + prediction_length
    for key, idx in dataset.groupby(['書店コード','書名']).indices.items():
        store, book = key
        group = dataset.iloc[idx]
        n_slices = group['n_slice'].iloc[0].astype(np.int32)
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
    num_cores = max(1, cpu_count() - 1)
    with Pool(num_cores) as pool:
        print(f"Processing {len(slices)} samples (with {augmentation_factor}x augmentation)...")
        result = pool.map(process_group, slices)
    return result

def process_group(args):
    key, group = args
    static_cat = [int(group[col].iloc[0]) for col in STATIC_CAT_COLS if col in group.columns]
    static_real = [float(group[col].iloc[0]) for col in STATIC_REAL_COLS if col in group.columns]
    for time_col in [c for c in TIME_COLS if c in group.columns]:
        time_val = group[time_col].iloc[0]
        static_real.append(float(time_val.hour + time_val.minute / 60.0))

    start_date = group['日付'].iloc[0]
    feat_dynamic_real = np.stack([group[col].values for col in TEMPORAL_COLS if col in group.columns], axis=0).astype(np.float32)

    roll_cols = sorted([c for c in group.columns if 'POS販売冊数_roll' in c],
                      key=lambda x: int(x.split('roll_')[1].split('mean')[-1].split('std')[-1].split('max')[-1]))
    past_feat_dynamic_real = group[roll_cols].values.T.astype(np.float32) if roll_cols else np.array([]).astype(np.float32)

    target_values = group['POS販売冊数'].values.astype(np.float32)

    return {
        "start": pd.Period(start_date, freq="D"),
        "target": target_values,
        "feat_static_cat": static_cat,
        "feat_static_real": static_real,
        "feat_dynamic_real": feat_dynamic_real,
        "past_feat_dynamic_real": past_feat_dynamic_real
    }
