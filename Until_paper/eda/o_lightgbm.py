import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings('ignore')

print("=== Loading Data ===")
df = pd.read_parquet('data/sales_df.parquet')

rename_map = {
    '日付': 'date', '書店コード': 'store_id', '書名': 'book_id', 'POS販売冊数': 'sales',
    '駅距離': 'dist_station', '駅構内': 'is_in_station', '複合施設': 'is_complex', '独立店舗': 'is_standalone',
    '市区別_夜間人口': 'night_pop', '市区別_昼間人口': 'day_pop',
    '開店時間(平)': 'open_time_weekday', '閉店時間(平)': 'close_time_weekday',
    '営業時間(平)': 'business_hours_weekday', '開店時間(特)': 'open_time_special',
    '閉店時間(特)': 'close_time_special', '営業時間(特)': 'business_hours_special',
    '出版社': 'publisher', '著者名': 'author',
    '大分類': 'cat_large', '中分類': 'cat_medium', '小分類': 'cat_small', '本体価格': 'price'
}
df = df.rename(columns=rename_map)
df = df.sort_values(['store_id', 'book_id', 'date']).reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])

cat_cols = ['book_id', 'store_id', 'publisher', 'author', 'cat_large']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

print("=== Feature Engineering (Cyclical & Hierarchical) ===")

df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12).astype(np.float32)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12).astype(np.float32)

df['dow_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7).astype(np.float32)
df['dow_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7).astype(np.float32)

df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype('int8')

holidays = [
    '2024-01-01', '2024-01-08', '2024-02-11', '2024-02-12', '2024-02-23',
    '2024-03-20', '2024-04-29', '2024-05-03', '2024-05-04', '2024-05-05', '2024-05-06',
    '2024-07-15', '2024-08-11', '2024-08-12', '2024-09-16', '2024-09-22', '2024-09-23',
    '2024-10-14', '2024-11-03', '2024-11-04', '2024-11-23',
    '2024-12-29', '2024-12-30', '2024-12-31', '2025-01-01', '2025-01-02', '2025-01-03'
]
holiday_dt = pd.to_datetime(holidays)
df['is_holiday'] = df['date'].isin(holiday_dt).astype('int8')
df['days_to_weekend'] = (4 - df['date'].dt.dayofweek).clip(lower=0).astype('int8')

SHIFT_DAYS = 7
print(f"Creating Lags with shift={SHIFT_DAYS} days (No Leak)...")

grp = df.groupby(['store_id', 'book_id'], observed=True)['sales']

df['lag_7'] = grp.shift(SHIFT_DAYS).astype(np.float32)
df['lag_14'] = grp.shift(SHIFT_DAYS + 7).astype(np.float32)
df['rolling_mean_28'] = grp.transform(lambda x: x.shift(SHIFT_DAYS).rolling(28).mean()).astype(np.float32)
df['rolling_std_28'] = grp.transform(lambda x: x.shift(SHIFT_DAYS).rolling(28).std()).astype(np.float32)

print("Creating Hierarchical Lags...")

global_sales = df.groupby(['date', 'book_id'], observed=True)['sales'].sum().reset_index()
global_sales = global_sales.sort_values(['book_id', 'date'])

grp_global = global_sales.groupby('book_id', observed=True)['sales']
global_sales['global_lag_7'] = grp_global.shift(SHIFT_DAYS).astype(np.float32)
global_sales['global_rolling_mean_14'] = grp_global.transform(lambda x: x.shift(SHIFT_DAYS).rolling(14).mean()).astype(np.float32)

df = df.merge(global_sales[['date', 'book_id', 'global_lag_7', 'global_rolling_mean_14']],
              on=['date', 'book_id'], how='left')

cat_sales = df.groupby(['date', 'store_id', 'cat_large'], observed=True)['sales'].sum().reset_index()
cat_sales = cat_sales.sort_values(['store_id', 'cat_large', 'date'])

grp_cat = cat_sales.groupby(['store_id', 'cat_large'], observed=True)['sales']
cat_sales['cat_lag_7'] = grp_cat.shift(SHIFT_DAYS).astype(np.float32)
cat_sales['cat_rolling_mean_28'] = grp_cat.transform(lambda x: x.shift(SHIFT_DAYS).rolling(28).mean()).astype(np.float32)

df = df.merge(cat_sales[['date', 'store_id', 'cat_large', 'cat_lag_7', 'cat_rolling_mean_28']],
              on=['date', 'store_id', 'cat_large'], how='left')

df = df.dropna(subset=['rolling_mean_28'])
fill_cols = ['global_lag_7', 'global_rolling_mean_14', 'cat_lag_7', 'cat_rolling_mean_28']
df[fill_cols] = df[fill_cols].fillna(0)
df['sales'] = df['sales'].clip(lower=0)

del global_sales, cat_sales, grp, grp_global, grp_cat
gc.collect()

print("=== Weekly Walk-Forward Validation (Seed Averaging) ===")

time_cols = ['open_time_weekday', 'close_time_weekday', 'open_time_special', 'close_time_special']
for col in time_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

target = 'sales'
ignore_cols = ['date', 'sales']
features = [c for c in df.columns if c not in ignore_cols]
print(f"Features: {len(features)} variables")

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.33,
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1
}

start_date = pd.to_datetime('2024-11-01')
end_date = pd.to_datetime('2024-12-31')
val_periods = []
current = start_date
while current + pd.Timedelta(days=7) <= end_date:
    val_periods.append(current)
    current += pd.Timedelta(days=7)

results = []
for i, val_start in enumerate(val_periods):
    val_start_dt = val_start
    val_end_dt = val_start + pd.Timedelta(days=7)

    train_mask = df['date'] < val_start_dt
    val_mask = (df['date'] >= val_start_dt) & (df['date'] < val_end_dt)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    X_val = df.loc[val_mask, features]
    y_val = df.loc[val_mask, target]

    if len(X_val) == 0: continue

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_cols)

    # Seed Averaging
    seeds = [42, 43, 44, 45, 46]
    preds_list = []

    print(f"\nRound {i+1}/{len(val_periods)}: {val_start_dt.date()} ~ {val_end_dt.date()}")
    for seed in seeds:
        params['seed'] = seed
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )
        preds_list.append(model.predict(X_val))

    # Average predictions
    preds = np.mean(preds_list, axis=0)

    score = np.sqrt(mean_squared_error(y_val, preds))
    results.append({'start_date': val_start_dt, 'rmse': score})
    print(f"RMSE (Seed Averaged): {score:.4f}")

    del lgb_train, lgb_val
    gc.collect()

res_df = pd.DataFrame(results)
print(f"\n=== Results ===")
print(res_df)
print(f"\nAverage RMSE: {res_df['rmse'].mean():.4f}")

plt.figure(figsize=(12, 6))
plt.plot(res_df['start_date'], res_df['rmse'], marker='o')
plt.title("Weekly Forecast RMSE (Seed Averaging)")
plt.xlabel("Date")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.savefig('rmse_seed_averaging.png', dpi=150, bbox_inches='tight')
print("Saved plot to 'rmse_seed_averaging.png'")

lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8))
plt.title("Feature Importance (Seed Averaging)")
plt.tight_layout()
plt.savefig('importance_seed_averaging.png', dpi=150, bbox_inches='tight')
print("Saved importance to 'importance_seed_averaging.png'")

del df, X_train, X_val, y_train, y_val
gc.collect()

"""
╭─dmainf@MacBook-Pro ~/mac/data_comp/eda 
╰─$ python3 o_lightgbm.py
=== Loading Data ===
=== Feature Engineering (Cyclical & Hierarchical) ===
Creating Lags with shift=7 days (No Leak)...
Creating Hierarchical Lags...
=== Weekly Walk-Forward Validation (Seed Averaging) ===
Features: 63 variables

Round 1/8: 2024-11-01 ~ 2024-11-08
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[118]	training's rmse: 1.08079	valid_1's rmse: 1.65012
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[121]	training's rmse: 1.08038	valid_1's rmse: 1.65816
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[119]	training's rmse: 1.07743	valid_1's rmse: 1.65563
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[115]	training's rmse: 1.08148	valid_1's rmse: 1.66288
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[92]	training's rmse: 1.09368	valid_1's rmse: 1.66003
RMSE (Seed Averaged): 1.6571

Round 2/8: 2024-11-08 ~ 2024-11-15
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[254]	training's rmse: 1.05499	valid_1's rmse: 0.7575
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[360]	training's rmse: 1.03873	valid_1's rmse: 0.748831
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[137]	training's rmse: 1.08174	valid_1's rmse: 0.763802
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[359]	training's rmse: 1.0373	valid_1's rmse: 0.754072
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[237]	training's rmse: 1.05623	valid_1's rmse: 0.753757
RMSE (Seed Averaged): 0.7534

Round 3/8: 2024-11-15 ~ 2024-11-22
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[247]	training's rmse: 1.04751	valid_1's rmse: 0.788371
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[373]	training's rmse: 1.03104	valid_1's rmse: 0.783963
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[325]	training's rmse: 1.0413	valid_1's rmse: 0.7845
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[304]	training's rmse: 1.03725	valid_1's rmse: 0.787632
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[504]	training's rmse: 1.02804	valid_1's rmse: 0.782121
RMSE (Seed Averaged): 0.7846

Round 4/8: 2024-11-22 ~ 2024-11-29
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[395]	training's rmse: 1.03355	valid_1's rmse: 0.885544
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[426]	training's rmse: 1.02894	valid_1's rmse: 0.887042
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[558]	training's rmse: 1.01959	valid_1's rmse: 0.888217
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[389]	training's rmse: 1.0339	valid_1's rmse: 0.884441
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[189]	training's rmse: 1.05534	valid_1's rmse: 0.893482
RMSE (Seed Averaged): 0.8868

Round 5/8: 2024-11-29 ~ 2024-12-06
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[105]	training's rmse: 1.08541	valid_1's rmse: 2.1426
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[102]	training's rmse: 1.08889	valid_1's rmse: 2.14204
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[91]	training's rmse: 1.09253	valid_1's rmse: 2.14406
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[95]	training's rmse: 1.0912	valid_1's rmse: 2.1417
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[103]	training's rmse: 1.08708	valid_1's rmse: 2.1438
RMSE (Seed Averaged): 2.1427

Round 6/8: 2024-12-06 ~ 2024-12-13
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[208]	training's rmse: 1.08108	valid_1's rmse: 0.868676
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[283]	training's rmse: 1.06933	valid_1's rmse: 0.86063
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[210]	training's rmse: 1.07759	valid_1's rmse: 0.861581
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[195]	training's rmse: 1.07108	valid_1's rmse: 0.858523
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[205]	training's rmse: 1.07694	valid_1's rmse: 0.856923
RMSE (Seed Averaged): 0.8594

Round 7/8: 2024-12-13 ~ 2024-12-20
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[141]	training's rmse: 1.09976	valid_1's rmse: 1.18115
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[143]	training's rmse: 1.09952	valid_1's rmse: 1.18029
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[142]	training's rmse: 1.09928	valid_1's rmse: 1.18051
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[139]	training's rmse: 1.098	valid_1's rmse: 1.18118
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[124]	training's rmse: 1.10628	valid_1's rmse: 1.17814
RMSE (Seed Averaged): 1.1799

Round 8/8: 2024-12-20 ~ 2024-12-27
[LightGBM] [Warning] Met categorical feature which contains sparse values. Consider renumbering to consecutive integers started from zero
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[91]	training's rmse: 1.12459	valid_1's rmse: 4.22855
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[129]	training's rmse: 1.10801	valid_1's rmse: 4.22079
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[148]	training's rmse: 1.09954	valid_1's rmse: 4.22663
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[118]	training's rmse: 1.1138	valid_1's rmse: 4.22339
Training until validation scores don't improve for 30 rounds
Early stopping, best iteration is:
[117]	training's rmse: 1.11035	valid_1's rmse: 4.22663
RMSE (Seed Averaged): 4.2251

=== Results ===
  start_date      rmse
0 2024-11-01  1.657122
1 2024-11-08  0.753445
2 2024-11-15  0.784611
3 2024-11-22  0.886811
4 2024-11-29  2.142684
5 2024-12-06  0.859434
6 2024-12-13  1.179898
7 2024-12-20  4.225058

Average RMSE: 1.5611
Saved plot to 'rmse_seed_averaging.png'
Saved importance to 'importance_seed_averaging.png'
"""