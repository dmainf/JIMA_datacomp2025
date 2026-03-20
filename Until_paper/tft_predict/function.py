import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.distributions import QuantileOutput, NegativeBinomialOutput
from pytorch_lightning.loggers import CSVLogger
import os
import shutil
from numba import jit
from typing import Tuple

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _get_latest_version(log_dir):
    version_dirs = [d for d in os.listdir(log_dir)
                    if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    return sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1] if version_dirs else None


def extract_decile_books(df):
    print("\n=== Identifying Representative Books (Deciles - 10% intervals) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    quantile_names = ["Min (0%)", "D1 (10%)", "D2 (20%)", "D3 (30%)", "D4 (40%)",
                      "Median (50%)", "D6 (60%)", "D7 (70%)", "D8 (80%)", "D9 (90%)", "Max (100%)"]
    decile_books = {}
    print(f"{'Decile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)
    num_books = len(total_sales)
    for q, name in zip(quantiles, quantile_names):
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]
        decile_books[book_name] = {
            "label": name,
            "total_sales": sales_val
        }
        print(f"{name:<15} | {sales_val:,.0f}冊       | {book_name}")
    return decile_books


@jit(nopython=True)
def find_first_significant_peak(
    rho: np.ndarray,
    min_lag: int,
    max_lag: int,
    threshold_ratio: float = 0.75,
    min_score: float = 0.25
) -> Tuple[int, float]:
    """
    YINアルゴリズム的First Significant Peak探索（Red Noise対策付き）

    1. Strict Local Peak: ρ(τ-1) < ρ(τ) > ρ(τ+1) を必須条件化
    2. 最小スコア閾値: min_score未満の弱い相関を無視
    3. First Peak優先: 倍音よりも基本波を優先
    """
    global_max = -1.0
    for tau in range(min_lag, max_lag + 1):
        if rho[tau] > global_max:
            global_max = rho[tau]

    if global_max < min_score:
        return 0, 0.0

    threshold = max(global_max * threshold_ratio, min_score)

    for tau in range(min_lag, max_lag):
        val = rho[tau]
        if val >= threshold:
            if val > rho[tau - 1] and val > rho[tau + 1]:
                return tau, global_max

    return 0, 0.0


@jit(nopython=True)
def compute_online_periodicity(
    sales: np.ndarray,
    min_lag: int = 3,
    max_lag: int = 60,
    alpha: float = 0.05,
    threshold_ratio: float = 0.75,
    min_score: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    オンライン自己相関 + Red Noise対策付きピーク選択による周期性スコア計算

    改良点：
    - Strict Local Peak: AR(1)プロセスの単調減衰を除外
    - 最小スコア閾値: 弱い相関による誤検知を防止
    - First Significant Peak: 倍音よりも基本波を優先
    """
    n = len(sales)
    scores = np.zeros(n, dtype=np.float32)
    detected_periods = np.zeros(n, dtype=np.float32)

    mu = sales[0]
    var = 0.0

    rho = np.zeros(max_lag + 2, dtype=np.float32)
    history = np.zeros(max_lag + 1, dtype=np.float32)

    for t in range(n):
        val = sales[t]

        diff = val - mu
        mu = mu + alpha * diff
        var = (1.0 - alpha) * var + alpha * (diff * (val - mu))

        std = 1.0 if var < 1e-6 else np.sqrt(var)

        z_t = (val - mu) / std
        z_t = max(-5.0, min(5.0, z_t))

        for lag in range(max_lag, 0, -1):
            history[lag] = history[lag - 1]
        history[0] = z_t

        for lag in range(max_lag + 1):
            rho[lag] = (1.0 - alpha) * rho[lag] + alpha * (z_t * history[lag])

        if t >= min_lag:
            best_lag, max_rho = find_first_significant_peak(
                rho, min_lag, max_lag, threshold_ratio, min_score
            )

            scores[t] = max_rho
            detected_periods[t] = float(best_lag)

    return scores, detected_periods


@jit(nopython=True)
def compute_regime_features(
    sales: np.ndarray,
    hawkes_decay: float = 0.1,
    initial_adi: float = 30.0,
    initial_cv2: float = 1.0,
    base_alpha: float = 0.1,
    min_period_lag: int = 3,
    max_period_lag: int = 60,
    period_alpha: float = 0.05,
    period_threshold_ratio: float = 0.75,
    period_min_score: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    オンライン自己相関 + Red Noise対策による周期性検出を含む特徴量計算

    Returns:
        - feat_adi: 平均需要間隔
        - feat_cv2: 需要間隔の変動係数の二乗
        - feat_hawkes: Hawkes Process値
        - feat_days_since: 前回売上からの経過日数
        - feat_periodicity_score: 周期性スコア（オンライン自己相関ベース）
        - feat_detected_period: 検出された周期（日数）
    """
    n = len(sales)

    feat_adi = np.full(n, initial_adi, dtype=np.float32)
    feat_cv2 = np.full(n, initial_cv2, dtype=np.float32)
    feat_hawkes = np.zeros(n, dtype=np.float32)
    feat_days_since = np.full(n, np.inf, dtype=np.float32)

    current_adi = initial_adi
    current_cv2 = initial_cv2
    current_sales_mean_interval = initial_adi
    current_sales_var_interval = 0.0

    last_hawkes = 0.0
    last_event_day = -1.0 - initial_adi

    for t in range(n):
        days_since_sales = float(t) - last_event_day
        feat_days_since[t] = days_since_sales

        effective_adi = max(current_adi, days_since_sales)
        feat_adi[t] = effective_adi
        feat_cv2[t] = current_cv2

        dt = 1.0
        current_hawkes_val = last_hawkes * np.exp(-hawkes_decay * dt)
        feat_hawkes[t] = current_hawkes_val

        if sales[t] > 0:
            last_hawkes = current_hawkes_val + sales[t]
            if last_event_day >= 0:
                new_interval = t - last_event_day
                delta = new_interval - current_sales_mean_interval
                current_sales_mean_interval += base_alpha * delta
                current_sales_var_interval = (1.0 - base_alpha) * (current_sales_var_interval + base_alpha * delta**2)

                current_adi = current_sales_mean_interval
                if current_sales_mean_interval > 1e-6:
                    current_cv2 = (np.sqrt(current_sales_var_interval) / current_sales_mean_interval) ** 2
                else:
                    current_cv2 = 1.0
            last_event_day = float(t)
        else:
            last_hawkes = current_hawkes_val

    feat_periodicity_score, feat_detected_period = compute_online_periodicity(
        sales,
        min_lag=min_period_lag,
        max_lag=max_period_lag,
        alpha=period_alpha,
        threshold_ratio=period_threshold_ratio,
        min_score=period_min_score
    )

    return feat_adi, feat_cv2, feat_hawkes, feat_days_since, feat_periodicity_score, feat_detected_period


def calculate_hazard_features(
    df,
    context_length,
    recent_period,
    spike_thresholds=[1.5, 2.0, 2.5, 3.0]
):
    print("\n=== Calculating Hazard Features ===")
    print(f"Context length: {context_length} days")
    print(f"Recent period (momentum): {recent_period} days")
    print(f"Spike thresholds: {spike_thresholds}")

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)
    n = len(df)

    momentum_arr = np.ones(n, dtype=np.float32)
    volatility_arr = np.zeros(n, dtype=np.float32)
    z_score_arr = np.zeros(n, dtype=np.float32)
    days_since_spike_dict = {threshold: np.zeros(n, dtype=np.int32) for threshold in spike_thresholds}

    for book_name, group in df.groupby('書名', observed=False):
        indices = group.index.values
        sales = group['POS販売冊数'].values

        last_spike_idx_dict = {threshold: -1 for threshold in spike_thresholds}

        for i in range(len(sales)):
            current_idx = indices[i]
            start_local = max(0, i - context_length + 1)

            current_z_score = 0.0
            if i > 0:
                hist_sales_excl_today = sales[start_local : i]

                if len(hist_sales_excl_today) >= recent_period:
                    rec_mean = np.mean(hist_sales_excl_today[-recent_period:])
                    hist_mean = np.mean(hist_sales_excl_today)
                    momentum = rec_mean / hist_mean if hist_mean > 0 else 1.0
                else:
                    momentum = 1.0

                volatility = np.std(hist_sales_excl_today) if len(hist_sales_excl_today) > 1 else 0.0

                if len(hist_sales_excl_today) > 0:
                    hist_mean = np.mean(hist_sales_excl_today)
                    hist_std = np.std(hist_sales_excl_today)
                    if hist_std > 0:
                        current_z_score = (sales[i] - hist_mean) / hist_std
            else:
                momentum = 1.0
                volatility = 0.0

            momentum_arr[current_idx] = momentum
            volatility_arr[current_idx] = volatility
            z_score_arr[current_idx] = current_z_score

            for threshold in spike_thresholds:
                last_spike_idx = last_spike_idx_dict[threshold]

                if last_spike_idx == -1:
                    days_since = context_length
                else:
                    days_since = i - last_spike_idx

                if current_z_score >= threshold:
                    last_spike_idx_dict[threshold] = i

                days_since_spike_dict[threshold][current_idx] = days_since

    df['momentum'] = momentum_arr
    df['volatility'] = volatility_arr
    df['z_score'] = z_score_arr

    for threshold in spike_thresholds:
        col_name = f'days_since_spike_{threshold}'
        df[col_name] = days_since_spike_dict[threshold]

    for threshold in spike_thresholds:
        col_name = f'is_spike_{threshold}'
        df[col_name] = (df['z_score'] >= threshold).astype(np.int8)

    print("\n=== Hazard Features Statistics ===")
    print(f"Momentum - Mean: {df['momentum'].mean():.4f}, Std: {df['momentum'].std():.4f}")
    print(f"Volatility - Mean: {df['volatility'].mean():.4f}, Std: {df['volatility'].std():.4f}")
    print(f"Z-score - Mean: {df['z_score'].mean():.4f}, Std: {df['z_score'].std():.4f}")
    for threshold in spike_thresholds:
        col_name = f'days_since_spike_{threshold}'
        print(f"Days since spike ({threshold}) - Mean: {df[col_name].mean():.1f}, Median: {df[col_name].median():.1f}")
    for threshold in spike_thresholds:
        col_name = f'is_spike_{threshold}'
        spike_rate = df[col_name].mean() * 100
        print(f"Is spike ({threshold}) - Spike rate: {spike_rate:.2f}%")

    return df


def calculate_regime_features(
    df,
    hawkes_decay=0.1,
    initial_adi=30.0,
    initial_cv2=1.0,
    base_alpha=0.1,
    min_period_lag=3,
    max_period_lag=60,
    period_alpha=0.05,
    period_threshold_ratio=0.75,
    period_min_score=0.25,
    lambda_adi=30.0
):
    """
    Red Noise対策付きオンライン自己相関による周期性検出を含む特徴量計算

    改良点：
    - スパイク依存を完全排除（is_spike不要）
    - オンライン自己相関による周期性検出
    - YIN的倍音抑制（7日vs14日で基本波を優先）
    - AR(1)プロセス（Red Noise）対策
    """
    print("\n=== Calculating Regime Features (Red Noise Resistant) ===")
    print(f"Hawkes decay: {hawkes_decay}")
    print(f"ADI parameters: initial={initial_adi}, base_alpha={base_alpha}")
    print(f"Periodicity detection:")
    print(f"  - min_period_lag={min_period_lag} (AR(1)ノイズ除外)")
    print(f"  - max_period_lag={max_period_lag}")
    print(f"  - period_alpha={period_alpha} (学習率)")
    print(f"  - period_threshold_ratio={period_threshold_ratio} (倍音抑制)")
    print(f"  - period_min_score={period_min_score} (弱い相関除外)")

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)

    regime_results = []
    for item_name, group in df.groupby('書名', observed=True):
        sales_array = group['POS販売冊数'].values.astype(np.float32)

        feat_adi, feat_cv2, feat_hawkes, feat_days_since, feat_periodicity_score, feat_detected_period = compute_regime_features(
            sales_array,
            hawkes_decay=hawkes_decay,
            initial_adi=initial_adi,
            initial_cv2=initial_cv2,
            base_alpha=base_alpha,
            min_period_lag=min_period_lag,
            max_period_lag=max_period_lag,
            period_alpha=period_alpha,
            period_threshold_ratio=period_threshold_ratio,
            period_min_score=period_min_score
        )

        regime_results.append(pd.DataFrame({
            'feat_adi': feat_adi,
            'feat_cv2': feat_cv2,
            'feat_hawkes': feat_hawkes,
            'feat_days_since': feat_days_since,
            'feat_periodicity_score': feat_periodicity_score,
            'feat_detected_period': feat_detected_period
        }, index=group.index))

    regime_features_df = pd.concat(regime_results, axis=0).sort_index()
    df = pd.concat([df, regime_features_df], axis=1)

    print("\n=== Calculating Regime Scores ===")
    df['score_sparse'] = (1.0 - np.exp(-df['feat_adi'] / lambda_adi)).astype(np.float32)

    rolling_max_hawkes = df.groupby('書名', observed=False)['feat_hawkes'].transform(
        lambda x: x.expanding(min_periods=1).max()
    )
    scale_hawkes = rolling_max_hawkes.replace(0, 1.0) * 0.5
    df['score_burst'] = np.tanh(df['feat_hawkes'] / scale_hawkes).astype(np.float32)

    df['score_periodic'] = df['feat_periodicity_score'].clip(0.0, 1.0).astype(np.float32)

    print("\n=== Regime Features Statistics ===")
    print(f"ADI - Mean: {df['feat_adi'].mean():.4f}, Std: {df['feat_adi'].std():.4f}")
    print(f"CV2 (Sales) - Mean: {df['feat_cv2'].mean():.4f}, Std: {df['feat_cv2'].std():.4f}")
    print(f"Hawkes - Mean: {df['feat_hawkes'].mean():.4f}, Std: {df['feat_hawkes'].std():.4f}")
    print(f"Days Since Event - Mean: {df['feat_days_since'].mean():.4f}, Median: {df['feat_days_since'].median():.4f}")
    print(f"Periodicity Score - Mean: {df['feat_periodicity_score'].mean():.4f}, Std: {df['feat_periodicity_score'].std():.4f}")
    print(f"Detected Period - Mean: {df['feat_detected_period'].mean():.1f} days, Median: {df['feat_detected_period'].median():.1f} days")

    detected_periods = df[df['feat_detected_period'] > 0]['feat_detected_period']
    if len(detected_periods) > 0:
        print(f"  Detection rate: {len(detected_periods) / len(df) * 100:.1f}%")
        period_counts = detected_periods.value_counts().head(5)
        print(f"  Top 5 detected periods:")
        for period, count in period_counts.items():
            print(f"    {period:.0f}日: {count}回 ({count/len(detected_periods)*100:.1f}%)")

    print("\n=== Regime Scores Statistics ===")
    print(f"Sparse Score - Mean: {df['score_sparse'].mean():.4f}, Std: {df['score_sparse'].std():.4f}")
    print(f"Periodic Score - Mean: {df['score_periodic'].mean():.4f}, Std: {df['score_periodic'].std():.4f}")
    print(f"  High periodicity (≥0.5): {(df['score_periodic'] >= 0.5).sum() / len(df) * 100:.1f}%")
    print(f"Burst Score - Mean: {df['score_burst'].mean():.4f}, Std: {df['score_burst'].std():.4f}")

    return df


def scaling_data(df, use_log_scale):
    if use_log_scale:
        df['POS販売冊数'] = np.log1p(df['POS販売冊数']).astype(np.float32)
        print("\n=== Applied log1p transformation to POS販売冊数 ===")

        df['log_本体価格'] = np.log1p(df['本体価格']).astype(np.float32)
        df = df.drop('本体価格', axis=1)
        print(f"\n=== Applied log1p transformation to 本体価格 ===")
        print(f"Created 'log_本体価格' column and removed original '本体価格' column")
        print(f"Log price range: {df['log_本体価格'].min():.4f} - {df['log_本体価格'].max():.4f}")
    else:
        df['log_本体価格'] = df['本体価格'].astype(np.float32)
        df = df.drop('本体価格', axis=1)
        print("\n=== No scaling applied ===")
        print("Renamed '本体価格' to 'log_本体価格'")

    print("\n=== POS Sales Statistics ===")
    print(f"Mean: {df['POS販売冊数'].mean():.4f}")
    print(f"Variance: {df['POS販売冊数'].var():.4f}")
    print(f"Std Dev: {df['POS販売冊数'].std():.4f}")

    return df


def calculate_temporal_relative_features(df, target_col, category_cols=['大分類', '中分類', '小分類'], context_length=180):
    print(f"\n=== Calculating Temporal Relative Features for {target_col} ===")
    print(f"Category columns: {category_cols}")
    print(f"Context length: {context_length} days (using past data only to prevent data leakage)")

    df_result = df.copy()

    if target_col not in df_result.columns:
        print(f"Warning: {target_col} not found in dataframe, skipping...")
        return df_result

    print(f"{target_col} range: min={df_result[target_col].min()}, max={df_result[target_col].max()}")

    df_result = df_result.sort_values(['書名', '日付']).reset_index(drop=True)

    for category_col in category_cols:
        if category_col not in df_result.columns:
            print(f"Warning: {category_col} not found in dataframe, skipping...")
            continue

        print(f"\n--- Processing {category_col} for {target_col} ---")

        relative_col = f'{category_col}_{target_col}_relative'
        z_score_col = f'{category_col}_{target_col}_z_score'
        category_daily = df_result.groupby([category_col, '日付'], observed=False)[target_col].mean().reset_index()
        category_daily = category_daily.rename(columns={target_col: f'{target_col}_category_mean'})
        category_daily = category_daily.sort_values([category_col, '日付'])
        category_daily['category_rolling_mean'] = category_daily.groupby(category_col, observed=False)[f'{target_col}_category_mean'].transform(
            lambda x: x.shift(1).rolling(window=context_length, min_periods=1).mean()
        )

        category_daily['category_rolling_std'] = category_daily.groupby(category_col, observed=False)[f'{target_col}_category_mean'].transform(
            lambda x: x.shift(1).rolling(window=context_length, min_periods=1).std()
        )
        df_result = df_result.merge(
            category_daily[[category_col, '日付', 'category_rolling_mean', 'category_rolling_std']],
            on=[category_col, '日付'],
            how='left'
        )
        df_result[relative_col] = (
            df_result[target_col] - df_result['category_rolling_mean']
        ).fillna(0).astype(np.float32)

        df_result[z_score_col] = (
            (df_result[target_col] - df_result['category_rolling_mean']) /
            np.maximum(df_result['category_rolling_std'], 1e-8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10).astype(np.float32)
        df_result = df_result.drop(['category_rolling_mean', 'category_rolling_std'], axis=1)
        print(f"Created features:")
        print(f"  - {relative_col} (Mean: {df_result[relative_col].mean():.4f}, Std: {df_result[relative_col].std():.4f})")
        print(f"  - {z_score_col} (Mean: {df_result[z_score_col].mean():.4f}, Std: {df_result[z_score_col].std():.4f})")

    print(f"\n=== Temporal Relative Features for {target_col} Complete ===")
    return df_result


def calculate_static_relative_features(df, target_col, category_cols=['大分類', '中分類', '小分類']):
    print(f"\n=== Calculating Static Relative Features for {target_col} ===")
    print(f"Category columns: {category_cols}")

    unique_books = df[['書名', target_col] + category_cols].drop_duplicates()

    for cat_col in category_cols:
        if cat_col not in unique_books.columns:
            print(f"Warning: {cat_col} not found in dataframe, skipping...")
            continue

        print(f"\n--- Processing {cat_col} for {target_col} ---")

        grp = unique_books.groupby(cat_col, observed=False)[target_col]
        mean_val = grp.transform('mean')
        std_val = grp.transform('std').fillna(1.0)

        relative_col = f'{cat_col}_{target_col}_relative'
        z_score_col = f'{cat_col}_{target_col}_z_score'

        unique_books[relative_col] = (unique_books[target_col] - mean_val).astype(np.float32)
        unique_books[z_score_col] = ((unique_books[target_col] - mean_val) / np.maximum(std_val, 1e-8)).astype(np.float32)

        print(f"Created features:")
        print(f"  - {relative_col} (Mean: {unique_books[relative_col].mean():.4f}, Std: {unique_books[relative_col].std():.4f})")
        print(f"  - {z_score_col} (Mean: {unique_books[z_score_col].mean():.4f}, Std: {unique_books[z_score_col].std():.4f})")

    static_features = [c for c in unique_books.columns if f'_{target_col}_' in c]
    df_result = df.merge(unique_books[['書名'] + static_features], on='書名', how='left')

    print(f"\nCreated static features: {static_features}")
    print(f"=== Static Relative Features for {target_col} Complete ===")
    return df_result


def verify_features(df, feature_list, feature_category):
    print(f"\n=== Verifying {feature_category} Features ===")
    missing_features = []
    existing_features = []
    for feature in feature_list:
        if feature in df.columns:
            existing_features.append(feature)
        else:
            missing_features.append(feature)
    if existing_features:
        print(f"Found {len(existing_features)}/{len(feature_list)} features:")
        for feature in existing_features:
            print(f"  ✓ {feature}")
    if missing_features:
        print(f"\nMISSING {len(missing_features)} features:")
        for feature in missing_features:
            print(f"  ✗ {feature}")
        raise ValueError(f"Missing required features: {missing_features}")
    print(f"All {len(feature_list)} {feature_category} features verified successfully!")
    return True


def verify_all_features(df, static_cols, time_cols, relative_cols, hazard_cols):
    verify_features(df, static_cols, "Static")
    verify_features(df, time_cols, "Time")
    verify_features(df, relative_cols, "Relative")
    verify_features(df, hazard_cols, "Hazard")
    past_dynamic_cols = relative_cols + hazard_cols
    print(f"\n=== Feature Summary ===")
    print(f"Static features: {len(static_cols)}")
    print(f"Time features (known covariates): {len(time_cols)}")
    print(f"Relative features: {len(relative_cols)}")
    print(f"Hazard features: {len(hazard_cols)}")
    print(f"Total past dynamic features: {len(past_dynamic_cols)}")
    print(f"Total features: {len(static_cols) + len(time_cols) + len(past_dynamic_cols)}")


def save_training_data(df, filepath='train.parquet'):
    print("\n=== Saving Training Data ===")
    df.to_parquet(filepath, index=False)
    print(f"Training data saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")


def prepare_dataset(df, static_cols, time_feature_cols, past_dynamic_cols, prediction_length):
    print(f"\n=== Preparing Dataset ===")
    print(f"Static features: {static_cols}")
    print(f"Time features (known covariates): {time_feature_cols}")
    print(f"Past dynamic features (observed covariates): {past_dynamic_cols}")

    df['id'] = df['書名'].astype(str)
    df = df.sort_values(['id', '日付']).reset_index(drop=True)

    for col in static_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.remove_unused_categories()

    static_df = df.groupby('id', observed=False)[static_cols].first().reset_index()
    cardinality = [static_df[col].cat.categories.size for col in static_cols]
    print(f"Cardinality: {dict(zip(static_cols, cardinality))}")
    static_df_formatted = static_df[['id'] + static_cols].set_index('id')

    dataset = PandasDataset.from_long_dataframe(
        df,
        target='POS販売冊数',
        item_id='id',
        timestamp='日付',
        freq='D',
        static_features=static_df_formatted,
        feat_dynamic_real=time_feature_cols,
        past_feat_dynamic_real=past_dynamic_cols
    )
    full_dataset = list(dataset)
    print(f"Number of time series: {len(full_dataset)}")
    print("\n=== Creating Training-Specific Dataset (No Data Leakage) ===")
    train_dataset = []
    for entry in full_dataset:
        train_entry = entry.copy()
        train_entry["target"] = entry["target"][:-prediction_length]
        if "feat_dynamic_real" in entry:
            train_entry["feat_dynamic_real"] = entry["feat_dynamic_real"]
        if "past_feat_dynamic_real" in entry:
            train_entry["past_feat_dynamic_real"] = entry["past_feat_dynamic_real"][:, :-prediction_length]
        train_dataset.append(train_entry)
    print("Training dataset created by removing prediction length from target and past features.")
    print("Note: feat_dynamic_real (known future covariates) is NOT truncated to preserve calendar/price information.")
    return full_dataset, train_dataset, cardinality


def plot_training_history():
    log_dir = 'lightning_logs/tft_training'
    save_path = 'tft/training_history.png'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    latest_version = _get_latest_version(log_dir)
    if not latest_version:
        print("No lightning_logs version directory found.")
        return
    metrics_file = os.path.join(log_dir, latest_version, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print("No metrics file found in the latest version.")
        return
    print(f"Reading metrics from: {metrics_file}")
    df = pd.read_csv(metrics_file)
    plt.figure(figsize=(12, 6))
    if 'train_loss' in df.columns:
        train_df = df[['epoch', 'train_loss']].dropna()
        if not train_df.empty:
            plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    if 'val_loss' in df.columns:
        val_df = df[['epoch', 'val_loss']].dropna()
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training history', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def train_model(train_dataset, cardinality, prediction_length, context_length, epochs, quantiles, distribution="negative_binomial"):
    csv_logger = CSVLogger("lightning_logs", name="tft_training")
    print("\n=== Training ===")

    batch_size = 128
    num_workers = os.cpu_count() or 4
    print(f"Training config: batch_size={batch_size}, num_workers={num_workers}")

    if distribution == "quantile":
        print("Distribution: QuantileOutput (Non-parametric, flexible for long-tail and intermittent demand)")
        print(f"Quantiles: {quantiles}")
        distr_output = QuantileOutput(quantiles=quantiles)
    else:  # negative_binomial
        print("Distribution: NegativeBinomialOutput (Count data distribution for non-negative integers)")
        distr_output = NegativeBinomialOutput()
    estimator = TemporalFusionTransformerEstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        static_cardinalities=cardinality,
        distr_output=distr_output,
        batch_size=batch_size,
        trainer_kwargs={
            "max_epochs": epochs,
            "accelerator": "auto",
            "logger": csv_logger,
            "enable_progress_bar": True,
            "enable_model_summary": False
        }
    )
    predictor = estimator.train(training_data=train_dataset, num_workers=num_workers)
    return predictor


def process_logs():
    log_dir = 'lightning_logs/tft_training'
    print("\n=== Plotting Training History ===")
    plot_training_history()
    print("\n=== Cleaning up logs ===")
    if not os.path.exists(log_dir):
        print("lightning_logs directory not found, no cleanup performed.")
        return
    latest_version = _get_latest_version(log_dir)
    if not latest_version:
        print("No versioned lightning_logs directories found to clean.")
        return
    latest_version_dir_path = os.path.join(log_dir, latest_version)
    version_dirs = [d for d in os.listdir(log_dir)
                    if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    for item in version_dirs:
        item_path = os.path.join(log_dir, item)
        if item_path != latest_version_dir_path:
            shutil.rmtree(item_path)
    print("Older lightning_logs versions removed")


def save_forecast(forecast, actual_data_log, save_path, title, use_log_scale, prediction_length, context_length, quantiles):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
    history_end_date = start_timestamp - pd.Timedelta(days=1)
    history_start_date = start_timestamp - pd.Timedelta(days=context_length)
    if hasattr(actual_data_log.index, 'to_timestamp'):
        actual_data_log.index = actual_data_log.index.to_timestamp()
    past_series = actual_data_log[history_start_date:history_end_date]
    future_end_date = start_timestamp + pd.Timedelta(days=prediction_length - 1)
    target_series = actual_data_log[start_timestamp:future_end_date]
    history_dates = past_series.index if not past_series.empty else []
    history_values = past_series.values.flatten() if not past_series.empty else []
    target_dates = target_series.index if not target_series.empty else []
    target_values = target_series.values.flatten() if not target_series.empty else []
    forecast_dates = [start_timestamp + pd.Timedelta(days=i) for i in range(prediction_length)]
    q_vals = {q: forecast.quantile(q) for q in quantiles}
    if use_log_scale:
        history_values = np.expm1(history_values) if len(history_values) > 0 else history_values
        target_values = np.expm1(target_values) if len(target_values) > 0 else target_values
        q_vals = {q: np.expm1(v) for q, v in q_vals.items()}
    plt.figure(figsize=(12, 6))
    if len(history_dates) > 0:
        plt.plot(history_dates, history_values, label="Actual History", color='black', linestyle='--')
    if len(target_dates) > 0:
        plt.plot(target_dates, target_values, label="Actual Target", color='black', marker='o', markersize=3)
    quantile_set = set(quantiles)
    pairs = []
    for q in sorted(quantiles):
        if q < 0.5:
            complement = round(1 - q, 2)
            if complement in quantile_set:
                pairs.append((q, complement))
    colors = ['blue', 'cyan', 'lightgreen', 'yellow']
    alphas = [0.15, 0.3, 0.45, 0.6]
    for idx, (lower, upper) in enumerate(pairs):
        coverage = int((upper - lower) * 100)
        color = colors[idx % len(colors)]
        alpha = alphas[idx % len(alphas)]
        plt.fill_between(forecast_dates, q_vals[lower], q_vals[upper],
                        color=color, alpha=alpha,
                        label=f"{coverage}% Interval ({int(lower*100)}-{int(upper*100)}%)")
    if 0.5 in q_vals:
        plt.plot(forecast_dates, q_vals[0.5], label="Median (50%)", color='blue', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("日付")
    plt.ylabel("POS販売冊数")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_prediction(predictor, full_dataset, use_log_scale=False):
    print("\n=== Evaluating ===")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    if use_log_scale:
        print("Log scale mode: Skipping GluonTS Evaluator (metrics would be on log-transformed data)")
        print("Accurate metrics will be calculated by evaluate_predictions() with inverse transformation")
    else:
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(tss, forecasts)
        print("\n=== Metrics ===")
        for key, value in agg_metrics.items():
            print(f"{key}: {value:.4f}")
    return forecasts, tss


def process_predictions(forecasts, tss, decile_books, df, all_title_predict, use_log_scale, prediction_length, context_length, quantiles):
    print("\n=== Saving Predictions ===")
    print("Mode: Saving ALL book predictions" if all_title_predict else "Mode: Saving ONLY representative book predictions (min, deciles, max)")
    os.makedirs('tft/All_predict', exist_ok=True)

    saved_count = 0
    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id
        actual_data_log = tss[i]
        safe_book_id = book_id.replace("/", "_").replace("\\", "_")
        if all_title_predict:
            base_save_path = f"tft/All_predict/forecast_{safe_book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=base_save_path,
                title=f"{book_id}",
                use_log_scale=use_log_scale,
                prediction_length=prediction_length,
                context_length=context_length,
                quantiles=quantiles
            )
            saved_count += 1
        if book_id in decile_books:
            info = decile_books[book_id]
            label = info["label"]
            total_sales = info["total_sales"]
            print(f"Plotting Representative: {label} - {book_id}")
            safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
            rep_save_path = f"tft/forecast_{safe_label}_{book_id}.png"
            save_forecast(
                forecast=forecast,
                actual_data_log=actual_data_log,
                save_path=rep_save_path,
                title=f"[{label}] {book_id}\nTotal Sales: {total_sales:,.0f} books",
                use_log_scale=use_log_scale,
                prediction_length=prediction_length,
                context_length=context_length,
                quantiles=quantiles
            )
            if not all_title_predict:
                saved_count += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(forecasts)}")
    print(f"All {len(forecasts)} prediction plots saved to tft/All_predict/" if all_title_predict
          else f"{saved_count} representative prediction plots saved to tft/")


def calculate_all_features(
    df,
    use_log_scale,
    use_hazard_features,
    use_sales_relative_features,
    use_price_relative_features,
    context_length,
    recent_period=14
):
    print("\n=== Feature Calculation ===")
    print(f"Log Scale: {'ENABLED' if use_log_scale else 'DISABLED'}")
    print(f"Hazard Features: {'ENABLED' if use_hazard_features else 'DISABLED'}")
    print(f"Sales Relative Features: {'ENABLED' if use_sales_relative_features else 'DISABLED'}")
    print(f"Price Relative Features: {'ENABLED' if use_price_relative_features else 'DISABLED'}")

    df = scaling_data(df, use_log_scale)

    if use_hazard_features:
        df = calculate_hazard_features(df, context_length=context_length, recent_period=recent_period)
        df = calculate_regime_features(df)

    if use_sales_relative_features:
        df = calculate_temporal_relative_features(
            df,
            target_col='POS販売冊数',
            category_cols=['大分類', '中分類', '小分類'],
            context_length=context_length
        )

    if use_price_relative_features:
        df = calculate_static_relative_features(
            df,
            target_col='log_本体価格',
            category_cols=['大分類', '中分類', '小分類']
        )

    return df


def apply_leakage_shift(df, use_sales_relative_features, use_hazard_features):
    print("\n=== Applying Lag-1 Shift to Prevent Data Leakage ===")
    print("Shifting features that use current timestep's sales to next timestep")
    print("This ensures: Row t = Target(t) + Features(t-1)")
    print("\nNote: feat_hawkes, feat_adi, feat_cv2, feat_days_since, score_sparse, score_burst")
    print("are NOT shifted because they are calculated from past events only (no current sales leakage)")

    leakage_cols = []

    if use_sales_relative_features:
        leakage_cols.extend([
            '大分類_POS販売冊数_relative',
            '大分類_POS販売冊数_z_score',
            '中分類_POS販売冊数_relative',
            '中分類_POS販売冊数_z_score',
            '小分類_POS販売冊数_relative',
            '小分類_POS販売冊数_z_score',
        ])

    if use_hazard_features:
        leakage_cols.extend([
            'z_score',
            'is_spike_1.5',
            'is_spike_2.0',
            'is_spike_2.5',
            'is_spike_3.0',
            'days_since_spike_1.5',
            'days_since_spike_2.0',
            'days_since_spike_2.5',
            'days_since_spike_3.0',
            'feat_periodicity_score',
            'feat_detected_period',
            'score_periodic'
        ])

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)

    for col in leakage_cols:
        if col in df.columns:
            df[col] = df.groupby('書名', observed=False)[col].shift(1).fillna(0).astype(np.float32)
            print(f"  ✓ Shifted: {col}")
        else:
            print(f"  ✗ Not found: {col}")

    print(f"Total shifted features: {len([c for c in leakage_cols if c in df.columns])}")
    print("First row of each book now has lagged features = 0 (no history)")

    return df


def build_feature_lists(
    use_calendar_features,
    use_sales_relative_features,
    use_price_relative_features,
    use_hazard_features,
    calendar_feature_cols,
    sales_relative_cols,
    price_relative_cols,
    hazard_feature_cols
):
    print("\n=== Building Feature Lists ===")

    time_feature_cols = []
    if use_calendar_features:
        time_feature_cols.extend(calendar_feature_cols)
        print(f"Added calendar features: {len(calendar_feature_cols)} features")
    if use_price_relative_features:
        time_feature_cols.extend(price_relative_cols)
        print(f"Added price relative features: {len(price_relative_cols)} features")

    past_dynamic_cols = []
    if use_sales_relative_features:
        past_dynamic_cols.extend(sales_relative_cols)
        print(f"Added sales relative features: {len(sales_relative_cols)} features")
    if use_hazard_features:
        past_dynamic_cols.extend(hazard_feature_cols)
        print(f"Added hazard features: {len(hazard_feature_cols)} features")

    print(f"\nTotal time features (known covariates): {len(time_feature_cols)}")
    print(f"Total past dynamic features (observed covariates): {len(past_dynamic_cols)}")

    return time_feature_cols, past_dynamic_cols


def evaluate_predictions(forecasts, tss, use_log_scale, prediction_length):
    print("\n=== Calculating Evaluation Metrics ===")

    total_metrics = {
        "mae": 0.0,
        "mse": 0.0,
        "wql_10": 0.0,
        "wql_50": 0.0,
        "wql_90": 0.0,
        "coverage_80": 0.0,
        "total_sales_sum": 0.0,
        "count_points": 0
    }

    item_results = []

    for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
        book_name = forecast.item_id

        start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
        future_end_date = start_timestamp + pd.Timedelta(days=prediction_length - 1)

        if hasattr(ts.index, 'to_timestamp'):
            ts_indexed = ts.copy()
            ts_indexed.index = ts.index.to_timestamp()
        else:
            ts_indexed = ts

        target_series = ts_indexed[start_timestamp:future_end_date]
        target = target_series.values.flatten()

        if len(target) != prediction_length:
            print(f"Warning: {book_name} has {len(target)} points instead of {prediction_length}, skipping")
            continue

        q10 = forecast.quantile(0.1)
        q50 = forecast.quantile(0.5)
        q90 = forecast.quantile(0.9)

        if use_log_scale:
            target = np.expm1(target)
            q10 = np.expm1(q10)
            q50 = np.expm1(q50)
            q90 = np.expm1(q90)

        ae = np.abs(target - q50)

        se = (target - q50) ** 2

        def pinball_loss(y_true, y_pred, quantile):
            diff = y_true - y_pred
            return np.maximum(quantile * diff, (quantile - 1) * diff)

        loss_10 = pinball_loss(target, q10, 0.1)
        loss_50 = pinball_loss(target, q50, 0.5)
        loss_90 = pinball_loss(target, q90, 0.9)

        in_interval = ((target >= q10) & (target <= q90)).astype(float)

        pred_len = len(target)
        item_sales_sum = np.sum(np.abs(target))

        total_metrics["mae"] += np.sum(ae)
        total_metrics["mse"] += np.sum(se)
        total_metrics["wql_10"] += np.sum(loss_10)
        total_metrics["wql_50"] += np.sum(loss_50)
        total_metrics["wql_90"] += np.sum(loss_90)
        total_metrics["coverage_80"] += np.sum(in_interval)
        total_metrics["total_sales_sum"] += item_sales_sum
        total_metrics["count_points"] += pred_len

        item_mae = np.mean(ae)
        item_rmse = np.sqrt(np.mean(se))
        item_wql_10 = 2 * np.sum(loss_10)
        item_wql_50 = 2 * np.sum(loss_50)
        item_wql_90 = 2 * np.sum(loss_90)
        item_wql_mean = (item_wql_10 + item_wql_50 + item_wql_90) / 3
        item_coverage = np.mean(in_interval)

        item_results.append({
            "book_name": book_name,
            "MAE": item_mae,
            "RMSE": item_rmse,
            "wQL_0.1": item_wql_10,
            "wQL_0.5": item_wql_50,
            "wQL_0.9": item_wql_90,
            "wQL_Mean": item_wql_mean,
            "Coverage_80%": item_coverage,
            "Total_Sales": np.sum(target)
        })

    final_metrics = {}

    final_metrics["MAE"] = total_metrics["mae"] / total_metrics["count_points"]
    final_metrics["RMSE"] = np.sqrt(total_metrics["mse"] / total_metrics["count_points"])

    denom = total_metrics["total_sales_sum"] + 1e-9
    final_metrics["wQL_0.1"] = 2 * total_metrics["wql_10"] / denom
    final_metrics["wQL_0.5"] = 2 * total_metrics["wql_50"] / denom
    final_metrics["wQL_0.9"] = 2 * total_metrics["wql_90"] / denom
    final_metrics["wQL_Mean"] = (final_metrics["wQL_0.1"] + final_metrics["wQL_0.5"] + final_metrics["wQL_0.9"]) / 3

    final_metrics["Coverage_80%"] = total_metrics["coverage_80"] / total_metrics["count_points"]

    print("-" * 40)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 40)

    eval_df = pd.DataFrame(item_results)
    eval_csv_path = os.path.join("tft", "evaluation_all.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nBook-level evaluation saved to: {eval_csv_path}")
