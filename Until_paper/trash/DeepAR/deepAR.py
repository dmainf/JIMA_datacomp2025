import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.prepro import *
from lib.fueture_eng import *

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.distributions import NormalOutput
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count

# 基準日を設定（累積日数=1が2024-01-01に対応）
base_date = pd.Timestamp("2024-01-01")

def process_group(args):
    _, group = args  # keyは使用しないのでアンダースコアで受け取る
    group_sorted = group.sort_values('累積日数')

    # 静的特徴量（各時系列で1つの値）
    static_cat = [
        int(group_sorted['出版社'].iloc[0]),
        int(group_sorted['著者名'].iloc[0]),
        int(group_sorted['大分類'].iloc[0]),
        int(group_sorted['中分類'].iloc[0]),
        int(group_sorted['小分類'].iloc[0])
    ]
    static_real = [float(group_sorted['本体価格'].iloc[0])]

    # 累積日数から実際の日付を計算
    start_day = int(group_sorted['累積日数'].iloc[0])
    start_date = base_date + pd.Timedelta(days=start_day - 1)

    return {
        "start": pd.Period(start_date, freq="D"),
        "target": group_sorted['売上高'].values,
        "feat_static_cat": static_cat,
        "feat_static_real": static_real
    }

if __name__ == '__main__':
    print("loading data...")
    df = pd.read_parquet('../data/df.parquet')
    print(f"Data loaded! Shape: {df.shape}")
    df = remove_volume_number(df)
    df['売上高'] = df['本体価格'] * df['POS販売冊数']
    aggregation_rules = {
        '売上高': 'sum',       # 売上高は合計する
        '月': 'first',         # 累積日数に紐づくため、最初の値を取得
        '日': 'first',         # 累積日数に紐づくため、最初の値を取得
        '出版社': 'first',     # 書名に紐づくため、最初の値を取得
        '著者名': 'first',     # 書名に紐づくため、最初の値を取得
        '大分類': 'first',     # 書名に紐づくため、最初の値を取得
        '中分類': 'first',     # 書名に紐づくため、最初の値を取得
        '小分類': 'first',     # 書名に紐づくため、最初の値を取得
        '本体価格': 'first'    # 書名に紐づくため、最初の値を取得
    }
    group_keys = ['書店コード', '書名', '累積日数']
    dataset = df.groupby(group_keys).agg(aggregation_rules).reset_index()

    label_columns = [
        '出版社',
        '著者名',
        '大分類',
        '中分類',
        '小分類'
    ]
    dataset = label_enc(dataset, label_columns)

    print("=== データの形状 ===")
    print(dataset.shape)
    print("\n=== データ型 ===")
    print(dataset.dtypes)

    # カーディナリティ（各カテゴリカル特徴量のユニーク数）を計算
    print("\n=== カーディナリティ ===")
    for col in label_columns:
        print(f"{col}: {dataset[col].nunique()}")

    print("saving dataflame to dataset.parquet...")
    dataset.to_parquet('dataset.parquet')
    print("complete!")

    PREDICTION_LENGTH = 7
    CONTEXT_LENGTH = 30
    NUM_EPOCHS = 20  # 高速化: 10 → 5エポックに削減
    BATCH_SIZE = 128  # GPU使用時は大きなバッチサイズで高速化（M4 ProのGPUメモリを活用）

    # カテゴリカル特徴量のカーディナリティを指定
    # 順序: 出版社, 著者名, 大分類, 中分類, 小分類
    CARDINALITY = [1389, 48233, 40, 304, 2258]
    NUM_FEAT_STATIC_CAT = len(CARDINALITY)  # 5個のカテゴリカル特徴量
    NUM_FEAT_STATIC_REAL = 1  # 1個の静的数値特徴量（本体価格）

    estimator = DeepAREstimator(
        prediction_length=PREDICTION_LENGTH,
        freq='D',
        context_length=CONTEXT_LENGTH,
        num_feat_static_cat=NUM_FEAT_STATIC_CAT,  # カテゴリカル特徴量の数
        cardinality=CARDINALITY,  # 各カテゴリカル特徴量のユニーク数
        num_feat_static_real=NUM_FEAT_STATIC_REAL,  # 静的数値特徴量の数
        num_layers=2,
        hidden_size=128,
        dropout_rate=0.1,
        lr=1e-3,
        batch_size=BATCH_SIZE,
        distr_output=NormalOutput(),
        trainer_kwargs={
            "max_epochs": NUM_EPOCHS,
            "accelerator": "mps",  # M4 MacBook ProのGPUを使用
            "devices": 1,
            "gradient_clip_val": 10.0,  # 勾配クリッピングで安定化
        }
    )
    # データセットを訓練用とテスト用に分割
    # 各時系列の最後のPREDICTION_LENGTH日分をテストデータとして確保
    print("\n===== Creating Train/Test Split =====")

    # 高速化: groupby().apply()を使わずにベクトル化された方法で分割
    # 各グループの累積日数の範囲を計算
    group_stats = dataset.groupby(['書店コード', '書名'])['累積日数'].agg(['min', 'max', 'count'])
    # 十分な長さ（CONTEXT_LENGTH + PREDICTION_LENGTH以上）の時系列のみを使用
    valid_groups = group_stats[group_stats['count'] >= CONTEXT_LENGTH + PREDICTION_LENGTH].index
    print(f"Total time series: {len(group_stats)}, Valid time series (>={CONTEXT_LENGTH + PREDICTION_LENGTH} points): {len(valid_groups)}")

    # 有効な時系列のみをフィルタ
    dataset_filtered = dataset.set_index(['書店コード', '書名']).loc[valid_groups].reset_index()

    # 各グループの累積日数の最大値を計算
    group_max = dataset_filtered.groupby(['書店コード', '書名'])['累積日数'].transform('max')

    # 訓練データ: 最後のPREDICTION_LENGTH日を除く
    train_data = dataset_filtered[dataset_filtered['累積日数'] <= group_max - PREDICTION_LENGTH].copy()
    test_data = dataset_filtered.copy()


    # ListDataset形式に変換（並列処理で高速化）
    print("Converting to ListDataset format (using parallel processing)...")

    # 並列処理で訓練データを変換
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing...")

    with Pool(num_cores) as pool:
        train_groups = list(train_data.groupby(['書店コード', '書名'], sort=False))
        train_list = pool.map(process_group, train_groups)

        test_groups = list(test_data.groupby(['書店コード', '書名'], sort=False))
        test_list = pool.map(process_group, test_groups)

    print(f"Number of time series: {len(train_list)}")

    train_dataset = ListDataset(train_list, freq="D")
    test_dataset = ListDataset(test_list, freq="D")

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    print("\n===== Training DeepAR Model =====")
    predictor = estimator.train(train_dataset)
    print("\n===== Making Predictions on Test Set =====")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=50  # 高速化: 100 → 50サンプルに削減（予測の不確実性推定の精度はやや下がるが高速）
    )
    print("\n===== Evaluating =====")
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # 評価指標を計算
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    print("\n===== Evaluation Metrics =====")
    for key, value in agg_metrics.items():
        print(f"{key}: {value:.4f}")

    # メトリクスをCSVで保存
    print("\n===== Saving Metrics =====")
    agg_metrics_df = pd.DataFrame([agg_metrics])
    agg_metrics_df.to_csv('deepar_agg_metrics.csv', index=False)
    item_metrics.to_csv('deepar_item_metrics.csv', index=True)
    print("Metrics saved to CSV files")

    # 性能可視化グラフを作成（簡潔版）
    print("\n===== Creating Performance Visualizations =====")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 主要メトリクスのバープロット
    main_metrics = {k: v for k, v in agg_metrics.items() if k in ['MASE', 'RMSE', 'MAPE', 'sMAPE']}
    ax.bar(main_metrics.keys(), main_metrics.values(), color='steelblue')
    ax.set_title('DeepAR Evaluation Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(main_metrics.items()):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('deepar_performance_metrics.png', dpi=150, bbox_inches='tight')
    print("Performance metrics visualization saved as 'deepar_performance_metrics.png'")

    # 予測サンプルのプロット（3サンプルのみ）
    print("\n===== Plotting Sample Predictions =====")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, (forecast, ts) in enumerate(zip(forecasts[:3], tss[:3])):
        ax = axes2[i]
        ts[-60:].plot(ax=ax, label='Actual', color='black', linewidth=1.5)
        forecast.plot(ax=ax, show_label=True, color='blue')
        ax.legend()
        ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Sales')

    plt.tight_layout()
    plt.savefig('deepar_sample_predictions.png', dpi=150, bbox_inches='tight')
    print("Sample predictions saved as 'deepar_sample_predictions.png'")


    print("\n===== Saving Model =====")
    model_path = Path("deepar_model")
    model_path.mkdir(exist_ok=True)
    predictor.serialize(model_path)
    print("Model saved to 'deepar_model/' directory")
    print("\n===== Done! =====")