import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
import matplotlib.pyplot as plt

print("===== DeepAR予測スクリプト =====\n")

# 学習済みモデルを読み込み
print("Loading trained model...")
predictor = Predictor.deserialize(Path("deepar_model"))
print("Model loaded successfully!\n")

# 予測用データを読み込み
print("Loading dataset...")
dataset = pd.read_parquet('dataset.parquet')
print(f"Dataset loaded! Shape: {dataset.shape}\n")

# 予測対象の設定
PREDICTION_LENGTH = 7  # 7日先まで予測
FORECAST_START_DATE = pd.Timestamp("2024-01-01")  # 予測開始日の基準

# 各時系列の最新データを使って予測
print("===== Creating prediction dataset =====")

# 書店コード × 書名 でグループ化
grouped = dataset.groupby(['書店コード', '書名'])

# 予測用データリストを作成
prediction_list = []
series_info = []  # 各時系列の情報を保存

for (bookstore_code, book_name), group in grouped:
    group_sorted = group.sort_values('累積日数')

    # 静的特徴量
    static_cat = [
        int(group_sorted['出版社'].iloc[0]),
        int(group_sorted['著者名'].iloc[0]),
        int(group_sorted['大分類'].iloc[0]),
        int(group_sorted['中分類'].iloc[0]),
        int(group_sorted['小分類'].iloc[0])
    ]
    static_real = [float(group_sorted['本体価格'].iloc[0])]

    # 開始日の計算
    start_day = int(group_sorted['累積日数'].iloc[0])
    start_date = FORECAST_START_DATE + pd.Timedelta(days=start_day - 1)

    prediction_list.append({
        "start": pd.Period(start_date, freq="D"),
        "target": group_sorted['売上高'].values,
        "feat_static_cat": static_cat,
        "feat_static_real": static_real
    })

    # 情報を保存
    series_info.append({
        "書店コード": bookstore_code,
        "書名": book_name,
        "データ数": len(group_sorted),
        "最終日": group_sorted['累積日数'].max()
    })

print(f"Total time series to predict: {len(prediction_list)}\n")

# ListDataset形式に変換
prediction_dataset = ListDataset(prediction_list, freq="D")

# 予測実行
print("===== Making Predictions =====")
forecasts = list(predictor.predict(prediction_dataset))
print(f"Predictions completed for {len(forecasts)} time series!\n")

# 予測結果を保存
print("===== Saving Predictions =====")

# 予測結果をDataFrameに変換
predictions_data = []
for i, (forecast, info) in enumerate(zip(forecasts, series_info)):
    for day_idx in range(PREDICTION_LENGTH):
        predictions_data.append({
            "書店コード": info["書店コード"],
            "書名": info["書名"],
            "予測日数": info["最終日"] + day_idx + 1,
            "予測売上高_mean": forecast.mean[day_idx],
            "予測売上高_median": forecast.quantile(0.5)[day_idx],
            "予測売上高_lower": forecast.quantile(0.1)[day_idx],
            "予測売上高_upper": forecast.quantile(0.9)[day_idx]
        })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv('deepar_predictions.csv', index=False)
print("Predictions saved to 'deepar_predictions.csv'\n")

# サマリー統計
print("===== Prediction Summary =====")
print(f"Total predictions: {len(predictions_df)}")
print(f"Unique series: {predictions_df['書名'].nunique()}")
print(f"\n予測売上高の統計:")
print(predictions_df['予測売上高_mean'].describe())

# 予測結果の可視化（サンプル5件）
print("\n===== Visualizing Sample Predictions =====")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# 最初の6つの時系列について可視化
for i in range(min(6, len(forecasts))):
    ax = axes[i]

    # 実績データ（最新30日分）
    actual_data = prediction_list[i]['target'][-30:]
    actual_dates = pd.date_range(
        end=pd.Timestamp(prediction_list[i]['start'].to_timestamp()) + pd.Timedelta(days=len(prediction_list[i]['target'])-1),
        periods=len(actual_data),
        freq='D'
    )

    # 予測データ
    forecast = forecasts[i]
    forecast_dates = pd.date_range(
        start=actual_dates[-1] + pd.Timedelta(days=1),
        periods=PREDICTION_LENGTH,
        freq='D'
    )

    # プロット
    ax.plot(actual_dates, actual_data, 'o-', color='black', label='Actual', linewidth=2)
    ax.plot(forecast_dates, forecast.mean, 's-', color='blue', label='Forecast (mean)', linewidth=2)
    ax.fill_between(
        forecast_dates,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        alpha=0.3,
        color='blue',
        label='80% interval'
    )

    ax.set_title(f"{series_info[i]['書名'][:20]}...", fontsize=10, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('deepar_forecast_samples.png', dpi=150, bbox_inches='tight')
print("Sample forecasts saved as 'deepar_forecast_samples.png'")

print("\n===== Done! =====")
print("出力ファイル:")
print("  - deepar_predictions.csv: 全予測結果")
print("  - deepar_forecast_samples.png: サンプル予測のグラフ")
