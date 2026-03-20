import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import pickle
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset
from common import load_encoders

matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'YuGothic', 'Hiragino Maru Gothic Pro']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

print("Loading model and artifacts...")

try:
    predictor = PyTorchPredictor.deserialize(Path("model"))
    print("  ✓ TFT model loaded")
except FileNotFoundError:
    print("ERROR: TFT model (model/) not found. Run o_transformer.py first.")
    exit()

try:
    encoders = load_encoders('encoders.pkl')
    print("  ✓ Encoders loaded")
except FileNotFoundError:
    print("ERROR: encoders.pkl not found. Run o_transformer.py with MAKE_DATASET=True first.")
    exit()

try:
    dataset = pd.read_parquet('dataset.parquet')
    print("  ✓ Dataset loaded")
except FileNotFoundError:
    print("ERROR: dataset.parquet not found. Run o_transformer.py with MAKE_DATASET=True first.")
    exit()

print("All models loaded!\n")

from o_feature_eng import (
    STATIC_CAT_COLS,
    STATIC_REAL_COLS,
    TEMPORAL_COLS,
    TIME_COLS
)

from o_transformer import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH
)

print(f"Context length: {CONTEXT_LENGTH}, Prediction length: {PREDICTION_LENGTH}\n")
print(f"Dataset shape: {dataset.shape}")
print(f"Dataset columns: {list(dataset.columns)}\n")

unique_stores = dataset['書店コード'].unique()
print(f"Number of unique stores (encoded): {len(unique_stores)}")
print(f"Unique stores (first 10): {unique_stores[:10]}\n")

output_dir = Path("score/predictions_o_transformer")
output_dir.mkdir(exist_ok=True, parents=True)

for encoded_store in unique_stores[:35]:
    try:
        decoded_store = encoders['書店コード'].inverse_transform([encoded_store])[0]
    except:
        decoded_store = f"Store_{encoded_store}"

    print(f"\nProcessing Store {decoded_store} (encoded: {encoded_store})...")

    store_data = dataset[dataset['書店コード'] == encoded_store]

    if len(store_data) == 0:
        print(f"  Skipped: No data for store {decoded_store}")
        continue

    sales_by_book = store_data.groupby('書名')['POS販売冊数'].agg(['sum', 'count'])
    sales_by_book = sales_by_book[sales_by_book['count'] >= CONTEXT_LENGTH + PREDICTION_LENGTH]
    sales_by_book = sales_by_book.sort_values('sum', ascending=False)

    if len(sales_by_book) < 1:
        print(f"  Skipped: Insufficient books")
        continue

    top_book_codes = sales_by_book.head(5).index.tolist()

    top_books_display = []
    for encoded_book in top_book_codes:
        try:
            decoded_book = encoders['書名'].inverse_transform([encoded_book])[0]
            top_books_display.append((encoded_book, decoded_book))
        except:
            continue

    if len(top_books_display) == 0:
        print(f"  Skipped: No valid books")
        continue

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (encoded_book, book_name_display) in enumerate(top_books_display):
        if idx >= 5:
            break

        ax = axes[idx]

        book_features = store_data[store_data['書名'] == encoded_book].sort_values('日付').reset_index(drop=True)

        if len(book_features) < CONTEXT_LENGTH + PREDICTION_LENGTH:
            print(f"  Skipped {book_name_display[:30]}: insufficient data length")
            continue

        dates = pd.to_datetime(book_features['日付'].values)
        sales = book_features['POS販売冊数'].values

        min_split = CONTEXT_LENGTH
        max_split = len(sales) - PREDICTION_LENGTH
        if max_split <= min_split:
            print(f"  Skipped {book_name_display[:30]}: insufficient length for slicing")
            continue

        num_slices = min(4, max_split - min_split)
        if num_slices < 2:
            num_slices = 1
        slice_points = np.linspace(min_split, max_split, num_slices, dtype=int)

        ax.plot(dates, sales, color='#95A5A6', linewidth=0.8,
                label='全データ', marker='o', markersize=1.5, alpha=0.4)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        all_maes = []
        all_mapes = []

        try:
            for slice_idx, split_point in enumerate(slice_points):
                split_point = int(split_point)
                test_start = split_point
                test_end = min(split_point + PREDICTION_LENGTH, len(sales))

                if test_end > len(book_features):
                    continue

                test_dates = dates[test_start:test_end]
                test_sales = sales[test_start:test_end]

                context_start = max(0, split_point - CONTEXT_LENGTH)
                context_dates = dates[context_start:split_point]
                context_sales = sales[context_start:split_point]
                color = colors[slice_idx % len(colors)]

                if len(context_dates) > 0:
                    ax.plot(context_dates, context_sales, color=color, linewidth=1.0,
                           label=f'Context{slice_idx+1}', alpha=0.5, linestyle='-')

                static_cat = [int(book_features[col].iloc[0]) for col in STATIC_CAT_COLS if col in book_features.columns]
                static_real = [float(book_features[col].iloc[0]) for col in STATIC_REAL_COLS if col in book_features.columns]

                for time_col in [c for c in TIME_COLS if c in book_features.columns]:
                    time_val = book_features[time_col].iloc[0]
                    static_real.append(float(time_val.hour + time_val.minute / 60.0))

                roll_cols = sorted([c for c in book_features.columns if 'POS販売冊数_roll' in c],
                                  key=lambda x: int(x.split('roll_')[1].split('mean')[-1].split('std')[-1].split('max')[-1]))

                entry_tft = {
                    "start": pd.Period(book_features['日付'].iloc[0], freq="D"),
                    "target": sales[:split_point].astype(np.float32),
                    "feat_static_cat": static_cat,
                    "feat_static_real": static_real,
                    "feat_dynamic_real": book_features[TEMPORAL_COLS].values[:test_end].T.astype(np.float32),
                    "past_feat_dynamic_real": book_features[roll_cols].values[:split_point].T.astype(np.float32) if roll_cols else np.array([]).astype(np.float32)
                }
                tft_dataset = ListDataset([entry_tft], freq="D")

                forecast_it = predictor.predict(tft_dataset)
                forecast = list(forecast_it)[0]
                prediction = forecast.median[:len(test_sales)]

                ax.plot(test_dates, test_sales, color='black', linewidth=1.0,
                       label=f'実測{slice_idx+1}', marker='o', markersize=4, alpha=0.8)

                ax.plot(test_dates, prediction, color=color, linewidth=1.0,
                       label=f'予測{slice_idx+1}', marker='s', markersize=3, linestyle='--', alpha=0.9)

                if hasattr(forecast, 'quantile'):
                    lower = forecast.quantile(0.1)[:len(test_sales)]
                    upper = forecast.quantile(0.9)[:len(test_sales)]
                    ax.fill_between(test_dates, lower, upper,
                                   color=color, alpha=0.1)

                mae = np.mean(np.abs(test_sales - prediction))
                mape = np.mean(np.abs((test_sales - prediction) / (test_sales + 1e-6))) * 100
                all_maes.append(mae)
                all_mapes.append(mape)

            if len(all_maes) > 0:
                avg_mae = np.mean(all_maes)
                avg_mape = np.mean(all_mapes)
                ax.text(0.02, 0.98, f'平均 MAE: {avg_mae:.2f}冊\n平均 MAPE: {avg_mape:.1f}%\nスライス数: {num_slices}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        except Exception as e:
            import traceback
            print(f"  Error: {book_name_display[:30]}: {str(e)[:80]}")
            if idx == 0:
                traceback.print_exc()

        ax.set_title(f'{book_name_display[:50]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('日付', fontsize=8)
        ax.set_ylabel('POS販売冊数', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))

    for idx in range(len(top_books_display), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'書店 {decoded_store} - TFT予測 Top 5 (各{PREDICTION_LENGTH}日間)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = output_dir / f'store_{decoded_store}_predictions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

print(f"\nCompleted! Check {output_dir}/")
