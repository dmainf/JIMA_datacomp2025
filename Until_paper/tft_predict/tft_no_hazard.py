import pandas as pd
import warnings
import logging
from function import *

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("gluonts").setLevel(logging.ERROR)

PREDICTION_LENGTH = 14
CONTEXT_LENGTH = 180
EPOC = 100

ALL_TITLE_PREDICT = False
QUANTILES = [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]
USE_LOG_SCALE = True

static_cols = ['出版社', '著者名', '大分類', '中分類', '小分類', '書名_base']

calendar_feature_cols = [
    'month_sin',
    'month_cos',
    'dayofweek_sin',
    'dayofweek_cos'
]

sales_relative_cols = [
    '大分類_POS販売冊数_relative',
    '大分類_POS販売冊数_z_score',
    '中分類_POS販売冊数_relative',
    '中分類_POS販売冊数_z_score',
    '小分類_POS販売冊数_relative',
    '小分類_POS販売冊数_z_score',
]

price_relative_cols = [
    '大分類_log_本体価格_relative',
    '大分類_log_本体価格_z_score',
    '中分類_log_本体価格_relative',
    '中分類_log_本体価格_z_score',
    '小分類_log_本体価格_relative',
    '小分類_log_本体価格_z_score',
]

time_feature_cols = calendar_feature_cols + price_relative_cols
past_dynamic_cols = sales_relative_cols


def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    print("Complete!")

    decile_books = extract_decile_books(df)
    df = scaling_data(df, USE_LOG_SCALE)
    df = calculate_temporal_relative_features(
        df,
        target_col='POS販売冊数',
        category_cols=['大分類', '中分類', '小分類'],
        context_length=CONTEXT_LENGTH
    )
    df = calculate_static_relative_features(
        df,
        target_col='log_本体価格',
        category_cols=['大分類', '中分類', '小分類']
    )

    print("\n=== Applying Lag-1 Shift to Prevent Data Leakage ===")
    print("Shifting features that use current timestep's sales to next timestep")
    print("This ensures: Row t = Target(t) + Features(t-1)")
    print("\nNote: This version uses NO hazard features, only sales relative features")
    print("All sales_relative_cols must be shifted as they include current timestep sales")

    leakage_cols = [
        '大分類_POS販売冊数_relative',
        '大分類_POS販売冊数_z_score',
        '中分類_POS販売冊数_relative',
        '中分類_POS販売冊数_z_score',
        '小分類_POS販売冊数_relative',
        '小分類_POS販売冊数_z_score',
    ]

    df = df.sort_values(['書名', '日付']).reset_index(drop=True)

    for col in leakage_cols:
        if col in df.columns:
            df[col] = df.groupby('書名', observed=False)[col].shift(1).fillna(0).astype(np.float32)
            print(f"  ✓ Shifted: {col}")
        else:
            print(f"  ✗ Not found: {col}")

    print(f"Total shifted features: {len([c for c in leakage_cols if c in df.columns])}")
    print("First row of each book now has lagged features = 0 (no history)")

    verify_all_features(df, static_cols, calendar_feature_cols, sales_relative_cols + price_relative_cols, [])
    save_training_data(df, 'train.parquet')
    df.describe().to_csv("df_describe.csv")
    print("=== Preparing Dataset and Training Model ===")
    full_dataset, train_dataset, cardinality = prepare_dataset(df, static_cols, time_feature_cols, past_dynamic_cols, PREDICTION_LENGTH)
    predictor = train_model(
        train_dataset,
        cardinality,
        PREDICTION_LENGTH,
        CONTEXT_LENGTH,
        EPOC,
        QUANTILES
    )
    process_logs()
    forecasts, tss = evaluate_prediction(predictor, full_dataset)
    process_predictions(
        forecasts,
        tss,
        decile_books,
        df,
        ALL_TITLE_PREDICT,
        USE_LOG_SCALE,
        PREDICTION_LENGTH,
        CONTEXT_LENGTH,
        QUANTILES
    )
    print("\nAll plots saved. Process Complete.")
if __name__ == "__main__":
    main()
