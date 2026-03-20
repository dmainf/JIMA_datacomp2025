import pandas as pd
import warnings
import logging
from function import *
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("gluonts").setLevel(logging.ERROR)

PREDICTION_LENGTH = 64
CONTEXT_LENGTH = 180
EPOC = 125

ALL_TITLE_PREDICT = True
QUANTILES = [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]
USE_LOG_SCALE = False
DISTRIBUTION = "quantile"   #整数じゃないと動かない
#DISTRIBUTION = "negative_binomial"

USE_CALENDAR_FEATURES = False#True
USE_SALES_RELATIVE_FEATURES = False#True
USE_PRICE_RELATIVE_FEATURES = False#True
USE_HAZARD_FEATURES = False#True

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

hazard_feature_cols = [
    'momentum',
    'volatility',
    'z_score',
    'days_since_spike_1.5',
    'days_since_spike_2.0',
    'days_since_spike_2.5',
    'days_since_spike_3.0',
    'is_spike_1.5',
    'is_spike_2.0',
    'is_spike_2.5',
    'is_spike_3.0',
    'feat_adi',
    'feat_cv2',
    'feat_hawkes',
    'feat_days_since',
    'feat_periodicity_score',
    'feat_detected_period',
    'score_sparse',
    'score_periodic',
    'score_burst',
]

def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')
    print("Complete!")

    decile_books = extract_decile_books(df)

    df = calculate_all_features(
        df,
        use_log_scale=USE_LOG_SCALE,
        use_hazard_features=USE_HAZARD_FEATURES,
        use_sales_relative_features=USE_SALES_RELATIVE_FEATURES,
        use_price_relative_features=USE_PRICE_RELATIVE_FEATURES,
        context_length=CONTEXT_LENGTH,
        recent_period=14
    )

    df = apply_leakage_shift(
        df,
        use_sales_relative_features=USE_SALES_RELATIVE_FEATURES,
        use_hazard_features=USE_HAZARD_FEATURES
    )

    time_feature_cols, past_dynamic_cols = build_feature_lists(
        use_calendar_features=USE_CALENDAR_FEATURES,
        use_sales_relative_features=USE_SALES_RELATIVE_FEATURES,
        use_price_relative_features=USE_PRICE_RELATIVE_FEATURES,
        use_hazard_features=USE_HAZARD_FEATURES,
        calendar_feature_cols=calendar_feature_cols,
        sales_relative_cols=sales_relative_cols,
        price_relative_cols=price_relative_cols,
        hazard_feature_cols=hazard_feature_cols
    )

    verify_time_cols = [col for cols, flag in [(calendar_feature_cols, USE_CALENDAR_FEATURES),
                                                (sales_relative_cols, USE_SALES_RELATIVE_FEATURES),
                                                (price_relative_cols, USE_PRICE_RELATIVE_FEATURES)]
                        for col in cols if flag]
    verify_hazard_cols = hazard_feature_cols if USE_HAZARD_FEATURES else []
    verify_all_features(df, static_cols, [], verify_time_cols, verify_hazard_cols)
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
        QUANTILES,
        DISTRIBUTION
    )
    process_logs()
    forecasts, tss = evaluate_prediction(predictor, full_dataset, USE_LOG_SCALE)
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
    evaluate_predictions(forecasts, tss, USE_LOG_SCALE, PREDICTION_LENGTH)
    print("\nAll plots saved. Process Complete.")
if __name__ == "__main__":
    main()
