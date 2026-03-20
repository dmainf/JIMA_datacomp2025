import pandas as pd
import numpy as np
from pathlib import Path
from old_feature_eng import *
from common import *

CONTEXT_LENGTH = 56
PREDICTION_LENGTH = 14
MAX_WINDOW = 7
REGION_EXTEND = 10
AUGMENTATION_FACTOR = 10

NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 5e-4       # 1e-3 から減らす
HIDDEN_DIM = 256         # 128 から増やす
VARIABLE_DIM = 64          # 32 から増やす
NUM_HEADS = 4
DROPOUT_RATE = 0.25      # 0.15 から増やす
PATIENCE = 4

USE_PRIORITIZED_REPLAY = False
REPLAY_TOP_RATIO = 0.5
REPLAY_MULTIPLIER = 2
REPLAY_EPOCHS = 20

MAKE_DATASET = False

if __name__ == '__main__':
    if MAKE_DATASET:
        print("\n=== Loading All Store Data ===")
        df = load_all_stores(data='by_store', exclude_stores=[26, 27])
        print(f"Total records: {len(df)}")

        print("\n=== Preparing Dataset ===")
        dataset, encoders = prepare_dataset(
            df,
            context_length = CONTEXT_LENGTH,
            prediction_length = PREDICTION_LENGTH,
            max_window = MAX_WINDOW
        )
        print(f"\nDataset Shape: {dataset.shape}")
        dataset.to_parquet('dataset.parquet')
        print("Saved dataset to dataset.parquet")
        save_encoders(encoders, 'encoders.pkl')
        print("Saved encoders to encoders.pkl")
    else:
        print("\n=== Loading Dataset ===")
        dataset = pd.read_parquet('dataset.parquet')
        print(f"Dataset Shape: {dataset.shape}")

    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
    from gluonts.evaluation import make_evaluation_predictions, Evaluator
    from gluonts.model.forecast import SampleForecast, QuantileForecast
    from sklearn.model_selection import train_test_split
    import torch

    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device_type}")

    print("\n=== Creating Slice Dataset ===")
    dict_list = create_slices(
        dataset,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        region_extend=REGION_EXTEND,
        max_window=MAX_WINDOW,
        augmentation_factor=AUGMENTATION_FACTOR
    )
    print(f"Total series: {len(dict_list)}")

    train_list, test_list = train_test_split(dict_list, test_size=0.2, random_state=42)
    print(f"Train series: {len(train_list)}")
    print(f"Test series: {len(test_list)}")

    sample = dict_list[0]
    NUM_DYNAMIC_REAL_FEATURES = sample['feat_dynamic_real'].shape[0]
    NUM_PAST_DYNAMIC_FEATURES = sample['past_feat_dynamic_real'].shape[0]
    NUM_STATIC_REAL = len(sample['feat_static_real'])

    static_cat_cols = [c for c in STATIC_CAT_COLS if c in dataset.columns]
    CARDINALITY = [int(dataset[col].max()) + 1 for col in static_cat_cols]

    print(f"Feature dimensions: dynamic_real={NUM_DYNAMIC_REAL_FEATURES}, past_dynamic={NUM_PAST_DYNAMIC_FEATURES}, static_real={NUM_STATIC_REAL}")
    print(f"Cardinality (max+1): {CARDINALITY}")

    estimator = TemporalFusionTransformerEstimator(
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        freq='D',
        static_cardinalities=CARDINALITY,
        hidden_dim=HIDDEN_DIM,
        variable_dim=VARIABLE_DIM,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        trainer_kwargs={
            "max_epochs": NUM_EPOCHS,
            "accelerator": device_type,
            "devices": 1,
            "gradient_clip_val": 0.5,
        }
    )

    train_dataset = ListDataset(train_list, freq="D")
    test_dataset = ListDataset(test_list, freq="D")

    print(f"\n=== Training ===")
    predictor = estimator.train(
        train_dataset,
        validation_data=test_dataset
    )

    print(f"\n=== Evaluation ===")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=100
    )

    forecasts_log = list(forecast_it)
    tss_log = list(ts_it)

    forecasts = []
    for f_log in forecasts_log:
        if hasattr(f_log, 'samples'):
            inv_samples = np.expm1(f_log.samples)
            forecasts.append(SampleForecast(
                samples=inv_samples,
                start_date=f_log.start_date,
                item_id=f_log.item_id
            ))
        elif hasattr(f_log, 'forecast_array'):
            inv_forecast_array = np.expm1(f_log.forecast_array)
            forecasts.append(QuantileForecast(
                forecast_arrays=inv_forecast_array,
                start_date=f_log.start_date,
                forecast_keys=f_log.forecast_keys,
                item_id=f_log.item_id
            ))

    tss = [ts_log_obj.apply(np.expm1) for ts_log_obj in tss_log]

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    if USE_PRIORITIZED_REPLAY:
        print(f"\n=== Prioritized Experience Replay ===")
        print("Calculating training errors...")
        train_forecast_it, train_ts_it = make_evaluation_predictions(
            dataset=train_dataset,
            predictor=predictor,
            num_samples=100
        )
        train_forecasts = list(train_forecast_it)
        train_tss = list(train_ts_it)

        errors = []
        for i, (forecast, ts) in enumerate(zip(train_forecasts, train_tss)):
            if hasattr(forecast, 'mean'):
                pred_mean_log = np.array(forecast.mean)
            else:
                pred_mean_log = np.median(forecast.samples, axis=0)

            actual_values_log = ts[-PREDICTION_LENGTH:].values if hasattr(ts[-PREDICTION_LENGTH:], 'values') else np.array(ts[-PREDICTION_LENGTH:])

            pred_mean_yen = np.expm1(pred_mean_log)
            actual_values_yen = np.expm1(actual_values_log)

            mae = np.mean(np.abs(pred_mean_yen - actual_values_yen))
            errors.append((i, mae))

        errors_sorted = sorted(errors, key=lambda x: x[1], reverse=True)
        n_replay = int(len(errors_sorted) * REPLAY_TOP_RATIO)
        high_error_indices = [idx for idx, _ in errors_sorted[:n_replay]]

        print(f"Top {REPLAY_TOP_RATIO*100:.0f}% high-error samples: {n_replay}")
        print(f"Average error (top): {np.mean([e for _, e in errors_sorted[:n_replay]]):.4f}")
        print(f"Average error (all): {np.mean([e for _, e in errors]):.4f}")

        replay_samples = [train_list[i] for i in high_error_indices]
        augmented_train_list = train_list + replay_samples * REPLAY_MULTIPLIER
        print(f"Augmented training size: {len(train_list)} -> {len(augmented_train_list)}")

        augmented_train_dataset = ListDataset(augmented_train_list, freq="D")

        print(f"\n=== Fine-tuning with Prioritized Samples ===")
        estimator_finetune = TemporalFusionTransformerEstimator(
            prediction_length=PREDICTION_LENGTH,
            context_length=CONTEXT_LENGTH,
            freq='D',
            static_cardinalities=CARDINALITY,
            hidden_dim=HIDDEN_DIM,
            variable_dim=VARIABLE_DIM,
            num_heads=NUM_HEADS,
            dropout_rate=DROPOUT_RATE,
            lr=LEARNING_RATE * 0.1,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            trainer_kwargs={
                "max_epochs": REPLAY_EPOCHS,
                "accelerator": device_type,
                "devices": 1,
                "gradient_clip_val": 0.5,
            }
        )

        predictor = estimator_finetune.train(
            augmented_train_dataset,
            validation_data=test_dataset,
            from_predictor=predictor
        )

        print(f"\n=== Re-evaluation after Prioritized Replay ===")
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,
            predictor=predictor,
            num_samples=100
        )

        forecasts_log = list(forecast_it)
        tss_log = list(ts_it)

        forecasts = []
        for f_log in forecasts_log:
            if hasattr(f_log, 'samples'):
                inv_samples = np.expm1(f_log.samples)
                forecasts.append(SampleForecast(
                    samples=inv_samples,
                    start_date=f_log.start_date,
                    item_id=f_log.item_id
                ))
            elif hasattr(f_log, 'forecast_array'):
                inv_forecast_array = np.expm1(f_log.forecast_array)
                forecasts.append(QuantileForecast(
                    forecast_arrays=inv_forecast_array,
                    start_date=f_log.start_date,
                    forecast_keys=f_log.forecast_keys,
                    item_id=f_log.item_id
                ))

        tss = [ts_log_obj.apply(np.expm1) for ts_log_obj in tss_log]

        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(tss, forecasts)

    print("\n" + "="*60)
    print("Performance Metrics")
    print("="*60)
    """
    MAE(Mean Absolute Error)
    RMSE(Root Mean Square Error)
    MAPE(Mean Absolute Percentage Error)    誤差/実測値
    sMAPE                                   誤差/(予測+実測値)
    MASE (Mean Absolute Scaled Error)       誤差/1期間の平均誤差
    """
    for key in ['MASE', 'RMSE', 'MAE', 'MAPE', 'sMAPE']:
        if key in agg_metrics:
            print(f"{key:10s}: {agg_metrics[key]:.4f}")
    mase_value = agg_metrics.get('MASE', float('inf'))
    agg_metrics_df = pd.DataFrame([agg_metrics])

    score_dir = Path("score")
    score_dir.mkdir(exist_ok=True)
    agg_metrics_df.to_csv(score_dir / 'transformer_agg_metrics.csv', index=False)
    item_metrics.to_csv(score_dir / 'transformer_item_metrics.csv', index=True)

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    predictor.serialize(model_dir)
    print(f"\nModel saved to: {model_dir}/")
    print(f"Metrics saved to: {score_dir}/")
    print("="*60)