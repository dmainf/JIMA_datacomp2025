import pandas as pd
import numpy as np
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.dataset.common import ListDataset
import torch
from o_feature_eng import STATIC_CAT_COLS

CONTEXT_LENGTH = 56
PREDICTION_LENGTH = 14
HIDDEN_DIM = 128
VARIABLE_DIM = 64
NUM_HEADS = 4
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
PATIENCE = 4

print("Loading dataset...")
dataset = pd.read_parquet('dataset.parquet')
print(f"Dataset shape: {dataset.shape}")

CARDINALITY = [int(dataset[col].max()) + 1 for col in STATIC_CAT_COLS]
print(f"\nCardinalities: {CARDINALITY}")

dummy_data = {
    "start": pd.Period("2023-01-01", freq="D"),
    "target": np.random.randn(CONTEXT_LENGTH + PREDICTION_LENGTH).astype(np.float32),
    "feat_static_cat": [0] * len(CARDINALITY),
    "feat_static_real": [0.0] * 10,
    "feat_dynamic_real": np.random.randn(18, CONTEXT_LENGTH + PREDICTION_LENGTH).astype(np.float32),
    "past_feat_dynamic_real": np.random.randn(27, CONTEXT_LENGTH + PREDICTION_LENGTH).astype(np.float32)
}

print(f"\nFeature dimensions:")
print(f"  Static categorical: {len(dummy_data['feat_static_cat'])}")
print(f"  Static real: {len(dummy_data['feat_static_real'])}")
print(f"  Dynamic real: {dummy_data['feat_dynamic_real'].shape[0]}")
print(f"  Past dynamic real: {dummy_data['past_feat_dynamic_real'].shape[0]}")

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
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
    }
)

print("\nCreating model...")
train_dataset = ListDataset([dummy_data], freq="D")
predictor = estimator.train(train_dataset, cache_data=False, shuffle_buffer_length=1)

model = predictor.prediction_net
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "="*60)
print("Model Parameter Count")
print("="*60)
print(f"Total parameters:      {total_params:,}")
print(f"Trainable parameters:  {trainable_params:,}")
print("="*60)

print("\nParameter breakdown by layer:")
for name, param in model.named_parameters():
    print(f"{name:60s} {param.numel():>10,}")
