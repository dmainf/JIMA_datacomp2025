import pandas as pd
import numpy as np
from o_feature_eng import create_slices

CONTEXT_LENGTH = 56
PREDICTION_LENGTH = 14
REGION_EXTEND = 10
AUGMENTATION_FACTOR = 10

print("Loading dataset...")
dataset = pd.read_parquet('dataset.parquet')
print(f"Dataset shape: {dataset.shape}")
print(f"Dataset columns: {len(dataset.columns)}")

print("\n=== Creating Slices ===")
np.random.seed(42)
slices = create_slices(
    dataset,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    region_extend=REGION_EXTEND,
    augmentation_factor=AUGMENTATION_FACTOR
)

print(f"\n=== Data Size Analysis ===")
print(f"Total slices: {len(slices)}")

sample = slices[0]
target_length = len(sample['target'])
feat_dynamic_shape = sample['feat_dynamic_real'].shape
past_feat_dynamic_shape = sample['past_feat_dynamic_real'].shape
n_static_cat = len(sample['feat_static_cat'])
n_static_real = len(sample['feat_static_real'])

print(f"\n1 slice contains:")
print(f"  Time steps: {target_length}")
print(f"  Target values: {target_length}")
print(f"  Dynamic real features: {feat_dynamic_shape[0]} features × {feat_dynamic_shape[1]} time steps = {feat_dynamic_shape[0] * feat_dynamic_shape[1]} values")
print(f"  Past dynamic real features: {past_feat_dynamic_shape[0]} features × {past_feat_dynamic_shape[1]} time steps = {past_feat_dynamic_shape[0] * past_feat_dynamic_shape[1]} values")
print(f"  Static categorical features: {n_static_cat}")
print(f"  Static real features: {n_static_real}")

total_per_slice = (
    target_length +  # target
    feat_dynamic_shape[0] * feat_dynamic_shape[1] +  # feat_dynamic_real
    past_feat_dynamic_shape[0] * past_feat_dynamic_shape[1] +  # past_feat_dynamic_real
    n_static_cat +  # feat_static_cat
    n_static_real  # feat_static_real
)

print(f"\n  Total values per slice: {total_per_slice:,}")

print(f"\n=== Total Dataset Size ===")
print(f"Total slices: {len(slices):,}")
print(f"Values per slice: {total_per_slice:,}")
print(f"Total values: {len(slices) * total_per_slice:,}")

memory_per_slice_bytes = (
    target_length * 4 +  # float32
    feat_dynamic_shape[0] * feat_dynamic_shape[1] * 4 +  # float32
    past_feat_dynamic_shape[0] * past_feat_dynamic_shape[1] * 4 +  # float32
    n_static_cat * 4 +  # int32
    n_static_real * 4  # float32
)
memory_per_slice_kb = memory_per_slice_bytes / 1024

print(f"\n=== Memory Estimation ===")
print(f"Memory per slice: {memory_per_slice_kb:.2f} KB")
print(f"Total memory for all slices: {len(slices) * memory_per_slice_kb / 1024:.2f} MB")

print(f"\n=== Breakdown by Component ===")
print(f"Context length: {CONTEXT_LENGTH}")
print(f"Prediction length: {PREDICTION_LENGTH}")
print(f"Total sequence length: {CONTEXT_LENGTH + PREDICTION_LENGTH}")
print(f"")
print(f"Each time step has:")
print(f"  - 1 target value")
print(f"  - {feat_dynamic_shape[0]} dynamic real features")
print(f"  - {past_feat_dynamic_shape[0]} past dynamic real features")
print(f"  Total per time step: {1 + feat_dynamic_shape[0] + past_feat_dynamic_shape[0]} values")
print(f"")
print(f"Static features (same for all time steps):")
print(f"  - {n_static_cat} categorical features")
print(f"  - {n_static_real} real features")
