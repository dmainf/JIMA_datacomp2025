import pandas as pd
import numpy as np
from pathlib import Path
from o_feature_eng import *
from common import *
import warnings
import logging
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

PREDICTION_LENGTH = 14

REGION_EXTEND = 10
AUGMENTATION_FACTOR = 1

MAKE_DATASET = True


if __name__ == '__main__':
    print("\n=== Loading All Store Data ===")
    df = pd.read_parquet('data/sales_df.parquet', engine='pyarrow')
    df = df[~df['書店コード'].isin(['26','27'])].copy()
    print(f"Total records: {len(df)}")

    context_lengths = list(range(7, 64, 7))
    slice_lengths = []

    for ctx_len in context_lengths:
        print(f"\n=== Processing CONTEXT_LENGTH = {ctx_len} ===")

        print("Preparing Dataset...")
        dataset, encoders = prepare_dataset(
            df,
            context_length=ctx_len,
            prediction_length=PREDICTION_LENGTH,
        )

        print("Creating Slice Dataset...")
        slice = create_slices(
            dataset,
            context_length=ctx_len,
            prediction_length=PREDICTION_LENGTH,
            region_extend=REGION_EXTEND,
            augmentation_factor=AUGMENTATION_FACTOR
        )

        slice_len = len(slice)
        slice_lengths.append(slice_len)
        print(f"CONTEXT_LENGTH={ctx_len}: len(slice)={slice_len}")

    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, slice_lengths, marker='o', linewidth=2, markersize=8)
    plt.xlabel('CONTEXT_LENGTH', fontsize=12)
    plt.ylabel('len(slice)', fontsize=12)
    plt.title('Relationship between CONTEXT_LENGTH and len(slice)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(context_lengths)
    plt.tight_layout()
    plt.savefig('context_length_vs_slice_length.png', dpi=300)
    print("\n=== Plot saved to context_length_vs_slice_length.png ===")
    plt.show()