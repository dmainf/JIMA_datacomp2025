import os
import pandas as pd
from func_lstm import *

ALL_PREDICT = True
DO_TRAIN = False


def main():
    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    decile_books = extract_decile_books(df)

    if DO_TRAIN:
        checkpoint_path = train_model(df)
    else:
        checkpoint_path = CONFIG["checkpoint_path"]
        if os.path.exists(checkpoint_path):
            print(f"Using existing checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Running with random weights.")
            checkpoint_path = None

    print(f"\n=== Starting Inference ===")
    metadata, forecasts = run_inference(df, decile_books, ALL_PREDICT, checkpoint_path)
    print(f"Completed {len(metadata)} books.")

    save_results(metadata, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(metadata, forecasts, ALL_PREDICT)
    print("Inference and saving complete.")


if __name__ == "__main__":
    main()
