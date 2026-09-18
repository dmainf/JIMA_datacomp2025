import pandas as pd
from func_prophet import *

ALL_PREDICT = True


def main():
    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    decile_books = extract_decile_books(df)

    print(f"\n=== Starting Inference ===")
    metadata, forecasts = run_inference(df, decile_books, ALL_PREDICT)
    print(f"Completed {len(metadata)} books.")

    save_results(metadata, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(metadata, forecasts, ALL_PREDICT)
    print("Inference and saving complete.")


if __name__ == "__main__":
    main()
