import pandas as pd
import os
from func_sarima import *

ALL_PREDICT = True


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    decile_books = extract_decile_books(df)

    print("\n=== Starting SARIMA ===")
    samples, forecasts = run_sarima(df, ALL_PREDICT)

    if not samples:
        print("No valid predictions.")
        return

    save_results(samples, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(samples, forecasts, ALL_PREDICT)
    evaluate_spike_predictions(samples, forecasts, z_thresh=2.0, all_predict=ALL_PREDICT)
    print("Done.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
