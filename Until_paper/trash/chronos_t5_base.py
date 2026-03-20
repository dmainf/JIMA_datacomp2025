import pandas as pd
from ch_function import *

ALL_PREDICT = True

def main():
    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    adapter_path = None

    print("\n=== Starting Inference (Zero-Shot) ===")
    decile_books = extract_decile_books(df)

    pipeline, accelerator = load_model(adapter_path)

    samples = preprocess_data(df, decile_books, ALL_PREDICT)
    print(f"Prepared {len(samples)} valid samples for inference.")

    forecasts = run_inference(pipeline, samples, accelerator)
    save_results(samples, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(samples, forecasts, CONFIG)
    print("Inference and saving complete.")

if __name__ == "__main__":
    main()