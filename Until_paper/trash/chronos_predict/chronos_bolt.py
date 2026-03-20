import pandas as pd
import os
from func_chronos_bolt import *

ALL_PREDICT = True
DO_TRAIN = False

def main():
    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    adapter_path = None

    if DO_TRAIN:
        adapter_path = train_model(df)
    else:
        possible_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
        if os.path.exists(possible_path):
            adapter_path = possible_path
            print(f"Using existing adapter: {adapter_path}")
        else:
            print(f"Adapter not found at {possible_path}. Running zero-shot inference.")

    CONFIG["output_dir"] = "chronos_bolt+FT" if adapter_path else "chronos_bolt(zero-shot)"

    print("\n=== Starting Inference ===")
    decile_books = extract_decile_books(df)

    pipeline, accelerator = load_model(adapter_path)

    samples = preprocess_data(df, decile_books, ALL_PREDICT)
    print(f"Prepared {len(samples)} valid samples for inference.")

    forecasts = run_inference(pipeline, samples, accelerator)
    save_results(samples, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(samples, forecasts, CONFIG, ALL_PREDICT)
    print("Inference and saving complete.")

if __name__ == "__main__":
    main()
