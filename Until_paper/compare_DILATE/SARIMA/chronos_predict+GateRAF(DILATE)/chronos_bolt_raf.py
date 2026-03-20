import pandas as pd
import numpy as np
import os
from func_chronos_bolt_raf import *

ALL_PREDICT = True
DO_TRAIN = False
USE_RAF = True


def main():
    CONFIG["use_raf"] = USE_RAF
    raf_suffix = "+RAF" if CONFIG["use_raf"] else ""
    CONFIG["lora_output_dir"] = f"dora_checkpoints{raf_suffix}"

    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    #df['POS販売冊数'] = np.log1p(df['POS販売冊数'])
    print("Complete!")

    retriever = None
    if CONFIG["use_raf"]:
        retriever = TimeSeriesRetriever(
            context_length=CONFIG["context_length"],
            retrieval_length=CONFIG["retrieval_length"]
        )
        retriever.build_index(df, step=CONFIG["index_step"])

    adapter_path = None

    if DO_TRAIN:
        adapter_path = train_model(df, retriever=retriever)
    else:
        adapter_name = "final_adapter"
        possible_path = os.path.join(CONFIG["lora_output_dir"], adapter_name)
        if os.path.exists(possible_path):
            adapter_path = possible_path
            print(f"Using existing adapter: {adapter_path}")
        else:
            print(f"Adapter not found at {possible_path}. Running zero-shot inference.")

    base_name = f"chronos_bolt(DoRA){raf_suffix}" if adapter_path else f"chronos_bolt(zero-shot){raf_suffix}"
    CONFIG["output_dir"] = base_name

    print(f"\n=== Starting Inference ===")
    decile_books = extract_decile_books(df)

    model = load_model(adapter_path)

    infer_ds = ChronosBoltFiDDataset(
        df=df, prediction_length=CONFIG["prediction_length"],
        mode="inference", retriever=retriever,
        use_raf=CONFIG["use_raf"], context_length=CONFIG["context_length"],
        top_k=CONFIG["top_k"], decile_books=decile_books, all_predict=ALL_PREDICT
    )
    print(f"Prepared {len(infer_ds)} valid samples for inference.")

    forecasts = run_inference(model, infer_ds)
    save_results(infer_ds.metadata, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(infer_ds.metadata, forecasts, ALL_PREDICT)
    print("Inference and saving complete.")


if __name__ == "__main__":
    main()
