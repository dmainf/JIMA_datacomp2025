import pandas as pd
import os
from func_gate_raf import *

ALL_PREDICT = True
DO_TRAIN = True


def main():
    CONFIG["lora_output_dir"] = "gate_raf_checkpoints"

    print("=== Loading Data ===")
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)
    print("Complete!")

    retriever = TimeSeriesRetriever(
        context_length=CONFIG["context_length"],
        retrieval_length=CONFIG["retrieval_length"],
        prediction_length=CONFIG["prediction_length"],
    )
    retriever.build_index(df, step=CONFIG["index_step"])

    gate_mlp_path = None

    if DO_TRAIN:
        gate_mlp_path = train_model(df, retriever=retriever)
    else:
        possible_path = os.path.join(CONFIG["lora_output_dir"], "gate_mlp.pt")
        if os.path.exists(possible_path):
            gate_mlp_path = possible_path
            print(f"Using existing gate_mlp: {gate_mlp_path}")
        else:
            print("gate_mlp not found. Running with untrained gate.")

    CONFIG["output_dir"] = "newGateRAF"

    print(f"\n=== Starting Inference ===")
    decile_books = extract_decile_books(df)

    model = load_model(gate_mlp_path)

    infer_ds = ChronosBoltFiDDataset(
        df=df, prediction_length=CONFIG["prediction_length"],
        mode="inference", retriever=retriever,
        context_length=CONFIG["context_length"],
        top_k=CONFIG["top_k"], decile_books=decile_books, all_predict=ALL_PREDICT
    )
    print(f"Prepared {len(infer_ds)} valid samples for inference.")

    forecasts = run_inference(model, infer_ds)
    save_results(infer_ds.metadata, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(infer_ds.metadata, forecasts, ALL_PREDICT)
    print("Inference and saving complete.")


if __name__ == "__main__":
    main()
