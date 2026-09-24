import argparse
import os
import pandas as pd
from func_gate_raf import *

ALL_PREDICT = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    K = args.k
    CONFIG["top_k"] = K
    CONFIG["lora_output_dir"] = f"gate_raf_k{K}_checkpoints"
    CONFIG["output_dir"] = f"GateRAF-k{K}"
    if args.tag:
        CONFIG["output_dir"] = f"GateRAF-{args.tag}"
    if args.steps:
        CONFIG["num_steps"] = args.steps
        CONFIG["eval_steps"] = max(1, args.steps // 2)
        CONFIG["lora_output_dir"] += "_smoke"
        CONFIG["output_dir"] += "-smoke"

    print(f"=== GateRAF top_k={K} ===", flush=True)
    print(f"checkpoints: {CONFIG['lora_output_dir']}", flush=True)
    print(f"predictions: {CONFIG['output_dir']}", flush=True)

    print("=== Loading Data ===", flush=True)
    required_cols = ['書名', '日付', 'POS販売冊数']
    df = pd.read_parquet('data/df_for.parquet', columns=required_cols)

    retriever = TimeSeriesRetriever(
        context_length=CONFIG["context_length"],
        retrieval_length=CONFIG["retrieval_length"]
    )
    retriever.build_index(df, step=CONFIG["index_step"])

    final_path = args.adapter or os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    if args.no_train or args.adapter:
        if not os.path.exists(final_path):
            raise SystemExit(f"Adapter not found: {final_path}")
        adapter_path = final_path
        print(f"Using existing adapter: {adapter_path}", flush=True)
    else:
        adapter_path = train_model(df, retriever=retriever)

    print("\n=== Starting Inference ===", flush=True)
    decile_books = extract_decile_books(df)
    model = load_model(adapter_path)

    infer_ds = ChronosBoltFiDDataset(
        df=df, prediction_length=CONFIG["prediction_length"],
        mode="inference", retriever=retriever,
        context_length=CONFIG["context_length"],
        top_k=K, decile_books=decile_books, all_predict=ALL_PREDICT
    )
    print(f"Prepared {len(infer_ds)} valid samples for inference.", flush=True)

    forecasts = run_inference(model, infer_ds)
    save_results(infer_ds.metadata, forecasts, decile_books, ALL_PREDICT)
    evaluate_predictions(infer_ds.metadata, forecasts, ALL_PREDICT)
    print(f"=== Done: top_k={K} ===", flush=True)


if __name__ == "__main__":
    main()
