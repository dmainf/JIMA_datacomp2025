import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, AdaLoraConfig, PeftModel
import os
from tqdm import tqdm
import warnings
import logging

import faiss
from model_raf import ChronosBoltFiDModel

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "model_name": "amazon/chronos-bolt-tiny",
    "prediction_length": 64,
    "context_length": 128,
    "retrieval_length": 128,
    "batch_size": 16,
    "output_dir": "chronos_bolt",
    "lora_output_dir": "dora_checkpoints",
    "learning_rate": 1e-5,
    "grad_accum_steps": 16,
    "epochs": 1,
    "top_k": 3,
    "index_step": 1,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "optim": "adamw_torch",
    "adalora_total_steps": 1000,
}

PEFT_CONFIG = AdaLoraConfig(
    inference_mode=False,
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=100,
    tfinal=700,
    deltaT=10,
    total_step=CONFIG["adalora_total_steps"],
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
)


def scale_series(series):
    scale = np.mean(np.abs(series)) + 1.0
    return series / scale, scale


def extract_decile_books(df):
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    labels = [f"{i*5}%" for i in range(1, 21)]
    categories = pd.qcut(total_sales, 20, labels=labels, duplicates='drop')
    deciles = {}
    for label, group in total_sales.groupby(categories, observed=True):
        book = group.index[len(group) // 2]
        deciles[book] = {
            "label": str(label),
            "file_prefix": str(label),
            "total_sales": group[book]
        }
    return deciles


class TimeSeriesRetriever:
    def __init__(self, context_length, retrieval_length):
        self.context_length = context_length
        self.retrieval_length = retrieval_length
        self.index = None
        self.timestamps = None
        self.vectors_store = None
        self.scales_store = None

    def build_index(self, df, step=30):
        print(f"=== Building Vector Index (len={self.retrieval_length}, step={step}) ===")
        timestamps_list = []
        vectors_list = []
        scales_list = []
        for _, group in tqdm(df.groupby('書名', observed=True)):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
            dates = group['日付'].values
            if len(series) < self.retrieval_length:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(series, self.retrieval_length)[::step]
            end_dates = dates[self.retrieval_length - 1:][::step]
            valid_mask = np.sum(np.abs(windows), axis=1) > 0
            if not np.any(valid_mask):
                continue
            windows = windows[valid_mask]
            end_dates = end_dates[valid_mask]
            scales = np.mean(np.abs(windows), axis=1, keepdims=True) + 1.0
            normalized_windows = windows / scales
            vectors_list.append(normalized_windows)
            scales_list.append(scales.flatten())
            timestamps_list.extend(end_dates)
        if not vectors_list:
            print("No vectors created.")
            return
        all_vectors = np.concatenate(vectors_list).astype('float32')
        all_scales = np.concatenate(scales_list).astype('float32')
        self.vectors_store = all_vectors
        self.scales_store = all_scales
        self.timestamps = pd.to_datetime(timestamps_list).values
        self.index = faiss.IndexFlatL2(self.retrieval_length)
        self.index.add(all_vectors)
        print(f"Index built: {self.index.ntotal} vectors of dim {self.retrieval_length}.")

    def search_batch(self, query_batch, query_dates, query_scales, k=1):
        """
        Returns:
            raf_contexts: List of [K, L_ret] arrays (raw values, not normalized)
            raf_masks: List of [K, L_ret] boolean arrays
            raf_scale_ratios: List of [K] arrays (retrieved_scale / query_scale)
        """
        N = len(query_batch)
        empty_raf = np.full((k, self.retrieval_length), np.nan, dtype=np.float32)
        empty_mask = np.zeros((k, self.retrieval_length), dtype=bool)
        empty_ratio = np.ones(k, dtype=np.float32)

        if self.index is None:
            return [empty_raf.copy() for _ in range(N)], [empty_mask.copy() for _ in range(N)], [empty_ratio.copy() for _ in range(N)]

        faiss.omp_set_num_threads(1)

        scales = np.mean(np.abs(query_batch), axis=1, keepdims=True) + 1.0
        normalized_queries = (query_batch / scales).astype('float32')
        normalized_queries = np.ascontiguousarray(normalized_queries)

        search_k = min(k * 5 + 10, self.index.ntotal)
        D, I = self.index.search(normalized_queries, search_k)

        found_dates = self.timestamps[I]
        query_dates_vec = pd.to_datetime(query_dates).values.reshape(-1, 1)
        valid_mask = found_dates < query_dates_vec

        raf_contexts = []
        raf_masks = []
        raf_scale_ratios = []

        for i in range(N):
            valid_indices = I[i][valid_mask[i]]
            if len(valid_indices) == 0:
                raf_contexts.append(empty_raf.copy())
                raf_masks.append(empty_mask.copy())
                raf_scale_ratios.append(empty_ratio.copy())
                continue

            top_indices = valid_indices[:k]
            num_found = len(top_indices)

            raf_ctx = np.full((k, self.retrieval_length), np.nan, dtype=np.float32)
            raf_msk = np.zeros((k, self.retrieval_length), dtype=bool)
            raf_ratio = np.ones(k, dtype=np.float32)

            for j, idx in enumerate(top_indices):
                vec = self.vectors_store[idx]
                retrieved_scale = self.scales_store[idx]
                raf_ctx[j] = vec * retrieved_scale
                raf_msk[j] = True
                raf_ratio[j] = retrieved_scale / query_scales[i]

            raf_contexts.append(raf_ctx)
            raf_masks.append(raf_msk)
            raf_scale_ratios.append(raf_ratio)

        return raf_contexts, raf_masks, raf_scale_ratios


class ChronosBoltFiDDataset(Dataset):
    def __init__(self, df, prediction_length, mode="train", split_ratio=0.9,
                 retriever=None, use_raf=True, context_length=512,
                 top_k=1, decile_books=None, all_predict=True):
        self.samples = []
        self.metadata = []
        sample_info = []
        retrieval_length = retriever.retrieval_length if retriever else 0

        for book_name, group in tqdm(df.groupby('書名', observed=True), desc="Processing Groups"):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
            dates = group['日付'].values
            total_len = len(series)
            if total_len <= prediction_length:
                continue

            if mode == "inference":
                if not all_predict and book_name not in (decile_books or {}):
                    continue
                target = series[-prediction_length:]
                query = series[:-prediction_length]
                query_date = dates[len(query) - 1]

                local_ctx = query[-context_length:] if len(query) >= context_length else query
                if len(local_ctx) < context_length:
                    local_ctx = np.concatenate([
                        np.full(context_length - len(local_ctx), np.nan, dtype=np.float32),
                        local_ctx
                    ])

                query_slice = None
                query_scale = 1.0
                if retriever:
                    use_len = min(len(query), retrieval_length)
                    query_slice = query[-use_len:]
                    query_scale = np.mean(np.abs(query_slice)) + 1.0
                    if use_len < retrieval_length:
                        query_slice = np.pad(query_slice, (retrieval_length - use_len, 0), 'constant')

                sample_info.append({
                    'is_inference': True,
                    'local_ctx': local_ctx,
                    'query_slice': query_slice,
                    'query_scale': query_scale,
                    'query_date': query_date,
                    'target': target,
                    'query_raw': query,
                    'book_name': book_name,
                })
            else:
                split_idx = int((total_len - prediction_length) * split_ratio)
                indices = range(0, split_idx) if mode == "train" else range(split_idx, total_len - prediction_length)
                for i in indices:
                    target = series[i: i + prediction_length]
                    ctx_start = max(0, i - context_length)
                    local_ctx = series[ctx_start: i]
                    if len(local_ctx) < context_length:
                        local_ctx = np.concatenate([
                            np.full(context_length - len(local_ctx), np.nan, dtype=np.float32),
                            local_ctx
                        ])

                    query_slice = None
                    query_scale = 1.0
                    if retriever:
                        q_start = max(0, i - retrieval_length)
                        query_slice = series[q_start:i]
                        query_scale = np.mean(np.abs(query_slice)) + 1.0
                        if len(query_slice) < retrieval_length:
                            query_slice = np.pad(query_slice, (retrieval_length - len(query_slice), 0), 'constant')

                    sample_info.append({
                        'is_inference': False,
                        'local_ctx': local_ctx,
                        'query_slice': query_slice,
                        'query_scale': query_scale,
                        'query_date': dates[max(0, i - 1)],
                        'target': target,
                    })

        raf_contexts = [None] * len(sample_info)
        raf_masks = [None] * len(sample_info)
        raf_scale_ratios = [None] * len(sample_info)

        if use_raf and retriever and retriever.index is not None and sample_info:
            query_matrix = np.stack([info['query_slice'] for info in sample_info])
            query_dates_arr = np.array([info['query_date'] for info in sample_info])
            query_scales_arr = np.array([info['query_scale'] for info in sample_info])
            print(f"Batch searching {len(query_matrix)} samples...")
            raf_contexts, raf_masks, raf_scale_ratios = retriever.search_batch(
                query_matrix, query_dates_arr, query_scales_arr, k=top_k
            )

        for idx, info in enumerate(tqdm(sample_info, desc="Building Samples")):
            local_ctx = info['local_ctx']

            context_tensor = torch.tensor(local_ctx, dtype=torch.float32)
            mask_tensor = ~torch.isnan(context_tensor)
            context_tensor = torch.nan_to_num(context_tensor, nan=0.0)

            sample = {
                "context": context_tensor,
                "mask": mask_tensor,
            }

            if use_raf and raf_contexts[idx] is not None:
                raf_ctx = raf_contexts[idx]
                raf_msk = raf_masks[idx]
                raf_ratio = raf_scale_ratios[idx]

                raf_context_tensor = torch.tensor(raf_ctx, dtype=torch.float32)
                raf_mask_tensor = torch.tensor(raf_msk, dtype=torch.bool)
                raf_context_tensor = torch.nan_to_num(raf_context_tensor, nan=0.0)

                sample["raf_context"] = raf_context_tensor
                sample["raf_mask"] = raf_mask_tensor
                sample["raf_scale_ratio"] = torch.tensor(raf_ratio, dtype=torch.float32)

            if info['is_inference']:
                self.metadata.append({
                    "id": info['book_name'],
                    "target": info['target'],
                    "query": info['query_raw'],
                })
            else:
                sample["target"] = torch.tensor(info['target'], dtype=torch.float32)

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def fid_collate_fn(batch):
    result = {}
    for k in batch[0]:
        if k in ["raf_context", "raf_mask", "raf_scale_ratio"]:
            if k in batch[0]:
                result[k] = torch.stack([item[k] for item in batch if k in item])
        else:
            result[k] = torch.stack([item[k] for item in batch])
    return result


class ChronosFiDTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        for p in model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)
        model.update_and_allocate(self.state.global_step)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
        return (loss.detach(), None, None)


def save_loss_plot(log_history, output_dir, filename="loss_curve.png"):
    history = pd.DataFrame(log_history)
    plt.figure(figsize=(10, 6))
    if "loss" in history.columns:
        train = history.dropna(subset=["loss"])
        plt.plot(train["step"], train["loss"], label="Training Loss", color="blue", alpha=0.6)
    if "eval_loss" in history.columns:
        val = history.dropna(subset=["eval_loss"])
        plt.plot(val["step"], val["eval_loss"], label="Validation Loss", color="red", linewidth=2, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def train_model(df, retriever=None):
    print("\n=== Training Model ===", flush=True)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}", flush=True)
    model = ChronosBoltFiDModel.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype
    ).to(device)
    model.chronos_config.context_length = CONFIG["context_length"]
    model.config.chronos_config["context_length"] = CONFIG["context_length"]
    model = get_peft_model(model, PEFT_CONFIG)

    common_args = {
        "df": df, "prediction_length": CONFIG["prediction_length"],
        "retriever": retriever, "use_raf": CONFIG["use_raf"],
        "context_length": CONFIG["context_length"],
        "top_k": CONFIG["top_k"]
    }
    train_ds = ChronosBoltFiDDataset(mode="train", **common_args)
    val_ds = ChronosBoltFiDDataset(mode="val", **common_args)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    args = TrainingArguments(
        output_dir=CONFIG["lora_output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        gradient_accumulation_steps=CONFIG["grad_accum_steps"],
        max_grad_norm=CONFIG["max_grad_norm"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        optim=CONFIG["optim"],
        num_train_epochs=CONFIG["epochs"],
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50,
        logging_steps=50, report_to="none",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        remove_unused_columns=False
    )
    trainer = ChronosFiDTrainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=fid_collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    save_loss_plot(trainer.state.log_history, CONFIG["lora_output_dir"], "loss_curve.png")
    adapter_name = "final_adapter"
    save_path = os.path.join(CONFIG["lora_output_dir"], adapter_name)
    model.save_pretrained(save_path)
    return save_path


def load_model(adapter_path=None):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    model = ChronosBoltFiDModel.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype
    ).to(device)
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def run_inference(model, dataset):
    print(f"\n=== Inference ===")
    loader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=fid_collate_fn
    )
    device = next(model.parameters()).device
    pred_len = CONFIG["prediction_length"]
    model.eval()
    forecasts = []
    for batch in tqdm(loader):
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch_gpu)
            preds = output.quantile_preds[:, :, :pred_len].cpu()
        forecasts.extend(list(preds))
    return forecasts


def save_results(samples, forecasts, decile_books, all_predict):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if all_predict:
        os.makedirs(f"{CONFIG['output_dir']}/All_predict", exist_ok=True)
    count = 0
    for i, sample in enumerate(tqdm(samples, desc="Saving Plots")):
        book_name = sample['id']
        if not all_predict and book_name not in decile_books:
            continue
        pred = forecasts[i].numpy()
        target = sample["target"]
        history = sample.get("query", np.array([]))
        len_history = len(history)
        fig, ax = plt.subplots(figsize=(12, 6))
        if len_history > 0:
            plot_start = max(0, len_history - 200)
            ax.plot(range(plot_start, len_history), history[plot_start:],
                    label="History", color="black", alpha=0.5)
        x_pred = range(len_history, len_history + len(target))
        ax.plot(x_pred, target, label="Actual", color="gray", linestyle="--")
        median = pred[pred.shape[0] // 2]
        low = pred[0] if pred.shape[0] > 1 else median
        high = pred[-1] if pred.shape[0] > 1 else median
        ax.plot(x_pred, median, label="Forecast", color="blue")
        ax.fill_between(x_pred, low, high, color="blue", alpha=0.2)
        if len_history > 0:
            ax.axvline(x=len_history, color="red", linestyle=":", alpha=0.7, label="Forecast Start")
        title = f"{book_name}"
        if book_name in decile_books:
            title = f"[{decile_books[book_name]['label']}] {title}"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        safe_name = str(book_name).replace("/", "_")
        fname = f"{safe_name}.png"
        if book_name in decile_books:
            fname = f"{decile_books[book_name]['file_prefix']}_{fname}"
            fig.savefig(f"{CONFIG['output_dir']}/{fname}")
        elif all_predict:
            fig.savefig(f"{CONFIG['output_dir']}/All_predict/{fname}")
        plt.close(fig)
        count += 1
    print(f"Saved {count} plots.")


def _pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.sum(np.maximum(q * diff, (q - 1) * diff))


def evaluate_predictions(samples, forecasts, all_predict):
    print("\n=== Calculating Evaluation Metrics ===")
    targets = np.stack([s["target"] for s in samples])
    pred_array = np.stack([f.numpy() for f in forecasts])
    book_names = [s["id"] for s in samples]

    q10 = pred_array[:, 0, :]
    q50 = pred_array[:, pred_array.shape[1] // 2, :]
    q90 = pred_array[:, -1, :]

    abs_err = np.abs(targets - q50)
    sq_err = (targets - q50) ** 2
    in_interval = (targets >= q10) & (targets <= q90)

    item_mae = np.mean(abs_err, axis=1)
    item_rmse = np.sqrt(np.mean(sq_err, axis=1))
    item_coverage = np.mean(in_interval, axis=1)
    item_total_sales = np.sum(targets, axis=1)
    pred_len = targets.shape[1]

    diff_10 = targets - q10
    diff_50 = targets - q50
    diff_90 = targets - q90
    item_wql_10 = 2 * np.sum(np.maximum(0.1 * diff_10, -0.9 * diff_10), axis=1)
    item_wql_50 = 2 * np.sum(np.maximum(0.5 * diff_50, -0.5 * diff_50), axis=1)
    item_wql_90 = 2 * np.sum(np.maximum(0.9 * diff_90, -0.1 * diff_90), axis=1)

    df_res = pd.DataFrame({
        "book_name": book_names,
        "MAE": item_mae,
        "RMSE": item_rmse,
        "wQL_0.1": item_wql_10,
        "wQL_0.5": item_wql_50,
        "wQL_0.9": item_wql_90,
        "wQL_Mean": (item_wql_10 + item_wql_50 + item_wql_90) / 3,
        "Coverage_80%": item_coverage,
        "Total_Sales": item_total_sales,
    })
    fname = "evaluation_all.csv" if all_predict else "evaluation_decile.csv"
    df_res.to_csv(f"{CONFIG['output_dir']}/{fname}", index=False)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(sq_err))
    coverage = np.mean(in_interval)
    total_sales = np.sum(np.abs(targets))
    if total_sales > 0:
        wql_10 = 2 * _pinball_loss(targets, q10, 0.1) / total_sales
        wql_50 = 2 * _pinball_loss(targets, q50, 0.5) / total_sales
        wql_90 = 2 * _pinball_loss(targets, q90, 0.9) / total_sales
    else:
        wql_10, wql_50, wql_90 = 0.0, 0.0, 0.0
    final_metrics = {
        "MAE": mae, "RMSE": rmse, "Coverage_80%": coverage,
        "wQL_0.1": wql_10, "wQL_0.5": wql_50, "wQL_0.9": wql_90,
        "wQL_Mean": (wql_10 + wql_50 + wql_90) / 3,
    }
    print("-" * 40)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 40)
