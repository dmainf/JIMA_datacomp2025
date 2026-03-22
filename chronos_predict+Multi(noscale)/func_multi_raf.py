import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, PeftModel
import os
from tqdm import tqdm
import warnings
import logging

import faiss
from model_multi_raf import ChronosBoltFiDModel

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "model_name": "amazon/chronos-bolt-base",
    "prediction_length": 64,
    "context_length": 128,
    "retrieval_length": 128,
    "batch_size": 16,
    "output_dir": "Multi-noscale",
    "lora_output_dir": "multi_raf_noscale_checkpoints",
    "learning_rate": 1e-5,
    "grad_accum_steps": 4,
    "num_steps": 10000,
    "patience": 3,
    "top_k": 3,
    "index_step": 10,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "optim": "adamw_torch",
    "eval_steps": 500,
    "logging_steps": 100,
}

PEFT_CONFIG = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules="all-linear",
    use_dora=True
)


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
            means = np.mean(windows, axis=1, keepdims=True)
            stds = np.std(windows, axis=1, keepdims=True) + 1e-5
            normalized_windows = (windows - means) / stds
            vectors_list.append(normalized_windows)
            scales_list.append(stds.flatten())
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
        N = len(query_batch)
        empty_raf = np.full((k, self.retrieval_length), np.nan, dtype=np.float32)
        empty_mask = np.zeros((k, self.retrieval_length), dtype=bool)
        empty_ratio = np.ones(k, dtype=np.float32)

        if self.index is None:
            return [empty_raf.copy() for _ in range(N)], [empty_mask.copy() for _ in range(N)], [empty_ratio.copy() for _ in range(N)]

        faiss.omp_set_num_threads(1)

        means = np.mean(query_batch, axis=1, keepdims=True)
        stds = np.std(query_batch, axis=1, keepdims=True) + 1e-5
        normalized_queries = ((query_batch - means) / stds).astype('float32')
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

            raf_ctx = np.full((k, self.retrieval_length), np.nan, dtype=np.float32)
            raf_msk = np.zeros((k, self.retrieval_length), dtype=bool)
            raf_ratio = np.ones(k, dtype=np.float32)

            for j, idx in enumerate(top_indices):
                vec = self.vectors_store[idx]
                retrieved_scale = self.scales_store[idx]
                raf_ctx[j] = vec * retrieved_scale
                raf_msk[j] = True
                raf_ratio[j] = retrieved_scale / stds[i, 0]

            raf_contexts.append(raf_ctx)
            raf_masks.append(raf_msk)
            raf_scale_ratios.append(raf_ratio)

        return raf_contexts, raf_masks, raf_scale_ratios


class ChronosBoltFiDDataset(Dataset):
    def __init__(self, df, prediction_length, mode="train", split_ratio=0.9,
                 retriever=None, context_length=512,
                 top_k=1, decile_books=None, all_predict=True):
        self.samples = []
        self.metadata = []
        sample_info = []
        retrieval_length = retriever.retrieval_length

        for book_name, group in tqdm(df.groupby('書名', observed=True), desc="Processing Groups"):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
            dates = group['日付'].values
            total_len = len(series)
            if total_len <= prediction_length + context_length:
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

                use_len = min(len(query), retrieval_length)
                query_slice = query[-use_len:]
                query_scale = np.std(query_slice) + 1e-5
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
                max_start = total_len - 2 * prediction_length
                split_idx = int(max_start * split_ratio)
                indices = range(0, split_idx) if mode == "train" else range(split_idx, max_start)
                for i in indices:
                    target = series[i: i + prediction_length]
                    ctx_start = max(0, i - context_length)
                    local_ctx = series[ctx_start: i]
                    if len(local_ctx) < context_length:
                        local_ctx = np.concatenate([
                            np.full(context_length - len(local_ctx), np.nan, dtype=np.float32),
                            local_ctx
                        ])

                    q_start = max(0, i - retrieval_length)
                    query_slice = series[q_start:i]
                    query_scale = np.std(query_slice) + 1e-5
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

            raf_ctx = raf_contexts[idx]
            raf_msk = raf_masks[idx]
            raf_ratio = raf_scale_ratios[idx]

            raf_context_tensor = torch.tensor(raf_ctx, dtype=torch.float32)
            raf_mask_tensor = torch.tensor(raf_msk, dtype=torch.bool)
            raf_context_tensor = torch.nan_to_num(raf_context_tensor, nan=0.0)

            sample = {
                "context": context_tensor,
                "mask": mask_tensor,
                "raf_context": raf_context_tensor,
                "raf_mask": raf_mask_tensor,
                "raf_scale_ratio": torch.tensor(raf_ratio, dtype=torch.float32),
            }

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
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


class ChronosFiDTrainer(Trainer):
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


def _get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def train_model(df, retriever):
    print("\n=== Training Model ===", flush=True)
    device, dtype = _get_device_dtype()
    print(f"Using device: {device}, dtype: {dtype}", flush=True)
    model = ChronosBoltFiDModel.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype
    ).to(device)
    model.chronos_config.context_length = CONFIG["context_length"]
    model.config.chronos_config["context_length"] = CONFIG["context_length"]
    model = get_peft_model(model, PEFT_CONFIG)

    common_args = {
        "df": df, "prediction_length": CONFIG["prediction_length"],
        "retriever": retriever, "context_length": CONFIG["context_length"],
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
        max_steps=CONFIG["num_steps"],
        eval_strategy="steps", eval_steps=CONFIG["eval_steps"],
        save_strategy="steps", save_steps=CONFIG["eval_steps"],
        logging_steps=CONFIG["logging_steps"], report_to="none",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        remove_unused_columns=False
    )
    trainer = ChronosFiDTrainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=fid_collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["patience"])]
    )
    trainer.train()
    save_loss_plot(trainer.state.log_history, CONFIG["lora_output_dir"], "loss_curve.png")
    save_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    model.save_pretrained(save_path)
    return save_path


def load_model(adapter_path=None):
    device, dtype = _get_device_dtype()
    model = ChronosBoltFiDModel.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype
    ).to(device)
    model.chronos_config.context_length = CONFIG["context_length"]
    model.config.chronos_config["context_length"] = CONFIG["context_length"]
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


def evaluate_predictions(samples, forecasts, all_predict):
    pred_array = np.stack([f.numpy() for f in forecasts])
    n = pred_array.shape[1]
    q_indices = [0, n // 2, n - 1]

    rows = []
    for i, sample in enumerate(samples):
        book_name = sample['id']
        target = sample["target"]
        pred = pred_array[i]
        for day in range(len(target)):
            rows.append({
                "書名": book_name, "day": day, "actual": target[day],
                "q0.1": pred[q_indices[0], day],
                "q0.5": pred[q_indices[1], day],
                "q0.9": pred[q_indices[2], day],
            })

    df_out = pd.DataFrame(rows)
    fname = "predictions_all.csv" if all_predict else "predictions_decile.csv"
    df_out.to_csv(f"{CONFIG['output_dir']}/{fname}", index=False)
    print(f"Saved {df_out['書名'].nunique()} books, {len(df_out)} rows → {CONFIG['output_dir']}/{fname}")
