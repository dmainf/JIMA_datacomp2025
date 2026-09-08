import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
import faiss
from model_lstm_gate_raf import LSTMForecaster

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "prediction_length": 64,
    "context_length": 128,
    "retrieval_length": 128,
    "top_k": 3,
    "index_step": 10,
    "batch_size": 64,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "epochs": 100,
    "warmup_ratio": 0.05,
    "patience": 5,
    "output_dir": "LSTM-GateRAF",
    "checkpoint_path": "lstm_gate_raf_checkpoints/best_model.pt",
    "quantiles": [0.1, 0.5, 0.9],
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


class LSTMDataset(Dataset):
    def __init__(self, df, prediction_length, context_length, mode="train",
                 split_ratio=0.9, decile_books=None, all_predict=True,
                 retriever=None, top_k=3):
        self.samples = []
        self.metadata = []
        sample_info = []
        retrieval_length = retriever.retrieval_length if retriever else context_length

        for book_name, group in tqdm(df.groupby('書名', observed=True), desc=f"Processing ({mode})"):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
            dates = group['日付'].values
            total_len = len(series)

            if mode == "inference":
                if total_len <= prediction_length + context_length:
                    continue
            else:
                if total_len <= 2 * prediction_length + context_length:
                    continue

            if mode == "inference":
                if not all_predict and book_name not in (decile_books or {}):
                    continue
                query = series[:-prediction_length]
                target = series[-prediction_length:]
                query_date = dates[len(query) - 1]

                local_ctx = query[-context_length:]
                scale = np.mean(np.abs(local_ctx)) + 1.0
                ctx_scaled = local_ctx / scale

                use_len = min(len(query), retrieval_length)
                query_slice = query[-use_len:]
                query_scale = np.std(query_slice) + 1e-5
                if use_len < retrieval_length:
                    query_slice = np.pad(query_slice, (retrieval_length - use_len, 0), 'constant')

                sample_info.append({
                    'is_inference': True,
                    'ctx_scaled': ctx_scaled,
                    'scale': scale,
                    'query_slice': query_slice,
                    'query_scale': query_scale,
                    'query_date': query_date,
                    'target': target,
                    'query_raw': query,
                    'book_name': book_name,
                })
            else:
                train_val_series = series[:-prediction_length]
                tv_len = len(train_val_series)
                max_start = tv_len - context_length - prediction_length
                split_point = int((max_start + 1) * split_ratio)

                if mode == "train":
                    start_idx = 0
                    end_idx = split_point - 1
                else:
                    start_idx = split_point
                    end_idx = max_start

                for i in range(start_idx, end_idx + 1):
                    ctx = train_val_series[i: i + context_length]
                    target = train_val_series[i + context_length: i + context_length + prediction_length]
                    scale = np.mean(np.abs(ctx)) + 1.0
                    ctx_scaled = ctx / scale
                    target_scaled = target / scale

                    query_slice = ctx
                    query_scale = np.std(query_slice) + 1e-5
                    if len(query_slice) < retrieval_length:
                        query_slice = np.pad(query_slice, (retrieval_length - len(query_slice), 0), 'constant')
                    query_date = dates[i + context_length - 1] if i + context_length - 1 < len(dates) else dates[-1]

                    sample_info.append({
                        'is_inference': False,
                        'ctx_scaled': ctx_scaled,
                        'scale': scale,
                        'target_scaled': target_scaled,
                        'query_slice': query_slice,
                        'query_scale': query_scale,
                        'query_date': query_date,
                    })

        if retriever and retriever.index is not None and len(sample_info) > 0:
            query_matrix = np.stack([info['query_slice'] for info in sample_info])
            query_dates_arr = np.array([info['query_date'] for info in sample_info])
            query_scales_arr = np.array([info['query_scale'] for info in sample_info])
            print(f"Batch searching {len(query_matrix)} samples...")
            raf_contexts, raf_masks, _ = retriever.search_batch(
                query_matrix, query_dates_arr, query_scales_arr, k=top_k
            )
        else:
            raf_contexts = [np.full((top_k, retrieval_length), np.nan, dtype=np.float32) for _ in sample_info]
            raf_masks = [np.zeros((top_k, retrieval_length), dtype=bool) for _ in sample_info]

        for idx, info in enumerate(tqdm(sample_info, desc="Building Samples")):
            scale = info['scale']
            raf_ctx = raf_contexts[idx] / scale
            raf_msk = raf_masks[idx]

            raf_context_tensor = torch.tensor(raf_ctx, dtype=torch.float32)
            raf_mask_tensor = torch.tensor(raf_msk, dtype=torch.bool)
            raf_context_tensor = torch.nan_to_num(raf_context_tensor, nan=0.0)

            sample = {
                "context": torch.tensor(info['ctx_scaled'], dtype=torch.float32),
                "scale": scale,
                "raf_context": raf_context_tensor,
                "raf_mask": raf_mask_tensor,
            }

            if info['is_inference']:
                self.metadata.append({
                    "id": info['book_name'],
                    "target": info['target'],
                    "query": info['query_raw'],
                })
            else:
                sample["target"] = torch.tensor(info['target_scaled'], dtype=torch.float32)

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    result = {
        "context": torch.stack([b["context"] for b in batch]),
        "scale": torch.tensor([b["scale"] for b in batch], dtype=torch.float32),
        "raf_context": torch.stack([b["raf_context"] for b in batch]),
        "raf_mask": torch.stack([b["raf_mask"] for b in batch]),
    }
    if "target" in batch[0]:
        result["target"] = torch.stack([b["target"] for b in batch])
    return result


def pinball_loss(preds, targets, quantiles):
    total = 0.0
    for i, q in enumerate(quantiles):
        diff = targets - preds[:, i, :]
        total = total + torch.mean(torch.maximum(q * diff, (q - 1) * diff))
    return total / len(quantiles)


def train_model(df, retriever):
    print("\n=== Training LSTM+GateRAF ===")
    device = get_device()
    print(f"Device: {device}")

    common_args = {
        "df": df, "prediction_length": CONFIG["prediction_length"],
        "context_length": CONFIG["context_length"],
        "retriever": retriever, "top_k": CONFIG["top_k"],
    }
    train_ds = LSTMDataset(mode="train", **common_args)
    val_ds = LSTMDataset(mode="val", **common_args)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = LSTMForecaster(
        context_length=CONFIG["context_length"],
        prediction_length=CONFIG["prediction_length"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        num_quantiles=len(CONFIG["quantiles"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    total_steps = CONFIG["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    quantiles = CONFIG["quantiles"]

    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)
    best_val_loss = float("inf")
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False):
            ctx = batch["context"].to(device)
            tgt = batch["target"].to(device)
            raf_ctx = batch["raf_context"].to(device)
            raf_msk = batch["raf_mask"].to(device)
            preds = model(ctx, raf_context=raf_ctx, raf_mask=raf_msk)
            loss = pinball_loss(preds, tgt, quantiles)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context"].to(device)
                tgt = batch["target"].to(device)
                raf_ctx = batch["raf_context"].to(device)
                raf_msk = batch["raf_mask"].to(device)
                preds = model(ctx, raf_context=raf_ctx, raf_mask=raf_msk)
                val_loss += pinball_loss(preds, tgt, quantiles).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_loss_plot(train_losses, val_losses, CONFIG["checkpoint_path"].replace("best_model.pt", "loss_curve.png"))
    return CONFIG["checkpoint_path"]


def load_model(checkpoint_path=None):
    device = get_device()
    model = LSTMForecaster(
        context_length=CONFIG["context_length"],
        prediction_length=CONFIG["prediction_length"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        num_quantiles=len(CONFIG["quantiles"]),
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found. Using random weights.")
    return model


def run_inference(df, decile_books, all_predict, checkpoint_path, retriever):
    model = load_model(checkpoint_path)

    infer_ds = LSTMDataset(
        df=df, prediction_length=CONFIG["prediction_length"],
        context_length=CONFIG["context_length"],
        mode="inference", decile_books=decile_books, all_predict=all_predict,
        retriever=retriever, top_k=CONFIG["top_k"],
    )
    print(f"Prepared {len(infer_ds)} valid samples for inference.")

    print("\n=== Inference ===")
    device = next(model.parameters()).device
    loader = DataLoader(infer_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    model.eval()
    forecasts = []
    with torch.no_grad():
        for batch in tqdm(loader):
            ctx = batch["context"].to(device)
            scales = batch["scale"].to(device)
            raf_ctx = batch["raf_context"].to(device)
            raf_msk = batch["raf_mask"].to(device)
            preds = model(ctx, raf_context=raf_ctx, raf_mask=raf_msk)
            preds_unscaled = preds * scales.unsqueeze(1).unsqueeze(2)
            forecasts.extend(list(preds_unscaled.cpu()))
    return infer_ds.metadata, forecasts


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue", alpha=0.6)
    plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


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

        median = pred[1]
        low = pred[0]
        high = pred[2]
        ax.plot(x_pred, median, label="Forecast (q50)", color="blue")
        ax.fill_between(x_pred, low, high, color="blue", alpha=0.2, label="80% Interval (q10-q90)")
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
