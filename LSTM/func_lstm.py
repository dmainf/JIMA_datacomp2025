import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "prediction_length": 64,
    "context_length": 128,
    "batch_size": 64,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "epochs": 50,
    "patience": 3,
    "output_dir": "LSTM",
    "checkpoint_path": "lstm_checkpoints/best_model.pt",
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


class LSTMDataset(Dataset):
    def __init__(self, df, prediction_length, context_length, mode="train",
                 split_ratio=0.9, decile_books=None, all_predict=True):
        self.samples = []
        self.metadata = []

        for book_name, group in tqdm(df.groupby('書名', observed=True), desc=f"Processing ({mode})"):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
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

                local_ctx = query[-context_length:]
                scale = np.mean(np.abs(local_ctx)) + 1.0
                ctx_scaled = local_ctx / scale

                self.samples.append({
                    "context": torch.tensor(ctx_scaled, dtype=torch.float32),
                    "scale": scale,
                })
                self.metadata.append({
                    "id": book_name,
                    "target": target,
                    "query": query,
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
                    self.samples.append({
                        "context": torch.tensor(ctx_scaled, dtype=torch.float32),
                        "target": torch.tensor(target_scaled, dtype=torch.float32),
                        "scale": scale,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    result = {
        "context": torch.stack([b["context"] for b in batch]),
        "scale": torch.tensor([b["scale"] for b in batch], dtype=torch.float32),
    }
    if "target" in batch[0]:
        result["target"] = torch.stack([b["target"] for b in batch])
    return result


class LSTMForecaster(nn.Module):
    def __init__(self, context_length, prediction_length, hidden_size, num_layers, dropout, num_quantiles=3):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_length * num_quantiles),
        )

    def forward(self, context):
        x = context.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1])
        return out.view(-1, self.num_quantiles, self.prediction_length)


def pinball_loss(preds, targets, quantiles):
    total = 0.0
    for i, q in enumerate(quantiles):
        diff = targets - preds[:, i, :]
        total = total + torch.mean(torch.maximum(q * diff, (q - 1) * diff))
    return total / len(quantiles)


def train_model(df):
    print("\n=== Training LSTM ===")
    device = get_device()
    print(f"Device: {device}")

    train_ds = LSTMDataset(df, CONFIG["prediction_length"], CONFIG["context_length"], mode="train")
    val_ds = LSTMDataset(df, CONFIG["prediction_length"], CONFIG["context_length"], mode="val")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
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
            preds = model(ctx)
            loss = pinball_loss(preds, tgt, quantiles)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context"].to(device)
                tgt = batch["target"].to(device)
                preds = model(ctx)
                val_loss += pinball_loss(preds, tgt, quantiles).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

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


def run_inference(df, decile_books, all_predict, checkpoint_path):
    model = load_model(checkpoint_path)

    infer_ds = LSTMDataset(
        df=df, prediction_length=CONFIG["prediction_length"],
        context_length=CONFIG["context_length"],
        mode="inference", decile_books=decile_books, all_predict=all_predict
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
            preds = model(ctx)
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
