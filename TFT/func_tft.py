import pandas as pd
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "prediction_length": 64,
    "context_length": 128,
    "batch_size": 64,
    "hidden_size": 128,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 64,
    "learning_rate": 1e-3,
    "epochs": 50,
    "patience": 3,
    "output_dir": "TFT",
    "checkpoint_dir": "tft_checkpoints",
    "quantiles": [0.1, 0.5, 0.9],
}


def get_device():
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _build_tv_df(df):
    pred_len = CONFIG["prediction_length"]
    ctx_len = CONFIG["context_length"]
    rows = []
    for book_name, group in tqdm(df.groupby('書名', observed=True), desc="Preparing train/val data"):
        series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
        if len(series) <= pred_len + ctx_len:
            continue
        tv = series[:-pred_len]
        for t, val in enumerate(tv):
            rows.append({"series_id": str(book_name), "time_idx": t, "value": float(val)})
    return pd.DataFrame(rows)


def _make_training_dataset(tv_df):
    return TimeSeriesDataSet(
        tv_df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        min_encoder_length=CONFIG["context_length"],
        max_encoder_length=CONFIG["context_length"],
        min_prediction_length=CONFIG["prediction_length"],
        max_prediction_length=CONFIG["prediction_length"],
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


class LossHistory(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss_epoch", trainer.callback_metrics.get("train_loss"))
        if v is not None:
            self.train_losses.append(float(v))

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))


def train_model(df):
    print("\n=== Training TFT ===")
    device = get_device()
    print(f"Device: {device}")

    tv_df = _build_tv_df(df)
    training_ds = _make_training_dataset(tv_df)
    val_ds = TimeSeriesDataSet.from_dataset(training_ds, tv_df, predict=True, stop_randomization=True)
    print(f"Train: {len(training_ds)}, Val: {len(val_ds)}")

    train_loader = training_ds.to_dataloader(train=True, batch_size=CONFIG["batch_size"], num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=CONFIG["batch_size"], num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=CONFIG["learning_rate"],
        hidden_size=CONFIG["hidden_size"],
        attention_head_size=CONFIG["attention_head_size"],
        dropout=CONFIG["dropout"],
        hidden_continuous_size=CONFIG["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=CONFIG["quantiles"]),
        log_interval=-1,
        reduce_on_plateau_patience=2,
    )

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    loss_history = LossHistory()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], mode="min"),
        ModelCheckpoint(
            dirpath=CONFIG["checkpoint_dir"],
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        loss_history,
    ]

    accelerator = device if device in ("gpu", "mps") else "cpu"
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    checkpoint_path = trainer.checkpoint_callback.best_model_path
    save_loss_plot(
        loss_history.train_losses, loss_history.val_losses,
        os.path.join(CONFIG["checkpoint_dir"], "loss_curve.png")
    )
    return checkpoint_path


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


def run_inference(df, decile_books, all_predict, checkpoint_path):
    pred_len = CONFIG["prediction_length"]
    ctx_len = CONFIG["context_length"]

    tv_df = _build_tv_df(df)
    training_ds = _make_training_dataset(tv_df)

    metadata = []
    infer_rows = []
    for book_name, group in tqdm(df.groupby('書名', observed=True), desc="Building inference data"):
        if not all_predict and book_name not in (decile_books or {}):
            continue
        series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
        if len(series) <= pred_len + ctx_len:
            continue
        metadata.append({
            "id": book_name,
            "target": series[-pred_len:],
            "query": series[:-pred_len],
        })
        sub = series[-(ctx_len + pred_len):]
        for t, val in enumerate(sub):
            infer_rows.append({"series_id": str(book_name), "time_idx": t, "value": float(val)})

    infer_df = pd.DataFrame(infer_rows)
    infer_ds = TimeSeriesDataSet.from_dataset(training_ds, infer_df, predict=True, stop_randomization=True)
    print(f"Prepared {len(infer_ds)} valid samples for inference.")

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    print("\n=== Inference ===")
    infer_loader = infer_ds.to_dataloader(train=False, batch_size=CONFIG["batch_size"], num_workers=0)
    raw_preds = model.predict(infer_loader, mode="quantiles", return_x=False)
    # raw_preds: (n_samples, pred_len, n_quantiles) → (n_samples, n_quantiles, pred_len)
    raw_preds = raw_preds.permute(0, 2, 1)
    forecasts = [raw_preds[i] for i in range(raw_preds.shape[0])]

    return metadata, forecasts


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
