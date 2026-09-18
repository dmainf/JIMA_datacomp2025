import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, PeftModel
from chronos.chronos_bolt import ChronosBoltModelForForecasting
import os
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "model_name": "amazon/chronos-bolt-base",
    "prediction_length": 64,
    "context_length": 128,
    "batch_size": 16,
    "output_dir": "chronos_bolt",
    "lora_output_dir": "chronos_bolt_checkpoints",
    "learning_rate": 1e-5,
    "grad_accum_steps": 4,
    "num_steps": 10000,
    "patience": 3,
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
    use_dora=True,
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


class ChronosBoltDataset(Dataset):
    def __init__(self, df, prediction_length, mode="train", split_ratio=0.9,
                 context_length=128, decile_books=None, all_predict=True):
        self.samples = []
        self.metadata = []

        for book_name, group in tqdm(df.groupby('書名', observed=True), desc="Processing Groups"):
            series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
            total_len = len(series)
            if total_len <= prediction_length + context_length:
                continue

            if mode == "inference":
                if not all_predict and book_name not in (decile_books or {}):
                    continue
                target = series[-prediction_length:]
                query = series[:-prediction_length]

                local_ctx = query[-context_length:] if len(query) >= context_length else query
                if len(local_ctx) < context_length:
                    local_ctx = np.concatenate([
                        np.full(context_length - len(local_ctx), np.nan, dtype=np.float32),
                        local_ctx
                    ])

                context_tensor = torch.tensor(local_ctx, dtype=torch.float32)
                mask_tensor = ~torch.isnan(context_tensor)
                context_tensor = torch.nan_to_num(context_tensor, nan=0.0)

                self.metadata.append({
                    "id": book_name,
                    "target": target,
                    "query": query,
                })
                self.samples.append({"context": context_tensor, "mask": mask_tensor})
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

                    context_tensor = torch.tensor(local_ctx, dtype=torch.float32)
                    mask_tensor = ~torch.isnan(context_tensor)
                    context_tensor = torch.nan_to_num(context_tensor, nan=0.0)

                    self.samples.append({
                        "context": context_tensor,
                        "mask": mask_tensor,
                        "target": torch.tensor(target, dtype=torch.float32),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


class ChronosBoltTrainer(Trainer):
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


def train_model(df):
    print("\n=== Training Model ===", flush=True)
    device, dtype = _get_device_dtype()
    print(f"Using device: {device}, dtype: {dtype}", flush=True)

    model = ChronosBoltModelForForecasting.from_pretrained(
        CONFIG["model_name"], torch_dtype=dtype
    ).to(device)
    model.chronos_config.context_length = CONFIG["context_length"]
    model.config.chronos_config["context_length"] = CONFIG["context_length"]
    model = get_peft_model(model, PEFT_CONFIG)

    common_args = {
        "df": df,
        "prediction_length": CONFIG["prediction_length"],
        "context_length": CONFIG["context_length"],
    }
    train_ds = ChronosBoltDataset(mode="train", **common_args)
    val_ds = ChronosBoltDataset(mode="val", **common_args)
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
    trainer = ChronosBoltTrainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["patience"])]
    )
    trainer.train()
    save_loss_plot(trainer.state.log_history, CONFIG["lora_output_dir"], "loss_curve.png")
    save_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    model.save_pretrained(save_path)
    return save_path


def load_model(adapter_path=None):
    device, dtype = _get_device_dtype()
    model = ChronosBoltModelForForecasting.from_pretrained(
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
        dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn
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
