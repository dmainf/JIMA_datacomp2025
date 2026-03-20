import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosBoltPipeline
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator
from torch.utils.data import Dataset, random_split
import os
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    "model_name": "amazon/chronos-bolt-base",
    "prediction_length": 64,
    "context_length": 180,
    "batch_size": 16,
    "output_dir": "chronos_bolt+FT",
    "lora_output_dir": "ch_bolt_dora_checkpoints",
    "learning_rate": 1e-4,
    "epochs": 3
}

PEFT_CONFIG = LoraConfig(
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    #target_modules=["q", "v"],
    target_modules="all-linear",
    use_dora=True
)

class ChronosBoltDataset(Dataset):
    def __init__(self, df, context_length, prediction_length, mode="train"):
        self.samples = []
        grouped = df.groupby('書名', observed=True)

        for _, group in grouped:
            if len(group) < context_length + prediction_length:
                continue

            group = group.sort_values('日付')
            series_data = group['POS販売冊数'].values.astype(np.float32)

            total_len = context_length + prediction_length

            if mode == "train":
                valid_end_idx = len(series_data) - prediction_length
            else:
                valid_end_idx = len(series_data)

            max_start_idx = valid_end_idx - total_len + 1
            if max_start_idx <= 0:
                continue

            for i in range(max_start_idx):
                window = series_data[i : i + total_len]
                context = window[:context_length]
                target = window[context_length:]

                scale = np.mean(np.abs(context)) + 1e-8
                scaled_context = context / scale
                scaled_target = target / scale

                self.samples.append({
                    "context": torch.tensor(scaled_context, dtype=torch.float32),
                    "target": torch.tensor(scaled_target, dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    contexts = torch.stack([item["context"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return {
        "context": contexts,
        "target": targets
    }

def extract_decile_books(df):
    print("\n=== Identifying Representative Books (5% intervals) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    quantiles = [i / 20 for i in range(21)]
    decile_books = {}
    print(f"{'Percentile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)
    num_books = len(total_sales)
    for q in quantiles:
        percent = int(round(q * 100))
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]
        label_name = f"{percent}%"
        file_prefix = f"{percent}%"
        decile_books[book_name] = {
            "label": label_name,
            "total_sales": sales_val,
            "file_prefix": file_prefix
        }
        print(f"{label_name:<15} | {sales_val:,.0f}冊       | {book_name}")
    return decile_books

def train_model(df):
    print("\n=== Starting DoRA Fine-Tuning (Chronos-Bolt SCALED) ===")

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    pipeline = ChronosBoltPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch_dtype
    )
    model = pipeline.model

    model = get_peft_model(model, PEFT_CONFIG)
    model.print_trainable_parameters()

    full_dataset = ChronosBoltDataset(
        df,
        context_length=CONFIG["context_length"],
        prediction_length=CONFIG["prediction_length"],
        mode="train"
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total samples: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=CONFIG["lora_output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_names=["target"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    trainer.train()

    final_adapter_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"DoRA adapter saved to: {final_adapter_path}")
    return final_adapter_path

def load_model(adapter_path=None):
    accelerator = Accelerator()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipeline = ChronosBoltPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=dtype
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading DoRA adapter from: {adapter_path}")
        pipeline.model = PeftModel.from_pretrained(pipeline.model, adapter_path)
    else:
        print("Using base model.")

    print(f"Using device: {accelerator.device}")
    return pipeline, accelerator

def preprocess_data(df, decile_books=None, all_predict=True):
    samples = []
    grouped = df.groupby('書名', observed=True)

    for book_name, group in grouped:
        if not all_predict and decile_books is not None and book_name not in decile_books:
            continue
        if len(group) < CONFIG["context_length"] + CONFIG["prediction_length"]:
            continue

        group = group.sort_values('日付')
        series_data = group['POS販売冊数'].values.astype(np.float32)

        total_len = CONFIG["context_length"] + CONFIG["prediction_length"]
        context = series_data[-total_len:-CONFIG["prediction_length"]]
        target = series_data[-CONFIG["prediction_length"]:]

        scale = np.mean(np.abs(context)) + 1e-8
        scaled_context = context / scale

        samples.append({
            "id": book_name,
            "context": torch.tensor(scaled_context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "raw_target": series_data[-CONFIG["prediction_length"]:],
            "scale": scale
        })
    return samples

def run_inference(pipeline, samples, accelerator):
    forecasts = []
    context_list = [s["context"] for s in samples]

    for i in tqdm(range(0, len(context_list), CONFIG["batch_size"])):
        batch_context = context_list[i:i + CONFIG["batch_size"]]

        batch_forecast = pipeline.predict(
            batch_context,
            prediction_length=CONFIG["prediction_length"],
            limit_prediction_length=False
        )
        for j in range(batch_forecast.shape[0]):
            forecasts.append(batch_forecast[j])

    return forecasts

def save_results(samples, forecasts, decile_books, all_predict):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if all_predict:
        os.makedirs(f"{CONFIG['output_dir']}/All_predict", exist_ok=True)
    else:
        print("\n=== Saving representative book predictions (decile_books) ===")

    saved_count = 0
    for i, sample in enumerate(samples):
        book_name = sample['id']
        scale = sample.get("scale", 1.0)

        quantile_forecast = forecasts[i].numpy() * scale
        context = sample["context"].numpy() * scale
        target = sample["target"].numpy()

        plt.figure(figsize=(10, 6))
        context_idx = np.arange(len(context))
        target_idx = np.arange(len(context), len(context) + len(target))
        plt.plot(context_idx, context, color="black", alpha=0.5, label="History")
        plt.plot(target_idx, target, color="gray", linestyle="--", label="Actual")

        median_idx = quantile_forecast.shape[0] // 2
        median = quantile_forecast[median_idx]

        if quantile_forecast.shape[0] >= 3:
            low_10 = quantile_forecast[0]
            high_90 = quantile_forecast[-1]
        else:
            low_10 = median
            high_90 = median

        plt.plot(target_idx, median, color="blue", label="Median Forecast")
        plt.fill_between(target_idx, low_10, high_90, color="blue", alpha=0.2, label="80% CI")

        if book_name in decile_books:
            info = decile_books[book_name]
            label = info["label"]
            total_sales = info["total_sales"]
            plt.title(f"[{label}] {book_name}\nTotal Sales: {total_sales:,.0f} books")
        else:
            plt.title(f"Forecast: {book_name}")

        plt.legend()
        plt.grid(True, alpha=0.3)
        safe_name = str(book_name).replace("/", "_").replace("\\", "_")

        if book_name in decile_books:
            prefix = decile_books[book_name]["file_prefix"]
            filename = f"{prefix}_{safe_name}.png"
            save_path = f"{CONFIG['output_dir']}/{filename}"
            plt.savefig(save_path)
            saved_count += 1

        if all_predict:
            filename = f"{safe_name}.png"
            save_path = f"{CONFIG['output_dir']}/All_predict/{filename}"
            plt.savefig(save_path)
            saved_count += 1

        plt.close()
        if (saved_count) % 100 == 0:
            print(f"Saved {saved_count} predictions...")
    print(f"Total saved: {saved_count} predictions")

def evaluate_predictions(samples, forecasts, config, all_predict):
    print("\n=== Calculating Evaluation Metrics ===")
    total_metrics = {"mae": 0.0, "mse": 0.0, "wql_10": 0.0, "wql_50": 0.0, "wql_90": 0.0, "coverage_80": 0.0, "total_sales_sum": 0.0, "count_points": 0}
    item_results = []

    for i, sample in enumerate(samples):
        book_name = sample['id']
        scale = sample.get("scale", 1.0)
        quantile_forecast = forecasts[i].numpy() * scale
        target = sample["target"].numpy()

        median_idx = quantile_forecast.shape[0] // 2
        q50 = quantile_forecast[median_idx]

        if quantile_forecast.shape[0] >= 3:
            q10 = quantile_forecast[0]
            q90 = quantile_forecast[-1]
        else:
            q10 = q50
            q90 = q50

        ae = np.abs(target - q50)
        se = (target - q50) ** 2

        def pinball_loss(y_true, y_pred, quantile):
            diff = y_true - y_pred
            return np.maximum(quantile * diff, (quantile - 1) * diff)

        loss_10 = pinball_loss(target, q10, 0.1)
        loss_50 = pinball_loss(target, q50, 0.5)
        loss_90 = pinball_loss(target, q90, 0.9)
        in_interval = ((target >= q10) & (target <= q90)).astype(float)

        total_metrics["mae"] += np.sum(ae)
        total_metrics["mse"] += np.sum(se)
        total_metrics["wql_10"] += np.sum(loss_10)
        total_metrics["wql_50"] += np.sum(loss_50)
        total_metrics["wql_90"] += np.sum(loss_90)
        total_metrics["coverage_80"] += np.sum(in_interval)
        total_metrics["total_sales_sum"] += np.sum(np.abs(target))
        total_metrics["count_points"] += len(target)

        item_wql_10 = 2 * np.sum(loss_10)
        item_wql_50 = 2 * np.sum(loss_50)
        item_wql_90 = 2 * np.sum(loss_90)
        item_results.append({
            "book_name": book_name,
            "MAE": np.mean(ae),
            "RMSE": np.sqrt(np.mean(se)),
            "wQL_0.1": item_wql_10,
            "wQL_0.5": item_wql_50,
            "wQL_0.9": item_wql_90,
            "wQL_Mean": (item_wql_10 + item_wql_50 + item_wql_90) / 3,
            "Coverage_80%": np.mean(in_interval),
            "Total_Sales": np.sum(target)
        })

    final_metrics = {}
    final_metrics["MAE"] = total_metrics["mae"] / total_metrics["count_points"]
    final_metrics["RMSE"] = np.sqrt(total_metrics["mse"] / total_metrics["count_points"])

    final_metrics["wQL_0.1"] = 2 * total_metrics["wql_10"] / total_metrics["total_sales_sum"]
    final_metrics["wQL_0.5"] = 2 * total_metrics["wql_50"] / total_metrics["total_sales_sum"]
    final_metrics["wQL_0.9"] = 2 * total_metrics["wql_90"] / total_metrics["total_sales_sum"]
    final_metrics["wQL_Mean"] = (final_metrics["wQL_0.1"] + final_metrics["wQL_0.5"] + final_metrics["wQL_0.9"]) / 3
    final_metrics["Coverage_80%"] = total_metrics["coverage_80"] / total_metrics["count_points"]

    print("-" * 40)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 40)

    eval_df = pd.DataFrame(item_results)
    eval_filename = "evaluation_all.csv" if all_predict else "evaluation_decile.csv"
    eval_csv_path = os.path.join(config["output_dir"], eval_filename)
    eval_df.to_csv(eval_csv_path, index=False)
