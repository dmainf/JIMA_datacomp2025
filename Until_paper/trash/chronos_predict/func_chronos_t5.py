import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
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
    "model_name": "amazon/chronos-t5-small",
    "prediction_length": 64,
    "context_length": 180,
    "batch_size": 16,
    "num_samples": 20,
    "output_dir": "chronos_t5+FT",
    "lora_output_dir": "ch_lora_checkpoints",
    "learning_rate": 1e-4,
    "epochs": 3
}

PEFT_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "k", "v", "o", "wi", "wo"]
)

class ChronosDataset(Dataset):
    def __init__(self, df, tokenizer, context_length, prediction_length, mode="train"):
        self.samples = []
        self.tokenizer = tokenizer

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

                self.samples.append((torch.tensor(context), torch.tensor(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]

        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context.unsqueeze(0))
        labels, _ = self.tokenizer.label_input_transform(target.unsqueeze(0), scale)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
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
    print("\n=== Starting LoRA Fine-Tuning (Baseline) ===")

    pipeline = ChronosPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    tokenizer = pipeline.tokenizer
    model = pipeline.inner_model

    model = get_peft_model(model, PEFT_CONFIG)
    model.print_trainable_parameters()

    full_dataset = ChronosDataset(
        df,
        tokenizer,
        context_length=CONFIG["context_length"],
        prediction_length=CONFIG["prediction_length"],
        mode="train"
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total samples: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = Seq2SeqTrainingArguments(
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
        metric_for_best_model="eval_loss"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None
    )

    trainer.train()

    final_adapter_path = os.path.join(CONFIG["lora_output_dir"], "final_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"LoRA adapter saved to: {final_adapter_path}")
    return final_adapter_path

def load_model(adapter_path=None):
    accelerator = Accelerator()

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    base_pipeline = ChronosPipeline.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        torch_dtype=torch_dtype
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        inner_model = PeftModel.from_pretrained(base_pipeline.inner_model, adapter_path)
        base_pipeline.model.model = inner_model
        base_pipeline.inner_model = inner_model
    else:
        print("Using base model.")

    print(f"Using device: {accelerator.device}")
    return base_pipeline, accelerator

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

        samples.append({
            "id": book_name,
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
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
            num_samples=CONFIG["num_samples"],
            limit_prediction_length=False
        )
        forecasts.extend(batch_forecast)

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

        forecast = forecasts[i].numpy()
        context = sample["context"].numpy()
        target = sample["target"].numpy()

        plt.figure(figsize=(10, 6))

        context_idx = np.arange(len(context))
        target_idx = np.arange(len(context), len(context) + len(target))

        plt.plot(context_idx, context, color="black", alpha=0.5, label="History")
        plt.plot(target_idx, target, color="gray", linestyle="--", label="Actual")

        median = np.median(forecast, axis=0)
        low_10 = np.quantile(forecast, 0.1, axis=0)
        high_90 = np.quantile(forecast, 0.9, axis=0)

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

    total_metrics = {
        "mae": 0.0,
        "mse": 0.0,
        "wql_10": 0.0,
        "wql_50": 0.0,
        "wql_90": 0.0,
        "coverage_80": 0.0,
        "total_sales_sum": 0.0,
        "count_points": 0
    }

    item_results = []

    for i, sample in enumerate(samples):
        book_name = sample['id']

        forecast_samples = forecasts[i].numpy()
        target = sample["target"].numpy()

        q10 = np.quantile(forecast_samples, 0.1, axis=0)
        q50 = np.quantile(forecast_samples, 0.5, axis=0)
        q90 = np.quantile(forecast_samples, 0.9, axis=0)

        ae = np.abs(target - q50)

        se = (target - q50) ** 2

        def pinball_loss(y_true, y_pred, quantile):
            diff = y_true - y_pred
            return np.maximum(quantile * diff, (quantile - 1) * diff)

        loss_10 = pinball_loss(target, q10, 0.1)
        loss_50 = pinball_loss(target, q50, 0.5)
        loss_90 = pinball_loss(target, q90, 0.9)

        in_interval = ((target >= q10) & (target <= q90)).astype(float)

        pred_len = len(target)
        item_sales_sum = np.sum(np.abs(target))

        total_metrics["mae"] += np.sum(ae)
        total_metrics["mse"] += np.sum(se)
        total_metrics["wql_10"] += np.sum(loss_10)
        total_metrics["wql_50"] += np.sum(loss_50)
        total_metrics["wql_90"] += np.sum(loss_90)
        total_metrics["coverage_80"] += np.sum(in_interval)
        total_metrics["total_sales_sum"] += item_sales_sum
        total_metrics["count_points"] += pred_len

        item_mae = np.mean(ae)
        item_rmse = np.sqrt(np.mean(se))
        item_wql_10 = 2 * np.sum(loss_10)
        item_wql_50 = 2 * np.sum(loss_50)
        item_wql_90 = 2 * np.sum(loss_90)
        item_wql_mean = (item_wql_10 + item_wql_50 + item_wql_90) / 3
        item_coverage = np.mean(in_interval)

        item_results.append({
            "book_name": book_name,
            "MAE": item_mae,
            "RMSE": item_rmse,
            "wQL_0.1": item_wql_10,
            "wQL_0.5": item_wql_50,
            "wQL_0.9": item_wql_90,
            "wQL_Mean": item_wql_mean,
            "Coverage_80%": item_coverage,
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
    print(f"\nBook-level evaluation saved to: {eval_csv_path}")
