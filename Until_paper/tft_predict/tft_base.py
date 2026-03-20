import pandas as pd
import numpy as np
import math
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import warnings
import logging
import torch
from torch.nn import functional as F
from gluonts.model.forecast_generator import SampleForecastGenerator

import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
import os
import shutil

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("gluonts").setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

PREDICTION_LENGTH = 45
CONTEXT_LENGTH = 180
EPOC = 125
USE_LOG_SCALE = True #True

def plot_training_history(log_dir='lightning_logs', save_path='figure/training_history.png'):
    metrics_file = None
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]
    if not version_dirs:
        print("No lightning_logs version directory found.")
        return

    latest_version = sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1]
    metrics_file = os.path.join(log_dir, latest_version, 'metrics.csv')

    if not os.path.exists(metrics_file):
        print("No metrics file found in the latest version.")
        return

    print(f"Reading metrics from: {metrics_file}")
    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(12, 6))

    if 'train_loss' in df.columns:
        train_df = df[['epoch', 'train_loss']].dropna()
        if not train_df.empty:
            plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss', marker='o', linewidth=2)

    if 'val_loss' in df.columns:
        val_df = df[['epoch', 'val_loss']].dropna()
        if not val_df.empty:
            plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss', marker='s', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('学習履歴', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"学習履歴のプロットを {save_path} に保存しました。")


def main():
    os.makedirs('figure', exist_ok=True)
    os.makedirs('figure/predict', exist_ok=True)

    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')

    print("\n=== Identifying Representative Books (Quantiles) ===")
    total_sales = df.groupby('書名')['POS販売冊数'].sum().sort_values()
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    quantile_names = ["Min (0%)", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max (100%)"]
    target_books = {}

    print(f"{'Quantile':<15} | {'Total Sales':<12} | {'Book Name'}")
    print("-" * 60)

    num_books = len(total_sales)
    for q, name in zip(quantiles, quantile_names):
        idx = int((num_books - 1) * q)
        book_name = total_sales.index[idx]
        sales_val = total_sales.iloc[idx]

        target_books[book_name] = {
            "label": name,
            "total_sales": sales_val
        }
        print(f"{name:<15} | {sales_val:,.0f}冊       | {book_name}")

    if USE_LOG_SCALE:
        df['POS販売冊数'] = np.log1p(df['POS販売冊数']).astype(np.float32)
        print("\n=== Applied log1p transformation ===")
    else:
        df['POS販売冊数'] = df['POS販売冊数'].astype(np.float32)
        print("\n=== No scaling applied ===")

    df['id'] = df['書名'].astype(str)

    print("\n=== POS Sales Statistics ===")
    print(f"Mean: {df['POS販売冊数'].mean():.4f}")
    print(f"Variance: {df['POS販売冊数'].var():.4f}")
    print(f"Std Dev: {df['POS販売冊数'].std():.4f}")

    static_cols = ['出版社', '著者名', '大分類', '中分類', '小分類']

    for col in static_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.remove_unused_categories()

    df_static = df.groupby('id', observed=False)[static_cols].first().reset_index()

    cardinality = [df_static[col].cat.categories.size for col in static_cols]
    print(f"Cardinality: {dict(zip(static_cols, cardinality))}")

    df_static_formatted = df_static[['id'] + static_cols].set_index('id')

    dataset = PandasDataset.from_long_dataframe(
        df,
        target='POS販売冊数',
        item_id='id',
        timestamp='日付',
        freq='D',
        static_features=df_static_formatted
    )

    full_dataset = list(dataset)
    print(f"Number of time series: {len(full_dataset)}")

    print("\n=== Creating Training-Specific Dataset (No Data Leakage) ===")
    train_dataset = []
    for entry in full_dataset:
        train_entry = entry.copy()
        train_entry["target"] = entry["target"][:-PREDICTION_LENGTH]
        train_dataset.append(train_entry)
    print("Training dataset created by removing prediction length from the end of each series.")

    csv_logger = CSVLogger("lightning_logs", name="tft_training")

    estimator = TemporalFusionTransformerEstimator(
        freq="D",
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        static_cardinalities=cardinality,
        batch_size=64,
        trainer_kwargs={
            "max_epochs": EPOC,
            "accelerator": "auto",
            "logger": csv_logger,
            "enable_progress_bar": True,
            "enable_model_summary": False
        }
    )

    print("\n=== Training ===")
    predictor = estimator.train(training_data=train_dataset)

    print("\n=== Plotting Training History ===")
    plot_training_history('lightning_logs/tft_training', 'figure/training_history.png')

    print("\n=== Cleaning up logs ===")
    if os.path.exists('lightning_logs/tft_training'):
        version_dirs = [d for d in os.listdir('lightning_logs/tft_training') if d.startswith('version_') and os.path.isdir(os.path.join('lightning_logs/tft_training', d))]
        if version_dirs:
            latest_version = sorted(version_dirs, key=lambda x: int(x.split('_')[1]))[-1]
            latest_version_dir_path = os.path.join('lightning_logs/tft_training', latest_version)
            for item in version_dirs:
                item_path = os.path.join('lightning_logs/tft_training', item)
                if os.path.isdir(item_path) and item_path != latest_version_dir_path:
                    shutil.rmtree(item_path)
            print("Older lightning_logs versions removed")
        else:
            print("No versioned lightning_logs directories found to clean.")
    else:
        print("lightning_logs/tft_training directory not found, no cleanup performed.")

    print("\n=== Evaluating ===")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=full_dataset,
        predictor=predictor,
        num_samples=100
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    print("\n=== Metrics ===")
    for key, value in agg_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Visualizing Representative Predictions ===")

    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id

        if book_id in target_books:
            info = target_books[book_id]
            label = info["label"]
            total_sales = info["total_sales"]

            print(f"Plotting {label}: {book_id} (Total: {total_sales})")

            actual_data_log = tss[i]

            start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
            forecast_dates = start_timestamp + pd.to_timedelta(np.arange(PREDICTION_LENGTH), unit='D')
            history_dates = start_timestamp + pd.to_timedelta(np.arange(-CONTEXT_LENGTH, 0), unit='D')

            plt.figure(figsize=(12, 6))

            if USE_LOG_SCALE:
                history_actual = np.expm1(actual_data_log[-CONTEXT_LENGTH:])
                target_actual = np.expm1(actual_data_log[-PREDICTION_LENGTH:])
            else:
                history_actual = actual_data_log[-CONTEXT_LENGTH:]
                target_actual = actual_data_log[-PREDICTION_LENGTH:]

            plt.plot(history_dates, history_actual,
                     label="Actual History", color='black', linestyle='--')
            plt.plot(forecast_dates, target_actual,
                     label="Actual Target", color='black', marker='o', markersize=3)

            if USE_LOG_SCALE:
                q10_sales = np.expm1(forecast.quantile(0.1))
                q90_sales = np.expm1(forecast.quantile(0.9))
                q25_sales = np.expm1(forecast.quantile(0.25))
                q75_sales = np.expm1(forecast.quantile(0.75))
                median_sales = np.expm1(forecast.quantile(0.5))
            else:
                q10_sales = forecast.quantile(0.1)
                q90_sales = forecast.quantile(0.9)
                q25_sales = forecast.quantile(0.25)
                q75_sales = forecast.quantile(0.75)
                median_sales = forecast.quantile(0.5)

            plt.fill_between(forecast_dates, q10_sales, q90_sales,
                             color='blue', alpha=0.15, label="80% Interval (10-90%)")
            plt.fill_between(forecast_dates, q25_sales, q75_sales,
                             color='cyan', alpha=0.5, label="IQR (25-75%)")

            plt.plot(forecast_dates, median_sales, label="Median (50%)", color='blue', linewidth=2)

            plt.title(f"[{label}] {book_id}\nTotal Sales: {total_sales:,.0f} books", fontsize=14)
            plt.xlabel("日付")
            if USE_LOG_SCALE:
                plt.ylabel("POS販売冊数 (expm1)")
            else:
                plt.ylabel("POS販売冊数")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()

            safe_label = label.replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
            save_name = f"figure/forecast_{safe_label}_{book_id}.png"
            plt.savefig(save_name)
            plt.close()

    print("\n=== Saving All Book Predictions to figure/predict ===")
    for i, forecast in enumerate(forecasts):
        book_id = forecast.item_id
        actual_data_log = tss[i]

        start_timestamp = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else pd.Timestamp(forecast.start_date)
        forecast_dates = start_timestamp + pd.to_timedelta(np.arange(PREDICTION_LENGTH), unit='D')
        history_dates = start_timestamp + pd.to_timedelta(np.arange(-CONTEXT_LENGTH, 0), unit='D')

        plt.figure(figsize=(12, 6))

        if USE_LOG_SCALE:
            history_actual = np.expm1(actual_data_log[-CONTEXT_LENGTH:])
            target_actual = np.expm1(actual_data_log[-PREDICTION_LENGTH:])
        else:
            history_actual = actual_data_log[-CONTEXT_LENGTH:]
            target_actual = actual_data_log[-PREDICTION_LENGTH:]

        plt.plot(history_dates, history_actual,
                 label="Actual History", color='black', linestyle='--')
        plt.plot(forecast_dates, target_actual,
                 label="Actual Target", color='black', marker='o', markersize=3)

        if USE_LOG_SCALE:
            q10_sales = np.expm1(forecast.quantile(0.1))
            q90_sales = np.expm1(forecast.quantile(0.9))
            q25_sales = np.expm1(forecast.quantile(0.25))
            q75_sales = np.expm1(forecast.quantile(0.75))
            median_sales = np.expm1(forecast.quantile(0.5))
        else:
            q10_sales = forecast.quantile(0.1)
            q90_sales = forecast.quantile(0.9)
            q25_sales = forecast.quantile(0.25)
            q75_sales = forecast.quantile(0.75)
            median_sales = forecast.quantile(0.5)

        plt.fill_between(forecast_dates, q10_sales, q90_sales,
                         color='blue', alpha=0.15, label="80% Interval (10-90%)")
        plt.fill_between(forecast_dates, q25_sales, q75_sales,
                         color='cyan', alpha=0.5, label="IQR (25-75%)")

        plt.plot(forecast_dates, median_sales, label="Median (50%)", color='blue', linewidth=2)

        plt.title(f"{book_id}", fontsize=14)
        plt.xlabel("日付")
        if USE_LOG_SCALE:
            plt.ylabel("POS販売冊数 (expm1)")
        else:
            plt.ylabel("POS販売冊数")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_book_id = book_id.replace("/", "_").replace("\\", "_")
        save_name = f"figure/predict/forecast_{safe_book_id}.png"
        plt.savefig(save_name)
        plt.close()

        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1}/{len(forecasts)} predictions")

    print(f"All {len(forecasts)} prediction plots saved to figure/predict/")

    print("\nAll plots saved. Process Complete.")

if __name__ == "__main__":
    main()



"""
class Tweedie(Distribution):
    arg_constraints = {}

    def __init__(self, mu, rho, validate_args=None):
        if mu.dim() > 2 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
        self.mu = mu
        self.rho = rho
        super().__init__(batch_shape=mu.shape, validate_args=validate_args)

    def log_prob(self, value):
        loss = F.tweedie_loss(self.mu, value, p=self.rho, reduction='none')
        return -loss

    def sample(self, sample_shape=torch.Size()):
        extended_shape = sample_shape + self.mu.shape
        mu_expanded = self.mu.expand(extended_shape)

        phi = 1.0
        p = self.rho
        mu_safe = torch.clamp(mu_expanded, min=1e-8)

        lambda_val = (mu_safe ** (2 - p)) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        beta = phi * (p - 1) * (mu_safe ** (p - 1))

        n_samples = torch.poisson(lambda_val)

        gamma_samples = torch.zeros_like(n_samples)
        mask = n_samples > 0

        if mask.any():
            valid_n = n_samples[mask]
            m = torch.distributions.Gamma(concentration=valid_n * alpha, rate=1.0/beta[mask])
            gamma_samples[mask] = m.sample()
        return gamma_samples

    @property
    def mean(self):
        return self.mu

class LightGBMTweedieOutput(Output):
    args_dim = {"mu": 1}

    def __init__(self, rho: float = 1.5):
        assert 1.0 < rho < 2.0, "rho (p) must be between 1.0 and 2.0"
        self.rho = rho

    def domain_map(self, mu):
        # 平均値は正である必要がある
        return F.softplus(mu)

    def distribution(self, distr_args, loc=None, scale=None):
        return LightGBMTweedie(distr_args, rho=self.rho)

    def loss(self, target, distr_args, loc=None, scale=None):
        distr = self.distribution(distr_args, loc=loc, scale=scale)
        return -distr.log_prob(target)

    @property
    def event_shape(self):
        return ()

    @property
    def forecast_generator(self):
        # サンプリングベースの予測生成器を指定
        return SampleForecastGenerator()
"""