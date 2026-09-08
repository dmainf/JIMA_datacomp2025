import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pmdarima as pm
from joblib import Parallel, delayed
import torch
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "prediction_length": 64,
    "context_length": 128,
    "output_dir": "ARIMA",
    "n_jobs": -1,
    "alpha": 0.2,
}


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


def fit_predict_arima(book_name, series, prediction_length, context_length):
    if len(series) <= prediction_length + context_length:
        return None

    query = series[:-prediction_length]
    target = series[-prediction_length:]
    train = query[-context_length:]

    scale = np.mean(np.abs(train)) + 1e-6
    train_scaled = train / scale

    try:
        model = pm.auto_arima(
            train_scaled,
            seasonal=False,
            d=None, test='kpss',
            start_p=0, start_q=0, max_p=5, max_q=5,
            information_criterion='aicc',
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        preds_scaled, conf_int_scaled = model.predict(
            n_periods=prediction_length, return_conf_int=True, alpha=CONFIG["alpha"]
        )
        preds = preds_scaled * scale
        conf_int = conf_int_scaled * scale

        q10 = conf_int[:, 0]
        q50 = preds
        q90 = conf_int[:, 1]
        forecast_quantiles = np.vstack([q10, q50, q90])

        return {
            "id": book_name,
            "target": target,
            "query": query,
            "forecast": forecast_quantiles
        }
    except Exception:
        return None


def run_arima(df, decile_books, all_predict):
    prediction_length = CONFIG["prediction_length"]
    context_length = CONFIG["context_length"]

    tasks = []
    for book_name in df['書名'].unique():
        if not all_predict and book_name not in (decile_books or {}):
            continue
        group = df[df['書名'] == book_name]
        series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
        if len(series) > prediction_length + context_length:
            tasks.append((book_name, series))

    print(f"Starting ARIMA for {len(tasks)} series (n_jobs={CONFIG['n_jobs']})...")
    results = Parallel(n_jobs=CONFIG["n_jobs"])(
        delayed(fit_predict_arima)(name, series, prediction_length, context_length)
        for name, series in tqdm(tasks)
    )

    valid_results = [r for r in results if r is not None]
    print(f"Valid predictions: {len(valid_results)} / {len(tasks)}")

    samples = []
    forecasts = []
    for res in valid_results:
        samples.append({"id": res["id"], "target": res["target"], "query": res["query"]})
        forecasts.append(torch.tensor(res["forecast"], dtype=torch.float32))

    return samples, forecasts


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
