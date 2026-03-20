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
from tslearn.metrics import dtw_path
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "prediction_length": 64,
    "seasonal_period": 7,
    "output_dir": "sarima_baseline",
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


def fit_predict_sarima(book_name, series, prediction_length, m):
    if len(series) <= prediction_length:
        return None

    train = series[:-prediction_length]
    target = series[-prediction_length:]

    scale = np.mean(np.abs(train)) + 1e-6
    train_scaled = train / scale

    try:
        model = pm.auto_arima(
            train_scaled,
            seasonal=True, m=m,
            d=None, test='adf',
            start_p=0, start_q=0, max_p=3, max_q=3,
            start_P=0, start_Q=0, max_P=2, max_Q=2,
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
            "query": train,
            "forecast": forecast_quantiles
        }
    except Exception:
        return None


def run_sarima(df, all_predict):
    prediction_length = CONFIG["prediction_length"]

    tasks = []
    for book_name in df['書名'].unique():
        group = df[df['書名'] == book_name]
        series = group.sort_values('日付')['POS販売冊数'].values.astype(np.float32)
        if len(series) > prediction_length:
            tasks.append((book_name, series))

    print(f"Starting SARIMA for {len(tasks)} series (n_jobs={CONFIG['n_jobs']})...")
    results = Parallel(n_jobs=CONFIG["n_jobs"])(
        delayed(fit_predict_sarima)(name, series, prediction_length, CONFIG["seasonal_period"])
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


def _pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.sum(np.maximum(q * diff, (q - 1) * diff))


def _protrusion_area_batch(y_true, y_lower, y_upper, dt=1.0):
    upper_protrusion = np.maximum(0, y_true - y_upper)
    lower_protrusion = np.maximum(0, y_lower - y_true)
    total_protrusion = upper_protrusion + lower_protrusion
    return np.trapz(total_protrusion, dx=dt, axis=1)


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

    item_protrusion_area = _protrusion_area_batch(targets, q10, q90)

    diff_10 = targets - q10
    diff_50 = targets - q50
    diff_90 = targets - q90
    item_wql_10 = 2 * np.sum(np.maximum(0.1 * diff_10, -0.9 * diff_10), axis=1)
    item_wql_50 = 2 * np.sum(np.maximum(0.5 * diff_50, -0.5 * diff_50), axis=1)
    item_wql_90 = 2 * np.sum(np.maximum(0.9 * diff_90, -0.1 * diff_90), axis=1)

    item_shape_err = []
    item_time_err = []
    for i in range(len(targets)):
        path, dtw_score = dtw_path(targets[i], q50[i])
        item_shape_err.append(dtw_score)
        item_time_err.append(np.mean([abs(idx_t - idx_p) for idx_t, idx_p in path]))
    item_shape_err = np.array(item_shape_err)
    item_time_err = np.array(item_time_err)

    df_res = pd.DataFrame({
        "book_name": book_names,
        "MAE": item_mae,
        "RMSE": item_rmse,
        "wQL_0.1": item_wql_10,
        "wQL_0.5": item_wql_50,
        "wQL_0.9": item_wql_90,
        "wQL_Mean": (item_wql_10 + item_wql_50 + item_wql_90) / 3,
        "Coverage_80%": item_coverage,
        "Protrusion_Area": item_protrusion_area,
        "DTW_Shape": item_shape_err,
        "DTW_Time": item_time_err,
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
    mean_shape = np.mean(item_shape_err)
    mean_time = np.mean(item_time_err)
    mean_protrusion = np.mean(item_protrusion_area)
    final_metrics = {
        "MAE": mae, "RMSE": rmse, "Coverage_80%": coverage,
        "wQL_0.1": wql_10, "wQL_0.5": wql_50, "wQL_0.9": wql_90,
        "wQL_Mean": (wql_10 + wql_50 + wql_90) / 3,
        "Protrusion_Area": mean_protrusion,
        "DTW_Shape": mean_shape,
        "DTW_Time": mean_time,
    }
    print("-" * 40)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 40)


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
