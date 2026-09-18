import os
os.environ["CMDSTAN_VERBOSE"] = "FALSE"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.INFO)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from prophet import Prophet
from joblib import Parallel, delayed
import torch

plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    # 固定パラメータ
    "prediction_length": 64,
    "context_length": 128,
    "interval_width": 0.8,
    "output_dir": "Prophet",
    # チューニング対象パラメータ（Optunaで上書き可能）
    "changepoint_prior_scale": 0.1,       # 論文デフォルト0.05，書籍の急変動に対応するため0.1
    "seasonality_prior_scale": 10.0,      # 論文デフォルト，フーリエ級数が低域フィルタとして機能するため弱正則化で十分
    "holidays_prior_scale": 10.0,         # 論文デフォルト，add_country_holidaysと対応
    "seasonality_mode": "multiplicative", # 売上規模に比例して季節変動が拡大する小売データに適切
    "uncertainty_samples": 1000,          # 論文デフォルト，点予測には無影響だが区間推定の信頼性のため
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


def _fit_and_predict(series_df, prediction_length, context_length, freq, params=None):
    if params is None:
        params = CONFIG
    train = series_df.iloc[:-prediction_length].copy()
    train = train.iloc[-context_length:]
    train_prophet = train.rename(columns={'日付': 'ds', 'POS販売冊数': 'y'})
    train_prophet['ds'] = pd.to_datetime(train_prophet['ds'])

    is_daily = (freq == 'D')

    model = Prophet(
        interval_width=CONFIG["interval_width"],
        yearly_seasonality=True,
        weekly_seasonality=is_daily,
        daily_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
        uncertainty_samples=params["uncertainty_samples"],
    )

    if is_daily:
        model.add_country_holidays(country_name='JP')

    model.fit(train_prophet)

    future = model.make_future_dataframe(periods=prediction_length, freq=freq, include_history=False)
    forecast = model.predict(future)

    q50 = forecast['yhat'].values
    q10 = forecast['yhat_lower'].values
    q90 = forecast['yhat_upper'].values

    return np.stack([q10, q50, q90])


def infer_freq(dates):
    dates_sorted = pd.to_datetime(np.sort(dates))
    if len(dates_sorted) < 2:
        return 'W'
    diffs = dates_sorted[1:] - dates_sorted[:-1]
    median_days = np.median([d.days for d in diffs])
    if median_days <= 1.5:
        return 'D'
    elif median_days <= 8:
        return 'W'
    elif median_days <= 16:
        return '2W'
    else:
        return 'MS'


def _process_single_book(book_name, df_group, pred_len, context_length, freq, params=None):
    group = df_group.sort_values('日付')
    if len(group) <= pred_len + context_length:
        return None

    series = group['POS販売冊数'].values.astype(np.float32)
    query = series[:-pred_len]
    target = series[-pred_len:]

    try:
        pred = _fit_and_predict(group[['日付', 'POS販売冊数']], pred_len, context_length, freq, params)
    except Exception:
        pred = np.zeros((3, pred_len), dtype=np.float32)

    return {
        "metadata": {"id": book_name, "target": target, "query": query},
        "forecast": pred
    }


def run_inference(df, decile_books, all_predict):
    pred_len = CONFIG["prediction_length"]

    books = df['書名'].unique()
    if not all_predict:
        books = [b for b in books if b in decile_books]

    freq = infer_freq(df['日付'].values)

    results = Parallel(n_jobs=-1)(
        delayed(_process_single_book)(
            book_name,
            df[df['書名'] == book_name],
            pred_len,
            CONFIG["context_length"],
            freq,
        ) for book_name in tqdm(books)
    )

    metadata = []
    forecasts = []
    for res in results:
        if res is not None:
            metadata.append(res["metadata"])
            forecasts.append(torch.tensor(res["forecast"], dtype=torch.float32))

    return metadata, forecasts


def save_results(samples, forecasts, decile_books, all_predict):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if all_predict:
        os.makedirs(f"{CONFIG['output_dir']}/All_predict", exist_ok=True)
    count = 0
    for i, sample in enumerate(tqdm(samples)):
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
            ax.plot(range(plot_start, len_history), history[plot_start:], color="black", alpha=0.5)

        x_pred = range(len_history, len_history + len(target))
        ax.plot(x_pred, target, color="gray", linestyle="--")

        median = pred[pred.shape[0] // 2]
        low = pred[0] if pred.shape[0] > 1 else median
        high = pred[-1] if pred.shape[0] > 1 else median
        ax.plot(x_pred, median, color="blue")
        ax.fill_between(x_pred, low, high, color="blue", alpha=0.2)

        if len_history > 0:
            ax.axvline(x=len_history, color="red", linestyle=":", alpha=0.7)

        title = f"{book_name}"
        if book_name in decile_books:
            title = f"[{decile_books[book_name]['label']}] {title}"
        ax.set_title(title)
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
