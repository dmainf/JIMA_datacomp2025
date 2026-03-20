import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

CONFIG = {
    "data_path": "data/df_for.parquet",
    "output_dir": "lgbm_ts_output",
    "prediction_length": 64,
    "valid_ratio": 0.1,
    "min_history": 200,
    "seed": 42,

    "n_estimators": 4000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "early_stopping_rounds": 200,
    "verbose_eval": 200,
}

COL_ID = "書名"
COL_TIME = "日付"
COL_Y_RAW = "POS販売冊数"

CAT_COLS = ["出版社", "大分類", "中分類", "小分類"]
NUM_COLS = ["本体価格"]

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14]

TS_COLS = [f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_7"]

FEATURE_COLS = CAT_COLS + NUM_COLS + TS_COLS + ["horizon"]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not Found : {path}")

    df = pd.read_parquet(path).copy()

    required = [COL_ID, COL_TIME, COL_Y_RAW] + CAT_COLS + NUM_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Not Found : {missing}")

    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df["y"] = df[COL_Y_RAW].astype(float).clip(lower=0.0)

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df.sort_values([COL_ID, COL_TIME]).reset_index(drop=True)


def compute_ts_features(y: np.ndarray, t: int) -> dict:
    feat = {}

    for k in LAGS:
        feat[f"lag_{k}"] = float(y[t - k])

    for w in ROLL_WINDOWS:
        window = y[t - w + 1 : t + 1]
        feat[f"roll_mean_{w}"] = float(np.mean(window))

    w = 7
    window = y[t - w + 1 : t + 1]
    feat["roll_std_7"] = float(np.std(window))

    return feat


def build_supervised_and_inference_samples(df: pd.DataFrame):
    pred_len = CONFIG["prediction_length"]
    valid_ratio = CONFIG["valid_ratio"]
    min_history = CONFIG["min_history"]

    need_back = max(max(LAGS), max(ROLL_WINDOWS))

    train_rows, valid_rows = [], []
    inference_samples = []

    for book, g in df.groupby(COL_ID, observed=True):
        g = g.sort_values(COL_TIME).reset_index(drop=True)
        if len(g) < (min_history + pred_len + need_back + 1):
            continue

        y = g["y"].values

        inference_samples.append({
            "id": book,
            "target": y[-pred_len:],
            "query": y[:-pred_len],
        })

        max_t = len(y) - pred_len - 1
        if max_t <= need_back:
            continue

        split_t = int(max_t * (1.0 - valid_ratio))

        for t in range(need_back, max_t + 1):
            base = {}
            for c in CAT_COLS:
                base[c] = g.iloc[t][c]
            for c in NUM_COLS:
                base[c] = float(g.iloc[t][c])

            ts_feat = compute_ts_features(y, t)

            for h in range(1, pred_len + 1):
                rec = dict(base)
                rec.update(ts_feat)
                rec["horizon"] = h
                rec["y_label"] = float(y[t + h])

                if t < split_t:
                    train_rows.append(rec)
                else:
                    valid_rows.append(rec)

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)

    if len(train_df) == 0 or len(valid_df) == 0 or len(inference_samples) == 0:
        raise ValueError("Error")

    for c in CAT_COLS:
        train_df[c] = train_df[c].astype("category")
        valid_df[c] = valid_df[c].astype("category")

    return train_df, valid_df, inference_samples


def train_quantile_models(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df["y_label"].values
    X_va = valid_df[FEATURE_COLS]
    y_va = valid_df["y_label"].values

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=CAT_COLS, reference=dtrain, free_raw_data=False)

    base_params = {
        "learning_rate": CONFIG["learning_rate"],
        "num_leaves": CONFIG["num_leaves"],
        "min_data_in_leaf": CONFIG["min_data_in_leaf"],
        "feature_fraction": CONFIG["feature_fraction"],
        "bagging_fraction": CONFIG["bagging_fraction"],
        "bagging_freq": CONFIG["bagging_freq"],
        "lambda_l2": CONFIG["lambda_l2"],
        "verbosity": -1,
        "seed": CONFIG["seed"],
    }

    models = {}
    for q in [0.1, 0.5, 0.9]:
        params = dict(base_params)
        params.update({"objective": "quantile", "alpha": q, "metric": "quantile"})

        print(f"\n=== Train LightGBM Quantile: alpha={q} ===")
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=CONFIG["n_estimators"],
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(CONFIG["verbose_eval"]),
            ],
        )
        models[q] = booster

    return models


def forecast_per_book(df: pd.DataFrame, inference_samples, models):
    pred_len = CONFIG["prediction_length"]
    need_back = max(max(LAGS), max(ROLL_WINDOWS))
    grouped = {b: g for b, g in df.groupby(COL_ID, observed=True)}

    forecasts = np.zeros((len(inference_samples), 3, pred_len), dtype=np.float32)
    q_list = [0.1, 0.5, 0.9]

    for i, s in enumerate(inference_samples):
        book = s["id"]
        g = grouped.get(book)
        if g is None:
            continue
        g = g.sort_values(COL_TIME).reset_index(drop=True)

        y = g["y"].values
        t0 = len(g) - pred_len - 1

        if t0 < need_back:
            continue

        base = {}
        for c in CAT_COLS:
            base[c] = g.iloc[t0][c]
        for c in NUM_COLS:
            base[c] = float(g.iloc[t0][c])

        ts_feat = compute_ts_features(y, t0)

        X_inf = []
        for h in range(1, pred_len + 1):
            rec = dict(base)
            rec.update(ts_feat)
            rec["horizon"] = h
            X_inf.append(rec)

        X_inf = pd.DataFrame(X_inf)
        for c in CAT_COLS:
            X_inf[c] = X_inf[c].astype("category")
        X_inf = X_inf[FEATURE_COLS]

        for qi, q in enumerate(q_list):
            pred = models[q].predict(X_inf, num_iteration=models[q].best_iteration)
            forecasts[i, qi, :] = pred.astype(np.float32)

    return forecasts


def save_results_per_book(inference_samples, forecasts):
    out_dir = CONFIG["output_dir"]
    pred_len = CONFIG["prediction_length"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "All_predict"), exist_ok=True)

    for i, s in enumerate(inference_samples):
        book = s["id"]

        history = s["query"]
        target = s["target"]
        pred_q = forecasts[i]
        q10, q50, q90 = pred_q[0], pred_q[1], pred_q[2]

        len_history = len(history)
        x_hist = np.arange(len_history)
        x_pred = np.arange(len_history, len_history + pred_len)

        fig, ax = plt.subplots(figsize=(12, 6))
        if len_history > 0:
            plot_start = max(0, len_history - 200)
            ax.plot(x_hist[plot_start:], history[plot_start:], label="History", alpha=0.6)
        ax.plot(x_pred, target, label="Actual", linestyle="--", alpha=0.8)
        ax.plot(x_pred, q50, label="Forecast(q50)")
        ax.fill_between(x_pred, q10, q90, alpha=0.2, label="q10-q90")
        ax.axvline(x=len_history, linestyle=":", alpha=0.7, label="Forecast Start")
        ax.set_title(str(book))
        ax.legend()
        fig.tight_layout()

        safe_name = str(book).replace("/", "_")
        fig.savefig(os.path.join(out_dir, "All_predict", f"{safe_name}.png"))
        plt.close(fig)



def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    df = load_data(CONFIG["data_path"])
    train_df, valid_df, inference_samples = build_supervised_and_inference_samples(df)

    print(f"\nTrain rows: {len(train_df):,}")
    print(f"Valid rows: {len(valid_df):,}")
    print(f"Books(inference): {len(inference_samples):,}")

    models = train_quantile_models(train_df, valid_df)
    forecasts = forecast_per_book(df, inference_samples, models)

    for q, m in models.items():
        m.save_model(os.path.join(CONFIG["output_dir"], f"lgbm_quantile_{q}.txt"))

    save_results_per_book(inference_samples, forecasts)

    metrics, _ = evaluate_predictions(inference_samples, forecasts, all_predict=True)
    print("\nSaved:", os.path.join(CONFIG["output_dir"], "evaluation_all.csv"),
          "and per-book plots in", os.path.join(CONFIG["output_dir"], "All_predict"))
    return metrics


def _pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.sum(np.maximum(q * diff, (q - 1) * diff))


def evaluate_predictions(samples, forecasts, all_predict: bool):
    print("\n=== Calculating Evaluation Metrics ===")

    targets = np.stack([s["target"] for s in samples])
    pred_array = np.array(forecasts)
    book_names = [s["id"] for s in samples]

    q10 = pred_array[:, 0, :]
    q50 = pred_array[:, 1, :]
    q90 = pred_array[:, 2, :]

    abs_err = np.abs(targets - q50)
    sq_err = (targets - q50) ** 2
    in_interval = (targets >= q10) & (targets <= q90)

    item_mae = np.mean(abs_err, axis=1)
    item_rmse = np.sqrt(np.mean(sq_err, axis=1))
    item_coverage = np.mean(in_interval, axis=1)
    item_total_sales = np.sum(targets, axis=1)

    diff_10 = targets - q10
    diff_50 = targets - q50
    diff_90 = targets - q90
    item_wql_10 = 2 * np.sum(np.maximum(0.1 * diff_10, -0.9 * diff_10), axis=1)
    item_wql_50 = 2 * np.sum(np.maximum(0.5 * diff_50, -0.5 * diff_50), axis=1)
    item_wql_90 = 2 * np.sum(np.maximum(0.9 * diff_90, -0.1 * diff_90), axis=1)

    df_res = pd.DataFrame({
        "book_name": book_names,
        "MAE": item_mae,
        "RMSE": item_rmse,
        "wQL_0.1": item_wql_10,
        "wQL_0.5": item_wql_50,
        "wQL_0.9": item_wql_90,
        "wQL_Mean": (item_wql_10 + item_wql_50 + item_wql_90) / 3,
        "Coverage_80%": item_coverage,
        "Total_Sales": item_total_sales,
    })

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    fname = "evaluation_all.csv" if all_predict else "evaluation_decile.csv"
    df_res.to_csv(os.path.join(CONFIG["output_dir"], fname), index=False)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    coverage = float(np.mean(in_interval))

    total_sales = float(np.sum(np.abs(targets)))
    if total_sales > 0:
        wql_10 = float(2 * _pinball_loss(targets, q10, 0.1) / total_sales)
        wql_50 = float(2 * _pinball_loss(targets, q50, 0.5) / total_sales)
        wql_90 = float(2 * _pinball_loss(targets, q90, 0.9) / total_sales)
    else:
        wql_10 = wql_50 = wql_90 = 0.0

    final_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Coverage_80%": coverage,
        "wQL_0.1": wql_10,
        "wQL_0.5": wql_50,
        "wQL_0.9": wql_90,
        "wQL_Mean": (wql_10 + wql_50 + wql_90) / 3,
    }

    print("-" * 40)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20} | {v:.4f}")
    print("-" * 40)

    return final_metrics, df_res


if __name__ == "__main__":
    main()