import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_distribution(df, col_name, save_dir):
    """カラムの分布をプロットして保存する関数"""
    data = df[col_name].dropna()
    if len(data) == 0:
        print(f"Skipping {col_name} (no data)")
        return

    mean_val = data.mean()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    zeros = (data == 0).sum()
    zero_pct = zeros / len(data) * 100

    plt.figure(figsize=(10, 6))

    plt.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

    title = (f"Distribution of {col_name}\n"
             f"Mean: {mean_val:.2f}, Std: {std_val:.2f}\n"
             f"Range: [{min_val:.2f}, {max_val:.2f}], Zeros: {zero_pct:.1f}%")
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')
    plt.legend()

    safe_name = col_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    save_path = os.path.join(save_dir, f"{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    print("=== Loading Data ===")
    if os.path.exists('train.parquet'):
        file_path = 'train.parquet'
    elif os.path.exists('data/df_for.parquet'):
        file_path = 'data/df_for.parquet'
    else:
        file_path = 'train.parquet'

    print(f"Reading from: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Data shape: {df.shape}")

    time_features = [
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_holiday'
    ]

    past_dynamic_features = [
        '大分類_POS販売冊数_relative', '大分類_POS販売冊数_z_score',
        '中分類_POS販売冊数_relative', '中分類_POS販売冊数_z_score',
        '小分類_POS販売冊数_relative', '小分類_POS販売冊数_z_score',
        '大分類_log_price_relative', '大分類_log_price_z_score',
        '中分類_log_price_relative', '中分類_log_price_z_score',
        '小分類_log_price_relative', '小分類_log_price_z_score',
        'momentum', 'volatility', 'z_score',
        'days_since_spike_2.0', 'days_since_spike_2.5', 'days_since_spike_3.0'
    ]

    target_cols = ['POS販売冊数', 'log_price', '本体価格']

    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    known_cols = set(time_features + past_dynamic_features + target_cols)
    other_cols = [c for c in all_numeric if c not in known_cols]

    base_dir = 'figure/distributions'
    dirs = {
        'time': os.path.join(base_dir, 'time_features'),
        'dynamic': os.path.join(base_dir, 'past_dynamic_features'),
        'target': os.path.join(base_dir, 'target_price'),
        'others': os.path.join(base_dir, 'others')
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("\n=== Plotting Distributions ===")

    print(f"\n--- Time Features ({len(time_features)}) ---")
    for col in time_features:
        if col in df.columns:
            plot_distribution(df, col, dirs['time'])
        else:
            print(f"Warning: {col} not found in dataframe")

    print(f"\n--- Past Dynamic Features ({len(past_dynamic_features)}) ---")
    for col in past_dynamic_features:
        if col in df.columns:
            plot_distribution(df, col, dirs['dynamic'])
        else:
            print(f"Warning: {col} not found in dataframe")

    print(f"\n--- Target & Price Features ---")
    for col in target_cols:
        if col in df.columns:
            plot_distribution(df, col, dirs['target'])

    if len(other_cols) > 0:
        print(f"\n--- Other Numeric Features ({len(other_cols)}) ---")
        for col in other_cols:
            plot_distribution(df, col, dirs['others'])

    print("\n=== Saving Summary Statistics ===")
    summary_path = os.path.join(base_dir, 'distribution_summary.csv')
    df.describe().transpose().to_csv(summary_path)
    print(f"Saved summary to: {summary_path}")

    print(f"\nAll plots saved in: {base_dir}")

if __name__ == "__main__":
    main()
