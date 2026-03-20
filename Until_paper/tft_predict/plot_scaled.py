import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

def process_book_optimized(book_name, book_df, output_dir):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    safe_name = "".join(c for c in book_name if c.isalnum() or c in (' ', '_', '-')).strip()[:100]

    dates = pd.to_datetime(book_df['日付'])
    original_values = book_df['POS販売冊数'].values

    log_values = np.log1p(original_values)
    scaler = RobustScaler()
    robust_values = scaler.fit_transform(original_values.reshape(-1, 1)).flatten()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(dates, original_values, linewidth=1)
    axes[0].set_title(f'{book_name} - Original')
    axes[0].set_ylabel('POS販売冊数')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, log_values, linewidth=1, color='orange')
    axes[1].set_title('Log1p Transformation')
    axes[1].set_ylabel('log1p(Count)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, robust_values, linewidth=1, color='green')
    axes[2].set_title('Robust Scaler')
    axes[2].set_ylabel('Scaled Value')
    axes[2].set_xlabel('日付')
    axes[2].grid(True, alpha=0.3)

    plt.setp(axes[2].get_xticklabels(), rotation=45)

    plt.tight_layout()

    save_path = output_dir / f"{safe_name}_combined.png"
    plt.savefig(save_path, dpi=80)

    plt.close(fig)

    return book_name

def main():
    print("=== Loading Data ===")
    df = pd.read_parquet('data/df_for.parquet')

    print("=== Selecting 100 Books at Equal Intervals ===")
    book_sales = df.groupby('書名', observed=True)['POS販売冊数'].sum().sort_values(ascending=False)
    total_books = len(book_sales)
    indices = np.linspace(0, total_books - 1, 100, dtype=int)
    selected_books = book_sales.iloc[indices].index.tolist()
    df = df[df['書名'].isin(selected_books)].copy()
    df['書名'] = df['書名'].cat.remove_unused_categories()
    print(f"Selected 100 books from {total_books} total books at equal intervals")

    print("=== Pre-sorting Data ===")
    df['日付'] = pd.to_datetime(df['日付'])
    df = df.sort_values(['書名', '日付'])

    print("=== Creating Plots ===")
    output_dir = Path('scaled_plots')
    output_dir.mkdir(exist_ok=True)

    grouped_data = list(df.groupby('書名', observed=True))

    n_jobs = multiprocessing.cpu_count()
    print(f"Total books: {len(grouped_data)}")
    print(f"Using {n_jobs} CPU cores")

    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_book_optimized)(name, group, output_dir)
        for name, group in grouped_data
    )

    print("=== Complete! ===")
    print(f"Plots saved in {output_dir}/")

if __name__ == "__main__":
    main()
