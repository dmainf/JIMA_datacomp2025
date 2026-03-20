import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from multiprocessing import Pool, cpu_count

# グラフ関数
def plot_sum_by_book(df, output_dir='figure', store_id=None, book_name_filter='薬屋のひとりごと'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    df = df.copy()

    # 対象書名でフィルタ
    df = df[df['書名'].astype(str).str.contains(book_name_filter, na=False)]
    if df.empty:
        print(f"Store {store_id}: No data for '{book_name_filter}', skipping...")
        return

    df['巻数'] = df['書名'].astype(str).str.extract(r'_(\d+)$')
    df['巻数'] = df['巻数'].fillna('不明').astype(str)

    if not {'本体価格', 'POS販売冊数'}.issubset(df.columns):
        raise ValueError("必要な列（本体価格, POS販売冊数）が見つかりません。")
    df['売上'] = df['本体価格'] * df['POS販売冊数']

    if '累積日数' not in df.columns:
        raise ValueError("累積日数カラムが存在しません。")
    df['累積日数'] = pd.to_numeric(df['累積日数'], errors='coerce').fillna(0).astype(int).clip(1, 365)

    grouped = (
        df.groupby(['巻数', '累積日数'], observed=True)['売上']
        .sum()
        .reset_index()
        .sort_values(['巻数', '累積日数'])
    )

    all_days = pd.Index(np.arange(1, 366), name='累積日数')
    filled_list = []
    for vol, g in grouped.groupby('巻数'):
        g = g.set_index('累積日数').reindex(all_days, fill_value=0).reset_index()
        g['巻数'] = vol
        filled_list.append(g)
    grouped_full = pd.concat(filled_list, ignore_index=True)

    grouped_full['累積売上'] = grouped_full.groupby('巻数', observed=True)['売上'].cumsum()

    final_sales = grouped_full.groupby('巻数', observed=True)['累積売上'].last().sort_values(ascending=False)
    top10_volumes = final_sales.head(10)
    top10_set = set(top10_volumes.index)

    print("上位10巻（累積売上）:")
    for i, (vol, sales) in enumerate(top10_volumes.items(), 1):
        print(f"{i}. {book_name_filter}_{vol}巻: {sales:,.0f}円")

    # グラフ描画
    plt.figure(figsize=(16, 10))
    other = grouped_full[~grouped_full['巻数'].isin(top10_set)]
    for vol, g in other.groupby('巻数'):
        plt.plot(g['累積日数'], g['累積売上'], color='gray', alpha=0.3, linewidth=0.5)
    for vol in top10_volumes.index:
        g = grouped_full[grouped_full['巻数'] == vol]
        plt.plot(g['累積日数'], g['累積売上'], linewidth=2, label=f"{book_name_filter}_{vol}巻")

    plt.xlabel("累積日数", fontsize=12)
    plt.ylabel("累積売上（円）", fontsize=12)
    plt.title(f"書名ごとの累積売上推移（{book_name_filter} 各巻） - 書店{store_id}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="上位10巻", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    save_path = output_dir / f"Kusuriya_sales_store_{store_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {save_path}")


def process_store(i):
    """各書店のデータを処理する関数"""
    file_path = f'data/df_{i}.parquet'
    book_name_filter = '薬屋のひとりごと'
    if not os.path.exists(file_path):
        return f"Store {i}: File not found, skipping..."

    try:
        df = pd.read_parquet(file_path)
        if '大分類' not in df.columns:
            return f"Store {i}: '大分類' column not found, skipping..."
        df = df[df['大分類'].astype(str).str.contains('コミ', na=False)]
        df = df[df['書名'].str.contains(book_name_filter, na=False)]
        if df.empty:
            return f"Store {i}: No data for '{book_name_filter}', skipping..."

        plot_sum_by_book(df, output_dir='figure', store_id=i, book_name_filter=book_name_filter)
        return f"Store {i}: Completed for '{book_name_filter}' ({len(df)} rows)"
    except Exception as e:
        return f"Store {i}: Error - {str(e)}"


if __name__ == '__main__':
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    print(f"\n{'='*60}")
    print("Starting parallel processing for '薬屋のひとりごと'...")
    print('='*60)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_store, range(1, 36))

    print(f"\n{'='*60}")
    print("Processing Results:")
    print('='*60)
    for result in results:
        print(result)

    print(f"\n{'='*60}")
    print("All stores processed successfully for '薬屋のひとりごと'!")
    print('='*60)
