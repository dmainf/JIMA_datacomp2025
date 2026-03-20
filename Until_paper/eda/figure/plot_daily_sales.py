import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from multiprocessing import Pool, cpu_count
import time
import numpy as np
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
PLOT_CUMULATION = True
def create_cumulative_plot(df, i):
    grouped = df.groupby(['書名', '日付'], observed=True)['POS販売冊数'].sum().reset_index()
    grouped['累積販売冊数'] = grouped.groupby('書名', observed=True)['POS販売冊数'].cumsum()
    final_sales = grouped.groupby('書名', observed=True)['累積販売冊数'].last()
    top30_book_names_sorted = final_sales.nlargest(30).index.tolist()
    fig, ax = plt.subplots(figsize=(16, 10))
    other_books_df = grouped[~grouped['書名'].isin(top30_book_names_sorted)]
    if not other_books_df.empty:
        import matplotlib.dates as mdates
        segments = [g[['日付', '累積販売冊数']].copy() for _, g in other_books_df.groupby('書名', observed=True) if len(g) > 1]
        for seg in segments:
            seg['日付'] = mdates.date2num(seg['日付'])
        segments = [seg.values for seg in segments]
        if segments:
            line_collection = LineCollection(segments, color='gray', alpha=0.3, linewidth=0.5, rasterized=True)
            ax.add_collection(line_collection)
            ax.autoscale_view()
    lines = []
    labels = []
    for book_name in top30_book_names_sorted:
        book_data = grouped[grouped['書名'] == book_name]
        if not book_data.empty:
            book_data = book_data.sort_values('日付')
            line, = ax.plot(book_data['日付'], book_data['累積販売冊数'], label=str(book_name), alpha=0.8, linewidth=1.5, rasterized=True)
            lines.append(line)
            labels.append(str(book_name))
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('累積POS販売冊数', fontsize=12)
    ax.set_title(f'書名ごとの累積POS販売冊数推移（上位30位を強調表示） - 書店{i}', fontsize=14)
    ax.grid(True, alpha=0.3)
    import datetime
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    ax.set_xlim(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    class DayResetFormatter:
        def __init__(self):
            self.last_day = -1
        def __call__(self, x, pos=None):
            date = mdates.num2date(x)
            day_str = date.strftime('%d')
            if date.day < self.last_day:
                month_str = date.strftime('%m')
                label = f"{day_str}\n{month_str}"
            else:
                label = day_str
            self.last_day = date.day
            return label
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(DayResetFormatter())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    all_sales_values = grouped['累積販売冊数'].values
    min_y = all_sales_values.min()
    max_y = all_sales_values.max()
    y_buffer = (max_y - min_y) * 0.05
    y_range = (max_y + y_buffer) - (min_y - y_buffer)
    if y_range > 5000:
        tick_interval = 1000
    elif y_range > 1000:
        tick_interval = 200
    elif y_range > 500:
        tick_interval = 100
    else:
        tick_interval = 50
    start_tick = np.floor((min_y - y_buffer) / tick_interval) * tick_interval
    end_tick = np.ceil((max_y + y_buffer) / tick_interval) * tick_interval
    yticks = np.arange(start_tick, end_tick + tick_interval, tick_interval)
    ax.set_ylim(min_y - y_buffer, max_y + y_buffer)
    ax.set_yticks(yticks)
    ax.ticklabel_format(style='plain', axis='y')
    if lines:
        ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    total_store_sales = df['POS販売冊数'].sum()
    ax.text(0.98, 0.02, f'合計販売冊数: {total_store_sales:,.0f}冊',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    os.makedirs('cumulative', exist_ok=True)
    output_file = f'cumulative/cumulative_sales_store_{i}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    num_books = len(grouped['書名'].unique())
    return f"Store {i}: Plotted cumulative sales for {num_books} books, highlighting top 30."
def process_store(args):
    i, df_store = args
    df_store['日付'] = pd.to_datetime(df_store['日付'])
    df_store = df_store.sort_values('日付')
    grouped = df_store.groupby(['書名', '日付'], observed=True)['POS販売冊数'].sum().reset_index()
    total_sales = grouped.groupby('書名', observed=True)['POS販売冊数'].sum()
    top30_book_names_sorted = total_sales.nlargest(30).index.tolist()
    fig, ax = plt.subplots(figsize=(16, 10))
    other_books_df = grouped[~grouped['書名'].isin(top30_book_names_sorted)]
    if not other_books_df.empty:
        import matplotlib.dates as mdates
        segments = [g[['日付', 'POS販売冊数']].copy() for _, g in other_books_df.groupby('書名', observed=True) if len(g) > 1]
        for seg in segments:
            seg['日付'] = mdates.date2num(seg['日付'])
        segments = [seg.values for seg in segments]
        if segments:
            line_collection = LineCollection(segments, color='gray', alpha=0.3, linewidth=0.5, rasterized=True)
            ax.add_collection(line_collection)
            ax.autoscale_view()
    lines = []
    labels = []
    for book_name in top30_book_names_sorted:
        book_data = grouped[grouped['書名'] == book_name]
        if not book_data.empty:
            book_data = book_data.sort_values('日付')
            line, = ax.plot(book_data['日付'], book_data['POS販売冊数'], label=str(book_name), alpha=0.8, linewidth=1.5, rasterized=True)
            lines.append(line)
            labels.append(str(book_name))
    ax.set_xlabel('日付', fontsize=12)
    ax.set_ylabel('POS販売冊数', fontsize=12)
    ax.set_title(f'書名ごとの日次POS販売冊数推移（上位30位を強調表示） - 書店{i}', fontsize=14)
    ax.grid(True, alpha=0.3)
    import datetime
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    ax.set_xlim(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    class DayResetFormatter:
        def __init__(self):
            self.last_day = -1
        def __call__(self, x, pos=None):
            date = mdates.num2date(x)
            day_str = date.strftime('%d')
            if date.day < self.last_day:
                month_str = date.strftime('%m')
                label = f"{day_str}\n{month_str}"
            else:
                label = day_str
            self.last_day = date.day
            return label
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(DayResetFormatter())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    all_sales_values = grouped['POS販売冊数'].values
    min_y = all_sales_values.min()
    max_y = all_sales_values.max()
    y_buffer = (max_y - min_y) * 0.05
    y_range = (max_y + y_buffer) - (min_y - y_buffer)
    if y_range > 100:
        tick_interval = 20
    elif y_range > 50:
        tick_interval = 10
    else:
        tick_interval = 5
    start_tick = np.floor((min_y - y_buffer) / tick_interval) * tick_interval
    end_tick = np.ceil((max_y + y_buffer) / tick_interval) * tick_interval
    yticks = np.arange(start_tick, end_tick + tick_interval, tick_interval)
    ax.set_ylim(min_y - y_buffer, max_y + y_buffer)
    ax.set_yticks(yticks)
    ax.ticklabel_format(style='plain', axis='y')
    if lines:
        ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    total_store_sales = df_store['POS販売冊数'].sum()
    ax.text(0.98, 0.02, f'合計販売冊数: {total_store_sales:,.0f}冊',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    os.makedirs('daily', exist_ok=True)
    output_file = f'daily/daily_sales_store_{i}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    num_books = len(grouped['書名'].unique())
    daily_msg = f"Store {i}: Plotted {num_books} books, highlighting top 30 with sorted legend."
    cumulative_msg = ""
    if PLOT_CUMULATION:
        cumulative_msg = "\n" + create_cumulative_plot(df_store, i)
    return daily_msg + cumulative_msg
if __name__ == '__main__':
    print(f"{ '='*60}")
    print(f"Starting parallel processing with {cpu_count()} CPU cores")
    print(f"{ '='*60}")
    start_time = time.time()
    df = pd.read_parquet('../data/sales_df.parquet')
    store_ids = sorted(df['書店コード'].unique())
    store_data_list = [(i, df[df['書店コード'] == i].copy()) for i in store_ids]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_store, store_data_list)
    for result in results:
        print(result)
        print(f"{ '='*60}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{ '='*60}")
    print(f"All stores processed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per store: {elapsed_time/len(store_ids):.2f} seconds")
    print(f"{ '='*60}")