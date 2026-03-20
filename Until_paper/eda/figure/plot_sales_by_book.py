import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

def remove_volume(df):
    """
    書名から一番右の「_」以降を除去する
    「_」がなければそのまま返す

    例:
        「作品_シリーズ_巻数」→「作品_シリーズ」(remove volume-number)
        「作品_巻数」→「作品」(remove series)
        「作品」→「作品」(変更なし)
    """
    before_counts = df['書名'].str.count('_').value_counts().to_dict()
    def process_volume(title):
        if pd.isna(title):
            return title
        if '_' not in title:
            return title
        return title.rsplit('_', 1)[0]
    df = df.copy()
    df['書名'] = df['書名'].apply(process_volume)
    after_counts = df['書名'].str.count('_').value_counts().to_dict()
    count_2_before = before_counts.get(2, 0)
    count_1_before = before_counts.get(1, 0)
    count_0_before = before_counts.get(0, 0)
    count_1_after = after_counts.get(1, 0)
    count_0_after = after_counts.get(0, 0)
    expected_count_1_after = count_1_before + count_2_before
    expected_count_0_after = count_0_before + count_1_before
    if count_2_before > 0 and count_1_after != expected_count_1_after:
        raise ValueError(f"「_」が2つあるものの変換が不整合: 処理前(2つ)={count_2_before}, 処理後(1つ)={count_1_after}, 期待値={expected_count_1_after}")
    if count_1_before > 0 and count_0_after != expected_count_0_after:
        raise ValueError(f"「_」が1つあるものの変換が不整合: 処理前(1つ)={count_1_before}, 処理後(0つ)={count_0_after}, 期待値={expected_count_0_after}")
    if count_2_before > 0:
        print("You remove volume-number on '書名'")
    if count_1_before > 0:
        print("You remove series on '書名'")
    return df

df = pd.read_parquet('data/sales_df.parquet')
df = remove_volume(df)
daily_sales = df.groupby(['日付', '書名'])['POS販売冊数'].sum().reset_index()
pivot_data = daily_sales.pivot(index='日付', columns='書名', values='POS販売冊数')
total_sales = pivot_data.sum().sort_values(ascending=False)
top_30_columns = total_sales.head(30).index
fig, ax = plt.subplots(figsize=(10, 8))
for column in top_30_columns:
    ax.plot(pivot_data.index, pivot_data[column], label=column, alpha=0.7)

ax.set_xlabel('日付')
ax.set_ylabel('POS販売冊数')
ax.set_title('日付ごとの書名別POS販売冊数(上位30書名)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.savefig('sales_by_book.png', dpi=300, bbox_inches='tight')
print('グラフを plot_sales_by_book.png に保存しました')
