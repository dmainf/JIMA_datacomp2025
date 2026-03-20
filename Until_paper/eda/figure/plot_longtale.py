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
    expected_count_1_after = count_2_before
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

TOPK = 300

df = pd.read_parquet('data/df.parquet')
df = remove_volume(df)
df = remove_volume(df)

total_sales_by_book = df.groupby('書名')['POS販売冊数'].sum().sort_values(ascending=False)

if TOPK is not None:
    total_sales_by_book = total_sales_by_book.head(TOPK)
    print(f'上位{TOPK}件の書名でグラフを作成します')

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(len(total_sales_by_book)), total_sales_by_book.values, linewidth=2, color='steelblue')
ax.set_xlabel('書名(sorted by POS販売冊数)', fontsize=12)
ax.set_ylabel('POS販売冊数', fontsize=12)
ax.set_title('Long-tail Distribution of POS販売冊数 by 書名', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('longtail.png', dpi=300, bbox_inches='tight')
print(f'longtail.png に保存しました（書名数: {len(total_sales_by_book)}）')

print(f'\n統計情報:')
print(f'総書名数: {len(total_sales_by_book)}')
print(f'上位10書名の販売数合計: {total_sales_by_book.head(10).sum()}')
print(f'全体の販売数合計: {total_sales_by_book.sum()}')
print(f'上位10書名が占める割合: {total_sales_by_book.head(10).sum() / total_sales_by_book.sum() * 100:.1f}%')

print(f'\nジニ係数の計算:')
sorted_values = total_sales_by_book.values
n = len(sorted_values)
print(f'n (書名数) = {n}')

cumsum = sorted_values.cumsum()
print(f'累積和の総計 = {cumsum[-1]}')

numerator = 2 * sum((i + 1) * sorted_values[i] for i in range(n))
print(f'分子: 2 * Σ(i+1)*x_i = {numerator}')

denominator = n * cumsum[-1]
print(f'分母: n * Σx_i = {denominator}')

gini = (numerator / denominator) - (n + 1) / n
print(f'\nジニ係数 = (分子/分母) - (n+1)/n')
print(f'ジニ係数 = ({numerator}/{denominator}) - ({n+1}/{n})')
print(f'ジニ係数 = {numerator/denominator:.6f} - {(n+1)/n:.6f}')
print(f'ジニ係数 = {gini:.4f}')
