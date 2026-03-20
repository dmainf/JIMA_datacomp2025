import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False
import os

def plot_distribution(column, df=None):
    """指定されたカラムの分布をプロットする"""
    if df is None:
        print("loading data...")
        df = pd.read_csv('../data/data.txt', sep='\t')

    print(f"総行数: {len(df)}")
    print(f"ユニークな{column}数: {df[column].nunique()}")
    null_count = df[column].isnull().sum()
    null_percentage = (null_count/len(df[column])*100).round(2)
    print(f"欠損値の数: {null_count}")
    print(f"欠損値の割合: {null_percentage}")

    counts = df[column].value_counts().head(20)

    plt.figure(figsize=(14, 9))
    # 数値型と文字列型の両方に対応
    if counts.index.dtype == 'object':
        names = [str(name).replace('\u3000', ' ').strip() for name in counts.index]
    else:
        names = [str(name) for name in counts.index]
    plt.barh(range(len(counts)), counts.values)
    plt.yticks(range(len(counts)), names, fontsize=10)
    plt.xlabel('出現回数', fontsize=11)
    plt.ylabel(column, fontsize=11)
    plt.title(f'{column}の分布(上位20件)', fontsize=13)
    plt.gca().invert_yaxis()

    # 欠損値情報をグラフに表示
    plt.text(0.98, 0.98, f'欠損値: {null_count:,}件 ({null_percentage}%)',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    print(f"\n上位20件の{column}:")
    print(counts)

    os.makedirs('figure', exist_ok=True)
    output_file = f'figure/{column}_dist.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_distribution(column)
    else:
        print("使用方法: python plot_dist.py [カラム名1] [カラム名2] ...")


def plot_count_distribution(column, df=None, log_x=False, log_y=False):
    """指定されたカラムの出現回数ごとの頻度分布をプロットする"""
    if df is None:
        print("loading data...")
        df = pd.read_csv('../data/data.txt', sep='\t')
    # 元データから出現回数を計算
    value_counts = df[column].value_counts()
    # 欠損値をカウント
    null_count = df[column].isnull().sum()
    total_count = len(df)
    null_percentage = (null_count / total_count * 100).round(2)

    # 出現回数ごとの頻度を計算
    counts_of_counts = value_counts.value_counts().sort_index()

    plt.figure(figsize=(20, 9))
    plt.bar(counts_of_counts.index, counts_of_counts.values)

    # x軸ラベル
    if log_x:
        plt.xlabel(f'各{column}の出現回数（対数スケール）', fontsize=11)
        plt.xscale('log')
    else:
        plt.xlabel(f'各{column}の出現回数', fontsize=11)

    # y軸ラベル
    if log_y:
        plt.ylabel(f'該当する{column}の種類数（対数スケール）', fontsize=11)
        plt.yscale('log')
    else:
        plt.ylabel(f'該当する{column}の種類数', fontsize=11)

    plt.title(f'{column}の出現回数ごとの頻度分布', fontsize=13)
    plt.grid(True, alpha=0.3, which='both')

    # 欠損値情報をグラフに表示
    plt.text(0.98, 0.98, f'欠損値: {null_count:,}件 ({null_percentage}%)',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    print(f"\n出現回数ごとの{column}種類数:")
    print(counts_of_counts)

    os.makedirs('figure', exist_ok=True)
    log_suffix = ''
    if log_x and log_y:
        log_suffix = '_loglog'
    elif log_x:
        log_suffix = '_logx'
    elif log_y:
        log_suffix = '_logy'
    output_file = f'figure/{column}_counts_dist{log_suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを {output_file} に保存しました")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        columns = sys.argv[1:]
        for column in columns:
            plot_count_distribution(column)
    else:
        print("使用方法: python plot_counts_dist.py [カラム名1] [カラム名2] ...")
