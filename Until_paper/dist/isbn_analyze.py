
import pandas as pd
from collections import Counter

file_path = 'data/embed_data.parquet'

# ハイフンパターンのカウンター
total_pattern_counts = Counter()

try:
    # Parquetファイルから'ISBN'カラムのみを読み込む
    df = pd.read_parquet(file_path, columns=['ISBN'])

    # --- ここからの処理はDataFrame全体に対して一度に行う ---

    # ISBNを文字列に変換し、スペースを削除
    cleaned_isbn = df['ISBN'].fillna('').astype(str).str.replace(' ', '', regex=False).str.replace('　', '', regex=False)

    # 17桁のデータのみをフィルタリング
    isbn_17_digits = cleaned_isbn[cleaned_isbn.str.len() == 17]

    # パターンを分析する関数
    def get_hyphen_pattern(isbn_string):
        parts = isbn_string.split('-')
        lengths = [str(len(p)) for p in parts]
        return "-".join(lengths)

    # 17桁のISBNにパターン分析を適用
    if not isbn_17_digits.empty:
        patterns = isbn_17_digits.apply(get_hyphen_pattern)
        total_pattern_counts.update(patterns)

    print("17桁ISBNのハイフン区切りパターン分析 (from embed_data.parquet):")
    if not total_pattern_counts:
        print("分析対象の17桁ISBNデータが見つかりませんでした。")
    else:
        # 件数が多い順に表示
        for pattern, count in total_pattern_counts.most_common():
            print(f'{pattern}: {count}件')

except ImportError:
    print("エラー: Parquetファイルの読み込みには `pyarrow` ライブラリが必要です。")
    print("pip install pyarrow を実行してください。")
except FileNotFoundError:
    print(f"エラー: {file_path} が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
