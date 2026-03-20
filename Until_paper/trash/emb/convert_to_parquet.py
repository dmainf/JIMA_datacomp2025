#!/usr/bin/env python3
"""
TSVファイルをParquet形式に変換するスクリプト
data/data.txt を data/data.parquet に変換します
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def convert_txt_to_parquet(
    input_file: str = "data/data.txt",
    output_file: str = "data/data.parquet",
    chunksize: int = 100000
):
    """
    大きなTSVファイルをParquet形式に変換

    Parameters:
    -----------
    input_file : str
        入力TSVファイルのパス
    output_file : str
        出力Parquetファイルのパス
    chunksize : int
        一度に読み込む行数（メモリ効率のため）
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    # 入力ファイルの存在確認
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    print(f"変換開始: {input_file} -> {output_file}")
    print(f"ファイルサイズ: {input_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquetライターの初期化用フラグ
    first_chunk = True
    parquet_writer = None
    total_rows = 0

    try:
        # チャンクごとにデータを読み込んで処理
        for i, chunk in enumerate(pd.read_csv(
            input_file,
            sep='\t',
            chunksize=chunksize,
            low_memory=False
        )):
            total_rows += len(chunk)

            # PyArrow Tableに変換
            table = pa.Table.from_pandas(chunk)

            # 最初のチャンクでParquetライターを初期化
            if first_chunk:
                parquet_writer = pq.ParquetWriter(output_file, table.schema)
                first_chunk = False

            # チャンクを書き込み
            parquet_writer.write_table(table)

            if (i + 1) % 10 == 0:
                print(f"処理中... {total_rows:,} 行完了")

        print(f"\n変換完了!")
        print(f"総行数: {total_rows:,} 行")
        print(f"出力ファイル: {output_file}")
        print(f"出力サイズ: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    finally:
        # ライターを閉じる
        if parquet_writer:
            parquet_writer.close()

if __name__ == "__main__":
    convert_txt_to_parquet()
