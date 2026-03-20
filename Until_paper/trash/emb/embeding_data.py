import sys
import os

# スクリプトの場所から親ディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from lib.prepro import *
from lib.fueture_eng import *
from sentence_transformers import SentenceTransformer, util
import gc
from tqdm import tqdm

def emb2vec(df, columns_to_process):
    print("loading model...")
    # 軽量なモデル: 384次元
    model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device='mps'
    )
    print("complete!")

    for column in columns_to_process:
        print(f"processing {column}...")
        # ユニークな値のみを取得
        unique_texts = df[column].fillna('').astype(str).unique()
        print(f"unique values: {len(unique_texts)} (from {len(df)} rows)")

        print(f"encoding {column}...")
        # ユニーク値のみエンコード
        unique_embeddings = model.encode(
            unique_texts.tolist(),
            batch_size=64,  # バッチサイズを増やして高速化
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype('float32')  # エンコード時点でfloat32に変換

        print(f"mapping embeddings...")
        # ユニーク値とインデックスのマッピング
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}

        # インデックス配列を作成（メモリ効率的）
        texts = df[column].fillna('').astype(str)
        indices = np.array([text_to_idx[text] for text in texts], dtype=np.int32)

        # インデックスを使って埋め込みをマッピング（メモリ効率的）
        embeddings = unique_embeddings[indices]

        print(f"creating embedding dataframe...")
        # 埋め込みを直接DataFrameに変換
        n_components = embeddings.shape[1]
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f'{column}_emb_{j}' for j in range(n_components)],
            index=df.index,
            dtype='float32'
        )

        # 元のカラムと結合
        df = pd.concat([df, emb_df], axis=1)

        # メモリ解放
        del unique_texts, unique_embeddings, text_to_idx, texts, indices, embeddings, emb_df
        gc.collect()

    print("all process complete!")
    return df

print("loading data...")
data_path = os.path.join(parent_dir, 'data', 'data.parquet')
# 必要なカラムのみ読み込み
df = pd.read_parquet(data_path, columns=['書名'])
print("complete!")

# ユニークな書名のみに絞る
print(f"original rows: {len(df)}")
df = df.drop_duplicates(subset=['書名']).reset_index(drop=True)
print(f"unique rows: {len(df)}")

emb2vec_columns = ['書名']
df = emb2vec(df, emb2vec_columns)

print("saving embedded data...")
output_path = os.path.join(script_dir, 'embed_書名.parquet')
df.to_parquet(output_path,
            engine='pyarrow',
            compression='snappy',
            index=False)
print(f"saved to {output_path}")
