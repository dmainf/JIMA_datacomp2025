# Apache Parquet (.parquet) フォーマット解説

---

## 1. 概要

Apache Parquet は、Twitter と Cloudera が共同開発した**オープンソースの列指向データファイル形式**。
Google の Dremel 論文にインスパイアされており、大規模データの効率的な保存・検索を目的として設計されている。

- ファイルは `PAR1` というマジックナンバーで識別される
- Hadoop エコシステムの標準フォーマットとして普及
- クラウド（S3, Azure Data Lake, GCS）でのデータレイク基盤として事実上の標準

---

## 2. ファイル構造

```
[PAR1 Magic Number]
├─ Row Group 1
│  ├─ Column Chunk A
│  │  ├─ Dictionary Page
│  │  └─ Data Page(s)
│  └─ Column Chunk B
│     └─ Data Page(s)
├─ Row Group 2 ...
[File Metadata / Footer]
[PAR1 Magic Number]
```

| 構成要素 | 説明 |
|---------|------|
| **Row Group** | データの水平パーティション。独立して処理可能 |
| **Column Chunk** | 特定の Row Group 内の特定列のデータ（連続配置が保証される） |
| **Page** | 最小の保存単位（Data Page / Dictionary Page） |
| **Footer / Metadata** | スキーマ・圧縮方式・列統計（min / max / null count）を格納 |

### 設計上のポイント

- メタデータはデータの**後に書き込む** → シングルパス書き込みが可能
- リーダーはフッターを先に読み込み、必要な列のみを選択して取得できる
- 列統計を活用して不要な Row Group をスキップする**データスキップ**が可能

---

## 3. 列指向ストレージの利点

### 行ベースとの概念比較

```
行ベース (CSV/JSON)           列指向 (Parquet)
┌────┬──────┬────────┐        列A: [A1, A2, A3, ...]
│ A1 │  B1  │   C1   │        列B: [B1, B2, B3, ...]
│ A2 │  B2  │   C2   │        列C: [C1, C2, C3, ...]
│ A3 │  B3  │   C3   │
└────┴──────┴────────┘
→ 1列だけ読みたくても        → 必要な列だけ読み込める
  全行を走査する必要あり
```

| 特性 | 行ベース (CSV/JSON) | 列指向 (Parquet) |
|-----|------------------|-----------------|
| 一部列の読み取り | 行全体を読む必要あり | 必要列のみ読み込む |
| 圧縮効率 | 低い | 高い（同型データが連続配置） |
| 向いているワークロード | OLTP（更新・単行アクセス） | OLAP（集計・分析） |
| 読み取り速度 | 基準 | **10〜100倍高速** (Databricks調査) |

---

## 3-A. 全列読み込み時でも Parquet が速い理由

「列を絞らなければ Parquet の優位性はない」と思われがちだが、**全列フルスキャン時でも Parquet は CSV/JSON に対して 2〜7 倍高速**（DuckDB TPC-H ベンチマーク等）。以下にその理由を示す。

---

### ベンチマーク数値（フルスキャン時）

| フォーマット | ファイルサイズ (TPC-H lineitem, 1.7億行) | フルスキャン相対速度 |
|------------|----------------------------------------|-------------------|
| **Parquet** | **72 MB** | **基準 (1x)** |
| ORC | 110 MB | 同程度〜やや遅い |
| Avro | 116 MB | 1.5〜2x 遅い |
| CSV | 230 MB | **2〜7x 遅い** |
| JSON | CSV の 2〜4 倍大きい | **5〜20x 遅い** |

> DuckDB TPC-H スケール 20 では CSV は Parquet の **約 5 倍のファイルサイズ**になり、フルスキャン速度でも Parquet が大幅に優位。
> 実世界データでは「194GB CSV → 4.7GB Parquet (97.6% 削減)」のような例もある。

---

### 速さの理由（寄与度順）

#### 理由 1：圧縮による I/O バイト削減（寄与度 70〜80%）

これが**最大の要因**。フルスキャンでも「読み込むバイト数」が 2〜10 倍少ない。

```
CSV:     A1, B1, C1\n A2, B2, C2\n ...  ← テキスト、1バイト/文字
Parquet: [圧縮済み列データ]              ← バイナリ + 辞書/RLE で大幅削減
```

- **列ごとに圧縮**するため、同型データが連続 → 圧縮アルゴリズムが効きやすい
- CSV はデータ型をテキストで表現するため本質的に非効率（数値 `123456789` を9バイト格納）
- ディスク I/O とネットワーク I/O（クラウドストレージ利用時）の両方で効く

#### 理由 2：エンコーディングによる二重の効率化（寄与度 5〜15%）

圧縮の前段階として、Parquet は列ごとに適切なエンコーディングを自動適用する。

| エンコーディング | 仕組み | 効果的な列 |
|--------------|------|-----------|
| **Dictionary Encoding** | 繰り返し値を整数インデックスに置換 | カテゴリ列（国コード、ステータスなど） |
| **RLE** | 連続する同一値をカウントとして圧縮 | NULL 列・ブール列 |
| **Delta Encoding** | 差分のみ格納 | タイムスタンプ・連番 ID |

エンコーディング後のデータを圧縮コーデック（ZSTD 等）で圧縮するため**二段階の削減**が起きる。

#### 理由 3：CPU キャッシュ効率と SIMD ベクトル化（寄与度 5〜10%）

列指向レイアウトは CPU との相性が良い。

```
行ベース（CSV パース後）:    [A1, B1, C1, A2, B2, C2, ...]  ← 型が混在
列指向（Parquet 読み込み後）: [A1, A2, A3, ...]              ← 同型が連続
```

- **CPU キャッシュミス削減**: 同じ型のデータが連続 → キャッシュラインを無駄なく使用
- **SIMD ベクトル化**: AVX-512 対応 CPU では 16 個の float を 1 命令で処理可能
  - Hive + Parquet ベクトル化読み込みで平均 **26.5% のクエリ高速化** (Cloudera 計測)
- CSV はテキストパース（文字列 → 数値変換）が必要で CPU コストが高い

#### 理由 4：Row Group 統計によるスキップ（寄与度 状況依存）

フルスキャンでも WHERE 句がある場合は効く。

- 各 Row Group のフッターに `min / max / null count` が記録される
- 条件に合致しない Row Group を**読み込まずにスキップ**
- 例：`WHERE age > 60` → 各 Row Group の `max(age)` を確認し、60 以下なら丸ごとスキップ

---

### 要因まとめ

| 要因 | フルスキャン時の寄与度 | 備考 |
|-----|-------------------|----|
| 圧縮による I/O 削減 | **70〜80%** | ファイルが 2〜10 倍小さい |
| エンコーディング効率 | **5〜15%** | Dictionary / RLE / Delta の二重削減 |
| CPU キャッシュ効率・SIMD | **5〜10%** | 同型連続配置、テキストパース不要 |
| Row Group スキップ | 状況依存 | WHERE 句があるときのみ効果大 |
| 列プルーニング（列の絞り込み） | フルスキャンでは 0% | ただし列を絞れば追加で 10〜20% 削減 |

> **結論**: フルスキャンでも速い理由の大部分は「**圧縮で読み込むバイト数が少ない**」こと。
> I/O ボトルネックが支配的な現代のデータ処理では、これが直接的にスループット向上に繋がる。

---

## 4. 圧縮・エンコーディング

### 対応圧縮コーデック

| コーデック | 特性 | 推奨用途 |
|-----------|------|---------|
| **ZSTD** | 高速 + 高圧縮率 | **現在の推奨デフォルト** |
| **Snappy** | 高速 + 中程度圧縮率 | 速度重視 |
| **GZIP** | 最高圧縮率 + 低速 | ストレージコスト最小化 |
| **LZ4** | 最高速 + 低圧縮率 | 超高速処理 |
| **UNCOMPRESSED** | 非圧縮 | テスト用途 |

CSV/JSON 比で **2〜5倍のサイズ削減** が一般的。

### エンコーディング技術

| 技術 | 仕組み | 効果的なデータ |
|-----|--------|--------------|
| **Dictionary Encoding** | 繰り返し値を辞書インデックスに変換 | カテゴリ列・低カーディナリティ列 |
| **Run-Length Encoding (RLE)** | 連続する同一値をまとめる | NULL 値・ブール列 |
| **Delta Encoding** | 差分のみ格納 | 単調増加する数値列・タイムスタンプ |

---

## 5. スキーマ進化

| 操作 | サポート | 互換性 |
|-----|---------|--------|
| 列の追加（末尾） | ✅ | 後方互換性あり |
| 列の削除 | ✅ | 前方互換性あり |
| 列の型変更 | ⚠️ 一部のみ | 要注意 |
| 列のリネーム | ❌ | 非対応 |

**Spark での利用:**
```python
spark.conf.set("spark.sql.parquet.mergeSchema", "true")
# 異なるスキーマを持つ複数ファイルを自動マージ
```

---

## 6. エコシステム互換性

| カテゴリ | ツール・サービス |
|--------|----------------|
| **処理エンジン** | Apache Spark, Hive, Impala, DuckDB, Trino / Presto |
| **クラウド (AWS)** | Athena, Redshift Spectrum, Glue |
| **クラウド (GCP)** | BigQuery, Dataproc |
| **クラウド (Azure)** | Data Lake Storage, Synapse |
| **SaaS** | Snowflake, Databricks |
| **Python** | PyArrow, Pandas, PySpark, Polars |

---

## 7. 他フォーマットとの比較

| 特性 | **Parquet** | ORC | Avro | CSV | JSON |
|-----|:-----------:|:---:|:----:|:---:|:----:|
| ストレージ形式 | 列指向 | 列指向 | 行ベース | 行ベース | 行ベース |
| 圧縮効率 | ◎ | ◎ | ○ | △ | △ |
| 読み取り速度 | ◎ | ◎ | ○ | △ | ✕ |
| 書き込み速度 | △ | △ | ◎ | ○ | ○ |
| スキーマ進化 | ○ | ○ | ◎ | ✕ | ✕ |
| 人間可読性 | ✕ | ✕ | ✕ | ◎ | ◎ |
| ネスト構造 | ○ | ○ | ○ | ✕ | ◎ |
| ACID トランザクション | ✕ | ○ | ✕ | ✕ | ✕ |

### 用途別推奨フォーマット

| ユースケース | 推奨 |
|------------|------|
| 大規模分析・データレイク | **Parquet** |
| Hive 統合・ACID 必須 | **ORC** |
| ストリーミング・Kafka | **Avro** |
| API データ交換 | **JSON** |
| 簡易ファイル共有 | **CSV** |

---

## 8. ユースケース

### 向いている用途

- 大規模 OLAP クエリ・データウェアハウス
- クラウドデータレイク（S3, Azure Data Lake など）での長期保存
- AWS Athena / BigQuery でのサーバーレスアドホッククエリ
- 機械学習の訓練データ（特定列のみ読み込みで高速化）
- 複数チーム・組織間のデータシェアリング

### 向いていない用途

- 高頻度の行単位更新（OLTP）
- リアルタイムストリーミング書き込み
- 小規模・単発アクセス（CSV の方がシンプル）
- 人間が直接確認・編集する用途

---

## 9. Python での読み書き

### インストール

```bash
pip3 install pandas pyarrow
```

### 基本的な読み書き（Pandas）

```python
import pandas as pd

# 書き込み
df.to_parquet('data.parquet', compression='zstd')

# 読み込み（全列）
df = pd.read_parquet('data.parquet')

# 読み込み（特定列のみ） ← I/O削減・高速化
df = pd.read_parquet('data.parquet', columns=['name', 'salary'])
```

### PyArrow を使った書き込み（詳細制御）

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(df)
pq.write_table(
    table,
    'data.parquet',
    compression='zstd',
    use_dictionary=True,       # Dictionary Encoding 有効化
    row_group_size=100_000     # Row Group サイズ指定
)
```

### 大規模データの分割書き込み

```python
import pyarrow.parquet as pq
import pyarrow as pa

schema = pa.schema([('id', pa.int64()), ('value', pa.float64())])
writer = pq.ParquetWriter('large_file.parquet', schema, compression='zstd')

for chunk_df in large_dataframe_chunks:
    writer.write_table(pa.Table.from_pandas(chunk_df))

writer.close()
```

### メタデータの確認

```python
import pyarrow.parquet as pq

f = pq.ParquetFile('data.parquet')
print(f.schema)                              # スキーマ確認
print(f.metadata)                            # メタデータ確認
print(f.metadata.num_row_groups)             # Row Group 数

for i in range(f.metadata.num_row_groups):
    rg = f.metadata.row_group(i)
    print(f"Row Group {i}: {rg.num_rows} rows, {rg.total_byte_size} bytes")
```

### パーティション書き込み（Hive スタイル）

```python
import pyarrow.parquet as pq
import pyarrow as pa

table = pa.Table.from_pandas(df)

# category 列でパーティション分割して保存
# → output_dir/category=A/part-0.parquet, output_dir/category=B/...
pq.write_to_dataset(
    table,
    root_path='output_dir',
    partition_cols=['category']
)
```

---

## 10. パフォーマンスのポイント

| 最適化手法 | 説明 |
|----------|------|
| **必要列のみ読み込む** | `columns=['a', 'b']` 指定で不要な列の I/O をスキップ |
| **適切な Row Group サイズ** | 大きすぎると並列性低下、小さすぎるとオーバーヘッド増（目安: 64MB〜512MB） |
| **ZSTD 圧縮の使用** | 圧縮率・速度のバランスが最も良い |
| **パーティション分割** | クエリ条件に合わせたパーティションで読み込み量を削減 |
| **Dictionary Encoding** | カテゴリ列に有効。カーディナリティが低い列に特に効果的 |
| **フィルタリング (Predicate Pushdown)** | Row Group の統計を使って不要ブロックを読み込まずスキップ |

---

---

## 11. 実測ベンチマーク（本データセットで検証）

### 対象データ

- **ファイル**: 書店 POS 販売データ（日付・書店コード・ISBN・出版社・書名・著者名・分類・価格・販売冊数 の 12 列）
- **行数**: 3,969,881 行（約 397 万行）
- **環境**: macOS, Python 3.x, pandas 2.3.3, pyarrow 22.0.0
- **計測**: 各フォーマット 3 回読み込みの平均値

---

### ファイルサイズ比較

| フォーマット | ファイルサイズ | Parquet 比 |
|------------|-------------|-----------|
| TXT（タブ区切り） | 1,193 MB | **8.7x 大きい** |
| CSV | 664 MB | **4.8x 大きい** |
| **Parquet** | **137 MB** | 基準 |

---

### 全列読み込み速度比較（3回平均）

| 読み込み方法 | 平均読み込み時間 | 最速 | Parquet(pandas) 比 |
|------------|--------------|-----|-------------------|
| TXT → pandas | 6,880 ms | 6,525 ms | **7.4x 遅い** |
| CSV → pandas | 4,388 ms | 4,217 ms | **4.7x 遅い** |
| Parquet → pandas | 934 ms | 618 ms | 基準 |
| **Parquet → PyArrow** | **253 ms** | **171 ms** | **3.7x 速い** |
| Parquet 列絞り（数値2列のみ） | 13 ms | 11 ms | **72x 速い** |

---

### スループット比較

| 読み込み方法 | 実ファイルサイズ | スループット |
|------------|--------------|-----------|
| TXT → pandas | 1,193 MB | 173 MB/s |
| CSV → pandas | 664 MB | 151 MB/s |
| Parquet → pandas | 137 MB | 147 MB/s |
| **Parquet → PyArrow** | **137 MB** | **544 MB/s** |

> Parquet(PyArrow) は**読み込むバイト数が CSV の 1/4.8** かつ**スループットが 3.6 倍高い**ため、
> 実時間では CSV の **17 倍以上高速**（4,388 ms → 253 ms）。

---

### 考察：なぜ全列読み込みでも速いのか（実測から）

**1. I/O 削減が支配的（ファイルサイズ差 4.8〜8.7 倍）**

- CSV 664MB の読み込みに 4,388 ms かかるのに対し、Parquet 137MB は 253 ms（PyArrow）。
- ファイルサイズ差（4.8x）がほぼそのまま速度差に反映されており、**I/O 削減が主要因**であることが確認できる。

**2. PyArrow ネイティブ読み込みの効率（pandas 変換コスト）**

- Parquet → PyArrow（253 ms）と Parquet → pandas（934 ms）の差（約 3.7x）は、
  pandas の DataFrame 変換コストによるもの。
- 集計・フィルタリングだけなら PyArrow のまま操作することで大幅に速くなる。

**3. 列絞り込みの絶大な効果**

- 数値 2 列だけ読む場合は **13 ms**（CSV 全列の 338 分の 1）。
- 機械学習の特徴量選択後の読み込みや、集計クエリなど「必要な列だけ読む」設計が効果的。

---

### Python コード（ベンチマーク再現）

```python
import time
import pandas as pd
import pyarrow.parquet as pq

# CSV
t0 = time.perf_counter()
df_csv = pd.read_csv('data/data.csv', low_memory=False)
print(f"CSV:            {(time.perf_counter()-t0)*1000:.0f} ms")

# Parquet → pandas
t0 = time.perf_counter()
df_prq = pd.read_parquet('data/data.parquet')
print(f"Parquet(pandas): {(time.perf_counter()-t0)*1000:.0f} ms")

# Parquet → PyArrow（最速）
t0 = time.perf_counter()
tbl = pq.read_table('data/data.parquet')
print(f"Parquet(arrow):  {(time.perf_counter()-t0)*1000:.0f} ms")

# 列絞り込み
t0 = time.perf_counter()
df_cols = pd.read_parquet('data/data.parquet', columns=['本体価格', 'POS販売冊数'])
print(f"Parquet(2列):    {(time.perf_counter()-t0)*1000:.0f} ms")
```

---

*参考: Apache Parquet 公式ドキュメント (https://parquet.apache.org/docs/)*
