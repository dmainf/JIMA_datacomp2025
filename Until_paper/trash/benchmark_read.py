import os
import time
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

DATA_DIR = "/Users/dmainf/lab/JIMA_datacomp2025/data"
TXT_PATH = f"{DATA_DIR}/data.txt"
CSV_PATH = f"{DATA_DIR}/data.csv"
PRQ_PATH = f"{DATA_DIR}/data.parquet"
N_TRIALS = 3

def file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

def bench(label, fn, n=N_TRIALS):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean_ms = sum(times) / len(times) * 1000
    min_ms  = min(times) * 1000
    return mean_ms, min_ms, result

print("=" * 60)
print("ファイルサイズ比較")
print("=" * 60)
for label, path in [("TXT", TXT_PATH), ("CSV", CSV_PATH), ("Parquet", PRQ_PATH)]:
    mb = file_size_mb(path)
    print(f"  {label:8s}: {mb:8.1f} MB")

# Parquet を基準にした圧縮率
prq_mb = file_size_mb(PRQ_PATH)
txt_mb = file_size_mb(TXT_PATH)
csv_mb = file_size_mb(CSV_PATH)
print(f"\n  TXT / Parquet = {txt_mb / prq_mb:.1f}x 大きい")
print(f"  CSV / Parquet = {csv_mb / prq_mb:.1f}x 大きい")

print()
print("=" * 60)
print(f"読み込み速度ベンチマーク（{N_TRIALS}回平均）")
print("=" * 60)

results = {}

# TXT (タブ区切り)
mean, mn, df = bench("TXT", lambda: pd.read_csv(TXT_PATH, sep="\t", low_memory=False))
results["TXT"] = (mean, mn, len(df))
print(f"  TXT     全列: 平均 {mean:7.0f} ms  最速 {mn:7.0f} ms  行数: {len(df):,}")

# CSV
mean, mn, df = bench("CSV", lambda: pd.read_csv(CSV_PATH, low_memory=False))
results["CSV"] = (mean, mn, len(df))
print(f"  CSV     全列: 平均 {mean:7.0f} ms  最速 {mn:7.0f} ms  行数: {len(df):,}")

# Parquet 全列 (Pandas)
mean, mn, df = bench("Parquet(pandas)", lambda: pd.read_parquet(PRQ_PATH))
results["Parquet_all"] = (mean, mn, len(df))
print(f"  Parquet 全列: 平均 {mean:7.0f} ms  最速 {mn:7.0f} ms  行数: {len(df):,}")

# Parquet 全列 (PyArrow native)
mean, mn, tbl = bench("Parquet(arrow)", lambda: pq.read_table(PRQ_PATH))
results["Parquet_arrow"] = (mean, mn, len(tbl))
print(f"  Parquet 全列(PyArrow): 平均 {mean:7.0f} ms  最速 {mn:7.0f} ms  行数: {len(tbl):,}")

# Parquet 列絞り込み (数値列のみ)
num_cols = ["本体価格", "POS販売冊数"]
mean, mn, df = bench("Parquet(cols)", lambda: pd.read_parquet(PRQ_PATH, columns=num_cols))
results["Parquet_cols"] = (mean, mn, len(df))
print(f"  Parquet 列絞り({num_cols}): 平均 {mean:7.0f} ms  最速 {mn:7.0f} ms  行数: {len(df):,}")

print()
print("=" * 60)
print("速度比較（Parquet全列 Pandas を基準）")
print("=" * 60)
base = results["Parquet_all"][0]
for label, (mean, mn, rows) in results.items():
    ratio = mean / base
    bar = "▌" * int(ratio * 10)
    print(f"  {label:25s}: {ratio:5.2f}x  {bar}")

print()
print("=" * 60)
print("スループット比較（MB/s）")
print("=" * 60)
for label, path, key in [
    ("TXT", TXT_PATH, "TXT"),
    ("CSV", CSV_PATH, "CSV"),
    ("Parquet 全列(pandas)", PRQ_PATH, "Parquet_all"),
    ("Parquet 全列(pyarrow)", PRQ_PATH, "Parquet_arrow"),
]:
    mb = file_size_mb(path)
    mean_s = results[key][0] / 1000
    throughput = mb / mean_s
    print(f"  {label:25s}: {throughput:7.1f} MB/s  ({mb:.0f} MB / {results[key][0]:.0f} ms)")
