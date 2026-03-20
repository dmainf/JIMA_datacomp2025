import os
import time
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Hiragino Sans'

DATA_DIR = "/Users/dmainf/lab/JIMA_datacomp2025/data"
TXT_PATH = f"{DATA_DIR}/data.txt"
CSV_PATH = f"{DATA_DIR}/data.csv"
PRQ_PATH = f"{DATA_DIR}/data.parquet"
N_TRIALS = 3

def file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

def bench(fn, n=N_TRIALS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000  # ms

print("計測中...")
results = {
    "TXT\n(pandas)":        bench(lambda: pd.read_csv(TXT_PATH, sep="\t", low_memory=False)),
    "CSV\n(pandas)":        bench(lambda: pd.read_csv(CSV_PATH, low_memory=False)),
    "Parquet\n(pandas)":    bench(lambda: pd.read_parquet(PRQ_PATH)),
    "Parquet\n(PyArrow)":   bench(lambda: pq.read_table(PRQ_PATH)),
    "Parquet\n列絞り2列":    bench(lambda: pd.read_parquet(PRQ_PATH, columns=["本体価格", "POS販売冊数"])),  # 計測のみ（グラフ非表示）
}
sizes = {
    "TXT":     file_size_mb(TXT_PATH),
    "CSV":     file_size_mb(CSV_PATH),
    "Parquet": file_size_mb(PRQ_PATH),
}
print("計測完了")

COLORS = {
    "TXT\n(pandas)":      "#e07b54",
    "CSV\n(pandas)":      "#e0b454",
    "Parquet\n(pandas)":  "#5488c8",
    "Parquet\n(PyArrow)": "#3a6ea8",
    "Parquet\n列絞り2列":  "#2a9d6a",
}
SIZE_COLORS = {"TXT": "#e07b54", "CSV": "#e0b454", "Parquet": "#5488c8"}

SUBTITLE = "書店POS販売データ, 397万行 × 12列"

# ── 1. ファイルサイズ ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
labels = list(sizes.keys())
vals   = list(sizes.values())
colors = [SIZE_COLORS[l] for l in labels]
bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
ax.set_title(f"ファイルサイズ\n({SUBTITLE})", fontsize=12, pad=10)
ax.set_ylabel("MB")
ax.set_ylim(0, max(vals) * 1.3)
base_size = sizes["CSV"]
gap = max(vals) * 0.02
for bar, v in zip(bars, vals):
    ratio = v / base_size
    cx = bar.get_x() + bar.get_width() / 2
    ax.text(cx, v / 2,       f"{ratio*100:.0f}%", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(cx, v + gap,     f"{v:.0f} MB",   ha="center", va="bottom", fontsize=9, color="#555")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out1 = "/Users/dmainf/lab/JIMA_datacomp2025/benchmark_filesize.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"保存: {out1}")

# ── 2. 読み込み時間 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
plot_keys = ["TXT\n(pandas)", "CSV\n(pandas)", "Parquet\n(pandas)", "Parquet\n(PyArrow)"]
labels2 = plot_keys
vals2   = [results[k] for k in plot_keys]
colors2 = [COLORS[l] for l in labels2]
bars2 = ax.bar(labels2, vals2, color=colors2, width=0.55, edgecolor="white", linewidth=1.2)
ax.set_title(f"全列読み込み時間（3回平均）\n({SUBTITLE})", fontsize=12, pad=10)
ax.set_ylabel("ms")
ax.set_ylim(0, max(vals2) * 1.3)
base = results["CSV\n(pandas)"]
gap2 = max(vals2) * 0.02
for bar, v, key in zip(bars2, vals2, plot_keys):
    ratio = v / base
    cx = bar.get_x() + bar.get_width() / 2
    if key == "Parquet\n(PyArrow)":
        # バーが小さいので値の上に表示
        ax.text(cx, v + gap2,       f"{v:.0f} ms",   ha="center", va="bottom", fontsize=9, color="#555")
        ax.text(cx, v + gap2 * 5,   f"{ratio*100:.0f}%", ha="center", va="bottom", fontsize=9, color="#333", fontweight="bold")
    else:
        ax.text(cx, v / 2,   f"{ratio*100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(cx, v + gap2, f"{v:.0f} ms",   ha="center", va="bottom", fontsize=9, color="#555")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out2 = "/Users/dmainf/lab/JIMA_datacomp2025/benchmark_readtime.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"保存: {out2}")

# ── 3. スループット ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
throughput_data = {
    "TXT\n(pandas)":      (sizes["TXT"],     results["TXT\n(pandas)"]),
    "CSV\n(pandas)":      (sizes["CSV"],      results["CSV\n(pandas)"]),
    "Parquet\n(pandas)":  (sizes["Parquet"],  results["Parquet\n(pandas)"]),
    "Parquet\n(PyArrow)": (sizes["Parquet"],  results["Parquet\n(PyArrow)"]),
}
tp_labels = list(throughput_data.keys())
tp_vals   = [mb / (ms / 1000) for mb, ms in throughput_data.values()]
tp_colors = [COLORS[l] for l in tp_labels]
bars3 = ax.bar(tp_labels, tp_vals, color=tp_colors, width=0.55, edgecolor="white", linewidth=1.2)
ax.set_title(f"スループット（ファイルサイズ ÷ 読込時間）\n({SUBTITLE})", fontsize=12, pad=10)
ax.set_ylabel("MB/s")
ax.set_ylim(0, max(tp_vals) * 1.3)
base_tp = tp_vals[1]  # CSV(pandas) 基準
gap3 = max(tp_vals) * 0.02
for bar, v in zip(bars3, tp_vals):
    ratio = v / base_tp
    cx = bar.get_x() + bar.get_width() / 2
    ax.text(cx, v / 2,   f"{ratio*100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(cx, v + gap3, f"{v:.0f} MB/s", ha="center", va="bottom", fontsize=9, color="#555")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out3 = "/Users/dmainf/lab/JIMA_datacomp2025/benchmark_throughput.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"保存: {out3}")
