import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

BG = '#0f172a'
GRID = '#1e293b'
TEXT = '#e2e8f0'
TEXT_DIM = '#94a3b8'
CYAN = '#22d3ee'
CYAN_FILL = '#0e7490'

np.random.seed(42)

n_titles = 3_000_000
alpha = 1.8
xmin = 1

samples = (xmin * (1 - np.random.uniform(0, 1, n_titles)) ** (-1 / (alpha - 1))).astype(int)
samples = np.clip(samples, 1, 10_000_000)

bin_edges = np.logspace(0, 7, 80)
counts, edges = np.histogram(samples, bins=bin_edges)
bin_centers = np.sqrt(edges[:-1] * edges[1:])

mask = counts > 0
bin_centers = bin_centers[mask]
counts = counts[mask]

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150, facecolor=BG)
ax.set_facecolor(BG)

ax.fill_between(bin_centers, counts, alpha=0.15, color=CYAN_FILL, zorder=1)
ax.plot(bin_centers, counts, color=CYAN, linewidth=2.2, zorder=3, solid_capstyle='round')

ax.set_xscale('log')
ax.set_xlabel('売れた冊数（1タイトルあたり）', fontsize=13, color=TEXT_DIM)
ax.set_ylabel('タイトル数', fontsize=13, color=TEXT_DIM)
ax.set_title('書籍市場のロングテール分布', fontsize=16, fontweight='bold',
             color=TEXT, loc='left', pad=12)

ax.set_xlim(0.8, 1.2e7)
ax.set_ylim(0, counts.max() * 1.1)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{int(x):,}' if x >= 1 else ''))

ax.tick_params(colors=TEXT_DIM, labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.grid(True, which='both', color=GRID, linewidth=0.6, linestyle='-')

peak_i = int(np.argmax(counts))
ax.annotate(f'最頻値: {int(bin_centers[peak_i]):,} 冊付近\n({counts[peak_i]:,} タイトル)',
            xy=(bin_centers[peak_i], counts[peak_i]),
            xytext=(bin_centers[peak_i] * 8, counts[peak_i] * 0.85),
            fontsize=9, color=TEXT_DIM,
            arrowprops=dict(arrowstyle='->', color=TEXT_DIM, lw=0.8),
            ha='left')

tail_start = 1000
tail_mask = bin_centers >= tail_start
tail_titles = counts[tail_mask].sum()
ax.axvline(x=tail_start, color=TEXT_DIM, linestyle='--', alpha=0.3, linewidth=0.8)
ax.text(tail_start * 1.3, counts.max() * 0.7,
        f'{tail_start:,}冊以上:\n{tail_titles:,} タイトル',
        fontsize=8, color=TEXT_DIM, va='top')

ax.text(0.97, 0.02, '参考: Fenner et al. (2010), NPD BookScan',
        transform=ax.transAxes, fontsize=7.5, color=TEXT_DIM,
        va='bottom', ha='right')

plt.tight_layout()
plt.savefig('/Users/dmainf/lab/datacomp_2025/book_market_longtail.png',
            dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
print("保存完了: book_market_longtail.png")
