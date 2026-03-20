import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

years = list(range(2000, 2025))

books_rate = [31.5, 31.5, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 38.5,
              39.0, 38.5, 38.0, 37.3, 36.0, 35.0, 34.0, 33.5, 33.0,
              32.0, 31.5, 32.0, 32.5, 33.4, 31.6]

mag_rate =   [31.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 36.0, 36.5, 37.0,
              37.5, 38.0, 38.5, 38.8, 39.0, 39.5, 39.5, 39.5, 40.0,
              40.0, 42.0, 43.0, 45.0, 47.3, 49.7]

books_confirmed = {2010: 39.0, 2013: 37.3, 2019: 32.0, 2021: 32.0, 2023: 33.4, 2024: 31.6}
mag_confirmed = {2006: 35.0, 2013: 38.8, 2019: 40.0, 2023: 47.3, 2024: 49.7}

BG = '#0f172a'
GRID = '#1e293b'
TEXT_DIM = '#94a3b8'
BLUE = '#60a5fa'
BLUE_FILL = '#1d4ed8'
RED = '#f87171'
RED_FILL = '#b91c1c'

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=BG)
fig.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.95)

ax.set_facecolor(BG)
ax.tick_params(colors=TEXT_DIM, labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.grid(axis='y', color=GRID, linewidth=0.6)
ax.set_xlim(1999, 2025.5)
ax.set_xticks(range(2000, 2026, 5))
ax.set_ylim(10, 56)
ax.set_ylabel('返品率 (%)', fontsize=13, color=TEXT_DIM)
ax.set_xlabel('年度', fontsize=13, color=TEXT_DIM)

ax.fill_between(years, books_rate, alpha=0.1, color=BLUE_FILL, zorder=1)
ax.plot(years, books_rate, color=BLUE, linewidth=2.2, zorder=3, solid_capstyle='round', label='書籍')
ax.fill_between(years, mag_rate, alpha=0.1, color=RED_FILL, zorder=1)
ax.plot(years, mag_rate, color=RED, linewidth=2.2, zorder=3, solid_capstyle='round', label='雑誌')

def annotate_confirmed(ax, confirmed, color, offset_y=10):
    for y, v in confirmed.items():
        ax.plot(y, v, 'o', color='white', markersize=6, zorder=5, markeredgewidth=0)
        ax.annotate(f'{v:.1f}%', (y, v), textcoords='offset points',
                    xytext=(0, offset_y), ha='center', fontsize=13,
                    color=color, fontweight='bold')

annotate_confirmed(ax, books_confirmed, BLUE)
annotate_confirmed(ax, mag_confirmed, RED, offset_y=-18)

leg = ax.legend(fontsize=13, loc='upper left', framealpha=0.3,
                edgecolor=GRID, facecolor=BG, labelcolor=[BLUE, RED])

fig.savefig('/Users/dmainf/lab/datacomp_2025/return_rate_chart.png',
            bbox_inches='tight', facecolor=BG, edgecolor='none')
print("Saved: return_rate_chart.png")
