import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

fm.fontManager.addfont('/usr/local/texlive/2026/texmf-dist/fonts/truetype/public/ipaex/ipaexm.ttf')
import matplotlib as mpl
mpl.rc('font', family='IPAexMincho')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EPS_DIR = os.path.join(BASE_PATH, 'eps')
PNG_DIR = os.path.join(BASE_PATH, 'png')
os.makedirs(EPS_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

ACTUAL_LABEL = '実績値'
CEIL_FACTOR = 1.25   # 実測値ベースの天井（主パネル上限 = 実測最大 × この係数）

BOOKS = [
    ('ぼっち・ざ・ろっく！_original', 'figure_book_bocchi'),
    ('デラックスクロスワード_original', 'figure_book_crossword'),
]

MODELS_BASE = [
    ('GateRAF',   'GateRAF',   '#d6336c', '-',  1.6),
    ('Multi-RAF', 'Multi-RAF', '#1971c2', '--', 1.2),
    ('ProtoRAF',  'ProtoRAF',  '#2f9e44', ':',  1.2),
    ('baseline',  'Baseline',  '#e67700', '-.', 1.0),
]

SARIMA  = ('SARIMA',  'SARIMA',  '#0c8599', (0, (5, 1)),       1.0)
PROPHET = ('Prophet', 'Prophet', '#7048e8', (0, (3, 1, 1, 1)), 1.0)

VARIANTS = [
    ('sarima',         [SARIMA]),
    ('prophet',        [PROPHET]),
    ('sarima_prophet', [SARIMA, PROPHET]),
]

all_files = set(m[0] for m in MODELS_BASE) | {SARIMA[0], PROPHET[0]}
model_dfs = {f: pd.read_csv(os.path.join(BASE_PATH, f'{f}.csv')) for f in all_files}


def wave_at(ax, yfrac):
    """ax の縦軸方向 yfrac の高さに、横全幅の波線（軸スキップ記号）を描く。"""
    bb = ax.get_window_extent()
    amp = 3.2 / bb.height           # 軸座標での振幅（約3pxに固定）
    ncyc = max(8, int(bb.width / 13))
    xs = np.linspace(0, 1, ncyc * 24)
    ys = yfrac + amp * np.sin(2 * np.pi * ncyc * xs)
    ax.plot(xs, ys, transform=ax.transAxes, color='k', lw=1.1,
            clip_on=False, zorder=11, solid_capstyle='round')


def plot_book(book_key, suffix, models, stem):
    sub_ref = model_dfs['GateRAF'][model_dfs['GateRAF']['書名'] == book_key].reset_index(drop=True)
    actual = sub_ref['actual'].values
    days = np.arange(len(actual))
    maxx = len(actual) - 1

    series = []
    for mf, label, color, ls, lw in models:
        q05 = model_dfs[mf][model_dfs[mf]['書名'] == book_key].reset_index(drop=True)['q0.5'].values
        mae = np.abs(actual - q05).mean()
        series.append((q05, f'{label} (MAE={mae:.2f})', color, ls, lw))

    ceil = actual.max() * CEIL_FACTOR
    qmax = max(s[0].max() for s in series)
    qmin = min(s[0].min() for s in series)
    need_top = qmax > ceil * 1.02
    need_bot = qmin < -ceil * 0.05

    ratios, kinds = [], []
    if need_top:
        ratios.append(2.2); kinds.append('top')
    ratios.append(4.0); kinds.append('main')
    if need_bot:
        ratios.append(2.2); kinds.append('bot')

    fig_h = 3.2 + 1.5 * (need_top + need_bot)
    fig, axes = plt.subplots(len(kinds), 1, sharex=True, figsize=(8, fig_h),
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.08})
    axes = np.atleast_1d(axes)
    ax_by = dict(zip(kinds, axes))
    ax_main = ax_by['main']

    def draw_series(ax):
        for q05, lbl, color, ls, lw in series:
            ax.plot(days, q05, color=color, linestyle=ls, linewidth=lw, label=lbl, zorder=3)
        ax.plot(days, actual, color='black', linewidth=2.0, label=ACTUAL_LABEL, zorder=4)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, maxx)

    for ax in axes:
        draw_series(ax)

    nb = ceil * 0.10                      # 主パネルの負側余白（0と波線の間隔）
    main_bottom = -nb if need_bot else 0
    ax_main.set_ylim(main_bottom, ceil)
    if need_top:
        ax_by['top'].set_ylim(ceil, ceil + (qmax - ceil) * 1.18)
        ax_by['top'].spines['bottom'].set_visible(False)
        ax_main.spines['top'].set_visible(False)
        ax_by['top'].tick_params(labelbottom=False, bottom=False)
        ax_by['top'].yaxis.set_major_locator(plt.MaxNLocator(3))
    if need_bot:
        ax_by['bot'].set_ylim(qmin * 1.18, main_bottom)
        ax_by['bot'].spines['top'].set_visible(False)
        ax_main.spines['bottom'].set_visible(False)
        ax_by['bot'].yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel('予測日', fontsize=9)
    fig.supylabel('販売冊数', fontsize=9, x=0.04)
    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.01),
               bbox_transform=axes[0].transAxes, ncol=4, fontsize=8,
               framealpha=0.9, columnspacing=1.2)

    fig.canvas.draw()
    if need_bot:  # 主パネルは 0 以上のみ表示、下パネルは境界(負)より下のみ表示
        ax_main.set_yticks([t for t in ax_main.get_yticks() if -1e-6 <= t <= ceil])
        ax = ax_by['bot']
        ax.set_yticks([t for t in ax.get_yticks() if qmin * 1.18 <= t < main_bottom])
    if need_top:
        wave_at(ax_main, 1.0)
        wave_at(ax_by['top'], 0.0)
    if need_bot:
        wave_at(ax_main, 0.0)
        wave_at(ax_by['bot'], 1.0)

    name = f'{stem}_{suffix}'
    fig.savefig(os.path.join(EPS_DIR, f'{name}.eps'), format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PNG_DIR, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}  (top={need_top}, bottom={need_bot})")


def print_mae_summary():
    all_models = MODELS_BASE + [SARIMA, PROPHET]
    headers = [b[1] for b in BOOKS] + ['全書名']
    print(f"\n{'Model':<10} " + " ".join(f"{h:>26}" for h in headers))
    for mf, label, *_ in all_models:
        df = model_dfs[mf]
        maes = []
        for book_key, _ in BOOKS:
            sub = df[df['書名'] == book_key]
            maes.append(np.abs(sub['actual'] - sub['q0.5']).mean())
        maes.append(np.abs(df['actual'] - df['q0.5']).mean())
        print(f"{label:<10} " + " ".join(f"{m:>26.4f}" for m in maes))
    print()


print_mae_summary()

for book_key, stem in BOOKS:
    for suffix, extra in VARIANTS:
        plot_book(book_key, suffix, MODELS_BASE + extra, stem)
