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
    nb = ceil * 0.10                       # 主パネルの負側余白（0と波線の間隔）
    qmax = max(s[0].max() for s in series)
    qmin = min(s[0].min() for s in series)

    need_top = qmax > ceil * 1.02
    deep_bot = qmin < -ceil                # 負が天井を超えて深い時だけ波線で省略
    if deep_bot:
        main_bottom = -nb
    elif qmin < 0:                         # 浅い負は波線を使わず主パネルにそのまま含める
        main_bottom = qmin * 1.1
    else:
        main_bottom = 0

    ratios, kinds = [], []
    if need_top:
        ratios.append(2.2); kinds.append('top')
    ratios.append(4.0); kinds.append('main')
    if deep_bot:
        ratios.append(2.2); kinds.append('bot')

    fig_h = 3.2 + 1.5 * (need_top + deep_bot)
    fig, axes = plt.subplots(len(kinds), 1, sharex=True, figsize=(8, fig_h),
                             gridspec_kw={'height_ratios': ratios, 'hspace': 0.08})
    axes = np.atleast_1d(axes)
    ax_by = dict(zip(kinds, axes))
    ax_main = ax_by['main']

    for ax in axes:
        for q05, lbl, color, ls, lw in series:
            ax.plot(days, q05, color=color, linestyle=ls, linewidth=lw, label=lbl, zorder=3)
        ax.plot(days, actual, color='black', linewidth=2.0, label=ACTUAL_LABEL, zorder=4)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, maxx)

    ax_main.set_ylim(main_bottom, ceil)
    if need_top:                           # 上パネルは全時点の上振れを圧縮表示
        ax_t = ax_by['top']
        ax_t.set_ylim(ceil, ceil + (qmax - ceil) * 1.18)
        ax_t.spines['bottom'].set_visible(False)
        ax_main.spines['top'].set_visible(False)
        ax_t.tick_params(labelbottom=False, bottom=False)
        ax_t.yaxis.set_major_locator(plt.MaxNLocator(3))
    if deep_bot:                           # 下パネルは全時点の下振れを圧縮表示
        ax_b = ax_by['bot']
        ax_b.set_ylim(qmin * 1.18, main_bottom)
        ax_b.spines['top'].set_visible(False)
        ax_main.spines['bottom'].set_visible(False)
        ax_b.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel('予測日', fontsize=9)
    fig.supylabel('販売冊数', fontsize=9, x=0.04)
    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.01),
               bbox_transform=axes[0].transAxes, ncol=4, fontsize=8,
               framealpha=0.9, columnspacing=1.2)

    fig.canvas.draw()
    if need_top:
        wave_at(ax_main, 1.0)
        wave_at(ax_by['top'], 0.0)
    if deep_bot:
        wave_at(ax_main, 0.0)
        wave_at(ax_by['bot'], 1.0)

    name = f'{stem}_{suffix}'
    fig.savefig(os.path.join(EPS_DIR, f'{name}.eps'), format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PNG_DIR, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}  (top={need_top}, bottom={deep_bot})")


def print_mae_summary():
    all_models = MODELS_BASE + [SARIMA, PROPHET]
    scopes = [(b[0].split('_')[0], b[0]) for b in BOOKS] + [('全書名', None)]
    for scope_name, book_key in scopes:
        for mf, *_ in [MODELS_BASE[0]]:
            base_df = model_dfs[mf]
            sub = base_df if book_key is None else base_df[base_df['書名'] == book_key]
            pos = sub['actual'].sum()
        print(f"\n=== {scope_name}  (POS販売冊数={pos:.0f}) ===")
        print(f"{'Model':<10}{'過剰入荷':>12}{'入荷数':>12}{'返本率':>11}"
              f"{'機会損失':>12}{'機会損失率':>12}")
        for mf, label, *_ in all_models:
            df = model_dfs[mf]
            sub = df if book_key is None else df[df['書名'] == book_key]
            err = sub['q0.5'] - sub['actual']
            over = err.clip(lower=0).sum()        # 過剰入荷（返本の素）= Σmax(q0.5−実測,0)
            short = (-err).clip(lower=0).sum()     # 機会損失       = Σmax(実測−q0.5,0)
            nyuka = pos + over
            print(f"{label:<10}{over:>12.1f}{nyuka:>12.1f}{over/nyuka*100:>10.2f}%"
                  f"{short:>12.1f}{short/pos*100:>11.2f}%")
    print()


print_mae_summary()

for book_key, stem in BOOKS:
    for suffix, extra in VARIANTS:
        plot_book(book_key, suffix, MODELS_BASE + extra, stem)
