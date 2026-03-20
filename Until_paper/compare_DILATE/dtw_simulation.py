import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# ── カラーパレット ──────────────────────────────────────────────────────────
BG       = '#0d1117'
PANEL    = '#161b22'
PLOT_BG  = '#1c2128'
BORDER   = '#30363d'
A_CLR    = '#58a6ff'
B_CLR    = '#f85149'
CONN_CLR = '#c9d1d9'
PATH_CLR = '#d29922'
TXT      = '#c9d1d9'
MUTED    = '#8b949e'

# ── DTW アルゴリズム ────────────────────────────────────────────────────────
def compute_dtw(s1, s2):
    n, m = len(s1), len(s2)
    D = np.zeros((n, m))
    D[0, 0] = abs(s1[0] - s2[0])
    for i in range(1, n):
        D[i, 0] = D[i-1, 0] + abs(s1[i] - s2[0])
    for j in range(1, m):
        D[0, j] = D[0, j-1] + abs(s1[0] - s2[j])
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = abs(s1[i] - s2[j]) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # バックトラック
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([D[i-1, j-1], D[i-1, j], D[i, j-1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    return D, list(reversed(path)), D[n-1, m-1]

def alignment_label(cost, n):
    v = cost / max(n, 1)
    if v < 0.05: return "Perfect"
    if v < 0.2:  return "Excellent"
    if v < 0.5:  return "Good"
    if v < 1.0:  return "Fair"
    return "Poor"

def make_series(n, shift, warp):
    cycles = 3
    t = np.linspace(0, cycles * 2 * np.pi, n)
    A = np.sin(t)
    t_b = np.linspace(0, cycles * 2 * np.pi * warp, n)
    phase = cycles * 2 * np.pi * shift / n
    B = np.sin(t_b - phase)
    return A, B

# ── Figure セットアップ ─────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color':       TXT,
    'axes.labelcolor':  MUTED,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
})

fig = plt.figure(figsize=(10, 9), facecolor=BG)
try:
    fig.canvas.manager.set_window_title('Dynamic Time Warping (DTW)')
except Exception:
    pass

# メインプロット
ax_wave = fig.add_axes([0.05, 0.50, 0.90, 0.38], facecolor=PLOT_BG)
ax_cost = fig.add_axes([0.10, 0.14, 0.82, 0.32], facecolor=PLOT_BG)

# 統計パネル（右上）
ax_stat = fig.add_axes([0.54, 0.895, 0.43, 0.082], facecolor=PANEL)
ax_stat.set_xticks([]); ax_stat.set_yticks([])
for sp in ax_stat.spines.values():
    sp.set_edgecolor(BORDER)

# 縦区切り線
for xpos in [0.705, 0.845]:
    fig.add_artist(plt.Line2D(
        [xpos, xpos], [0.895, 0.978],
        transform=fig.transFigure, color=BORDER, linewidth=1, zorder=5))

# タイトル
fig.text(0.05, 0.97, 'Dynamic Time Warping (DTW)',
         fontsize=14, fontweight='bold', color=TXT, va='top')

# 統計ラベル
for xf, label in [(0.625, 'DATA POINTS'), (0.775, 'TOTAL COST'), (0.920, 'ALIGNMENT')]:
    fig.text(xf, 0.968, label, color=MUTED, fontsize=7, ha='center', va='top')

txt_n = fig.text(0.625, 0.920, '30',      color=TXT, fontsize=13, ha='center', va='top', fontweight='bold')
txt_c = fig.text(0.775, 0.920, '0.00',    color=TXT, fontsize=13, ha='center', va='top', fontweight='bold')
txt_a = fig.text(0.920, 0.920, 'Perfect', color=TXT, fontsize=13, ha='center', va='top', fontweight='bold')

# ── スライダー & ボタン ────────────────────────────────────────────────────
ax_sl_shift = fig.add_axes([0.19, 0.09, 0.22, 0.025], facecolor=PANEL)
ax_sl_warp  = fig.add_axes([0.60, 0.09, 0.30, 0.025], facecolor=PANEL)
ax_sl_pts   = fig.add_axes([0.19, 0.04, 0.22, 0.025], facecolor=PANEL)
ax_btn      = fig.add_axes([0.60, 0.025, 0.10, 0.04],  facecolor=PANEL)

fig.text(0.05, 0.102, 'Time Shift',  color=TXT, fontsize=10, va='center')
fig.text(0.05, 0.052, 'Points',      color=TXT, fontsize=10, va='center')
fig.text(0.465, 0.102, 'Warp Factor', color=TXT, fontsize=10, va='center')

sl_shift  = widgets.Slider(ax_sl_shift, '', -15, 15,   valinit=0,   valstep=1,   color=A_CLR)
sl_warp   = widgets.Slider(ax_sl_warp,  '', 0.25, 3.0, valinit=1.0,              color=A_CLR)
sl_pts    = widgets.Slider(ax_sl_pts,   '', 10,   80,   valinit=30,  valstep=1,   color=A_CLR)
btn_reset = widgets.Button(ax_btn, 'Reset', color=PANEL, hovercolor=BORDER)

for sl in [sl_shift, sl_warp, sl_pts]:
    sl.valtext.set_color(TXT)
    sl.valtext.set_fontweight('bold')
    sl.valtext.set_bbox(dict(facecolor=BORDER, edgecolor=BORDER, boxstyle='round,pad=0.3'))
btn_reset.label.set_color(TXT)

# ── 更新関数 ────────────────────────────────────────────────────────────────
def update(val=None):
    n     = int(sl_pts.val)
    shift = int(sl_shift.val)
    warp  = float(sl_warp.val)

    A, B = make_series(n, shift, warp)
    D, path, cost = compute_dtw(A, B)

    txt_n.set_text(str(n))
    txt_c.set_text(f'{cost:.2f}')
    txt_a.set_text(alignment_label(cost, n))

    # ── 波形アライメントプロット ──────────────────────────────────────────
    ax_wave.cla()
    ax_wave.set_facecolor(PLOT_BG)

    sep = 2.5
    x = np.arange(n)
    ax_wave.plot(x, A,       color=A_CLR, linewidth=2.5, zorder=3)
    ax_wave.plot(x, B - sep, color=B_CLR, linewidth=2.5, zorder=3)

    # アライメント接続線
    step = max(1, len(path) // 40)
    for ia, ib in path[::step]:
        ax_wave.plot([ia, ib], [A[ia], B[ib] - sep],
                     color=CONN_CLR, alpha=0.3, linewidth=1, zorder=2)

    # シリーズラベル
    ax_wave.text(0, A[0] + 0.25, 'SERIES A (REFERENCE)', color=A_CLR, fontsize=8,
                 bbox=dict(facecolor=PANEL, edgecolor=A_CLR, boxstyle='round,pad=0.3'))
    ax_wave.text(0, B[0] - sep - 0.5, 'SERIES B (QUERY)', color=B_CLR, fontsize=8,
                 bbox=dict(facecolor=PANEL, edgecolor=B_CLR, boxstyle='round,pad=0.3'))

    ax_wave.set_xlim(-1, n)
    ax_wave.set_ylim(B.min() - sep - 0.9, A.max() + 0.9)
    ax_wave.axis('off')
    for sp in ax_wave.spines.values():
        sp.set_visible(True); sp.set_edgecolor(BORDER)

    # ── 累積コスト行列プロット ────────────────────────────────────────────
    ax_cost.cla()
    ax_cost.set_facecolor(PLOT_BG)

    # D.T → x軸=Series A (Reference), y軸=Series B (Query)
    ax_cost.imshow(D.T, cmap='Blues', origin='lower', aspect='auto', interpolation='nearest')

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax_cost.plot(path_x, path_y, color=PATH_CLR, linewidth=2.5)

    ax_cost.set_xlabel('Series A (Reference Index)', fontsize=9)
    ax_cost.set_ylabel('Series B (Query Index)',     fontsize=9)
    ax_cost.set_title('CUMULATIVE COST MATRIX', fontsize=9, color=TXT, pad=8,
                      bbox=dict(facecolor=PANEL, edgecolor=BORDER, boxstyle='round,pad=0.4'))
    ax_cost.tick_params(labelsize=8)
    for sp in ax_cost.spines.values():
        sp.set_edgecolor(BORDER)

    fig.canvas.draw_idle()

sl_shift.on_changed(update)
sl_warp.on_changed(update)
sl_pts.on_changed(update)
btn_reset.on_clicked(lambda e: [sl_shift.reset(), sl_warp.reset(), sl_pts.reset()])

update()
plt.show()
