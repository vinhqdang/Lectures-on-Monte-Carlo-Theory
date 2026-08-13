"""
Generate all figures for Chapter 2.6 "Computability Theory" Beamer slides.
Run with: conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


# ----------------------------------------------------------------------
# Figure 1: Turing machine schematic (tape + head + finite control)
# ----------------------------------------------------------------------
def fig_turing_machine():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    # tape cells
    n_cells = 11
    cell_w = 1.0
    x0 = 0.3
    y0 = 0.6
    symbols = ['B', 'B', '1', '0', '1', '1', '0', 'B', 'B', 'B', 'B']
    head_index = 4  # points at cell holding '1' (5th visible cell)

    for i in range(n_cells):
        x = x0 + i * cell_w
        rect = Rectangle((x, y0), cell_w, 1.0, fill=False, edgecolor='black', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y0 + 0.5, symbols[i], ha='center', va='center', fontsize=14)

    # finite control box
    fc_x, fc_y, fc_w, fc_h = 5.3, 3.0, 3.4, 1.6
    fc = FancyBboxPatch((fc_x, fc_y), fc_w, fc_h, boxstyle="round,pad=0.05",
                         edgecolor='navy', facecolor='#dbe7f5', linewidth=1.6)
    ax.add_patch(fc)
    ax.text(fc_x + fc_w / 2, fc_y + fc_h / 2 + 0.18, "finite control", ha='center', va='center',
            fontsize=13, fontweight='bold', color='navy')
    ax.text(fc_x + fc_w / 2, fc_y + fc_h / 2 - 0.28, r"state $q_i$, $\delta$", ha='center', va='center',
            fontsize=12, color='navy')

    # single connector/arrow: head position on tape up to the finite control
    hx = x0 + head_index * cell_w + cell_w / 2
    fc_bottom_x = fc_x + fc_w / 2
    ax.annotate('', xy=(fc_bottom_x, fc_y), xytext=(hx, y0 + 1.0),
                arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.2,
                                 connectionstyle='arc3,rad=0.0'))
    ax.text(hx - 0.15, y0 + 1.75, 'head', ha='right', va='center', color='crimson', fontsize=12)

    # caption for the transition function's action
    ax.text(fc_bottom_x, fc_y + fc_h + 0.35,
            r"reads symbol, writes symbol, moves $L$ or $R$", ha='center', fontsize=10.5, color='dimgray')

    ax.text(x0 + n_cells * cell_w / 2, y0 - 0.55, "tape (infinite in both directions)",
            ha='center', fontsize=11, color='dimgray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "turing_machine.pdf"), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: Implications among computable functions (Theorem 2.6.14)
# ----------------------------------------------------------------------
def fig_implications():
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, color='#eef3fb', edge='navy', fs=12):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.06",
                            edgecolor=edge, facecolor=color, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6,
                                      shrinkA=8, shrinkB=8))

    box(5, 9, 3.6, 1.0, "finitely computable", fs=12)
    box(5, 7, 3.6, 1.0, r"estimable $= \Delta_1^0$", fs=12)
    box(2.4, 5, 3.6, 1.0, r"$\Sigma_1^0 = $ lower semicomp.", fs=11)
    box(7.6, 5, 3.6, 1.0, r"$\Pi_1^0 = $ upper semicomp.", fs=11)
    box(5, 3, 3.6, 1.0, r"approximable $= \Delta_2^0$", fs=12)

    arrow(5, 8.48, 5, 7.52)
    arrow(4.55, 6.55, 3.2, 5.45)
    arrow(5.45, 6.55, 6.8, 5.45)
    arrow(2.85, 4.55, 4.5, 3.45)
    arrow(7.15, 4.55, 5.5, 3.45)

    ax.text(5, 1.2,
            "Arrow = implies. If both upper and lower semicomputable $\\Rightarrow$ estimable.",
            ha='center', fontsize=10.5, color='dimgray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "implications.pdf"), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Arithmetic hierarchy chain of strict inclusions
# ----------------------------------------------------------------------
def fig_arithmetic_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    labels_top = [r"$\Sigma_n^0$", r"$\Sigma_{n+1}^0$"]
    labels_bot = [r"$\Pi_n^0$", r"$\Pi_{n+1}^0$"]
    mid_labels = [r"$\Delta_n^0$", r"$\Delta_{n+1}^0$", r"$\Delta_{n+2}^0$"]

    xs_mid = [1.0, 5.5, 10.5]
    xs_top = [3.2, 8.0]
    xs_bot = [3.2, 8.0]
    y_mid, y_top, y_bot = 2.0, 3.0, 1.0

    for x, lab in zip(xs_mid, mid_labels):
        ax.text(x, y_mid, lab, ha='center', va='center', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.3', fc='#eef3fb', ec='navy'))
    for x, lab in zip(xs_top, labels_top):
        ax.text(x, y_top, lab, ha='center', va='center', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.3', fc='#fdf0e4', ec='darkorange'))
    for x, lab in zip(xs_bot, labels_bot):
        ax.text(x, y_bot, lab, ha='center', va='center', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.3', fc='#eafaea', ec='seagreen'))

    def sub(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4, shrinkA=14, shrinkB=14))

    # mid -> top/bot -> next mid
    sub(1.35, 2.05, 2.85, 2.9)
    sub(1.35, 1.95, 2.85, 1.1)
    sub(3.55, 2.9, 5.15, 2.05)
    sub(3.55, 1.1, 5.15, 1.95)
    sub(5.85, 2.05, 7.65, 2.9)
    sub(5.85, 1.95, 7.65, 1.1)
    sub(8.35, 2.9, 10.15, 2.05)
    sub(8.35, 1.1, 10.15, 1.95)

    ax.text(6, 3.85, r"$\cdots \subset \Delta_n^0 \subset \Sigma_n^0,\Pi_n^0 \subset \Delta_{n+1}^0 \subset \Sigma_{n+1}^0,\Pi_{n+1}^0 \subset \Delta_{n+2}^0 \subset \cdots$",
            ha='center', fontsize=11.5, color='dimgray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "arithmetic_hierarchy.pdf"), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: Upper/lower semicomputable convergence (numeric illustration)
# ----------------------------------------------------------------------
def fig_semicomputable_convergence():
    k = np.arange(1, 21)
    true_val = np.sqrt(2)
    # lower bound sequence: monotonically increasing rational approximations from below
    lower = true_val - 1.0 / (k + 0.3)
    lower = np.maximum.accumulate(lower)  # ensure monotone increasing
    # upper bound sequence: monotonically decreasing rational approximations from above
    upper = true_val + 1.0 / (k + 0.3)
    upper = np.minimum.accumulate(upper)  # ensure monotone decreasing

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(k, lower, 'o-', color='seagreen', label=r'$\phi(x,k)$ lower semicomp. (nondecreasing)')
    ax.plot(k, upper, 's-', color='darkorange', label=r'$\phi(x,k)$ upper semicomp. (nonincreasing)')
    ax.axhline(true_val, color='navy', linestyle='--', linewidth=1.5, label=r'true value $f(x)=\sqrt{2}$')
    ax.set_xlabel(r'$k$ (computation budget)')
    ax.set_ylabel('approximation value')
    ax.set_title('Monotone convergence of semicomputable approximations')
    ax.legend(fontsize=9.5, loc='center right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "semicomputable_convergence.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_turing_machine()
    fig_implications()
    fig_arithmetic_hierarchy()
    fig_semicomputable_convergence()
    print("All figures written to", OUTDIR)
