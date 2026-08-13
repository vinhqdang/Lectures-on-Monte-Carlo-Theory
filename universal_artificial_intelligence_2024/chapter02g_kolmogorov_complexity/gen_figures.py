"""
Generate all figures needed for chapter02g_kolmogorov_complexity_slides.tex

Book: An Introduction to Universal Artificial Intelligence (2024)
Section covered: 2.7 Kolmogorov Complexity (pp. 91-107)

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(HERE),
    "An Introduction to Universal Artificial Intelligence 2024.pdf",
)

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Figure 1 (crop): Figure 2.19 from the book -- schematic plot of K(x).
# We reproduce it via a crop from the book PDF (it is a hand-drawn style
# schematic, not a plottable closed-form function), using PyMuPDF.
# ---------------------------------------------------------------------------
def crop_book_figure_2_19():
    import fitz  # PyMuPDF
    import PIL.Image
    import io

    doc = fitz.open(BOOK_PDF)
    # Printed page 98 (0-indexed PDF page = printed_page + 20, verified against
    # running headers "98 CHAPTER 2. BACKGROUND" / "2.7. KOLMOGOROV COMPLEXITY 99").
    PAGE_INDEX = 118
    page = doc[PAGE_INDEX]
    mat = fitz.Matrix(4, 4)  # 4x super-sampling for crisp crop
    pix = page.get_pixmap(matrix=mat)
    img = PIL.Image.open(io.BytesIO(pix.tobytes('png')))
    # Crop box found by inspection (pixel coords at 4x scale, page is 504x720pt)
    crop = img.crop((560, 200, 1550, 840))
    crop.save(os.path.join(FIG_DIR, 'kcomplexity_graph.png'))
    print("Saved kcomplexity_graph.png (cropped from book p.98, Figure 2.19)")


# ---------------------------------------------------------------------------
# Figure 2: Simple vs random-looking vs truly-random string illustration.
# Motivational figure for Section 2.7.1 -- three bit strings of the same
# length, and how compressible each one is.
# ---------------------------------------------------------------------------
def fig_three_strings():
    x = "10" * 20
    y = "1100100100001111110110101010010001000100001"[:40]
    z = "10111011001101011111100011101011010111011000"[:40]

    fig, axes = plt.subplots(3, 1, figsize=(9, 4.2))
    strings = [
        (x, "x  (simple pattern: '10' repeated 20 times)", "#2b6cb0"),
        (y, "y  (first 40 bits of the binary expansion of $\\pi$)", "#6b46c1"),
        (z, "z  (physically measured 'true' random bits)", "#b83280"),
    ]
    for ax, (s, label, color) in zip(axes, strings):
        bits = [int(b) for b in s]
        ax.bar(range(len(bits)), bits, width=0.9, color=color)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(label, fontsize=10, loc='left')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Same length, very different Kolmogorov complexity", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG_DIR, 'three_strings.pdf'))
    plt.close(fig)
    print("Saved three_strings.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Fraction of strings compressible by more than k bits (Thm 2.7.13)
# 2^{-n}|{x : l(C(x)) < n-k}| < 2^{-k}
# ---------------------------------------------------------------------------
def fig_compressible_fraction():
    k = np.arange(0, 21)
    frac = 2.0 ** (-k)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(k, frac, 'o-', color="#2b6cb0", lw=2, markersize=5)
    ax.set_xlabel(r'$k$ = number of bits saved by compression')
    ax.set_ylabel(r'fraction of length-$n$ strings compressible by $>k$ bits')
    ax.set_title(r'Fewer than $2^{-k}$ of all strings compress by more than $k$ bits')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'compressible_fraction.pdf'))
    plt.close(fig)
    print("Saved compressible_fraction.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Dovetailing schedule illustration for approximating K from above
# (Theorem 2.7.28) -- which machine runs at which global time step.
# ---------------------------------------------------------------------------
def fig_dovetail():
    # Standard dovetailing schedule: machine i gets run at steps where a
    # diagonal enumeration of (i, t) pairs visits it. We draw the classic
    # triangular schedule.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    n_machines = 6
    n_steps = 9
    schedule = []
    # anti-diagonal enumeration: (machine, local_step) pairs in order
    order = []
    for s in range(1, n_steps + n_machines):
        for m in range(1, n_machines + 1):
            t = s - m + 1
            if 1 <= t <= n_steps:
                order.append((m, t))
    for global_t, (m, t) in enumerate(order[:40], start=1):
        schedule.append((global_t, m, t))

    for global_t, m, t in schedule:
        ax.scatter(global_t, m, color="#2b6cb0", s=140, zorder=3)
        ax.text(global_t, m, str(t), color='white', ha='center', va='center',
                fontsize=7, zorder=4)

    ax.set_yticks(range(1, n_machines + 1))
    ax.set_yticklabels([rf'$T_{{{i}}}$' for i in range(1, n_machines + 1)])
    ax.set_xlabel('global time step of the single dovetailing machine')
    ax.set_ylabel('simulated machine')
    ax.set_title('Dovetailing: interleaving infinitely many machines $T_1,T_2,\\dots$')
    ax.set_xlim(0, 41)
    ax.set_ylim(0.5, n_machines + 0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dovetail.pdf'))
    plt.close(fig)
    print("Saved dovetail.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Symmetry-of-information Venn-style diagram
# K(x,y) =+ K(x) + K(y|x) =+ K(y) + K(x|y)
# ---------------------------------------------------------------------------
def fig_symmetry_of_info():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axis('off')

    # Draw two overlapping bars representing K(x) and K(y), with K(x,y) as
    # a bracket above, illustrating K(x)+K(y|x) ~= K(y)+K(x|y) ~= K(x,y).
    ax.barh(2, 5, left=0, height=0.8, color="#2b6cb0", alpha=0.85)
    ax.text(2.5, 2, r'$K(x)$', color='white', ha='center', va='center', fontsize=13)
    ax.barh(2, 3, left=5, height=0.8, color="#c05621", alpha=0.85)
    ax.text(6.5, 2, r'$K(y|x)$', color='white', ha='center', va='center', fontsize=12)

    ax.barh(0.6, 4, left=0, height=0.8, color="#c05621", alpha=0.85)
    ax.text(2, 0.6, r'$K(x|y)$', color='white', ha='center', va='center', fontsize=12)
    ax.barh(0.6, 4, left=4, height=0.8, color="#805ad5", alpha=0.85)
    ax.text(6, 0.6, r'$K(y)$', color='white', ha='center', va='center', fontsize=13)

    ax.annotate('', xy=(0, 3.1), xytext=(8, 3.1),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.2))
    ax.text(4, 3.35, r'$K(x,y)$ (total length of both bars, within $O(\log)$)',
            ha='center', fontsize=11)

    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-0.2, 3.8)
    ax.set_title('Symmetry of information: two routes to the same total length',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'symmetry_info.pdf'))
    plt.close(fig)
    print("Saved symmetry_info.pdf")


if __name__ == '__main__':
    crop_book_figure_2_19()
    fig_three_strings()
    fig_compressible_fraction()
    fig_dovetail()
    fig_symmetry_of_info()
    print("All figures generated in", FIG_DIR)
