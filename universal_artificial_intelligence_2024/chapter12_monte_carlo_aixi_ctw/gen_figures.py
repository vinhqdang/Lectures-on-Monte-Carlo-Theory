"""
Generate all figures needed for Chapter 12 (Monte Carlo AIXI with Context
Tree Weighting) slides.

Two kinds of figures:
  1. Diagrams cropped directly from the book PDF with PyMuPDF (fitz), since
     they are hand-drawn architecture/tree diagrams that are not easily
     reproduced by plotting code.
  2. A matplotlib-generated illustrative plot of the UCB-vs-UCT bandit
     behaviour described in Example 12.2.5 (not in the book -- built from
     scratch to make the textual argument concrete with real numbers).

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import os
import fitz
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_PDF = os.path.join(os.path.dirname(HERE),
                         "An Introduction to Universal Artificial Intelligence 2024.pdf")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Crop diagrams from the book PDF.
#    Page mapping: fitz page index (0-indexed) = printed_page_number + 20
#    (verified: printed page 315 == filename p336.png == fitz index 335)
# ----------------------------------------------------------------------
doc = fitz.open(BOOK_PDF)
ZOOM = 4
mat = fitz.Matrix(ZOOM, ZOOM)

# crops[name] = (printed_page_number, (x0,y0,x1,y1) in PDF points, out padding)
crops = {
    "fig12_1_loop":        (317, (35, 50, 468, 330)),
    "fig12_2_expectimax":  (318, (30, 50, 470, 285)),
    "fig12_3_mcts_steps":  (320, (30, 222, 470, 465)),
    "fig12_5_grid":        (340, (85, 55, 215, 190)),
    "fig12_6_cheese":      (340, (225, 50, 445, 205)),
    "fig12_7_kuhn":        (341, (55, 235, 445, 495)),
    "fig12_8_pocman":      (342, (120, 50, 445, 350)),
    "fig12_9_performance": (342, (85, 405, 445, 625)),
    "fig12_10_grid_explore": (344, (85, 50, 445, 193)),
    "fig12_11_reward_curves": (344, (50, 465, 465, 602)),
}

for name, (printed_page, rect) in crops.items():
    page_index = printed_page + 20
    page = doc[page_index]
    clip = fitz.Rect(*rect)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    outpath = os.path.join(OUTDIR, f"{name}.png")
    pix.save(outpath)
    print("saved", outpath, pix.width, pix.height)

doc.close()

# ----------------------------------------------------------------------
# 2. Illustrative plot: UCB' (no log term) vs UCT (with log term) on the
#    two-armed bandit of Example 12.2.5.
#      a1: deterministic reward 1
#      a2: reward 10'000 with probability 0.001, else 0 (expected value 10)
#    We simulate both action-selection rules and plot how many times each
#    chooses a2 over time, to make concrete the claim that UCB' can get
#    permanently stuck on the suboptimal arm a1 while UCT (with the log
#    term) keeps exploring a2 infinitely often.
# ----------------------------------------------------------------------
rng = np.random.default_rng(7)

def simulate(use_log, T=20000, C=2.0, m=10, seed=0):
    rng = np.random.default_rng(seed)
    N = np.zeros(2)      # visit counts
    Q = np.zeros(2)      # value estimates
    picks_a2 = np.zeros(T, dtype=int)
    for t in range(1, T + 1):
        # UCB-style score; infinite for unvisited actions
        scores = np.empty(2)
        for a in range(2):
            if N[a] == 0:
                scores[a] = np.inf
            else:
                bonus = C * np.sqrt((np.log(t) if use_log else 1.0) / N[a])
                scores[a] = Q[a] / m + bonus
        a = int(np.argmax(scores))
        if a == 0:
            r = 1.0
        else:
            r = 10000.0 if rng.random() < 1e-3 else 0.0
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        picks_a2[t - 1] = 1 if a == 1 else 0
    return np.cumsum(picks_a2)

T = 20000
cum_noLog = simulate(use_log=False, T=T, seed=1)
cum_log = simulate(use_log=True, T=T, seed=1)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
t_axis = np.arange(1, T + 1)
ax.plot(t_axis, cum_log, color="#1b6ca8", lw=2.0, label=r"UCT (with $\ln N(h)$ term)")
ax.plot(t_axis, cum_noLog, color="#c0392b", lw=2.0, label=r"UCB$'$ (no log term)")
ax.set_xlabel("Interaction step $t$")
ax.set_ylabel(r"Cumulative number of times $a_2$ chosen")
ax.set_title("Example 12.2.5: exploring the rare high-reward action $a_2$")
ax.legend(loc="upper left", frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_ucb_vs_uct_bandit.pdf"))
plt.close(fig)
print("saved fig_ucb_vs_uct_bandit.pdf")

print("All figures generated.")
