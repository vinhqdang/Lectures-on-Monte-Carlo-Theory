#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 2: Hilbert Spaces
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer, 2017).

Figures produced (saved as vector PDF in this directory):

  fig_parallelogram.pdf
      The parallelogram law ||x+y||^2 + ||x-y||^2 = 2||x||^2 + 2||y||^2
      illustrated geometrically in R^2 with the running-example vectors
      x = (3,4), y = (1,-2): the parallelogram spanned by x and y, its two
      diagonals x+y and x-y, and the numeric identity check.

  fig_weak_vs_strong.pdf
      Two-panel figure contrasting weak and strong convergence using the
      standard orthonormal sequence (e_n) in ell^2(N) (Example 2.32 in the
      book). Left: a "truncated coordinate" stem-plot view of e_1,...,e_8,
      showing the mass of each vector marching off to a new, previously-zero
      coordinate (so no coordinate/functional sees a moving target forever).
      Right: ||e_n|| is identically 1 (no strong convergence to 0) while
      |<e_n | u>| -> 0 for a fixed u in ell^2(N) (weak convergence to 0),
      and correspondingly ||e_n - e_m|| = sqrt(2) for all n != m (no Cauchy
      subsequence, hence no strongly convergent subsequence).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(0)

# ---------------------------------------------------------------------------
# Running example vectors (used throughout the slide deck)
# ---------------------------------------------------------------------------
x = np.array([3.0, 4.0])
y = np.array([1.0, -2.0])

# ===========================================================================
# Figure (a): parallelogram law in R^2
# ===========================================================================
s = x + y   # one diagonal
d = x - y   # other diagonal

norm_x2 = float(np.dot(x, x))
norm_y2 = float(np.dot(y, y))
norm_s2 = float(np.dot(s, s))
norm_d2 = float(np.dot(d, d))

fig, ax = plt.subplots(figsize=(7.4, 6.4))

O = np.array([0.0, 0.0])

# The parallelogram has vertices O, x, x+y, y (in order O -> x -> x+y -> y -> O)
poly = np.array([O, x, s, y, O])
ax.fill(poly[:, 0], poly[:, 1], color='steelblue', alpha=0.10, zorder=1)
ax.plot(poly[:, 0], poly[:, 1], color='steelblue', linewidth=1.8, zorder=2)

# Sides: x (O->x), y (O->y), and the two translated copies
ax.annotate('', xy=x, xytext=O,
            arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.4))
ax.annotate('', xy=y, xytext=O,
            arrowprops=dict(arrowstyle='-|>', color='darkorange', lw=2.4))
ax.annotate('', xy=s, xytext=x,
            arrowprops=dict(arrowstyle='-', color='darkorange', lw=1.6, linestyle='--'))
ax.annotate('', xy=s, xytext=y,
            arrowprops=dict(arrowstyle='-', color='crimson', lw=1.6, linestyle='--'))

# Diagonals: x+y (O -> s) and x-y (y -> x)
ax.annotate('', xy=s, xytext=O,
            arrowprops=dict(arrowstyle='-|>', color='seagreen', lw=2.4))
ax.annotate('', xy=x, xytext=y,
            arrowprops=dict(arrowstyle='-|>', color='purple', lw=2.4))

ax.plot(*O, marker='o', color='black', markersize=5, zorder=5)

labels = [
    (x, "$x=(3,4)$", 'crimson', (0.10, 0.15)),
    (y, "$y=(1,-2)$", 'darkorange', (0.10, -0.35)),
    (s, "$x+y=(4,2)$", 'seagreen', (0.10, 0.15)),
]
for pt, lab, c, off in labels:
    ax.annotate(lab, xy=pt, xytext=(pt[0] + off[0], pt[1] + off[1]),
                color=c, fontsize=11, fontweight='bold')

mid_diag = (x + y) / 2.0
ax.annotate("$x-y=(2,6)$", xy=mid_diag, xytext=(mid_diag[0] - 2.7, mid_diag[1] - 0.3),
            color='purple', fontsize=11, fontweight='bold')

ax.set_xlim(-1.5, 5.5)
ax.set_ylim(-3.5, 6.5)
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
ax.axvline(0, color='gray', linewidth=0.5, zorder=0)
ax.set_xlabel("$\\xi_1$")
ax.set_ylabel("$\\xi_2$")
ax.set_title("Parallelogram law: $\\|x+y\\|^2+\\|x-y\\|^2 = 2\\|x\\|^2+2\\|y\\|^2$")

legend_elems = [
    Line2D([0], [0], color='crimson', lw=2.4, label='side $x$'),
    Line2D([0], [0], color='darkorange', lw=2.4, label='side $y$'),
    Line2D([0], [0], color='seagreen', lw=2.4, label='diagonal $x+y$'),
    Line2D([0], [0], color='purple', lw=2.4, label='diagonal $x-y$'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=9, framealpha=0.9)

textstr = (
    f"$\\|x\\|^2={norm_x2:g}$   $\\|y\\|^2={norm_y2:g}$\n"
    f"$\\|x+y\\|^2={norm_s2:g}$   $\\|x-y\\|^2={norm_d2:g}$\n"
    f"LHS $=\\|x+y\\|^2+\\|x-y\\|^2={norm_s2+norm_d2:g}$\n"
    f"RHS $=2\\|x\\|^2+2\\|y\\|^2={2*norm_x2+2*norm_y2:g}$"
)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

fig.tight_layout()
fig.savefig('fig_parallelogram.pdf')
plt.close(fig)

assert abs((norm_s2 + norm_d2) - (2 * norm_x2 + 2 * norm_y2)) < 1e-9

# ===========================================================================
# Figure (b): weak vs. strong convergence -- orthonormal sequence in ell^2(N)
# ===========================================================================
N_show = 8     # number of basis vectors to draw as coordinate stems
N_big = 200    # ambient truncated dimension used to fix u in ell^2(N)

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0))

# --- Left panel: e_1,...,e_8 as spikes in coordinate space -----------------
offset = 1.15  # vertical spacing between stacked stem-rows
for n in range(1, N_show + 1):
    row = N_show - n  # draw e_1 at top
    coords = np.zeros(N_show + 2)
    coords[n] = 1.0
    baseline = row * offset
    markerline, stemlines, baseline_ = axL.stem(
        range(N_show + 2), coords + baseline, bottom=baseline,
        basefmt=' ')
    plt.setp(markerline, color='steelblue', markersize=4)
    plt.setp(stemlines, color='steelblue', linewidth=1.2)
    axL.axhline(baseline, color='gray', linewidth=0.4, zorder=0)
    axL.text(-1.3, baseline, f"$e_{{{n}}}$", fontsize=10, va='center', ha='right')

axL.set_xlim(-2.0, N_show + 1.5)
axL.set_ylim(-0.3, N_show * offset + 0.3)
axL.set_yticks([])
axL.set_xlabel("coordinate index $i$")
axL.set_title("Orthonormal sequence $(e_n)$: unit spike\nmarches off to new coordinates")

# --- Right panel: ||e_n|| stays 1, but <e_n|u> -> 0 (weak convergence) -----
n_vals = np.arange(1, 31)
u = 1.0 / n_vals  # a fixed vector u=(1/1,1/2,1/3,...) truncated; in ell^2(N)
u_full = 1.0 / np.arange(1, N_big + 1)
norm_u = np.linalg.norm(u_full)

norm_en = np.ones_like(n_vals, dtype=float)          # ||e_n|| = 1 for every n
inner_en_u = np.array([u_full[n - 1] if n - 1 < N_big else 0.0 for n in n_vals])  # <e_n|u> = u_n

axR.plot(n_vals, norm_en, 'o-', color='crimson', label=r'$\|e_n\|=1$ (no strong $\to 0$)')
axR.plot(n_vals, np.abs(inner_en_u), 's-', color='seagreen',
         label=r'$|\langle e_n\mid u\rangle|\to 0$ (weak $\to 0$)')
axR.axhline(0, color='gray', linewidth=0.5)
axR.set_xlabel("$n$")
axR.set_ylabel("value")
axR.set_title(r"$e_n \rightharpoonup 0$ weakly but $e_n \not\to 0$ strongly")
axR.legend(loc='upper right', fontsize=9, framealpha=0.9)
axR.set_ylim(-0.05, 1.15)

fig2.suptitle("Weak vs. strong convergence of the orthonormal sequence $(e_n)_{n\\in\\mathbb{N}}$ "
              "(Example 2.32)", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig('fig_weak_vs_strong.pdf')
plt.close(fig2)

# Sanity check printed to console: pairwise distances are all sqrt(2), so
# (e_n) has no Cauchy (hence no strongly convergent) subsequence.
pair_dist = np.sqrt(2.0)
print("Generated fig_parallelogram.pdf and fig_weak_vs_strong.pdf")
print(f"Parallelogram check: LHS={norm_s2+norm_d2:g}, RHS={2*norm_x2+2*norm_y2:g}")
print(f"||e_n - e_m|| = sqrt(2) = {pair_dist:.6f} for all n != m (no Cauchy subsequence)")
print(f"<e_n|u> = u_n -> 0 as n -> infinity, since u in ell^2(N) (||u||={norm_u:.4f})")
