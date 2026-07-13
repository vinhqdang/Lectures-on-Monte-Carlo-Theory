#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 2: Notation and Mathematical Foundations
(The Krasnosel'skii-Mann Iterative Method, Dong-Cho-He-Pardalos-Rassias, 2022).

Figures produced (saved as vector PDF in this directory):
  fig_projection.pdf         -- P_C for C = closed unit disk in R^2,
                                 several external/internal points and their
                                 nearest-point projections onto C.
  fig_firmly_nonexpansive.pdf -- numeric check, over many random pairs
                                 (x, y) in R^2, that
                                 ||P_C(x)-P_C(y)||^2 <= <P_C(x)-P_C(y), x-y>
                                 (the firmly-nonexpansive inequality, Lemma 2.4(1)).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)


def project_to_unit_disk(x):
    """Metric projection P_C(x) onto C = closed unit disk in R^2."""
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if norm <= 1.0:
        return x.copy()
    return x / norm


# ---------------------------------------------------------------------------
# Figure (a): the projection operator P_C onto the closed unit disk
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.8, zorder=2)
ax.fill(np.cos(theta), np.sin(theta), color='steelblue', alpha=0.12, zorder=1)

# A handful of points: some outside C, one inside C (projection = itself),
# including the running example point (3,4)/... actually (3,4) scaled is
# used elsewhere; here we keep a representative spread of points.
points = [
    (3.0, 4.0),
    (0.0, 2.0),
    (-1.5, -1.2),
    (2.0, -0.5),
    (0.3, 0.4),   # already inside C
    (-2.2, 1.0),
]

colors = ['crimson', 'darkorange', 'seagreen', 'purple', 'saddlebrown', 'teal']

for (pt, c) in zip(points, colors):
    x = np.array(pt)
    px = project_to_unit_disk(x)
    ax.plot(*x, marker='o', markersize=7, color=c, zorder=5)
    ax.plot(*px, marker='s', markersize=7, color=c, zorder=5,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.annotate('', xy=px, xytext=x,
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.4,
                                shrinkA=4, shrinkB=4))
    label = f"({pt[0]:g},{pt[1]:g})"
    ax.annotate(label, xy=x, xytext=(x[0] + 0.10, x[1] + 0.10), color=c,
                fontsize=9)

ax.set_xlim(-3.2, 4.6)
ax.set_ylim(-3.0, 5.0)
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
ax.axvline(0, color='gray', linewidth=0.5, zorder=0)
ax.set_title(r"Metric projection $P_C$ onto $C = $ closed unit disk in $\mathbb{R}^2$")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

# legend proxies
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, label='original point $x$'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
           markeredgecolor='gray', markeredgewidth=1.8, markersize=8,
           label='projection $P_C(x)$'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig('fig_projection.pdf')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (b): numeric verification of the firmly-nonexpansive inequality
#   ||P_C(x) - P_C(y)||^2 <= <P_C(x) - P_C(y), x - y>     for all x,y in R^2
# ---------------------------------------------------------------------------
n_samples = 4000
# sample points from a fairly wide box so that both "inside C", "outside C",
# and "mixed" configurations are well represented.
X = np.random.uniform(-4, 4, size=(n_samples, 2))
Y = np.random.uniform(-4, 4, size=(n_samples, 2))

lhs = np.empty(n_samples)
rhs = np.empty(n_samples)

for i in range(n_samples):
    px = project_to_unit_disk(X[i])
    py = project_to_unit_disk(Y[i])
    diff_p = px - py
    diff_xy = X[i] - Y[i]
    lhs[i] = np.dot(diff_p, diff_p)          # ||P_C(x)-P_C(y)||^2
    rhs[i] = np.dot(diff_p, diff_xy)         # <P_C(x)-P_C(y), x-y>

violations = np.sum(lhs > rhs + 1e-9)

fig2, ax2 = plt.subplots(figsize=(5.8, 5.4))
ax2.scatter(rhs, lhs, s=6, alpha=0.35, color='steelblue',
            label=f'{n_samples} random pairs $(x,y)$')
lims = [min(rhs.min(), lhs.min()) - 0.5, max(rhs.max(), lhs.max()) + 0.5]
ax2.plot(lims, lims, color='crimson', linewidth=1.5, linestyle='--',
         label=r'$\mathrm{LHS}=\mathrm{RHS}$ (equality line)')
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel(r"RHS $= \langle P_C(x)-P_C(y),\, x-y\rangle$")
ax2.set_ylabel(r"LHS $=\|P_C(x)-P_C(y)\|^2$")
ax2.set_title("Firmly nonexpansive inequality for $P_C$ (Lemma 2.4(1))\n"
              f"violations found: {violations} / {n_samples}")
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)

fig2.tight_layout()
fig2.savefig('fig_firmly_nonexpansive.pdf')
plt.close(fig2)

print("Generated fig_projection.pdf and fig_firmly_nonexpansive.pdf")
print(f"Violations of LHS <= RHS out of {n_samples} random pairs: {violations}")
