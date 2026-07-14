#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 7: Support Functions and Polar Sets
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer, 2017).

Figures produced (saved as vector PDF in this directory):
  fig_support_disk.pdf     -- Support function sigma_C(u) = ||u|| of the
                               closed unit disk C = B(0;1) in R^2, shown for
                               a unit-norm direction and a longer direction,
                               with the corresponding supporting lines.
  fig_support_polygon.pdf  -- Support function of a general convex polygon C:
                               a direction u, the supporting hyperplane H_u,
                               and the distance sigma_C(u) from the origin
                               to H_u (mirrors Fig. 7.1 in the book, for a
                               pentagon instead of a generic blob).
  fig_polar_square.pdf     -- The square C = [-1,1]^2 (the closed unit ball
                               of the sup-norm) together with its polar set
                               C^o, the closed unit ball of the l^1 norm
                               (a diamond), illustrating C^o = {u : sigma_C(u)
                               <= 1}.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(0)

# ---------------------------------------------------------------------------
# Figure 1: support function of the closed unit disk, sigma_C(u) = ||u||
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 6.2))

theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.8, zorder=2)
ax.fill(np.cos(theta), np.sin(theta), color='steelblue', alpha=0.15, zorder=1)
ax.plot(0, 0, marker='o', color='black', markersize=4, zorder=6)

def draw_support(ax, u, color, label):
    """Draw direction u, the support point x* = u/||u|| (for the unit disk),
    the supporting line at x*, and the segment of length sigma_C(u)=||u||."""
    u = np.asarray(u, dtype=float)
    nu = np.linalg.norm(u)
    xstar = u / nu  # sup is attained at x* = u/||u|| since ||x*||=1
    # arrow for direction u (drawn from the origin, actual length ||u||)
    ax.annotate('', xy=u, xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.0))
    ax.plot(*xstar, marker='o', markersize=7, color=color, zorder=5)
    # supporting line: perpendicular to u, passing through x*
    perp = np.array([-u[1], u[0]]) / nu
    p1 = xstar + 2.6 * perp
    p2 = xstar - 2.6 * perp
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle='-',
            linewidth=1.6, zorder=3)
    ax.annotate(label, xy=u, xytext=(u[0] + 0.12, u[1] + 0.12), color=color,
                fontsize=11)

draw_support(ax, (np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))),
             'crimson', r"$u_1$, $\|u_1\|=1$")
draw_support(ax, (1.8 * np.cos(np.deg2rad(150)), 1.8 * np.sin(np.deg2rad(150))),
             'darkorange', r"$u_2$, $\|u_2\|=1.8$")

ax.set_xlim(-3.0, 2.2)
ax.set_ylim(-1.6, 2.6)
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
ax.axvline(0, color='gray', linewidth=0.5, zorder=0)
ax.set_title(r"Support function of the unit disk: $\sigma_C(u) = \|u\|$")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

legend_elems = [
    Line2D([0], [0], color='crimson', lw=2,
           label=r'$u_1$: supporting line at $x^*=u_1$, $\sigma_C(u_1)=1$'),
    Line2D([0], [0], color='darkorange', lw=2,
           label=r'$u_2$: supporting line at $x^*=u_2/\|u_2\|$, $\sigma_C(u_2)=1.8$'),
]
ax.legend(handles=legend_elems, loc='lower left', fontsize=7.5, framealpha=0.9)

fig.tight_layout()
fig.savefig('fig_support_disk.pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: support function of a convex pentagon (mirrors book Fig. 7.1)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6.4, 6.0))

# a convex pentagon C, not centered at the origin
pentagon = np.array([
    [2.4, 1.2],
    [3.4, 2.0],
    [3.0, 3.2],
    [1.8, 3.4],
    [1.2, 2.1],
])
poly_closed = np.vstack([pentagon, pentagon[0]])
ax2.fill(poly_closed[:, 0], poly_closed[:, 1], color='steelblue', alpha=0.20,
          zorder=1)
ax2.plot(poly_closed[:, 0], poly_closed[:, 1], color='black', linewidth=1.8,
          zorder=2)
ax2.annotate("$C$", xy=(2.3, 2.3), fontsize=13)
ax2.plot(0, 0, marker='o', color='black', markersize=5, zorder=6)
ax2.annotate("$0$", xy=(0.08, -0.28), fontsize=11)

# direction u (unit vector) and sigma_C(u) = max_{x in C} <x|u>
angle = np.deg2rad(28)
u = np.array([np.cos(angle), np.sin(angle)])
vals = pentagon @ u
sigma = vals.max()
xstar = pentagon[np.argmax(vals)]

# arrow for u, scaled for visibility, drawn from the origin
ax2.annotate('', xy=1.3 * u, xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.0))
ax2.annotate(r"$u$  ($\|u\|=1$)", xy=1.3 * u,
             xytext=(1.3 * u[0] + 0.05, 1.3 * u[1] - 0.35), color='crimson',
             fontsize=11)

# dashed segment from 0 along u of length sigma_C(u), marking sigma_C(u)
foot = sigma * u
ax2.plot([0, foot[0]], [0, foot[1]], color='crimson', linestyle='--',
          linewidth=1.4, zorder=3)
ax2.plot(*foot, marker='o', markersize=5, color='crimson', zorder=6)
ax2.annotate(r"$\sigma_C(u)$", xy=foot, xytext=(foot[0] - 1.3, foot[1] + 0.15),
             color='crimson', fontsize=11)

# supporting hyperplane H_u = {x : <x|u> = sigma_C(u)}, through x*
perp = np.array([-u[1], u[0]])
p1 = xstar + 2.6 * perp
p2 = xstar - 2.6 * perp
ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color='darkgreen', linewidth=1.6,
          zorder=3, label=r'$H_u=\{x:\langle x\mid u\rangle=\sigma_C(u)\}$')
ax2.plot(*xstar, marker='s', markersize=8, color='darkgreen', zorder=6,
          markerfacecolor='white', markeredgewidth=1.8)
ax2.annotate("$x^*$ (support point)", xy=xstar,
             xytext=(xstar[0] + 0.1, xstar[1] + 0.25), color='darkgreen',
             fontsize=10)

ax2.set_xlim(-1.0, 4.4)
ax2.set_ylim(-1.0, 4.4)
ax2.set_aspect('equal', adjustable='box')
ax2.axhline(0, color='gray', linewidth=0.5, zorder=0)
ax2.axvline(0, color='gray', linewidth=0.5, zorder=0)
ax2.set_title(r"$H_u$ is the smallest closed half-space with outer normal"
              " $u$ containing $C$")
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")

fig2.tight_layout()
fig2.savefig('fig_support_polygon.pdf')
plt.close(fig2)

# ---------------------------------------------------------------------------
# Figure 3: the square C=[-1,1]^2 and its polar set C^o = {u : ||u||_1 <= 1}
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(6.2, 6.2))

square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])

ax3.fill(square[:, 0], square[:, 1], color='steelblue', alpha=0.18, zorder=1)
ax3.plot(square[:, 0], square[:, 1], color='steelblue', linewidth=2.0,
          zorder=2, label=r"$C=[-1,1]^2$  (unit ball of $\|\cdot\|_\infty$)")

ax3.fill(diamond[:, 0], diamond[:, 1], color='darkorange', alpha=0.18,
          zorder=1)
ax3.plot(diamond[:, 0], diamond[:, 1], color='darkorange', linewidth=2.0,
          zorder=2,
          label=r"$C^{\odot}=\{u:\|u\|_1\leq 1\}$  (unit ball of $\|\cdot\|_1$)")

# unit circle for reference: the self-polar case
theta = np.linspace(0, 2 * np.pi, 300)
ax3.plot(np.cos(theta), np.sin(theta), color='gray', linewidth=1.2,
          linestyle=':', zorder=1,
          label=r"unit disk $B(0;1)$ (self-polar: $B(0;1)^{\odot}=B(0;1)$)")

# mark the point u=(1,1): sigma_C(1,1) = 2 > 1, so (1,1) not in C^o
pt = np.array([1.0, 1.0])
ax3.plot(*pt, marker='o', color='crimson', markersize=7, zorder=6)
ax3.annotate(r"$(1,1)$: $\sigma_C(1,1)=2>1 \Rightarrow (1,1)\notin C^{\odot}$",
             xy=pt, xytext=(pt[0] - 1.85, pt[1] + 0.25), color='crimson',
             fontsize=8)

# mark the point u=(0.5,0.5): on boundary of C^o since ||u||_1 = 1
pt2 = np.array([0.5, 0.5])
ax3.plot(*pt2, marker='o', color='seagreen', markersize=7, zorder=6)
ax3.annotate(r"$(0.5,0.5)$: $\|u\|_1=1 \Rightarrow$ bdry $C^{\odot}$",
             xy=pt2, xytext=(pt2[0] + 0.15, pt2[1] - 0.55), color='seagreen',
             fontsize=8)

ax3.set_xlim(-2.3, 2.3)
ax3.set_ylim(-2.3, 2.3)
ax3.set_aspect('equal', adjustable='box')
ax3.axhline(0, color='gray', linewidth=0.5, zorder=0)
ax3.axvline(0, color='gray', linewidth=0.5, zorder=0)
ax3.set_title(r"The square $C=[-1,1]^2$ and its polar set $C^{\odot}$")
ax3.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
ax3.set_xlabel("$u_1$")
ax3.set_ylabel("$u_2$")

fig3.tight_layout()
fig3.savefig('fig_polar_square.pdf')
plt.close(fig3)

print("Generated fig_support_disk.pdf, fig_support_polygon.pdf, "
      "fig_polar_square.pdf")
