"""
gen_figures.py -- Generate all figures for Chapter 3: Convex Sets
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer 2017)

Run with: python3 gen_figures.py

Produces (as vector PDFs, in this directory):
  fig_convex_nonconvex.pdf   -- convex vs. non-convex sets, side by side
  fig_convex_hull.pdf        -- convex hull of a finite point set (running
                                 numeric example: triangle vertices)
  fig_projection.pdf         -- projection onto a closed convex set,
                                 illustrating the obtuse-angle characterization
                                 of Theorem 3.16
  fig_separation.pdf         -- (strong) separation of two disjoint closed
                                 convex sets by a hyperplane (Theorem 3.50 /
                                 Definition 3.49)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: convex vs. non-convex sets
# ─────────────────────────────────────────────────────────────────────────
def fig_convex_nonconvex():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))

    def style(ax, title, good):
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        color = '#2c6aad' if good else '#c0392b'
        ax.set_title(title, fontsize=11, color=color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.6)

    # (a) disk -- convex
    ax = axes[0]
    style(ax, "Disk (convex)", True)
    disk = mpatches.Circle((0, 0), 1.2, facecolor='#dce9f5', edgecolor='#2c6aad', lw=2)
    ax.add_patch(disk)
    p, q = np.array([-0.75, -0.55]), np.array([0.85, 0.65])
    ax.plot(*zip(p, q), color='#c0392b', lw=2, marker='o', ms=5)

    # (b) triangle -- convex
    ax = axes[1]
    style(ax, "Triangle (convex)", True)
    tri = mpatches.Polygon([(-1.3, -1.0), (1.3, -1.0), (0.0, 1.3)],
                            facecolor='#dce9f5', edgecolor='#2c6aad', lw=2)
    ax.add_patch(tri)
    p, q = np.array([-0.9, -0.7]), np.array([0.6, 0.3])
    ax.plot(*zip(p, q), color='#c0392b', lw=2, marker='o', ms=5)

    # (c) crescent -- non-convex
    ax = axes[2]
    style(ax, "Crescent (not convex)", False)
    theta = np.linspace(0.15 * np.pi, 1.85 * np.pi, 200)
    outer = np.array([1.3 * np.cos(theta), 1.3 * np.sin(theta)]).T
    theta2 = np.linspace(1.85 * np.pi, 0.15 * np.pi, 200)
    inner = np.array([0.55 + 1.1 * np.cos(theta2), 1.1 * np.sin(theta2)]).T
    verts = np.vstack([outer, inner])
    crescent = mpatches.Polygon(verts, closed=True, facecolor='#fde8e8',
                                 edgecolor='#c0392b', lw=2)
    ax.add_patch(crescent)
    p, q = np.array([-1.1, 0.55]), np.array([-1.1, -0.55])
    ax.plot(*zip(p, q), color='#c0392b', lw=2, ls='--', marker='o', ms=5)
    ax.annotate("segment leaves the set!", xy=(-1.1, 0), xytext=(-1.55, -1.45),
                fontsize=8, color='#c0392b')

    # (d) two disjoint disks (union) -- non-convex
    ax = axes[3]
    style(ax, "Two disks (not convex)", False)
    d1 = mpatches.Circle((-0.75, 0), 0.55, facecolor='#fde8e8', edgecolor='#c0392b', lw=2)
    d2 = mpatches.Circle((0.75, 0), 0.55, facecolor='#fde8e8', edgecolor='#c0392b', lw=2)
    ax.add_patch(d1)
    ax.add_patch(d2)
    p, q = np.array([-0.75, 0]), np.array([0.75, 0])
    ax.plot(*zip(p, q), color='#c0392b', lw=2, ls='--', marker='o', ms=5)

    fig.suptitle(r"Convexity: for every $x,y \in C$, the segment $[x,y]$ stays inside $C$",
                 fontsize=11, y=1.04)
    savefig("fig_convex_nonconvex.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: convex hull of a finite point set -- running numeric example
# vertices of the running-example triangle T = conv{(0,0),(4,0),(0,3)}
# plus two extra sample points used to test hull membership.
# ─────────────────────────────────────────────────────────────────────────
def fig_convex_hull():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    pts = np.array([[0, 0], [4, 0], [0, 3]])  # the three generators
    hull_poly = mpatches.Polygon(pts, closed=True, facecolor='#dce9f5',
                                  edgecolor='#2c6aad', lw=2.2, zorder=1,
                                  label=r'$\mathrm{conv}\{(0,0),(4,0),(0,3)\}$')
    ax.add_patch(hull_poly)

    for (x, y) in pts:
        ax.plot(x, y, 'o', color='#2c6aad', ms=8, zorder=3)
    ax.annotate("(0,0)", (0, 0), textcoords="offset points", xytext=(-28, -12), fontsize=10)
    ax.annotate("(4,0)", (4, 0), textcoords="offset points", xytext=(4, -16), fontsize=10)
    ax.annotate("(0,3)", (0, 3), textcoords="offset points", xytext=(-8, 10), fontsize=10)

    # a point that IS a convex combination -> inside the hull
    inside = np.array([1.0, 1.0])
    ax.plot(*inside, marker='*', color='#1e8449', ms=16, zorder=4)
    ax.annotate(r"$(1,1)\in T$", inside, textcoords="offset points",
                xytext=(8, 6), fontsize=10, color='#1e8449')

    # a point that is NOT a convex combination -> outside the hull
    outside = np.array([3.0, 3.0])
    ax.plot(*outside, marker='X', color='#c0392b', ms=12, zorder=4)
    ax.annotate(r"$(3,3)\notin T$", outside, textcoords="offset points",
                xytext=(8, 4), fontsize=10, color='#c0392b')

    # a redundant / interior generator, to show it does not change the hull
    redundant = np.array([1.0, 0.5])
    ax.plot(*redundant, 's', color='#7f8c8d', ms=7, zorder=4)
    ax.annotate("redundant point\n(already in $T$)", redundant,
                textcoords="offset points", xytext=(10, -30), fontsize=8.5,
                color='#555555')

    ax.set_xlim(-1.2, 4.8)
    ax.set_ylim(-1.2, 4.0)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title("Convex hull of a finite point set (running example $T$)", fontsize=11)
    ax.grid(alpha=0.25)
    savefig("fig_convex_hull.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: projection onto a nonempty closed convex set (Theorem 3.16)
# obtuse/right-angle characterization: <y - p | x - p> <= 0 for all y in C
# ─────────────────────────────────────────────────────────────────────────
def fig_projection():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # closed convex set: an ellipse-ish blob
    theta = np.linspace(0, 2 * np.pi, 300)
    cx, cy = 0.0, 0.0
    rx, ry = 1.6, 1.1
    C = np.array([cx + rx * np.cos(theta), cy + ry * np.sin(theta)]).T
    blob = mpatches.Polygon(C, closed=True, facecolor='#dce9f5',
                             edgecolor='#2c6aad', lw=2)
    ax.add_patch(blob)
    ax.text(0, 0, r'$C$', fontsize=13, color='#2c6aad', ha='center', va='center')

    x = np.array([3.6, 2.2])  # point outside C
    # numerically find closest point on the ellipse boundary to x
    dists = np.linalg.norm(C - x, axis=1)
    p = C[np.argmin(dists)]

    ax.plot(*x, 'o', color='#c0392b', ms=8)
    ax.annotate(r'$x$', x, textcoords="offset points", xytext=(6, 6), fontsize=12)
    ax.plot(*p, 'o', color='black', ms=6)
    ax.annotate(r'$p = P_C x$', p, textcoords="offset points", xytext=(8, -4), fontsize=11)

    ax.annotate("", xy=p, xytext=x,
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.8))

    # a generic point y in C, and the vector y - p
    y = np.array([-1.0, -0.6])
    ax.plot(*y, 'o', color='#1e8449', ms=6)
    ax.annotate(r'$y$', y, textcoords="offset points", xytext=(-14, -12), fontsize=11)
    ax.annotate("", xy=y, xytext=p,
                arrowprops=dict(arrowstyle='->', color='#1e8449', lw=1.6))

    ax.set_xlim(-2.6, 4.6)
    ax.set_ylim(-2.6, 3.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r"$\langle y-p,\ x-p\rangle \leq 0$ for every $y \in C$"
                 "\n(the angle at $p$ is right or obtuse)", fontsize=10.5)
    savefig("fig_projection.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: separation of two disjoint closed convex sets (Theorem 3.50)
# ─────────────────────────────────────────────────────────────────────────
def fig_separation():
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    # C: a disk
    C = mpatches.Circle((-1.6, 0.2), 1.15, facecolor='#dce9f5', edgecolor='#2c6aad', lw=2)
    ax.add_patch(C)
    ax.text(-1.6, 0.2, r'$C$', fontsize=13, color='#2c6aad', ha='center', va='center')

    # D: a rotated square-ish blob
    Dpts = np.array([[1.0, -1.1], [3.1, -0.6], [2.7, 1.4], [1.1, 1.2]])
    D = mpatches.Polygon(Dpts, closed=True, facecolor='#fde8e8', edgecolor='#c0392b', lw=2)
    ax.add_patch(D)
    ax.text(2.0, 0.1, r'$D$', fontsize=13, color='#c0392b', ha='center', va='center')

    # separating line: vertical-ish, x-coordinate ~ 0.15 (some line u.x = eta)
    xs = np.linspace(-0.35, 0.55, 2)
    # line: direction perpendicular to u=(1,0.15) roughly, through (0.1,*)
    u = np.array([1.0, 0.12])
    eta = 0.05
    # points on the line u1*x+u2*y = eta -> y = (eta - u1 x)/u2 ... but u2 small; instead
    # parametrize the line directly: passes near x=0.1, with slight tilt.
    line_x = np.array([-2.6, 0.0]) * 0 + 0.1  # placeholder, recompute below
    t = np.linspace(-3.0, 3.0, 2)
    px = 0.1 - 0.12 * t
    py = t
    ax.plot(px, py, color='black', lw=1.8)
    ax.annotate(r'$H = \{z : \langle z,\ u\rangle = \eta\}$', (0.15, 2.15), fontsize=10)

    # normal vector u
    ax.annotate("", xy=(0.1 + 0.9, 0.0 + 0.9 * 0.12 * 0 - 0.0), xytext=(0.1, 0.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.6))
    ax.annotate(r'$u$', (1.05, 0.05), fontsize=11)

    ax.set_xlim(-3.2, 3.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r"$\sup\langle C,\ u\rangle \leq \eta \leq \inf\langle D,\ u\rangle$",
                 fontsize=11)
    savefig("fig_separation.pdf")


if __name__ == "__main__":
    fig_convex_nonconvex()
    fig_convex_hull()
    fig_projection()
    fig_separation()
    print("All figures generated.")
