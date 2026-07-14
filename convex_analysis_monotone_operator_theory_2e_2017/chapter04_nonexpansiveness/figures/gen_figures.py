#!/usr/bin/env python3
"""
gen_figures.py -- Generate all figures for Chapter 4: Convexity and Notions
of Nonexpansiveness
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer 2017)

Running example used throughout the chapter (and reused from this repo's
Krasnosel'skii-Mann book): C = closed unit disk in R^2, T = P_C the metric
projector onto C, x0 = (3,4), unique fixed point of interest (0.6, 0.8).

Run with: python3 gen_figures.py

Produces (as vector PDFs, in this directory):
  fig_projection.pdf         -- P_C for C = closed unit disk, several points
                                 and arrows to their projections.
  fig_firmly_nonexpansive.pdf-- numeric scatter, over many random pairs
                                 (x,y) in R^2, verifying
                                 ||P_C(x)-P_C(y)||^2 <= <P_C(x)-P_C(y), x-y>
                                 (the firmly-nonexpansive inequality).
  fig_averaged_family.pdf    -- the averaged-operator family
                                 T_alpha = (1-alpha) Id + alpha R,
                                 R = 2P_C - Id, applied to x0=(3,4),
                                 illustrating how alpha "damps" the
                                 nonexpansive reflection R down to P_C
                                 (alpha=1/2) and beyond.
  fig_soft_threshold.pdf     -- the 1-D soft-thresholding operator
                                 (Example 4.17(i)), shown to be firmly
                                 nonexpansive.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


def P_C(z):
    """Metric projection onto C = closed unit disk in R^2."""
    z = np.asarray(z, dtype=float)
    n = np.linalg.norm(z)
    return z.copy() if n <= 1.0 else z / n


# ---------------------------------------------------------------------------
# Figure (a): the projector P_C onto the closed unit disk, several points
# ---------------------------------------------------------------------------
def fig_projection():
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.8, zorder=2)
    ax.fill(np.cos(theta), np.sin(theta), color='steelblue', alpha=0.12, zorder=1)

    points = [
        (3.0, 4.0),    # the running-example point x0
        (0.0, 2.0),
        (-2.2, -1.0),
        (1.8, -0.6),
        (0.4, 0.3),    # already inside C: P_C fixes it
    ]
    colors = ['crimson', 'darkorange', 'seagreen', 'purple', 'gray']

    for (x, y), c in zip(points, colors):
        px, py = P_C((x, y))
        ax.plot(x, y, 'o', color=c, markersize=7, zorder=4)
        ax.plot(px, py, 's', color=c, markersize=7, zorder=4, markerfacecolor='white',
                markeredgewidth=1.8)
        if (x, y) != (px, py):
            ax.annotate('', xy=(px, py), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.6), zorder=3)
        else:
            ax.plot(x, y, 'o', color=c, markersize=11, markerfacecolor='none',
                    markeredgewidth=1.6, zorder=3)

    ax.annotate(r'$x_0=(3,4)$', xy=(3.0, 4.0), xytext=(3.15, 4.15), fontsize=11)
    ax.annotate(r'$P_C(x_0)=(0.6,0.8)$', xy=(0.6, 0.8), xytext=(0.75, 1.55), fontsize=10)

    ax.set_xlim(-3.2, 4.6)
    ax.set_ylim(-2.4, 4.8)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(r'The projector $P_C$ onto the closed unit disk $C$')
    ax.grid(alpha=0.25)
    savefig('fig_projection.pdf')


# ---------------------------------------------------------------------------
# Figure (b): numeric verification of the firmly-nonexpansive inequality
# ---------------------------------------------------------------------------
def fig_firmly_nonexpansive():
    n_samples = 400
    pts = np.random.uniform(-5, 5, size=(n_samples, 2, 2))  # (sample, {x,y}, coord)

    lhs = np.empty(n_samples)
    rhs = np.empty(n_samples)
    for k in range(n_samples):
        x, y = pts[k, 0], pts[k, 1]
        px, py = P_C(x), P_C(y)
        lhs[k] = np.dot(px - py, px - py)
        rhs[k] = np.dot(px - py, x - y)

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.scatter(rhs, lhs, s=14, alpha=0.55, color='steelblue',
               label=r'$400$ random pairs $(x,y)$')
    lims = [min(rhs.min(), lhs.min()) - 0.5, max(rhs.max(), lhs.max()) + 0.5]
    ax.plot(lims, lims, 'k--', lw=1.4, label=r'$\mathrm{LHS}=\mathrm{RHS}$')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\langle P_C x - P_C y \mid x-y\rangle$  (RHS)')
    ax.set_ylabel(r'$\|P_C x - P_C y\|^2$  (LHS)')
    ax.set_title('Firmly nonexpansive inequality for $P_C$:\n'
                 r'every point lies on or below the diagonal')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    assert np.all(lhs <= rhs + 1e-9), "firmly nonexpansive inequality violated!"
    savefig('fig_firmly_nonexpansive.pdf')


# ---------------------------------------------------------------------------
# Figure (c): the averaged family T_alpha = (1-alpha) Id + alpha R, R = 2P_C-Id
# ---------------------------------------------------------------------------
def fig_averaged_family():
    x0 = np.array([3.0, 4.0])

    def R(z):
        return 2 * P_C(z) - z

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [r'$\alpha=0$ (Id, no move)', r'$\alpha=0.25$', r'$\alpha=0.5\ (=P_C)$',
              r'$\alpha=0.75$', r'$\alpha=1$ (reflection $R$)']
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(alphas)))

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.6, zorder=2)
    ax.fill(np.cos(theta), np.sin(theta), color='steelblue', alpha=0.10, zorder=1)

    ax.plot(*x0, 'o', color='crimson', markersize=9, zorder=5, label=r'$x_0=(3,4)$')
    r0 = R(x0)
    for alpha, lab, c in zip(alphas, labels, colors):
        Ta = (1 - alpha) * x0 + alpha * r0
        ax.plot(*Ta, 'D', color=c, markersize=8, zorder=4)
        ax.annotate(lab, xy=Ta, xytext=(Ta[0] + 0.15, Ta[1] + 0.12), fontsize=8.5)
    # segment from x0 through the interpolation path to R(x0)
    t = np.linspace(0, 1, 50)
    path = np.outer(1 - t, x0) + np.outer(t, r0)
    ax.plot(path[:, 0], path[:, 1], color='gray', lw=1.2, ls=':', zorder=3)

    ax.set_xlim(-1.8, 4.6)
    ax.set_ylim(-3.4, 4.8)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(r'Averaged family $T_\alpha=(1-\alpha)\mathrm{Id}+\alpha R$, '
                r'$R=2P_C-\mathrm{Id}$')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.25)
    savefig('fig_averaged_family.pdf')


# ---------------------------------------------------------------------------
# Figure (d): the 1-D soft-thresholding operator (Example 4.17(i))
# ---------------------------------------------------------------------------
def fig_soft_threshold():
    rho = 1.5
    xs = np.linspace(-5, 5, 400)

    def soft(x, rho):
        return np.sign(x) * np.maximum(np.abs(x) - rho, 0.0)

    ys = soft(xs, rho)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(xs, xs, color='gray', lw=1.0, ls='--', label=r'Identity (reference)')
    ax.plot(xs, ys, color='darkorange', lw=2.4,
            label=rf'Soft threshold $T_1$, $\rho={rho}$')
    ax.axvspan(-rho, rho, color='steelblue', alpha=0.12)
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$T_1 x$')
    ax.set_title('Soft thresholder: firmly nonexpansive (Example 4.17(i))')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    savefig('fig_soft_threshold.pdf')


if __name__ == '__main__':
    fig_projection()
    fig_firmly_nonexpansive()
    fig_averaged_family()
    fig_soft_threshold()
    print("All figures generated.")
