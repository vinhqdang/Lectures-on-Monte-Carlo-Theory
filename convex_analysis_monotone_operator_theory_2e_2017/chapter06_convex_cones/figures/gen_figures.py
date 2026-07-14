"""
gen_figures.py -- Chapter 6: Convex Cones and Generalized Interiors
Bauschke & Combettes, Convex Analysis and Monotone Operator Theory in
Hilbert Spaces, 2nd ed. (2017), CMS Books in Mathematics, Springer.

Generates all figures for the Beamer slides using matplotlib (Agg backend).
Every figure is saved as a vector PDF into figures/.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

FIGDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIGDIR, exist_ok=True)


def savefig(name, fig=None, **kw):
    path = os.path.join(FIGDIR, name)
    (fig or plt).savefig(path, bbox_inches='tight', **kw)
    plt.close('all')
    print(f"  saved {path}")


def style_axes(ax, lim=3.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.7, zorder=1)
    ax.axvline(0, color='gray', lw=0.7, zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------------------------------------------------
# Figure 1: the nonnegative quadrant K = R^2_+ and its polar cone K^- = R^2_-
# -----------------------------------------------------------------------
def fig_orthant_polar():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))

    ax = axes[0]
    style_axes(ax)
    ax.fill_between([0, 3.2], 0, 3.2, color='royalblue', alpha=0.35)
    ax.annotate('', xy=(3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2))
    ax.annotate('', xy=(0, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2))
    ax.text(1.6, 1.6, r'$K=\mathbb{R}^2_+$', fontsize=15, color='navy',
            ha='center')
    ax.set_title('The cone $K$', fontsize=13)

    ax = axes[1]
    style_axes(ax)
    ax.fill_between([-3.2, 0], -3.2, 0, color='darkorange', alpha=0.35)
    ax.annotate('', xy=(-3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='darkorange', lw=2))
    ax.annotate('', xy=(0, -3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='darkorange', lw=2))
    ax.text(-1.6, -1.6, r'$K^{\ominus}=\mathbb{R}^2_-$', fontsize=15,
            color='saddlebrown', ha='center')
    ax.set_title(r'The polar cone $K^{\ominus}$', fontsize=13)

    fig.suptitle(r'$K=\mathbb{R}^2_+$ is self-dual: $K^{\ominus}=-K$,'
                  r' $K^{\oplus}=K$', fontsize=12, y=1.03)
    savefig('fig_orthant_polar.pdf', fig=fig)


# -----------------------------------------------------------------------
# Figure 2: the ice-cream cone K = {(x,y): y >= |x|}, its polar K^- = -K,
# and the dual K^+ = K (self-dual cone), drawn together.
# -----------------------------------------------------------------------
def fig_icecream_dual():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    style_axes(ax, lim=3.0)

    t = np.linspace(0, 3.0, 50)
    # K = {y >= |x|}: boundary rays (1,1) and (-1,1)
    xK = np.concatenate([-t[::-1], t])
    yK = np.abs(xK)
    ax.fill_between(xK, yK, 3.05, color='royalblue', alpha=0.35,
                     label=r'$K=\{(x,y):y\geq|x|\}$')
    ax.plot([-3, 0, 3], [3, 0, 3], color='navy', lw=2)

    # K^- = -K = {y <= -|x|}
    ax.fill_between(xK, -3.05, -yK, color='darkorange', alpha=0.35,
                     label=r'$K^{\ominus}=-K=\{(x,y):y\leq -|x|\}$')
    ax.plot([-3, 0, 3], [-3, 0, -3], color='saddlebrown', lw=2, ls='--')

    ax.annotate('', xy=(1, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='navy', lw=1.6))
    ax.annotate('', xy=(-1, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='navy', lw=1.6))
    ax.text(1.55, 2.15, r'$K$', fontsize=15, color='navy')
    ax.text(-1.9, -2.35, r'$K^{\ominus}=-K$', fontsize=13,
            color='saddlebrown')

    ax.set_title(r'Ice-cream cone $K$: $K^{\ominus}=-K$, so'
                 r' $K^{\oplus}=-K^{\ominus}=K$ (self-dual)', fontsize=11.5)
    savefig('fig_icecream_dual.pdf', fig=fig)


# -----------------------------------------------------------------------
# Figure 3: tangent cone / normal cone to the unit ball at a boundary point
# (Example 6.39) and at an interior point.
# -----------------------------------------------------------------------
def fig_ball_tangent_normal():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    style_axes(ax, lim=2.3)

    circle = plt.Circle((0, 0), 1, color='royalblue', alpha=0.30, zorder=2)
    ax.add_patch(circle)
    ax.plot(np.cos(np.linspace(0, 2 * np.pi, 200)),
            np.sin(np.linspace(0, 2 * np.pi, 200)), color='navy', lw=2,
            zorder=3)
    ax.text(-0.35, -0.15, r'$C=B(0;1)$', fontsize=13, color='navy')

    # boundary point
    theta = np.pi / 4
    x0 = np.array([np.cos(theta), np.sin(theta)])
    ax.plot(*x0, 'o', color='black', ms=6, zorder=5)
    ax.text(x0[0] + 0.08, x0[1] + 0.12, r'$x$, $\|x\|=1$', fontsize=12)

    # tangent cone: half-plane {y : <y|x> <= 0}, drawn translated at x0
    tang_dir = np.array([-x0[1], x0[0]])  # direction orthogonal to x0
    L = 1.55
    p1 = x0 + L * tang_dir
    p2 = x0 - L * tang_dir
    # shade the half-plane (points x0 + s*tang_dir + t*(-x0), t>=0)
    verts = []
    for s in np.linspace(-L, L, 2):
        verts.append(x0 + s * tang_dir)
    inward = -x0
    poly_pts = [x0 + L * tang_dir, x0 - L * tang_dir,
                x0 - L * tang_dir + 1.3 * inward,
                x0 + L * tang_dir + 1.3 * inward]
    ax.add_patch(plt.Polygon(poly_pts, closed=True, color='seagreen',
                              alpha=0.25, zorder=1))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='seagreen', lw=2,
            zorder=4)
    mid = x0 + 0.75 * L * tang_dir
    ax.text(mid[0] + 0.05, mid[1] + 0.05, r'$T_C x$', fontsize=13,
            color='seagreen')

    # normal cone: ray R_+ x, drawn as an arrow starting at x0
    ax.annotate('', xy=tuple(x0 + 0.85 * x0), xytext=tuple(x0),
                arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.2),
                zorder=6)
    ax.text(*(x0 + 0.95 * x0 + np.array([0.05, 0.0])), r'$N_C x=\mathbb{R}_+x$',
            fontsize=12, color='crimson')

    # interior point
    xi = np.array([-0.35, -0.4])
    ax.plot(*xi, 's', color='dimgray', ms=6, zorder=5)
    ax.text(xi[0] - 0.85, xi[1] - 0.28,
            r'interior point: $T_Cx=\mathcal{H},\ N_Cx=\{0\}$', fontsize=10.5,
            color='dimgray')

    ax.set_title('Tangent and normal cone to the unit ball at a boundary '
                  'point', fontsize=12)
    savefig('fig_ball_tangent_normal.pdf', fig=fig)


# -----------------------------------------------------------------------
# Figure 4: tangent cone / normal cone to the ice-cream cone K at the
# boundary point x0 = (1,1)  (running numerical example, Example 6.40)
# -----------------------------------------------------------------------
def fig_cone_tangent_normal():
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    style_axes(ax, lim=3.0)

    t = np.linspace(0, 3.0, 50)
    xK = np.concatenate([-t[::-1], t])
    yK = np.abs(xK)
    ax.fill_between(xK, yK, 3.05, color='royalblue', alpha=0.30,
                     zorder=1)
    ax.plot([-3, 0, 3], [3, 0, 3], color='navy', lw=2, zorder=2)
    ax.text(-2.55, 1.85, r'$K=\{(x,y):y\geq|x|\}$', fontsize=12.5,
            color='navy')

    x0 = np.array([1.0, 1.0])
    ax.plot(*x0, 'o', color='black', ms=6, zorder=6)
    ax.text(x0[0] + 0.08, x0[1] - 0.28, r'$x_0=(1,1)$', fontsize=12)

    # tangent cone T_K(x0) = {(x,y): y >= x}: half-plane through the origin
    # bounded by the line y = x, shade it (translate not needed -- it is a
    # cone through 0, but we highlight it as attached at x0 for clarity)
    ax.fill_between([-3, 3], [-3, 3], [3, 3], color='seagreen', alpha=0.20,
                     zorder=0)
    ax.plot([-3, 3], [-3, 3], color='seagreen', lw=2, ls='--', zorder=2)
    ax.text(-2.9, -1.55, r'$T_K x_0=\{(x,y):y\geq x\}$', fontsize=12,
            color='seagreen')

    # normal cone N_K(x0) = R_+ (1,-1), drawn as arrow from x0
    ndir = np.array([1.0, -1.0]) / np.sqrt(2)
    ax.annotate('', xy=tuple(x0 + 1.3 * ndir), xytext=tuple(x0),
                arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.2),
                zorder=6)
    ax.text(*(x0 + 1.4 * ndir + np.array([0.05, -0.05])),
            r'$N_K x_0=\mathbb{R}_+(1,-1)$', fontsize=11.5, color='crimson')

    ax.set_title(r'Tangent and normal cone to $K$ at the boundary point'
                 r' $x_0=(1,1)$', fontsize=12)
    savefig('fig_cone_tangent_normal.pdf', fig=fig)


if __name__ == '__main__':
    fig_orthant_polar()
    fig_icecream_dual()
    fig_ball_tangent_normal()
    fig_cone_tangent_normal()
    print("All figures generated.")
