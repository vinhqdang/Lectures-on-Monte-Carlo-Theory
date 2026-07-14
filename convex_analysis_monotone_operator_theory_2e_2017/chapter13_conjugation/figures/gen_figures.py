"""
gen_figures.py -- Generate all figures for Chapter 13: Conjugation
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory
in Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer 2017)

Run with: python3 gen_figures.py
All output figures are saved as vector PDFs into this "figures" directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIGURES_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: geometric picture of the Fenchel conjugate as the "best affine
# minorant" construction, cf. Fig. 13.1 in the book, illustrated for the
# self-conjugate function f(x) = x^2/2.
# ─────────────────────────────────────────────────────────────────────────
def fig_geometric_conjugate():
    x = np.linspace(-3.2, 3.2, 600)
    f = 0.5 * x**2

    u = 1.5                      # the point at which we evaluate f*
    xstar = u                    # argmax of  u*x - f(x)  is x = u
    fstar_u = 0.5 * u**2         # f*(u) = u^2/2  (self-conjugate)

    line_through_origin = u * x            # graph of <.|u>
    tangent_minorant = u * x - fstar_u      # the best affine minorant with slope u

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(x, f, color='#1f4e79', lw=2.5, label=r'$\mathrm{gra}\,f,\ f(x)=x^2/2$')
    ax.plot(x, line_through_origin, color='#c0392b', lw=1.8,
            label=r'$\mathrm{gra}\,\langle\cdot\mid u\rangle,\ u=1.5$')
    ax.plot(x, tangent_minorant, color='#c0392b', lw=1.8, ls='--',
            label=r'$\langle\cdot\mid u\rangle - f^*(u)$ (best affine minorant)')

    # mark the touching point
    ax.plot([xstar], [f[np.argmin(np.abs(x - xstar))]], 'o', color='#1f4e79', ms=6, zorder=5)

    # vertical dashed arrow at x=0 showing the gap f*(u)
    ax.annotate('', xy=(0, 0), xytext=(0, -fstar_u),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax.text(0.12, -fstar_u / 2, r'$f^*(u)$', fontsize=13, va='center')

    ax.axhline(0, color='gray', lw=0.6)
    ax.axvline(0, color='gray', lw=0.6)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3, 6)
    ax.set_xlabel(r'$x$')
    ax.set_title(r'$f^*(u)=\sup_{x}\,(\langle x\mid u\rangle - f(x))$'
                 r' as the largest vertical gap')
    ax.legend(loc='upper center', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    savefig("fig_geometric_conjugate.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: f and f* side by side for f(x) = |x|, showing that
# f* = iota_{[-1,1]} (the indicator function of [-1,1]).
# ─────────────────────────────────────────────────────────────────────────
def fig_abs_and_conjugate():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # left panel: f(x) = |x|
    x = np.linspace(-3, 3, 400)
    f = np.abs(x)
    axes[0].plot(x, f, color='#1f4e79', lw=2.5)
    axes[0].set_title(r'$f(x) = |x|$')
    axes[0].set_xlabel(r'$x$')
    axes[0].axhline(0, color='gray', lw=0.6)
    axes[0].axvline(0, color='gray', lw=0.6)
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(-0.5, 3.2)

    # right panel: f*(u) = 0 on [-1,1], +infty outside
    axes[1].plot([-1, 1], [0, 0], color='#c0392b', lw=3.5, solid_capstyle='round',
                 label=r'$f^*(u)=0$ on $[-1,1]$')
    # vertical dashed rays indicating +infinity outside [-1,1]
    axes[1].annotate('', xy=(-1, 3.0), xytext=(-1, 0),
                      arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=2, ls='dashed'))
    axes[1].annotate('', xy=(1, 3.0), xytext=(1, 0),
                      arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=2, ls='dashed'))
    axes[1].text(-1, 3.15, r'$+\infty$', ha='center', fontsize=10, color='#c0392b')
    axes[1].text(1, 3.15, r'$+\infty$', ha='center', fontsize=10, color='#c0392b')
    axes[1].plot([-3, -1], [3.0, 3.0], color='#c0392b', lw=1.2, ls=':')
    axes[1].plot([1, 3], [3.0, 3.0], color='#c0392b', lw=1.2, ls=':')

    axes[1].set_title(r'$f^*(u) = \iota_{[-1,1]}(u)$')
    axes[1].set_xlabel(r'$u$')
    axes[1].axhline(0, color='gray', lw=0.6)
    axes[1].axvline(0, color='gray', lw=0.6)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-0.5, 3.2)

    fig.suptitle(r'Fenchel conjugation turns a "kink" into a "hard wall": '
                 r'$f=|\cdot|\ \longleftrightarrow\ f^*=\iota_{[-1,1]}$', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    savefig("fig_abs_and_conjugate.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: numerically computed conjugate via a discretized sup, compared
# against the closed-form analytic conjugate, for both running examples.
# ─────────────────────────────────────────────────────────────────────────
def numerical_conjugate(f, x_grid, u_grid):
    """Brute-force f*(u) = sup_x (u*x - f(x)) over a discretized grid."""
    fx = f(x_grid)
    # outer difference: shape (len(u_grid), len(x_grid))
    vals = np.outer(u_grid, x_grid) - fx[None, :]
    return np.max(vals, axis=1)


def fig_numeric_vs_analytic():
    x_grid = np.linspace(-8, 8, 4001)
    u_grid = np.linspace(-2.5, 2.5, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) f(x) = x^2/2  -->  f*(u) = u^2/2
    f1 = lambda x: 0.5 * x**2
    fstar_numeric_1 = numerical_conjugate(f1, x_grid, u_grid)
    fstar_analytic_1 = 0.5 * u_grid**2

    axes[0].plot(u_grid, fstar_analytic_1, color='#1f4e79', lw=3, alpha=0.4,
                 label='analytic $f^*(u)=u^2/2$')
    axes[0].plot(u_grid, fstar_numeric_1, color='#c0392b', lw=1.4, ls='--',
                 label='discretized sup')
    axes[0].set_title(r'$f(x)=x^2/2$ (self-conjugate)')
    axes[0].set_xlabel(r'$u$')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # (b) f(x) = |x|  -->  f*(u) = iota_{[-1,1]}(u), truncated to a cap for plotting
    f2 = lambda x: np.abs(x)
    fstar_numeric_2 = numerical_conjugate(f2, x_grid, u_grid)
    cap = 6.0
    fstar_numeric_2_capped = np.minimum(fstar_numeric_2, cap)

    axes[1].plot(u_grid, fstar_numeric_2_capped, color='#c0392b', lw=1.6,
                 label='discretized sup (capped at plot border)')
    axes[1].axvline(-1, color='gray', ls=':', lw=1)
    axes[1].axvline(1, color='gray', ls=':', lw=1)
    axes[1].set_title(r'$f(x)=|x|\ \Rightarrow\ f^*=\iota_{[-1,1]}$')
    axes[1].set_xlabel(r'$u$')
    axes[1].set_ylim(-0.5, cap + 0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle('Numerically evaluating the conjugate via a discretized supremum', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    savefig("fig_numeric_vs_analytic.pdf")


if __name__ == "__main__":
    fig_geometric_conjugate()
    fig_abs_and_conjugate()
    fig_numeric_vs_analytic()
    print("All figures generated.")
