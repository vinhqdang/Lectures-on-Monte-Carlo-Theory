#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 16: Subdifferentiability of Convex Functions
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd ed.

Generates (as vector PDFs, saved into this directory):
  fig_abs_subdifferential.pdf   -- f(x)=|x|, its supporting lines, and the
                                    graph of the multifunction x -> d|.|(x)
  fig_fermat_rule.pdf           -- Fermat's rule 0 in df(x*) illustrated on
                                    f(x) = |x-2| + x^2
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ----------------------------------------------------------------------
# Figure (a): f(x) = |x| and its subdifferential
# ----------------------------------------------------------------------

def make_abs_subdifferential_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    x = np.linspace(-2, 2, 400)
    fx = np.abs(x)

    # --- Left panel: graph of |x| with supporting affine minorants ---
    ax1.plot(x, fx, color='black', linewidth=2.5, label=r'$f(x)=|x|$', zorder=5)

    # Supporting lines at x=0 for several slopes in [-1,1]
    slopes_at_0 = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(slopes_at_0)))
    xs = np.linspace(-2, 2, 50)
    for s, c in zip(slopes_at_0, colors):
        ax1.plot(xs, 0 + s * (xs - 0), color=c, linewidth=1.3, alpha=0.85,
                  linestyle='--')

    # Single supporting line (the unique tangent / subgradient) at x=-1.2 and x=1.2
    for x0 in [-1.2, 1.2]:
        u = np.sign(x0)  # the unique subgradient
        f0 = abs(x0)
        xs2 = np.linspace(x0 - 0.9, x0 + 0.9, 20)
        ax1.plot(xs2, f0 + u * (xs2 - x0), color='green', linewidth=1.6)
        ax1.plot([x0], [f0], marker='o', color='green', markersize=5, zorder=6)

    ax1.plot([0], [0], marker='o', color='black', markersize=6, zorder=6)
    ax1.annotate(r'kink at $x=0$:' + '\n' + r'$\partial f(0)=[-1,1]$',
                 xy=(0, 0), xytext=(-1.95, 1.55),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate(r'unique slope $+1$', xy=(1.2, 1.2), xytext=(0.15, 1.85),
                 fontsize=9.5, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(r'unique slope $-1$', xy=(-1.2, 1.2), xytext=(-1.95, 0.55),
                 fontsize=9.5, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-0.3, 2.1)
    ax1.axhline(0, color='gray', linewidth=0.6)
    ax1.axvline(0, color='gray', linewidth=0.6)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\mathbb{R}$')
    ax1.set_title('Supporting affine minorants of $|x|$')
    ax1.legend(loc='upper center')

    # --- Right panel: graph of the multifunction x -> partial|.|(x) ---
    ax2.plot([-2, 0], [-1, -1], color='crimson', linewidth=2.5)
    ax2.plot([0, 2], [1, 1], color='crimson', linewidth=2.5)
    ax2.plot([0, 0], [-1, 1], color='crimson', linewidth=2.5)
    ax2.plot([0], [-1], marker='o', mfc='white', mec='crimson', markersize=6)
    ax2.plot([0], [1], marker='o', mfc='white', mec='crimson', markersize=6)
    ax2.plot([-2], [-1], marker='o', mfc='crimson', mec='crimson', markersize=5)
    ax2.plot([2], [1], marker='o', mfc='crimson', mec='crimson', markersize=5)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.6, 1.6)
    ax2.axhline(0, color='gray', linewidth=0.6)
    ax2.axvline(0, color='gray', linewidth=0.6)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$u \in \partial|\cdot|(x)$')
    ax2.set_title(r'The multifunction $x \mapsto \partial|\cdot|(x)$')
    ax2.text(0.05, 1.25, r'$\partial|\cdot|(0)=[-1,1]$', color='crimson', fontsize=10)

    fig.tight_layout()
    fig.savefig('fig_abs_subdifferential.pdf')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure (b): Fermat's rule for f(x) = |x-2| + x^2
# ----------------------------------------------------------------------

def make_fermat_rule_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    def f(x):
        return np.abs(x - 2) + x ** 2

    def subdiff_single(x):
        # returns the (unique) subgradient for x != 2
        return (2 * x - 1) if x < 2 else (2 * x + 1)

    x = np.linspace(-1.5, 4, 400)
    ax1.plot(x, f(x), color='black', linewidth=2.5,
              label=r'$f(x)=|x-2|+x^2$')

    xstar = 0.5
    fstar = f(xstar)
    ax1.plot([xstar], [fstar], marker='o', color='crimson', markersize=7, zorder=5)
    ax1.plot(np.linspace(xstar - 1, xstar + 1, 10),
              fstar + 0 * np.linspace(-1, 1, 10),
              color='crimson', linewidth=1.6, linestyle='--',
              label=r'horizontal support line ($0\in\partial f(x^\star)$)')
    ax1.annotate(r'$x^\star=1/2$' + '\n' + r'$0\in\partial f(x^\star)$',
                 xy=(xstar, fstar), xytext=(xstar - 1.3, fstar + 2.4),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))

    xkink = 2.0
    fkink = f(xkink)
    ax1.plot([xkink], [fkink], marker='s', color='blue', markersize=6, zorder=5)
    ax1.annotate(r'kink at $x=2$' + '\n' + r'$\partial f(2)=[3,5]\not\ni 0$',
                 xy=(xkink, fkink), xytext=(2.05, fkink - 2.6),
                 fontsize=10, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$f(x)$')
    ax1.set_title(r"Fermat's rule: $\mathrm{Argmin}\,f=\mathrm{zer}\,\partial f$")
    ax1.legend(loc='upper center', fontsize=8.5)
    ax1.set_ylim(-0.5, 9)

    # --- Right panel: the multifunction x -> partial f(x) ---
    xs_left = np.linspace(-1.5, 2, 200)
    xs_right = np.linspace(2, 4, 200)
    ax2.plot(xs_left, 2 * xs_left - 1, color='teal', linewidth=2.2,
              label=r'$2x-1$ (for $x<2$)')
    ax2.plot(xs_right, 2 * xs_right + 1, color='darkorange', linewidth=2.2,
              label=r'$2x+1$ (for $x>2$)')
    ax2.plot([2, 2], [3, 5], color='blue', linewidth=2.2,
              label=r'$[3,5]$ (at $x=2$)')
    ax2.plot([2], [3], marker='o', mfc='blue', mec='blue', markersize=5)
    ax2.plot([2], [5], marker='o', mfc='blue', mec='blue', markersize=5)

    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.axvline(xstar, color='crimson', linewidth=1.2, linestyle='--')
    ax2.plot([xstar], [0], marker='o', color='crimson', markersize=7, zorder=6)
    ax2.annotate(r'$0\in\partial f(x^\star)$', xy=(xstar, 0),
                 xytext=(-1.4, 1.8), fontsize=10, color='crimson',
                 arrowprops=dict(arrowstyle='->', color='crimson'))

    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$u\in\partial f(x)$')
    ax2.set_title(r'$\partial f$ for $f(x)=|x-2|+x^2$')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim(-1.5, 4)

    fig.tight_layout()
    fig.savefig('fig_fermat_rule.pdf')
    plt.close(fig)


if __name__ == '__main__':
    make_abs_subdifferential_figure()
    make_fermat_rule_figure()
    print("Figures written: fig_abs_subdifferential.pdf, fig_fermat_rule.pdf")
