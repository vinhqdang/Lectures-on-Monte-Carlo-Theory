"""
gen_figures.py -- Generate all figures for Chapter 9: Lower Semicontinuous
Convex Functions
(Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in
Hilbert Spaces", 2nd ed., CMS Books in Mathematics, Springer 2017)

Run with: python3 gen_figures.py

Produces (as vector PDFs, in this directory):
  fig_lsc_vs_not.pdf     -- an lsc function (closed epigraph) side by side
                            with a non-lsc function (epigraph has a "hole"
                            at one point, caused by an upward spike)
  fig_closure.pdf        -- the lower semicontinuous convex envelope
                            (closure) of a convex function: the boundary
                            values are "filled in" by the limiting value so
                            that the epigraph becomes closed
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: lsc function vs. non-lsc function, epigraphs side by side
#
#   Left  (lsc):     f(x) = x^2                       everywhere continuous,
#                     hence lsc; epi f is closed.
#   Right (not lsc): g(x) = x^2 for x != 0, g(0) = 1   an upward spike at 0;
#                     liminf_{x->0} g(x) = 0 < g(0) = 1, so g violates
#                     lower semicontinuity at x = 0. The epigraph has a
#                     vertical "hole": the segment {0} x [0,1) is a limit of
#                     epigraph points but is NOT itself in epi g.
# ─────────────────────────────────────────────────────────────────────────
def fig_lsc_vs_not():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    xs = np.linspace(-1.6, 1.6, 400)

    # ---- left panel: lsc function f(x) = x^2 ----
    ax = axes[0]
    f = xs ** 2
    ax.fill_between(xs, f, 2.8, color='#dce9f5', alpha=0.9, zorder=1)
    ax.plot(xs, f, color='#2c6aad', lw=2.4, zorder=3)
    ax.plot(0, 0, 'o', color='#2c6aad', ms=8, zorder=4)
    ax.set_title(r"$f(x) = x^2$  (lsc: epigraph is closed)",
                 fontsize=11, color='#2c6aad', fontweight='bold')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.3, 2.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\xi$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.25)

    # ---- right panel: non-lsc function with an upward spike at 0 ----
    ax = axes[1]
    xs_left = np.linspace(-1.6, -0.02, 200)
    xs_right = np.linspace(0.02, 1.6, 200)
    ax.fill_between(xs_left, xs_left ** 2, 2.8, color='#fde8e8', alpha=0.9, zorder=1)
    ax.fill_between(xs_right, xs_right ** 2, 2.8, color='#fde8e8', alpha=0.9, zorder=1)
    # the epigraph "hole": the open vertical segment {0} x [0,1) is missing
    ax.fill_between([-0.02, 0.02], [1, 1], [2.8, 2.8], color='#fde8e8', alpha=0.9, zorder=1)
    ax.plot(xs_left, xs_left ** 2, color='#c0392b', lw=2.4, zorder=3)
    ax.plot(xs_right, xs_right ** 2, color='#c0392b', lw=2.4, zorder=3)
    # open circle at the removed limit point (0,0): NOT attained, epigraph
    # is missing the segment {0} x [0,1) right above it
    ax.plot(0, 0, 'o', mfc='white', mec='#c0392b', mew=2, ms=9, zorder=4)
    # filled dot at the actual (too high) function value g(0) = 1
    ax.plot(0, 1, 'o', color='#c0392b', ms=9, zorder=4)
    ax.annotate("actual value\n" + r"$g(0)=1$", (0, 1), xytext=(0.35, 1.55),
                fontsize=9, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax.annotate(r"$\liminf_{x\to0} g(x) = 0$", (0, 0), xytext=(-1.55, 0.55),
                fontsize=9, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax.annotate("missing sliver of\nthe epigraph (a \"hole\")", (0, 0.5),
                xytext=(0.15, -0.28), fontsize=8.5, color='#7f2d20')
    ax.set_title(r"$g(0){=}1,\ g(x){=}x^2\ (x{\neq}0)$: not lsc at $0$",
                 fontsize=11, color='#c0392b', fontweight='bold')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.3, 2.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\xi$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.25)

    fig.suptitle(r"$f$ is lsc at $x$ iff $f(x) \le \liminf_{y\to x} f(y)$ "
                 r"iff epi $f$ is closed",
                 fontsize=11.5, y=1.03)
    savefig("fig_lsc_vs_not.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: closure (lsc convex envelope) of a convex function
#
#   g(x) = x^2 on (-1,1), and g(x) = +infty otherwise (including AT the
#   endpoints x = +-1, even though lim_{x -> +-1} x^2 = 1 is finite: g is
#   defined to be +infty there by fiat, so g fails to be lsc at +-1).
#
#   The closure g-check "fills in" the boundary with the honest limiting
#   value: g-check(x) = x^2 on the closed interval [-1,1], +infty outside.
#   This matches Proposition 9.33/9.34: the lsc envelope of a convex
#   function that is continuous on its open domain assigns the boundary
#   the limiting value of the function.
# ─────────────────────────────────────────────────────────────────────────
def fig_closure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    xs_open = np.linspace(-0.995, 0.995, 300)

    # ---- left panel: g, BEFORE closure ----
    ax = axes[0]
    ax.fill_between(xs_open, xs_open ** 2, 2.6, color='#fde8e8', alpha=0.9, zorder=1)
    ax.plot(xs_open, xs_open ** 2, color='#c0392b', lw=2.4, zorder=3)
    # open circles at the excluded endpoint values (x=+-1, height 1)
    ax.plot([-1, 1], [1, 1], 'o', mfc='white', mec='#c0392b', mew=2, ms=9, zorder=4)
    ax.axvline(-1, color='#c0392b', lw=1.0, ls=':')
    ax.axvline(1, color='#c0392b', lw=1.0, ls=':')
    ax.annotate(r"$g(\pm1) = +\infty$" "\n(declared by fiat)", (1, 1),
                xytext=(1.05, 1.9), fontsize=9, color='#c0392b')
    ax.set_title(r"$g$: $x^2$ on $(-1,1)$, $+\infty$ elsewhere (not lsc)",
                 fontsize=10.8, color='#c0392b', fontweight='bold')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-0.3, 2.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\xi$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.25)

    # ---- right panel: the closure g-check, AFTER closure ----
    ax = axes[1]
    xs_closed = np.linspace(-1, 1, 300)
    ax.fill_between(xs_closed, xs_closed ** 2, 2.6, color='#dce9f5', alpha=0.9, zorder=1)
    ax.plot(xs_closed, xs_closed ** 2, color='#2c6aad', lw=2.4, zorder=3)
    # filled dots at the now-included endpoint values (limits filled in)
    ax.plot([-1, 1], [1, 1], 'o', color='#2c6aad', ms=9, zorder=4)
    ax.axvline(-1, color='#2c6aad', lw=1.0, ls=':')
    ax.axvline(1, color='#2c6aad', lw=1.0, ls=':')
    ax.annotate(r"$\check{g}(\pm1)=\lim_{y\to\pm1}g(y)=1$"
                "\n(limit filled in)", (1, 1),
                xytext=(1.05, 1.9), fontsize=9, color='#2c6aad')
    ax.set_title(r"$\check{g}$: $x^2$ on $[-1,1]$, $+\infty$ elsewhere (lsc)",
                 fontsize=10.8, color='#2c6aad', fontweight='bold')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-0.3, 2.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\xi$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.25)

    fig.suptitle(r"Closure: $\check{g} = \sup\{h \in \Gamma(\mathcal{H}) : h \le g\}$, "
                 r"  epi $\check{g}$ = closed convex hull of epi $g$",
                 fontsize=11, y=1.03)
    savefig("fig_closure.pdf")


if __name__ == "__main__":
    fig_lsc_vs_not()
    fig_closure()
    print("All figures generated.")
