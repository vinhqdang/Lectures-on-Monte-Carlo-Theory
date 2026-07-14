#!/usr/bin/env python3
"""
Figure generator for Chapter 11 (Convex Minimization Problems) slides.

Generates, as vector PDFs in the current directory:
  1. fig_unique_vs_flat.pdf
       Two convex functions side by side: one with a unique minimizer
       (a strictly/strongly convex quadratic) and one with a flat
       region of minimizers (a convex function that is affine, hence
       constant at its minimum value, on an interval).
  2. fig_minimizing_sequence.pdf
       The values f(x_n) of a minimizing sequence for the running
       example f(x) = (x-3)^2 + 1, decreasing monotonically toward
       the infimum value 1, together with x_n -> 3 in the domain.
  3. fig_quadratic_running_example.pdf
       The running numeric example: f(x) = (x-3)^2 + 1 minimized
       over R (minimizer x* = 3, min value 1) versus minimized over
       the restricted closed convex set C = [0,2] (minimizer x* = 2,
       min value 2, attained at the boundary of C).

Plain python3 + matplotlib only, Agg backend, no LaTeX rendering
dependency beyond matplotlib's own mathtext.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_unique_vs_flat():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel 1: unique minimizer -- strictly convex quadratic
    x = np.linspace(-1, 7, 400)
    f = (x - 3) ** 2 + 1
    ax = axes[0]
    ax.plot(x, f, color="tab:blue", lw=2.5)
    ax.plot([3], [1], "o", color="tab:red", ms=8, zorder=5)
    ax.annotate(r"unique minimizer $\bar x = 3$",
                xy=(3, 1), xytext=(3.2, 6),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title(r"Strictly convex: $f(x)=(x-3)^2+1$" "\n"
                 r"$\mathrm{Argmin}\,f=\{3\}$ (singleton)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_ylim(-1, 14)

    # Panel 2: flat region of minimizers -- convex but not strictly so
    def g(x):
        # Convex "valley with a flat bottom": quadratic outside [1,3],
        # constant (value 0) on [1,3].
        y = np.zeros_like(x)
        left = x < 1
        right = x > 3
        y[left] = (x[left] - 1) ** 2
        y[right] = (x[right] - 3) ** 2
        return y

    x2 = np.linspace(-2, 6, 400)
    ax = axes[1]
    ax.plot(x2, g(x2), color="tab:blue", lw=2.5)
    ax.axhspan(0, 0, color="none")
    ax.plot([1, 3], [0, 0], color="tab:red", lw=4, zorder=5,
            solid_capstyle="round")
    ax.annotate(r"$\mathrm{Argmin}\,g=[1,3]$" "\n(a whole interval)",
                xy=(2, 0), xytext=(2.3, 5),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("Merely convex (not strictly):\n"
                 "flat region of minimizers")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$g(x)$")
    ax.set_ylim(-1, 14)

    fig.suptitle("Uniqueness of minimizers depends on strict convexity "
                 r"(Corollary 11.9)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("fig_unique_vs_flat.pdf")
    plt.close(fig)


def fig_minimizing_sequence():
    n = np.arange(1, 16)
    xn = 3.0 - 1.0 / n           # x_n -> 3, minimizing sequence in domain
    fn = (xn - 3) ** 2 + 1       # f(x_n) -> 1 = inf f(R)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    ax.plot(n, fn, "o-", color="tab:blue", ms=5)
    ax.axhline(1.0, color="tab:red", ls="--", lw=1.5,
               label=r"$\inf f(\mathbb{R}) = 1$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$f(x_n)$")
    ax.set_title(r"$f(x_n)\downarrow \inf f(\mathbb{R})$")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(n, xn, "o-", color="tab:green", ms=5)
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.5,
               label=r"minimizer $\bar x = 3$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$x_n$")
    ax.set_title(r"$x_n = 3 - 1/n \to \bar x$")
    ax.legend(loc="lower right")

    fig.suptitle(r"Minimizing sequence for $f(x)=(x-3)^2+1$")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("fig_minimizing_sequence.pdf")
    plt.close(fig)


def fig_quadratic_running_example():
    x = np.linspace(-1, 7, 400)
    f = (x - 3) ** 2 + 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, f, color="tab:blue", lw=2.5, label=r"$f(x)=(x-3)^2+1$")

    # Highlight C = [0,2]
    xc = np.linspace(0, 2, 200)
    ax.plot(xc, (xc - 3) ** 2 + 1, color="tab:orange", lw=5, alpha=0.5,
            label=r"$f$ restricted to $C=[0,2]$", zorder=1)

    # Unconstrained minimizer over R
    ax.plot([3], [1], "o", color="tab:red", ms=9, zorder=5)
    ax.annotate(r"min over $\mathbb{R}$:" "\n" r"$\bar x=3,\ f(\bar x)=1$",
                xy=(3, 1), xytext=(3.6, 4.5),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Constrained minimizer over C = [0,2] -- attained at boundary x=2
    ax.plot([2], [2], "s", color="tab:purple", ms=9, zorder=5)
    ax.annotate(r"min over $C=[0,2]$:" "\n" r"$x_C=2,\ f(x_C)=2$"
                "\n(attained at $\\mathrm{bdry}\\,C$)",
                xy=(2, 2), xytext=(-0.9, 7.5),
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.axvspan(0, 2, color="tab:orange", alpha=0.08)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_ylim(-1, 14)
    ax.set_title("Running example: minimizing over $\\mathbb{R}$ "
                 "vs.\\ over a restricted convex set $C$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("fig_quadratic_running_example.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_unique_vs_flat()
    fig_minimizing_sequence()
    fig_quadratic_running_example()
    print("Figures written:")
    print("  fig_unique_vs_flat.pdf")
    print("  fig_minimizing_sequence.pdf")
    print("  fig_quadratic_running_example.pdf")
