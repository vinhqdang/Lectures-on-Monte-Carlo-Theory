#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 21 (Finer Properties of Monotone Operators)
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd ed.

Generates three vector (PDF) figures used by chapter21_slides.tex:

  fig_minty.pdf       -- Minty's theorem for the 1D operator A(x) = x^3:
                         Id + A is a strictly increasing bijection of R onto R,
                         illustrating "A maximally monotone iff ran(Id+A)=H".

  fig_domain_range.pdf -- The running set-valued example A = d|.| (subdifferential
                          of the absolute value): dom A = R (unbounded) but
                          ran A = [-1,1] (bounded), showing domain and range can
                          behave very differently.

  fig_local_bdd.pdf   -- Local boundedness / Rockafellar-Vesely theorem: an
                          operator whose domain is the open interval (-1,1)
                          blows up as x approaches the boundary, illustrating
                          that local boundedness can fail exactly at bdry(dom A).

All figures are plain matplotlib, saved as PDF (vector), no external data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------
# Figure 1: Minty's theorem for A(x) = x^3
# ---------------------------------------------------------------------------
def fig_minty():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    x = np.linspace(-1.8, 1.8, 400)
    Tx = x + x ** 3  # T = Id + A

    ax.plot(x, Tx, color="#1f77b4", lw=2.5, label=r"$T(x) = x + x^3 = (\mathrm{Id}+A)(x)$")
    ax.plot(x, x, color="#999999", lw=1.2, ls="--", label=r"$y = x$ (identity, for reference)")

    # Mark a few explicit (x, T(x)) pairs used in the worked example on the slides.
    sample_xs = [-1.5, -1.0, 0.0, 0.5, 1.0, 1.2]
    for xs_ in sample_xs:
        ys_ = xs_ + xs_ ** 3
        ax.plot([xs_], [ys_], "o", color="#d62728", zorder=5)
        ax.annotate(f"({xs_:g}, {ys_:g})", (xs_, ys_),
                    textcoords="offset points", xytext=(6, 6), fontsize=8.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T(x)$")
    ax.set_title("Minty's Theorem for $A(x)=x^3$:\n"
                 r"$T=\mathrm{Id}+A$ is a strictly increasing bijection $\mathbb{R}\to\mathbb{R}$")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_minty.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Domain and range of A = d|.|  (subdifferential of the abs. value)
# ---------------------------------------------------------------------------
def fig_domain_range():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # Graph of A(x) = sign(x) for x != 0, A(0) = [-1, 1].
    x_pos = np.linspace(0.02, 2.0, 200)
    x_neg = np.linspace(-2.0, -0.02, 200)

    ax.plot(x_pos, np.ones_like(x_pos), color="#1f77b4", lw=2.5)
    ax.plot(x_neg, -np.ones_like(x_neg), color="#1f77b4", lw=2.5)
    # Vertical segment at x=0 representing the set A(0) = [-1,1].
    ax.plot([0, 0], [-1, 1], color="#1f77b4", lw=2.5,
            label=r"$\mathrm{gra}\, A,\ A = \partial|\cdot|$")
    ax.plot([0], [1], "o", mfc="white", mec="#1f77b4", zorder=5)
    ax.plot([0], [-1], "o", mfc="white", mec="#1f77b4", zorder=5)
    ax.plot([0], [1], ".", color="#1f77b4", zorder=6, ms=4)
    ax.plot([0], [-1], ".", color="#1f77b4", zorder=6, ms=4)

    # Shade / mark dom A (the whole x-axis) and ran A (the segment [-1,1] on y-axis).
    ax.axhspan(-1, 1, xmin=0, xmax=1, color="#ff7f0e", alpha=0.10)
    ax.annotate(r"$\mathrm{ran}\,A=[-1,1]$" "\n(bounded)", xy=(1.55, 0.0),
                fontsize=9, color="#d62728", ha="left", va="center")
    ax.annotate(r"$\mathrm{dom}\,A=\mathbb{R}$ (unbounded)", xy=(0.0, -1.55),
                fontsize=9, color="#2ca02c", ha="center")

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u \in A(x)$")
    ax.set_title(r"Domain vs.\ Range: $A=\partial|\cdot|$ on $\mathbb{R}$" "\n"
                 r"unbounded domain, bounded range")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_domain_range.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Local boundedness fails at the boundary of dom A
# ---------------------------------------------------------------------------
def fig_local_bdd():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # A(x) = x / (1 - x^2) on (-1, 1): maximally monotone-type blow-up example,
    # dom A = (-1,1) is open so bdry(dom A) = {-1, 1}. |A(x)| -> infinity as
    # x -> +-1, illustrating that local boundedness must fail there
    # (Theorem 21.18, Rockafellar-Vesely).
    x = np.linspace(-0.985, 0.985, 600)
    y = x / (1 - x ** 2)
    y_clip = np.clip(y, -12, 12)

    ax.plot(x, y_clip, color="#1f77b4", lw=2.2, label=r"$A(x)=\dfrac{x}{1-x^2}$, $\mathrm{dom}\,A=(-1,1)$")
    ax.axvline(-1, color="#d62728", lw=1.3, ls="--")
    ax.axvline(1, color="#d62728", lw=1.3, ls="--")
    ax.annotate(r"$x\to -1^+ \Rightarrow |A(x)|\to\infty$", xy=(-0.97, -10.5),
                fontsize=8.5, color="#d62728", ha="left")
    ax.annotate(r"$x\to 1^- \Rightarrow |A(x)|\to\infty$", xy=(0.97, 10.5),
                fontsize=8.5, color="#d62728", ha="right")

    # Mark an interior point where A is locally bounded.
    x0 = 0.3
    ax.plot([x0], [x0 / (1 - x0 ** 2)], "o", color="#2ca02c", zorder=5)
    ax.annotate("interior point:\nlocally bounded here", xy=(x0, x0 / (1 - x0 ** 2)),
                textcoords="offset points", xytext=(-90, 25), fontsize=8.5, color="#2ca02c")

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-13, 13)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$A(x)$")
    ax.set_title("Local Boundedness Can Fail Only at $\\mathrm{bdry}(\\mathrm{dom}\\,A)$\n"
                 "(Rockafellar–Veselý, Theorem 21.18)")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_local_bdd.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_minty()
    fig_domain_range()
    fig_local_bdd()
    print("Wrote fig_minty.pdf, fig_domain_range.pdf, fig_local_bdd.pdf")
