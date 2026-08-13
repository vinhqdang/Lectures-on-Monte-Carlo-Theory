#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 2.1 "Binary Strings" slides
(An Introduction to Universal Artificial Intelligence, Hutter et al. 2024)

Run with:
    conda run -n py313 python3 gen_figures.py

All figures are saved as PDF into ./figures/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


# ----------------------------------------------------------------------------
# Figure 1: Binary tree showing strings as nodes, and a prefix-free code
#           as a set of "cut" leaves that block their own descendants.
# ----------------------------------------------------------------------------
def fig_binary_tree():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    def draw_tree(ax, depth, highlight=None, title=""):
        highlight = highlight or set()
        # positions: node at (level, index within level), x in [0,1]
        pos = {}
        pos[""] = (0.5, depth)
        for lvl in range(1, depth + 1):
            n = 2 ** lvl
            for i in range(n):
                label = format(i, "0" + str(lvl) + "b")
                x = (i + 0.5) / n
                pos[label] = (x, depth - lvl)
        # draw edges
        for lvl in range(1, depth + 1):
            n = 2 ** lvl
            for i in range(n):
                label = format(i, "0" + str(lvl) + "b")
                parent = label[:-1]
                x0, y0 = pos[parent]
                x1, y1 = pos[label]
                ax.plot([x0, x1], [y0, y1], color="0.6", lw=1.2, zorder=1)
        # draw nodes
        for label, (x, y) in pos.items():
            is_hi = label in highlight
            # a node is "blocked" (descendant of a highlighted codeword)
            blocked = any(label != h and label.startswith(h) for h in highlight)
            if is_hi:
                color = "#1b6ca8"
                ax.scatter([x], [y], s=260, color=color, zorder=3,
                           edgecolor="black", linewidth=1.2)
                txt = label if label != "" else r"$\epsilon$"
                ax.text(x, y - 0.32, txt, ha="center", va="top",
                        fontsize=11, color=color, fontweight="bold",
                        family="monospace")
            elif blocked:
                ax.scatter([x], [y], s=60, color="0.85", zorder=2,
                           edgecolor="0.6", linewidth=0.8)
            else:
                ax.scatter([x], [y], s=70, color="white", zorder=2,
                           edgecolor="black", linewidth=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, depth + 0.3)
        ax.set_title(title)
        ax.axis("off")

    draw_tree(axes[0], 3, highlight={"0", "10", "110", "111"},
              title="Prefix-free set\n" r"$\mathcal{P}=\{0,10,110,111\}$")
    axes[0].text(0.5, -0.55,
                 "No codeword lies on the path to another:\nevery message is uniquely decodable.",
                 ha="center", va="top", fontsize=10.5)

    draw_tree(axes[1], 3, highlight={"0", "01", "011"},
              title="NOT prefix-free\n" r"$\{0,01,011\}$")
    axes[1].text(0.5, -0.55,
                 r"$0$ is a proper prefix of $01$ and of $011$:"
                 "\nreceiving 011 is ambiguous.",
                 ha="center", va="top", fontsize=10.5)

    fig.suptitle("Binary tree view of prefix codes (depth-3 strings shown)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_binary_tree.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: Cylinder sets Gamma_x as nested closed intervals of [0,1]
#           (own rendition of the book's Figure 2.1 idea).
# ----------------------------------------------------------------------------
def fig_cylinder_sets():
    fig, ax = plt.subplots(figsize=(10, 4.6))

    # helper: f(x) = interval [b(x)/2^l(x), (b(x)+1)/2^l(x)] for the cylinder Gamma_x
    def interval(x):
        if x == "":
            return (0.0, 1.0)
        l = len(x)
        b = int(x, 2)
        return (b / 2 ** l, (b + 1) / 2 ** l)

    rows = [
        ("", 3),
        ("0", 2), ("1", 2),
        ("00", 1), ("01", 1), ("10", 1), ("11", 1),
    ]
    colors = {"": "#444444", "0": "#1b6ca8", "1": "#c0392b",
              "00": "#1b6ca8", "01": "#1b6ca8", "10": "#c0392b", "11": "#c0392b"}

    for label, y in rows:
        lo, hi = interval(label)
        c = colors[label]
        ax.plot([lo, hi], [y, y], color=c, lw=6, solid_capstyle="butt", alpha=0.85, zorder=2)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=c, lw=2)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=c, lw=2)
        mid = (lo + hi) / 2
        txt = r"$\Gamma_\epsilon$" if label == "" else r"$\Gamma_{%s}$" % label
        ax.text(mid, y + 0.22, txt, ha="center", va="bottom", fontsize=12, color=c,
                family="monospace" if label else None)

    for x in np.linspace(0, 1, 5):
        ax.axvline(x, color="0.9", lw=0.8, zorder=0)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.4, 3.7)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(r"real line via $f(\omega)=\sum_{n\geq1} 2^{-n}\omega_n$")
    ax.set_title(r"Cylinder sets $\Gamma_x=\{x\omega : \omega\in\mathbb{B}^\infty\}$ as nested closed intervals of $[0,1]$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_cylinder_sets.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: Codeword-length overhead of E_0 (zeroth order), x-bar (first
#           order) and x' (second order) self-delimiting prefix codes.
# ----------------------------------------------------------------------------
def fig_code_length_growth():
    ell = np.arange(1, 33)  # length l(x) of the underlying string x, from 1 to 32 bits

    # l(E_0(x)) = <x> + 1, and by Prop 2.1.3, 2^l(x)-1 <= <x> <= 2^(l(x)+1)-2
    # we plot the *upper bound* 2^(l(x)+1)-1 for l(E_0(x)) as the representative curve
    len_E0_upper = 2.0 ** (ell + 1) - 1  # exponential
    len_xbar = 2 * ell + 1  # exact: l(xbar) = 2 l(x) + 1
    len_xprime = ell + 2 * np.log2(np.maximum(ell, 1)) + 1  # approx: l(x) + 2 log2(l(x)) + O(1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    ax.plot(ell, len_E0_upper, "o-", color="#c0392b", label=r"$\ell(E_0(x))$ (0th order, worst case)")
    ax.plot(ell, len_xbar, "s-", color="#1b6ca8", label=r"$\ell(\overline{x})$ (1st order)")
    ax.plot(ell, len_xprime, "^-", color="#27ae60", label=r"$\ell(x')$ (2nd order)")
    ax.plot(ell, ell, "--", color="0.5", label=r"$\ell(x)$ (no overhead)")
    ax.set_yscale("log")
    ax.set_xlabel(r"length of the message $\ell(x)$ (bits)")
    ax.set_ylabel("codeword length (bits, log scale)")
    ax.set_title("Codeword length vs. message length")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ell, len_xbar - ell, "s-", color="#1b6ca8", label=r"overhead of $\overline{x}$: $2\log_2\ell(x)+1$")
    ax.plot(ell, len_xprime - ell, "^-", color="#27ae60", label=r"overhead of $x'$: $\approx\log_2\ell(x)+2\log_2\log_2\ell(x)$")
    ax.set_xlabel(r"length of the message $\ell(x)$ (bits)")
    ax.set_ylabel("overhead (bits)")
    ax.set_title("Overhead grows only logarithmically")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_code_length_growth.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_binary_tree()
    fig_cylinder_sets()
    fig_code_length_growth()
    print("Figures written to", FIGDIR)
