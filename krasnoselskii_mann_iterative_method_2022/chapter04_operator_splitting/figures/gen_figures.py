"""
gen_figures.py  --  Chapter 4: Relations of the KM Iteration and Operator Splitting Methods
Generates all figures needed for chapter4_slides.tex
Saves PDFs (vector) to ./figures/  (this directory)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np
import os

FIGDIR = os.path.dirname(os.path.abspath(__file__))

def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = "#2166ac"
LBLUE  = "#a6cee3"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#e08214"
PURPLE = "#762a83"
GRAY   = "#666666"
DGRAY  = "#333333"


# =============================================================================
# Figure 1 -- Family tree: everything is a KM / averaged-operator iteration
# =============================================================================
def fig_family_tree():
    fig, ax = plt.subplots(figsize=(11.5, 7.7))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 7.9)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=9.5, textcolor='black'):
        b = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.4, edgecolor=DGRAY, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, linespacing=1.35)
        return (x + w/2, y, x + w/2, y + h)  # (bottom center, top center)

    def arrow(p0, p1, color=DGRAY):
        a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=14,
                             lw=1.3, color=color, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    # Root: KM iteration
    root_w, root_h = 5.6, 1.05
    root_x, root_y = (11.5 - root_w)/2, 6.0
    box(root_x, root_y, root_w, root_h,
        "Krasnosel'skiĭ--Mann iteration\n"
        r"$x_{n+1}=(1-\lambda_n)x_n+\lambda_n T x_n$" + "\n"
        r"$T$ averaged / nonexpansive, Fix$(T)\ne\emptyset$",
        "#fdd", fontsize=9.5)
    root_bottom = (root_x + root_w/2, root_y)

    # Second level: 3 branches
    lvl2_y = 4.35
    lvl2_h = 1.0
    specs = [
        (0.2, 3.1, "Gradient Descent\n(4.1)\n" r"$T=\mathrm{Id}-\frac{\gamma}{\lambda}\nabla f$", LBLUE),
        (3.6, 3.1, "Proximal Point\nAlgorithm (4.2)\n" r"$T=J_{\gamma A}$", "#cde6c7"),
        (7.4, 3.9, "Operator Splitting\nMethods (4.3)\n" "handle $A,B$ separately", "#ffe1b3"),
    ]
    tops = []
    for x, w, txt, color in specs:
        cx, cb, ct, ctop = box(x, lvl2_y, w, lvl2_h, txt, color, fontsize=9)
        tops.append((cx, ctop))
        arrow((cx, ctop), root_bottom)

    # Third level: the 4 splitting methods, children of "Operator Splitting"
    split_cx = tops[2][0]
    split_bottom_y = lvl2_y
    lvl3_y = 0.25
    lvl3_h = 1.55
    lvl3_specs = [
        (0.1, 2.55, "Forward-Backward /\nBackward-Forward\n(4.3.1)\n"
                     r"$T_{FB}=J_{\gamma A}(\mathrm{Id}-\gamma B)$", "#f6c9c1"),
        (2.85, 2.55, "Douglas--Rachford\n(4.3.2)\n"
                      r"$T_{DR}=\frac{1}{2}(R_{\gamma A_1}R_{\gamma A_2}+\mathrm{Id})$", "#d9c9ee"),
        (5.6, 2.55, "Davis--Yin\n(4.3.3)\n"
                     r"$T_{DY}=$ (4.13)" + "\n"
                     "generalizes FBS and DR", "#c2e0f2"),
        (8.35, 2.75, "Primal--Dual\nSplitting (4.3.4)\n"
                      r"FBS in a renormed" + "\n" r"product space $\mathcal{K}_{\mathbf{V}}$", "#f2e0a3"),
    ]
    for x, w, txt, color in lvl3_specs:
        cx, cb, ct, ctop = box(x, lvl3_y, w, lvl3_h, txt, color, fontsize=8.7)
        arrow((cx, ctop), (split_cx, split_bottom_y))

    ax.text(5.75, 7.65, "All roads lead to: iterate an averaged operator", ha='center',
            fontsize=13, fontweight='bold', color=DGRAY)

    savefig("fig_family_tree.pdf")


# =============================================================================
# Figure 2 -- Running example: FBS iterates converging to P_C(p) = (0.6, 0.8)
# =============================================================================
def fig_fbs_convergence():
    p = np.array([3.0, 4.0])
    target = np.array([0.6, 0.8])

    def P_C(x):
        n = np.linalg.norm(x)
        return x / n if n > 1 else x

    x0 = np.array([-2.0, 3.0])
    gamma = 0.5

    x = x0.copy()
    pts = [x.copy()]
    for _ in range(12):
        y = x - gamma * (x - p)
        x = P_C(y)
        pts.append(x.copy())
    pts = np.array(pts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- left panel: geometric picture in the plane ---
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.3)
    ax.fill(np.cos(theta), np.sin(theta), color=LBLUE, alpha=0.25)
    ax.plot(*p, marker='*', color=RED, ms=16, ls='none', zorder=5)
    ax.annotate('$p=(3,4)$', xy=p, xytext=(p[0]+0.15, p[1]+0.05), fontsize=10, color=RED)
    ax.plot(pts[:, 0], pts[:, 1], '-o', color=BLUE, ms=4.5, lw=1.3, zorder=4)
    ax.plot(*pts[0], marker='s', color=GREEN, ms=9, zorder=6)
    ax.annotate('$x_0$', xy=pts[0], xytext=(pts[0][0]-0.55, pts[0][1]+0.15), fontsize=10, color=GREEN)
    ax.plot(*target, marker='*', color=PURPLE, ms=14, zorder=6)
    ax.annotate(r'$P_C(p)=(0.6,0.8)$', xy=target, xytext=(target[0]+0.15, target[1]-0.55),
                fontsize=10, color=PURPLE)
    ax.set_xlim(-3.0, 4.2)
    ax.set_ylim(-1.0, 4.6)
    ax.set_aspect('equal')
    ax.axhline(0, color='#cccccc', lw=0.7, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.7, zorder=0)
    ax.set_title("Forward--backward splitting iterates\n"
                  r"$f(x)=\frac{1}{2}\|x-p\|^2$ (smooth), $g=\delta_C$ (nonsmooth)", fontsize=10.5)
    ax.set_xlabel("$x^{(1)}$"); ax.set_ylabel("$x^{(2)}$")

    # --- right panel: distance to solution vs iteration, log scale ---
    ax2 = axes[1]
    dist = np.linalg.norm(pts - target, axis=1)
    ax2.semilogy(range(len(pts)), dist, '-o', color=BLUE, ms=4.5)
    ax2.set_xlabel("iteration $n$")
    ax2.set_ylabel(r"$\|x_n - P_C(p)\|$ (log scale)")
    ax2.set_title("Linear convergence of FBS\nto the running-example solution", fontsize=10.5)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    savefig("fig_fbs_convergence.pdf")


# =============================================================================
# Figure 3 -- Gradient descent vs. proximal point vs. FBS, same target
# =============================================================================
def fig_three_methods_compare():
    p = np.array([3.0, 4.0])
    target = np.array([0.6, 0.8])

    def P_C(x):
        n = np.linalg.norm(x)
        return x / n if n > 1 else x

    x0 = np.array([-2.0, 3.0])

    # Gradient descent on f alone (unconstrained), converges to p, NOT to target.
    x = x0.copy()
    gd_pts = [x.copy()]
    gamma = 0.5
    for _ in range(10):
        x = x - gamma * (x - p)
        gd_pts.append(x.copy())
    gd_pts = np.array(gd_pts)

    # Proximal point algorithm for A = subdifferential of indicator of C:
    # J_{gamma A} = P_C independent of gamma. x_{n+1} = P_C(x_n).
    x = x0.copy()
    ppa_pts = [x.copy()]
    for _ in range(4):
        x = P_C(x)
        ppa_pts.append(x.copy())
    ppa_pts = np.array(ppa_pts)

    # FBS (same as before)
    x = x0.copy()
    fbs_pts = [x.copy()]
    gamma = 0.5
    for _ in range(12):
        y = x - gamma * (x - p)
        x = P_C(y)
        fbs_pts.append(x.copy())
    fbs_pts = np.array(fbs_pts)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.3)
    ax.fill(np.cos(theta), np.sin(theta), color=LBLUE, alpha=0.2)

    ax.plot(gd_pts[:, 0], gd_pts[:, 1], '-o', color=ORANGE, ms=4, lw=1.2,
            label=r"Gradient descent on $f$ alone $\to p$ (ignores $C$)")
    ax.plot(ppa_pts[:, 0], ppa_pts[:, 1], '-s', color=GREEN, ms=7, lw=1.4,
            label=r"Proximal point ($A=\partial\delta_C$) $\to P_C(x_0)$ in 1 step")
    ax.plot(fbs_pts[:, 0], fbs_pts[:, 1], '-^', color=BLUE, ms=4.5, lw=1.2,
            label=r"Forward--backward splitting $\to P_C(p)$")

    ax.plot(*p, marker='*', color=RED, ms=16, ls='none', zorder=5)
    ax.annotate('$p=(3,4)$', xy=p, xytext=(p[0]+0.1, p[1]+0.05), fontsize=10, color=RED)
    ax.plot(*x0, marker='D', color='black', ms=6, zorder=6)
    ax.annotate('$x_0$', xy=x0, xytext=(x0[0]-0.5, x0[1]+0.1), fontsize=10)
    ax.plot(*target, marker='*', color=PURPLE, ms=14, zorder=6)

    ax.set_xlim(-3.0, 4.2)
    ax.set_ylim(-1.2, 4.6)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title("Three KM instances on the same running example", fontsize=11)
    ax.set_xlabel("$x^{(1)}$"); ax.set_ylabel("$x^{(2)}$")

    plt.tight_layout()
    savefig("fig_three_methods_compare.pdf")


if __name__ == "__main__":
    fig_family_tree()
    fig_fbs_convergence()
    fig_three_methods_compare()
    print("All figures generated.")
