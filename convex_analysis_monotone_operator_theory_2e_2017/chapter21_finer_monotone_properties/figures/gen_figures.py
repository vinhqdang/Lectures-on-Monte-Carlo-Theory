#!/usr/bin/env python3
"""
gen_figures.py -- Figures for Chapter 21 (Finer Properties of Monotone Operators)
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd ed.

Generates vector (PDF) figures used by chapter21_slides.tex:

  fig_minty.pdf       -- Minty's theorem for the 1D operator A(x) = x^3
  fig_domain_range.pdf -- Domain vs. range: A = d|.| (subdifferential of abs value)
  fig_local_bdd.pdf   -- Local boundedness fails at boundary (Rockafellar-Vesely)
  fig_monotone_property.pdf -- Basic monotonicity illustration
  fig_example_operator.pdf -- Example: subdifferential as monotone operator
  fig_summary.pdf     -- Summary diagram of Chapter 21 concepts

All figures are plain matplotlib, saved as PDF (vector), no external data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------
# Figure 0: Basic monotone property visualization
# ---------------------------------------------------------------------------
def fig_monotone_property():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw coordinate system
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Illustrate monotone property: <x-y, u-v> >= 0
    x1, y1 = 1.0, 1.5
    x2, y2 = 2.5, 3.0

    # Plot two points in the graph
    ax.scatter([x1, x2], [y1, y2], s=150, c=['red', 'blue'], zorder=5)
    ax.text(x1-0.3, y1-0.4, r'$(x,u)$', fontsize=12, color='red', fontweight='bold')
    ax.text(x2+0.1, y2+0.2, r'$(y,v)$', fontsize=12, color='blue', fontweight='bold')

    # Draw arrows
    arrow1 = FancyArrowPatch((0.5, 0.5), (x2, 0.5), arrowstyle='<->',
                            mutation_scale=20, color='darkgreen', linewidth=2.5, alpha=0.7)
    ax.add_patch(arrow1)
    ax.text((x1+x2)/2, 0.1, r'$x - y$ (difference)', fontsize=11, ha='center',
            color='darkgreen', fontweight='bold')

    arrow2 = FancyArrowPatch((0.2, y1), (0.2, y2), arrowstyle='<->',
                            mutation_scale=20, color='darkviolet', linewidth=2.5, alpha=0.7)
    ax.add_patch(arrow2)
    ax.text(-0.5, (y1+y2)/2, r'$u - v$', fontsize=11, ha='center', color='darkviolet',
            fontweight='bold', rotation=90)

    # Property box
    ax.text(1.5, 4.5, r'Monotonicity: $\langle x-y, u-v \rangle \geq 0$',
           fontsize=13, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='darkred', linewidth=2))

    ax.set_xlim(-1, 4)
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel(r'Domain $H$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Range', fontsize=12, fontweight='bold')
    ax.set_title(r'Monotone Operator: Core Property', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_monotone_property.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Minty's theorem for A(x) = x^3
# ---------------------------------------------------------------------------
def fig_minty():
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(-1.8, 1.8, 400)
    Tx = x + x ** 3  # T = Id + A

    ax.plot(x, Tx, color="#1f77b4", lw=2.8, label=r"$T(x) = x + x^3 = (\mathrm{Id}+A)(x)$")
    ax.plot(x, x, color="#999999", lw=1.5, ls="--", alpha=0.6, label=r"$y = x$ (identity)")

    # Mark sample points
    sample_xs = [-1.5, -1.0, 0.0, 0.5, 1.0, 1.2]
    for xs_ in sample_xs:
        ys_ = xs_ + xs_ ** 3
        ax.plot([xs_], [ys_], "o", color="#d62728", zorder=5, markersize=6)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$x$", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"$T(x)$", fontsize=12, fontweight='bold')
    ax.set_title(r"Minty's Theorem: $T=\mathrm{Id}+A$ is Surjective when $A$ is Maximally Monotone" "\n"
                 r"Example: $A(x)=x^3$ on $\mathbb{R}$", fontsize=12, fontweight='bold')
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_minty.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Domain and range of A = d|.|
# ---------------------------------------------------------------------------
def fig_domain_range():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Graph of A(x) = sign(x) for x != 0, A(0) = [-1, 1]
    x_pos = np.linspace(0.02, 2.5, 200)
    x_neg = np.linspace(-2.5, -0.02, 200)

    ax.plot(x_pos, np.ones_like(x_pos), color="#1f77b4", lw=2.8)
    ax.plot(x_neg, -np.ones_like(x_neg), color="#1f77b4", lw=2.8)
    ax.plot([0, 0], [-1, 1], color="#1f77b4", lw=2.8,
            label=r"$\mathrm{gra}\, A,\ A = \partial|\cdot|$")
    ax.plot([0], [1], "o", mfc="white", mec="#1f77b4", zorder=5, markersize=7)
    ax.plot([0], [-1], "o", mfc="white", mec="#1f77b4", zorder=5, markersize=7)
    ax.plot([0], [1], ".", color="#1f77b4", zorder=6, ms=5)
    ax.plot([0], [-1], ".", color="#1f77b4", zorder=6, ms=5)

    # Shade dom A and ran A
    ax.axhspan(-1, 1, xmin=0, xmax=1, color="#ff7f0e", alpha=0.15, label=r'$\mathrm{ran}\,A$')
    ax.annotate(r"$\mathrm{ran}\,A=[-1,1]$ (bounded)", xy=(1.8, 0.0),
                fontsize=11, color="#d62728", ha="left", va="center", fontweight='bold')
    ax.annotate(r"$\mathrm{dom}\,A=\mathbb{R}$ (unbounded)", xy=(0.0, -1.65),
                fontsize=11, color="#2ca02c", ha="center", fontweight='bold')

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$x$", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"$u \in A(x)$", fontsize=12, fontweight='bold')
    ax.set_title(r"Domain vs.\ Range: $A=\partial|\cdot|$ on $\mathbb{R}$" "\n"
                 r"Unbounded domain, bounded range (Section 21.3)", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_domain_range.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Local boundedness fails at boundary
# ---------------------------------------------------------------------------
def fig_local_bdd():
    fig, ax = plt.subplots(figsize=(8, 6))

    # A(x) = x / (1 - x^2) on (-1, 1)
    x = np.linspace(-0.985, 0.985, 600)
    y = x / (1 - x ** 2)
    y_clip = np.clip(y, -15, 15)

    ax.plot(x, y_clip, color="#1f77b4", lw=2.8,
            label=r"$A(x)=\dfrac{x}{1-x^2}$, $\mathrm{dom}\,A=(-1,1)$")
    ax.axvline(-1, color="#d62728", lw=2, ls="--", alpha=0.7, label='Boundary of dom $A$')
    ax.axvline(1, color="#d62728", lw=2, ls="--", alpha=0.7)
    ax.annotate(r"$|A(x)|\to\infty$", xy=(-0.95, -12),
                fontsize=10, color="#d62728", ha="left", fontweight='bold')
    ax.annotate(r"$|A(x)|\to\infty$", xy=(0.95, 12),
                fontsize=10, color="#d62728", ha="right", fontweight='bold')

    # Interior point
    x0 = 0.3
    y0 = x0 / (1 - x0 ** 2)
    ax.plot([x0], [y0], "o", color="#2ca02c", zorder=5, markersize=10)
    ax.annotate("Interior: locally bounded", xy=(x0, y0),
                textcoords="offset points", xytext=(-100, 35), fontsize=10,
                color="#2ca02c", fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-16, 16)
    ax.set_xlabel(r"$x$", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"$A(x)$", fontsize=12, fontweight='bold')
    ax.set_title(r"Rockafellar–Veselý Theorem: Local Boundedness at $\mathrm{bdry}(\mathrm{dom}\,A)$" "\n"
                 r"(Theorem 21.18, Section 21.4)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper center", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_local_bdd.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Example operator - subdifferential of ||x||^2/2
# ---------------------------------------------------------------------------
def fig_example_operator():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: The convex function
    ax = axes[0]
    x = np.linspace(-3, 3, 100)
    f_vals = 0.5 * x**2
    ax.plot(x, f_vals, 'b-', linewidth=2.8, label=r'$f(x) = \frac{1}{2}\|x\|^2$')

    x_pts = [-2, -1, 0, 1, 2]
    f_pts = [0.5 * xi**2 for xi in x_pts]
    ax.scatter(x_pts, f_pts, s=120, c='red', zorder=5)

    for xi, fi in zip(x_pts, f_pts):
        if xi != 0:
            slope = xi
            x_range = [xi - 0.7, xi + 0.7]
            y_range = [fi - 0.7*slope, fi + 0.7*slope]
            ax.plot(x_range, y_range, 'r--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel(r'$x \in \mathbb{R}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$f(x)$', fontsize=12, fontweight='bold')
    ax.set_title(r'Convex Function: $f(x) = \frac{1}{2}\|x\|^2$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper center')
    ax.set_ylim(-0.5, 5)

    # Right: Subdifferential as monotone operator
    ax = axes[1]
    x = np.linspace(-3, 3, 100)
    subdiff = x
    ax.plot(x, subdiff, 'g-', linewidth=2.8, label=r'$\partial f(x) = x$')

    x1, x2 = -1.5, 1.5
    u1, u2 = x1, x2
    ax.scatter([x1, x2], [u1, u2], s=120, c=['purple', 'orange'], zorder=5)
    ax.text(x1-0.4, u1-0.4, r'$(x_1, u_1)$', fontsize=11, color='purple', fontweight='bold')
    ax.text(x2+0.2, u2+0.3, r'$(x_2, u_2)$', fontsize=11, color='orange', fontweight='bold')

    # Monotonicity
    arrow = FancyArrowPatch((x1-0.1, u1+0.1), (x2+0.1, u2-0.1),
                           arrowstyle='<->', color='red', lw=2, alpha=0.7)
    ax.add_patch(arrow)
    ax.text((x1+x2)/2, (u1+u2)/2+0.5, r'$\langle x_2-x_1, u_2-u_1 \rangle > 0$',
           fontsize=11, color='red', fontweight='bold', ha='center')

    ax.set_xlabel(r'$x$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$u = \partial f(x)$', fontsize=12, fontweight='bold')
    ax.set_title(r'Subdifferential $\partial f$ as Monotone Operator', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.tight_layout()
    fig.savefig("fig_example_operator.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Summary diagram
# ---------------------------------------------------------------------------
def fig_summary():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, 'Chapter 21: Finer Monotone Properties - Summary',
           fontsize=14, ha='center', fontweight='bold')

    # Central concept
    central = Rectangle((3, 7.2), 4, 1.2, linewidth=3, edgecolor='darkblue',
                        facecolor='lightblue', alpha=0.6)
    ax.add_patch(central)
    ax.text(5, 7.8, r'$A: H \to 2^H$ Maximally Monotone', fontsize=12,
           ha='center', fontweight='bold')

    # Five main sections
    sections = [
        {'name': "21.1: Minty's Theorem", 'x': 1, 'y': 5, 'result': r'ran$(Id+A)=H$', 'color': '#FFE5B4'},
        {'name': '21.2: Debrunner-Flor', 'x': 3, 'y': 5, 'result': 'Separation property', 'color': '#FFE5CC'},
        {'name': '21.3: Domain & Range', 'x': 5, 'y': 5, 'result': r'dom $A$, ran $A$', 'color': '#E5F3FF'},
        {'name': '21.4: Local Bdedness', 'x': 7, 'y': 5, 'result': r'Thm 21.18, Cor 21.23-25', 'color': '#FFE5E5'},
        {'name': '21.5: Kenderov', 'x': 9, 'y': 5, 'result': 'Generic $G_\delta$ differentiability', 'color': '#F0E5FF'},
    ]

    for sec in sections:
        box = Rectangle((sec['x']-0.9, sec['y']-0.5), 1.8, 1.5,
                        linewidth=2, edgecolor='black', facecolor=sec['color'], alpha=0.7)
        ax.add_patch(box)
        ax.text(sec['x'], sec['y']+0.5, sec['name'], fontsize=9, ha='center', fontweight='bold')
        ax.text(sec['x'], sec['y']-0.1, sec['result'], fontsize=8, ha='center', style='italic')

        # Draw arrows from central box
        arrow = FancyArrowPatch((5, 7.2), (sec['x'], sec['y']+0.75),
                              arrowstyle='->', mutation_scale=15, color='gray', lw=1.5, alpha=0.6)
        ax.add_patch(arrow)

    # Key theorems box
    ax.text(5, 3.5, 'Key Results', fontsize=11, ha='center', fontweight='bold')
    theorems_text = (
        r'• Minty: $A$ maximal monotone $\Rightarrow$ ran$(Id+A)=H$' + '\n' +
        r'• Debrunner-Flor: Separation property for monotone sets' + '\n' +
        r'• Rockafellar-Vesely: Local boundedness iff $x \notin$ bdry(dom $A$)' + '\n' +
        r'• Kenderov: Generic $G_\delta$ single-valuedness & continuity'
    )
    ax.text(5, 2, theorems_text, fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig("fig_summary.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_monotone_property()
    fig_minty()
    fig_domain_range()
    fig_local_bdd()
    fig_example_operator()
    fig_summary()
    print("Generated: fig_monotone_property.pdf, fig_minty.pdf, fig_domain_range.pdf,")
    print("           fig_local_bdd.pdf, fig_example_operator.pdf, fig_summary.pdf")
