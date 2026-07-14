#!/usr/bin/env python3
"""
Generate figures for Pathak Chapter 1a: Discrete & Topological Structures
Includes: lattice diagrams, norm visualizations, metric space examples
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configure for LaTeX output (disable usetex to avoid LaTeX dependencies)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def fig_partial_order():
    """Visualize a partial order (poset) as a Hasse diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Draw a Hasse diagram for a simple poset
    # Elements: {1, 2, 3, 4, 5, 6}
    # Relations: 1 < 2,3; 2 < 4; 3 < 4,5; 4 < 6; 5 < 6

    positions = {
        1: (2, 0.5),
        2: (1, 1.5),
        3: (3, 1.5),
        4: (2, 2.5),
        5: (3.5, 2.5),
        6: (2.5, 3.5)
    }

    edges = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6)]

    # Draw edges
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, zorder=1)

    # Draw nodes
    for node, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.25, color='lightblue', ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'Hasse Diagram of a Partially Ordered Set', fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig('figures/fig_poset.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_poset.pdf")

def fig_lattice():
    """Visualize a lattice structure"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Boolean lattice B3 (subsets of {1,2,3})
    positions = {
        r'$\emptyset$': (3.5, 0.5),
        r'$\{1\}$': (1.5, 1.5),
        r'$\{2\}$': (3.5, 1.5),
        r'$\{3\}$': (5.5, 1.5),
        r'$\{1,2\}$': (1.5, 2.5),
        r'$\{1,3\}$': (3.5, 2.5),
        r'$\{2,3\}$': (5.5, 2.5),
        r'$\{1,2,3\}$': (3.5, 3.5),
    }

    edges = [
        (r'$\emptyset$', r'$\{1\}$'),
        (r'$\emptyset$', r'$\{2\}$'),
        (r'$\emptyset$', r'$\{3\}$'),
        (r'$\{1\}$', r'$\{1,2\}$'),
        (r'$\{1\}$', r'$\{1,3\}$'),
        (r'$\{2\}$', r'$\{1,2\}$'),
        (r'$\{2\}$', r'$\{2,3\}$'),
        (r'$\{3\}$', r'$\{1,3\}$'),
        (r'$\{3\}$', r'$\{2,3\}$'),
        (r'$\{1,2\}$', r'$\{1,2,3\}$'),
        (r'$\{1,3\}$', r'$\{1,2,3\}$'),
        (r'$\{2,3\}$', r'$\{1,2,3\}$'),
    ]

    # Draw edges
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, zorder=1)

    # Draw nodes
    for label, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.3, color='lightyellow', ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=3)

    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'Boolean Lattice $B_3$ (Power Set of $\{1,2,3\}$)', fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig('figures/fig_lattice.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_lattice.pdf")

def fig_unit_spheres():
    """Visualize unit spheres for different p-norms in R^2"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    norms_info = [
        (1, r'$\ell^1$ norm: $\|x\|_1 = |x_1| + |x_2|$', axes[0, 0]),
        (2, r'$\ell^2$ norm: $\|x\|_2 = \sqrt{|x_1|^2 + |x_2|^2}$', axes[0, 1]),
        (4, r'$\ell^4$ norm: $\|x\|_4 = (|x_1|^4 + |x_2|^4)^{1/4}$', axes[1, 0]),
        (np.inf, r'$\ell^\infty$ norm: $\|x\|_\infty = \max(|x_1|, |x_2|)$', axes[1, 1]),
    ]

    for p, title, ax in norms_info:
        # Generate points on the unit sphere
        theta = np.linspace(0, 2*np.pi, 1000)

        if p == 1:
            # L1 ball: diamond shape
            x = np.array([1, 0, -1, 0, 1])
            y = np.array([0, 1, 0, -1, 0])
        elif p == 2:
            # L2 ball: circle
            x = np.cos(theta)
            y = np.sin(theta)
        elif p == 4:
            # L4 ball
            x = np.cos(theta)
            y = np.sin(theta)
            # Approximate by power
            radius = (np.abs(x)**4 + np.abs(y)**4)**(1/4)
            x = x / radius
            y = y / radius
        elif p == np.inf:
            # L-infinity ball: square
            x = np.array([-1, 1, 1, -1, -1])
            y = np.array([-1, -1, 1, 1, -1])

        ax.plot(x, y, 'b-', linewidth=2.5)
        ax.fill(x, y, alpha=0.3, color='lightblue')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$', fontsize=11)
        ax.set_ylabel(r'$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig('figures/fig_unit_spheres.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_unit_spheres.pdf")

def fig_continuity():
    """Visualize epsilon-delta definition of continuity"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot a continuous function
    x = np.linspace(-1, 3, 1000)
    y = 0.5 * x**2 - x + 0.5

    ax.plot(x, y, 'b-', linewidth=2.5, label=r'$f(x) = 0.5x^2 - x + 0.5$')

    # Show a point and epsilon-delta
    x0, y0 = 1.5, 0.625
    epsilon = 0.4
    delta = 0.3

    ax.plot(x0, y0, 'ro', markersize=10, label=r'$(x_0, f(x_0))$', zorder=5)

    # Draw epsilon band
    ax.axhline(y0 + epsilon, color='r', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y0 - epsilon, color='r', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.fill_between([-1, 3], y0 - epsilon, y0 + epsilon, alpha=0.15, color='red', label=r'$\epsilon$-band')

    # Draw delta band
    ax.axvline(x0 + delta, color='g', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x0 - delta, color='g', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.fill_betweenx([-0.5, 3], x0 - delta, x0 + delta, alpha=0.15, color='green', label=r'$\delta$-band')

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$f(x)$', fontsize=12)
    ax.set_title(r'Continuity: For $\epsilon > 0$, there exists $\delta > 0$ such that $|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \epsilon$',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_continuity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_continuity.pdf")

def fig_semicontinuity():
    """Visualize upper and lower semicontinuity"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Upper semicontinuous
    x = np.linspace(0, 2*np.pi, 500)
    y_usc = np.sin(x)
    y_usc[np.where(x > np.pi)[0]] = np.sin(x[np.where(x > np.pi)[0]]) - 0.5

    ax = axes[0]
    ax.plot(x, y_usc, 'b-', linewidth=2.5, label='Upper semicontinuous')
    ax.plot(np.pi, np.sin(np.pi), 'ro', markersize=8, zorder=5)
    ax.plot(np.pi, np.sin(np.pi) - 0.5, 'o', color='white', markeredgecolor='b',
            markeredgewidth=2, markersize=8, zorder=5)
    ax.axvline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x$', fontsize=11)
    ax.set_ylabel(r'$\varphi(x)$', fontsize=11)
    ax.set_title(r'Upper Semicontinuous at $x_0 = \pi$', fontsize=11)
    ax.legend(fontsize=10)

    # Lower semicontinuous
    y_lsc = np.sin(x)
    y_lsc[np.where(x > np.pi)[0]] = np.sin(x[np.where(x > np.pi)[0]]) + 0.5

    ax = axes[1]
    ax.plot(x, y_lsc, 'g-', linewidth=2.5, label='Lower semicontinuous')
    ax.plot(np.pi, np.sin(np.pi) + 0.5, 'ro', markersize=8, zorder=5)
    ax.plot(np.pi, np.sin(np.pi), 'o', color='white', markeredgecolor='g',
            markeredgewidth=2, markersize=8, zorder=5)
    ax.axvline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x$', fontsize=11)
    ax.set_ylabel(r'$\varphi(x)$', fontsize=11)
    ax.set_title(r'Lower Semicontinuous at $x_0 = \pi$', fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_semicontinuity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_semicontinuity.pdf")

def fig_separation_axioms():
    """Visualize separation axioms T0, T1, T2"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # T0 space
    ax = axes[0]
    ax.text(0.5, 0.8, r'$T_0$ Space (Kolmogorov)', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.6, r'For distinct $x, y \in X$:', ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.45, r'$\exists$ open set $U$ containing', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.35, r'one but not the other', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.15, r'(Can distinguish points)', ha='center', fontsize=9, style='italic',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.axis('off')

    # T1 space
    ax = axes[1]
    ax.text(0.5, 0.8, r'$T_1$ Space', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.6, r'For distinct $x, y \in X$:', ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.45, r'$\exists$ open set $U_x$ with', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.35, r'$x \in U_x, y \notin U_x$ and', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.25, r'$\exists$ open set $U_y$ with', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.15, r'$y \in U_y, x \notin U_y$', ha='center', fontsize=9, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.axis('off')

    # T2 space (Hausdorff)
    ax = axes[2]
    ax.text(0.5, 0.8, r'$T_2$ Space (Hausdorff)', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.6, r'For distinct $x, y \in X$:', ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.45, r'$\exists$ disjoint open sets', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.35, r'$U \ni x$ and $V \ni y$ with', ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.15, r'$U \cap V = \emptyset$', ha='center', fontsize=9, style='italic',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/fig_separation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_separation.pdf")

def fig_net_convergence():
    """Visualize net (generalized sequence) convergence"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw topological space as a rectangle
    rect = FancyBboxPatch((0.5, 0.5), 5, 3, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 4, r'Topological Space $X$', fontsize=12, ha='center', fontweight='bold')

    # Draw limit point x
    x_pos = (2.5, 2)
    ax.plot(x_pos[0], x_pos[1], 'ro', markersize=12, zorder=5, label='Limit point $x$')
    ax.text(2.5, 1.4, r'$x$', fontsize=11, ha='center', fontweight='bold')

    # Draw neighborhood U of x
    circle = plt.Circle(x_pos, 0.6, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.text(3.3, 2.3, r'Neighborhood $U$ of $x$', fontsize=10, color='green', fontweight='bold')

    # Draw some net points
    net_points = [
        (1.8, 2.7, r'$x_{\alpha_1}$'),
        (2.2, 1.8, r'$x_{\alpha_2}$'),
        (2.6, 2.5, r'$x_{\alpha_3}$'),
        (3.2, 1.9, r'$x_{\alpha_4}$'),
        (1.5, 2.1, r'$x_{\alpha_5}$'),
    ]

    for px, py, label in net_points:
        ax.plot(px, py, 'bs', markersize=8, zorder=4)
        ax.text(px, py-0.25, label, fontsize=9, ha='center')

    # Draw arrows indicating net tail
    ax.annotate('', xy=(2.5, 2), xytext=(1.8, 2.7),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', alpha=0.5))
    ax.annotate('', xy=(2.5, 2), xytext=(3.2, 1.9),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', alpha=0.5))

    ax.text(3, 0.8, r'A net $\{x_\alpha\}$ converges to $x$ if for every neighborhood $U$ of $x$,',
            fontsize=10, ha='center')
    ax.text(3, 0.3, r'there exists $\alpha_0$ such that $x_\alpha \in U$ for all $\alpha \geq \alpha_0$',
            fontsize=10, ha='center')

    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/fig_net.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_net.pdf")

def fig_compactness():
    """Visualize compactness concept"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Non-compact
    ax = axes[0]
    ax.text(0.5, 0.95, 'Non-Compact Space', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Draw horizontal line
    ax.plot([0.1, 0.9], [0.5, 0.5], 'b-', linewidth=3, transform=ax.transAxes)
    ax.text(0.05, 0.5, r'$\mathbb{R}$', fontsize=11, va='center', transform=ax.transAxes, fontweight='bold')

    # Draw open intervals (open covering)
    intervals = [(0.15, 0.25), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.85, 0.95)]
    colors = ['red', 'orange', 'yellow', 'green', 'cyan']
    for (x1, x2), color in zip(intervals, colors):
        ax.add_patch(patches.Rectangle((x1, 0.45), x2-x1, 0.1, transform=ax.transAxes,
                                       facecolor=color, alpha=0.6, edgecolor='black', linewidth=1))

    ax.text(0.5, 0.25, 'Open covering: No finite subcover!', ha='center', fontsize=10,
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')

    # Compact
    ax = axes[1]
    ax.text(0.5, 0.95, 'Compact Space', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Draw closed interval
    ax.plot([0.2, 0.8], [0.5, 0.5], 'b-', linewidth=3, transform=ax.transAxes)
    ax.plot(0.2, 0.5, 'bo', markersize=8, transform=ax.transAxes)
    ax.plot(0.8, 0.5, 'bo', markersize=8, transform=ax.transAxes)
    ax.text(0.05, 0.5, r'$[a,b]$', fontsize=11, va='center', transform=ax.transAxes, fontweight='bold')

    # Draw covering intervals
    intervals2 = [(0.15, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 0.85)]
    colors2 = ['red', 'orange', 'green', 'cyan']
    for (x1, x2), color in zip(intervals2, colors2):
        ax.add_patch(patches.Rectangle((x1, 0.45), x2-x1, 0.1, transform=ax.transAxes,
                                       facecolor=color, alpha=0.6, edgecolor='black', linewidth=1))

    ax.text(0.5, 0.25, 'Finite subcover exists!', ha='center', fontsize=10,
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/fig_compactness.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig_compactness.pdf")

if __name__ == '__main__':
    print("Generating figures for Chapter 1a...")
    fig_partial_order()
    fig_lattice()
    fig_unit_spheres()
    fig_continuity()
    fig_semicontinuity()
    fig_separation_axioms()
    fig_net_convergence()
    fig_compactness()
    print("\nAll figures generated successfully!")
