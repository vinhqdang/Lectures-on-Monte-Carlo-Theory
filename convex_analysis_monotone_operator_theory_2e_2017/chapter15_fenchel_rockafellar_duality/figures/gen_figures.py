#!/usr/bin/env python3
"""
Generate figures for Chapter 15: Fenchel-Rockafellar Duality
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

# Set up figure parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2

# Create output directory
os.makedirs('.', exist_ok=True)


def fig_conjugate_function():
    """Visualize conjugate function and Fenchel duality"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original function: f(x) = x^2
    x = np.linspace(-3, 3, 200)
    f = x**2
    ax1.plot(x, f, 'b-', linewidth=2.5, label=r'$f(x) = x^2$')

    # Tangent lines to illustrate dual
    for u in [-1, 0, 1]:
        y_intersect = u**2 / 4
        ax1.plot(x, u*x - u**2/4, 'r--', alpha=0.5, linewidth=1)

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 4)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f(x)$', fontsize=12)
    ax1.set_title(r'Original Function: $f(x) = x^2$', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Conjugate function: f*(u) = u^2/4
    u = np.linspace(-4, 4, 200)
    f_conj = u**2 / 4
    ax2.plot(u, f_conj, 'g-', linewidth=2.5, label=r"$f^*(u) = u^2/4$")

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.5, 5)
    ax2.set_xlabel(r'$u$', fontsize=12)
    ax2.set_ylabel(r"$f^*(u)$", fontsize=12)
    ax2.set_title(r"Conjugate Function", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('fenchel_conjugate.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_duality_gap():
    """Illustrate primal-dual problems and duality gap"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw boxes for primal and dual problems
    primal_box = FancyBboxPatch((0.1, 0.6), 0.3, 0.25, boxstyle="round,pad=0.02",
                                edgecolor='blue', facecolor='lightblue', linewidth=2)
    dual_box = FancyBboxPatch((0.6, 0.6), 0.3, 0.25, boxstyle="round,pad=0.02",
                              edgecolor='red', facecolor='lightcoral', linewidth=2)

    ax.add_patch(primal_box)
    ax.add_patch(dual_box)

    # Text in boxes
    ax.text(0.25, 0.75, 'Primal Problem', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(0.25, 0.67, r'$\min_x [f(x) + g(x)]$', ha='center', va='center', fontsize=11)
    ax.text(0.75, 0.75, 'Dual Problem', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(0.75, 0.67, r'$\min_u [f^*(-u) + g^*(u)]$', ha='center', va='center', fontsize=11)

    # Arrow showing duality gap
    arrow = FancyArrowPatch((0.4, 0.72), (0.6, 0.72), arrowstyle='<->', mutation_scale=20,
                           color='black', linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.5, 0.78, r'Duality Gap: $\Delta(f,g)$', ha='center', fontsize=11, weight='bold')

    # Key theorem box
    theorem_box = FancyBboxPatch((0.05, 0.15), 0.9, 0.35, boxstyle="round,pad=0.02",
                                 edgecolor='green', facecolor='lightgreen', linewidth=2, alpha=0.3)
    ax.add_patch(theorem_box)

    ax.text(0.5, 0.45, r'Fenchel-Rockafellar Duality', ha='center', fontsize=12, weight='bold')
    ax.text(0.5, 0.36, r'If $0 \in \text{sri}(\text{dom} f - \text{dom} g)$, then:', ha='center', fontsize=11)
    ax.text(0.5, 0.28, r'$\min_x[f(x) + g(x)] = -\min_u[f^*(-L^*u) + g^*(u)]$', ha='center', fontsize=10)
    ax.text(0.5, 0.20, r'Zero duality gap: $\Delta(f,g,L) = 0$', ha='center', fontsize=11, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('duality_gap.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_convex_sets():
    """Illustrate convex sets and polar cones"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Convex set
    circle = patches.Circle((0.5, 0.5), 0.3, fill=True, alpha=0.3,
                            edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(circle)
    ax1.plot([0.2, 0.8], [0.2, 0.8], 'r-', linewidth=2, label='Chord')
    ax1.scatter([0.5, 0.5], [0.5, 0.5], c='blue', s=100, zorder=5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Convex Set: Chords stay inside', fontsize=12)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.grid(True, alpha=0.3)

    # Right: Cone structure
    angles = np.array([0, 30, 60]) * np.pi / 180
    for angle in angles:
        r = np.linspace(0, 1, 50)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        ax2.plot(x, y, 'b-', linewidth=1.5)

    # Fill cone
    theta_cone = np.linspace(0, np.pi/3, 50)
    r_cone = np.linspace(0, 1, 50)
    for r in np.linspace(0.2, 1, 5):
        x = r * np.cos(theta_cone)
        y = r * np.sin(theta_cone)
        ax2.fill(x, y, alpha=0.1, color='blue')

    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('Convex Cone: Rays stay inside', fontsize=12)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convex_sets.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_examples_1d():
    """Examples in 1D with numerical values"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example 1: f(x) = |x|, g(x) = 0
    x = np.linspace(-3, 3, 200)
    ax = axes[0, 0]
    ax.plot(x, np.abs(x), 'b-', linewidth=2.5, label=r'$f(x) = |x|$')
    ax.axhline(y=0, color='r', linewidth=2.5, linestyle='--', label=r'$g(x) = 0$')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x) + g(x)$')
    ax.set_title(r'Example: $f(x) = |x|, g(x) = 0$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Example 2: f(x) = x^2, g(x) = sqrt(1 + x^2)
    x = np.linspace(-2, 2, 200)
    ax = axes[0, 1]
    f = x**2
    g = np.sqrt(1 + x**2)
    fg = f + g
    ax.plot(x, f, 'b-', linewidth=1.5, label=r'$f(x) = x^2$')
    ax.plot(x, g, 'r-', linewidth=1.5, label=r'$g(x) = \sqrt{1+x^2}$')
    ax.plot(x, fg, 'g-', linewidth=2.5, label=r'$f(x) + g(x)$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 6)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Value')
    ax.set_title(r'Example: Sum of Functions', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Example 3: Conjugate pair visualization
    x = np.linspace(-3, 3, 200)
    ax = axes[1, 0]
    ax.plot(x, x**2, 'b-', linewidth=2.5, label=r'$f(x) = x^2$')
    ax.plot(x, np.abs(x) - 1, 'r-', linewidth=2.5, label=r"Comparison")
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 4)
    ax.set_xlabel(r'$x$ or $u$')
    ax.set_ylabel(r'Value')
    ax.set_title(r'Example: Function and Dual', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Example 4: Optimization problem
    x = np.linspace(0, 3, 100)
    ax = axes[1, 1]
    primal = 0.5 * x**2 + np.abs(x - 1)
    ax.plot(x, primal, 'b-', linewidth=2.5, label='Primal Cost')
    ax.fill_between(x, primal, alpha=0.2, color='blue')

    # Add annotations
    min_idx = np.argmin(primal)
    ax.scatter(x[min_idx], primal[min_idx], color='red', s=100, zorder=5, label='Optimal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'Cost')
    ax.set_title(r'Example: Optimization Problem', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('examples_1d.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_algorithm_flow():
    """Flow diagram for duality framework"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Define boxes for algorithm flow
    boxes = [
        {'xy': (0.1, 0.85), 'label': 'Start: Primal\nmin [f(x)+g(x)]', 'color': 'lightblue'},
        {'xy': (0.1, 0.65), 'label': 'Compute\nConjugates', 'color': 'lightyellow'},
        {'xy': (0.1, 0.45), 'label': 'Dual Problem\nmin [f*(-L*u)+g*(u)]', 'color': 'lightgreen'},
        {'xy': (0.1, 0.25), 'label': 'Check:\n0 in sri(...)', 'color': 'lightyellow'},
        {'xy': (0.55, 0.45), 'label': 'YES: Strong\nDuality', 'color': 'lightgreen'},
        {'xy': (0.55, 0.25), 'label': 'NO: Duality\nGap > 0', 'color': 'lightcoral'},
    ]

    for box in boxes:
        rect = FancyBboxPatch(box['xy'], 0.35, 0.12, boxstyle="round,pad=0.01",
                             edgecolor='black', facecolor=box['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + 0.175, box['xy'][1] + 0.06, box['label'],
               ha='center', va='center', fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')

    # Main flow
    ax.annotate('', xy=(0.275, 0.77), xytext=(0.275, 0.73), arrowprops=arrow_props)
    ax.annotate('', xy=(0.275, 0.57), xytext=(0.275, 0.53), arrowprops=arrow_props)
    ax.annotate('', xy=(0.275, 0.37), xytext=(0.275, 0.33), arrowprops=arrow_props)

    # Branch arrows
    ax.annotate('', xy=(0.45, 0.51), xytext=(0.4, 0.37), arrowprops=arrow_props)
    ax.annotate('', xy=(0.45, 0.31), xytext=(0.4, 0.31), arrowprops=arrow_props)

    # Add title
    ax.text(0.5, 0.98, 'Fenchel-Rockafellar Duality Framework', ha='center', fontsize=13, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('algorithm_flow.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_minimax_theorem():
    """Illustration of von Neumann's minimax theorem"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 2D example with linear operator
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Define a simple bilinear function: <Lx, y>
    Z = X * Y

    contour1 = ax1.contourf(X, Y, Z, levels=15, cmap='RdBu_r', alpha=0.7)
    contour2 = ax1.contour(X, Y, Z, levels=8, colors='black', alpha=0.3, linewidths=0.5)
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label(r'$\langle Lx, y \rangle$')

    ax1.set_xlabel(r'$x$', fontsize=11)
    ax1.set_ylabel(r'$y$', fontsize=11)
    ax1.set_title(r'Bilinear Form: $\langle Lx, y \rangle = xy$', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Right: Minimax property
    x_vals = np.linspace(-2, 2, 100)

    # For different values of y, plot the inner minimization
    for y_val in [-1.5, -0.75, 0, 0.75, 1.5]:
        z_vals = x_vals * y_val
        ax2.plot(x_vals, z_vals, linewidth=1.5, label=f'y={y_val:.2f}')

    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'$x$', fontsize=11)
    ax2.set_ylabel(r'$\langle Lx, y \rangle$', fontsize=11)
    ax2.set_title(r'Von Neumann: min(x) max(y) = max(y) min(x)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('minimax_theorem.pdf', dpi=150, bbox_inches='tight')
    plt.close()


def fig_attouchbrezis_illustration():
    """Illustrate the Attouch-Brézis theorem"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Domain illustration
    ax.fill_between([-2, 2], -1, 1, alpha=0.3, color='blue', label=r'$\text{dom} f$')
    ax.fill_between([-1, 3], -1, 1, alpha=0.3, color='red', label=r'$\text{dom} g$')
    ax.fill_between([-2, 3], -0.5, 0.5, alpha=0.2, color='green', label=r'$\text{dom} f - \text{dom} g$')

    ax.text(-1, -0.7, r'Overlap', ha='center', fontsize=11, weight='bold')
    ax.text(1, 0.8, r'Functions in $\Gamma_0(\mathcal{H})$', ha='center', fontsize=12)
    ax.text(0.5, 0, r'Contains origin', ha='center', fontsize=10, style='italic')

    ax.set_xlim(-3, 4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r'Space dimension', fontsize=11)
    ax.set_title(r'Attouch-Brézis Theorem: Domain Condition', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('attouchbrezis_illustration.pdf', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Generating figures for Chapter 15...")

    fig_conjugate_function()
    print("Generated: fenchel_conjugate.pdf")

    fig_duality_gap()
    print("Generated: duality_gap.pdf")

    fig_convex_sets()
    print("Generated: convex_sets.pdf")

    fig_examples_1d()
    print("Generated: examples_1d.pdf")

    fig_algorithm_flow()
    print("Generated: algorithm_flow.pdf")

    fig_minimax_theorem()
    print("Generated: minimax_theorem.pdf")

    fig_attouchbrezis_illustration()
    print("Generated: attouchbrezis_illustration.pdf")

    print("\nAll figures generated successfully!")
