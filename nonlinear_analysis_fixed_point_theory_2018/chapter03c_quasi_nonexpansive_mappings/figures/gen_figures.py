#!/usr/bin/env python3
"""
Generate figures for Pathak Chapter 5.3: Fixed Point Theorems of Brouwer and Schauder
Figures illustrate key concepts related to nonexpansive and quasi-nonexpansive mappings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set matplotlib parameters for publication-quality figures
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3


def fig_nonexpansive_mapping():
    """Visualize nonexpansive mapping property: ||Tx - Ty|| <= ||x - y||"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: domain space
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('Domain: K', fontsize=12, fontweight='bold')

    # Draw a circle (compact convex set)
    circle_k = Circle((1, 1), 0.8, fill=False, edgecolor='blue', linewidth=2)
    ax1.add_patch(circle_k)

    # Two points in K
    x = np.array([0.5, 1.3])
    y = np.array([1.5, 0.8])
    ax1.plot(*x, 'ro', markersize=8, label='x')
    ax1.plot(*y, 'go', markersize=8, label='y')

    # Draw line connecting x and y
    ax1.plot([x[0], y[0]], [x[1], y[1]], 'k--', linewidth=1.5, alpha=0.7)

    # Distance annotation
    mid_point = (x + y) / 2
    ax1.text(mid_point[0] + 0.15, mid_point[1], r'$\|x-y\|$', fontsize=11, fontweight='bold')

    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.text(0.5, 2.3, r'Nonexpansive: $\|Tx - Ty\| \leq \|x - y\|$',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right panel: range space
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_title('Range: T(K)', fontsize=12, fontweight='bold')

    # Draw the range set (smaller)
    circle_tk = Circle((1.2, 1.1), 0.5, fill=False, edgecolor='green', linewidth=2)
    ax2.add_patch(circle_tk)

    # Images of x and y under T
    Tx = np.array([0.9, 1.4])
    Ty = np.array([1.3, 0.9])
    ax2.plot(*Tx, 'ro', markersize=8, label='Tx')
    ax2.plot(*Ty, 'go', markersize=8, label='Ty')

    # Draw line connecting Tx and Ty
    ax2.plot([Tx[0], Ty[0]], [Tx[1], Ty[1]], 'k--', linewidth=1.5, alpha=0.7)

    # Distance annotation
    mid_point_t = (Tx + Ty) / 2
    ax2.text(mid_point_t[0] + 0.15, mid_point_t[1], r'$\|Tx-Ty\|$', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Tx', fontsize=11)
    ax2.set_ylabel('Ty', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.text(0.3, 2.3, r'Contraction: $\|Tx-Ty\| < \|x-y\|$',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/fig_nonexpansive_mapping.pdf', bbox_inches='tight')
    print("Saved: fig_nonexpansive_mapping.pdf")
    plt.close()


def fig_brouwer_fixed_point():
    """Illustrate Brouwer's Fixed Point Theorem concept"""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title("Brouwer's Fixed Point Theorem", fontsize=14, fontweight='bold')

    # Draw closed convex set K (ball)
    circle_k = Circle((1, 1), 1, fill=True, facecolor='lightblue',
                      edgecolor='darkblue', linewidth=2.5, alpha=0.6)
    ax.add_patch(circle_k)
    ax.text(1, 1, 'K', fontsize=18, fontweight='bold',
            ha='center', va='center', color='darkblue')

    # Draw some sample points and their images under continuous map
    theta = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    points_x = 1 + 0.9 * np.cos(theta)
    points_y = 1 + 0.9 * np.sin(theta)

    # Fixed point
    fixed_pt = np.array([1.0, 1.0])
    ax.plot(*fixed_pt, 'r*', markersize=25, label=r'Fixed point: $x^* = Tx^*$', zorder=5)

    # Draw arrows showing continuous deformation (not actual T)
    for i in range(len(theta)):
        # Create a continuous deformation towards center
        scale = 0.1 + 0.3 * np.sin(theta[i])
        Tx = 1 + 0.6 * np.cos(theta[i]) * (1 - scale)
        Ty = 1 + 0.6 * np.sin(theta[i]) * (1 - scale)

        ax.arrow(points_x[i], points_y[i], Tx - points_x[i], Ty - points_y[i],
                head_width=0.08, head_length=0.06, fc='red', ec='red', alpha=0.6, linewidth=1.5)
        ax.plot(points_x[i], points_y[i], 'ko', markersize=6)

    # Add theorem statement
    theorem_text = r"Every continuous map $f: K \to K$ where K is" + "\n" + \
                   r"compact and convex in $\mathbb{R}^n$ has a fixed point."
    ax.text(1, -0.3, theorem_text, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    plt.tight_layout()
    plt.savefig('figures/fig_brouwer_fixed_point.pdf', bbox_inches='tight')
    print("Saved: fig_brouwer_fixed_point.pdf")
    plt.close()


def fig_schauder_fixed_point():
    """Illustrate Schauder's Fixed Point Theorem"""
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.set_title("Schauder's Fixed Point Theorem", fontsize=14, fontweight='bold')

    # Draw compact convex set K (irregular shape)
    # Use a polygon to represent arbitrary compact convex set
    vertices = np.array([[0.5, 0.5], [2.5, 0.3], [2.8, 2.2], [0.8, 2.5], [0.3, 1.5]])
    polygon = Polygon(vertices, fill=True, facecolor='lightcyan',
                     edgecolor='darkblue', linewidth=2.5, alpha=0.6)
    ax.add_patch(polygon)
    ax.text(1.5, 1.3, 'K', fontsize=18, fontweight='bold',
            ha='center', va='center', color='darkblue')

    # Draw image of K under continuous map T
    vertices_tk = vertices * 0.7 + np.array([0.6, 0.5])
    polygon_tk = Polygon(vertices_tk, fill=True, facecolor='lightgreen',
                        edgecolor='darkgreen', linewidth=2, alpha=0.4, linestyle='--')
    ax.add_patch(polygon_tk)
    ax.text(1.2, 0.9, 'T(K)', fontsize=12, fontweight='bold',
            ha='center', va='center', color='darkgreen')

    # Fixed point (intersection point)
    fixed_pt = np.array([1.1, 1.0])
    ax.plot(*fixed_pt, 'r*', markersize=30, label=r'Fixed point: $x^* = Tx^*$', zorder=5)

    # Add conditions
    conditions = r"$T: K \to K$ continuous, " + "\n" + \
                r"$K$ compact, closed, convex, nonempty"
    ax.text(1.5, 2.8, conditions, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])

    plt.tight_layout()
    plt.savefig('figures/fig_schauder_fixed_point.pdf', bbox_inches='tight')
    print("Saved: fig_schauder_fixed_point.pdf")
    plt.close()


def fig_asymptotically_nonexpansive():
    """Visualize asymptotically nonexpansive mapping: ||T^n x - T^n y|| <= k_n ||x - y||"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_values = np.arange(1, 21)
    k_n_values = 1 + 0.8 * np.exp(-0.3 * n_values)

    ax.plot(n_values, k_n_values, 'b-', linewidth=2.5, label=r'$k_n$ (Lipschitz constant)')
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label=r'Limit: $\lim_{n \to \infty} k_n = 1$')

    # Add markers at specific points
    ax.plot(n_values[::2], k_n_values[::2], 'bo', markersize=7)

    # Shade the region between k_n and 1
    ax.fill_between(n_values, 1, k_n_values, alpha=0.2, color='blue')

    ax.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$k_n$', fontsize=12, fontweight='bold')
    ax.set_title('Asymptotically Nonexpansive Mapping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, 21)
    ax.set_ylim(0.9, 1.85)

    # Add annotation
    ax.text(12, 1.4, r'$\|T^n x - T^n y\| \leq k_n \|x - y\|$' + '\n' + r'with $k_n \to 1$',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/fig_asymptotically_nonexpansive.pdf', bbox_inches='tight')
    print("Saved: fig_asymptotically_nonexpansive.pdf")
    plt.close()


def fig_fixed_point_iteration():
    """Illustrate fixed point iteration convergence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Graphical fixed point iteration
    x = np.linspace(0, 2, 200)
    f = lambda t: 0.5 * t + 0.3
    y = f(x)

    ax1.plot(x, y, 'b-', linewidth=2.5, label='y = T(x)')
    ax1.plot(x, x, 'k--', linewidth=1.5, label='y = x')

    # Fixed point
    x_fixed = 0.6  # Solve x = 0.5*x + 0.3
    ax1.plot(x_fixed, x_fixed, 'r*', markersize=20, label=f'Fixed point: x* = {x_fixed}')

    # Show iteration
    x_iter = [0.1]
    for i in range(8):
        y_new = f(x_iter[-1])
        x_iter.append(y_new)

    x_iter = np.array(x_iter)

    # Plot web diagram
    for i in range(len(x_iter)-1):
        # Vertical line to curve
        ax1.plot([x_iter[i], x_iter[i]], [x_iter[i], f(x_iter[i])], 'r-', linewidth=1, alpha=0.7)
        # Horizontal line to diagonal
        ax1.plot([x_iter[i], f(x_iter[i])], [f(x_iter[i]), f(x_iter[i])], 'r-', linewidth=1, alpha=0.7)
        ax1.plot(x_iter[i+1], x_iter[i+1], 'ro', markersize=4, alpha=0.6)

    ax1.set_xlabel('x', fontsize=12, fontweight='bold')
    ax1.set_ylabel('T(x)', fontsize=12, fontweight='bold')
    ax1.set_title('Graphical Fixed Point Iteration', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 2)
    ax1.set_ylim(-0.1, 2)

    # Right: Convergence of iterates
    errors = np.abs(x_iter - x_fixed)
    iterations = np.arange(len(x_iter))

    ax2.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=6, label='|x_n - x*|')
    ax2.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence: Error vs Iteration', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_fixed_point_iteration.pdf', bbox_inches='tight')
    print("Saved: fig_fixed_point_iteration.pdf")
    plt.close()


def fig_contraction_mapping():
    """Compare different types of contractions"""
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.linspace(0, 1, 200)

    # Different types of mappings
    contraction = x * 0.7
    nonexpansive = x.copy()
    quasi_nonexp = np.minimum(x, 0.95)
    expansion = x * 1.2

    ax.plot(x, contraction, 'b-', linewidth=2.5, label='Contraction (k < 1)')
    ax.plot(x, nonexpansive, 'g--', linewidth=2.5, label='Nonexpansive (k = 1)')
    ax.plot(x, quasi_nonexp, 'r-.', linewidth=2.5, label='Quasi-nonexpansive')
    ax.plot(x, expansion, 'k:', linewidth=2.5, label='Expansion (k > 1)')
    ax.plot(x, x, 'gray', linewidth=1, alpha=0.5)

    # Fixed points
    ax.plot(0, 0, 'bs', markersize=10, label='FP: Contraction')
    ax.plot(1, 1, 'gD', markersize=10, label='FP: Nonexpansive')
    ax.plot(0.95, 0.95, 'r^', markersize=10, label='FP: Quasi-nonexp')

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('T(x)', fontsize=12, fontweight='bold')
    ax.set_title('Classification of Contractive Mappings', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.3)

    # Add theorem reference box
    theorem_box = r"$\|Tx - Ty\| \leq \lambda\|x - y\|$" + "\n" + \
                  r"• $\lambda < 1$: Contraction (unique FP)" + "\n" + \
                  r"• $\lambda = 1$: Nonexpansive (FP may exist)" + "\n" + \
                  r"• $\lambda > 1$: Expansion (might have FP)"
    ax.text(0.55, 0.25, theorem_box, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig('figures/fig_contraction_mapping.pdf', bbox_inches='tight')
    print("Saved: fig_contraction_mapping.pdf")
    plt.close()


def fig_pseudocontractive_mapping():
    """Visualize pseudocontractive mapping: (Fx - Fy, x - y) <= ||x - y||^2"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a 2D space
    x_range = np.linspace(-1, 3, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Define a pseudocontractive mapping (example)
    # F(x,y) = (0.5*x - 0.3*y, 0.2*x + 0.6*y)
    Fx = 0.5 * X - 0.3 * Y
    Fy = 0.2 * X + 0.6 * Y

    # Compute the pseudocontractivity measure: (Fx - F0, x - 0)
    # where F0 = F(0,0) = (0, 0)
    pseudo_measure = Fx * X + Fy * Y
    norm_sq = X**2 + Y**2

    # Plot contours
    levels = np.linspace(0, 10, 15)
    cs = ax.contourf(X, Y, pseudo_measure, levels=20, cmap='RdYlBu_r', alpha=0.7)
    cbar = plt.colorbar(cs, ax=ax, label='(Fx - F0, x - 0)')

    # Draw some vector field arrows
    step = 5
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              Fx[::step, ::step], Fy[::step, ::step],
              alpha=0.5, scale=20, width=0.003)

    # Fixed point
    ax.plot(0, 0, 'g*', markersize=25, label='Fixed point: (0,0)')

    ax.set_xlabel(r'$x$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y$', fontsize=12, fontweight='bold')
    ax.set_title('Pseudocontractive Mapping', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)

    # Add definition
    definition = r"Pseudocontractive if:" + "\n" + \
                 r"$(Fx - Fy, x - y) \leq \|x - y\|^2$"
    ax.text(0.5, 2.5, definition, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('figures/fig_pseudocontractive_mapping.pdf', bbox_inches='tight')
    print("Saved: fig_pseudocontractive_mapping.pdf")
    plt.close()


def main():
    """Generate all figures"""
    print("Generating figures for Chapter 5.3: Fixed Point Theorems...")

    fig_nonexpansive_mapping()
    fig_brouwer_fixed_point()
    fig_schauder_fixed_point()
    fig_asymptotically_nonexpansive()
    fig_fixed_point_iteration()
    fig_contraction_mapping()
    fig_pseudocontractive_mapping()

    print("\nAll figures generated successfully!")
    print("Location: ./figures/")


if __name__ == '__main__':
    main()
