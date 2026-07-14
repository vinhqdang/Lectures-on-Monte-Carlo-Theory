#!/usr/bin/env python3
"""
Generate figures for Chapter 5.4: Fixed Point Theorems for Multifunctions
Multivalued Mappings & Fixed Points
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, Rectangle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'highlight': '#d62728',
    'light': '#9467bd',
}

def figure_multivalued_mapping():
    """Figure: Concept of a multivalued mapping"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Domain X on the left
    domain_circle = Circle((-2, 0), 1.2, color=colors['primary'], alpha=0.2, ec=colors['primary'], lw=2)
    ax.add_patch(domain_circle)
    ax.text(-2, 0, 'X', fontsize=14, ha='center', va='center', weight='bold')

    # Codomain 2^X on the right (power set)
    codomain_circle = Circle((2, 0), 1.5, color=colors['secondary'], alpha=0.2, ec=colors['secondary'], lw=2)
    ax.add_patch(codomain_circle)
    ax.text(2, 0, r'$2^X$', fontsize=14, ha='center', va='center', weight='bold')

    # Points in domain
    x_points = [-2.5, -2, -1.5]
    for i, x in enumerate(x_points):
        ax.plot(x, 0.3 + i*0.2, 'o', color=colors['primary'], markersize=8)
        ax.text(x-0.3, 0.3 + i*0.2, f'$x_{i+1}$', fontsize=11, ha='right')

    # Sets in codomain (showing multivalued nature)
    y_sets = [
        [Circle((2-0.3, -0.8), 0.25, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5)],
        [Circle((2-0.2, 0), 0.2, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5),
         Circle((2+0.2, 0), 0.2, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5),
         Circle((2, 0.2), 0.2, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5)],
        [Circle((2-0.25, 0.8), 0.2, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5),
         Circle((2+0.25, 0.8), 0.2, color=colors['highlight'], alpha=0.3, ec=colors['highlight'], lw=1.5)]
    ]

    for circles in y_sets:
        for circle in circles:
            ax.add_patch(circle)

    # Arrows showing the multivalued mapping
    for i, x in enumerate(x_points):
        y_pos = 0.3 + i*0.2
        y_targets = [-0.8, 0, 0.8]
        arrow = FancyArrowPatch((x+0.3, y_pos), (2-1.3, y_targets[i]),
                               arrowstyle='->', mutation_scale=20, lw=2, color=colors['primary'])
        ax.add_patch(arrow)

    # Mapping label
    ax.text(0, -2.2, r'$T: X \to 2^X$ is a multivalued mapping', fontsize=13, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('multivalued_mapping.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_hausdorff_metric():
    """Figure: Hausdorff-Pompeu metric visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Draw two sets A and B
    theta = np.linspace(0, 2*np.pi, 100)

    # Set A (ellipse-like)
    A_x = 0.8 * np.cos(theta) - 0.5
    A_y = 0.6 * np.sin(theta)
    ax.fill(A_x, A_y, color=colors['primary'], alpha=0.3, edgecolor=colors['primary'], lw=2.5, label='Set A')

    # Set B (circle)
    B_x = 0.6 * np.cos(theta) + 0.8
    B_y = 0.6 * np.sin(theta) + 0.3
    ax.fill(B_x, B_y, color=colors['secondary'], alpha=0.3, edgecolor=colors['secondary'], lw=2.5, label='Set B')

    # Mark points to show distances
    p_A = np.array([-1.2, 0])
    p_B = np.array([1.2, 0.3])

    ax.plot(p_A[0], p_A[1], 'o', color=colors['primary'], markersize=10, zorder=5)
    ax.plot(p_B[0], p_B[1], 'o', color=colors['secondary'], markersize=10, zorder=5)

    # Draw distance line
    ax.plot([p_A[0], p_B[0]], [p_A[1], p_B[1]], 'k--', lw=1.5, alpha=0.6)
    ax.text((p_A[0]+p_B[0])/2, (p_A[1]+p_B[1])/2 + 0.15, r'$d(a,b)$', fontsize=12, ha='center')

    # Draw ρ(A,B) - maximum distance from A to B
    arrow_rho = FancyArrowPatch((0.8, 0.5), (0.8, 1.0), arrowstyle='<->', mutation_scale=15,
                                lw=2, color=colors['highlight'], alpha=0.7)
    ax.add_patch(arrow_rho)
    ax.text(1.0, 0.75, r'$\rho(A,B)$', fontsize=11, color=colors['highlight'], weight='bold')

    # Hausdorff metric formula
    formula_text = r'$H^+(A,B) = \frac{1}{2}(\rho(A,B) + \rho(B,A))$'
    ax.text(0, -1.2, formula_text, fontsize=13, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.5))

    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('', fontsize=11)
    ax.set_ylabel('', fontsize=11)
    ax.set_title('Hausdorff-Pompeu Metric on CB(X)', fontsize=14, weight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('hausdorff_metric.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_fixed_point_existence():
    """Figure: Fixed point existence for multivalued mapping"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Domain with fixed point
    x = np.linspace(-2, 2, 100)

    # Draw compact convex set K
    K_vertices = np.array([[-1.5, -0.8], [1.5, -0.8], [1.8, 1.2], [-1.8, 1.2]])
    K_polygon = Polygon(K_vertices, closed=True, edgecolor=colors['primary'],
                       facecolor=colors['primary'], alpha=0.15, lw=2.5)
    ax1.add_patch(K_polygon)

    # Show several points and their images
    test_points = [-1.2, -0.5, 0, 0.8, 1.2]
    for pt in test_points:
        y_center = 0.3 * np.sin(pt)
        radius = 0.4 + 0.15 * np.abs(pt)
        circle = Circle((pt, y_center), radius, color=colors['highlight'], alpha=0.2,
                       ec=colors['highlight'], lw=1.5)
        ax1.add_patch(circle)
        ax1.plot(pt, 0, 'o', color=colors['primary'], markersize=7)

    # Mark the fixed point
    fixed_pt = 0.15
    ax1.plot(fixed_pt, 0.2, '*', color=colors['accent'], markersize=25, zorder=5)
    ax1.text(fixed_pt + 0.3, 0.35, r'$x^* \in T(x^*)$', fontsize=12, weight='bold',
            color=colors['accent'])

    ax1.text(0, -1.3, r'$T: K \to \mathit{CB}(K)$ is u.s.c.', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Fixed Point in Compact Convex Set', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right plot: Contraction property
    iterations = np.arange(0, 6)
    # Simulating H(T^n(x), T^n(x)) with contraction
    distances = 1.0 * np.exp(-0.8 * iterations)

    ax2.plot(iterations, distances, 'o-', color=colors['primary'], linewidth=2.5,
            markersize=8, label=r'$H(T^n(x), T^n(x))$')

    # Add exponential decay line
    smooth_iter = np.linspace(0, 5, 100)
    decay = 1.0 * np.exp(-0.8 * smooth_iter)
    ax2.plot(smooth_iter, decay, '--', color=colors['secondary'], linewidth=2,
            alpha=0.7, label='Exponential decay')

    ax2.fill_between(smooth_iter, decay, alpha=0.1, color=colors['secondary'])
    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel(r'$H^+(T^n(x), \{p\})$', fontsize=12)
    ax2.set_title('Contraction Property', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('fixed_point_existence.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_nadler_theorem():
    """Figure: Nadler's Contraction Theorem visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Draw metric space
    circle_radius = 1.5
    metric_space = Circle((0, 0), circle_radius, color=colors['primary'],
                         alpha=0.15, ec=colors['primary'], lw=2.5)
    ax.add_patch(metric_space)
    ax.text(0, -1.95, 'Complete Metric Space (X, d)', fontsize=12, ha='center', weight='bold')

    # Draw sets for iteration
    colors_iter = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    theta = np.linspace(0, 2*np.pi, 100)
    x0_pos = np.array([0.8, 0.6])

    # Sequence of contracted sets
    for n in range(4):
        radius = 1.0 * (0.6 ** n)
        angle = n * 0.3
        center_x = 0.8 * np.cos(angle)
        center_y = 0.6 * np.sin(angle)

        circle = Circle((center_x, center_y), radius,
                       color=colors_iter[n], alpha=0.2, ec=colors_iter[n], lw=2)
        ax.add_patch(circle)
        ax.text(center_x, center_y - radius - 0.25, f'$T^{n}(x_0)$',
               fontsize=11, ha='center', weight='bold')

    # Show convergence to fixed point
    ax.plot(0, 0, '*', color=colors['accent'], markersize=30, zorder=5, label='Fixed point')
    ax.text(0.15, 0.2, r'$p = \bigcap_{n=0}^{\infty} T^n(x_0)$', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Contraction condition
    condition_text = (r'If $H^+(T(x), T(y)) \leq k \cdot H^+(x, y)$ for $k \in (0,1)$'
                     '\nthen $T$ has a unique fixed point')
    ax.text(0, 2.0, condition_text, fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.7))

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('nadler_theorem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_upper_semicontinuous():
    """Figure: Upper semicontinuous multivalued mapping"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: u.s.c. mapping
    x_vals = np.linspace(-2, 2, 50)
    y_center = 0.3 * x_vals

    for x in np.linspace(-2, 2, 9):
        y = 0.3 * x
        width = 0.4 + 0.1 * x**2
        # Draw continuous family of sets
        for dy in np.linspace(-width/2, width/2, 5):
            ax1.plot(x, y + dy, '.', color=colors['primary'], markersize=5, alpha=0.6)

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title('Upper Semicontinuous (u.s.c.) Mapping', fontsize=12, weight='bold')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('T(x)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(0, -1.2, 'Continuous graph variation', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Right: Definition
    definition_text = (
        r'$T: X \to 2^Y$ is u.s.c. if for each closed $B \subseteq Y$,' '\n'
        r'the set $T^{-1}(B) = \{x: T(x) \cap B \neq \emptyset\}$ is closed' '\n'
        '\n'
        r'Equivalently: For each $x \in X$ and open set $U \supseteq T(x)$,' '\n'
        r'there exists open $V \ni x$ such that $U \supseteq T(y)$ for all $y \in V$'
    )
    ax2.text(0.5, 0.5, definition_text, transform=ax2.transAxes, fontsize=11.5,
            verticalalignment='center', horizontalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Definition of Upper Semicontinuity', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('upper_semicontinuous.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_coincidence_fixed_points():
    """Figure: Coincidence and fixed points for two mappings"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Domain
    domain_circle = Circle((-1.5, 0), 1.0, color=colors['primary'], alpha=0.2,
                          ec=colors['primary'], lw=2.5)
    ax.add_patch(domain_circle)
    ax.text(-1.5, 0, 'X', fontsize=13, ha='center', va='center', weight='bold')

    # Codomain
    codomain_circle = Circle((1.5, 0), 1.0, color=colors['secondary'], alpha=0.2,
                            ec=colors['secondary'], lw=2.5)
    ax.add_patch(codomain_circle)
    ax.text(1.5, 0, 'Y', fontsize=13, ha='center', va='center', weight='bold')

    # Show two mappings
    x_test = [-1.8, -1.2]
    for i, x in enumerate(x_test):
        y_f = -0.3 + i * 0.6
        y_g = 0.3 + i * 0.4

        # Point in X
        ax.plot(x, -0.1, 'o', color=colors['primary'], markersize=8)

        # Arrows for f
        arrow_f = FancyArrowPatch((x+0.2, -0.1), (1.2, y_f), arrowstyle='->',
                                 mutation_scale=15, lw=2, color=colors['accent'], alpha=0.8)
        ax.add_patch(arrow_f)

        # Arrows for g
        arrow_g = FancyArrowPatch((x+0.2, -0.1), (1.2, y_g), arrowstyle='->',
                                 mutation_scale=15, lw=2, color=colors['highlight'],
                                 alpha=0.8, linestyle='--')
        ax.add_patch(arrow_g)

        # Points in Y
        ax.plot(1.5, y_f, 's', color=colors['accent'], markersize=8, alpha=0.8)
        ax.plot(1.5, y_g, '^', color=colors['highlight'], markersize=8, alpha=0.8)

    # Highlight coincidence point
    ax.plot(-1.2, -0.1, 'o', color=colors['primary'], markersize=10,
           markeredgewidth=2.5, markerfacecolor='none', markeredgecolor=colors['primary'])
    ax.text(-1.2, -0.5, 'Coincidence\npoint', fontsize=10, ha='center', weight='bold')

    # Legend
    ax.plot([], [], 's-', color=colors['accent'], linewidth=2, markersize=8, label='f: X → Y')
    ax.plot([], [], '^--', color=colors['highlight'], linewidth=2, markersize=8, label='g: X → Y')

    ax.text(0, -1.8, r'Coincidence point: $p \in X$ such that $f(p) = g(p)$',
           fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('coincidence_fixed_points.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def figure_convergence_iteration():
    """Figure: Convergence behavior of iterative schemes"""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    # Simulate convergence for multivalued case
    n_iter = 15
    iterations = np.arange(0, n_iter)

    # Different convergence rates
    fast = np.exp(-1.5 * iterations) * 0.5
    medium = np.exp(-0.8 * iterations) * 0.5
    slow = np.exp(-0.3 * iterations) * 0.5

    ax.semilogy(iterations, fast, 'o-', linewidth=2.5, markersize=7,
               color=colors['accent'], label=r'Fast: $k=0.22$')
    ax.semilogy(iterations, medium, 's-', linewidth=2.5, markersize=7,
               color=colors['primary'], label=r'Medium: $k=0.45$')
    ax.semilogy(iterations, slow, '^-', linewidth=2.5, markersize=7,
               color=colors['secondary'], label=r'Slow: $k=0.74$')

    # Add reference line for linear convergence
    linear_ref = 0.5 * (0.5 ** iterations)
    ax.semilogy(iterations, linear_ref, ':', linewidth=2, color='gray',
               alpha=0.6, label='Linear convergence (k=0.5)')

    ax.set_xlabel('Iteration n', fontsize=12, weight='bold')
    ax.set_ylabel(r'$d(x^{(n)}, x^*)$ (log scale)', fontsize=12, weight='bold')
    ax.set_title('Convergence of Iterative Schemes for Multivalued Mappings',
                fontsize=13, weight='bold', pad=15)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(bottom=1e-5)

    # Add shaded region for convergence
    ax.axhspan(1e-5, 1e-2, alpha=0.1, color=colors['accent'])
    ax.text(11, 5e-4, 'Convergence\nRegion', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('convergence_iteration.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 5.4: Fixed Point Theorems for Multifunctions")

    print("1. Generating multivalued_mapping.pdf...")
    figure_multivalued_mapping()

    print("2. Generating hausdorff_metric.pdf...")
    figure_hausdorff_metric()

    print("3. Generating fixed_point_existence.pdf...")
    figure_fixed_point_existence()

    print("4. Generating nadler_theorem.pdf...")
    figure_nadler_theorem()

    print("5. Generating upper_semicontinuous.pdf...")
    figure_upper_semicontinuous()

    print("6. Generating coincidence_fixed_points.pdf...")
    figure_coincidence_fixed_points()

    print("7. Generating convergence_iteration.pdf...")
    figure_convergence_iteration()

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
