#!/usr/bin/env python3
"""
Figure generation script for Chapter 27: Fermat's Rule in Convex Optimization
Generates all figures used in the Beamer slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def generate_constrained_opt_figure():
    """
    Generate figure showing constrained minimization:
    minimize ||x - c||^2 subject to x in [0,1]^2
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Feasible region [0,1]^2
    rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='blue',
                      facecolor='lightblue', alpha=0.3, label='Feasible set C')
    ax.add_patch(rect)

    # Target point c
    c = np.array([0.7, 0.8])
    ax.plot(c[0], c[1], 'r*', markersize=20, label='Target point c = (0.7, 0.8)')

    # Projection of c onto [0,1]^2
    c_proj = np.clip(c, 0, 1)
    ax.plot(c_proj[0], c_proj[1], 'go', markersize=12, label=f'Optimal point (0.7, 0.8)')

    # Draw line from c to projection
    ax.plot([c[0], c_proj[0]], [c[1], c_proj[1]], 'g--', linewidth=2, alpha=0.7)

    # Add level sets of objective function ||x - c||^2
    x_range = np.linspace(-0.2, 1.2, 100)
    y_range = np.linspace(-0.2, 1.2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - c[0])**2 + (Y - c[1])**2

    # Draw contour lines
    contours = ax.contour(X, Y, Z, levels=8, colors='red', alpha=0.5, linewidths=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # Add gradient vector at c
    grad_magnitude = 0.15
    grad_direction = (c_proj - c) / np.linalg.norm(c_proj - c + 1e-10)
    ax.arrow(c[0], c[1], -grad_direction[0] * grad_magnitude,
             -grad_direction[1] * grad_magnitude,
             head_width=0.03, head_length=0.03, fc='darkred', ec='darkred', alpha=0.7)
    ax.text(c[0] - 0.1, c[1] - 0.15, r'$\nabla f(c)$', fontsize=12, color='darkred')

    # Labels and formatting
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=14)
    ax.set_ylabel(r'$x_2$', fontsize=14)
    ax.set_title(r'Constrained Minimization: $\min_{x \in [0,1]^2} \|x - c\|^2$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_constrained_opt.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_constrained_opt.png', dpi=150, bbox_inches='tight')
    print("Generated: fig_constrained_opt.pdf")
    plt.close()


def generate_fermat_rule_visualization():
    """
    Generate figure showing Fermat's rule in action.
    Shows a convex function and its subdifferential at the minimum.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: 1D convex function
    x = np.linspace(-3, 3, 200)
    f = x**2  # Simple quadratic

    ax1.plot(x, f, 'b-', linewidth=2.5, label=r'$f(x) = x^2$')

    # Minimum point
    x_min = 0
    ax1.plot(x_min, f[np.argmin(np.abs(x - x_min))], 'ro', markersize=10, label='Minimizer')

    # Subdifferential at minimum (just {0})
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.plot(x_min, 0, 'r^', markersize=8, label=r'$0 \in \partial f(x^*)$')

    # Tangent line at minimum (horizontal)
    x_tangent = np.array([-2, 2])
    ax1.plot(x_tangent, np.zeros_like(x_tangent), 'r--', linewidth=2, alpha=0.5)

    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f(x)$', fontsize=12)
    ax1.set_title(r"Fermat's Rule: $x^* \in \text{Argmin}(f) \Leftrightarrow 0 \in \partial f(x^*)$",
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 8)

    # Right plot: Absolute value function (non-smooth)
    x2 = np.linspace(-3, 3, 200)
    f2 = np.abs(x2)

    ax2.plot(x2, f2, 'g-', linewidth=2.5, label=r'$f(x) = |x|$')

    # Minimum point
    x_min2 = 0
    ax2.plot(x_min2, 0, 'ro', markersize=10, label='Minimizer')

    # Subdifferential at minimum (interval [-1, 1])
    subdiff_x = np.array([-0.2, 0.2])
    for v in [-1, -0.5, 0, 0.5, 1]:
        ax2.plot([x_min2 - 0.1, x_min2 + 0.1],
                [x_min2 - 0.1 + v, x_min2 + 0.1 + v],
                'r--', alpha=0.3, linewidth=1)

    ax2.set_xlabel(r'$x$', fontsize=12)
    ax2.set_ylabel(r'$f(x)$', fontsize=12)
    ax2.set_title(r'Non-smooth function: $0 \in \partial f(0) = [-1, 1]$',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.savefig('fig_fermat_rule.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_fermat_rule.png', dpi=150, bbox_inches='tight')
    print("Generated: fig_fermat_rule.pdf")
    plt.close()


def generate_subdifferential_visualization():
    """
    Generate figure showing subdifferentials of various convex functions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Function 1: Smooth convex function
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 200)
    f1 = (x**2 + np.abs(x)) / 2
    ax.plot(x, f1, 'b-', linewidth=2.5)
    ax.fill_between(x, 0, f1, alpha=0.2, color='blue')
    ax.set_title('Smooth convex function', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$f(x)$', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Function 2: Maximum function
    ax = axes[0, 1]
    x2 = np.linspace(-2, 4, 200)
    f2 = np.maximum(x2, -x2 + 2)
    ax.plot(x2, f2, 'g-', linewidth=2.5)
    ax.fill_between(x2, 0, f2, alpha=0.2, color='green')
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.set_title(r'Max function: $\max(x, 2-x)$ (non-smooth at $x=1$)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$f(x)$', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Function 3: Absolute value
    ax = axes[1, 0]
    x3 = np.linspace(-3, 3, 200)
    f3 = np.abs(x3)
    ax.plot(x3, f3, 'm-', linewidth=2.5)
    ax.fill_between(x3, 0, f3, alpha=0.2, color='magenta')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(r'Absolute value: $|x|$ (non-smooth at $x=0$)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$f(x)$', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Function 4: Hinge loss
    ax = axes[1, 1]
    x4 = np.linspace(-2, 4, 200)
    f4 = np.maximum(0, 1 - x4)
    ax.plot(x4, f4, 'c-', linewidth=2.5)
    ax.fill_between(x4, 0, f4, alpha=0.2, color='cyan')
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.set_title(r'Hinge loss: $\max(0, 1-x)$ (non-smooth at $x=1$)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'$f(x)$', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Examples of Convex Functions and Their Subdifferentials',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('fig_subdifferentials.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_subdifferentials.png', dpi=150, bbox_inches='tight')
    print("Generated: fig_subdifferentials.pdf")
    plt.close()


def generate_affine_constraint_figure():
    """
    Generate figure showing minimization subject to affine constraints.
    minimize ||x||^2 subject to sum(x_i) = 1
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: 2D slice
    # minimize x1^2 + x2^2 subject to x1 + x2 = 1
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_constraint = 1 - x1_range

    # Level sets of objective function
    x1_level = np.linspace(-0.5, 1.5, 100)
    x2_level = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x1_level, x2_level)
    Z = X**2 + Y**2

    contours = ax1.contour(X, Y, Z, levels=10, colors='blue', alpha=0.5)
    ax1.clabel(contours, inline=True, fontsize=8)

    # Constraint line
    ax1.plot(x1_range, x2_constraint, 'r-', linewidth=3, label='Constraint: $x_1 + x_2 = 1$')

    # Optimal point: (0.5, 0.5)
    x_opt = np.array([0.5, 0.5])
    ax1.plot(x_opt[0], x_opt[1], 'go', markersize=12, label='Optimal: (0.5, 0.5)')

    # Gradient at optimal point (perpendicular to constraint)
    grad = 2 * x_opt
    grad = grad / np.linalg.norm(grad)
    ax1.arrow(x_opt[0], x_opt[1], grad[0]*0.2, grad[1]*0.2,
             head_width=0.03, head_length=0.03, fc='green', ec='green')

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$x_1$', fontsize=12)
    ax1.set_ylabel(r'$x_2$', fontsize=12)
    ax1.set_title(r'2D: $\min \|x\|^2$ subject to $x_1 + x_2 = 1$',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right plot: 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')

    # Create data for 3D plot
    n_points = 50
    x1 = np.linspace(-0.5, 1.5, n_points)
    x2 = np.linspace(-0.5, 1.5, n_points)
    X_3d, Y_3d = np.meshgrid(x1, x2)
    Z_3d = X_3d**2 + Y_3d**2

    # Plot surface
    ax2.plot_surface(X_3d, Y_3d, Z_3d, cmap='viridis', alpha=0.6, linewidth=0)

    # Plot constraint line
    x1_constraint = np.linspace(-0.2, 1.2, 50)
    x2_constraint = 1 - x1_constraint
    z_constraint = x1_constraint**2 + x2_constraint**2
    ax2.plot(x1_constraint, x2_constraint, z_constraint, 'r-', linewidth=3,
             label='Constraint curve')

    # Optimal point
    ax2.scatter([0.5], [0.5], [0.5], color='green', s=100, marker='o',
               label='Optimal: (0.5, 0.5, 0.5)')

    ax2.set_xlabel(r'$x_1$', fontsize=11)
    ax2.set_ylabel(r'$x_2$', fontsize=11)
    ax2.set_zlabel(r'$\|x\|^2$', fontsize=11)
    ax2.set_title('3D View of Objective and Constraint', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)

    # Remove the original ax2 to avoid overlap
    fig.delaxes(ax2)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_3d, Y_3d, Z_3d, cmap='viridis', alpha=0.6, linewidth=0)
    ax2.plot(x1_constraint, x2_constraint, z_constraint, 'r-', linewidth=3)
    ax2.scatter([0.5], [0.5], [0.5], color='green', s=100, marker='o')
    ax2.set_xlabel(r'$x_1$', fontsize=11)
    ax2.set_ylabel(r'$x_2$', fontsize=11)
    ax2.set_zlabel(r'$\|x\|^2$', fontsize=11)
    ax2.set_title('3D View', fontsize=12, fontweight='bold')

    plt.suptitle('Affine Constraints: Lagrange Multipliers', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_affine_constraints.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_affine_constraints.png', dpi=150, bbox_inches='tight')
    print("Generated: fig_affine_constraints.pdf")
    plt.close()


def generate_convergence_figure():
    """
    Generate figure showing convergence of iterative optimization methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Simulate convergence of proximal gradient method
    np.random.seed(42)

    # Algorithm 1: Fast convergence (smooth + strongly convex)
    n_iter = 50
    f_values_fast = 10 * np.exp(-0.15 * np.arange(n_iter)) + 0.01 * np.random.randn(n_iter)
    f_values_fast[0] = 10
    f_values_fast = np.maximum.accumulate(-np.sort(-f_values_fast))

    ax = axes[0]
    ax.semilogy(f_values_fast, 'b-o', linewidth=2, markersize=4, label='Proximal Gradient')
    ax.semilogy(0.5 * (0.9 ** np.arange(n_iter)), 'r--', linewidth=2, label='Exponential decay')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'$f(x_k) - f(x^*)$', fontsize=12)
    ax.set_title('Linear Convergence (Smooth + Strongly Convex)',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Algorithm 2: Slower convergence (non-smooth)
    f_values_slow = 1.0 / (np.arange(n_iter) + 1) + 0.01 * np.random.randn(n_iter)
    f_values_slow = np.maximum.accumulate(-np.sort(-f_values_slow))

    ax = axes[1]
    ax.loglog(f_values_slow, 'g-s', linewidth=2, markersize=4, label='Subgradient Method')
    ax.loglog(1.0 / (np.arange(n_iter) + 1), 'r--', linewidth=2, label=r'$O(1/k)$ rate')
    ax.set_xlabel('Iteration (log scale)', fontsize=12)
    ax.set_ylabel(r'$f(x_k) - f(x^*)$ (log scale)', fontsize=12)
    ax.set_title('Sublinear Convergence (Non-smooth)',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('Convergence Rates of Optimization Algorithms',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_convergence.png', dpi=150, bbox_inches='tight')
    print("Generated: fig_convergence.pdf")
    plt.close()


def main():
    """Generate all figures for Chapter 27."""
    print("Generating figures for Chapter 27: Fermat's Rule in Convex Optimization")
    print("=" * 70)

    generate_constrained_opt_figure()
    generate_fermat_rule_visualization()
    generate_subdifferential_visualization()
    generate_affine_constraint_figure()
    generate_convergence_figure()

    print("=" * 70)
    print("All figures generated successfully!")


if __name__ == '__main__':
    main()
