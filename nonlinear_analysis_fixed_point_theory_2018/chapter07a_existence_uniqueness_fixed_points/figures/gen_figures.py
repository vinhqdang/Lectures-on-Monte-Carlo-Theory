#!/usr/bin/env python3
"""
Generate figures for Chapter 7a: Variational Methods and Optimization
Pathak's "An Introduction to Nonlinear Analysis and Fixed Point Theory"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'neutral': '#6C757D'
}

def set_figure_params():
    """Configure matplotlib for publication-quality figures"""
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['patch.linewidth'] = 1.5

set_figure_params()

# ==============================================================================
# Figure 1: Variational Principle - Convex Function
# ==============================================================================
def figure_variational_principle():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.linspace(-3, 3, 200)
    y = x**2 - 2*x + 1  # (x-1)^2

    ax.plot(x, y, color=colors['primary'], linewidth=2.5, label=r'$f(x) = (x-1)^2$')

    # Mark the minimum
    x_min = 1
    y_min = 0
    ax.plot(x_min, y_min, 'o', color=colors['accent'], markersize=12,
            label=f'Minimum: $\\bar{{x}} = {x_min}$, $f(\\bar{{x}}) = {y_min}$', zorder=5)

    # Add vertical line
    ax.vlines(x_min, 0, 2, colors=colors['accent'], linestyles='dashed', alpha=0.6)

    # Add annotation
    ax.annotate('Global Minimum', xy=(x_min, y_min), xytext=(x_min + 0.8, 1.5),
                fontsize=11, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.5))

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$f(x)$', fontsize=13)
    ax.set_title('Variational Principle: Minimization of Convex Function',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim([-0.5, 5])

    plt.tight_layout()
    plt.savefig('figures/fig_variational_principle.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_variational_principle.pdf")
    plt.close()

# ==============================================================================
# Figure 2: Ekeland's Variational Principle
# ==============================================================================
def figure_ekeland_principle():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    x = np.linspace(0, 10, 300)
    # A function that has a near-minimum
    f = 0.5 + 0.3*np.sin(x/2) + 0.1*x

    ax.plot(x, f, color=colors['primary'], linewidth=2.5, label='$f(x)$')

    # x_0 - initial point
    x0_idx = 30
    x0 = x[x0_idx]
    f_x0 = f[x0_idx]

    ax.plot(x0, f_x0, 'o', color=colors['secondary'], markersize=11,
            label=f'Initial point $x_0$', zorder=5)

    # x_bar - the point from Ekeland's theorem
    x_bar_idx = 80
    x_bar = x[x_bar_idx]
    f_x_bar = f[x_bar_idx]

    ax.plot(x_bar, f_x_bar, 's', color=colors['accent'], markersize=11,
            label=f'Point $\\bar{{x}}$ from Ekeland', zorder=5)

    # Perturbation cone
    epsilon = 0.3
    lambda_param = 0.5
    cone_x = np.linspace(x_bar - 3, x_bar + 3, 100)
    cone_y = f_x_bar + (epsilon/lambda_param) * np.abs(cone_x - x_bar)

    ax.fill_between(cone_x, f_x_bar, cone_y, alpha=0.15, color=colors['accent'],
                     label='Perturbation cone')
    ax.plot(cone_x, cone_y, '--', color=colors['accent'], linewidth=1.5, alpha=0.7)

    ax.vlines(x0, 0, f_x0, colors=colors['secondary'], linestyles='dotted', alpha=0.5)
    ax.vlines(x_bar, 0, f_x_bar, colors=colors['accent'], linestyles='dotted', alpha=0.5)

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$f(x)$', fontsize=13)
    ax.set_title("Ekeland's Variational Principle", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 3])

    plt.tight_layout()
    plt.savefig('figures/fig_ekeland_principle.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_ekeland_principle.pdf")
    plt.close()

# ==============================================================================
# Figure 3: Contraction Mapping and Fixed Point
# ==============================================================================
def figure_contraction_fixed_point():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Non-contractive mapping
    x = np.linspace(0, 1, 100)
    y1 = x**0.5

    ax1.plot(x, y1, color=colors['primary'], linewidth=2.5, label=r'$T(x) = \sqrt{x}$')
    ax1.plot(x, x, 'k--', linewidth=1.5, alpha=0.7, label='$y = x$')

    # Fixed point
    x_fp = (1 + np.sqrt(5))/2 - 1  # Golden ratio - 1 ≈ 0.618
    y_fp = x_fp
    ax1.plot(x_fp, y_fp, 'o', color=colors['accent'], markersize=12,
            label=f'Fixed point $\\bar{{x}}$', zorder=5)

    # Iteration arrows
    x_iter = 0.8
    for i in range(4):
        y_iter = np.sqrt(x_iter)
        ax1.annotate('', xy=(y_iter, y_iter), xytext=(x_iter, y_iter),
                    arrowprops=dict(arrowstyle='->', color=colors['secondary'], lw=1.5))
        ax1.annotate('', xy=(y_iter, x_iter), xytext=(y_iter, y_iter),
                    arrowprops=dict(arrowstyle='->', color=colors['secondary'], lw=1.5))
        x_iter = y_iter

    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$y = T(x)$', fontsize=12)
    ax1.set_title('Contraction Mapping: Fixed Point Iteration', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Right plot: Convergence of iterates
    iterations = 20
    x_seq = np.zeros(iterations)
    x_seq[0] = 0.8
    for i in range(1, iterations):
        x_seq[i] = np.sqrt(x_seq[i-1])

    ax2.plot(range(iterations), x_seq, 'o-', color=colors['primary'], linewidth=2,
            markersize=8, label='Iterates $x_n = T(x_{n-1})$')
    ax2.axhline(y=x_fp, color=colors['accent'], linestyle='--', linewidth=2,
               label=f'Fixed point $\\bar{{x}} \\approx {x_fp:.3f}$')

    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel(r'$x_n$', fontsize=12)
    ax2.set_title('Convergence of Iterates to Fixed Point', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim([0, 19])
    ax2.set_ylim([0.55, 0.85])

    plt.tight_layout()
    plt.savefig('figures/fig_contraction_fixed_point.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_contraction_fixed_point.pdf")
    plt.close()

# ==============================================================================
# Figure 4: Variational Inequality in 2D
# ==============================================================================
def figure_variational_inequality():
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # Create convex set K (a circle)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 2 + 1.2*np.cos(theta)
    circle_y = 2 + 1.2*np.sin(theta)
    ax.fill(circle_x, circle_y, alpha=0.15, color=colors['primary'], label='Convex set $K$')
    ax.plot(circle_x, circle_y, color=colors['primary'], linewidth=2.5)

    # Point x_bar in K (solution)
    x_bar = np.array([2.0, 2.0])
    ax.plot(x_bar[0], x_bar[1], 'o', color=colors['accent'], markersize=14,
           label='Solution $\\bar{x} \\in K$', zorder=5)

    # Vector field f(x) = A*x + b (example of pseudo-monotone operator)
    A = np.array([[1.5, -0.3], [-0.3, 1.2]])
    b = np.array([-2, -2])

    X = np.linspace(0, 4, 12)
    Y = np.linspace(0, 4, 12)
    U = np.zeros((len(Y), len(X)))
    V = np.zeros((len(Y), len(X)))

    for i, y in enumerate(Y):
        for j, x in enumerate(X):
            vec = A @ np.array([x, y]) + b
            U[i, j] = vec[0]
            V[i, j] = vec[1]

    # Normalize for better visualization
    N = np.sqrt(U**2 + V**2)
    U2, V2 = U/N, V/N

    ax.quiver(X, Y, U2, V2, N, cmap='viridis', alpha=0.6, scale=30)

    # Show the variational inequality condition
    y_test = np.array([2.5, 1.5])
    f_x_bar = A @ x_bar + b
    v_minus_x = y_test - x_bar

    ax.arrow(x_bar[0], x_bar[1], 0.6*v_minus_x[0], 0.6*v_minus_x[1],
            head_width=0.15, head_length=0.1, fc=colors['secondary'],
            ec=colors['secondary'], alpha=0.7, linewidth=2)

    ax.arrow(x_bar[0], x_bar[1], 0.5*f_x_bar[0], 0.5*f_x_bar[1],
            head_width=0.12, head_length=0.08, fc=colors['success'],
            ec=colors['success'], alpha=0.8, linewidth=2)

    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title('Variational Inequality: $(x - \\bar{x})^T f(\\bar{x}) \\geq 0$ for all $x \\in K$',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figures/fig_variational_inequality.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_variational_inequality.pdf")
    plt.close()

# ==============================================================================
# Figure 5: Convex vs Non-Convex Functions
# ==============================================================================
def figure_convex_nonconvex():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.linspace(-2, 2, 200)

    # Left: Convex function
    y_convex = x**2
    ax1.plot(x, y_convex, color=colors['primary'], linewidth=2.5, label='$f(x) = x^2$ (Convex)')

    # Tangent line at x=0.5
    x0 = 0.5
    f0 = x0**2
    slope = 2*x0
    y_tangent = f0 + slope*(x - x0)
    ax1.plot(x, y_tangent, '--', color=colors['secondary'], linewidth=2, alpha=0.7,
            label='Tangent line')

    ax1.fill_between(x, y_convex, y_tangent, where=(y_tangent <= y_convex),
                    alpha=0.2, color=colors['success'], label='$f(x) \\geq$ tangent')

    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f(x)$', fontsize=12)
    ax1.set_title('Convex Function: $f(\\alpha x + (1-\\alpha)y) \\leq \\alpha f(x) + (1-\\alpha)f(y)$',
                 fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper center', fontsize=10)
    ax1.set_ylim([-0.5, 4])

    # Right: Non-convex function
    y_nonconvex = -x**2 + 2*x + 1
    ax2.plot(x, y_nonconvex, color=colors['primary'], linewidth=2.5, label='$f(x) = -x^2 + 2x + 1$')

    # Mark local maxima
    ax2.plot(1, 2, 'o', color=colors['accent'], markersize=12, label='Local maximum', zorder=5)

    # Two points for illustration
    x1, x2 = -0.5, 1.8
    f1, f2 = -(x1**2) + 2*x1 + 1, -(x2**2) + 2*x2 + 1

    alpha = 0.5
    x_mid = alpha*x1 + (1-alpha)*x2
    f_mid = -(x_mid**2) + 2*x_mid + 1
    f_convex_combo = alpha*f1 + (1-alpha)*f2

    ax2.plot([x1, x2], [f1, f2], 'o', color=colors['secondary'], markersize=10)
    ax2.plot([x1, x2], [f1, f2], '--', color=colors['secondary'], linewidth=2, alpha=0.6,
            label='Secant line')
    ax2.plot(x_mid, f_mid, 's', color=colors['accent'], markersize=10)
    ax2.plot(x_mid, f_convex_combo, '^', color=colors['secondary'], markersize=10)

    ax2.annotate('', xy=(x_mid, f_mid), xytext=(x_mid, f_convex_combo),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    ax2.set_xlabel(r'$x$', fontsize=12)
    ax2.set_ylabel(r'$f(x)$', fontsize=12)
    ax2.set_title('Non-Convex Function: Not necessarily satisfying convexity',
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim([-0.5, 2.5])

    plt.tight_layout()
    plt.savefig('figures/fig_convex_nonconvex.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_convex_nonconvex.pdf")
    plt.close()

# ==============================================================================
# Figure 6: Gradient Descent Convergence
# ==============================================================================
def figure_gradient_descent():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Gradient descent path on contour plot
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2  # Simple quadratic function

    contour = ax1.contour(X, Y, Z, levels=10, colors='gray', alpha=0.5, linewidths=0.5)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Gradient descent iterations
    x_current = np.array([-1.5, 1.5])
    learning_rate = 0.3
    iterations_gd = []

    for _ in range(20):
        iterations_gd.append(x_current.copy())
        gradient = 2 * x_current
        x_current = x_current - learning_rate * gradient

    iterations_gd = np.array(iterations_gd)

    ax1.plot(iterations_gd[:, 0], iterations_gd[:, 1], 'o-', color=colors['accent'],
            linewidth=2, markersize=6, label='Gradient descent path')
    ax1.plot(0, 0, '*', color=colors['success'], markersize=20, label='Minimum', zorder=5)
    ax1.plot(iterations_gd[0, 0], iterations_gd[0, 1], 's', color=colors['secondary'],
            markersize=10, label='Starting point', zorder=5)

    ax1.set_xlabel(r'$x_1$', fontsize=12)
    ax1.set_ylabel(r'$x_2$', fontsize=12)
    ax1.set_title('Gradient Descent: Path to Minimum', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.grid(True, alpha=0.3)

    # Right: Convergence of objective function
    obj_values = np.sum(iterations_gd**2, axis=1)

    ax2.semilogy(range(len(obj_values)), obj_values, 'o-', color=colors['primary'],
                linewidth=2, markersize=7, label='Objective value $f(x_n)$')
    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel(r'$f(x_n) = \|x_n\|^2$', fontsize=12)
    ax2.set_title('Convergence of Objective Function (log scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_gradient_descent.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_gradient_descent.pdf")
    plt.close()

# ==============================================================================
# Figure 7: Banach Contraction Principle Illustration
# ==============================================================================
def figure_banach_contraction():
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    x = np.linspace(0, 4, 200)
    k = 0.6  # Contraction constant
    T = lambda x: k*x + 0.5

    # Plot T(x) and y=x line
    y_Tx = T(x)
    ax.plot(x, y_Tx, color=colors['primary'], linewidth=2.5, label='$T(x) = 0.6x + 0.5$ (Contraction)')
    ax.plot(x, x, 'k--', linewidth=2, alpha=0.7, label='$y = x$')

    # Fixed point: solve x = 0.6x + 0.5 => x = 1.25
    x_fp = 0.5 / (1 - k)
    ax.plot(x_fp, x_fp, 'o', color=colors['accent'], markersize=14,
           label=f'Unique fixed point $\\bar{{x}} = {x_fp:.3f}$', zorder=5)

    # Show iterations with cobweb diagram
    x_iter = 0.2
    colors_iter = [colors['secondary'], colors['accent'], colors['success']]

    for i in range(8):
        y_iter = T(x_iter)

        # Draw vertical line from (x_iter, x_iter) to (x_iter, y_iter)
        ax.plot([x_iter, x_iter], [x_iter, y_iter], '--', color=colors_iter[i % 3],
               alpha=0.6, linewidth=1.5)

        # Draw horizontal line from (x_iter, y_iter) to (y_iter, y_iter)
        ax.plot([x_iter, y_iter], [y_iter, y_iter], '--', color=colors_iter[i % 3],
               alpha=0.6, linewidth=1.5)

        # Update x for next iteration
        x_iter = y_iter

    # Final convergence
    ax.text(x_fp + 0.15, x_fp - 0.15, 'Converges here', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$y = T(x)$', fontsize=13)
    ax.set_title('Banach Contraction Principle: Cobweb Diagram', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])

    # Add theoretical condition box
    textstr = 'Contraction Condition:\n' + r'$\exists k \in [0,1): \|T(x) - T(y)\| \leq k\|x - y\|$'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('figures/fig_banach_contraction.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_banach_contraction.pdf")
    plt.close()

# ==============================================================================
# Figure 8: Normal Cone and Projection
# ==============================================================================
def figure_normal_cone():
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    # Convex set K (a triangle)
    K_x = np.array([0, 3, 1.5, 0])
    K_y = np.array([0, 0, 3, 0])
    ax.fill(K_x, K_y, alpha=0.15, color=colors['primary'], label='Convex set $K$')
    ax.plot(K_x, K_y, color=colors['primary'], linewidth=2.5)

    # Point on boundary
    z_boundary = np.array([1.5, 0])
    ax.plot(z_boundary[0], z_boundary[1], 'o', color=colors['secondary'], markersize=12,
           label='Point $z$ on boundary of $K$', zorder=5)

    # Point inside K
    z_inside = np.array([1.5, 1.0])
    ax.plot(z_inside[0], z_inside[1], 's', color=colors['accent'], markersize=12,
           label='Point $z$ inside $K$', zorder=5)

    # Normal cone at boundary point
    normal_direction = np.array([0, -1])
    normal_length = 1.5
    ax.arrow(z_boundary[0], z_boundary[1],
            normal_direction[0]*normal_length, normal_direction[1]*normal_length,
            head_width=0.15, head_length=0.15, fc=colors['secondary'],
            ec=colors['secondary'], linewidth=2.5, label='Normal cone $N_K(z)$', zorder=5)

    # Projection illustration
    P_z = np.array([1.5, 0])  # projection of z onto K
    ax.plot([z_inside[0], P_z[0]], [z_inside[1], P_z[1]], ':', color=colors['success'],
           linewidth=2.5, label='Projection $P_K(z)$')
    ax.plot(P_z[0], P_z[1], 'X', color=colors['success'], markersize=14, zorder=5)

    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title('Normal Cone and Projection onto Convex Set', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim([-1, 4])
    ax.set_ylim([-2.5, 3.5])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figures/fig_normal_cone.pdf', dpi=300, bbox_inches='tight')
    print("Created: fig_normal_cone.pdf")
    plt.close()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 7a...")
    print()

    figure_variational_principle()
    figure_ekeland_principle()
    figure_contraction_fixed_point()
    figure_variational_inequality()
    figure_convex_nonconvex()
    figure_gradient_descent()
    figure_banach_contraction()
    figure_normal_cone()

    print()
    print("All figures generated successfully!")
