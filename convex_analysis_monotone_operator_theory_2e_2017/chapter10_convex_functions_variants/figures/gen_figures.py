#!/usr/bin/env python3
"""
Generate figures for Chapter 10: Convex Functions: Variants
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os

# Set up matplotlib for publication-quality figures
plt.style.use('default')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Figure 1: Strictly Convex vs Uniformly Convex vs Strongly Convex
# ============================================================================
def fig_convexity_variants():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.linspace(-2, 2, 200)

    # Plot 1: Convex
    y_convex = x**2
    axes[0].plot(x, y_convex, 'b-', linewidth=2, label=r'$f(x) = x^2$')
    # Show convex inequality for two points
    x1, x2 = -1.5, 1.2
    y1, y2 = x1**2, x2**2
    alpha = 0.4
    x_mid = alpha * x1 + (1 - alpha) * x2
    y_mid_actual = x_mid**2
    y_mid_linear = alpha * y1 + (1 - alpha) * y2

    axes[0].plot([x1, x2], [y1, y2], 'r--', alpha=0.5, linewidth=1.5)
    axes[0].plot([x1, x2], [y1, y2], 'ro', markersize=6)
    axes[0].plot(x_mid, y_mid_actual, 'b*', markersize=12)
    axes[0].plot(x_mid, y_mid_linear, 'g^', markersize=8)
    axes[0].set_ylim(-0.5, 5)
    axes[0].set_title('Convex Function', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f(x)$')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend([r'$f(x)$', 'Secant', r'$f(\alpha x_1 + (1-\alpha)x_2)$',
                    r'Linear combo $\leq$ secant'], fontsize=9)

    # Plot 2: Strictly Convex
    y_strictly = x**2 + 0.1 * x**4
    axes[1].plot(x, y_strictly, 'b-', linewidth=2, label=r'$f(x) = x^2 + 0.1x^4$')
    axes[1].plot([x1, x2], [x1**2 + 0.1*x1**4, x2**2 + 0.1*x2**4], 'r--', alpha=0.5, linewidth=1.5)
    axes[1].plot([x1, x2], [x1**2 + 0.1*x1**4, x2**2 + 0.1*x2**4], 'ro', markersize=6)
    axes[1].plot(x_mid, x_mid**2 + 0.1*x_mid**4, 'b*', markersize=12)
    axes[1].plot(x_mid, alpha*(x1**2 + 0.1*x1**4) + (1-alpha)*(x2**2 + 0.1*x2**4),
                 'g^', markersize=8)
    axes[1].set_ylim(-0.5, 5)
    axes[1].set_title('Strictly Convex Function', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$f(x)$')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend([r'$f(x)$', 'Secant', r'$f(\alpha x_1 + (1-\alpha)x_2)$',
                    r'Strict inequality'], fontsize=9)

    # Plot 3: Strongly Convex (quadratic growth)
    y_strong = x**2 + x**4
    axes[2].plot(x, y_strong, 'b-', linewidth=2, label=r'$f(x) = x^2 + x^4$')
    axes[2].plot([x1, x2], [x1**2 + x1**4, x2**2 + x2**4], 'r--', alpha=0.5, linewidth=1.5)
    axes[2].plot([x1, x2], [x1**2 + x1**4, x2**2 + x2**4], 'ro', markersize=6)
    axes[2].set_ylim(-0.5, 5)
    axes[2].set_title('Strongly Convex Function', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$f(x)$')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend([r'$f(x)$', 'Secant'], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convexity_variants.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 2: Modulus of Convexity
# ============================================================================
def fig_modulus_of_convexity():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use ||·||^2 to illustrate modulus
    t = np.linspace(0, 2, 100)
    # For f(x) = ||x||^2, the exact modulus is phi(t) = t^2/4
    phi = t**2 / 4

    ax.plot(t, phi, 'b-', linewidth=2.5, label=r'$\phi(t) = \frac{t^2}{4}$ (exact modulus for $\|x\|^2$)')
    ax.fill_between(t, 0, phi, alpha=0.2, color='blue')

    # Add horizontal lines and annotations
    t_vals = [0.5, 1.0, 1.5]
    for t_val in t_vals:
        phi_val = t_val**2 / 4
        ax.plot([t_val, t_val], [0, phi_val], 'k--', alpha=0.3, linewidth=0.8)
        ax.plot(t_val, phi_val, 'ro', markersize=6)
        ax.text(t_val + 0.05, phi_val + 0.05, f'  $t={t_val}$', fontsize=9)

    ax.set_xlabel(r'$t = \|x - y\|$ (distance)', fontsize=12)
    ax.set_ylabel(r'$\phi(t)$ (gap measure)', fontsize=12)
    ax.set_title('Exact Modulus of Convexity: $f(x) = \\|x\\|^2$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modulus_of_convexity.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 3: Quasiconvex vs Convex
# ============================================================================
def fig_quasiconvex_vs_convex():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.linspace(0.01, 5, 200)

    # Quasiconvex but not convex: f(x) = sqrt(x)
    y_quasi = np.sqrt(x)
    axes[0].plot(x, y_quasi, 'b-', linewidth=2.5, label=r'$f(x) = \sqrt{x}$ (quasiconvex)')

    # Show level sets
    levels = [0.5, 1.0, 1.5, 2.0]
    for level in levels:
        x_level = level**2
        if x_level <= 5:
            axes[0].axvline(x_level, color='green', linestyle='--', alpha=0.4, linewidth=1)
            axes[0].fill_between(x, 0, 5, where=(x >= x_level), alpha=0.05, color='green')

    axes[0].set_xlabel('$x$', fontsize=11)
    axes[0].set_ylabel('$f(x)$', fontsize=11)
    axes[0].set_title('Quasiconvex: Level Sets are Convex', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 2.5)

    # Convex: f(x) = x^2
    y_conv = x**2
    axes[1].plot(x, y_conv, 'r-', linewidth=2.5, label=r'$f(x) = x^2$ (convex)')

    # Show some convex combinations
    x1, x2 = 1, 4
    y1, y2 = 1, 16
    alphas = [0.2, 0.5, 0.8]
    for alpha in alphas:
        x_m = alpha * x1 + (1 - alpha) * x2
        y_m_actual = x_m**2
        y_m_linear = alpha * y1 + (1 - alpha) * y2
        axes[1].plot(x_m, y_m_linear, 'g^', markersize=7, alpha=0.7)
        axes[1].plot(x_m, y_m_actual, 'b*', markersize=10)

    axes[1].plot([x1, x2], [y1, y2], 'k--', alpha=0.4, linewidth=1.5, label='Secant line')
    axes[1].plot([x1, x2], [y1, y2], 'ko', markersize=6)

    axes[1].set_xlabel('$x$', fontsize=11)
    axes[1].set_ylabel('$f(x)$', fontsize=11)
    axes[1].set_title('Convex: Epigraph is Convex', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quasiconvex_vs_convex.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 4: Norm Convexity in Different Spaces
# ============================================================================
def fig_norm_convexity():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    x = np.linspace(-2, 2, 300)

    # L1 norm: p=1, NOT uniformly convex
    y_l1 = np.abs(x)
    axes[0].plot(x, y_l1, 'b-', linewidth=2.5, label=r'$f(x) = |x|$ ($\ell^1$ norm)')
    axes[0].plot([-1.5, 1.5], [1.5, 1.5], 'r--', alpha=0.5, linewidth=1.5)
    axes[0].plot([-1.5, 1.5], [1.5, 1.5], 'ro', markersize=6)
    axes[0].set_xlabel('$x$', fontsize=11)
    axes[0].set_ylabel('$\\|x\\|_1$', fontsize=11)
    axes[0].set_title(r'$p=1$: Not Uniformly Convex', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 3)
    axes[0].legend(fontsize=10)

    # L2 norm: p=2, STRONGLY convex
    y_l2 = np.abs(x)**2
    axes[1].plot(x, y_l2, 'b-', linewidth=2.5, label=r'$f(x) = x^2$ ($\ell^2$ norm squared)')
    x1, x2 = -1.2, 1.5
    y1, y2 = x1**2, x2**2
    axes[1].plot([x1, x2], [y1, y2], 'r--', alpha=0.5, linewidth=1.5)
    axes[1].plot([x1, x2], [y1, y2], 'ro', markersize=6)
    axes[1].set_xlabel('$x$', fontsize=11)
    axes[1].set_ylabel('$\\|x\\|_2^2$', fontsize=11)
    axes[1].set_title(r'$p=2$: Strongly Convex with $\beta=2$', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 4)
    axes[1].legend(fontsize=10)

    # Lp norm: p>2, uniformly convex
    y_lp = np.abs(x)**3
    axes[2].plot(x, y_lp, 'b-', linewidth=2.5, label=r'$f(x) = |x|^3$ ($\ell^3$ norm)')
    axes[2].plot([x1, x2], [np.abs(x1)**3, np.abs(x2)**3], 'r--', alpha=0.5, linewidth=1.5)
    axes[2].plot([x1, x2], [np.abs(x1)**3, np.abs(x2)**3], 'ro', markersize=6)
    axes[2].set_xlabel('$x$', fontsize=11)
    axes[2].set_ylabel('$\\|x\\|_p$', fontsize=11)
    axes[2].set_title(r'$p>2$: Uniformly Convex', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 4)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'norm_convexity.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 5: Hierarchy of Convexity Classes
# ============================================================================
def fig_convexity_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 8))

    positions = {
        'Convex': (0.5, 0.2, 'lightblue'),
        'Strictly Convex': (0.5, 0.35, 'lightgreen'),
        'Uniformly Convex': (0.5, 0.5, 'lightyellow'),
        'Strongly Convex': (0.5, 0.65, 'lightcoral'),
        'Quasiconvex': (0.25, 0.2, 'lavender'),
        'Strictly Quasiconvex': (0.25, 0.35, 'lightsteelblue'),
        'Uniformly Quasiconvex': (0.25, 0.5, 'lightcyan'),
    }

    # Draw boxes
    box_width = 0.18
    box_height = 0.08

    for label, (x, y, color) in positions.items():
        rect = plt.Rectangle((x - box_width/2, y - box_height/2), box_width, box_height,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows for implications
    ax.arrow(0.5, 0.625, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.5, 0.475, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.5, 0.325, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')

    ax.arrow(0.25, 0.475, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.25, 0.325, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')

    ax.arrow(0.41, 0.2, -0.08, 0, head_width=0.02, head_length=0.025, fc='red', ec='red', linewidth=2)

    ax.text(0.5, 0.85, 'Hierarchy of Convexity Classes', fontsize=14, fontweight='bold', ha='center')
    ax.text(0.5, 0.78, 'Strong implications (top to bottom)', fontsize=11, ha='center', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convexity_hierarchy.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 6: Strong Convexity Example
# ============================================================================
def fig_strong_convexity_example():
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.linspace(-2, 2, 200)
    y = x**2

    ax.plot(x, y, 'b-', linewidth=3, label=r'$f(x) = x^2$ (strongly convex, $\beta=2$)')

    # Show the strong convexity inequality with parabola
    x1, x2 = -1.5, 1.2
    alpha = 0.3
    x_mid = alpha * x1 + (1 - alpha) * x2

    y1, y2 = x1**2, x2**2
    y_mid_actual = x_mid**2
    y_mid_linear = alpha * y1 + (1 - alpha) * y2

    # Plot secant line
    ax.plot([x1, x2], [y1, y2], 'k--', linewidth=2, alpha=0.6, label='Secant line')
    ax.plot([x1, x2], [y1, y2], 'ko', markersize=8)

    # Plot parabola (strong convexity gap)
    beta = 2
    x_para = np.linspace(x1, x2, 100)
    y_para = y_mid_linear - beta * alpha * (1 - alpha) * (x_para - x_mid)**2
    ax.fill_between(x_para, x_para**2, y_para, alpha=0.3, color='red',
                     label=r'Strong convexity gap: $\frac{\beta\alpha(1-\alpha)}{2}\|x-y\|^2$')

    ax.plot(x_mid, y_mid_actual, 'b*', markersize=15, label=f'$f({x_mid:.2f}) = {y_mid_actual:.2f}$')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title(r'Strong Convexity: $f(x) = x^2$ with $\beta = 2$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper center')
    ax.set_ylim(-0.5, 5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strong_convexity_example.pdf'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Run all figure generation
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 10...")

    fig_convexity_variants()
    print("  - convexity_variants.pdf")

    fig_modulus_of_convexity()
    print("  - modulus_of_convexity.pdf")

    fig_quasiconvex_vs_convex()
    print("  - quasiconvex_vs_convex.pdf")

    fig_norm_convexity()
    print("  - norm_convexity.pdf")

    fig_convexity_hierarchy()
    print("  - convexity_hierarchy.pdf")

    fig_strong_convexity_example()
    print("  - strong_convexity_example.pdf")

    print("\nAll figures generated successfully!")
