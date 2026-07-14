#!/usr/bin/env python3
"""
Generate figures for Chapter 5a: Fixed Point Theorems
Illustrations for metric spaces, contractive mappings, and convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'warning': '#d62728'
}

def set_common_style(ax):
    """Apply common style to axes"""
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Figure 1: Contractive Mapping Illustration
def fig_contractive_mapping():
    """Illustrate a contractive mapping on the real line"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Fixed point of a contractive map T(x) = 0.5*x + 1
    x = np.linspace(-2, 6, 1000)
    Tx = 0.5 * x + 1

    ax1.plot(x, x, 'k--', linewidth=1.5, label='$y = x$', alpha=0.7)
    ax1.plot(x, Tx, linewidth=2.5, color=colors['primary'], label='$T(x) = 0.5x + 1$')

    # Mark fixed point
    fixed_point = 2
    ax1.plot(fixed_point, fixed_point, 'o', markersize=12, color=colors['warning'],
             label='Fixed point', zorder=5)
    ax1.axvline(fixed_point, color=colors['warning'], linestyle=':', alpha=0.5)
    ax1.axhline(fixed_point, color=colors['warning'], linestyle=':', alpha=0.5)

    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$T(x)$', fontsize=12)
    ax1.set_title('Contractive Mapping: $T(x) = 0.5x + 1$', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-0.5, 5)
    set_common_style(ax1)

    # Right: Iteration sequence showing convergence
    x0 = 5
    n_iter = 20
    x_vals = [x0]
    for i in range(n_iter):
        x_vals.append(0.5 * x_vals[-1] + 1)

    iterations = np.arange(len(x_vals))
    ax2.plot(iterations, x_vals, 'o-', linewidth=2, markersize=6,
             color=colors['primary'], label='Iteration sequence')
    ax2.axhline(fixed_point, color=colors['warning'], linestyle='--',
                linewidth=2, label=f'Fixed point = {fixed_point}', alpha=0.8)

    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel('$x^{(n)}$', fontsize=12)
    ax2.set_title('Convergence: Starting from $x_0 = 5$', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    set_common_style(ax2)

    plt.tight_layout()
    plt.savefig('figures/fig_contractive_mapping.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2: Banach Fixed Point Theorem
def fig_banach_theorem():
    """Illustration of Banach contraction principle"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw different contraction coefficients
    x = np.linspace(0, 5, 1000)

    contractions = [
        (0.3, '$\\alpha = 0.3$ (strong)'),
        (0.6, '$\\alpha = 0.6$ (moderate)'),
        (0.9, '$\\alpha = 0.9$ (weak)')
    ]

    color_idx = [colors['accent'], colors['primary'], colors['secondary']]

    for (alpha, label), color in zip(contractions, color_idx):
        y = alpha * x + 2
        ax.plot(x, y, linewidth=2.5, label=label, color=color)

    ax.plot(x, x, 'k--', linewidth=1.5, label='$y = x$', alpha=0.7)

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$T(x) = \\alpha x + 2$', fontsize=12)
    ax.set_title('Banach Contraction Principle: Different Contraction Coefficients',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    set_common_style(ax)

    plt.tight_layout()
    plt.savefig('figures/fig_banach_theorem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3: Metric Space Distance
def fig_metric_space():
    """Visualize metric space properties"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Metric properties illustration
    # 1. Non-negativity: d(x,y) >= 0
    ax1.scatter([1, 2], [2, 4], s=200, color=colors['primary'], alpha=0.7, zorder=3)
    ax1.text(1, 1.7, '$x$', fontsize=14, ha='center')
    ax1.text(2, 4.3, '$y$', fontsize=14, ha='center')
    ax1.annotate('', xy=(2, 4), xytext=(1, 2),
                arrowprops=dict(arrowstyle='<->', lw=2, color=colors['secondary']))
    ax1.text(1.4, 3.2, '$d(x,y) \\geq 0$', fontsize=12, rotation=45)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 5)
    ax1.set_title('Non-negativity', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # 2. Identity: d(x,x) = 0
    ax2.scatter([2], [2], s=200, color=colors['primary'], alpha=0.7, zorder=3)
    circle = Circle((2, 2), 0.3, fill=False, edgecolor=colors['accent'], linewidth=2)
    ax2.add_patch(circle)
    ax2.text(2, 1.5, '$x$', fontsize=14, ha='center')
    ax2.text(1.2, 3, '$d(x,x) = 0$', fontsize=12)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.set_title('Identity of Indiscernibles', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # 3. Symmetry: d(x,y) = d(y,x)
    ax3.scatter([1, 3], [2, 2], s=200, color=colors['primary'], alpha=0.7, zorder=3)
    ax3.annotate('', xy=(3, 2), xytext=(1, 2),
                arrowprops=dict(arrowstyle='<->', lw=2, color=colors['secondary']))
    ax3.text(1, 1.5, '$x$', fontsize=14, ha='center')
    ax3.text(3, 1.5, '$y$', fontsize=14, ha='center')
    ax3.text(2, 2.6, '$d(x,y) = d(y,x)$', fontsize=12, ha='center')
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 4)
    ax3.set_title('Symmetry', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # 4. Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
    x_pts = [1, 2, 3]
    y_pts = [1, 3, 1]
    ax4.scatter(x_pts, y_pts, s=200, color=colors['primary'], alpha=0.7, zorder=3)
    ax4.plot([1, 2, 3], [1, 3, 1], 'b-', linewidth=1.5, alpha=0.5)
    ax4.annotate('', xy=(3, 1), xytext=(1, 1),
                arrowprops=dict(arrowstyle='<->', lw=2, color=colors['secondary']))
    ax4.text(1, 0.5, '$x$', fontsize=14, ha='center')
    ax4.text(2, 3.4, '$y$', fontsize=14, ha='center')
    ax4.text(3, 0.5, '$z$', fontsize=14, ha='center')
    ax4.text(2, 1.8, '$d(x,y)+d(y,z)$', fontsize=11, ha='center', style='italic')
    ax4.text(2, -0.2, '$d(x,z)$', fontsize=11, ha='center', fontweight='bold')
    ax4.set_xlim(0, 4)
    ax4.set_ylim(-0.5, 4)
    ax4.set_title('Triangle Inequality', fontsize=12, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig_metric_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4: Fixed Point Theorems Comparison
def fig_theorems_comparison():
    """Comparison of different fixed point theorems"""
    fig, ax = plt.subplots(figsize=(11, 7))

    theorems = [
        'Brouwer\n(compact convex)',
        'Schauder\n(compact, convex)',
        'Banach\n(complete metric)',
        'Browder\n(uniformly convex)',
        'Kirk\n(hyperbolic space)'
    ]

    properties = {
        'Continuity': [1, 1, 0.8, 1, 1],
        'Compactness': [1, 1, 0, 0.8, 1],
        'Convexity': [1, 1, 0, 1, 0.5],
        'Contraction': [0, 0, 1, 0.5, 0.6],
        'Nonexpansive': [0.2, 0.2, 0, 1, 1],
    }

    x = np.arange(len(theorems))
    width = 0.15

    colors_props = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (prop, values) in enumerate(properties.items()):
        ax.bar(x + i*width, values, width, label=prop, color=colors_props[i], alpha=0.8)

    ax.set_xlabel('Theorem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Applicability', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Fixed Point Theorems', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(theorems, fontsize=10)
    ax.legend(fontsize=10, loc='upper right', ncol=1)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig_theorems_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5: Convergence Behavior
def fig_convergence_behavior():
    """Show different convergence behaviors"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    n = np.arange(0, 50)

    # Linear convergence
    e_linear = 0.9 ** n
    ax1.semilogy(n, e_linear, 'o-', linewidth=2, markersize=5,
                 color=colors['primary'], label='$e_n = 0.9^n$')
    ax1.set_title('Linear Convergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration $n$', fontsize=11)
    ax1.set_ylabel('Error $\\|e_n\\|$ (log scale)', fontsize=11)
    ax1.legend()
    set_common_style(ax1)
    ax1.grid(True, alpha=0.3)

    # Quadratic convergence
    e_quad = 0.5 ** (2**n)
    ax2.semilogy(n[:15], e_quad[:15], 'o-', linewidth=2, markersize=5,
                 color=colors['secondary'], label='$e_n = 0.5^{2^n}$')
    ax2.set_title('Quadratic Convergence', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('Error $\\|e_n\\|$ (log scale)', fontsize=11)
    ax2.legend()
    set_common_style(ax2)
    ax2.grid(True, alpha=0.3)

    # Superlinear convergence
    e_super = 0.8 ** (n * 1.5)
    ax3.semilogy(n, e_super, 'o-', linewidth=2, markersize=5,
                 color=colors['accent'], label='$e_n = 0.8^{1.5n}$')
    ax3.set_title('Superlinear Convergence', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Iteration $n$', fontsize=11)
    ax3.set_ylabel('Error $\\|e_n\\|$ (log scale)', fontsize=11)
    ax3.legend()
    set_common_style(ax3)
    ax3.grid(True, alpha=0.3)

    # Comparison of all three
    ax4.semilogy(n, e_linear, 'o-', label='Linear', color=colors['primary'],
                 markersize=4, alpha=0.8, linewidth=1.5)
    ax4.semilogy(n[:15], e_quad[:15], 's-', label='Quadratic', color=colors['secondary'],
                 markersize=4, alpha=0.8, linewidth=1.5)
    ax4.semilogy(n, e_super, '^-', label='Superlinear', color=colors['accent'],
                 markersize=4, alpha=0.8, linewidth=1.5)
    ax4.set_title('Convergence Rate Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Iteration $n$', fontsize=11)
    ax4.set_ylabel('Error $\\|e_n\\|$ (log scale)', fontsize=11)
    ax4.legend(fontsize=10)
    set_common_style(ax4)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_convergence_behavior.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 6: Iteration Diagram (Cobweb)
def fig_cobweb_diagram():
    """Cobweb diagram showing fixed point iteration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Contractive case
    x = np.linspace(0, 3, 1000)
    Tx = 0.6 * x + 0.5

    x0 = 2.5
    x_seq = [x0]
    for _ in range(15):
        x_seq.append(0.6 * x_seq[-1] + 0.5)

    ax1.plot(x, x, 'k--', linewidth=1.5, label='$y=x$', alpha=0.7)
    ax1.plot(x, Tx, linewidth=2, color=colors['primary'], label='$T(x)$')

    # Draw cobweb
    for i in range(len(x_seq)-1):
        ax1.plot([x_seq[i], x_seq[i]], [x_seq[i], Tx[np.argmin(np.abs(x - x_seq[i]))]],
                'r--', alpha=0.3, linewidth=0.8)
        ax1.plot([x_seq[i], x_seq[i+1]], [Tx[np.argmin(np.abs(x - x_seq[i]))], x_seq[i+1]],
                'r--', alpha=0.3, linewidth=0.8)

    ax1.plot(x_seq, 'o-', markersize=4, linewidth=1, color=colors['secondary'],
            alpha=0.6, label='Iteration sequence')
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_title('Contractive: Converges to Fixed Point', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)
    set_common_style(ax1)

    # Non-contractive case
    Tx2 = 1.5 * x - 0.5
    x2 = np.linspace(-1, 3, 1000)
    Tx2_eval = 1.5 * x2 - 0.5

    x0_2 = 1
    x_seq2 = [x0_2]
    for _ in range(8):
        x_seq2.append(1.5 * x_seq2[-1] - 0.5)

    ax2.plot(x2, x2, 'k--', linewidth=1.5, label='$y=x$', alpha=0.7)
    ax2.plot(x2, Tx2_eval, linewidth=2, color=colors['primary'], label='$T(x)$')

    ax2.plot(x_seq2[:4], x_seq2[:4], 'o-', markersize=4, linewidth=1,
            color=colors['secondary'], alpha=0.6, label='Iteration sequence')

    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_title('Non-contractive: Diverges', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-2, 4)
    set_common_style(ax2)

    plt.tight_layout()
    plt.savefig('figures/fig_cobweb_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == '__main__':
    print("Generating figures for Chapter 5a: Fixed Point Theorems...")

    fig_contractive_mapping()
    print("✓ Generated fig_contractive_mapping.pdf")

    fig_banach_theorem()
    print("✓ Generated fig_banach_theorem.pdf")

    fig_metric_space()
    print("✓ Generated fig_metric_space.pdf")

    fig_theorems_comparison()
    print("✓ Generated fig_theorems_comparison.pdf")

    fig_convergence_behavior()
    print("✓ Generated fig_convergence_behavior.pdf")

    fig_cobweb_diagram()
    print("✓ Generated fig_cobweb_diagram.pdf")

    print("\nAll figures generated successfully!")
