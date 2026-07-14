#!/usr/bin/env python3
"""
Generate figures for Chapter 5g: Fixed Point Theorems in Banach Algebra
and Lattice-Theoretic Fixed Point Theorems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import os

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_figure_params():
    """Set consistent figure parameters"""
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10

set_figure_params()

# Figure 1: Mapping hierarchy - Contractions vs Lipschitzian maps
def figure_mapping_hierarchy():
    """Create diagram showing mapping relationships"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define boxes
    boxes = [
        {'xy': (1, 7), 'label': 'Contraction\nmappings', 'color': '#FF6B6B'},
        {'xy': (6, 7), 'label': 'Lipschitzian\nmappings', 'color': '#4ECDC4'},
        {'xy': (1, 4), 'label': 'P-Lipschitzian\nmappings', 'color': '#45B7D1'},
        {'xy': (6, 4), 'label': 'D-Lipschitzian\nmappings', 'color': '#96CEB4'},
    ]

    for box in boxes:
        fancy_box = FancyBboxPatch(
            box['xy'], 2.5, 1.2,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=box['color'],
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(fancy_box)
        ax.text(box['xy'][0] + 1.25, box['xy'][1] + 0.6, box['label'],
                ha='center', va='center', fontsize=11, fontweight='bold')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(4.8, 7.3), xytext=(3.6, 7.3),
                arrowprops=arrow_props)
    ax.text(4.2, 7.7, 'weaker', ha='center', fontsize=10, style='italic')

    ax.annotate('', xy=(1.5, 5.8), xytext=(1.5, 6.9),
                arrowprops=arrow_props)
    ax.text(0.2, 6.3, 'special\ncase', ha='center', fontsize=9, style='italic')

    ax.annotate('', xy=(6.5, 5.8), xytext=(6.5, 6.9),
                arrowprops=arrow_props)
    ax.text(7.8, 6.3, 'includes', ha='center', fontsize=9, style='italic')

    ax.set_title('Hierarchy of Mapping Types', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'mapping_hierarchy.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2: Fixed point iteration visualization
def figure_fixed_point_iteration():
    """Visualize fixed point iteration for contraction mappings"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Contraction mapping example
    ax = axes[0]
    x = np.linspace(0, 1, 100)

    # Identity line
    ax.plot(x, x, 'k--', linewidth=2, label='$y = x$ (identity)')

    # Contraction mapping: f(x) = 0.5*x + 0.2
    f_x = 0.5 * x + 0.2
    ax.plot(x, f_x, 'b-', linewidth=2.5, label='$T(x) = 0.5x + 0.2$')

    # Fixed point
    fixed_point = 0.4
    ax.plot(fixed_point, fixed_point, 'ro', markersize=10, label='Fixed point', zorder=5)

    # Iteration visualization
    x_iter = [0.1]
    for i in range(5):
        y = 0.5 * x_iter[-1] + 0.2
        ax.plot([x_iter[-1], x_iter[-1]], [x_iter[-1], y], 'g-', alpha=0.5, linewidth=1)
        ax.plot([x_iter[-1], y], [y, y], 'g-', alpha=0.5, linewidth=1)
        x_iter.append(y)

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$T(x)$', fontsize=12)
    ax.set_title('Contraction Mapping Iteration', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right plot: Convergence rate
    ax = axes[1]
    iterations = np.arange(0, 8)

    # Error for different Lipschitz constants
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for alpha, color in zip(alpha_values, colors):
        error = alpha ** iterations
        ax.semilogy(iterations, error, 'o-', linewidth=2.5, markersize=8,
                   label=f'$\\alpha = {alpha}$', color=color)

    ax.set_xlabel('Iteration $n$', fontsize=12)
    ax.set_ylabel('Error $\\|x_n - x^*\\|$ (log scale)', fontsize=12)
    ax.set_title('Convergence: Effect of Lipschitz Constant', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'fixed_point_iteration.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3: Banach lattice structure
def figure_banach_lattice():
    """Visualize Banach lattice structure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Banach Lattice Structure', ha='center', fontsize=14, fontweight='bold')

    # Properties
    properties = [
        {'y': 8.5, 'text': 'Vector Space Structure', 'icon': '▼'},
        {'y': 7.8, 'text': 'Norm Structure', 'icon': '‖·‖'},
        {'y': 7.1, 'text': 'Lattice Order Structure', 'icon': '≤'},
        {'y': 6.4, 'text': 'Norm-Lattice Compatibility', 'icon': '⇔'},
    ]

    colors_prop = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, prop in enumerate(properties):
        # Background box
        rect = FancyBboxPatch(
            (0.5, prop['y'] - 0.35), 9,
            0.6,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=colors_prop[i],
            linewidth=1.5,
            alpha=0.6
        )
        ax.add_patch(rect)
        ax.text(1, prop['y'], prop['icon'], ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(2, prop['y'], prop['text'], ha='left', va='center', fontsize=11)

    # Key definitions box
    key_defs = [
        'Fixed point: $f(x^*) = x^*$',
        'Increasing function: $x ≤ y ⟹ f(x) ≤ f(y)$',
        'Lattice completeness: every subset has supremum and infimum'
    ]

    y_pos = 5.2
    ax.text(5, y_pos + 0.3, 'Key Definitions', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    for i, defn in enumerate(key_defs):
        ax.text(1, y_pos - i*0.5, f'• {defn}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'banach_lattice_structure.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4: Comparison of fixed point existence conditions
def figure_theorem_comparison():
    """Compare different fixed point theorems"""
    fig, ax = plt.subplots(figsize=(11, 7))

    theorems = [
        'Banach\nContraction',
        'Boyd-Wong\nGeneralization',
        'Kannan\nMapping',
        'P-Lipschitzian\n(Thm 5.183)',
        'Lattice-Theoretic\n(Tarski)',
    ]

    properties = ['Existence', 'Uniqueness', 'Contraction', 'Linear Ops', 'Lattice']

    # Existence conditions matrix (1 = satisfied, 0 = not required/satisfied)
    conditions = np.array([
        [1, 1, 1, 0, 0],  # Banach
        [1, 1, 1, 0, 0],  # Boyd-Wong
        [1, 1, 0, 0, 0],  # Kannan
        [1, 0, 0, 1, 0],  # P-Lipschitzian
        [1, 0, 0, 0, 1],  # Lattice-Theoretic
    ])

    im = ax.imshow(conditions, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(properties)))
    ax.set_yticks(np.arange(len(theorems)))
    ax.set_xticklabels(properties, fontsize=10)
    ax.set_yticklabels(theorems, fontsize=10)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(theorems)):
        for j in range(len(properties)):
            value = conditions[i, j]
            text = ax.text(j, i, '✓' if value else '✗',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    ax.set_title('Comparison of Fixed Point Theorem Conditions', fontsize=13, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Condition Satisfied', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'theorem_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5: Lipschitz constant effect on convergence
def figure_lipschitz_convergence():
    """Show how Lipschitz constant affects convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lipschitz_constants = [0.3, 0.5, 0.7, 0.95]

    for idx, (ax, alpha) in enumerate(zip(axes.flat, lipschitz_constants)):
        # Generate iterations
        x = np.linspace(0, 1, 100)
        f_x = alpha * x + (1 - alpha) * 0.5  # Function with Lipschitz constant alpha

        # Plot
        ax.plot(x, x, 'k--', linewidth=2, label='Identity')
        ax.plot(x, f_x, 'b-', linewidth=2.5, label=f'$T(x)$, $L={alpha}$')

        # Fixed point
        fixed_point = 0.5
        ax.plot(fixed_point, fixed_point, 'ro', markersize=10, zorder=5)

        # Iteration
        x_iter = [0.1]
        for i in range(8):
            y = alpha * x_iter[-1] + (1 - alpha) * 0.5
            if i < 5:  # Only show first few iterations for clarity
                ax.plot([x_iter[-1], x_iter[-1]], [x_iter[-1], y], 'g-', alpha=0.5, linewidth=1)
                ax.plot([x_iter[-1], y], [y, y], 'g-', alpha=0.5, linewidth=1)
            x_iter.append(y)

        ax.set_xlabel('$x$', fontsize=11)
        ax.set_ylabel('$T(x)$', fontsize=11)
        ax.set_title(f'Lipschitz Constant α = {alpha}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add convergence rate annotation
        if alpha < 1:
            ax.text(0.5, 0.05, f'Convergence Rate: $α^n$',
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        else:
            ax.text(0.5, 0.05, 'No Guaranteed Convergence',
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'lipschitz_convergence.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 6: Numerical example - operator equation solution
def figure_numerical_example():
    """Numerical example solving an operator equation"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Example: Solve x = Ax + b where A is P-Lipschitzian
    # Using fixed-point iteration

    # Left plot: Function behavior
    ax = axes[0]
    x = np.linspace(0, 1, 100)

    # Define T(x) = Ax + b where A is approximately 0.4-Lipschitzian
    A = 0.4
    b = 0.3
    T_x = A * x + b

    ax.plot(x, x, 'k--', linewidth=2, label='$y = x$')
    ax.plot(x, T_x, 'b-', linewidth=2.5, label='$T(x) = 0.4x + 0.3$')

    # Fixed point solution
    x_star = b / (1 - A)  # Analytical solution
    ax.plot(x_star, x_star, 'ro', markersize=12, label=f'$x^* ≈ {x_star:.4f}$', zorder=5)

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$T(x)$', fontsize=12)
    ax.set_title('Operator Equation Visualization', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right plot: Convergence history
    ax = axes[1]
    x_init = 0.0
    x_values = [x_init]

    for i in range(20):
        x_new = A * x_values[-1] + b
        x_values.append(x_new)

    errors = np.abs(np.array(x_values) - x_star)

    ax.semilogy(range(len(errors)), errors, 'bo-', linewidth=2, markersize=6, label='Error $|x_n - x^*|$')
    ax.set_xlabel('Iteration $n$', fontsize=12)
    ax.set_ylabel('Error (log scale)', fontsize=12)
    ax.set_title('Convergence History', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'numerical_example.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
if __name__ == '__main__':
    print("Generating figures for Chapter 5g...")

    figure_mapping_hierarchy()
    print("✓ mapping_hierarchy.pdf")

    figure_fixed_point_iteration()
    print("✓ fixed_point_iteration.pdf")

    figure_banach_lattice()
    print("✓ banach_lattice_structure.pdf")

    figure_theorem_comparison()
    print("✓ theorem_comparison.pdf")

    figure_lipschitz_convergence()
    print("✓ lipschitz_convergence.pdf")

    figure_numerical_example()
    print("✓ numerical_example.pdf")

    print("\nAll figures generated successfully!")
