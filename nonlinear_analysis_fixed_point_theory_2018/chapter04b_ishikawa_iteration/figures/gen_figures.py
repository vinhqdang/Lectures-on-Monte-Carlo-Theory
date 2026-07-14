#!/usr/bin/env python3
"""
Generate figures for Chapter 4b: Ishikawa Iteration and Common Fixed Point Theorems
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def setup_figure(title=""):
    fig, ax = plt.subplots(figsize=(10, 7))
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    return fig, ax

# ============================================================================
# Figure 1: Fixed Point Iteration Scheme
# ============================================================================
def create_iteration_scheme():
    fig, ax = setup_figure("Fixed Point Iteration Schemes")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9, "One-Step vs Two-Step Schemes", fontsize=13, fontweight='bold', ha='center')

    # One-step (Picard)
    ax.text(2.5, 7.5, "One-Step (Picard):", fontsize=11, fontweight='bold')
    ax.text(2.5, 6.8, r"$x_{n+1} = Tx_n$", fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Two-step (Ishikawa)
    ax.text(7.5, 7.5, "Two-Step (Ishikawa):", fontsize=11, fontweight='bold')
    ax.text(7.5, 6.8, r"$y_n = (1-\alpha_n)x_n + \alpha_n Tx_n$", fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(7.5, 5.9, r"$x_{n+1} = (1-\beta_n)x_n + \beta_n Ty_n$", fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Convergence rates
    ax.text(5, 3.8, "Convergence Characteristics", fontsize=12, fontweight='bold', ha='center')

    # One-step characteristics
    ax.text(2.5, 2.8, "Picard:", fontsize=10, fontweight='bold')
    ax.text(2.5, 2.2, "• Faster convergence", fontsize=9)
    ax.text(2.5, 1.7, "• Works for contractions", fontsize=9)
    ax.text(2.5, 1.2, "• Limited to certain mappings", fontsize=9)

    # Two-step characteristics
    ax.text(7.5, 2.8, "Ishikawa:", fontsize=10, fontweight='bold')
    ax.text(7.5, 2.2, "• More flexible", fontsize=9)
    ax.text(7.5, 1.7, "• Works for nonexpansive maps", fontsize=9)
    ax.text(7.5, 1.2, "• Controlled by parameters", fontsize=9)

    plt.tight_layout()
    plt.savefig('01_iteration_schemes.pdf', bbox_inches='tight', dpi=300)
    print("* Created 01_iteration_schemes.pdf")
    plt.close()

# ============================================================================
# Figure 2: Convergence Illustration
# ============================================================================
def create_convergence_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Simulation parameters
    np.random.seed(42)
    x0 = 0.1
    iterations = 50

    # One-step: Picard iteration
    x_picard = [x0]
    x = x0
    for i in range(iterations):
        x = 0.5 * np.sin(x) + 0.5  # Fixed point near 0.7
        x_picard.append(x)

    # Two-step: Ishikawa iteration
    x_ishikawa = [x0]
    x = x0
    alpha = 0.3
    beta = 0.4
    for i in range(iterations):
        y = (1 - alpha) * x + alpha * (0.5 * np.sin(x) + 0.5)
        x = (1 - beta) * x + beta * (0.5 * np.sin(y) + 0.5)
        x_ishikawa.append(x)

    # Plot Picard iteration
    ax1.semilogy(range(len(x_picard)), np.abs(np.array(x_picard) - 0.766), 'b-o', markersize=3, label='Picard')
    ax1.set_xlabel('Iteration n', fontsize=11)
    ax1.set_ylabel(r'$|x_n - x^*|$ (log scale)', fontsize=11)
    ax1.set_title('One-Step Iteration (Picard)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Ishikawa iteration
    ax2.semilogy(range(len(x_ishikawa)), np.abs(np.array(x_ishikawa) - 0.766), 'r-s', markersize=3, label='Ishikawa')
    ax2.set_xlabel('Iteration n', fontsize=11)
    ax2.set_ylabel(r'$|x_n - x^*|$ (log scale)', fontsize=11)
    ax2.set_title('Two-Step Iteration (Ishikawa)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('02_convergence_comparison.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 02_convergence_comparison.pdf")
    plt.close()

# ============================================================================
# Figure 3: Nonexpansive Mapping
# ============================================================================
def create_nonexpansive_illustration():
    fig, ax = setup_figure("Nonexpansive Mapping Property")
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')

    # Draw domain
    circle1 = Circle((1, 1), 1.5, fill=False, edgecolor='blue', linewidth=2, label='Domain K')
    ax.add_patch(circle1)
    ax.text(1, -0.3, 'K', fontsize=12, fontweight='bold', ha='center', color='blue')

    # Draw range
    circle2 = Circle((3, 1), 1.2, fill=False, edgecolor='green', linewidth=2, label='Range T(K)')
    ax.add_patch(circle2)
    ax.text(3, -0.3, 'T(K)', fontsize=12, fontweight='bold', ha='center', color='green')

    # Draw arrow representing mapping
    arrow = FancyArrowPatch((1.8, 1.3), (2.2, 1.3),
                           arrowstyle='->', mutation_scale=20,
                           color='black', linewidth=2)
    ax.add_patch(arrow)
    ax.text(2, 1.6, 'T', fontsize=12, fontweight='bold')

    # Add property box
    textstr = 'Nonexpansive Condition:\n' + r'$\|Tx - Ty\| \leq \|x - y\|$ for all $x, y \in K$'
    ax.text(2.5, 3.5, textstr, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='center')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('03_nonexpansive_mapping.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 03_nonexpansive_mapping.pdf")
    plt.close()

# ============================================================================
# Figure 4: Fixed Point Theorem Structure
# ============================================================================
def create_theorem_structure():
    fig, ax = setup_figure("Fixed Point Theorem Framework")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main theorem box
    boxes = [
        {'xy': (0.5, 8), 'width': 9, 'height': 1.2, 'text': 'Fixed Point Theorem',
         'color': 'lightblue'},
        {'xy': (0.5, 6.2), 'width': 4.3, 'height': 1.4, 'text': 'Conditions:\n• Metric Space\n• Mapping Properties',
         'color': 'lightgreen'},
        {'xy': (5.2, 6.2), 'width': 4.3, 'height': 1.4, 'text': 'Conclusion:\n• Fixed Point Exists\n• Fixed Point Unique',
         'color': 'lightyellow'},
        {'xy': (0.5, 3.8), 'width': 4.3, 'height': 1.8, 'text': 'Example Types:\n• Contraction\n• Nonexpansive\n• Kannan',
         'color': 'lightcoral'},
        {'xy': (5.2, 3.8), 'width': 4.3, 'height': 1.8, 'text': 'Application:\n• Find Solution\n• Use Iteration\n• Compute Numerically',
         'color': 'plum'},
    ]

    for box in boxes:
        fancy_box = FancyBboxPatch(box['xy'], box['width'], box['height'],
                                   boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor=box['color'],
                                   linewidth=2, alpha=0.7)
        ax.add_patch(fancy_box)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
               box['text'], fontsize=10, ha='center', va='center',
               fontweight='bold' if 'Theorem' in box['text'] else 'normal')

    # Arrows
    ax.annotate('', xy=(2.2, 6.2), xytext=(2.2, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(7.4, 6.2), xytext=(7.4, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(2.2, 3.8), xytext=(2.2, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(7.4, 3.8), xytext=(7.4, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    plt.tight_layout()
    plt.savefig('04_theorem_structure.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 04_theorem_structure.pdf")
    plt.close()

# ============================================================================
# Figure 5: Parameter Effect on Convergence
# ============================================================================
def create_parameter_sensitivity():
    fig, ax = setup_figure("Parameter Effect on Ishikawa Iteration")

    alphas = np.linspace(0.1, 0.9, 5)
    betas = np.linspace(0.1, 0.9, 5)

    x0 = 0.1
    T_func = lambda x: 0.5 * np.sin(x) + 0.5
    target = 0.766

    for alpha in [0.3, 0.6, 0.9]:
        errors = []
        for beta in np.linspace(0.1, 0.9, 15):
            x = x0
            error = abs(x - target)
            for _ in range(50):
                y = (1 - alpha) * x + alpha * T_func(x)
                x = (1 - beta) * x + beta * T_func(y)
                error = abs(x - target)
            errors.append(error)

        ax.plot(np.linspace(0.1, 0.9, 15), errors, 'o-',
               label=f'α = {alpha:.1f}', linewidth=2, markersize=5)

    ax.set_xlabel('Parameter β', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Error |x₅₀ - x*|', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_parameter_sensitivity.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 05_parameter_sensitivity.pdf")
    plt.close()

# ============================================================================
# Figure 6: b-Metric Space Illustration
# ============================================================================
def create_bmetric_illustration():
    fig, ax = setup_figure("b-Metric Space (s ≥ 1)")
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')

    # Three points
    points = {'x': (1, 1), 'y': (3, 2), 'z': (4, 0.5)}
    colors = {'x': 'red', 'y': 'blue', 'z': 'green'}

    for label, (px, py) in points.items():
        ax.plot(px, py, 'o', markersize=12, color=colors[label])
        ax.text(px, py + 0.4, label, fontsize=14, fontweight='bold', ha='center')

    # Draw "distances"
    ax.plot([points['x'][0], points['y'][0]], [points['x'][1], points['y'][1]],
           'k--', alpha=0.5, linewidth=1.5)
    ax.plot([points['y'][0], points['z'][0]], [points['y'][1], points['z'][1]],
           'k--', alpha=0.5, linewidth=1.5)
    ax.plot([points['x'][0], points['z'][0]], [points['x'][1], points['z'][1]],
           'k-', alpha=0.7, linewidth=2)

    # Add distance labels
    ax.text(2, 1.7, 'd(x,y)', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(3.5, 1, 'd(y,z)', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2.2, 0.1, 'd(x,z)', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # b-metric property
    textstr = 'b-Metric Property:\nd(x,z) ≤ s[d(x,y) + d(y,z)]\nfor some s ≥ 1'
    ax.text(4.5, 4, textstr, fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           ha='left', family='monospace')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('06_bmetric_space.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 06_bmetric_space.pdf")
    plt.close()

# ============================================================================
# Figure 7: Convergence Rate Comparison
# ============================================================================
def create_convergence_rates():
    fig, ax = setup_figure("Convergence Rates Comparison")

    n = np.arange(1, 31)

    # Different convergence rates
    linear = 0.9 ** n
    quadratic = 0.9 ** (n**2 / 10)
    superlinear = 0.9 ** (1.5 * n)

    ax.semilogy(n, linear, 'o-', linewidth=2, markersize=4, label='Linear: 0.9ⁿ')
    ax.semilogy(n, quadratic, 's-', linewidth=2, markersize=4, label='Quadratic-like')
    ax.semilogy(n, superlinear, '^-', linewidth=2, markersize=4, label='Superlinear: 0.9^(1.5n)')

    ax.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error |eₙ| (log scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('07_convergence_rates.pdf', bbox_inches='tight', dpi=300)
    print("✓ Created 07_convergence_rates.pdf")
    plt.close()

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 4b: Ishikawa Iteration...\n")

    create_iteration_scheme()
    create_convergence_plot()
    create_nonexpansive_illustration()
    create_theorem_structure()
    create_parameter_sensitivity()
    create_bmetric_illustration()
    create_convergence_rates()

    print("\n✓ All figures generated successfully!")
