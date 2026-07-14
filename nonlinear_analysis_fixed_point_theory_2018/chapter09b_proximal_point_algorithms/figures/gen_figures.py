#!/usr/bin/env python3
"""
Generate figures for Proximal Point Algorithms chapter
Includes convergence analysis, algorithm flowcharts, and numerical examples
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

def setup_figure(figname, figsize=(10, 6)):
    """Setup a figure with consistent styling"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

# Figure 1: Classic Proximal Point Algorithm Convergence
def fig_proximal_convergence():
    """Convergence of proximal point algorithm for maximal monotone operator"""
    fig, ax = setup_figure('fig_proximal_convergence')

    n = 100
    # Simulate convergence of a proximal algorithm
    # x_{n+1} = (I + r_n T)^{-1} x_n

    # Example: T x = x (simple case for visualization)
    x_init = 5.0
    r_seq = [0.5 * (1 - 0.05*i) for i in range(n)]  # r_n decreasing

    x_vals = [x_init]
    for i in range(n-1):
        # Approximate (I + r_n T)^{-1} for T x = x
        r_n = max(0.01, r_seq[i])  # Ensure positive parameter
        x_new = x_vals[-1] / (1 + r_n)
        x_vals.append(x_new)

    ax.semilogy(range(n), np.abs(np.array(x_vals)), 'o-', linewidth=2.5,
                markersize=4, color=COLORS[0], label=r'$\|x_n\|$')
    ax.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\|x_n\|$ (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence of Proximal Point Algorithm\n' +
                 r'$x_{n+1} = (I + r_n T)^{-1} x_n$',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/fig_proximal_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_proximal_convergence.pdf")

# Figure 2: Strong vs Weak Convergence
def fig_convergence_types():
    """Illustrate difference between weak and strong convergence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Strong convergence
    t = np.linspace(0, 1, 50)
    for i in range(5):
        offset = i * 0.2
        y = np.exp(-3*t) * (1 + 0.3*offset)
        ax1.plot(t, y, 'o-', markersize=3, alpha=0.7, color=COLORS[0])

    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Limit point')
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'$\|x_n - x^*\|$', fontsize=11, fontweight='bold')
    ax1.set_title('Strong Convergence\n' + r'$x_n \to x^*$ in norm',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Weak convergence (oscillating)
    t = np.linspace(0, 1, 50)
    for i in range(5):
        offset = i * 0.2
        y = np.sin(2*np.pi*t) * np.exp(-2*t) * (1 + 0.3*offset)
        ax2.plot(t, y, 'o-', markersize=3, alpha=0.7, color=COLORS[1])

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Weak limit')
    ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'$f(x_n) - f(x^*)$', fontsize=11, fontweight='bold')
    ax2.set_title('Weak Convergence\n' + r'$x_n \rightharpoonup x^*$ (weakly)',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_convergence_types.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_convergence_types.pdf")

# Figure 3: Algorithm Evolution
def fig_algorithm_evolution():
    """Timeline of proximal algorithms development"""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Timeline data
    years = [1962, 1976, 1976, 2001, 2004, 2012]
    names = ['Martinet\nProximal Point\nAlgorithm',
             'Rockafellar\nGeneral Theory\n(Hilbert)',
             'Multiple\nAuthors\nConvergence Studies',
             'Kamimura &\nTakahashi\nBanach Spaces',
             'Solodov &\nSvaiter\nStrong Convergence',
             'Pathak & Cho\nStrong Convergence\nHilbert Spaces']
    y_pos = [1, 0.5, -0.5, 1, 0.5, -0.5]

    # Plot timeline
    ax.plot(years, [0]*len(years), 'k-', linewidth=3)

    colors_timeline = [COLORS[i % len(COLORS)] for i in range(len(years))]
    for year, name, y, color in zip(years, names, y_pos, colors_timeline):
        ax.plot(year, 0, 'o', markersize=12, color=color)
        ax.text(year, y, name, ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7))

    ax.set_xlim(1960, 2015)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_title('Historical Development of Proximal Point Algorithms',
                fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('figures/fig_algorithm_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_algorithm_evolution.pdf")

# Figure 4: Rate of Convergence Comparison
def fig_convergence_rates():
    """Compare convergence rates for different algorithms"""
    fig, ax = setup_figure('fig_convergence_rates', figsize=(11, 6))

    n = np.arange(1, 31)

    # Different convergence rates
    linear = 0.9**n
    superlinear = 0.5**n
    quadratic = 0.5**(2**n)

    ax.semilogy(n, linear, 'o-', linewidth=2.5, markersize=5,
               color=COLORS[0], label='Linear: $r^n$ (r < 1)')
    ax.semilogy(n, superlinear, 's-', linewidth=2.5, markersize=5,
               color=COLORS[1], label='Superlinear: $c^{2^n}$')
    ax.semilogy(n, quadratic, '^-', linewidth=2.5, markersize=5,
               color=COLORS[2], label='Quadratic: $r^{n^2}$')

    ax.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error $\|x_n - x^*\|$ (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Rate Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, 31)

    plt.tight_layout()
    plt.savefig('figures/fig_convergence_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_convergence_rates.pdf")

# Figure 5: Banach Space Properties
def fig_banach_space_hierarchy():
    """Show hierarchy of Banach space properties"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define boxes with hierarchy
    boxes = [
        {'pos': (1, 8), 'text': 'General Banach Space', 'color': COLORS[0]},
        {'pos': (1, 6), 'text': 'Reflexive Banach Space', 'color': COLORS[1]},
        {'pos': (1, 4), 'text': 'Uniformly Convex\nBanach Space', 'color': COLORS[2]},
        {'pos': (1, 2), 'text': 'Hilbert Space\n(Complete Inner Product)', 'color': COLORS[3]},
        {'pos': (5.5, 6), 'text': 'Smooth\nBanach Space', 'color': COLORS[1]},
        {'pos': (5.5, 4), 'text': 'Strictly Convex\nBanach Space', 'color': COLORS[2]},
    ]

    for box in boxes:
        fancy_box = FancyBboxPatch((box['pos'][0]-0.8, box['pos'][1]-0.35),
                                  1.6, 0.7, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=box['color'],
                                  alpha=0.7, linewidth=2)
        ax.add_patch(fancy_box)
        ax.text(box['pos'][0], box['pos'][1], box['text'],
               ha='center', va='center', fontsize=10, fontweight='bold')

    # Add arrows showing implications
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(1, 5.7), xytext=(1, 7.65), arrowprops=arrow_props)
    ax.annotate('', xy=(1, 3.7), xytext=(1, 5.65), arrowprops=arrow_props)
    ax.annotate('', xy=(1, 1.7), xytext=(1, 3.65), arrowprops=arrow_props)

    ax.text(3, 7.5, 'Proximal algorithms\nrequire reflexivity\nfor convergence',
           fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', alpha=0.8))

    ax.set_title('Banach Space Hierarchy and Proximal Algorithm Requirements',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/fig_banach_hierarchy.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_banach_hierarchy.pdf")

# Figure 6: Monotone Operator Examples
def fig_monotone_operators():
    """Visualize examples of monotone operators"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example 1: Positive definite matrix
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 100)
    y = x**3  # Monotone odd function
    ax.plot(x, y, 'o-', linewidth=2.5, color=COLORS[0], markersize=3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.3, color=COLORS[0])
    ax.set_title(r'Monotone: $\langle T(x) - T(y), x - y \rangle \geq 0$',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('T(x)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Example 2: Subdifferential
    ax = axes[0, 1]
    x = np.linspace(-3, 3, 100)
    y = np.abs(x)  # Absolute value (convex)
    ax.plot(x, y, 'o-', linewidth=2.5, color=COLORS[1], markersize=3)
    ax.set_title(r'Subdifferential of $f(x) = |x|$',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Example 3: Maximal Monotone
    ax = axes[1, 0]
    x = np.linspace(-2, 2, 50)
    y = np.tanh(x)  # Bounded monotone
    ax.plot(x, y, 'o-', linewidth=2.5, color=COLORS[2], markersize=3)
    ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax.set_title(r'Maximal Monotone: Bounded Range',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('T(x)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Example 4: Resolvent
    ax = axes[1, 1]
    r_vals = np.linspace(0.1, 3, 50)
    # Resolvent norm behavior
    resolvent_error = 1.0 / (1 + r_vals)
    ax.plot(r_vals, resolvent_error, 'o-', linewidth=2.5,
           color=COLORS[3], markersize=4)
    ax.set_title(r'Resolvent $(I + rT)^{-1}$ Error vs r',
                fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Parameter $r$', fontsize=10)
    ax.set_ylabel(r'$\|(I+rT)^{-1}x - x^*\|$', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_monotone_operators.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_monotone_operators.pdf")

# Figure 7: Algorithm Flowchart
def fig_algorithm_flowchart():
    """Create flowchart for proximal point algorithm"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def draw_box(ax, x, y, width, height, text, color):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                           boxstyle="round,pad=0.1", edgecolor='black',
                           facecolor=color, alpha=0.8, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='black')
        ax.add_patch(arrow)

    # Flowchart
    y_start = 11
    draw_box(ax, 5, y_start, 2, 0.7, 'Start', COLORS[0])
    draw_arrow(ax, 5, y_start-0.35, 5, y_start-0.9)

    y = y_start - 1.3
    draw_box(ax, 5, y, 3, 0.8, r'Choose $x_0 \in X$, $r > 0$', COLORS[1])
    draw_arrow(ax, 5, y-0.4, 5, y-0.95)

    y -= 1.3
    draw_box(ax, 5, y, 3, 1, r'$0 \in v_n + \frac{1}{r}(y_n - x_n)$' + '\n' +
            r'$v_n \in Tx_n$', COLORS[2])
    draw_arrow(ax, 5, y-0.5, 5, y-1.05)

    y -= 1.3
    draw_box(ax, 5, y, 3.5, 1, r'Define $H_n, W_n$ half-spaces' + '\n' +
            r'based on inclusion', COLORS[3])
    draw_arrow(ax, 5, y-0.5, 5, y-1.05)

    y -= 1.3
    draw_box(ax, 5, y, 3, 0.8, r'$x_{n+1} = R_{H_n \cap W_n}x_0$' + '\n' +
            '(projection)', COLORS[4])
    draw_arrow(ax, 5, y-0.4, 5, y-0.95)

    y -= 1.3
    draw_box(ax, 5, y, 2.5, 0.8, 'Check Convergence\nCriteria', COLORS[1])
    draw_arrow(ax, 3.75, y, 2, y)
    draw_arrow(ax, 6.25, y, 8, y)

    # Continue branch
    ax.text(2.5, y+0.3, 'No', fontsize=11, fontweight='bold')
    draw_arrow(ax, 2, y-0.4, 2, y-1.2)
    draw_arrow(ax, 2, y-1.2, 3.5, y-1.2)

    # Convergence branch
    ax.text(8.3, y+0.3, 'Yes', fontsize=11, fontweight='bold')
    draw_arrow(ax, 8, y-0.4, 8, y-1.2)

    y -= 1.8
    draw_box(ax, 8, y, 2.2, 0.7, 'Output x*', COLORS[0])

    ax.text(5, 1.2, r'Repeat until $\|x_{n+1} - x_n\| < \epsilon$',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    ax.set_title('Proximal Point Algorithm Flowchart',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/fig_algorithm_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_algorithm_flowchart.pdf")

# Figure 8: Parameter sensitivity
def fig_parameter_sensitivity():
    """Show sensitivity to parameter r_n"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fixed r
    n = np.arange(0, 50)
    r_fixed = 0.5
    convergence_fixed = np.exp(-0.3*n)
    ax1.semilogy(n, convergence_fixed, 'o-', linewidth=2.5, markersize=5,
                color=COLORS[0], label=f'Fixed r = {r_fixed}')
    ax1.set_xlabel('Iteration n', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'$\|x_n - x^*\|$', fontsize=11, fontweight='bold')
    ax1.set_title('Fixed Parameter: Slow Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)

    # Adaptive r
    r_adaptive = [0.5 + 0.1/(n+1) for n in range(50)]  # Decreasing
    convergence_adaptive = np.exp(-0.5*n) * np.power(0.95, n)
    ax2.semilogy(n, convergence_adaptive, 's-', linewidth=2.5, markersize=5,
                color=COLORS[1], label=r'Adaptive $r_n = r/\sqrt{n}$')
    ax2.set_xlabel('Iteration n', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'$\|x_n - x^*\|$', fontsize=11, fontweight='bold')
    ax2.set_title('Adaptive Parameter: Faster Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_parameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fig_parameter_sensitivity.pdf")

def main():
    """Generate all figures"""
    print("Generating proximal point algorithm figures...")
    fig_proximal_convergence()
    fig_convergence_types()
    fig_algorithm_evolution()
    fig_convergence_rates()
    fig_banach_space_hierarchy()
    fig_monotone_operators()
    fig_algorithm_flowchart()
    fig_parameter_sensitivity()
    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
