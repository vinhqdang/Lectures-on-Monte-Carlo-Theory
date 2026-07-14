#!/usr/bin/env python3
"""
Generate figures for Chapter 7c: Approximating Fixed Points: Ishikawa
Visualizations include:
- Ishikawa iteration convergence patterns
- Comparison with Mann iteration
- Convergence rate analysis
- Two-dimensional iteration visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Set style for publication quality
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'ishikawa': '#2E86AB', 'mann': '#A23B72', 'picard': '#F18F01', 'true': '#C73E1D'}

def setup_figure():
    """Create a figure with proper sizing for Beamer slides"""
    fig = plt.figure(figsize=(10, 6))
    return fig

def save_figure(fig, name):
    """Save figure as PDF in the figures directory"""
    filepath = f"{name}.pdf"
    fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Generated: {filepath}")

# ============================================================================
# Figure 1: Ishikawa Iteration Convergence (Simple Mapping)
# ============================================================================
def fig_ishikawa_convergence():
    """Visualize Ishikawa iteration convergence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mapping: T(x) = 0.5*x + 0.3*sin(x)
    # Fixed point near x* ≈ 0
    def T(x):
        return 0.5 * x + 0.3 * np.sin(x)

    # Ishikawa parameters
    alpha_seq = [0.7, 0.6, 0.5, 0.4, 0.3]  # Main iteration parameter
    beta_seq = [0.5, 0.5, 0.5, 0.5, 0.5]   # Intermediate parameter

    x0 = 2.0
    iterations = 30

    # Left plot: Convergence trajectory
    x_vals = np.linspace(-0.5, 2.5, 1000)
    y_vals = T(x_vals)
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='$T(x) = 0.5x + 0.3\\sin(x)$')
    ax1.plot(x_vals, x_vals, 'k--', linewidth=1, alpha=0.5, label='$y = x$')

    # Perform Ishikawa iterations
    x = x0
    x_sequence = [x]
    for i in range(iterations):
        alpha = alpha_seq[i % len(alpha_seq)]
        beta = beta_seq[i % len(beta_seq)]

        # Intermediate step: y_n = beta * T(x_n) + (1 - beta) * x_n
        y = beta * T(x) + (1 - beta) * x
        # Main step: x_{n+1} = alpha * T(y_n) + (1 - alpha) * x_n
        x = alpha * T(y) + (1 - alpha) * x
        x_sequence.append(x)

    # Plot the iteration steps
    x = x0
    for i in range(min(8, iterations)):
        alpha = alpha_seq[i % len(alpha_seq)]
        beta = beta_seq[i % len(beta_seq)]

        y = beta * T(x) + (1 - beta) * x
        x_new = alpha * T(y) + (1 - alpha) * x

        # Draw arrows for iteration
        ax1.annotate('', xy=(x, T(x)), xytext=(x, x),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1))
        ax1.annotate('', xy=(T(x), T(x)), xytext=(x, T(x)),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1))
        ax1.plot(x, x, 'o', color=colors['ishikawa'], markersize=6, alpha=0.7)

        x = x_new

    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$T(x)$', fontsize=12)
    ax1.set_title('Ishikawa Iteration Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)

    # Right plot: Convergence error
    errors = np.abs(np.array(x_sequence) - x_sequence[-1])
    ax2.semilogy(range(len(errors)), errors, 'o-', color=colors['ishikawa'],
                linewidth=2, markersize=6, label='Ishikawa Iteration')
    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel('Error $|x_n - x^*|$', fontsize=12)
    ax2.set_title('Convergence Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'ishikawa_convergence')

# ============================================================================
# Figure 2: Comparison - Ishikawa vs Mann vs Picard
# ============================================================================
def fig_comparison_methods():
    """Compare convergence of Ishikawa, Mann, and Picard iterations"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mapping: T(x) = 0.6*x
    # Fixed point: x* = 0
    def T(x):
        return 0.6 * x

    x0 = 2.0
    iterations = 40

    # Picard (one-step)
    x_picard = [x0]
    for _ in range(iterations):
        x_picard.append(T(x_picard[-1]))

    # Mann iteration: x_{n+1} = alpha * T(x_n) + (1-alpha) * x_n
    alpha_mann = 0.7
    x_mann = [x0]
    for _ in range(iterations):
        x_mann.append(alpha_mann * T(x_mann[-1]) + (1 - alpha_mann) * x_mann[-1])

    # Ishikawa iteration
    alpha_ishi = 0.7
    beta_ishi = 0.5
    x_ishi = [x0]
    for _ in range(iterations):
        y = beta_ishi * T(x_ishi[-1]) + (1 - beta_ishi) * x_ishi[-1]
        x_ishi.append(alpha_ishi * T(y) + (1 - alpha_ishi) * x_ishi[-1])

    # Compute errors
    errors_picard = np.abs(np.array(x_picard))
    errors_mann = np.abs(np.array(x_mann))
    errors_ishi = np.abs(np.array(x_ishi))

    n_range = range(len(errors_picard))

    ax.semilogy(n_range, errors_picard, 's-', color=colors['picard'],
               linewidth=2.5, markersize=6, label='Picard (one-step)', alpha=0.8)
    ax.semilogy(n_range, errors_mann, 'o-', color=colors['mann'],
               linewidth=2.5, markersize=6, label='Mann ($\\alpha=0.7$)', alpha=0.8)
    ax.semilogy(n_range, errors_ishi, '^-', color=colors['ishikawa'],
               linewidth=2.5, markersize=6, label='Ishikawa ($\\alpha=0.7, \\beta=0.5$)', alpha=0.8)

    ax.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error $|x_n - x^*|$', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Comparison: Picard vs Mann vs Ishikawa',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save_figure(fig, 'comparison_methods')

# ============================================================================
# Figure 3: Effect of Parameters on Convergence
# ============================================================================
def fig_parameter_analysis():
    """Analyze effect of alpha and beta on Ishikawa convergence"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    def T(x):
        return 0.7 * x

    x0 = 2.0
    iterations = 30

    # Vary alpha (main parameter)
    alphas = [0.3, 0.5, 0.7, 0.9]
    beta_fixed = 0.5

    for alpha in alphas:
        x = [x0]
        for _ in range(iterations):
            y = beta_fixed * T(x[-1]) + (1 - beta_fixed) * x[-1]
            x.append(alpha * T(y) + (1 - alpha) * x[-1])
        errors = np.abs(np.array(x))
        ax1.semilogy(errors, 'o-', linewidth=2, markersize=5, label=f'$\\alpha = {alpha}$')

    ax1.set_xlabel('Iteration $n$', fontsize=11)
    ax1.set_ylabel('Error $|x_n - x^*|$', fontsize=11)
    ax1.set_title('Effect of $\\alpha$ (Main Parameter)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Vary beta (intermediate parameter)
    betas = [0.1, 0.3, 0.5, 0.8]
    alpha_fixed = 0.7

    for beta in betas:
        x = [x0]
        for _ in range(iterations):
            y = beta * T(x[-1]) + (1 - beta) * x[-1]
            x.append(alpha_fixed * T(y) + (1 - alpha_fixed) * x[-1])
        errors = np.abs(np.array(x))
        ax2.semilogy(errors, 's-', linewidth=2, markersize=5, label=f'$\\beta = {beta}$')

    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('Error $|x_n - x^*|$', fontsize=11)
    ax2.set_title('Effect of $\\beta$ (Intermediate Parameter)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Heat map: alpha vs beta convergence rate
    alphas_range = np.linspace(0.1, 0.95, 15)
    betas_range = np.linspace(0.1, 0.95, 15)
    convergence_rate = np.zeros((len(betas_range), len(alphas_range)))

    for i, beta in enumerate(betas_range):
        for j, alpha in enumerate(alphas_range):
            x = x0
            for _ in range(20):
                y = beta * T(x) + (1 - beta) * x
                x = alpha * T(y) + (1 - alpha) * x
            convergence_rate[i, j] = np.log10(np.abs(x) + 1e-10)

    im = ax3.contourf(alphas_range, betas_range, convergence_rate, levels=15, cmap='viridis')
    ax3.set_xlabel('$\\alpha$ (Main Parameter)', fontsize=11)
    ax3.set_ylabel('$\\beta$ (Intermediate Parameter)', fontsize=11)
    ax3.set_title('Convergence Rate Heatmap (log scale)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('$\log_{10}|x_{20} - x^*|$', fontsize=10)

    # Convergence region (alpha-beta plane)
    # For linear mapping T(x) = 0.7*x, need |1 - alpha| < alpha*beta*0.7
    alpha_mesh = np.linspace(0.01, 1, 100)
    beta_mesh = np.linspace(0.01, 1, 100)

    ax4.fill_between([0, 1], 0, 1, alpha=0.2, color='green', label='Convergence Region')
    ax4.plot([0.7, 0.7], [0, 1], 'r--', linewidth=2, label='$\\alpha = 0.7$')
    ax4.plot([0, 1], [0.5, 0.5], 'b--', linewidth=2, label='$\\beta = 0.5$')
    ax4.scatter([0.7], [0.5], s=200, c=colors['ishikawa'], marker='*',
               zorder=5, label='Standard choice', edgecolors='black', linewidths=2)

    ax4.set_xlabel('$\\alpha$', fontsize=11)
    ax4.set_ylabel('$\\beta$', fontsize=11)
    ax4.set_title('Parameter Space', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'parameter_analysis')

# ============================================================================
# Figure 4: Two-Dimensional Iteration Visualization
# ============================================================================
def fig_2d_iteration():
    """Visualize Ishikawa iteration in 2D space"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Define a 2D contraction mapping
    def T(p):
        """T(x, y) = (0.6*x + 0.1*y, 0.1*x + 0.5*y)"""
        return np.array([0.6 * p[0] + 0.1 * p[1],
                        0.1 * p[0] + 0.5 * p[1]])

    # Fixed point is at origin
    p0 = np.array([2.0, 2.0])

    # Ishikawa iteration parameters
    alpha = 0.7
    beta = 0.5
    iterations = 15

    # Generate iterations
    trajectory = [p0]
    p = p0.copy()
    for _ in range(iterations):
        q = beta * T(p) + (1 - beta) * p
        p = alpha * T(q) + (1 - alpha) * p
        trajectory.append(p.copy())

    trajectory = np.array(trajectory)

    # Left: Iteration path
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors['ishikawa'],
            linewidth=2, markersize=7, label='Iteration path', alpha=0.8)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 's', color=colors['picard'],
            markersize=12, label='Start ($x_0$)', zorder=5)
    ax1.plot(0, 0, '*', color=colors['true'], markersize=20, label='Fixed point ($x^*$)', zorder=5)

    # Draw arrows
    for i in range(len(trajectory)-1):
        ax1.annotate('', xy=tuple(trajectory[i+1]), xytext=tuple(trajectory[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))

    # Add contour of the mapping
    x_range = np.linspace(-0.5, 2.5, 50)
    y_range = np.linspace(-0.5, 2.5, 50)
    X, Y = np.meshgrid(x_range, y_range)

    Z = np.sqrt(X**2 + Y**2)  # Distance from origin
    contours = ax1.contour(X, Y, Z, levels=8, colors='gray', alpha=0.3, linewidths=0.5)
    ax1.clabel(contours, inline=True, fontsize=8)

    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_title('2D Ishikawa Iteration Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right: Norm convergence
    norms = np.linalg.norm(trajectory, axis=1)
    ax2.semilogy(range(len(norms)), norms, 'o-', color=colors['ishikawa'],
                linewidth=2.5, markersize=7, label='$\|\|x_n\|\|$')
    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel('$\|\|x_n\|\|$ (Euclidean Norm)', fontsize=12)
    ax2.set_title('Norm Convergence in 2D', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'iteration_2d')

# ============================================================================
# Figure 5: Convergence Rate Analysis
# ============================================================================
def fig_convergence_rate():
    """Analyze and compare convergence rates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Different contraction coefficients
    coefficients = [0.3, 0.5, 0.7, 0.85]
    iterations = 50

    for coeff in coefficients:
        def T(x):
            return coeff * x

        # Ishikawa
        alpha, beta = 0.7, 0.5
        x_ishi = [2.0]
        for _ in range(iterations):
            y = beta * T(x_ishi[-1]) + (1 - beta) * x_ishi[-1]
            x_ishi.append(alpha * T(y) + (1 - alpha) * x_ishi[-1])

        errors_ishi = np.abs(np.array(x_ishi))
        ax1.semilogy(errors_ishi, 'o-', linewidth=2, markersize=4,
                    label=f'$T(x) = {coeff}x$', alpha=0.8)

    ax1.set_xlabel('Iteration $n$', fontsize=12)
    ax1.set_ylabel('Error $|x_n|$', fontsize=12)
    ax1.set_title('Convergence for Different Contraction Coefficients',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Linear convergence rate analysis
    coeff = 0.7
    def T(x):
        return coeff * x

    # Compute convergence rates
    alpha_vals = np.linspace(0.1, 0.99, 20)
    rates = []

    for alpha in alpha_vals:
        beta = 0.5
        x = 2.0
        for _ in range(30):
            y = beta * T(x) + (1 - beta) * x
            x = alpha * T(y) + (1 - alpha) * x
        # Convergence rate is |x_30| relative to |x_0|
        rate = np.abs(x) / 2.0
        rates.append(rate)

    ax2.plot(alpha_vals, rates, 'o-', color=colors['ishikawa'],
            linewidth=2.5, markersize=8, label='Convergence Factor')
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.5, label='No Convergence')
    ax2.fill_between(alpha_vals, 0, rates, alpha=0.2, color=colors['ishikawa'])

    ax2.set_xlabel('Parameter $\\alpha$', fontsize=12)
    ax2.set_ylabel('Convergence Factor $|x_{30}|/|x_0|$', fontsize=12)
    ax2.set_title('Parameter Influence on Convergence', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([-0.05, 1.1])

    fig.tight_layout()
    save_figure(fig, 'convergence_rate')

# ============================================================================
# Figure 6: Iteration Scheme Diagram
# ============================================================================
def fig_iteration_scheme():
    """Create a flowchart/diagram of Ishikawa iteration"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'Ishikawa Iteration Scheme', fontsize=16, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Start
    circle_start = plt.Circle((5, 10.5), 0.3, color=colors['picard'], zorder=3)
    ax.add_patch(circle_start)
    ax.text(5, 10.5, 'Start', fontsize=10, ha='center', va='center', fontweight='bold', color='white')

    # Step 0: Initialize
    rect_init = mpatches.FancyBboxPatch((3.5, 9.3), 3, 0.8, boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect_init)
    ax.text(5, 9.7, 'Choose $x_0 \\in X$, $\\alpha, \\beta \\in (0,1)$', fontsize=10, ha='center', va='center')

    ax.arrow(5, 10.2, 0, -0.35, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Step 1: Intermediate step
    rect_step1 = mpatches.FancyBboxPatch((2, 7.8), 6, 0.8, boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect_step1)
    ax.text(5, 8.2, 'Step 1: Compute $y_n = \\beta T(x_n) + (1-\\beta) x_n$',
           fontsize=10, ha='center', va='center', family='monospace')

    ax.arrow(5, 9.3, 0, -0.35, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Step 2: Main step
    rect_step2 = mpatches.FancyBboxPatch((1.5, 6.3), 7, 0.8, boxstyle="round,pad=0.1",
                                        edgecolor='black', facecolor='lightcyan', linewidth=2)
    ax.add_patch(rect_step2)
    ax.text(5, 6.7, 'Step 2: Compute $x_{n+1} = \\alpha T(y_n) + (1-\\alpha) x_n$',
           fontsize=10, ha='center', va='center', family='monospace')

    ax.arrow(5, 7.8, 0, -0.35, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Convergence check
    diamond = mpatches.FancyBboxPatch((3, 4.8), 4, 0.8, boxstyle="round,pad=0.1",
                                     edgecolor='black', facecolor='lightsalmon', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 5.2, 'Convergence Check', fontsize=10, ha='center', va='center', fontweight='bold')

    ax.arrow(5, 6.3, 0, -0.35, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Decision branches
    ax.text(3.5, 4.3, 'No', fontsize=9, ha='center', fontweight='bold', color='red')
    ax.arrow(3.2, 5, -1.2, 1, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.arrow(3.8, 5, -0.5, 1.5, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.text(1.5, 6.5, '$n \\leftarrow n+1$', fontsize=9, ha='center', style='italic')

    # Loop back
    ax.annotate('', xy=(2, 7.2), xytext=(1.5, 6.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=.5"))

    # Success
    ax.text(6.5, 4.3, 'Yes', fontsize=9, ha='center', fontweight='bold', color='green')
    ax.arrow(6.8, 5, 1.2, -1, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)

    # Output
    rect_output = mpatches.FancyBboxPatch((3.5, 2.3), 3, 0.8, boxstyle="round,pad=0.1",
                                         edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect_output)
    ax.text(5, 2.7, 'Output: $x_n \\approx x^*$', fontsize=10, ha='center', va='center', fontweight='bold')

    # Key properties box
    properties_text = (
        'Key Properties:\n'
        '• Two-step iterative method\n'
        '• Convergence for nonexpansive mappings\n'
        '• Parameters $\\alpha, \\beta$ control stability\n'
        '• Faster than Mann iteration for some mappings'
    )
    ax.text(5, 0.8, properties_text, fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')

    fig.tight_layout()
    save_figure(fig, 'iteration_scheme')

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 7c: Approximating Fixed Points - Ishikawa")
    print("=" * 70)

    fig_ishikawa_convergence()
    fig_comparison_methods()
    fig_parameter_analysis()
    fig_2d_iteration()
    fig_convergence_rate()
    fig_iteration_scheme()

    print("=" * 70)
    print("All figures generated successfully!")
