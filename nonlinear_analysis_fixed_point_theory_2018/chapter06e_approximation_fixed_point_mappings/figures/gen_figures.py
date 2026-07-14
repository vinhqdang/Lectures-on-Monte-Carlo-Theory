#!/usr/bin/env python3
"""
Generate figures for Chapter 6e: Approximation of Fixed Point Mappings
Based on Pathak (2018) - Applications of Fixed Point Theorems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D62828'
}

def fig_iterative_convergence():
    """Figure 1: Iterative convergence of approximation sequence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Convergence to fixed point
    t = np.linspace(0, 10, 100)
    # Exponential convergence
    y_conv = np.exp(-0.3 * t)
    # Linear convergence
    y_linear = (0.8) ** t

    ax1.plot(t, y_conv, 'o-', color=colors['primary'], linewidth=2,
             markersize=4, label='Exponential convergence')
    ax1.plot(t, y_linear, 's-', color=colors['secondary'], linewidth=2,
             markersize=4, label='Linear convergence')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Iteration n', fontsize=12)
    ax1.set_ylabel('Error $||x_n - x^*||$', fontsize=12)
    ax1.set_title('Convergence Rates', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Subplot 2: Residual norm
    residuals = np.array([1.0, 0.25, 0.08, 0.03, 0.01, 0.003, 0.001])
    iterations = np.arange(len(residuals))

    ax2.bar(iterations, residuals, color=colors['accent'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Iteration m', fontsize=12)
    ax2.set_ylabel('$R_m = ||X_{m+1} - (Q + AX_m^{1/2}A + ...)||$', fontsize=11)
    ax2.set_title('Residual Error History', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('fig_01_iterative_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_fixed_point_iteration():
    """Figure 2: Fixed point iteration graphically"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define function T(x) = 0.5*x + 0.3*sin(x)
    T = lambda x: 0.5 * x + 0.3 * np.sin(x)
    x = np.linspace(-2, 4, 500)
    T_x = T(x)
    y = x  # Identity line

    ax.plot(x, y, 'k-', linewidth=2, label='$y = x$ (identity)')
    ax.plot(x, T_x, linewidth=2.5, color=colors['primary'], label='$y = T(x)$')

    # Find fixed point approximately (solve T(x) = x)
    from scipy.optimize import fsolve
    fixed_point = fsolve(lambda x_val: T(x_val) - x_val, 1.0)[0]

    # Draw cobweb diagram
    x0 = -1.5
    ax.plot(x0, 0, 'o', color=colors['success'], markersize=8, label='$x_0$ (initial guess)')

    for i in range(5):
        y_val = 0.5 * x0 + 0.3 * np.sin(x0)
        # Vertical line to curve
        ax.plot([x0, x0], [x0, y_val], 'r--', linewidth=1, alpha=0.7)
        # Horizontal line to identity
        ax.plot([x0, y_val], [y_val, y_val], 'r--', linewidth=1, alpha=0.7)
        x0 = y_val

    # Mark fixed point
    ax.plot(fixed_point, fixed_point, '*', color=colors['warning'],
            markersize=15, label='Fixed point $x^*$', zorder=5)

    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title('Fixed Point Iteration: Cobweb Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fig_02_fixed_point_iteration.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_approximation_error():
    """Figure 3: Approximation error bounds"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n = np.arange(0, 21)
    # Different error bounds
    bound1 = 1.0 * (0.7 ** n)  # Contraction rate 0.7
    bound2 = 1.0 * (0.5 ** n)  # Contraction rate 0.5
    bound3 = 1.0 * (0.3 ** n)  # Contraction rate 0.3
    bound4 = 1.0 / (1 + n)     # Slower convergence

    ax.semilogy(n, bound1, 'o-', color=colors['primary'], linewidth=2,
                markersize=6, label='$\lambda = 0.7$ (slow)')
    ax.semilogy(n, bound2, 's-', color=colors['secondary'], linewidth=2,
                markersize=6, label='$\lambda = 0.5$ (medium)')
    ax.semilogy(n, bound3, '^-', color=colors['success'], linewidth=2,
                markersize=6, label='$\lambda = 0.3$ (fast)')
    ax.semilogy(n, bound4, 'd--', color=colors['warning'], linewidth=2,
                markersize=6, label='Non-contraction')

    ax.set_xlabel('Iteration n', fontsize=12)
    ax.set_ylabel('Error bound $||x_n - x^*||$', fontsize=12)
    ax.set_title('Approximation Error vs. Contraction Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-4, 1.5)

    plt.tight_layout()
    plt.savefig('fig_03_approximation_error.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_sequence_convergence():
    """Figure 4: Convergence of iterative sequences"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate three different sequences
    np.random.seed(42)
    n_iter = 30

    # Sequence 1: Fast convergence
    x1 = 1.0 + 0.1 * np.exp(-0.5 * np.arange(n_iter))
    # Sequence 2: Medium convergence
    x2 = 1.0 + 0.1 * np.exp(-0.3 * np.arange(n_iter)) + 0.02 * np.random.randn(n_iter)
    # Sequence 3: Oscillatory convergence
    x3 = 1.0 + 0.1 * np.exp(-0.2 * np.arange(n_iter)) * np.cos(np.arange(n_iter) * 0.5)

    n = np.arange(n_iter)
    ax.plot(n, x1, 'o-', color=colors['primary'], linewidth=2, markersize=6,
            label='Monotone convergence')
    ax.plot(n, x2, 's-', color=colors['secondary'], linewidth=2, markersize=6,
            label='Convergence with noise')
    ax.plot(n, x3, '^-', color=colors['success'], linewidth=2, markersize=6,
            label='Oscillatory convergence')

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Fixed point $x^*$', alpha=0.7)

    ax.set_xlabel('Iteration n', fontsize=12)
    ax.set_ylabel('$x_n$', fontsize=12)
    ax.set_title('Convergence of Iterative Sequences {$x_n$}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_04_sequence_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_domain_diagram():
    """Figure 5: Domain and mapping diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw domain D
    domain = FancyBboxPatch((0.5, 1), 4, 3, boxstyle="round,pad=0.1",
                            edgecolor=colors['primary'], facecolor=colors['primary'],
                            alpha=0.2, linewidth=2.5)
    ax.add_patch(domain)
    ax.text(2.5, 2.5, 'Domain D\n(Banach Space)', fontsize=13, ha='center',
            va='center', fontweight='bold')

    # Draw mapping arrows
    points_x = [1.5, 2.5, 3.5]
    points_y = [1.5, 2.0, 2.5]

    for i, (px, py) in enumerate(zip(points_x, points_y)):
        circle = mpatches.Circle((px, py), 0.15, color=colors['accent'], zorder=5)
        ax.add_patch(circle)

        # Arrow showing iteration
        if i < len(points_x) - 1:
            dx = points_x[i+1] - px
            dy = points_y[i+1] - py
            arrow = FancyArrowPatch((px + 0.2, py), (points_x[i+1] - 0.2, points_y[i+1]),
                                   arrowstyle='->', mutation_scale=25,
                                   color=colors['secondary'], linewidth=2)
            ax.add_patch(arrow)
            ax.text((px + points_x[i+1])/2, (py + points_y[i+1])/2 + 0.3,
                   f'$T^{i+1}$', fontsize=11, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Label points
    ax.text(1.5, 1.2, '$x_0$', fontsize=12, ha='center', fontweight='bold')
    ax.text(2.5, 1.8, '$x_1=Tx_0$', fontsize=12, ha='center', fontweight='bold')
    ax.text(3.5, 2.3, '$x_2=T^2x_0$', fontsize=12, ha='center', fontweight='bold')
    ax.text(2.5, 0.5, '...convergence to $x^* \in D$', fontsize=12, ha='center',
           style='italic')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Fixed Point Mapping: Sequence Generation', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('fig_05_domain_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_contraction_mapping():
    """Figure 6: Contraction mapping principle"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 2, 500)

    # Define mappings
    T1_func = lambda x_val: 0.3 * x_val + 0.5  # Strong contraction
    T2_func = lambda x_val: 0.5 * x_val + 0.3  # Weak contraction
    T3_func = lambda x_val: 0.7 * x_val + 0.2  # Weaker contraction

    T1 = T1_func(x)
    T2 = T2_func(x)
    T3 = T3_func(x)

    ax.plot(x, x, 'k-', linewidth=2.5, label='Identity: $y = x$')
    ax.plot(x, T1, linewidth=2.5, color=colors['success'], label='$T_1(x)$ (strong, $\lambda=0.3$)')
    ax.plot(x, T2, linewidth=2.5, color=colors['primary'], label='$T_2(x)$ (weak, $\lambda=0.5$)')
    ax.plot(x, T3, linewidth=2.5, color=colors['warning'], label='$T_3(x)$ (weaker, $\lambda=0.7$)')

    # Mark fixed points
    from scipy.optimize import fsolve
    fp1 = fsolve(lambda x_val: T1_func(x_val) - x_val, 1.0)[0]
    fp2 = fsolve(lambda x_val: T2_func(x_val) - x_val, 1.0)[0]
    fp3 = fsolve(lambda x_val: T3_func(x_val) - x_val, 1.0)[0]

    ax.plot(fp1, fp1, 'o', color=colors['success'], markersize=10)
    ax.plot(fp2, fp2, 'o', color=colors['primary'], markersize=10)
    ax.plot(fp3, fp3, 'o', color=colors['warning'], markersize=10)

    # Add annotation box
    textstr = 'Contraction Mapping Principle:\n' \
              'If $||T(x) - T(y)|| \leq \lambda ||x-y||$\nwith $\lambda < 1$,\n' \
              'then unique fixed point exists'
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y = T(x)$', fontsize=12)
    ax.set_title('Contraction Mappings with Different Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('fig_06_contraction_mapping.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating figures for Chapter 6e...")
    fig_iterative_convergence()
    print("Generated: fig_01_iterative_convergence.pdf")

    fig_fixed_point_iteration()
    print("Generated: fig_02_fixed_point_iteration.pdf")

    fig_approximation_error()
    print("Generated: fig_03_approximation_error.pdf")

    fig_sequence_convergence()
    print("Generated: fig_04_sequence_convergence.pdf")

    fig_domain_diagram()
    print("Generated: fig_05_domain_diagram.pdf")

    fig_contraction_mapping()
    print("Generated: fig_06_contraction_mapping.pdf")

    print("\nAll figures generated successfully!")
