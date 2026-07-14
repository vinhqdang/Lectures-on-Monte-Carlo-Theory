#!/usr/bin/env python3
"""
Generate figures for Chapter 9: Applications of Fixed Point Theorems
Pages 689-710
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for PDF output and consistency
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'

def fig_convergence_history():
    """Figure 9.1: Convergence history of the iterative algorithm"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Simulate convergence behavior similar to Fig 9.1
    iterations = np.arange(0, 36)
    # Exponential decay with noise
    errors = 230 * np.exp(-0.15 * iterations) + np.random.normal(0, 2, len(iterations))
    errors = np.maximum(errors, 0.1)  # Keep positive

    ax.plot(iterations, errors, 'b-', linewidth=2, label='Error')
    ax.fill_between(iterations, errors, alpha=0.2)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Convergence History of Algorithm (9.16)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 250)

    plt.tight_layout()
    plt.savefig('figures/fig_convergence_history.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_matrix_iteration():
    """Visualization of matrix iteration convergence"""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Illustrate convergence of matrix sequence
    n_steps = 20

    # Step 1: Residual errors
    iterations = np.arange(0, n_steps)
    residuals = 150 * np.exp(-0.25 * iterations)

    ax.semilogy(iterations, residuals, 'o-', color='#2E86AB', linewidth=2.5,
                markersize=6, label='Residual Error $R_m$')

    ax.set_xlabel('Iteration $m$', fontsize=11)
    ax.set_ylabel('Residual Error $R_m$ (log scale)', fontsize=11)
    ax.set_title('Matrix Iteration Convergence: $X_{n+1} = Q + A^*X_n^{\\frac{1}{2}}A + B^*X_n^{-\\frac{1}{2}}B + C^*X_n^{\\frac{1}{2}}C$',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_matrix_iteration.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_control_theory_diagram():
    """Diagram for Control Theory Application"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw control system diagram
    # Boxes for components
    box_width = 1.2
    box_height = 0.6

    # System box
    system = FancyBboxPatch((1, 2), box_width, box_height,
                            boxstyle="round,pad=0.1",
                            edgecolor='#E63946', facecolor='#F1FAEE', linewidth=2)
    ax.add_patch(system)
    ax.text(1.6, 2.3, r'$\dot{x}(s) = f(x(s), \alpha(s))$',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Control box
    control = FancyBboxPatch((0.1, 3.2), 1.2, 0.6,
                             boxstyle="round,pad=0.1",
                             edgecolor='#A8DADC', facecolor='#F1FAEE', linewidth=2)
    ax.add_patch(control)
    ax.text(0.7, 3.5, r'Control $\alpha(s)$', ha='center', va='center', fontsize=10, fontweight='bold')

    # State box
    state = FancyBboxPatch((2.8, 2.5), 1.2, 0.6,
                          boxstyle="round,pad=0.1",
                          edgecolor='#457B9D', facecolor='#F1FAEE', linewidth=2)
    ax.add_patch(state)
    ax.text(3.4, 2.8, r'State $x(t)$', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(0.8, 3.2), xytext=(0.8, 3.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#A8DADC'))
    ax.annotate('', xy=(2.2, 2.3), xytext=(1.2, 2.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(2.8, 2.8), xytext=(2.2, 2.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#457B9D'))

    # Add text annotations
    ax.text(1.6, 0.8, 'Dynamic Programming:', fontsize=11, fontweight='bold')
    ax.text(1.6, 0.3, r'Control signal $\alpha(\cdot)$ optimally governs the ODE solution trajectory',
            fontsize=10, ha='center', style='italic')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/fig_control_system.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_differential_equations():
    """Solution trajectories for differential equations"""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Generate trajectories for a sample ODE
    t = np.linspace(0, 5, 200)

    # Multiple solution trajectories with different initial conditions
    x0_values = np.linspace(0.5, 3, 5)
    colors = plt.cm.viridis(np.linspace(0, 1, len(x0_values)))

    for i, x0 in enumerate(x0_values):
        # Exponential decay solution: x(t) = x0 * exp(-t)
        x = x0 * np.exp(-t)
        ax.plot(t, x, linewidth=2.2, color=colors[i],
                label=f'$x_0 = {x0:.1f}$')

    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('Solution $x(t)$', fontsize=11)
    ax.set_title('Solutions of Differential Equation: $\\frac{dx}{dt} = f(t, x(t))$',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.5)

    plt.tight_layout()
    plt.savefig('figures/fig_differential_equations.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_banach_contraction():
    """Visualization of Banach Contraction Mapping"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw the set E and visualization of contraction
    x = np.linspace(0, 2*np.pi, 100)

    # Original function and its iterate
    f_orig = 1.5 * np.sin(x) + 1
    f_iter1 = 0.7 * f_orig  # Contraction factor λ
    f_iter2 = 0.7 * f_iter1

    ax.plot(x, f_orig, 'o-', color='#E63946', linewidth=2.5,
            markersize=4, label='Original function $f$', alpha=0.8)
    ax.plot(x, f_iter1, 's-', color='#A8DADC', linewidth=2.5,
            markersize=4, label='First iterate $Ff = f$', alpha=0.8)
    ax.plot(x, f_iter2, '^-', color='#457B9D', linewidth=2.5,
            markersize=4, label='Converging sequence', alpha=0.8)

    ax.fill_between(x, 0, 2*np.pi, alpha=0.1, color='gray', label='Domain $E$')

    ax.set_xlabel('Parameter', fontsize=11)
    ax.set_ylabel('Function Value', fontsize=11)
    ax.set_title('Banach Contraction Theorem: Iterative Convergence $|Fφ_1 - Fφ_2| ≤ λ|φ_1 - φ_2|, λ < 1$',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2*np.pi)

    plt.tight_layout()
    plt.savefig('figures/fig_banach_contraction.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_fixed_point_property():
    """Illustration of fixed point property for nonexpansive mappings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Nonexpansive mapping
    t = np.linspace(0, 2*np.pi, 100)

    # Identity line
    ax1.plot(t, t, 'k--', linewidth=1.5, alpha=0.5, label='Identity: $y = x$')

    # Nonexpansive mapping (contraction)
    nonexp = 0.6 * t + 0.8
    ax1.plot(t, nonexp, 'b-', linewidth=2.5, label='Nonexpansive: $T(x)$')

    # Fixed point
    fixed_point = 2.0
    ax1.plot(fixed_point, fixed_point, 'ro', markersize=10, label='Fixed point $x^*$', zorder=5)

    ax1.set_xlabel('$x$', fontsize=11)
    ax1.set_ylabel('$T(x)$', fontsize=11)
    ax1.set_title('Nonexpansive Mapping', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(0, 2*np.pi)

    # Right plot: Convergence behavior
    x_seq = [1.0]
    for i in range(20):
        x_next = 0.6 * x_seq[-1] + 0.8
        x_seq.append(x_next)

    iterations = np.arange(len(x_seq))
    ax2.plot(iterations, x_seq, 'bo-', linewidth=2, markersize=6, label='$x_n = T(x_{n-1})$')
    ax2.axhline(y=fixed_point, color='r', linestyle='--', linewidth=2, label='Limit $x^*$')
    ax2.fill_between(iterations, x_seq, fixed_point, alpha=0.2, color='blue')

    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('$x_n$', fontsize=11)
    ax2.set_title('Convergence to Fixed Point', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_fixed_point_property.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def fig_operator_norm():
    """Visualization of operator norm and contractivity"""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Show the spectral radius and contractivity
    x = np.linspace(0, 1, 100)

    # Different contraction rates
    lambda_values = [0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(lambda_values)))

    for lambda_val, color in zip(lambda_values, colors):
        # Error decay
        error = np.exp(-lambda_val * x)
        ax.plot(x, error, linewidth=2.5, color=color,
                label=f'$\\lambda = {lambda_val}$')

    # Contractivity threshold
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.fill_between(x, 0, 1, alpha=0.1, color='green', label='Contraction region')

    ax.set_xlabel('Parameter or Iteration', fontsize=11)
    ax.set_ylabel('Error/Norm', fontsize=11)
    ax.set_title('Contraction Rates: $|T(x) - T(y)| ≤ λ|x - y|$ with $λ < 1$',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('figures/fig_operator_norm.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 9: Applications of Fixed Point Theorems...")

    fig_convergence_history()
    print("✓ Generated: fig_convergence_history.pdf")

    fig_matrix_iteration()
    print("✓ Generated: fig_matrix_iteration.pdf")

    fig_control_theory_diagram()
    print("✓ Generated: fig_control_system.pdf")

    fig_differential_equations()
    print("✓ Generated: fig_differential_equations.pdf")

    fig_banach_contraction()
    print("✓ Generated: fig_banach_contraction.pdf")

    fig_fixed_point_property()
    print("✓ Generated: fig_fixed_point_property.pdf")

    fig_operator_norm()
    print("✓ Generated: fig_operator_norm.pdf")

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
