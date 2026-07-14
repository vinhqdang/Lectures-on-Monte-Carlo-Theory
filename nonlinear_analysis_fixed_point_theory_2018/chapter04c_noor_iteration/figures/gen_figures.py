#!/usr/bin/env python3
"""
Generate figures for Chapter 4c: Noor Iteration
Based on Pathak's "An Introduction to Nonlinear Analysis and Fixed Point Theory"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'mann': '#2E86AB', 'ishikawa': '#A23B72', 'noor': '#F18F01'}

# ============================================================================
# Figure 1: Noor Iteration Process
# ============================================================================
def fig_noor_iteration_process():
    """Visualize the three-step Noor iteration process."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Step boxes
    steps = ['$x_n$', '$z_n = (1-c_n)x_n + c_n Tx_n$',
             '$y_n = (1-b_n)x_n + b_n Tz_n$',
             '$x_{n+1} = (1-a_n)x_n + a_n Ty_n$']

    y_positions = np.linspace(0.9, 0.1, 4)

    for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
        if i == 0:
            box = mpatches.FancyBboxPatch((0.15, y_pos-0.05), 0.7, 0.08,
                                         boxstyle="round,pad=0.01",
                                         facecolor=colors['noor'], alpha=0.3,
                                         edgecolor=colors['noor'], linewidth=2)
        else:
            box = mpatches.FancyBboxPatch((0.15, y_pos-0.05), 0.7, 0.08,
                                         boxstyle="round,pad=0.01",
                                         facecolor=colors['noor'], alpha=0.2,
                                         edgecolor=colors['noor'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(0.5, y_pos, step, ha='center', va='center',
               fontsize=14, fontweight='bold' if i==0 else 'normal')

        # Add arrows between steps
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((0.5, y_positions[i]-0.06),
                                   (0.5, y_positions[i+1]+0.06),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=2, color=colors['noor'])
            ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.98, 'Noor Iteration: Three-Step Process',
           ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('noor_iteration_process.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: noor_iteration_process.pdf")


# ============================================================================
# Figure 2: Iteration Methods Comparison
# ============================================================================
def fig_iteration_methods_comparison():
    """Compare Mann, Ishikawa, and Noor iterations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    methods = ['Mann', 'Ishikawa', 'Noor']
    steps = [1, 2, 3]

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, steps, color=[colors['mann'], colors['ishikawa'], colors['noor']],
                  alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_ylabel('Number of Steps per Iteration', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Iteration Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(0, 4)

    # Add value labels on bars
    for i, (bar, step) in enumerate(zip(bars, steps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{step}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add formulas as legend
    legend_text = (
        'Mann: $x_{n+1} = (1-a_n)x_n + a_n Tx_n$\n'
        'Ishikawa: Two intermediate steps\n'
        'Noor: Three intermediate steps'
    )
    ax.text(0.98, 0.97, legend_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)

    plt.tight_layout()
    plt.savefig('iteration_methods_comparison.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: iteration_methods_comparison.pdf")


# ============================================================================
# Figure 3: Noor Iteration Convergence Example
# ============================================================================
def fig_noor_convergence():
    """Show convergence of Noor iteration for a specific mapping."""
    # Define T(x) = sin(x) which has fixed point at 0
    def T(x):
        return 0.5 * np.sin(x)

    def noor_iteration(x0, n_iter, a_seq, b_seq, c_seq):
        """Perform Noor iteration."""
        x = x0
        sequence = [x]

        for n in range(n_iter):
            a_n = a_seq[n] if n < len(a_seq) else a_seq[-1]
            b_n = b_seq[n] if n < len(b_seq) else b_seq[-1]
            c_n = c_seq[n] if n < len(c_seq) else c_seq[-1]

            z = (1 - c_n) * x + c_n * T(x)
            y = (1 - b_n) * x + b_n * T(z)
            x = (1 - a_n) * x + a_n * T(y)
            sequence.append(x)

        return np.array(sequence)

    # Parameters
    x0 = 2.0
    n_iter = 15
    # Sequences for iteration parameters (can vary)
    a_seq = [0.7] * n_iter
    b_seq = [0.6] * n_iter
    c_seq = [0.5] * n_iter

    seq = noor_iteration(x0, n_iter, a_seq, b_seq, c_seq)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convergence plot
    ax1.plot(seq, 'o-', color=colors['noor'], linewidth=2, markersize=6, label='Noor Iteration')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Fixed Point (x*=0)')
    ax1.set_xlabel('Iteration Number n', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$x_n$', fontsize=12, fontweight='bold')
    ax1.set_title('Noor Iteration Convergence: $T(x) = 0.5\\sin(x)$', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Error plot
    errors = np.abs(seq)
    ax2.semilogy(errors, 'o-', color=colors['noor'], linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration Number n', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$|x_n - x^*|$ (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Decay', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('noor_convergence.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: noor_convergence.pdf")


# ============================================================================
# Figure 4: Phase Space Diagram
# ============================================================================
def fig_phase_space():
    """Show phase space diagram for fixed point iteration."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    x = np.linspace(-0.5, 3, 200)
    T_x = 0.5 * np.sin(x) + 0.3 * x  # Modified T

    # Plot the mapping and identity line
    ax.plot(x, T_x, linewidth=3, color=colors['noor'], label='$y = T(x)$')
    ax.plot(x, x, linewidth=2.5, color='red', linestyle='--', label='$y = x$ (Identity)')

    # Fixed point
    fixed_point_x = x[np.argmin(np.abs(T_x - x))]
    ax.plot(fixed_point_x, fixed_point_x, 'o', markersize=12,
           color='darkred', label=f'Fixed Point $x^*$', zorder=5)

    # Example iteration path
    x0 = 2.5
    x_curr = x0
    path_x = [x0]
    path_y = [0]

    for i in range(6):
        T_x_curr = 0.5 * np.sin(x_curr) + 0.3 * x_curr
        path_x.extend([x_curr, x_curr])
        path_y.extend([T_x_curr, T_x_curr])
        x_curr = T_x_curr
        path_x.append(x_curr)
        path_y.append(T_x_curr)

    ax.plot(path_x, path_y, 'o-', color='green', alpha=0.6, linewidth=1.5,
           markersize=4, label='Iteration Path')
    ax.plot(x0, 0, 'go', markersize=10, label='Starting Point')

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('Phase Space: Fixed Point Iteration', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.savefig('phase_space.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: phase_space.pdf")


# ============================================================================
# Figure 5: Cone Illustration
# ============================================================================
def fig_cone_illustration():
    """Illustrate cones in Banach spaces."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 2D cone
    ax = axes[0]
    theta = np.linspace(0, np.pi/3, 100)
    x_cone = np.concatenate([np.cos(theta), [0]])
    y_cone = np.concatenate([np.sin(theta), [0]])

    ax.fill(x_cone, y_cone, color=colors['noor'], alpha=0.3, edgecolor=colors['noor'], linewidth=2)
    ax.plot([0, 1], [0, 0], 'k-', linewidth=2)
    ax.plot([0, np.cos(np.pi/3)], [0, np.sin(np.pi/3)], 'k-', linewidth=2)

    # Add example points in cone
    for i in range(5):
        t = i / 4
        x_pt = t * np.cos(np.pi/6)
        y_pt = t * np.sin(np.pi/6)
        ax.plot(x_pt, y_pt, 'o', color='darkred', markersize=8)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1)
    ax.set_aspect('equal')
    ax.set_title('Cone K in $\\mathbb{R}^2$', fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.text(0.3, 0.15, 'Cone K', fontsize=12, fontweight='bold', color=colors['noor'])

    # Right: Properties of cones
    ax = axes[1]
    ax.axis('off')
    properties_text = """
Cone Properties (Definition 5.70)

(i)  Closed set

(ii) Convexity: If u, v ∈ K, then
     αu + βv ∈ K for all α,β ≥ 0

(iii) x, -x ∈ K ⟹ x = 0
     (Only origin in both K and -K)

Normal Cone (Definition 5.73)
A cone K is normal if ∃δ > 0:
||e₁ + e₂|| ≥ δ,  for e₁,e₂ ∈ K, ||e₁||=||e₂||=1
"""
    ax.text(0.05, 0.95, properties_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('cone_illustration.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: cone_illustration.pdf")


# ============================================================================
# Figure 6: Convergence Rates
# ============================================================================
def fig_convergence_rates():
    """Compare convergence rates of different iteration methods."""
    n = np.arange(0, 20)

    # Simulate convergence rates
    linear_conv = 0.8 ** n
    super_linear = 0.5 ** n
    quadratic = 0.3 ** (n**1.5)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.semilogy(n, linear_conv, 'o-', linewidth=2.5, markersize=6,
               label='Linear Convergence', color=colors['mann'])
    ax.semilogy(n, super_linear, 's-', linewidth=2.5, markersize=6,
               label='Super-linear Convergence', color=colors['ishikawa'])
    ax.semilogy(n, quadratic, '^-', linewidth=2.5, markersize=6,
               label='Quadratic Convergence', color=colors['noor'])

    ax.set_xlabel('Iteration Number n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error $||x_n - x^*||$ (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Rates of Fixed Point Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotations
    ax.text(15, 1e-2, 'Faster\nConvergence', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('convergence_rates.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: convergence_rates.pdf")


# ============================================================================
# Figure 7: Parameter Sensitivity
# ============================================================================
def fig_parameter_sensitivity():
    """Show how parameters affect convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Effect of a_n parameter
    ax = axes[0]
    a_values = np.linspace(0, 1, 50)
    iterations_to_convergence = []

    for a in a_values:
        x = 2.0
        for it in range(100):
            x_new = (1 - a) * x + a * 0.5 * np.sin(x)
            if abs(x_new - x) < 1e-6:
                iterations_to_convergence.append(it)
                break
        else:
            iterations_to_convergence.append(100)

    ax.plot(a_values, iterations_to_convergence, linewidth=2.5, color=colors['noor'], marker='o',
           markersize=4, alpha=0.7)
    ax.set_xlabel('Parameter $a_n$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.set_title('Effect of $a_n$ on Convergence Speed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Optimal?')

    # Effect of sequence type
    ax = axes[1]
    n_iter = 20
    n = np.arange(n_iter)

    constant_seq = np.ones(n_iter) * 0.7
    decreasing_seq = 0.7 / (1 + 0.1 * n)
    increasing_seq = 0.7 * (1 - np.exp(-0.2 * n))

    ax.plot(n, constant_seq, 'o-', linewidth=2, markersize=5, label='Constant', color=colors['mann'])
    ax.plot(n, decreasing_seq, 's-', linewidth=2, markersize=5, label='Decreasing', color=colors['ishikawa'])
    ax.plot(n, increasing_seq, '^-', linewidth=2, markersize=5, label='Increasing', color=colors['noor'])

    ax.set_xlabel('Iteration Number n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Value $a_n$', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sequences', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Saved: parameter_sensitivity.pdf")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Figures for Chapter 4c: Noor Iteration")
    print("="*60 + "\n")

    fig_noor_iteration_process()
    fig_iteration_methods_comparison()
    fig_noor_convergence()
    fig_phase_space()
    fig_cone_illustration()
    fig_convergence_rates()
    fig_parameter_sensitivity()

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60 + "\n")
