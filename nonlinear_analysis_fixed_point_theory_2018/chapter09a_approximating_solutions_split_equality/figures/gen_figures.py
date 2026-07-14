#!/usr/bin/env python3
"""Generate figures for Chapter 9a: Approximating Solutions"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.linewidth'] = 1.0

def create_convergence_plot():
    """Create convergence history plot for matrix equation"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulate convergence behavior
    iterations = np.arange(1, 36)
    error = 250 * np.exp(-0.15 * (iterations - 1)) + 5 * np.sin(iterations/10)
    error = np.maximum(error, 0.1)

    ax.semilogy(iterations, error, 'b-o', linewidth=2, markersize=4, label='Residual Error')
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title('Convergence History of Fixed Point Algorithm', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('convergence_history.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: convergence_history.pdf")

def create_banach_space_diagram():
    """Create Banach space geometry visualization"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw Banach space representation
    # Closed set C
    c_circle = Circle((0, 0), 2, color='lightblue', alpha=0.3, linewidth=2,
                       edgecolor='blue', linestyle='--', label='Closed set C')
    ax.add_patch(c_circle)

    # Point z outside C
    ax.plot(3.5, 0, 'ro', markersize=12, label='Point z (outside C)', zorder=5)
    ax.text(3.7, 0.3, 'z', fontsize=14, fontweight='bold')

    # Closed ball K(z,r)
    k_circle = Circle((3.5, 0), 2, color='lightyellow', alpha=0.3, linewidth=2,
                       edgecolor='orange', linestyle='--', label='Closed ball K(z,r)')
    ax.add_patch(k_circle)

    # Point p in intersection
    p_point = (2, 0.8)
    ax.plot(p_point[0], p_point[1], 'g*', markersize=20, label='Fixed point p in C ∩ ∂D(x,K)', zorder=5)

    # Sequence points
    seq_x = [0, 0.5, 1, 1.5, 1.8]
    seq_y = [0, 0.3, 0.5, 0.6, 0.7]
    ax.plot(seq_x, seq_y, 'k-o', linewidth=1.5, markersize=6, alpha=0.6, label='Sequence {xₙ}')

    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('X', fontsize=12, fontweight='bold')
    ax.set_title('Banach Space Geometry: Closed Sets and Convergence', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('banach_space_geometry.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: banach_space_geometry.pdf")

def create_nash_equilibrium():
    """Create Nash equilibrium payoff matrix visualization"""
    fig = plt.figure(figsize=(10, 7))

    # Two-person game payoff matrix
    strategies1 = ['Strat 1', 'Strat 2', 'Strat 3']
    strategies2 = ['Choice A', 'Choice B', 'Choice C', 'Choice D']

    # Random payoff matrix for illustration
    np.random.seed(42)
    payoff_p1 = np.random.randint(1, 10, (3, 4))
    payoff_p2 = np.random.randint(1, 10, (3, 4))

    # Create heatmap for Player 1
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(payoff_p1, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(strategies2)))
    ax1.set_yticks(range(len(strategies1)))
    ax1.set_xticklabels(strategies2, fontsize=9)
    ax1.set_yticklabels(strategies1, fontsize=9)

    # Annotate with values
    for i in range(len(strategies1)):
        for j in range(len(strategies2)):
            text = ax1.text(j, i, str(payoff_p1[i, j]),
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax1.set_title('Player 1 Payoffs\n(Maximizing Player)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Player 2 Choices', fontsize=11)
    ax1.set_ylabel('Player 1 Strategies', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Payoff')

    # Create heatmap for Player 2
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(payoff_p2, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(strategies2)))
    ax2.set_yticks(range(len(strategies1)))
    ax2.set_xticklabels(strategies2, fontsize=9)
    ax2.set_yticklabels(strategies1, fontsize=9)

    # Annotate with values
    for i in range(len(strategies1)):
        for j in range(len(strategies2)):
            text = ax2.text(j, i, str(payoff_p2[i, j]),
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax2.set_title('Player 2 Payoffs\n(Minimizing Player)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Player 2 Choices', fontsize=11)
    ax2.set_ylabel('Player 1 Strategies', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Payoff')

    plt.suptitle('Two-Person Zero-Sum Game: Nash Equilibrium Analysis',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('nash_equilibrium.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: nash_equilibrium.pdf")

def create_control_theory_diagram():
    """Create control theory system diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # System components
    time = np.linspace(0, 5, 100)

    # Reference trajectory
    reference = 5 * np.sin(time)

    # Controlled response (approaching reference)
    controlled = reference + 2 * np.exp(-0.5 * time) * np.cos(3 * time)

    # Cost function
    ax_cost = plt.subplot(2, 1, 1)
    ax_cost.plot(time, reference, 'r--', linewidth=2, label='Reference trajectory', alpha=0.7)
    ax_cost.plot(time, controlled, 'b-', linewidth=2, label='Controlled system response')
    ax_cost.fill_between(time, reference, controlled, alpha=0.2, color='green')
    ax_cost.set_ylabel('System State x(t)', fontsize=11, fontweight='bold')
    ax_cost.set_title('Optimal Control Theory: System Dynamics', fontsize=12, fontweight='bold')
    ax_cost.legend(fontsize=10)
    ax_cost.grid(True, alpha=0.3)

    # Cost functional
    ax_cost_func = plt.subplot(2, 1, 2)
    cumulative_cost = np.cumsum(np.abs(reference - controlled)) * (time[1] - time[0])
    ax_cost_func.fill_between(time, 0, cumulative_cost, alpha=0.3, color='orange')
    ax_cost_func.plot(time, cumulative_cost, 'orange', linewidth=2.5, label='Cumulative cost C(x,α)')
    ax_cost_func.set_xlabel('Time t', fontsize=11, fontweight='bold')
    ax_cost_func.set_ylabel('Cost', fontsize=11, fontweight='bold')
    ax_cost_func.set_title('Cumulative Cost Functional', fontsize=12, fontweight='bold')
    ax_cost_func.legend(fontsize=10)
    ax_cost_func.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('control_theory_system.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: control_theory_system.pdf")

def create_matrix_eigenvalues():
    """Create matrix eigenvalue spectrum visualization"""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Create a sample matrix
    np.random.seed(123)
    n = 4
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Plot 1: Eigenvalue spectrum
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue', fontsize=11)
    ax1.set_title('Eigenvalue Spectrum', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight dominant eigenvalue
    bars[0].set_color('red')
    bars[0].set_alpha(0.9)

    # Plot 2: Iteration convergence with contraction factor
    ax2 = plt.subplot(2, 2, 2)
    iterations = np.arange(0, 20)
    lambda_max = np.max(np.abs(eigenvalues))
    errors = (lambda_max ** iterations)
    ax2.semilogy(iterations, errors, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration n', fontsize=11)
    ax2.set_ylabel('Error ||xₙ - x*||', fontsize=11)
    ax2.set_title(f'Contraction with λ = {lambda_max:.3f}', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Matrix structure
    ax3 = plt.subplot(2, 2, 3)
    im = ax3.imshow(A, cmap='RdBu_r', aspect='auto')
    ax3.set_title('Matrix A Structure', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Column', fontsize=11)
    ax3.set_ylabel('Row', fontsize=11)
    plt.colorbar(im, ax=ax3, label='Value')

    # Plot 4: Linear system solution space
    ax4 = plt.subplot(2, 2, 4)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    # Create contours for Ax = b (simplified illustration)
    Z = X**2 + 0.5*Y**2 - 2*X + Y
    contours = ax4.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax4.clabel(contours, inline=True, fontsize=8)
    filled = ax4.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax4.plot(1, -0.5, 'r*', markersize=20, label='Solution x*')
    ax4.set_xlabel('x₁', fontsize=11)
    ax4.set_ylabel('x₂', fontsize=11)
    ax4.set_title('Solution Space (2D view)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=10)
    plt.colorbar(filled, ax=ax4, label='||Ax - b||²')

    plt.suptitle('Matrix Properties and Convergence Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matrix_eigenvalues.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: matrix_eigenvalues.pdf")

def create_variational_inequality():
    """Create visualization for variational inequality"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Two obstacle problem visualization
    x = np.linspace(0, 10, 200)

    # Upper and lower obstacles
    upper_obstacle = 3 + 0.2*x - 0.01*x**2
    lower_obstacle = 0.5 + 0.1*x - 0.005*x**2

    # Solution that respects obstacles
    solution = np.maximum(lower_obstacle, np.minimum(upper_obstacle,
                                                      2 + 0.15*x - 0.008*x**2))

    # Unconstrained solution
    unconstrained = 2 + 0.15*x - 0.008*x**2

    ax.fill_between(x, lower_obstacle, upper_obstacle, alpha=0.2, color='lightblue',
                     label='Feasible region')
    ax.plot(x, upper_obstacle, 'r--', linewidth=2.5, label='Upper obstacle φ(x)')
    ax.plot(x, lower_obstacle, 'b--', linewidth=2.5, label='Lower obstacle μ(x)')
    ax.plot(x, solution, 'g-', linewidth=3, label='Solution u(x) of VI(9.26)', alpha=0.8)
    ax.plot(x, unconstrained, 'orange', linewidth=2, linestyle=':', label='Unconstrained Lu - f = 0', alpha=0.7)

    ax.set_xlabel('Spatial domain x', fontsize=12, fontweight='bold')
    ax.set_ylabel('Function value u(x)', fontsize=12, fontweight='bold')
    ax.set_title('Two-Obstacle Variational Inequality Problem', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig('variational_inequality.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: variational_inequality.pdf")

def create_fixed_point_iteration():
    """Create fixed point iteration visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define function and fixed points
    x = np.linspace(0, 2, 500)
    y = np.linspace(0, 2, 500)

    # T(x) = 0.6*x + 0.5*sin(x) + 0.3
    T_x = 0.6*x + 0.5*np.sin(x) + 0.3

    # Plot the function and identity line
    ax.plot(x, x, 'k--', linewidth=2, label='y = x (fixed points)')
    ax.plot(x, T_x, 'b-', linewidth=2.5, label='y = T(x)', alpha=0.8)

    # Find intersection (fixed point)
    diff = np.abs(x - T_x)
    fixed_point_idx = np.argmin(diff)
    x_fixed = x[fixed_point_idx]
    y_fixed = T_x[fixed_point_idx]

    ax.plot(x_fixed, y_fixed, 'r*', markersize=25, label=f'Fixed point x* ≈ {x_fixed:.3f}', zorder=5)

    # Show iteration steps
    x_iter = 0.1
    for i in range(8):
        y_iter = 0.6*x_iter + 0.5*np.sin(x_iter) + 0.3
        ax.plot([x_iter, x_iter], [x_iter, y_iter], 'g-', linewidth=1.5, alpha=0.6)
        ax.plot([x_iter, y_iter], [y_iter, y_iter], 'g-', linewidth=1.5, alpha=0.6)
        ax.plot(x_iter, y_iter, 'go', markersize=5, alpha=0.6)
        x_iter = y_iter

    # Final point
    ax.plot(x_iter, y_iter, 'go', markersize=7, label='Final iterate', zorder=4)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('Fixed Point Iteration: Graphical Method', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fixed_point_iteration.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fixed_point_iteration.pdf")

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 9a...")
    create_convergence_plot()
    create_banach_space_diagram()
    create_nash_equilibrium()
    create_control_theory_diagram()
    create_matrix_eigenvalues()
    create_variational_inequality()
    create_fixed_point_iteration()
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
