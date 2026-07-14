#!/usr/bin/env python3
"""
Figure generation for Chapter 26: Resolvents & Fixed Points
Convex Analysis and Monotone Operator Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

# Configure matplotlib for LaTeX rendering
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def fig_resolvent_operator():
    """Figure: Resolvent operator visualization and properties"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Fixed point iteration via resolvent
    ax = axes[0]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(r'Resolvent Operator $J_A = (\mathrm{Id} + A)^{-1}$')

    # Example iteration points
    x_vals = [0.5, 2, 3.5]
    y_vals = [0.5, 1.5, 2.5]

    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        circle = Circle((x, y), 0.2, color='steelblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y-0.5, f'$x_{i}$', ha='center', fontsize=10)

    # Draw iteration arrows
    for i in range(len(x_vals)-1):
        arrow = FancyArrowPatch((x_vals[i]+0.2, y_vals[i]),
                               (x_vals[i+1]-0.2, y_vals[i+1]),
                               arrowstyle='->', mutation_scale=20, lw=2, color='darkblue')
        ax.add_patch(arrow)

    ax.text(3, 4.5, r'Fixed point: $x^* = J_A(x^*)$', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right plot: Douglas-Rachford splitting diagram
    ax = axes[1]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'Douglas-Rachford Splitting Operator')

    # Stages of DR algorithm
    stages = [
        (0.5, 4.5, 'Input: $x_n$', 'lightblue'),
        (0.5, 3.5, r'$y_n = J_A(x_n)$', 'lightgreen'),
        (0.5, 2.5, r'$z_n = J_B(2y_n - x_n)$', 'lightyellow'),
        (0.5, 1.5, r'$x_{n+1} = x_n + z_n - y_n$', 'lightcoral'),
    ]

    for x, y, text, color in stages:
        box = Rectangle((x, y-0.3), 4, 0.6,
                       facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(2.5, y, text, ha='center', va='center', fontsize=10, weight='bold')

        if y > 1.5:
            arrow = FancyArrowPatch((2.5, y-0.35), (2.5, y-0.65),
                                   arrowstyle='->', mutation_scale=15, lw=1.5)
            ax.add_patch(arrow)

    ax.text(2.5, 0.5, r'Repeat until convergence',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('resolvent_operator.pdf', dpi=150)
    print("Saved: resolvent_operator.pdf")
    plt.close()


def fig_fixed_point_theory():
    """Figure: Fixed point existence and uniqueness conditions"""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, r'Fixed Point Theory in Monotone Operator Theory',
            ha='center', fontsize=14, weight='bold')

    # Main theorem box
    main_box = Rectangle((0.5, 7), 9, 2,
                         facecolor='lightblue', edgecolor='darkblue', linewidth=2)
    ax.add_patch(main_box)
    ax.text(5, 8.5, r'$A: \mathcal{H} \to 2^{\mathcal{H}}$ maximally monotone operator',
            ha='center', fontsize=11, style='italic')
    ax.text(5, 7.9, r'$\Rightarrow$ Resolvent $J_A = (\mathrm{Id} + A)^{-1}$ is firmly nonexpansive',
            ha='center', fontsize=11, weight='bold')
    ax.text(5, 7.3, r'Every fixed point of $J_A$ is a zero of $A$',
            ha='center', fontsize=10)

    # Key properties
    y_pos = 6.5
    properties = [
        (r'Nonexpansivity: $\|J_A(x) - J_A(y)\| \leq \|x - y\|$', 'lightgreen'),
        (r'Firmly nonexpansive: $\|J_A(x) - J_A(y)\|^2 + \gamma \|x - y - (J_A(x) - J_A(y))\|^2 \leq \gamma \|x - y\|^2$', 'lightyellow'),
        (r'Fixed points: $\mathrm{Fix}(J_A) = \mathrm{zer}(A)$', 'lightcoral'),
    ]

    for i, (prop, color) in enumerate(properties):
        y = y_pos - i * 1.3
        box = Rectangle((0.7, y-0.35), 8.6, 0.7,
                       facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y, prop, ha='center', va='center', fontsize=10)

    # Applications section
    ax.text(0.7, 2.8, r'Applications:', fontsize=11, weight='bold')
    applications = [
        r'$\bullet$ Proximal Point Algorithm: $x_{n+1} = J_A(x_n)$',
        r'$\bullet$ Douglas-Rachford Splitting: $\mathrm{Fix}(T_{A,B})$ solves $0 \in Ax + Bx$',
        r'$\bullet$ Forward-Backward Splitting: Variational inequalities',
    ]

    for i, app in enumerate(applications):
        ax.text(0.9, 2.3 - i*0.5, app, fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('fixed_point_theory.pdf', dpi=150)
    print("Saved: fixed_point_theory.pdf")
    plt.close()


def fig_convergence_behavior():
    """Figure: Convergence behavior of splitting algorithms"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Convergence rate comparison
    ax = axes[0]
    n = np.arange(0, 30)
    error_linear = 0.9**n
    error_superlinear = (0.9**n) * (1 + 0.5*n)**(-1)

    ax.semilogy(n, error_linear, 'o-', linewidth=2, markersize=6,
                label='Linear convergence', color='steelblue')
    ax.semilogy(n, error_superlinear, 's--', linewidth=2, markersize=6,
                label='Superlinear convergence', color='coral')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel(r'Error $\|x_n - x^*\|$')
    ax.set_title('Convergence Rates')
    ax.legend(fontsize=10)
    ax.set_ylim([1e-8, 1])

    # Right: Distance to solution vs residual
    ax = axes[1]
    x = np.linspace(0, 3, 100)
    solution_distance = np.exp(-x**2)
    residual = np.abs(np.sin(x)) * np.exp(-0.5*x)

    ax.plot(x, solution_distance, linewidth=2.5, label=r'$\|x_n - x^*\|$', color='steelblue')
    ax.plot(x, residual, '--', linewidth=2.5, label=r'$\|Ax_n + Bx_n\|$', color='coral')
    ax.fill_between(x, solution_distance, alpha=0.2, color='steelblue')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Parameter $t$')
    ax.set_ylabel('Magnitude')
    ax.set_title('Solution Error vs Residual')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('convergence_behavior.pdf', dpi=150)
    print("Saved: convergence_behavior.pdf")
    plt.close()


def fig_monotone_operators():
    """Figure: Visual representation of monotone operators"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Top-left: Graph of a monotone operator
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 100)
    y = np.tanh(x)  # monotone increasing
    ax.plot(x, y, linewidth=2.5, color='steelblue')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$A(x)$ (monotone increasing)')
    ax.set_title('Monotone Increasing Operator')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Top-right: Maximally monotone operator
    ax = axes[0, 1]
    x = np.linspace(-2, 2, 100)
    # Subdifferential-like visualization
    for x0 in np.linspace(-1.5, 1.5, 5):
        y_min = -2 - (x0)**2
        y_max = 2 + (x0)**2
        ax.plot([x0, x0], [y_min, y_max], 'o-', linewidth=1.5,
                color='steelblue', alpha=0.6, markersize=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$A(x)$ (multivalued)')
    ax.set_title('Maximally Monotone Operator')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-6, 6)

    # Bottom-left: Resolvent operator iteration
    ax = axes[1, 0]
    x_vals = np.linspace(-1.5, 1.5, 50)
    y_vals = 0.8 * x_vals  # Simplified resolvent
    ax.plot(x_vals, y_vals, linewidth=2.5, label=r'$J_A(x)$', color='steelblue')
    ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', linewidth=1.5, label='Identity', alpha=0.5)

    # Show fixed point
    fixed_point = [0]
    ax.plot(0, 0, 'ro', markersize=10, label='Fixed point', zorder=5)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$J_A(x)$')
    ax.set_title('Resolvent Operator')
    ax.legend(fontsize=9)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Bottom-right: Split operators
    ax = axes[1, 1]
    ax.axis('off')

    # Text box for split operators
    text_content = r'''
    $\mathbf{Problem:}$ Find $x \in \mathcal{H}$ such that $0 \in Ax + Bx$

    $\mathbf{Douglas\text{-}Rachford:}$
    $y_n = J_A(x_n)$
    $x_{n+1} = x_n + J_B(2y_n - x_n) - y_n$

    $\mathbf{Forward\text{-}Backward:}$
    $y_n = x_n - \gamma B x_n$
    $x_{n+1} = J_{\gamma A}(y_n)$
    '''

    ax.text(0.1, 0.5, text_content, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('monotone_operators.pdf', dpi=150)
    print("Saved: monotone_operators.pdf")
    plt.close()


def fig_spingarn_method():
    """Figure: Spingarn's method of partial inverses"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, r"Spingarn's Method of Partial Inverses",
            ha='center', fontsize=13, weight='bold')

    # Problem statement
    problem_box = Rectangle((0.5, 8), 9, 1.2,
                           facecolor='lightblue', edgecolor='darkblue', linewidth=1.5)
    ax.add_patch(problem_box)
    ax.text(5, 8.8, r'Given: $A: \mathcal{H} \to 2^{\mathcal{H}}$ maximally monotone, closed subspace $V$',
            ha='center', fontsize=10)
    ax.text(5, 8.3, r'Find: $x \in V, u \in V^{\perp}$ such that $u \in Ax$',
            ha='center', fontsize=10)

    # Algorithm steps
    y_pos = 7.2
    steps = [
        r'$\mathbf{Step\ 1}$: Initialize $x_0 \in V, u_0 \in V^{\perp}$',
        r'$\mathbf{Step\ 2}$: Compute $y_n = J_A(x_n + u_n)$',
        r'$\mathbf{Step\ 3}$: Compute $v_n = x_n + u_n - y_n$',
        r'$\mathbf{Step\ 4}$: Project $(x_{n+1}, u_{n+1}) = (P_V y_n, P_{V^{\perp}} v_n)$',
    ]

    for i, step in enumerate(steps):
        y = y_pos - i * 1.0
        box = Rectangle((0.7, y-0.35), 8.6, 0.7,
                       facecolor='lightyellow', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y, step, ha='center', va='center', fontsize=9.5)

    # Convergence result
    conv_box = Rectangle((0.5, 1.5), 9, 1,
                        facecolor='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    ax.add_patch(conv_box)
    ax.text(5, 2.2, r'$\mathbf{Convergence:}$ $(x_n, u_n) \to (x, u)$ where $u \in Ax$',
            ha='center', fontsize=10, weight='bold')
    ax.text(5, 1.7, r'Solution to the constrained problem: $x \in V$ and $u \in V^{\perp}$',
            ha='center', fontsize=9, style='italic')

    # Key insight box
    insight_box = Rectangle((0.5, 0.1), 9, 1.2,
                           facecolor='wheat', edgecolor='orange', linewidth=1.5)
    ax.add_patch(insight_box)
    ax.text(5, 0.95, r'$\mathbf{Key\ Insight:}$',
            ha='center', fontsize=10, weight='bold')
    ax.text(5, 0.45, r'Decomposes constrained problem into independent subproblems',
            ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('spingarn_method.pdf', dpi=150)
    print("Saved: spingarn_method.pdf")
    plt.close()


def fig_numerical_example():
    """Figure: Numerical example of algorithm convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Simulate proximal point algorithm on convex function
    # Minimize f(x) = 0.5 * x^T Q x where Q is positive definite
    Q = np.array([[2, 0.5], [0.5, 1]])

    # Algorithm: x_{n+1} = J_{gamma A}(x_n) where A = nabla f
    def proximal_step(x, gamma=0.1):
        return np.linalg.solve(np.eye(2) + gamma * Q, x)

    # Run iterations
    x = np.array([2.0, 2.0])
    trajectory = [x.copy()]
    errors = [np.linalg.norm(x)]

    for _ in range(50):
        x = proximal_step(x)
        trajectory.append(x.copy())
        errors.append(np.linalg.norm(x))

    trajectory = np.array(trajectory)

    # Plot 1: Trajectory in 2D
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', linewidth=1.5,
            markersize=4, color='steelblue', alpha=0.7)
    ax.plot(0, 0, 'r*', markersize=15, label='Fixed point (solution)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Algorithm Trajectory')
    ax.legend()

    # Plot 2: Convergence history
    ax = axes[0, 1]
    ax.semilogy(errors, 'o-', linewidth=2, markersize=5, color='steelblue')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel(r'$\|x_n - x^*\|$')
    ax.set_title('Error vs Iteration')

    # Plot 3: Step size vs iteration
    ax = axes[1, 0]
    step_sizes = [np.linalg.norm(trajectory[i+1] - trajectory[i])
                  for i in range(len(trajectory)-1)]
    ax.semilogy(step_sizes, 's-', linewidth=2, markersize=5, color='coral')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel(r'Step size $\|x_{n+1} - x_n\|$')
    ax.set_title('Step Size Decrease')

    # Plot 4: Function value along trajectory
    ax = axes[1, 1]
    fvals = [0.5 * x @ Q @ x for x in trajectory]
    ax.semilogy(fvals, '^-', linewidth=2, markersize=5, color='green')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel(r'$f(x_n) = \frac{1}{2} x_n^T Q x_n$')
    ax.set_title('Objective Value')

    plt.tight_layout()
    plt.savefig('numerical_example.pdf', dpi=150)
    print("Saved: numerical_example.pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating figures for Chapter 26...")
    fig_resolvent_operator()
    fig_fixed_point_theory()
    fig_convergence_behavior()
    fig_monotone_operators()
    fig_spingarn_method()
    fig_numerical_example()
    print("All figures generated successfully!")
