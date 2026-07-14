#!/usr/bin/env python3
"""
Generate figures for Chapter 2b: Banach Contraction Mapping
Illustrates key concepts and numerical examples
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def save_pdf(fig, filename):
    """Save figure as PDF"""
    fig.savefig(f'figures/{filename}', bbox_inches='tight', dpi=300, format='pdf')
    plt.close(fig)

# Figure 1: Illustration of contraction mapping
def figure_contraction_mapping():
    """Show how contraction mapping shrinks distances"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Graphical interpretation
    x = np.linspace(0, 1, 1000)

    # Plot identity line
    ax1.plot(x, x, 'k--', label='y = x', linewidth=2, alpha=0.7)

    # Plot contraction mapping T(x) = 0.5*x + 0.2
    T = lambda x: 0.5 * x + 0.2
    y = T(x)
    ax1.plot(x, y, 'b-', label='T(x) = 0.5x + 0.2', linewidth=2.5)

    # Mark fixed point
    fixed_point = 0.4
    ax1.plot(fixed_point, fixed_point, 'ro', markersize=10, label=f'Fixed point x* = {fixed_point}')

    # Show iterations from starting point x0 = 0
    x0 = 0
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for i in range(5):
        x_new = T(x0)
        ax1.plot([x0, x0], [x0, x_new], color=colors[i], linewidth=1.5, alpha=0.8)
        ax1.plot([x0, x_new], [x_new, x_new], color=colors[i], linewidth=1.5, alpha=0.8)
        ax1.plot(x_new, x_new, 'o', color=colors[i], markersize=6)
        x0 = x_new

    ax1.set_xlim(-0.05, 1)
    ax1.set_ylim(-0.05, 1)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('T(x)', fontsize=12)
    ax1.set_title('Graphical Interpretation of Contraction Mapping', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right plot: Distance reduction
    n_iter = 10
    x0 = 0.9
    distances = [abs(x0 - fixed_point)]
    x = x0
    for _ in range(n_iter):
        x = T(x)
        distances.append(abs(x - fixed_point))

    ax2.semilogy(range(len(distances)), distances, 'b-o', linewidth=2, markersize=8, label='$|x_n - x^*|$')

    # Theoretical bound
    alpha = 0.5
    theoretical = [distances[0] * (alpha ** n) for n in range(len(distances))]
    ax2.semilogy(range(len(theoretical)), theoretical, 'r--s', linewidth=2, markersize=6,
                 label=f'$|x_0 - x^*|\\alpha^n$ (α={alpha})', alpha=0.8)

    ax2.set_xlabel('Iteration n', fontsize=12)
    ax2.set_ylabel('Distance to Fixed Point', fontsize=12)
    ax2.set_title('Exponential Convergence', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save_pdf(fig, 'contraction_mapping.pdf')

# Figure 2: Effect of contraction constant
def figure_contraction_constant():
    """Show how different contraction constants affect convergence"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 10, 100)
    fixed_point = 0.4
    x0 = 0.9

    alphas = [0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.cool(np.linspace(0, 1, len(alphas)))

    for alpha, color in zip(alphas, colors):
        distances = [abs(x0 - fixed_point) * (alpha ** n) for n in x]
        ax.semilogy(x, distances, linewidth=2.5, label=f'$\\alpha = {alpha}$', color=color)

    ax.set_xlabel('Iteration n', fontsize=12)
    ax.set_ylabel('Distance to Fixed Point (log scale)', fontsize=12)
    ax.set_title('Effect of Contraction Constant on Convergence Rate',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 10)

    fig.tight_layout()
    save_pdf(fig, 'contraction_constant.pdf')

# Figure 3: Fixed point theorem conditions
def figure_conditions():
    """Illustrate the three main conditions of Banach Fixed Point Theorem"""
    fig = plt.figure(figsize=(14, 10))

    # Condition 1: Complete metric space
    ax1 = plt.subplot(2, 3, 1)

    circle = Circle((0.5, 0.5), 0.35, fill=False, edgecolor='blue', linewidth=2.5)
    ax1.add_patch(circle)

    # Add points
    np.random.seed(42)
    points = np.random.rand(20, 2) * 0.6 + 0.2
    ax1.scatter(points[:, 0], points[:, 1], s=100, alpha=0.6, c='blue')

    ax1.text(0.5, 0.1, 'Complete\nMetric Space', ha='center', fontsize=11, fontweight='bold')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(1) Complete Metric Space $(X, d)$', fontsize=12, fontweight='bold')

    # Condition 2: Continuous mapping
    ax2 = plt.subplot(2, 3, 2)
    x = np.linspace(0, 1, 100)
    y1 = np.sin(np.pi * x)
    ax2.plot(x, y1, 'b-', linewidth=2.5, label='T(x)')
    ax2.plot(x, x, 'k--', alpha=0.5, label='Identity')
    ax2.scatter([0.5], [np.sin(np.pi * 0.5)], s=100, c='red', zorder=5)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('T(x)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(2) Continuous Mapping $T: X \\to X$', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)

    # Condition 3: Contraction property
    ax3 = plt.subplot(2, 3, 3)

    # Show distance shrinking
    points1 = np.array([[0.2, 0.5], [0.6, 0.5]])
    points2 = np.array([[0.35, 0.5], [0.55, 0.5]])

    ax3.scatter(points1[:, 0], points1[:, 1], s=150, c='blue', marker='o', label='$x, y$', zorder=5)
    ax3.scatter(points2[:, 0], points2[:, 1], s=150, c='red', marker='s', label='$T(x), T(y)$', zorder=5)

    ax3.plot([points1[0, 0], points1[1, 0]], [points1[0, 1], points1[1, 1]], 'b-', linewidth=2.5, label='$d(x,y)$')
    ax3.plot([points2[0, 0], points2[1, 0]], [points2[0, 1], points2[1, 1]], 'r-', linewidth=2.5, label='$d(T(x),T(y))$')

    ax3.text(0.4, 0.3, '$d(T(x),T(y)) \\leq \\alpha d(x,y)$\nwith $0 < \\alpha < 1$',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('(3) Contraction Property', fontsize=12, fontweight='bold')

    # Theorem statement
    ax4 = plt.subplot(2, 3, (4, 6))
    ax4.axis('off')

    theorem_text = """
    BANACH CONTRACTION MAPPING THEOREM (Banach Fixed Point Theorem)

    Let $(X, d)$ be a complete metric space and $T: X \\to X$ be a contraction mapping
    with Lipschitz constant $\\alpha \\in (0, 1)$, i.e.,

                    $d(T(x), T(y)) \\leq \\alpha \\, d(x, y)$ for all $x, y \\in X$

    Then:
      1. $T$ has a unique fixed point $x^* \\in X$
      2. For any $x_0 \\in X$, the sequence $\\{x_n\\}$ defined by $x_{n+1} = T(x_n)$ converges to $x^*$
      3. The rate of convergence is exponential: $d(x_n, x^*) \\leq \\alpha^n d(x_0, x^*)$
    """

    ax4.text(0.05, 0.5, theorem_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow',
                                         edgecolor='black', linewidth=2))

    fig.tight_layout()
    save_pdf(fig, 'theorem_conditions.pdf')

# Figure 4: Banach vs Non-Banach examples
def figure_examples():
    """Compare Banach contraction examples with non-contractions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(0, 1, 1000)

    # Example 1: Banach contraction T(x) = 0.5x + 0.2
    ax = axes[0, 0]
    T1 = lambda x: 0.5 * x + 0.2
    ax.plot(x, x, 'k--', alpha=0.5, linewidth=1.5)
    ax.plot(x, T1(x), 'b-', linewidth=2.5, label='$T(x) = 0.5x + 0.2$')
    fixed1 = 0.4
    ax.plot(fixed1, fixed1, 'ro', markersize=10, label=f'Fixed point: $x^* = {fixed1}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x)', fontsize=11)
    ax.set_title('✓ Banach Contraction\n(α = 0.5)', fontsize=11, fontweight='bold', color='green')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Example 2: Too steep - not a contraction
    ax = axes[0, 1]
    T2 = lambda x: 1.5 * x - 0.2
    T2_clipped = np.clip(T2(x), 0, 1)
    ax.plot(x, x, 'k--', alpha=0.5, linewidth=1.5)
    ax.plot(x, T2(x), 'r-', linewidth=2.5, label='$T(x) = 1.5x - 0.2$')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x)', fontsize=11)
    ax.set_title('✗ Not a Contraction\n(α = 1.5 > 1)', fontsize=11, fontweight='bold', color='red')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Example 3: Good contraction
    ax = axes[1, 0]
    T3 = lambda x: 0.3 * x + 0.35
    ax.plot(x, x, 'k--', alpha=0.5, linewidth=1.5)
    ax.plot(x, T3(x), 'g-', linewidth=2.5, label='$T(x) = 0.3x + 0.35$')
    fixed3 = 0.5
    ax.plot(fixed3, fixed3, 'ro', markersize=10, label=f'Fixed point: $x^* = {fixed3}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x)', fontsize=11)
    ax.set_title('✓ Stronger Contraction\n(α = 0.3)', fontsize=11, fontweight='bold', color='green')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Example 4: Error estimation
    ax = axes[1, 1]
    n = np.arange(0, 15)
    x0 = 0.9

    # Different alphas
    for alpha, label in [(0.3, '0.3'), (0.5, '0.5'), (0.7, '0.7')]:
        error = x0 * (alpha ** n)
        ax.semilogy(n, error, 'o-', linewidth=2, markersize=6, label=f'$\\alpha = {label}$')

    ax.set_xlabel('Iteration n', fontsize=11)
    ax.set_ylabel('Error $|x_n - x^*|$ (log scale)', fontsize=11)
    ax.set_title('Error Decay for Different α Values', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    save_pdf(fig, 'examples.pdf')

# Figure 5: Picard iteration convergence
def figure_picard_iteration():
    """Illustrate Picard iteration algorithm"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Step-by-step iteration
    x = np.linspace(-0.5, 2, 1000)
    T = lambda x: np.exp(-x)
    x_fixed = np.e ** (-x_fixed) if False else 0.567  # Approximate

    ax1.plot(x, x, 'k--', label='y = x', linewidth=2, alpha=0.7)
    ax1.plot(x, T(x), 'b-', label='$T(x) = e^{-x}$', linewidth=2.5)

    x0 = 1.5
    n_steps = 5
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_steps))

    for i in range(n_steps):
        x_new = T(x0)
        ax1.plot([x0, x0], [x0, x_new], color=colors[i], linewidth=1.5, alpha=0.8)
        ax1.plot([x0, x_new], [x_new, x_new], color=colors[i], linewidth=1.5, alpha=0.8)
        ax1.plot(x_new, x_new, 'o', color=colors[i], markersize=8)
        x0 = x_new

    ax1.plot(x_fixed, x_fixed, 'r*', markersize=20, label='Fixed point', zorder=10)
    ax1.set_xlim(-0.2, 2)
    ax1.set_ylim(-0.2, 2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Picard Iteration: Graphical', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right: Algorithm box
    ax2.axis('off')

    algorithm = """
    PICARD ITERATION ALGORITHM
    ═════════════════════════════════════════

    Input: Complete metric space (X, d),
           contraction mapping T: X → X with constant α < 1,
           starting point x₀ ∈ X, tolerance ε > 0

    Output: Approximate fixed point x*

    ─────────────────────────────────────────
    Algorithm:

    1.  Set n = 0

    2.  While |xₙ₊₁ - xₙ| > ε  do:

        a) Compute xₙ₊₁ = T(xₙ)

        b) If |xₙ₊₁ - xₙ| ≤ ε·(1-α)/α then
                return xₙ₊₁  (converged)

        c) Set n ← n + 1

    3.  If n exceeds max iterations:
            warn "Convergence may not be achieved"

    ─────────────────────────────────────────
    Properties:
    • Guaranteed convergence
    • Exponential error decay
    • Error estimate: |xₙ - x*| ≤ αⁿ|x₀ - x*|
    """

    ax2.text(0.05, 0.95, algorithm, fontsize=9, verticalalignment='top',
            family='monospace', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, linewidth=2))

    fig.tight_layout()
    save_pdf(fig, 'picard_iteration.pdf')

# Figure 6: Numerical example with error analysis
def figure_numerical_example():
    """Detailed numerical example"""
    fig = plt.figure(figsize=(13, 9))

    # Title
    fig.suptitle('Numerical Example: Solving x = 0.8·cos(x)',
                fontsize=14, fontweight='bold', y=0.98)

    # Left: Iteration table
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis('off')

    T = lambda x: 0.8 * np.cos(x)
    x0 = 1.0
    alpha = 0.8

    iterations = []
    x = x0
    for n in range(10):
        x_new = T(x)
        error = abs(x_new - x)
        ratio = error / abs(x - iterations[n-1][1]) if n > 0 else 0
        iterations.append((n, x, x_new, error))
        x = x_new

    table_text = "n    xₙ         xₙ₊₁       |xₙ₊₁ - xₙ|   Rate\n"
    table_text += "─" * 50 + "\n"
    for n, xn, xnp1, err in iterations[:6]:
        if n == 0:
            table_text += f"{n}    {xn:.6f}    {xnp1:.6f}   {err:.2e}     -\n"
        else:
            ratio = err / iterations[n-1][3]
            table_text += f"{n}    {xnp1:.6f}    T(xₙ)      {err:.2e}     {ratio:.4f}\n"

    ax1.text(0.05, 0.95, table_text, fontsize=9, verticalalignment='top',
            family='monospace', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.set_title('Iteration Table', fontsize=11, fontweight='bold')

    # Middle: Convergence plot
    ax2 = plt.subplot(2, 3, 2)
    n_vals = np.arange(len(iterations))
    x_vals = [it[1] for it in iterations]
    ax2.plot(n_vals, x_vals, 'bo-', linewidth=2, markersize=8, label='xₙ')
    ax2.axhline(y=iterations[-1][1], color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Limit')
    ax2.set_xlabel('Iteration n', fontsize=11)
    ax2.set_ylabel('xₙ', fontsize=11)
    ax2.set_title('Sequence Convergence', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Right: Error decay
    ax3 = plt.subplot(2, 3, 3)
    errors = [abs(it[1] - it[2]) for it in iterations]
    ax3.semilogy(n_vals, errors, 'rs-', linewidth=2, markersize=8, label='Actual error')

    # Theoretical bound
    theo_bound = [errors[0] * (alpha ** n) for n in n_vals]
    ax3.semilogy(n_vals, theo_bound, 'b--o', linewidth=2, markersize=6, alpha=0.7, label=f'Bound: α^n·e₀ (α={alpha})')

    ax3.set_xlabel('Iteration n', fontsize=11)
    ax3.set_ylabel('Error (log scale)', fontsize=11)
    ax3.set_title('Error Decay Rate', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    # Bottom: Graphical solution
    ax4 = plt.subplot(2, 1, 2)

    x = np.linspace(0, np.pi, 1000)
    y_id = x
    y_cos = 0.8 * np.cos(x)

    ax4.plot(x, y_id, 'k--', label='y = x', linewidth=2, alpha=0.7)
    ax4.plot(x, y_cos, 'b-', label='y = 0.8·cos(x)', linewidth=2.5)

    x_sol = iterations[-1][2]
    ax4.plot(x_sol, x_sol, 'r*', markersize=25, label=f'Fixed point: x* ≈ {x_sol:.6f}', zorder=10)

    # Show a few iterations
    x_curr = 1.0
    for i in range(3):
        x_next = T(x_curr)
        ax4.plot([x_curr, x_curr], [x_curr, x_next], 'g-', linewidth=1.5, alpha=0.6)
        ax4.plot([x_curr, x_next], [x_next, x_next], 'g-', linewidth=1.5, alpha=0.6)
        x_curr = x_next

    ax4.set_xlim(0, np.pi)
    ax4.set_ylim(0, np.pi)
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.set_title('Graphical Solution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    fig.tight_layout()
    save_pdf(fig, 'numerical_example.pdf')

# Figure 7: Applications overview
def figure_applications():
    """Show key applications of Banach Fixed Point Theorem"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    applications_text = """
    APPLICATIONS OF BANACH FIXED POINT THEOREM
    ═════════════════════════════════════════════════════════════════════════════════════════════════════════

    1. DIFFERENTIAL EQUATIONS
       • Existence and uniqueness of solutions for initial value problems (IVP)
       • Picard-Lindelöf theorem for ODEs
       • Nonlinear differential equations
       Example: y' = f(t,y),  y(t₀) = y₀

    2. INTEGRAL EQUATIONS
       • Volterra integral equations (existence of solutions)
       • Fredholm equations with small perturbations
       • Hammerstein equations
       Example: x(t) = g(t) + λ∫ₐᵇ K(t,s)f(s,x(s))ds

    3. NUMERICAL ANALYSIS
       • Newton-Raphson method (convergence analysis)
       • Fixed point iteration methods
       • Picard iteration for solving equations
       • Error estimation and convergence rates

    4. NONLINEAR ANALYSIS
       • Existence of solutions to nonlinear operator equations
       • Fixed point iterations in Banach spaces
       • Perturbation theory

    5. ECONOMICS & GAME THEORY
       • Nash equilibrium existence in games
       • Market equilibrium models
       • Dynamic programming and optimal control

    6. FRACTIONAL CALCULUS
       • Existence of solutions for fractional differential equations
       • Applied to modeling in physics and biology

    ═════════════════════════════════════════════════════════════════════════════════════════════════════════

    KEY ADVANTAGES:
    ✓ Constructive method: provides iterative algorithm to find fixed point
    ✓ Error estimation: quantifies approximation error at each step
    ✓ Applicable in infinite dimensional spaces (Banach spaces)
    ✓ Efficient computation with guaranteed convergence
    """

    ax.text(0.02, 0.98, applications_text, fontsize=9.5, verticalalignment='top',
           family='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, linewidth=2.5, pad=1))

    fig.tight_layout()
    save_pdf(fig, 'applications.pdf')

# Figure 8: Comparison with other fixed point theorems
def figure_comparison():
    """Compare different fixed point theorems"""
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    ax.axis('off')

    comparison_text = """
    COMPARISON OF FIXED POINT THEOREMS
    ═════════════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  THEOREM              │  SPACE TYPE  │  CONDITION       │  CONSTRUCTIVE  │  RATE      │  UNIQUENESS  │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  Banach Contraction   │  Complete    │  Contraction:    │  YES (Picard)  │  Linear    │  YES         │
    │                       │  Metric      │  d(Tx,Ty)≤αd(x,y)│  Algorithm     │  (αⁿ)      │              │
    │                       │              │  α < 1           │  available     │            │              │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  Brouwer's FPT        │  Compact     │  Continuous      │  NO            │  No info   │  Not always  │
    │                       │  Convex Set  │  mapping of      │  (exists but   │            │              │
    │                       │  in ℝⁿ       │  X into itself    │  no algorithm) │            │              │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  Schauder's FPT       │  Banach      │  Continuous +    │  NO            │  No info   │  Not always  │
    │                       │  Space       │  Compact image   │  construction  │            │              │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  Kakutani's FPT       │  Compact     │  Upper semi-     │  NO            │  No info   │  Not always  │
    │                       │  Convex Set  │  continuous      │  (for multi-   │            │              │
    │  (Set-valued)         │              │  multifunctions   │  functions)    │            │              │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────┘

    KEY DIFFERENCES:

    • BANACH: Requires CONTRACTION property → guarantees UNIQUE fixed point
             Provides EXPLICIT ALGORITHM with ERROR BOUNDS
             Works in any COMPLETE METRIC SPACE (including infinite dimensions)

    • BROUWER: Only requires CONTINUITY → may have multiple fixed points
              Existence only, NO ALGORITHM
              Limited to COMPACT CONVEX subsets of ℝⁿ

    • SCHAUDER: Generalizes Brouwer to infinite dimensions
               Existence only, NO ALGORITHM
               Requires COMPACTNESS condition on the mapping


    CHOICE OF THEOREM:

    ✓ Use BANACH when:  Contraction property is satisfied, uniqueness matters, efficient algorithm needed
    ✓ Use BROUWER when: Finite dimensions, continuity sufficient, domain is compact convex subset
    ✓ Use SCHAUDER when: Infinite dimensions, compactness is available, existence is sufficient

    ═════════════════════════════════════════════════════════════════════════════════════════════════════════
    """

    ax.text(0.01, 0.98, comparison_text, fontsize=8.5, verticalalignment='top',
           family='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, linewidth=2.5, pad=1))

    fig.tight_layout()
    save_pdf(fig, 'comparison_fixed_point_theorems.pdf')

if __name__ == "__main__":
    print("Generating figures for Chapter 2b: Banach Contraction Mapping...")

    figure_contraction_mapping()
    print("✓ Generated: contraction_mapping.pdf")

    figure_contraction_constant()
    print("✓ Generated: contraction_constant.pdf")

    figure_conditions()
    print("✓ Generated: theorem_conditions.pdf")

    figure_examples()
    print("✓ Generated: examples.pdf")

    figure_picard_iteration()
    print("✓ Generated: picard_iteration.pdf")

    figure_numerical_example()
    print("✓ Generated: numerical_example.pdf")

    figure_applications()
    print("✓ Generated: applications.pdf")

    figure_comparison()
    print("✓ Generated: comparison_fixed_point_theorems.pdf")

    print("\nAll figures generated successfully!")
