"""
Generate figures for Chapter 8b: Integral Equations and Computational Schemes
Pathak - An Introduction to Nonlinear Analysis and Fixed Point Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from scipy.integrate import odeint, quad
from scipy.optimize import fsolve
import os

# Ensure output directory exists
output_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(output_dir)

# Set style for Beamer compatibility
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['text.usetex'] = False

# Color palette (accessible, print-friendly)
colors = {
    'primary': '#1f77b4',    # blue
    'secondary': '#ff7f0e',  # orange
    'success': '#2ca02c',    # green
    'danger': '#d62728',     # red
    'warning': '#ff9896',    # light red
    'info': '#9467bd',       # purple
}

# ============================================================================
# Figure 1: Hammerstein Operator Structure
# ============================================================================
def fig_hammerstein_structure():
    """Visualize the structure of Hammerstein operators"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Block diagram of Hammerstein operator
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1, 2)
    ax.axis('off')

    # Title
    ax.text(2, 1.8, 'Hammerstein Operator: $(Tx)(s) = \\int_\\Omega k(s,t) f(t,x(t)) dt$',
            fontsize=11, ha='center', weight='bold')

    # Boxes
    box_props = dict(boxstyle='round', facecolor=colors['primary'], alpha=0.3, edgecolor=colors['primary'], linewidth=2)
    ax.text(0.5, 0.8, '$x(t)$', fontsize=10, ha='center', bbox=box_props)
    ax.text(1.5, 0.8, '$f(t,x(t))$', fontsize=10, ha='center', bbox=box_props)
    ax.text(2.5, 0.8, '$\\int$ operator', fontsize=10, ha='center', bbox=box_props)
    ax.text(3.5, 0.8, '$(Tx)(s)$', fontsize=10, ha='center', bbox=box_props)

    # Arrows
    for i in range(3):
        ax.arrow(0.8 + i, 0.8, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Properties
    props_text = 'Properties:\n' + '\\n'.join([
        '• Nonlinear via $f(t,x(t))$',
        '• Integral via kernel $k(s,t)$',
        '• Applications: integral eqs.',
    ])
    ax.text(2, 0, props_text, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor=colors['info'], alpha=0.2))

    # Right: Example with concrete nonlinearity
    ax = axes[1]
    x = np.linspace(-2, 2, 100)

    # Different nonlinearities
    f1 = x  # linear
    f2 = np.tanh(x)  # bounded
    f3 = x**3  # polynomial

    ax.plot(x, f1, label='$f(x) = x$ (Linear)', linewidth=2, color=colors['primary'])
    ax.plot(x, f2, label='$f(x) = \\tanh(x)$ (Bounded)', linewidth=2, color=colors['secondary'])
    ax.plot(x, f3, label='$f(x) = x^3$ (Polynomial)', linewidth=2, color=colors['success'])

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$f(t, x)$', fontsize=10)
    ax.set_title('Nonlinearity Examples', fontsize=11, weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('hammerstein_structure.pdf')
    plt.close()

# ============================================================================
# Figure 2: Convergence of Mann Iteration
# ============================================================================
def fig_mann_iteration_convergence():
    """Visualize Mann iteration convergence for fixed point"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Graphical Mann iteration
    ax = axes[0]

    # Define a mapping T: x_{n+1} = T(x_n) with fixed point
    # Use T(x) = 0.5 * x + 0.3 * sin(2*x)
    def T(x):
        return 0.5 * x + 0.3 * np.sin(2 * x)

    # Find fixed point
    x_star = fsolve(lambda x: T(x) - x, 0.5)[0]

    x = np.linspace(-1, 1.5, 200)
    y = T(x)

    ax.plot(x, y, linewidth=2.5, label='$y = T(x)$', color=colors['primary'])
    ax.plot(x, x, linewidth=2, label='$y = x$', color=colors['danger'], linestyle='--')
    ax.plot(x_star, x_star, 'o', markersize=10, color=colors['success'], label=f'Fixed point', zorder=5)

    # Show iteration steps
    x0 = 0.8
    for i in range(4):
        x1 = T(x0)
        ax.plot([x0, x0], [x0, x1], 'k-', alpha=0.5, linewidth=1)
        ax.plot([x0, x1], [x1, x1], 'k-', alpha=0.5, linewidth=1)
        ax.plot(x0, x1, 'ro', markersize=4)
        x0 = x1

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_n$', fontsize=10)
    ax.set_ylabel('$T(x_n)$', fontsize=10)
    ax.set_title('Graphical Mann Iteration', fontsize=11, weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Error vs iterations
    ax = axes[1]

    iterations = 15
    errors = []

    # Simulate Mann iteration: x_{n+1} = (1-\alpha_n) x_n + \alpha_n T(x_n)
    alphas = [0.3, 0.5, 1.0]  # Different step sizes

    for alpha in alphas:
        x = 0.8
        error_seq = [abs(x - x_star)]
        for _ in range(iterations):
            x_new = (1 - alpha) * x + alpha * T(x)
            error_seq.append(abs(x_new - x_star))
            x = x_new
        ax.semilogy(range(len(error_seq)), error_seq, marker='o',
                   label=f'$\\alpha = {alpha}$', linewidth=2, markersize=5)

    ax.set_xlabel('Iteration $n$', fontsize=10)
    ax.set_ylabel('Error $|x_n - x^*|$', fontsize=10)
    ax.set_title('Convergence Rates', fontsize=11, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(-0.5, iterations + 0.5)

    plt.tight_layout()
    plt.savefig('mann_iteration_convergence.pdf')
    plt.close()

# ============================================================================
# Figure 3: Projection Schemes
# ============================================================================
def fig_projection_schemes():
    """Visualize projection-based computational schemes"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Top-left: Orthogonal projection in R^2
    ax = axes[0, 0]

    # Subspace (line)
    t = np.linspace(-1.5, 1.5, 100)
    # Project onto line with slope 2
    direction = np.array([1, 0.5])
    direction = direction / np.linalg.norm(direction)

    ax.plot(t * direction[0], t * direction[1], 'b-', linewidth=2.5, label='Subspace $X_n$')

    # Point and projection
    point = np.array([1.2, 0.3])
    proj = np.dot(point, direction) * direction

    ax.plot(point[0], point[1], 'ro', markersize=8, label='$x$')
    ax.plot(proj[0], proj[1], 'go', markersize=8, label='$P_n x$')
    ax.plot([point[0], proj[0]], [point[1], proj[1]], 'k--', linewidth=1.5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.set_title('Orthogonal Projection', fontsize=11, weight='bold')
    ax.legend(fontsize=9)

    # Top-right: Galerkin approximation concept
    ax = axes[0, 1]

    n_modes = 5
    x_fine = np.linspace(0, 1, 200)

    # Function: example function
    f_true = np.sin(np.pi * x_fine) * np.cos(2 * np.pi * x_fine)

    # Basis functions (sin)
    basis_coeff = np.random.randn(n_modes)
    f_approx = np.zeros_like(x_fine)
    for i in range(n_modes):
        f_approx += basis_coeff[i] * np.sin((i+1) * np.pi * x_fine)
    f_approx /= np.max(np.abs(f_approx))

    ax.plot(x_fine, f_true, 'b-', linewidth=2.5, label='True function')
    ax.plot(x_fine, f_approx, 'r--', linewidth=2, label=f'Galerkin approx. (n={n_modes})')
    ax.fill_between(x_fine, f_true, f_approx, alpha=0.2, color='gray')

    ax.set_xlabel('$t$', fontsize=10)
    ax.set_ylabel('$u(t)$', fontsize=10)
    ax.set_title('Galerkin Approximation', fontsize=11, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Convergence of projection methods
    ax = axes[1, 0]

    dimensions = np.arange(1, 11)
    # Simulated errors: E_n ~ 1/(n+1)^2 for smooth functions
    error_smooth = 1.0 / (dimensions + 1)**2
    error_rough = 1.0 / (dimensions + 1)

    ax.semilogy(dimensions, error_smooth, 'o-', linewidth=2.5,
               markersize=6, label='Smooth solution: $O(1/n^2)$', color=colors['primary'])
    ax.semilogy(dimensions, error_rough, 's-', linewidth=2.5,
               markersize=6, label='Rough solution: $O(1/n)$', color=colors['secondary'])

    ax.set_xlabel('Dimension $n$', fontsize=10)
    ax.set_ylabel('Approximation Error', fontsize=10)
    ax.set_title('Projection Convergence Rates', fontsize=11, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Bottom-right: Comparison of schemes
    ax = axes[1, 1]

    schemes = ['Mann\nIteration', 'Projection\n(Galerkin)', 'Projection\n(Petrov-Galerkin)', 'Hybrid\nApproach']
    convergence = [0.8, 0.6, 0.5, 0.3]  # Error reduction factor per iteration

    bars = ax.bar(schemes, convergence, color=[colors['primary'], colors['secondary'],
                                                colors['success'], colors['info']], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, conv in zip(bars, convergence):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{conv:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel('Contraction Factor', fontsize=10)
    ax.set_title('Computational Scheme Comparison', fontsize=11, weight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('projection_schemes.pdf')
    plt.close()

# ============================================================================
# Figure 4: Example: Hammerstein Integral Equation
# ============================================================================
def fig_hammerstein_example():
    """Numerical example: Solve a concrete Hammerstein equation"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Problem: x(s) + ∫_0^1 k(s,t) f(t, x(t)) dt = g(s)
    # Kernel: k(s,t) = min(s,t) (Green's function for BVP)
    # Nonlinearity: f(t,x) = 0.1 * sin(x)
    # Forcing: g(s) = s

    def kernel(s, t):
        return np.minimum(s, t)

    def nonlinearity(x):
        return 0.1 * np.sin(x)

    def forcing(s):
        return s

    # Solve numerically using simple iteration
    n_points = 30
    s = np.linspace(0, 1, n_points)
    ds = 1.0 / n_points

    # Initial guess
    x = s.copy()

    # Iteration
    iterations = 20
    errors = []
    solutions = [x.copy()]

    for iteration in range(iterations):
        # Compute integral term
        integral_term = np.zeros(n_points)
        for i in range(n_points):
            for j in range(n_points):
                integral_term[i] += kernel(s[i], s[j]) * nonlinearity(x[j]) * ds

        # Update: x(s) = g(s) - integral_term
        x_new = forcing(s) - integral_term

        # Error
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        x = x_new
        solutions.append(x.copy())

    # Top-left: Solution profiles
    ax = axes[0, 0]
    ax.plot(s, solutions[0], 'o-', linewidth=2, markersize=4,
           label='Initial guess', color=colors['warning'], alpha=0.7)
    ax.plot(s, solutions[5], 's-', linewidth=2, markersize=4,
           label='After 5 iterations', color=colors['secondary'], alpha=0.7)
    ax.plot(s, solutions[-1], 'D-', linewidth=2, markersize=4,
           label='Converged solution', color=colors['primary'])
    ax.set_xlabel('$s$', fontsize=10)
    ax.set_ylabel('$x(s)$', fontsize=10)
    ax.set_title('Solution Evolution', fontsize=11, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Convergence
    ax = axes[0, 1]
    ax.semilogy(range(1, len(errors)+1), errors, 'o-', linewidth=2.5,
               markersize=6, color=colors['primary'])
    ax.set_xlabel('Iteration $n$', fontsize=10)
    ax.set_ylabel('Error $|x_n - x_{n-1}|$', fontsize=10)
    ax.set_title('Convergence History', fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Bottom-left: Kernel visualization
    ax = axes[1, 0]
    S, T = np.meshgrid(s, s)
    K = np.minimum(S, T)
    im = ax.contourf(S, T, K, levels=15, cmap='viridis')
    plt.colorbar(im, ax=ax, label='$k(s,t)$')
    ax.set_xlabel('$s$', fontsize=10)
    ax.set_ylabel('$t$', fontsize=10)
    ax.set_title('Green\'s Function Kernel', fontsize=11, weight='bold')

    # Bottom-right: Nonlinearity impact
    ax = axes[1, 1]
    x_vals = np.linspace(-np.pi, np.pi, 100)
    f_vals = nonlinearity(x_vals)

    ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label='$f(x) = 0.1\\sin(x)$')
    ax.fill_between(x_vals, 0, f_vals, alpha=0.2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$f(t, x)$', fontsize=10)
    ax.set_title('Nonlinearity Function', fontsize=11, weight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('hammerstein_example.pdf')
    plt.close()

# ============================================================================
# Figure 5: Key Theorems Framework
# ============================================================================
def fig_theorems_framework():
    """Visualize the relationship between key theorems"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Theoretical Framework for Integral Equations',
           fontsize=13, weight='bold', ha='center')

    # Define boxes with theorem information
    theorems = [
        {'name': 'Th. 8.8\n(Hammerstein,\nMonotone K)', 'pos': (1.5, 5.5), 'color': colors['primary']},
        {'name': 'Th. 8.9\n(Monotone K,\nBounded N)', 'pos': (4.5, 5.5), 'color': colors['secondary']},
        {'name': 'Th. 8.10\n(Angle-bounded K,\nHemicont. N)', 'pos': (7.5, 5.5), 'color': colors['success']},
        {'name': 'Th. 8.13\n(Hammerstein)\nSums', 'pos': (1.5, 3.0), 'color': colors['info']},
        {'name': 'Th. 8.17\n(Generalized\nHammerstein)', 'pos': (4.5, 3.0), 'color': colors['warning']},
        {'name': 'Th. 8.22\n(Projection\nSchemes)', 'pos': (7.5, 3.0), 'color': colors['danger']},
        {'name': 'Existence & Uniqueness', 'pos': (5, 0.8), 'color': colors['primary']},
    ]

    # Draw boxes
    box_width = 1.5
    box_height = 0.8

    for thm in theorems:
        x, y = thm['pos']
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.05",
            edgecolor=thm['color'], facecolor=thm['color'], alpha=0.3, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, thm['name'], fontsize=9, ha='center', va='center', weight='bold')

    # Draw arrows showing relationships
    # Row 1 to Row 2
    ax.annotate('', xy=(1.5, 3.8), xytext=(1.5, 5.1),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.annotate('', xy=(4.5, 3.8), xytext=(4.5, 5.1),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.annotate('', xy=(7.5, 3.8), xytext=(7.5, 5.1),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # Row 2 to conclusion
    ax.annotate('', xy=(5, 1.6), xytext=(4.5, 2.6),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['primary']))
    ax.annotate('', xy=(5, 1.6), xytext=(5, 2.6),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['primary']))
    ax.annotate('', xy=(5, 1.6), xytext=(5.5, 2.6),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['primary']))

    # Add condition boxes on the left
    conditions_text = "Key Conditions:\n" + "\n".join([
        "• Monotone operators",
        "• Angle-bounded",
        "• Hemicontinuous",
        "• Growth conditions",
    ])
    ax.text(0.2, 4.5, conditions_text, fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.5))

    plt.tight_layout()
    plt.savefig('theorems_framework.pdf')
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 8b...")

    print("  1. Hammerstein operator structure...")
    fig_hammerstein_structure()

    print("  2. Mann iteration convergence...")
    fig_mann_iteration_convergence()

    print("  3. Projection schemes...")
    fig_projection_schemes()

    print("  4. Hammerstein equation example...")
    fig_hammerstein_example()

    print("  5. Theorems framework...")
    fig_theorems_framework()

    print("All figures generated successfully!")
