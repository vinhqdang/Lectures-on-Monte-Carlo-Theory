#!/usr/bin/env python3
"""
Generate figures for Chapter 9d: Applications of Fixed Point Theorems
Illustrations for key concepts and theorems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib import patches
import os

# Set up output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

def set_style():
    """Set consistent matplotlib style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def fig_volterra_kernel():
    """Figure: Volterra Integral Equation Structure"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot: Time domain
    t = np.linspace(0, 2, 200)
    s = np.linspace(0, 2, 50)

    ax1.fill_between([0, 2], 0, 2, alpha=0.1, color='blue', label='Integration region')
    ax1.plot([0, 2], [0, 0], 'k-', linewidth=1)
    ax1.plot([0, 0], [0, 2], 'k-', linewidth=1)

    for ti in np.linspace(0.2, 2, 6):
        ax1.plot([ti, ti], [0, ti], 'b-', linewidth=0.5, alpha=0.6)
        ax1.plot([0, ti], [ti, ti], 'b-', linewidth=0.5, alpha=0.6)

    ax1.set_xlabel('Time $s$', fontsize=11)
    ax1.set_ylabel('Time $t$', fontsize=11)
    ax1.set_title('Volterra Integral Domain: $0 \leq s \leq t$', fontsize=12)
    ax1.set_xlim(-0.1, 2.2)
    ax1.set_ylim(-0.1, 2.2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right plot: Function behavior
    t_vals = np.linspace(0.01, 2, 100)
    y = np.exp(-t_vals) * (1 + np.sin(2*np.pi*t_vals))

    ax2.plot(t_vals, y, 'b-', linewidth=2, label='Solution $y(t)$')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.fill_between(t_vals, 0, y, alpha=0.2)
    ax2.set_xlabel('Time $t$', fontsize=11)
    ax2.set_ylabel('$y(t)$', fontsize=11)
    ax2.set_title('Typical Solution Behavior', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volterra_structure.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: volterra_structure.pdf")

def fig_fixed_point_geometry():
    """Figure: Fixed Point and Contraction Mapping"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Identity line and function
    x = np.linspace(0, 1, 200)
    f = 0.4 + 0.5*x + 0.1*np.sin(4*np.pi*x)

    ax1.plot(x, x, 'k--', linewidth=2, label='$y = x$ (Identity)')
    ax1.plot(x, f, 'b-', linewidth=2.5, label='$y = T(x)$ (Operator)')

    # Mark fixed point
    fp_idx = np.argmin(np.abs(x - f))
    fixed_pt = x[fp_idx]
    ax1.plot(fixed_pt, fixed_pt, 'ro', markersize=10, label='Fixed Point', zorder=5)
    ax1.plot([fixed_pt, fixed_pt], [0, fixed_pt], 'r--', linewidth=1, alpha=0.5)
    ax1.plot([0, fixed_pt], [fixed_pt, fixed_pt], 'r--', linewidth=1, alpha=0.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_title('Fixed Point of Operator $T$', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_aspect('equal')

    # Right: Convergence iteration
    x0 = 0.1
    iterations = []
    x_curr = x0

    for _ in range(15):
        y_curr = 0.4 + 0.5*x_curr + 0.1*np.sin(4*np.pi*x_curr)
        iterations.append((x_curr, y_curr))
        x_curr = y_curr

    for i, (xi, yi) in enumerate(iterations[:-1]):
        if i < 8:  # Only show first 8 iterations for clarity
            ax2.plot([xi, xi], [0, yi], 'b-', linewidth=0.8, alpha=0.6)
            ax2.plot([xi, yi], [yi, yi], 'b-', linewidth=0.8, alpha=0.6)

    iter_x = [p[0] for p in iterations]
    ax2.plot(iter_x, 'g-o', linewidth=2, markersize=4, label='Iterates $x_n$')
    ax2.axhline(y=fixed_pt, color='r', linewidth=2, linestyle='--', label='Fixed point', alpha=0.7)

    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel('Value $x_n$', fontsize=12)
    ax2.set_title('Convergence of Iterations', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fixed_point_geometry.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: fixed_point_geometry.pdf")

def fig_complementarity_problem():
    """Figure: Complementarity Problem Geometry"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Orthogonal complementarity
    theta = np.linspace(0, 2*np.pi, 100)

    # Cone K
    K_angle1 = np.pi/4
    K_angle2 = 3*np.pi/4
    K_radius = 1.5

    ax1.fill([0, K_radius*np.cos(K_angle1), K_radius*np.cos(K_angle2), 0],
             [0, K_radius*np.sin(K_angle1), K_radius*np.sin(K_angle2), 0],
             color='blue', alpha=0.3, label='Cone $K$')

    # Dual cone K*
    K_angle1_dual = K_angle1 + np.pi
    K_angle2_dual = K_angle2 + np.pi
    ax1.fill([0, K_radius*np.cos(K_angle1_dual), K_radius*np.cos(K_angle2_dual), 0],
             [0, K_radius*np.sin(K_angle1_dual), K_radius*np.sin(K_angle2_dual), 0],
             color='red', alpha=0.3, label='Dual cone $K^*$')

    # Orthogonality illustration
    x_vec = np.array([1, 0.5])
    y_vec = np.array([-1, -0.5])

    ax1.arrow(0, 0, x_vec[0], x_vec[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax1.arrow(0, 0, y_vec[0], y_vec[1], head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)

    ax1.text(x_vec[0]+0.1, x_vec[1]+0.1, '$x \\in K$', fontsize=11, color='blue')
    ax1.text(y_vec[0]-0.3, y_vec[1]-0.3, '$y \\in K^*$', fontsize=11, color='red')
    ax1.text(0.3, -0.2, '$\\langle x, y \\rangle = 0$', fontsize=11, fontweight='bold')

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.set_title('Cone and Dual Cone Relationship', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Complementarity conditions
    z = np.linspace(0, 3, 300)
    x_vals = np.maximum(z - 1, 0)  # x = (z-1)_+

    ax2.fill_between(z, 0, 1.5, where=(z >= 1), alpha=0.3, color='green', label='$x > 0$ region')
    ax2.plot(z, x_vals, 'b-', linewidth=2.5, label='$x = \\max(0, z-1)$')
    ax2.axvline(x=1, color='gray', linewidth=1, linestyle='--', alpha=0.7)

    ax2.text(0.3, 0.7, '$x = 0$\n$z > 0$', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax2.text(2.5, 1.2, '$x > 0$\n$z = 0$', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.5))

    ax2.set_xlim(0, 3)
    ax2.set_ylim(-0.2, 2)
    ax2.set_xlabel('$z$', fontsize=11)
    ax2.set_ylabel('$x$', fontsize=11)
    ax2.set_title('Complementarity Conditions: $0 \\leq x \\perp z \\geq 0$', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complementarity_problem.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: complementarity_problem.pdf")

def fig_contraction_mapping_rate():
    """Figure: Contraction Mapping and Convergence Rate"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Error reduction
    n_iter = np.arange(0, 20)

    # Different contraction constants
    k_values = [0.3, 0.5, 0.7, 0.9]
    colors = ['green', 'blue', 'orange', 'red']

    for k, color in zip(k_values, colors):
        errors = k**n_iter
        ax1.semilogy(n_iter, errors, 'o-', linewidth=2, markersize=5,
                    label=f'$k = {k}$', color=color)

    ax1.set_xlabel('Iteration $n$', fontsize=12)
    ax1.set_ylabel('Error $||x_n - x^*||$ (log scale)', fontsize=12)
    ax1.set_title('Convergence Rate for Different Contraction Constants', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11)

    # Right: Basin of attraction
    x = np.linspace(-2, 2, 300)
    T_x = 0.5*x**2 + 0.3*x

    ax2.plot(x, x, 'k--', linewidth=1.5, label='$y = x$')
    ax2.plot(x, T_x, 'b-', linewidth=2.5, label='$y = T(x)$')

    # Fixed points
    fp1 = -0.4  # Approximate fixed point
    ax2.plot([fp1, fp1], [0, fp1], 'g--', linewidth=1, alpha=0.5)
    ax2.plot(fp1, fp1, 'go', markersize=10, label='Fixed point')

    # Basin indication
    basin_x = np.linspace(fp1-0.8, fp1+0.8, 100)
    ax2.fill_between(basin_x, -2, 2, alpha=0.1, color='green', label='Basin of attraction')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_title('Basin of Attraction', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contraction_mapping_rate.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contraction_mapping_rate.pdf")

def fig_surjectivity_application():
    """Figure: Surjectivity and Open Mapping"""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Function with surjectivity
    ax = axes[0]
    x = np.linspace(-2, 2, 200)
    y = x**3 + 0.5*x

    ax.plot(x, y, 'b-', linewidth=2.5, label='$y = T(x)$')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Show range coverage
    y_vals = [-2, -1, 0, 1, 2]
    for yv in y_vals:
        x_val = np.cbrt(yv / (1 + 0.001))  # Approximate inverse
        if -2 <= x_val <= 2:
            ax.plot(x_val, yv, 'ro', markersize=6)
            ax.plot([x_val, x_val], [yv, 0], 'r--', linewidth=0.5, alpha=0.5)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Domain $X$', fontsize=12)
    ax.set_ylabel('Range (Codomain) $Y$', fontsize=12)
    ax.set_title('Surjective Operator: Every $y \\in Y$ has preimage', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Right: Directional contractor concept
    ax = axes[1]

    # Draw mapping regions
    rect1 = Rectangle((-1.5, -1), 1, 1, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.2, label='Domain region')
    rect2 = Rectangle((-0.8, -0.3), 0.6, 0.6, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Contracted region')
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Direction arrow
    arrow = FancyArrowPatch((-0.5, -0.5), (-0.2, -0.05),
                           arrowstyle='->', mutation_scale=30, linewidth=2, color='green')
    ax.add_patch(arrow)

    ax.text(-1, 0.3, 'Direction $\\Gamma(x)$', fontsize=11, color='green', fontweight='bold')
    ax.text(-0.8, -1.3, 'Generalized\nDirectional\nContractor', fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlim(-2, 1)
    ax.set_ylim(-1.5, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Space $X$', fontsize=12)
    ax.set_title('Generalized Directional Contractor', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'surjectivity_application.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: surjectivity_application.pdf")

def fig_chapter_overview():
    """Figure: Chapter 9d Overview - Applications of Fixed Point Theorems"""
    set_style()
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Chapter 9d: Applications of Fixed Point Theorems',
           ha='center', fontsize=16, fontweight='bold',
           transform=ax.transAxes)

    # Main sections
    sections = [
        ('9.6 Volterra\nIntegrodifferential\nEquations', 0.15, 0.75, 'lightblue'),
        ('9.7 Surjectivity\nTheorems', 0.5, 0.75, 'lightgreen'),
        ('9.8 Simultaneous\nComplementarity\nProblems', 0.85, 0.75, 'lightyellow'),
        ('Key Theorems:\n9.34, 9.37,\n9.38-9.40', 0.15, 0.45, 'lightcoral'),
        ('Generalized\nDirectional\nContractors', 0.5, 0.45, 'lightcyan'),
        ('Applications to\nVariational\nInequalities', 0.85, 0.45, 'plum'),
    ]

    for text, x, y, color in sections:
        bbox = dict(boxstyle='round,pad=0.8', facecolor=color, edgecolor='black', linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes, bbox=bbox)

    # Connections
    for i in range(3):
        x1, y1 = [0.15, 0.5, 0.85][i], 0.65
        x2, y2 = [0.15, 0.5, 0.85][i], 0.55
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
                   transform=ax.transAxes)

    # Key results box
    results_text = ('Key Results:\n'
                   '• Fixed point existence and uniqueness\n'
                   '• Convergence of iterative methods\n'
                   '• Contraction mapping principles\n'
                   '• Applications to differential and integral equations\n'
                   '• Solution of complementarity problems')

    ax.text(0.5, 0.15, results_text, ha='center', va='top', fontsize=10,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', edgecolor='black', linewidth=1.5, pad=1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chapter_overview.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: chapter_overview.pdf")

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 9d...\n")

    fig_chapter_overview()
    fig_volterra_kernel()
    fig_fixed_point_geometry()
    fig_complementarity_problem()
    fig_contraction_mapping_rate()
    fig_surjectivity_application()

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
