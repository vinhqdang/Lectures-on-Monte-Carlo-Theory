#!/usr/bin/env python3
"""
Generate figures for Chapter 20: Monotone Operators
Bauscke & Combettes, Convex Analysis and Monotone Operator Theory in Hilbert Spaces 2e 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent appearance
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'error': '#d62728',
    'neutral': '#7f7f7f'
}

def set_fig_params():
    """Set matplotlib parameters for consistent styling."""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def fig_monotone_function():
    """
    Illustration of a monotone function: (x - y | u - v) >= 0
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Monotone operator example
    ax = axes[0]
    x = np.linspace(-2, 2, 100)
    y = x**2
    ax.plot(x, y, linewidth=2.5, color=COLORS['primary'], label='$f(x) = x^2$')
    ax.fill_between(x, y, alpha=0.2, color=COLORS['primary'])

    # Mark two points
    x1, x2 = -1, 1
    y1, y2 = x1**2, x2**2
    ax.plot([x1, x2], [y1, y2], 'o', markersize=10, color=COLORS['accent'])
    ax.text(x1-0.3, y1+0.2, f'$(x_1, u_1)$', fontsize=11, color=COLORS['accent'])
    ax.text(x2+0.1, y2+0.2, f'$(x_2, u_2)$', fontsize=11, color=COLORS['accent'])

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u = f(x)$', fontsize=12)
    ax.set_title('Monotone Function: $\\langle x - y \\mid u - v \\rangle \\geq 0$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 3])

    # Right: Non-monotone counterexample
    ax = axes[1]
    x = np.linspace(-2, 2, 100)
    y = -x**2 + 1
    ax.plot(x, y, linewidth=2.5, color=COLORS['error'], label='$f(x) = -x^2 + 1$')
    ax.fill_between(x, y, alpha=0.2, color=COLORS['error'])

    x1, x2 = -1, 1
    y1, y2 = -(x1**2) + 1, -(x2**2) + 1
    ax.plot([x1, x2], [y1, y2], 'o', markersize=10, color=COLORS['neutral'])
    ax.text(x1-0.4, y1-0.3, f'$(x_1, u_1)$', fontsize=11, color=COLORS['neutral'])
    ax.text(x2+0.1, y2-0.3, f'$(x_2, u_2)$', fontsize=11, color=COLORS['neutral'])

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u = f(x)$', fontsize=12)
    ax.set_title('Non-Monotone Function', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 1.5])

    plt.tight_layout()
    plt.savefig('monotone_function.pdf', dpi=300, bbox_inches='tight')
    print("Generated: monotone_function.pdf")
    plt.close()

def fig_monotone_operator_examples():
    """
    Examples of monotone operators in 1D
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Linear operator
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 100)
    u = 2*x  # Linear monotone operator
    ax.plot(x, u, linewidth=2.5, color=COLORS['primary'], label='$A(x) = 2x$')
    ax.fill_between(x, 0, u, alpha=0.2, color=COLORS['primary'])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$u = A(x)$', fontsize=11)
    ax.set_title('(a) Linear: $A(x) = 2x$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-6, 6])

    # (b) Subgradient
    ax = axes[0, 1]
    x = np.linspace(-3, 3, 100)
    u = np.abs(x)
    ax.plot(x, u, linewidth=2.5, color=COLORS['accent'], label='$\\partial f(x)$')
    ax.fill_between(x, 0, u, alpha=0.2, color=COLORS['accent'])
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$u \\in \\partial|x|$', fontsize=11)
    ax.set_title('(b) Subgradient: $\\partial|\\cdot|$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 4])

    # (c) Nonexpansive operator
    ax = axes[1, 0]
    x = np.linspace(-3, 3, 100)
    u = 0.5*x  # Nonexpansive (contraction)
    ax.plot(x, u, linewidth=2.5, color=COLORS['secondary'], label='$T(x) = 0.5x$')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.plot(x, x, 'k--', linewidth=1.5, alpha=0.5, label='$y = x$')
    ax.fill_between(x, 0, u, alpha=0.2, color=COLORS['secondary'])
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$u = T(x)$', fontsize=11)
    ax.set_title('(c) Nonexpansive: $T(x) = 0.5x$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])
    ax.legend()

    # (d) Projector
    ax = axes[1, 1]
    # Illustration of projection onto interval [−1, 1]
    x = np.linspace(-3, 3, 100)
    u = np.clip(x, -1, 1)  # Projection onto [−1, 1]
    ax.plot(x, u, linewidth=2.5, color=COLORS['error'], label='$P_C(x)$')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axhline(1, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(-1, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.fill_between(x, 0, u, alpha=0.2, color=COLORS['error'])
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$u = P_C(x)$', fontsize=11)
    ax.set_title('(d) Projector onto $C = [-1, 1]$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])

    plt.tight_layout()
    plt.savefig('monotone_operators.pdf', dpi=300, bbox_inches='tight')
    print("Generated: monotone_operators.pdf")
    plt.close()

def fig_maximally_monotone():
    """
    Illustration of maximal monotonicity via graphs
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Monotone but not maximal
    ax = axes[0]
    x = np.linspace(-2, 2, 100)
    u = np.tanh(2*x)  # Restricted curve
    ax.plot(x, u, linewidth=2.5, color=COLORS['secondary'], label='Graph of $A$ (monotone)')
    ax.scatter(x[::10], u[::10], s=20, color=COLORS['secondary'], alpha=0.5)

    # Show possible extension
    x_ext = np.linspace(-2.5, 2.5, 100)
    u_ext = np.sign(x_ext) * 1.2
    ax.plot(x_ext, u_ext, linewidth=1.5, color=COLORS['accent'], linestyle='--',
            label='Possible monotone extension')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u$', fontsize=12)
    ax.set_title('Monotone but Not Maximally Monotone', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-1.5, 1.5])

    # Right: Maximally monotone (subdifferential)
    ax = axes[1]
    # Draw multi-valued graph
    x_vals = np.array([-1.5, -0.5, 0, 0.5, 1.5])
    for x_val in x_vals:
        if x_val == 0:
            # At 0, subdifferential of |·| is [-1, 1]
            ax.plot([0, 0], [-1, 1], linewidth=3, color=COLORS['primary'], marker='o', markersize=6)
            ax.text(x_val+0.15, 0.5, '$\\partial|0|=[-1,1]$', fontsize=10)
        else:
            # Away from 0, subdifferential is single point
            u_val = np.sign(x_val)
            ax.plot(x_val, u_val, 'o', markersize=8, color=COLORS['primary'])
            ax.text(x_val+0.1, u_val+0.15, f'$(x,\\text{{sgn}}(x))$', fontsize=9)

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u \\in A(x)$', fontsize=12)
    ax.set_title('Maximally Monotone: $A = \\partial|\\cdot|$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-1.5, 1.5])

    plt.tight_layout()
    plt.savefig('maximally_monotone.pdf', dpi=300, bbox_inches='tight')
    print("Generated: maximally_monotone.pdf")
    plt.close()

def fig_fitzpatrick_function():
    """
    Illustration of the Fitzpatrick function
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Fitzpatrick function for linear operator
    ax = axes[0]
    x = np.linspace(-2, 2, 100)
    u_range = np.linspace(-3, 3, 100)
    X, U = np.meshgrid(x, u_range)

    # For A(x) = 2x, F_A(x,u) = (1/4)||x - u||^2 + (1/2)||u||^2
    F = 0.25 * (X - U)**2 + 0.5 * U**2
    contour = ax.contourf(X, U, F, levels=20, cmap='viridis')
    ax.contour(X, U, F, levels=10, colors='k', alpha=0.2, linewidths=0.5)

    # Mark the graph of A
    A_x = x
    A_u = 2*x
    mask = (A_u >= -3) & (A_u <= 3)
    ax.plot(A_x[mask], A_u[mask], 'r-', linewidth=2.5, label='Graph of $A$')

    plt.colorbar(contour, ax=ax, label='$F_A(x,u)$')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u$', fontsize=12)
    ax.set_title('Fitzpatrick Function: $F_A(x,u)$ for $A(x)=2x$', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-3, 3])

    # Right: Characteristics of Fitzpatrick function
    ax = axes[1]
    ax.axis('off')

    props = [
        r'$\mathbf{(i)}$ $F_A(x, u) = \langle x \mid u \rangle - \inf_{(y,v) \in \mathrm{gra} A} \langle x - y \mid u - v \rangle$',
        r'$\mathbf{(ii)}$ $F_A(x, u) \geq \langle x \mid u \rangle$ for all $(x, u)$',
        r'$\mathbf{(iii)}$ $(x, u) \in \mathrm{gra} A \Rightarrow F_A(x, u) = \langle x \mid u \rangle$',
        r'$\mathbf{(iv)}$ $F_A \in \Gamma_0(\mathcal{H} \oplus \mathcal{H})$ when $A$ is m.m.',
        r'$\mathbf{(v)}$ $\text{gra} A = \{\,(x,u) \mid F_A(x, u) = \langle x \mid u \rangle\,\}$'
    ]

    y_pos = 0.9
    ax.text(0.05, y_pos, 'Key Properties of $F_A$:', fontsize=13, weight='bold',
            transform=ax.transAxes, va='top')
    y_pos -= 0.15

    for prop in props:
        ax.text(0.05, y_pos, prop, fontsize=11, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        y_pos -= 0.17

    plt.tight_layout()
    plt.savefig('fitzpatrick_function.pdf', dpi=300, bbox_inches='tight')
    print("Generated: fitzpatrick_function.pdf")
    plt.close()

def fig_numerical_example():
    """
    Numerical example: monotonicity of sum of operators
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Individual operators
    ax = axes[0]
    x = np.linspace(-2, 2, 100)
    A_x = np.exp(x)  # Strictly monotone
    B_x = np.sin(2*np.pi*x)  # Non-monotone
    C_x = A_x + B_x  # Sum

    ax.plot(x, A_x, linewidth=2, label='$A(x) = e^x$ (monotone)', color=COLORS['primary'])
    ax.plot(x, B_x, linewidth=2, label='$B(x) = \\sin(2\\pi x)$ (non-monotone)', color=COLORS['error'])
    ax.plot(x, C_x, linewidth=2.5, label='$A(x) + B(x)$', color=COLORS['accent'], linestyle='--')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u$', fontsize=12)
    ax.set_title('Operators and Their Sum', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 8])

    # Right: Verify monotonicity via inner products
    ax = axes[1]

    x_vals = np.linspace(-2, 1.5, 30)
    inner_products = []

    for i in range(len(x_vals)-1):
        x1, x2 = x_vals[i], x_vals[i+1]
        u1 = np.exp(x1)
        u2 = np.exp(x2)
        ip = (x1 - x2) * (u1 - u2)
        inner_products.append(ip)

    x_diffs = x_vals[:-1] + np.diff(x_vals)/2
    colors = [COLORS['primary'] if ip >= 0 else COLORS['error'] for ip in inner_products]

    ax.bar(range(len(inner_products)), inner_products, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xlabel('Pair Index', fontsize=12)
    ax.set_ylabel('$\\langle x_1 - x_2 \\mid A(x_1) - A(x_2) \\rangle$', fontsize=11)
    ax.set_title('Monotonicity Check: $\\langle x - y \\mid u - v \\rangle \\geq 0$', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('numerical_example.pdf', dpi=300, bbox_inches='tight')
    print("Generated: numerical_example.pdf")
    plt.close()

def fig_chapter_overview():
    """
    Overview diagram of Chapter 20 sections
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Chapter 20: Monotone Operators', fontsize=16, weight='bold', ha='center')

    # Main sections
    sections = [
        ('20.1 Monotone Operators', 1, 7.5, COLORS['primary']),
        ('20.2 Maximally Monotone', 4, 7.5, COLORS['accent']),
        ('20.3 Partial Inverse', 7, 7.5, COLORS['secondary']),
        ('20.4 Bivariate Functions', 2, 5, COLORS['error']),
        ('20.5 Fitzpatrick Function', 5, 5, COLORS['neutral']),
        ('Exercises', 8, 5, 'lightgray'),
    ]

    for title, x, y, color in sections:
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor=color, facecolor=color,
                             alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, title, fontsize=10, ha='center', va='center', weight='bold')

    # Key concepts
    concepts_y = 3.5
    ax.text(5, concepts_y+0.5, 'Key Concepts', fontsize=12, weight='bold', ha='center')

    concepts = [
        r'$\langle x - y \mid u - v \rangle \geq 0$',
        'Maximality via Zorn\'s Lemma',
        'Moreau\'s Theorem',
        'Spingarn\'s Formula',
        'Fitzpatrick Characterization'
    ]

    x_start = 0.5
    for i, concept in enumerate(concepts):
        ax.text(x_start + (i % 3)*3, concepts_y - (i//3)*0.6, f'• {concept}',
               fontsize=9, va='top')

    # Applications
    app_y = 1.2
    ax.text(5, app_y+0.3, 'Applications', fontsize=11, weight='bold', ha='center')
    ax.text(5, app_y-0.3,
           'Convex optimization • Fixed-point algorithms • Partial differential equations',
           fontsize=9, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('chapter_overview.pdf', dpi=300, bbox_inches='tight')
    print("Generated: chapter_overview.pdf")
    plt.close()

def main():
    """Generate all figures."""
    set_fig_params()

    print("Generating figures for Chapter 20: Monotone Operators...")
    fig_monotone_function()
    fig_monotone_operator_examples()
    fig_maximally_monotone()
    fig_fitzpatrick_function()
    fig_numerical_example()
    fig_chapter_overview()

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
