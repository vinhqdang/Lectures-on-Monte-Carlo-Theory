#!/usr/bin/env python3
"""
Generate figures for Chapter 6c: Normal Structure & Fixed Point Theory
Beamer slides based on Pathak's "An Introduction to Nonlinear Analysis and Fixed Point Theory"

This script creates supporting diagrams and visualizations for the presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

# Set up matplotlib style
plt.style.use('default')
colors = {
    'primary': '#0072B2',
    'secondary': '#D55E00',
    'accent': '#009E73',
    'neutral': '#595959'
}

output_dir = os.path.dirname(os.path.abspath(__file__))

def fig_banach_space_visualization():
    """Visualize a Banach space with normal structure and fixed point concepts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Banach Space with bounded set
    ax = axes[0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Bounded Convex Set K in Banach Space', fontsize=12, fontweight='bold')

    # Draw the convex set K
    circle = Circle((0, 0), 1.8, fill=True, facecolor=colors['primary'],
                    alpha=0.2, edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(circle)
    ax.text(0, 0, 'K', fontsize=16, ha='center', va='center', fontweight='bold')

    # Mark diameter
    ax.plot([-1.8, 1.8], [0, 0], 'k--', linewidth=1.5, label='Diameter δ(K)')
    ax.plot([-1.8], [0], 'ro', markersize=8)
    ax.plot([1.8], [0], 'ro', markersize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9)

    # Right: Fixed point in normal structure
    ax = axes[1]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Normal Structure & Nonexpansive Mappings', fontsize=12, fontweight='bold')

    # Draw the convex set K
    circle = Circle((0, 0), 1.8, fill=True, facecolor=colors['secondary'],
                    alpha=0.2, edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(circle)

    # Mark a fixed point
    ax.plot([0], [0], 'g*', markersize=20, label='Fixed point x*')
    ax.text(0.3, 0.3, 'x*', fontsize=12, fontweight='bold', color='green')

    # Show mapping illustration
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        x0 = 1.3 * np.cos(angle)
        y0 = 1.3 * np.sin(angle)
        dx = -x0 * 0.3
        dy = -y0 * 0.3
        ax.arrow(x0, y0, dx, dy, head_width=0.15, head_length=0.1,
                fc=colors['accent'], ec=colors['accent'], alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'banach_space_normal_structure.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: banach_space_normal_structure.pdf")

def fig_iterative_convergence():
    """Visualize convergence of iterative sequences."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Strong convergence
    ax = axes[0, 0]
    n_vals = np.arange(1, 31)
    strong_conv = 1 / n_vals**2
    ax.semilogy(n_vals, strong_conv, 'o-', color=colors['primary'], linewidth=2, markersize=6)
    ax.set_xlabel('Iteration n', fontsize=11)
    ax.set_ylabel('||xₙ - x*|| (log scale)', fontsize=11)
    ax.set_title('Strong Convergence (Quadratic)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Subplot 2: Weak convergence
    ax = axes[0, 1]
    weak_conv = 1 / n_vals
    ax.semilogy(n_vals, weak_conv, 's-', color=colors['secondary'], linewidth=2, markersize=6)
    ax.set_xlabel('Iteration n', fontsize=11)
    ax.set_ylabel('||xₙ - x*|| (log scale)', fontsize=11)
    ax.set_title('Weak Convergence (Linear)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Subplot 3: Hammerstein equation residual
    ax = axes[1, 0]
    residual = np.exp(-0.15 * n_vals)
    ax.semilogy(n_vals, residual, '^-', color=colors['accent'], linewidth=2, markersize=6)
    ax.set_xlabel('Iteration n', fontsize=11)
    ax.set_ylabel('Residual ||u + KNu|| (log scale)', fontsize=11)
    ax.set_title('Hammerstein Operator Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Subplot 4: Contraction rate
    ax = axes[1, 1]
    rates = [0.3, 0.5, 0.7, 0.9]
    for rate, color in zip(rates, [colors['primary'], colors['secondary'], colors['accent'], colors['neutral']]):
        contraction = rate ** n_vals
        ax.semilogy(n_vals, contraction, 'o-', color=color, linewidth=2, label=f'ρ = {rate}')
    ax.set_xlabel('Iteration n', fontsize=11)
    ax.set_ylabel('||xₙ - x₀|| (log scale)', fontsize=11)
    ax.set_title('Contraction Mapping: ρⁿ Decay', fontsize=12, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iterative_convergence.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: iterative_convergence.pdf")

def fig_monotone_operator_property():
    """Visualize monotone operator properties."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Monotone operator visualization
    ax = axes[0]
    x_vals = np.linspace(-2, 2, 100)
    y_vals = x_vals**3  # Example monotone operator

    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='T(x) = x³')

    # Show two points and their property
    x1, x2 = -1.5, 1.0
    T_x1 = x1**3
    T_x2 = x2**3

    ax.plot([x1], [T_x1], 'ro', markersize=10, label='(x₁, Tx₁)')
    ax.plot([x2], [T_x2], 'go', markersize=10, label='(x₂, Tx₂)')

    # Draw the monotonicity illustration
    ax.annotate('', xy=(x2, T_x2), xytext=(x1, T_x1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.6))

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('T(x)', fontsize=12, fontweight='bold')
    ax.set_title('Monotone Operator: (x₁-x₂, T(x₁)-T(x₂)) ≥ 0', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')

    # Right: Strongly monotone operator
    ax = axes[1]
    x_vals = np.linspace(-2, 2, 100)
    y_vals = 2*x_vals**3 + 0.5*x_vals  # Strongly monotone

    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Strongly monotone T(x)')

    x1, x2 = -1.5, 1.0
    T_x1 = 2*x1**3 + 0.5*x1
    T_x2 = 2*x2**3 + 0.5*x2

    ax.plot([x1], [T_x1], 'ro', markersize=10, label='(x₁, Tx₁)')
    ax.plot([x2], [T_x2], 'go', markersize=10, label='(x₂, Tx₂)')

    ax.annotate('', xy=(x2, T_x2), xytext=(x1, T_x1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.6))

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('T(x)', fontsize=12, fontweight='bold')
    ax.set_title('Strongly Monotone: (x₁-x₂, T(x₁)-T(x₂)) ≥ c||x₁-x₂||²', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monotone_operator_property.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: monotone_operator_property.pdf")

def fig_hammerstein_equation_schematic():
    """Create a schematic diagram of Hammerstein integral equation structure."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Title
    ax.text(0.5, 0.95, 'Hammerstein Integral Equation: x + KNₓ = y',
            ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

    # Draw operator decomposition
    y_pos = 0.75
    x_positions = [0.15, 0.5, 0.85]

    # Linear operator K
    rect1 = FancyBboxPatch((0.05, y_pos-0.08), 0.2, 0.15,
                           boxstyle="round,pad=0.01",
                           edgecolor=colors['primary'], facecolor=colors['primary'],
                           alpha=0.2, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect1)
    ax.text(0.15, y_pos, 'K\nLinear\nOperator', ha='center', va='center',
            fontsize=11, fontweight='bold', transform=ax.transAxes)

    # Nonlinear operator N
    rect2 = FancyBboxPatch((0.4, y_pos-0.08), 0.2, 0.15,
                           boxstyle="round,pad=0.01",
                           edgecolor=colors['secondary'], facecolor=colors['secondary'],
                           alpha=0.2, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect2)
    ax.text(0.5, y_pos, 'N\nNonlinear\nOperator', ha='center', va='center',
            fontsize=11, fontweight='bold', transform=ax.transAxes)

    # Solution x
    rect3 = FancyBboxPatch((0.75, y_pos-0.08), 0.2, 0.15,
                           boxstyle="round,pad=0.01",
                           edgecolor=colors['accent'], facecolor=colors['accent'],
                           alpha=0.2, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect3)
    ax.text(0.85, y_pos, 'x*\nFixed Point\nSolution', ha='center', va='center',
            fontsize=11, fontweight='bold', transform=ax.transAxes)

    # Arrows
    ax.annotate('', xy=(0.25, y_pos), xytext=(0.35, y_pos),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                transform=ax.transAxes)
    ax.annotate('', xy=(0.6, y_pos), xytext=(0.7, y_pos),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                transform=ax.transAxes)

    # Properties section
    y_pos = 0.55
    ax.text(0.05, y_pos+0.1, 'Key Properties:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    properties = [
        '• K: Bounded linear monotone operator (integral operator)',
        '• N: Continuous, bounded, possibly monotone',
        '• Solution x satisfies: x + KNx = y',
        '• Solvability depends on monotonicity structure'
    ]

    for i, prop in enumerate(properties):
        ax.text(0.07, y_pos - 0.08 - 0.05*i, prop, fontsize=10,
                transform=ax.transAxes, family='monospace')

    # Integral form at bottom
    y_pos = 0.1
    ax.text(0.5, y_pos+0.05, r'$x(s) + \int_{\Omega} k(s,t) f(t, x(t)) dt = y(s)$',
            ha='center', fontsize=13, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hammerstein_equation_schematic.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: hammerstein_equation_schematic.pdf")

def fig_fixed_point_theorem_hierarchy():
    """Visualize the hierarchy of fixed point theorems."""
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Hierarchy of Fixed Point Theorems',
            ha='center', fontsize=14, fontweight='bold')

    # Draw boxes for each theorem type
    theorems = [
        {'name': 'Brouwer\nFixed Point', 'pos': (1, 7), 'color': colors['primary']},
        {'name': 'Schauder\nFixed Point', 'pos': (4, 7), 'color': colors['secondary']},
        {'name': 'Banach\nContraction', 'pos': (7, 7), 'color': colors['accent']},
        {'name': 'Nonexpansive\nMappings', 'pos': (2.5, 4.5), 'color': colors['primary']},
        {'name': 'Kirk\'s\nTheorem', 'pos': (5.5, 4.5), 'color': colors['secondary']},
        {'name': 'Monotone\nOperators', 'pos': (8.5, 4.5), 'color': colors['accent']},
    ]

    for theorem in theorems:
        rect = FancyBboxPatch((theorem['pos'][0]-0.7, theorem['pos'][1]-0.5),
                              1.4, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor=theorem['color'],
                              facecolor=theorem['color'],
                              alpha=0.3, linewidth=2)
        ax.add_patch(rect)
        ax.text(theorem['pos'][0], theorem['pos'][1], theorem['name'],
                ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connection arrows
    connections = [
        ((1, 6.5), (2.5, 5)),
        ((4, 6.5), (5.5, 5)),
        ((7, 6.5), (8.5, 5)),
    ]

    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Add conditions boxes
    conditions_y = 2.5
    ax.text(5, conditions_y+0.5, 'Required Conditions for Existence:',
            ha='center', fontsize=11, fontweight='bold')

    conditions = [
        'Brouwer: Compact convex set in ℝⁿ',
        'Schauder: Compact convex set in Banach space',
        'Contraction: Complete metric space + contraction property',
        'Nonexpansive: Bounded convex set with normal structure',
        'Kirk: Uniformly convex Banach space',
        'Monotone: Monotone + coercivity conditions'
    ]

    for i, cond in enumerate(conditions):
        ax.text(0.5, conditions_y - 0.35 - 0.3*i, '• ' + cond, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fixed_point_theorem_hierarchy.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: fixed_point_theorem_hierarchy.pdf")

def main():
    """Generate all figures."""
    print("Generating figures for Chapter 6c: Normal Structure & Fixed Point Theory...")
    print()

    fig_banach_space_visualization()
    fig_iterative_convergence()
    fig_monotone_operator_property()
    fig_hammerstein_equation_schematic()
    fig_fixed_point_theorem_hierarchy()

    print()
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
