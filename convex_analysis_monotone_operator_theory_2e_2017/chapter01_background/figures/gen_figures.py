#!/usr/bin/env python3
"""
Generate figures for Chapter 1: Background
Convex Analysis and Monotone Operator Theory in Hilbert Spaces
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Set up matplotlib with TeX-like rendering
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define color scheme
color_main = '#1f77b4'
color_secondary = '#ff7f0e'
color_accent = '#2ca02c'

def fig_lower_semicontinuous():
    """Figure showing a lower semicontinuous function."""
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.linspace(-2, 3, 1000)
    # Create a piecewise function with a lower semicontinuity jump
    y = np.where(x < -1, 0.5*x**2 + 1, np.where(x < 1.5, 0.3*x**2 - 0.5*x + 0.5, -0.3*x + 2))

    ax.plot(x, y, 'b-', linewidth=2.5, label='$f(x)$')

    # Mark a point of interest
    x0 = 0.5
    f_x0 = 0.3*x0**2 - 0.5*x0 + 0.5
    ax.plot(x0, f_x0, 'ro', markersize=8)

    # Draw vertical line and neighborhood indicators
    ax.axvline(x0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(f_x0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add text annotations
    ax.text(x0 + 0.1, f_x0 - 0.1, f'$f(x)$', fontsize=11, ha='left')
    ax.text(x0 - 0.3, -0.3, f'$x$', fontsize=11)

    # Add neighborhood V
    V_left, V_right = x0 - 0.4, x0 + 0.4
    ax.axvline(V_left, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axvline(V_right, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text((V_left + V_right)/2, -0.6, 'V', fontsize=11, ha='center', color='green')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$\\mathbb{R}$', fontsize=12)
    ax.set_title('Lower Semicontinuous Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 3)

    plt.tight_layout()
    plt.savefig('lower_semicontinuous.pdf', dpi=300, bbox_inches='tight')
    print("Generated: lower_semicontinuous.pdf")
    plt.close()

def fig_order_relations():
    """Figure showing order relation properties."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Order Relations on a Set', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)

    y_pos = 0.85
    properties = [
        '1. Reflexive: $(\\forall a \\in A)$ $a \\preceq a$',
        '2. Transitive: $(\\forall a,b,c)$ $a \\preceq b$ and $b \\preceq c$ $\\Rightarrow a \\preceq c$',
        '3. Antisymmetric: $(\\forall a,b)$ $a \\preceq b$ and $b \\preceq a$ $\\Rightarrow a = b$',
        '4. Connex: $(\\forall a,b)$ $a \\preceq b$ or $b \\preceq a$',
    ]

    labels = ['Directed Set', 'Partially Ordered Set', 'Totally Ordered Set']
    requirements = [
        '(1), (2)',
        '(1), (2), (3)',
        '(1), (2), (3), (4)',
    ]

    # Draw requirement table
    y_pos = 0.75
    for i, (label, req) in enumerate(zip(labels, requirements)):
        color = ['#ffe6e6', '#fff4e6', '#e6ffe6'][i]
        rect = FancyBboxPatch((0.05, y_pos - 0.08), 0.9, 0.08,
                             boxstyle="round,pad=0.01",
                             transform=ax.transAxes,
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.1, y_pos - 0.04, f'{label}: {req}', fontsize=11,
               transform=ax.transAxes, va='center')
        y_pos -= 0.12

    # Add property definitions
    y_pos = 0.35
    for prop in properties:
        ax.text(0.08, y_pos, prop, fontsize=10, transform=ax.transAxes, va='top')
        y_pos -= 0.07

    plt.tight_layout()
    plt.savefig('order_relations.pdf', dpi=300, bbox_inches='tight')
    print("Generated: order_relations.pdf")
    plt.close()

def fig_extended_real_line():
    """Figure showing the extended real line."""
    fig, ax = plt.subplots(figsize=(8, 2))

    # Draw the number line
    ax.plot([-3.5, 3.5], [0, 0], 'k-', linewidth=2)

    # Mark points
    points = {'-\\infty': -3.2, '-2': -2, '-1': -1, '0': 0, '1': 1, '2': 2, '+\\infty': 3.2}
    for label, pos in points.items():
        if label in ['-\\infty', '+\\infty']:
            ax.plot(pos, 0, 'ro', markersize=10)
            ax.text(pos, -0.25, '$' + label + '$', fontsize=12, ha='center', fontweight='bold')
        else:
            ax.plot(pos, 0, 'bo', markersize=8)
            ax.text(pos, -0.25, label, fontsize=11, ha='center')

    # Add braces
    ax.text(-3.2, 0.35, 'Extended Real Line: $[-\\infty, +\\infty]$',
           fontsize=12, ha='center', fontweight='bold')
    ax.text(0, -0.7, 'Real Line: $\\mathbb{R}$', fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('extended_real_line.pdf', dpi=300, bbox_inches='tight')
    print("Generated: extended_real_line.pdf")
    plt.close()

def fig_function_concepts():
    """Figure showing function concepts: graph, epigraph, level sets."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Create a simple function
    x = np.linspace(-2, 3, 200)
    y = 0.3*x**2 - 0.5*x + 0.5

    # Plot 1: Graph of f
    ax = axes[0]
    ax.plot(x, y, 'b-', linewidth=2.5, label='$\\text{gra } f$')
    ax.fill_between(x, 0, y, alpha=0.2, color='blue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$f(x)$', fontsize=11)
    ax.set_title('Graph of $f$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-0.5, 3)

    # Plot 2: Epigraph of f
    ax = axes[1]
    ax.plot(x, y, 'b-', linewidth=2.5, label='$\\text{epi } f$')
    ax.fill_between(x, y, 3.5, alpha=0.2, color='blue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$\\xi$', fontsize=11)
    ax.set_title('Epigraph of $f$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-0.5, 3)

    # Plot 3: Level sets
    ax = axes[2]
    xi_levels = [0.3, 1.0, 1.5, 2.0]
    for xi in xi_levels:
        # Solve 0.3*x^2 - 0.5*x + 0.5 = xi
        # 0.3*x^2 - 0.5*x + (0.5 - xi) = 0
        a, b, c = 0.3, -0.5, 0.5 - xi
        disc = b**2 - 4*a*c
        if disc >= 0:
            x1 = (-b - np.sqrt(disc)) / (2*a)
            x2 = (-b + np.sqrt(disc)) / (2*a)
            ax.plot([x1, x2], [xi, xi], 'o-', markersize=8, linewidth=2)
            ax.text((x1+x2)/2, xi + 0.1, f'$\\xi={xi:.1f}$', fontsize=9, ha='center')

    ax.plot(x, y, 'b-', linewidth=2.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$\\xi = f(x)$', fontsize=11)
    ax.set_title('Level Sets of $f$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.savefig('function_concepts.pdf', dpi=300, bbox_inches='tight')
    print("Generated: function_concepts.pdf")
    plt.close()

def fig_net_and_sequence():
    """Figure comparing nets and sequences."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    ax.text(0.5, 0.95, 'Nets vs. Sequences', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Define boxes
    y_seq = 0.75
    y_net = 0.4

    # Sequence box
    seq_box = FancyBboxPatch((0.05, y_seq - 0.15), 0.9, 0.18,
                            boxstyle="round,pad=0.02",
                            transform=ax.transAxes,
                            facecolor='#e6f3ff', edgecolor='blue', linewidth=2)
    ax.add_patch(seq_box)

    ax.text(0.5, y_seq + 0.08, 'Sequence: Indexed by $\\mathbb{N}$',
           fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, y_seq - 0.02, '$(x_n)_{n \\in \\mathbb{N}} = (x_0, x_1, x_2, \\ldots)$',
           fontsize=11, ha='center', transform=ax.transAxes, family='monospace')
    ax.text(0.5, y_seq - 0.08, 'Limited: sequences insufficient for general topologies',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    # Net box
    net_box = FancyBboxPatch((0.05, y_net - 0.15), 0.9, 0.18,
                            boxstyle="round,pad=0.02",
                            transform=ax.transAxes,
                            facecolor='#fff4e6', edgecolor='#ff7f0e', linewidth=2)
    ax.add_patch(net_box)

    ax.text(0.5, y_net + 0.08, 'Net: Indexed by a Directed Set',
           fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, y_net - 0.02, '$(x_a)_{a \\in A}$ where $A$ is a directed set',
           fontsize=11, ha='center', transform=ax.transAxes, family='monospace')
    ax.text(0.5, y_net - 0.08, 'General: applies to all topological spaces',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    # Arrow
    ax.annotate('', xy=(0.5, y_net + 0.2), xytext=(0.5, y_seq - 0.18),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
               transform=ax.transAxes)

    ax.text(0.55, (y_net + y_seq) / 2, 'Generalization', fontsize=10,
           transform=ax.transAxes, style='italic')

    plt.tight_layout()
    plt.savefig('net_vs_sequence.pdf', dpi=300, bbox_inches='tight')
    print("Generated: net_vs_sequence.pdf")
    plt.close()

def fig_closure_and_interior():
    """Figure showing closure and interior of a set."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw the real line
    ax.plot([-4, 4], [0, 0], 'k-', linewidth=1.5)

    # Define a set (open interval)
    left, right = -1.5, 2

    # Interior (open interval)
    ax.plot([left, right], [0, 0], 'g-', linewidth=4, label='Interior: $(a, b)$', alpha=0.7)
    ax.plot(left, 0, 'go', markersize=8, markerfacecolor='none', markeredgewidth=2)
    ax.plot(right, 0, 'go', markersize=8, markerfacecolor='none', markeredgewidth=2)

    # Closure (closed interval)
    ax.plot([left - 0.1, right + 0.1], [0.2, 0.2], 'b-', linewidth=4, label='Closure: $[a, b]$', alpha=0.5)
    ax.plot(left - 0.1, 0.2, 'bo', markersize=8, markerfacecolor='blue')
    ax.plot(right + 0.1, 0.2, 'bo', markersize=8, markerfacecolor='blue')

    # Mark endpoints
    ax.text(left - 0.1, -0.35, '$a$', fontsize=12, ha='center', fontweight='bold')
    ax.text(right + 0.1, -0.35, '$b$', fontsize=12, ha='center', fontweight='bold')

    # Add text
    ax.text(0, 0.6, 'Open set: points at endpoints not included', fontsize=11, ha='center')
    ax.text(0, 0.45, 'Closed set: closure includes boundary points', fontsize=11, ha='center')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 1)
    ax.set_xlabel('$\\mathbb{R}$', fontsize=12)
    ax.set_title('Closure vs. Interior', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('closure_interior.pdf', dpi=300, bbox_inches='tight')
    print("Generated: closure_interior.pdf")
    plt.close()

def fig_compactness():
    """Figure illustrating compact vs non-compact sets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Compact set (closed interval)
    ax = axes[0]
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.5)

    # Draw the interval [0, 2]
    interval = Rectangle((0, 0.5), 2, 0.2, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(interval)

    # Mark endpoints
    ax.plot([0, 2], [0.6, 0.6], 'bo', markersize=10)
    ax.text(0, 0.3, '$a$', fontsize=12, ha='center', fontweight='bold')
    ax.text(2, 0.3, '$b$', fontsize=12, ha='center', fontweight='bold')

    ax.text(1, 1.1, 'Compact: $[a, b] \\subset \\mathbb{R}$', fontsize=12, ha='center', fontweight='bold')
    ax.text(1, 0.85, 'Closed and bounded', fontsize=11, ha='center', style='italic')

    ax.axis('off')
    ax.set_aspect('equal')

    # Non-compact set (open interval)
    ax = axes[1]
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.5)

    # Draw the interval (0, 2)
    interval = Rectangle((0.05, 0.5), 1.9, 0.2, facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(interval)

    # Mark endpoints with open circles
    ax.plot([0, 2], [0.6, 0.6], 'o', color='orange', markersize=10, markerfacecolor='none', markeredgewidth=2)
    ax.text(0, 0.3, '$a$', fontsize=12, ha='center', fontweight='bold')
    ax.text(2, 0.3, '$b$', fontsize=12, ha='center', fontweight='bold')

    ax.text(1, 1.1, 'Non-compact: $(a, b) \\subset \\mathbb{R}$', fontsize=12, ha='center', fontweight='bold')
    ax.text(1, 0.85, 'Open (boundary points excluded)', fontsize=11, ha='center', style='italic')

    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('compactness.pdf', dpi=300, bbox_inches='tight')
    print("Generated: compactness.pdf")
    plt.close()

def fig_operators_diagram():
    """Figure showing operator relationships."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis('off')

    ax.text(0.5, 0.95, 'Types of Operators', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Single-valued operator
    box1 = FancyBboxPatch((0.05, 0.65), 0.4, 0.2,
                         boxstyle="round,pad=0.02",
                         transform=ax.transAxes,
                         facecolor='#e6f3ff', edgecolor='blue', linewidth=2)
    ax.add_patch(box1)
    ax.text(0.25, 0.8, 'Single-Valued', fontsize=12, fontweight='bold',
           ha='center', transform=ax.transAxes)
    ax.text(0.25, 0.72, '$T: X \\to Y$', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.25, 0.67, 'Each $x$ maps to', fontsize=9, ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.25, 0.655, 'exactly one $Tx \\in Y$', fontsize=9, ha='center', transform=ax.transAxes, style='italic')

    # Set-valued operator
    box2 = FancyBboxPatch((0.55, 0.65), 0.4, 0.2,
                         boxstyle="round,pad=0.02",
                         transform=ax.transAxes,
                         facecolor='#fff4e6', edgecolor='#ff7f0e', linewidth=2)
    ax.add_patch(box2)
    ax.text(0.75, 0.8, 'Set-Valued', fontsize=12, fontweight='bold',
           ha='center', transform=ax.transAxes)
    ax.text(0.75, 0.72, '$A: X \\to 2^Y$', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.75, 0.67, 'Each $x$ maps to', fontsize=9, ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.75, 0.655, 'a set $Ax \\subseteq Y$', fontsize=9, ha='center', transform=ax.transAxes, style='italic')

    # Arrow
    ax.annotate('', xy=(0.45, 0.5), xytext=(0.3, 0.65),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
               transform=ax.transAxes)
    ax.annotate('', xy=(0.55, 0.5), xytext=(0.7, 0.65),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
               transform=ax.transAxes)

    # Graph
    box3 = FancyBboxPatch((0.05, 0.2), 0.9, 0.25,
                         boxstyle="round,pad=0.02",
                         transform=ax.transAxes,
                         facecolor='#f0f0f0', edgecolor='black', linewidth=1.5)
    ax.add_patch(box3)

    ax.text(0.5, 0.42, 'Graph and Inverse', fontsize=12, fontweight='bold',
           ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.355, 'Graph: $\\text{gra } A = \\{(x, y) \\in X \\times Y : y \\in Ax\\}$',
           fontsize=11, ha='center', transform=ax.transAxes, family='monospace')
    ax.text(0.5, 0.295, 'Inverse: $A^{-1} = \\{(y, x) \\in Y \\times X : (x, y) \\in \\text{gra } A\\}$',
           fontsize=11, ha='center', transform=ax.transAxes, family='monospace')
    ax.text(0.5, 0.235, 'Domain: $\\text{dom } A = \\{x \\in X : Ax \\ne \\emptyset\\}$',
           fontsize=11, ha='center', transform=ax.transAxes, family='monospace')

    plt.tight_layout()
    plt.savefig('operators_diagram.pdf', dpi=300, bbox_inches='tight')
    print("Generated: operators_diagram.pdf")
    plt.close()

def main():
    """Generate all figures."""
    print("Generating figures for Chapter 1: Background...")
    fig_lower_semicontinuous()
    fig_order_relations()
    fig_extended_real_line()
    fig_function_concepts()
    fig_net_and_sequence()
    fig_closure_and_interior()
    fig_compactness()
    fig_operators_diagram()
    print("All figures generated successfully!")

if __name__ == '__main__':
    main()
