#!/usr/bin/env python3
"""
Generate figures for Chapter 25: Sums of Monotone Operators
Includes diagrams for parallel sum, composition operations, and monotone operator examples
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
fig_dpi = 150

def create_operator_composition_diagram():
    """Create diagram showing parallel composition operation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Individual operators
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title(r'Parallel Composition: $L \triangledown A = (L \circ A^{-1} \circ L^*)^{-1}$',
                  fontsize=11, fontweight='bold')

    # Draw operator A
    rect_a = FancyBboxPatch((0.5, 6), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(rect_a)
    ax1.text(1.5, 7, r'$A$', fontsize=14, ha='center', va='center', fontweight='bold')
    ax1.text(1.5, 5.3, r'$\mathcal{H} \to 2^{\mathcal{H}}$', fontsize=9, ha='center')

    # Draw linear operator L
    rect_l = FancyBboxPatch((5, 6), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(rect_l)
    ax1.text(6, 7, r'$L$', fontsize=14, ha='center', va='center', fontweight='bold')
    ax1.text(6, 5.3, r'$\mathcal{H} \to \mathcal{K}$', fontsize=9, ha='center')

    # Arrow between them
    arrow = FancyArrowPatch((2.8, 7), (4.8, 7), arrowstyle='->',
                           mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow)

    # Result
    rect_result = FancyBboxPatch((2.5, 2), 3, 2, boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(rect_result)
    ax1.text(4, 3, r'$L \triangledown A$', fontsize=13, ha='center', va='center', fontweight='bold')
    ax1.text(4, 1.3, r'Parallel Composition', fontsize=9, ha='center', style='italic')

    # Right: Properties
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(5, 9.5, 'Key Properties', fontsize=12, ha='center', fontweight='bold')

    properties = [
        r'(i) Composition: $(L \triangledown A) \circ B = L \triangledown (A \circ B)$',
        r'(ii) Associativity: $M \triangledown (L \triangledown A) = (M \circ L) \triangledown A$',
        r'(iii) Extends infimal convolution to operators',
        r'(iv) Links conjugation and composition',
    ]

    y_pos = 8.5
    for i, prop in enumerate(properties):
        ax2.text(0.3, y_pos - i*1.8, prop, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('parallel_composition.pdf', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_monotone_sum_diagram():
    """Create diagram showing sum of monotone operators"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, r'Sum of Monotone Operators: $A + B$',
            fontsize=13, ha='center', fontweight='bold')

    # Operator A
    rect_a = FancyBboxPatch((0.5, 6.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                            edgecolor='darkblue', facecolor='lightblue', linewidth=2.5)
    ax.add_patch(rect_a)
    ax.text(1.75, 7.4, r'Operator $A$', fontsize=11, ha='center', fontweight='bold')
    ax.text(1.75, 6.8, r'Monotone', fontsize=9, ha='center', style='italic')

    # Operator B
    rect_b = FancyBboxPatch((3.5, 6.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                            edgecolor='darkred', facecolor='lightcoral', linewidth=2.5)
    ax.add_patch(rect_b)
    ax.text(4.75, 7.4, r'Operator $B$', fontsize=11, ha='center', fontweight='bold')
    ax.text(4.75, 6.8, r'Monotone', fontsize=9, ha='center', style='italic')

    # Plus sign
    ax.text(3.1, 7.4, r'$+$', fontsize=16, ha='center', fontweight='bold')

    # Result
    arrow_down = FancyArrowPatch((2.75, 6), (4, 5.3), arrowstyle='->',
                                mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow_down)

    rect_sum = FancyBboxPatch((2.5, 3.5), 3, 1.8, boxstyle="round,pad=0.1",
                             edgecolor='darkgreen', facecolor='lightgreen', linewidth=2.5)
    ax.add_patch(rect_sum)
    ax.text(4, 4.4, r'$A + B$', fontsize=12, ha='center', fontweight='bold')
    ax.text(4, 3.8, r'Maximally Monotone', fontsize=10, ha='center', style='italic')

    # Conditions box
    ax.text(7.5, 8.5, 'Conditions for', fontsize=11, ha='center', fontweight='bold')
    ax.text(7.5, 8.1, 'Maximal Monotonicity:', fontsize=10, ha='center', fontweight='bold')

    conditions = [
        r'(i) $\text{cone}(\text{dom } B - L(\text{dom } A))$',
        r'$\quad = \overline{\text{span}}(\text{dom } B - L(\text{dom } A))$',
        r'(ii) $A$ and $B$ are 3* monotone',
        r'(iii) $\text{dom } A \subset \text{dom } B$ and $B$ is 3* monotone',
    ]

    y_pos = 7.5
    for cond in conditions:
        ax.text(7.5, y_pos, cond, fontsize=8.5, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
        y_pos -= 0.9

    # Example box
    ax.text(7.5, 1.2, 'Example', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    ax.text(7.5, 0.5, r'$\partial f + N_C$ is maximally monotone', fontsize=8.5, ha='center')

    plt.tight_layout()
    plt.savefig('monotone_sum.pdf', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_operator_hierarchy():
    """Create hierarchy of monotone operator concepts"""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Monotone Operators Hierarchy', fontsize=13, ha='center', fontweight='bold')

    # Level 1: Monotone operators
    rect1 = FancyBboxPatch((2, 8), 6, 0.8, boxstyle="round,pad=0.05",
                           edgecolor='darkblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 8.4, 'Monotone Operators', fontsize=11, ha='center', fontweight='bold')

    # Level 2: Locally maximally monotone
    rect2a = FancyBboxPatch((0.5, 6.5), 4, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='darkblue', facecolor='#ADD8E6', linewidth=1.5)
    ax.add_patch(rect2a)
    ax.text(2.5, 6.9, 'Locally Maximally Monotone', fontsize=10, ha='center')

    # Level 2: Maximally monotone
    rect2b = FancyBboxPatch((5.5, 6.5), 4, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect2b)
    ax.text(7.5, 6.9, 'Maximally Monotone', fontsize=10, ha='center', fontweight='bold')

    # Arrows down
    ax.arrow(5, 7.95, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(3, 6.45, -0.5, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, 6.45, 0.5, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')

    # Level 3: Special classes
    rect3a = FancyBboxPatch((0.2, 5), 2.5, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='darkred', facecolor='#FFB6C6', linewidth=1.5)
    ax.add_patch(rect3a)
    ax.text(1.45, 5.4, '3* Monotone', fontsize=9, ha='center')

    rect3b = FancyBboxPatch((3.2, 5), 2.5, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='purple', facecolor='#E6D7FF', linewidth=1.5)
    ax.add_patch(rect3b)
    ax.text(4.45, 5.4, r'Strongly Monotone', fontsize=9, ha='center')

    rect3c = FancyBboxPatch((6.2, 5), 2.5, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='orange', facecolor='#FFE4B5', linewidth=1.5)
    ax.add_patch(rect3c)
    ax.text(7.45, 5.4, 'Cocoercive', fontsize=9, ha='center')

    # Examples box
    ax.text(5, 3.5, 'Important Examples', fontsize=11, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    examples = [
        r'$\partial f$ (subdifferential): maximally monotone',
        r'$N_C$ (normal cone): 3* monotone',
        r'$\text{Id} - T$ (firmly nonexpansive): monotone & cocoercive',
        r'Sum $A + B$: maximally monotone (under conditions)',
    ]

    y_pos = 3
    for ex in examples:
        ax.text(5, y_pos, ex, fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        y_pos -= 0.65

    plt.tight_layout()
    plt.savefig('operator_hierarchy.pdf', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_numerical_example():
    """Create numerical example plot"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Example 1: Sum of subdifferential and normal cone
    ax = axes[0, 0]
    x = np.linspace(-3, 3, 200)
    # Subdifferential of |x|
    y_subdiff = np.abs(x)
    ax.plot(x, y_subdiff, 'b-', linewidth=2.5, label=r'$\partial |x|$')
    ax.fill_between(x, y_subdiff, alpha=0.3, color='blue')
    ax.set_xlabel(r'$x$', fontsize=10)
    ax.set_ylabel(r'Value', fontsize=10)
    ax.set_title(r'Example: Subdifferential $\partial f$ (Monotone)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 3.5])

    # Example 2: Resolvent (nonexpansive operator)
    ax = axes[0, 1]
    t_vals = np.linspace(0.01, 3, 100)
    # J_A(x) for x=1 with parameter variations
    identity = np.linspace(-2, 2, 100)
    for t_val in [0.5, 1, 2]:
        resolvent = identity / (1 + t_val)
        ax.plot(identity, resolvent, linewidth=2, label=f'$J_{{{t_val}}}(x)$')

    ax.plot(identity, identity, 'k--', linewidth=1.5, alpha=0.5, label='Identity')
    ax.set_xlabel(r'Input $x$', fontsize=10)
    ax.set_ylabel(r'Output $J_t(x)$', fontsize=10)
    ax.set_title(r'Resolvent: Nonexpansive Mapping', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-1.5, 1.5])

    # Example 3: Operator norms
    ax = axes[1, 0]
    operators = ['Projection', r'Resolvent $J_A$', r'Prox$_f$', 'Normal Cone']
    properties = [1.0, 1.0, 1.0, 0]  # Nonexpansive property
    colors = ['blue', 'green', 'red', 'orange']
    bars = ax.bar(operators, [1]*4, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
    ax.set_ylabel('Operator Lipschitz Constant', fontsize=10)
    ax.set_title('Nonexpansive Operators (L ≤ 1)', fontsize=10, fontweight='bold')
    ax.set_ylim([0, 1.3])
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Nonexpansive bound')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Example 4: Convergence of Douglas-Rachford
    ax = axes[1, 1]
    iterations = np.arange(0, 50)
    # Simulated convergence for different step sizes
    for gamma_val in [0.5, 1.0, 2.0]:
        error = 10 * np.exp(-0.15 * gamma_val * iterations)
        ax.semilogy(iterations, error, 'o-', linewidth=2.5, markersize=4, label=f'λ = {gamma_val}')

    ax.set_xlabel('Iteration $k$', fontsize=10)
    ax.set_ylabel(r'Residual Error', fontsize=10)
    ax.set_title(r'Convergence: Operator Splitting Algorithms', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 50])

    plt.tight_layout()
    plt.savefig('numerical_examples.pdf', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

def create_composition_flow():
    """Create flow diagram for compositions and postcomposition"""
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5.5, 6.7, 'Compositions and Postcomposition Operations',
            fontsize=12, ha='center', fontweight='bold')

    # Infimal convolution
    rect1 = FancyBboxPatch((0.2, 5.5), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.3, 5.9, 'Infimal Conv.', fontsize=9, ha='center', fontweight='bold')
    ax.text(1.3, 5.55, '$f \\circ g$', fontsize=8, ha='center', style='italic')

    # Postcomposition
    rect2 = FancyBboxPatch((3.4, 5.5), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkred', facecolor='#FFB6C6', linewidth=2)
    ax.add_patch(rect2)
    ax.text(4.5, 5.9, 'Postcomposition', fontsize=9, ha='center', fontweight='bold')
    ax.text(4.5, 5.55, '$L \\triangledown f$', fontsize=8, ha='center', style='italic')

    # Infimal postcomposition
    rect3 = FancyBboxPatch((6.6, 5.5), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect3)
    ax.text(7.7, 5.9, 'Infimal Postcomp.', fontsize=9, ha='center', fontweight='bold')
    ax.text(7.7, 5.55, r'$L \triangledown A$', fontsize=8, ha='center', style='italic')

    # Arrows showing relationships
    ax.annotate('', xy=(3.2, 5.9), xytext=(2.5, 5.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(6.4, 5.9), xytext=(5.7, 5.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Function level
    ax.text(5.5, 4.8, 'Function Level: Convex Functions', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=1.5))

    # Operator level
    ax.text(5.5, 2.8, 'Operator Level: Monotone Operators', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='darkblue', linewidth=1.5))

    # Operator versions
    rect4 = FancyBboxPatch((0.2, 1.8), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkblue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect4)
    ax.text(1.3, 2.2, 'Sum: $A + B$', fontsize=9, ha='center', fontweight='bold')
    ax.text(1.3, 1.85, '(infimal conv.)', fontsize=7, ha='center', style='italic')

    rect5 = FancyBboxPatch((3.4, 1.8), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkred', facecolor='#FFB6C6', linewidth=2)
    ax.add_patch(rect5)
    ax.text(4.5, 2.2, 'Composition: $L \\circ A$', fontsize=8.5, ha='center', fontweight='bold')
    ax.text(4.5, 1.85, '(postcomposition)', fontsize=7, ha='center', style='italic')

    rect6 = FancyBboxPatch((6.6, 1.8), 2.2, 0.8, boxstyle="round,pad=0.08",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect6)
    ax.text(7.7, 2.2, 'Parallel: $L \\triangledown A$', fontsize=8.5, ha='center', fontweight='bold')
    ax.text(7.7, 1.85, '(infimal postcomp.)', fontsize=7, ha='center', style='italic')

    # Key property box
    ax.text(5.5, 1, 'Key Connection:', fontsize=9, ha='center', fontweight='bold')
    ax.text(5.5, 0.4, r'Infimal postcomposition extends the infimal convolution concept',
            fontsize=8.5, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.text(5.5, -0.15, r'from functions to monotone operators',
            fontsize=8.5, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('composition_flow.pdf', dpi=fig_dpi, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating figures for Chapter 25: Sums of Monotone Operators...")

    create_monotone_sum_diagram()
    print("  ✓ monotone_sum.pdf")

    create_operator_composition_diagram()
    print("  ✓ parallel_composition.pdf")

    create_operator_hierarchy()
    print("  ✓ operator_hierarchy.pdf")

    create_numerical_example()
    print("  ✓ numerical_examples.pdf")

    create_composition_flow()
    print("  ✓ composition_flow.pdf")

    print("\nAll figures generated successfully!")
