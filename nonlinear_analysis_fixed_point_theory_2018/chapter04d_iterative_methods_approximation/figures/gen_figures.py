#!/usr/bin/env python3
"""
Figure generation for Chapter 4d: Iterative Methods and Fixed Point Theory
Visualizations of key concepts from sections 5.8-5.9
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'light': '#E8F1F5'
}

def setup_figure(figsize=(8, 6)):
    """Create and setup a figure with consistent style."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    return fig, ax

def save_figure(fig, filename):
    """Save figure as PDF with proper formatting."""
    fig.tight_layout()
    fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)

# Figure 1: Contraction Mapping Concept
def fig_contraction_mapping():
    """Visualize the contraction mapping principle."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Function graph showing contraction
    x = np.linspace(0, 1, 100)
    y_identity = x
    y_contraction = 0.6 * x + 0.1

    ax1.plot(x, y_identity, 'k--', linewidth=2, label='$y = x$ (fixed point line)', alpha=0.7)
    ax1.plot(x, y_contraction, color=colors['primary'], linewidth=2.5, label='$y = T(x)$ (contraction)')
    ax1.fill_between(x, y_contraction, y_identity, alpha=0.2, color=colors['primary'])

    # Show iteration
    x0 = 0.2
    for i in range(5):
        y0 = y_contraction[int(x0 * 99)]
        ax1.plot([x0, x0], [x0, y0], 'r--', alpha=0.5, linewidth=1)
        ax1.plot([x0, y0], [y0, y0], 'r--', alpha=0.5, linewidth=1)
        ax1.plot(x0, y0, 'ro', markersize=6, alpha=0.7)
        x0 = y0

    fixed_point = 0.25  # Theoretical fixed point
    ax1.plot(fixed_point, fixed_point, 'g*', markersize=20, label=f'Fixed point $x^*$')

    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$T(x)$', fontsize=12)
    ax1.set_title('Contraction Mapping: Iteration Converges to Fixed Point', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Right: Contraction ratio visualization
    iterations = np.arange(0, 10)
    error_contraction = 0.1 * (0.6 ** iterations)
    error_noncontraction = 0.1 * np.exp(-0.3 * iterations)

    ax2.semilogy(iterations, error_contraction, 'o-', color=colors['primary'],
                 linewidth=2.5, markersize=8, label='Contraction (ratio=0.6)')
    ax2.semilogy(iterations, error_noncontraction, 's-', color=colors['secondary'],
                 linewidth=2.5, markersize=8, label='Nonexpansive', alpha=0.7)

    ax2.set_xlabel('Iteration $n$', fontsize=12)
    ax2.set_ylabel('Error $\|x_n - x^*\|$', fontsize=12)
    ax2.set_title('Convergence Rate Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    return fig

# Figure 2: Lipschitz Continuity and Contractiveness
def fig_lipschitz_contractive():
    """Compare Lipschitz continuous vs contractive mappings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 1, 100)

    # Different types of mappings
    y_lipschitz = 0.8 * x + 0.1
    y_contraction = 0.5 * x + 0.1
    y_identity = x
    y_nonexpansive = np.sqrt(x) * 0.8

    ax.plot(x, y_identity, 'k--', linewidth=2, label='Identity $y=x$', alpha=0.6)
    ax.plot(x, y_nonexpansive, '-', color=colors['accent'], linewidth=2.5,
            label='Nonexpansive: $\|Tx - Ty\| \leq \|x - y\|$ (ratio = 1)')
    ax.plot(x, y_contraction, '-', color=colors['primary'], linewidth=2.5,
            label='Contraction: $\|Tx - Ty\| \leq k\|x - y\|$ (ratio = 0.5)')
    ax.plot(x, y_lipschitz, '-', color=colors['secondary'], linewidth=2.5, alpha=0.7,
            label='Lipschitz: $\|Tx - Ty\| \leq L\|x - y\|$ (ratio = 0.8)')

    # Shade regions
    ax.fill_between(x, y_contraction, y_nonexpansive, alpha=0.15, color=colors['primary'],
                     label='Contraction region')
    ax.fill_between(x, y_nonexpansive, y_identity, alpha=0.1, color=colors['secondary'],
                     label='Nonexpansive region')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$T(x)$', fontsize=12)
    ax.set_title('Classification of Mappings by Lipschitz Constant', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add text box with definitions
    textstr = 'Contraction: $k < 1$\nNonexpansive: $k = 1$\nLipschitzian: $k > 1$'
    props = dict(boxstyle='round', facecolor=colors['light'], alpha=0.8, edgecolor=colors['primary'])
    ax.text(0.65, 0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    return fig

# Figure 3: Banach Space Operators
def fig_operator_types():
    """Visualize different types of Banach space operators."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Type 1: Compact Operator
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')

    circle_original = Circle((0, 0), 1, fill=False, edgecolor='gray',
                             linewidth=2, linestyle='--', label='Original set')
    ax1.add_patch(circle_original)

    # Compact image
    theta = np.linspace(0, 2*np.pi, 100)
    x_compact = 0.4 * np.cos(theta)
    y_compact = 0.3 * np.sin(theta)
    ax1.fill(x_compact, y_compact, color=colors['primary'], alpha=0.3)
    ax1.plot(x_compact, y_compact, color=colors['primary'], linewidth=2.5, label='Image $T(X)$')
    ax1.set_title('Compact Operator: Image is relatively compact', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('$\mathbb{R}$')
    ax1.set_ylabel('$\mathbb{R}$')

    # Type 2: Totally Bounded
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')

    circle2 = Circle((0, 0), 1, fill=False, edgecolor='gray',
                     linewidth=2, linestyle='--', label='Original set')
    ax2.add_patch(circle2)

    # Image of totally bounded set - spreads but covers finite epsilon-net
    points = np.random.randn(200, 2) * 0.6
    ax2.scatter(points[:, 0], points[:, 1], c=colors['secondary'], s=20,
               alpha=0.6, label='Image with $\epsilon$-net')

    # Show epsilon balls
    for i in range(3):
        idx = i * 50 + 30
        circle_eps = Circle((points[idx, 0], points[idx, 1]), 0.15,
                           fill=False, edgecolor=colors['secondary'],
                           linewidth=1, linestyle=':', alpha=0.5)
        ax2.add_patch(circle_eps)

    ax2.set_title('Totally Bounded Operator: Covered by finite $\epsilon$-balls',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('$\mathbb{R}$')
    ax2.set_ylabel('$\mathbb{R}$')

    # Type 3: Completely Continuous (both properties)
    ax3.text(0.5, 0.7, 'Completely Continuous Operator', fontsize=12, fontweight='bold',
            ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, 'Continuous + Totally Bounded', fontsize=11,
            ha='center', transform=ax3.transAxes, style='italic')

    props_list = [
        '• Continuous on its domain',
        '• Maps bounded sets to totally bounded sets',
        '• Compact operators are completely continuous',
        '• Every completely continuous is totally bounded'
    ]
    y_pos = 0.35
    for prop in props_list:
        ax3.text(0.1, y_pos, prop, fontsize=10, transform=ax3.transAxes)
        y_pos -= 0.10

    ax3.axis('off')

    # Type 4: Theorem relationships
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    # Create hierarchy
    boxes = [
        {'xy': (1, 7), 'text': 'Lipschitzian\nMappings', 'color': colors['primary']},
        {'xy': (5, 7), 'text': '$\mathcal{D}$-Lipschitzian\nMappings', 'color': colors['secondary']},
        {'xy': (3, 4), 'text': '$\mathcal{P}$-Lipschitzian\nMappings', 'color': colors['accent']},
    ]

    for box in boxes:
        rect = FancyBboxPatch(box['xy'], 2, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=box['color'], facecolor=colors['light'],
                             linewidth=2, alpha=0.8)
        ax4.add_patch(rect)
        ax4.text(box['xy'][0] + 1, box['xy'][1] + 0.6, box['text'],
                ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows showing relationships
    ax4.annotate('', xy=(3.8, 6.5), xytext=(2.3, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax4.text(3, 6.8, 'generalization', ha='center', fontsize=8, style='italic')

    ax4.annotate('', xy=(3, 5.2), xytext=(2.5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax4.annotate('', xy=(4, 5.2), xytext=(5.5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    ax4.set_title('Operator Classification Hierarchy', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(2.5, 9)

    fig.tight_layout()
    return fig

# Figure 4: Fixed Point Iteration Convergence
def fig_iteration_convergence():
    """Visualize convergence of fixed point iterations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    n_iter = 50

    # Case 1: Banach Contraction with good convergence
    iterations = np.arange(n_iter)
    error1 = 1.0 * (0.5 ** iterations)
    ax1.semilogy(iterations, error1, 'o-', color=colors['primary'], linewidth=2.5, markersize=5)
    ax1.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Machine precision')
    ax1.set_xlabel('Iteration $n$', fontsize=11)
    ax1.set_ylabel('Error $e_n$', fontsize=11)
    ax1.set_title('Banach Contraction: Exponential Convergence', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=9)

    # Case 2: Mann iteration (slower)
    error2 = 1.0 / (1 + 0.5 * iterations)
    error2[0] = 1.0
    ax2.semilogy(iterations, error2, 's-', color=colors['secondary'], linewidth=2.5, markersize=5)
    ax2.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Machine precision')
    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('Error $e_n$', fontsize=11)
    ax2.set_title('Mann Iteration: Slower Convergence', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=9)

    # Case 3: Comparison of different methods
    error_banach = 1.0 * (0.5 ** iterations)
    error_mann = 1.0 / (1 + 0.3 * iterations)
    error_mann[0] = 1.0
    error_krasnoselski = 1.0 * np.exp(-0.1 * iterations)

    ax3.semilogy(iterations[:30], error_banach[:30], 'o-', color=colors['primary'],
                linewidth=2, markersize=4, label='Banach Contraction')
    ax3.semilogy(iterations[:30], error_mann[:30], 's-', color=colors['secondary'],
                linewidth=2, markersize=4, label='Mann Iteration')
    ax3.semilogy(iterations[:30], error_krasnoselski[:30], '^-', color=colors['accent'],
                linewidth=2, markersize=4, label='Krasnoselski')

    ax3.set_xlabel('Iteration $n$', fontsize=11)
    ax3.set_ylabel('Error $e_n$', fontsize=11)
    ax3.set_title('Convergence Rate Comparison', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9)

    # Case 4: Convergence regions in parameter space
    lambda_vals = np.linspace(0, 1, 100)

    # Contraction region
    ax4.axvspan(0, 1, alpha=0.2, color=colors['primary'], label='Contraction: $k < 1$')
    ax4.axvline(x=1, color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # Plot boundary
    ax4.plot([1, 1], [0, 1], 'k--', linewidth=2)
    ax4.plot([0, 1.2], [0, 0], 'k-', linewidth=2)
    ax4.plot([0, 0], [0, 1.2], 'k-', linewidth=2)

    # Annotations
    ax4.text(0.4, 0.8, 'Guaranteed\nConvergence', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=colors['primary'], alpha=0.3))
    ax4.text(1.1, 0.8, 'No guarantee\nof convergence', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    ax4.set_xlabel('Lipschitz Constant $k$', fontsize=11)
    ax4.set_ylabel('Convergence', fontsize=11)
    ax4.set_title('Convergence Region in Parameter Space', fontsize=11, fontweight='bold')
    ax4.set_xlim(-0.1, 1.3)
    ax4.set_ylim(-0.1, 1.2)
    ax4.set_xticks([0, 0.5, 1.0, 1.2])
    ax4.set_yticks([0, 0.5, 1.0])
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    return fig

# Figure 5: Lattice Fixed Point Theorem
def fig_lattice_fixed_points():
    """Visualize lattice-theoretic fixed points."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Poset structure
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 5.5)
    ax1.set_aspect('equal')

    # Draw lattice elements
    positions = {
        'top': (2, 5),
        'middle_left': (1, 3),
        'middle_right': (3, 3),
        'bottom_left': (0.5, 1),
        'bottom_middle': (2, 1),
        'bottom_right': (3.5, 1),
        'bottom': (2, -0.2)
    }

    for name, pos in positions.items():
        if name != 'bottom':
            circle = Circle(pos, 0.25, color=colors['primary'], alpha=0.7, zorder=10)
            ax1.add_patch(circle)
            ax1.text(pos[0], pos[1], name[0].upper(), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white', zorder=11)

    # Draw partial order relations
    edges = [
        ('bottom_left', 'middle_left'),
        ('bottom_middle', 'middle_left'),
        ('bottom_middle', 'middle_right'),
        ('bottom_right', 'middle_right'),
        ('middle_left', 'top'),
        ('middle_right', 'top'),
    ]

    for start, end in edges:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax1.arrow(x1, y1 + 0.3, x2 - x1, y2 - y1 - 0.6, head_width=0.15,
                 head_length=0.1, fc='gray', ec='gray', alpha=0.6, zorder=1)

    ax1.set_title('Partially Ordered Set (Poset)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Monotone functions on lattice
    x = np.linspace(0, 10, 100)
    y_mono_inc = 0.8 * x + 1
    y_mono_dec = -0.6 * x + 8
    y_identity = x

    ax2.plot(x, y_identity, 'k--', linewidth=2, alpha=0.5, label='Identity')
    ax2.plot(x, y_mono_inc, 'o-', color=colors['primary'], linewidth=2.5,
            markersize=4, label='Monotone increasing')
    ax2.plot(x, y_mono_dec, 's-', color=colors['secondary'], linewidth=2.5,
            markersize=4, label='Monotone decreasing')

    # Mark fixed points
    fp1_x = 5 / (1 - 0.8)
    fp1_y = fp1_x
    ax2.plot(fp1_x if fp1_x < 10 else 10, fp1_y if fp1_y < 10 else 10,
            'g*', markersize=20, label='Fixed point', zorder=10)

    ax2.set_xlabel('$x$', fontsize=11)
    ax2.set_ylabel('$T(x)$', fontsize=11)
    ax2.set_title('Monotone Functions on Lattice', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Panel 3: Knaster-Tarski Theorem
    ax3.text(0.5, 0.9, 'Knaster-Tarski Fixed Point Theorem', fontsize=12, fontweight='bold',
            ha='center', transform=ax3.transAxes)

    theorem_text = (
        'If $f: L \\to L$ is a monotone increasing function\n'
        'on a complete lattice $L$, then:\n\n'
        '1. The set of fixed points is non-empty\n'
        '2. The set of fixed points forms a lattice\n'
        '3. $f$ has a least and greatest fixed point\n\n'
        'Applications:\n'
        '• Static analysis in program verification\n'
        '• Computation of reachable states\n'
        '• Dataflow analysis'
    )

    ax3.text(0.5, 0.45, theorem_text, fontsize=10, ha='center', va='center',
            transform=ax3.transAxes, bbox=dict(boxstyle='round',
            facecolor=colors['light'], alpha=0.8, edgecolor=colors['primary'], linewidth=2))
    ax3.axis('off')

    # Panel 4: Reflexivity property
    ax4.text(0.5, 0.9, 'Reflexivity in Banach Lattices', fontsize=12, fontweight='bold',
            ha='center', transform=ax4.transAxes)

    reflexivity_text = (
        'A Banach lattice is reflexive if:\n\n'
        '• Every bounded sequence has a\n'
        '  weakly convergent subsequence\n\n'
        'Properties of reflexive spaces:\n'
        '• Every continuous operator has\n'
        '  a property related to fixed points\n'
        '• Enables existence of fixed points for\n'
        '  wider classes of operators\n'
        '• Important for applications in\n'
        '  functional analysis and PDEs'
    )

    ax4.text(0.5, 0.45, reflexivity_text, fontsize=10, ha='center', va='center',
            transform=ax4.transAxes, bbox=dict(boxstyle='round',
            facecolor=colors['light'], alpha=0.8, edgecolor=colors['secondary'], linewidth=2))
    ax4.axis('off')

    fig.tight_layout()
    return fig

# Figure 6: Krasnoselski and Burton Theorems
def fig_krasnoselski_burton():
    """Visualize the geometric interpretation of Krasnoselski-Burton theorems."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Krasnoselski Theorem
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 5)
    ax1.set_aspect('equal')

    # Draw convex set S
    S_vertices = np.array([[0.5, 0.5], [4.5, 0.5], [4.5, 4.5], [0.5, 4.5]])
    S_patch = Polygon(S_vertices, fill=True, facecolor=colors['light'],
                     edgecolor=colors['primary'], linewidth=2.5, alpha=0.7)
    ax1.add_patch(S_patch)
    ax1.text(2.5, 2.5, 'Convex Set $S$', fontsize=11, ha='center', fontweight='bold')

    # Show fixed point
    fp = np.array([2.5, 2.5])
    ax1.plot(fp[0], fp[1], 'g*', markersize=20, label="$x^* \\in S$", zorder=10)

    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.set_title('Krasnoselski: $A$ is contraction, $B$ is completely continuous',
                 fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Burton modification
    ax2.text(0.05, 0.95, 'Burton\'s Modification of Krasnoselski', fontsize=11, fontweight='bold',
            ha='left', va='top', transform=ax2.transAxes)

    burton_text = (
        'Krasnoselski (1964):\n'
        '• $A$: contraction on $S$\n'
        '• $B$: completely continuous\n'
        '• $Ax + By \\in S$\n\n'
        'Burton (1995) weakened:\n'
        '• $A$: still contraction\n'
        '• $B$: maps $S$ to bounded sets\n'
        '• Weaker covering condition\n\n'
        'Both guarantee fixed points for\n'
        'operator equation: $Ax + Bx = x$'
    )

    ax2.text(0.05, 0.45, burton_text, fontsize=9.5, ha='left', va='top',
            transform=ax2.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor=colors['light'],
            alpha=0.8, edgecolor=colors['secondary']))
    ax2.axis('off')

    # Panel 3: $\mathcal{P}$-Lipschitzian mappings
    ax3.text(0.5, 0.95, '$\\mathcal{P}$-Lipschitzian Mappings', fontsize=11, fontweight='bold',
            ha='center', transform=ax3.transAxes)

    p_lips_text = (
        'Definition:\n'
        '$T$ is $\\mathcal{P}$-Lipschitzian if:\n\n'
        '$\\|Tx - Ty\\| \\leq \\phi(\\|x - y\\|)$\n\n'
        'where $\\phi$ is a continuous function\n'
        'with $\\phi(r) < r$ for $r > 0$\n\n'
        'Key property:\n'
        '• Generalizes contractions and\n'
        '  nonexpansive mappings\n'
        '• Enables analysis of iterative\n'
        '  methods in infinite dimensions'
    )

    ax3.text(0.5, 0.45, p_lips_text, fontsize=10, ha='center', va='center',
            transform=ax3.transAxes, bbox=dict(boxstyle='round',
            facecolor=colors['light'], alpha=0.8, edgecolor=colors['primary'], linewidth=2))
    ax3.axis('off')

    # Panel 4: $\mathcal{D}$-Lipschitzian mappings
    ax4.text(0.5, 0.95, '$\\mathcal{D}$-Lipschitzian Mappings', fontsize=11, fontweight='bold',
            ha='center', transform=ax4.transAxes)

    d_lips_text = (
        'Definition (Dhage 2003):\n'
        '$T$ is $\\mathcal{D}$-Lipschitzian if:\n\n'
        '$\\|Tx - Ty\\| \\leq \\phi(\\|x - y\\|)$\n\n'
        'where $\\phi: \\mathbb{R}^+ \\to \\mathbb{R}^+$\n'
        'is continuous, nondecreasing,\n'
        'with $\\phi(0) = 0$\n\n'
        'Key difference:\n'
        '• No requirement that $\\phi(r) < r$\n'
        '• More general than $\\mathcal{P}$-Lipschitzian\n'
        '• Includes broader class of mappings'
    )

    ax4.text(0.5, 0.45, d_lips_text, fontsize=10, ha='center', va='center',
            transform=ax4.transAxes, bbox=dict(boxstyle='round',
            facecolor=colors['light'], alpha=0.8, edgecolor=colors['secondary'], linewidth=2))
    ax4.axis('off')

    fig.tight_layout()
    return fig

# Generate all figures
def main():
    """Generate all figures."""
    print("Generating figures for Chapter 4d: Iterative Methods and Fixed Point Theory\n")

    fig = fig_contraction_mapping()
    save_figure(fig, 'fig_contraction_mapping.pdf')

    fig = fig_lipschitz_contractive()
    save_figure(fig, 'fig_lipschitz_contractive.pdf')

    fig = fig_operator_types()
    save_figure(fig, 'fig_operator_types.pdf')

    fig = fig_iteration_convergence()
    save_figure(fig, 'fig_iteration_convergence.pdf')

    fig = fig_lattice_fixed_points()
    save_figure(fig, 'fig_lattice_fixed_points.pdf')

    fig = fig_krasnoselski_burton()
    save_figure(fig, 'fig_krasnoselski_burton.pdf')

    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
