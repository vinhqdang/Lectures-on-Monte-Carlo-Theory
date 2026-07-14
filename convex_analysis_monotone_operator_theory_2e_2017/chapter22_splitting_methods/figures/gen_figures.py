#!/usr/bin/env python3
"""
Generate figures for Chapter 22: Stronger Notions of Monotonicity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
fig_dpi = 150

# Color scheme
color_primary = '#1f77b4'
color_secondary = '#ff7f0e'
color_accent = '#2ca02c'
color_neutral = '#7f7f7f'

# ============================================================================
# Figure 1: Monotone vs Paramonotone Operators
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Monotone operator
ax = axes[0]
x = np.linspace(-2, 2, 100)
y_monotone = 0.5 * x  # Monotone: y = 0.5*x

ax.plot(x, y_monotone, 'o-', color=color_primary, linewidth=2.5, markersize=3, label='Monotone A')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$u \\in Ax$', fontsize=12)
ax.set_title('Monotone Operator', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# Add annotation
ax.text(1.5, 1.2, '$\\langle x - y | u - v \\rangle \\geq 0$',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: Paramonotone operator
ax = axes[1]
# Example: graph of A and A^{-1}
x_vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
y_vals = -0.3 * x_vals  # Paramonotone example

ax.plot(x_vals, y_vals, 'o', color=color_secondary, markersize=8, label='Points in gra A')
ax.plot(y_vals, x_vals, 's', color=color_accent, markersize=8, label='Points in gra $A^{-1}$')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.plot([-2, 2], [-2, 2], 'k--', linewidth=1, alpha=0.3, label='Identity line')
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$u$', fontsize=12)
ax.set_title('Paramonotone: Both A and $A^{-1}$ Monotone', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/fig_monotonicity_types.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Cyclic Monotonicity Illustration
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Example: Subdifferential of f(x) = (1/2)||x||^2
x = np.linspace(-2, 2, 100)
f = 0.5 * x**2

ax.plot(x, f, 'b-', linewidth=2.5, label='$f(x) = \\frac{1}{2}\\|x\\|^2$')

# Mark some points on the graph
x_pts = np.array([-1.5, -0.75, 0, 0.75, 1.5])
f_pts = 0.5 * x_pts**2
u_pts = x_pts  # Gradient at these points

ax.plot(x_pts, f_pts, 'o', color=color_secondary, markersize=10)

# Draw tangent lines (subdifferential)
for i, (xi, fi, ui) in enumerate(zip(x_pts, f_pts, u_pts)):
    x_tan = np.linspace(xi - 1.5, xi + 1.5, 50)
    f_tan = fi + ui * (x_tan - xi)
    ax.plot(x_tan, f_tan, '--', color=color_accent, alpha=0.4, linewidth=1)

ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$f(x)$', fontsize=13)
ax.set_title('Cyclic Monotonicity: Subdifferential of Convex Function', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 3.5)

# Add text box
textstr = '$\\partial f$ is cyclically monotone\n$A = \\partial f \\Rightarrow$ Rockafellar\'s Theorem'
ax.text(-2, 2.8, textstr, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/fig_cyclic_monotonicity.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Cyclic Sequence in ℝ²
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Example: Cyclic monotonicity on R^2
# Matrix A = [0 -1; 1 0] (counterclockwise rotation by π/2)
theta = np.pi / 2

# Generate a cycle of points
n = 4
angles = np.linspace(0, 2*np.pi, n+1)[:-1]
r = 1.5
cycle_x = r * np.cos(angles)
cycle_y = r * np.sin(angles)

# For cyclic monotonicity visualization
ax.set_aspect('equal')

# Plot the cycle
for i in range(n):
    j = (i + 1) % n
    ax.arrow(cycle_x[i], cycle_y[i],
             cycle_x[j] - cycle_x[i], cycle_y[j] - cycle_y[i],
             head_width=0.15, head_length=0.1, fc=color_primary, ec=color_primary, alpha=0.7)

ax.plot(cycle_x, cycle_y, 'o', color=color_secondary, markersize=12, zorder=5)

# Add labels
for i in range(n):
    ax.text(cycle_x[i]*1.3, cycle_y[i]*1.3, f'$(x_{i}, u_{i})$',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title('Cyclic Monotonicity: Sequence in $\\mathbb{R}^2$', fontsize=13, fontweight='bold')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Add condition text
textstr = '$\\sum_{i=1}^{n} \\langle x_{i+1} - x_i | u_i \\rangle \\leq 0$\nfor all cycles'
ax.text(-2.5, -2.5, textstr, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/fig_cyclic_sequence.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Rockafellar Theorem Diagram
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, "Rockafellar's Cyclic Monotonicity Theorem",
        fontsize=14, fontweight='bold', ha='center')

# Left box
box1 = Rectangle((0.5, 6.5), 4, 2, linewidth=2, edgecolor=color_primary, facecolor='lightblue', alpha=0.3)
ax.add_patch(box1)
ax.text(2.5, 7.8, "Subdifferential", fontsize=12, fontweight='bold', ha='center')
ax.text(2.5, 7.3, "$A = \\partial f$", fontsize=11, ha='center')
ax.text(2.5, 6.8, "$f \\in \\Gamma_0(\\mathcal{H})$", fontsize=10, ha='center')

# Right box
box2 = Rectangle((5.5, 6.5), 4, 2, linewidth=2, edgecolor=color_secondary, facecolor='lightyellow', alpha=0.3)
ax.add_patch(box2)
ax.text(7.5, 7.8, "Cyclic Monotonicity", fontsize=12, fontweight='bold', ha='center')
ax.text(7.5, 7.3, "$A$ is maximally", fontsize=11, ha='center')
ax.text(7.5, 6.8, "cyclically monotone", fontsize=10, ha='center')

# Arrow: Left to Right
ax.arrow(4.7, 7.5, 0.6, 0, head_width=0.25, head_length=0.2, fc='black', ec='black')
ax.text(5.2, 8, "Prop. 22.14", fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Arrow: Right to Left
ax.arrow(5.3, 7, -0.6, 0, head_width=0.25, head_length=0.2, fc='black', ec='black')
ax.text(4.8, 6.5, "Thm. 22.18", fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Consequences box
box3 = Rectangle((1, 2), 8, 3.5, linewidth=2, edgecolor=color_accent, facecolor='lightgreen', alpha=0.2)
ax.add_patch(box3)
ax.text(5, 5.2, "Key Consequences", fontsize=12, fontweight='bold', ha='center')

consequences = [
    "1. Every subdifferential is cyclically monotone",
    "2. A is maximally cyclically monotone $\\Leftrightarrow$ $A = \\partial f$ for some $f$",
    "3. Provides structural understanding of monotone operators",
    "4. Connection to optimization: subdifferentials characterize convex functions"
]

y_pos = 4.7
for cons in consequences:
    ax.text(5, y_pos, cons, fontsize=10, ha='center')
    y_pos -= 0.65

# Bottom note
ax.text(5, 0.8, "This is a fundamental result in convex analysis connecting monotone operators to convex analysis",
        fontsize=10, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('figures/fig_rockafellar_theorem.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Comparison of Monotonicity Types
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, "Types of Monotonicity for Operators", fontsize=14, fontweight='bold', ha='center')

monotonicity_types = [
    {
        'name': 'Monotone',
        'def': '$\\langle x - y | u - v \\rangle \\geq 0$',
        'pos': (1.5, 7),
        'color': 'lightblue'
    },
    {
        'name': 'Paramonotone',
        'def': 'Both $A$ and $A^{-1}$ monotone',
        'pos': (5, 7),
        'color': 'lightyellow'
    },
    {
        'name': 'Strictly Monotone',
        'def': 'Monotone + inequality strict when $x \\neq y$',
        'pos': (8.5, 7),
        'color': 'lightcyan'
    },
    {
        'name': 'Uniformly Monotone',
        'def': '$\\langle x - y | u - v \\rangle \\geq \\phi(\\|x-y\\|)$',
        'pos': (1.5, 4),
        'color': 'lightcoral'
    },
    {
        'name': 'Strongly Monotone',
        'def': '$\\langle x - y | u - v \\rangle \\geq \\beta \\|x - y\\|^2$',
        'pos': (5, 4),
        'color': 'lightgray'
    },
    {
        'name': 'Cyclically Monotone',
        'def': '$\\sum_{i=1}^n \\langle x_{i+1} - x_i | u_i \\rangle \\leq 0$',
        'pos': (8.5, 4),
        'color': 'lightgreen'
    }
]

for mt in monotonicity_types:
    x, y = mt['pos']

    # Draw box
    box = Rectangle((x-1.3, y-1), 2.6, 1.5, linewidth=2,
                   edgecolor='black', facecolor=mt['color'], alpha=0.6)
    ax.add_patch(box)

    # Add text
    ax.text(x, y+0.4, mt['name'], fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(x, y-0.3, mt['def'], fontsize=8, ha='center', va='center')

# Hierarchy
ax.text(5, 1.8, "Relationships:", fontsize=11, fontweight='bold', ha='center')
ax.text(5, 1.2, "Strongly Monotone $\\Rightarrow$ Uniformly Monotone $\\Rightarrow$ Monotone",
        fontsize=9, ha='center')
ax.text(5, 0.6, "Subdifferential of Convex Function $\\Rightarrow$ Cyclically Monotone",
        fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig('figures/fig_monotonicity_types_detailed.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 6: Numerical Example - Monotone Operator
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Operator A(x) = x (identity, strongly monotone)
ax = axes[0]
x_vals = np.linspace(-3, 3, 50)
A_vals = x_vals  # A(x) = x

ax.plot(x_vals, A_vals, 'o-', color=color_primary, linewidth=2.5, markersize=4, label='$A(x) = x$')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$u = A(x)$', fontsize=12)
ax.set_title('Strongly Monotone: $A(x) = x$', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)

# Right: Cocoercive operator
ax = axes[1]
x_vals = np.linspace(-3, 3, 50)
A_vals = 0.5 * x_vals  # A(x) = 0.5*x

ax.plot(x_vals, A_vals, 's-', color=color_secondary, linewidth=2.5, markersize=4, label='$A(x) = 0.5x$')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$u = A(x)$', fontsize=12)
ax.set_title('Monotone and Cocoercive: $A(x) = 0.5x$', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)

plt.tight_layout()
plt.savefig('figures/fig_monotone_examples.pdf', dpi=fig_dpi, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Generated figures:")
print("  - figures/fig_monotonicity_types.pdf")
print("  - figures/fig_cyclic_monotonicity.pdf")
print("  - figures/fig_cyclic_sequence.pdf")
print("  - figures/fig_rockafellar_theorem.pdf")
print("  - figures/fig_monotonicity_types_detailed.pdf")
print("  - figures/fig_monotone_examples.pdf")
