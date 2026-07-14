#!/usr/bin/env python3
"""
Generate figures for Chapter 5d: Linear Operators - Basic Theory
Illustrates key concepts: operator continuity, boundedness, convergence
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
np.random.seed(42)

# Color scheme
color_x = '#1f77b4'      # Blue - domain
color_y = '#ff7f0e'      # Orange - codomain
color_map = '#2ca02c'    # Green - operator
color_bound = '#d62728'  # Red - bounded region

def save_pdf(filename):
    """Save figure as PDF to figures directory"""
    filepath = f'/home/user/Lectures-on-Monte-Carlo-Theory/nonlinear_analysis_fixed_point_theory_2018/chapter05d_linear_operators_basic_theory/figures/{filename}'
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Figure 1: Linear vs Nonlinear Mapping
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Linear mapping
x_vals = np.linspace(-3, 3, 100)
y_linear = 2 * x_vals
ax1.plot(x_vals, y_linear, color=color_map, linewidth=3, label='A(x) = 2x (Linear)')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.scatter([1, 2], [2, 4], color=color_x, s=100, zorder=5)
ax1.scatter([2, 4], [4, 8], color=color_y, s=100, zorder=5)
ax1.plot([1, 2], [2, 4], 'k--', alpha=0.3, linewidth=1)
ax1.plot([2, 4], [4, 8], 'k--', alpha=0.3, linewidth=1)
ax1.arrow(1, 2.5, 0.8, 1.5, head_width=0.15, head_length=0.2, fc='gray', ec='gray', alpha=0.5)
ax1.text(1.5, 3.5, 'A(1)=2', fontsize=10, ha='center')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('A(x)', fontsize=12)
ax1.set_title('Linear Operator: A(αx + βy) = αA(x) + βA(y)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-6, 6)

# Nonlinear mapping
y_nonlinear = x_vals**2
ax2.plot(x_vals, y_nonlinear, color=color_map, linewidth=3, label='T(x) = x² (Nonlinear)')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.scatter([1, 2], [1, 4], color=color_x, s=100, zorder=5)
ax2.scatter([1, 4], [1, 16], color=color_y, s=100, zorder=5)
ax2.plot([1, 2], [1, 4], 'k--', alpha=0.3, linewidth=1)
ax2.plot([1, 4], [1, 16], 'k--', alpha=0.3, linewidth=1)
ax2.text(1.5, 2.5, 'T(1)=1', fontsize=10, ha='center')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('T(x)', fontsize=12)
ax2.set_title('Nonlinear Operator: T(x) ≠ αT(x)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1, 9)

plt.tight_layout()
save_pdf('linear_vs_nonlinear.pdf')

# Figure 2: Bounded vs Unbounded Operators
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bounded operator
x_bounded = np.linspace(-2, 2, 100)
y_bounded = np.tanh(2 * x_bounded)  # Bounded by approximately [-1, 1]
ax1.plot(x_bounded, y_bounded, color=color_map, linewidth=3, label='Bounded: A(x) = tanh(2x)')
ax1.axhline(y=1, color=color_bound, linewidth=2, linestyle='--', label='‖A(x)‖ ≤ 1')
ax1.axhline(y=-1, color=color_bound, linewidth=2, linestyle='--')
ax1.fill_between(x_bounded, -1, 1, alpha=0.1, color=color_bound)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('A(x)', fontsize=12)
ax1.set_title('Bounded Operator: ‖A(x)‖ ≤ M‖x‖', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.5, 1.5)

# Unbounded operator
x_unbounded = np.linspace(-2, 2, 100)
# Avoid x=0 for unbounded function
x_unbounded = x_unbounded[np.abs(x_unbounded) > 0.1]
y_unbounded = 1 / x_unbounded  # Classic unbounded operator
ax2.plot(x_unbounded[x_unbounded > 0], y_unbounded[x_unbounded > 0],
         color=color_map, linewidth=3, label='Unbounded: T(x) = 1/x')
ax2.plot(x_unbounded[x_unbounded < 0], y_unbounded[x_unbounded < 0],
         color=color_map, linewidth=3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('T(x)', fontsize=12)
ax2.set_title('Unbounded Operator: No finite M', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-10, 10)
ax2.set_xlim(-2, 2)

plt.tight_layout()
save_pdf('bounded_vs_unbounded.pdf')

# Figure 3: Operator Norm Convergence
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

n_vals = np.arange(1, 51)

# Case 1: Uniform convergence (fastest)
uniform_conv = 1 / (n_vals + 1)
axes[0].semilogy(n_vals, uniform_conv, 'o-', color=color_map, linewidth=2, markersize=4, label='Uniform: ‖Tₙ - T‖_B → 0')
axes[0].axhline(y=0.01, color=color_bound, linestyle='--', alpha=0.5, label='Convergence threshold')
axes[0].set_xlabel('n (iteration)', fontsize=11)
axes[0].set_ylabel('‖Tₙ - T‖_B', fontsize=11)
axes[0].set_title('Uniform Convergence (Fastest)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, which='both')

# Case 2: Strong convergence (medium)
strong_conv = 1 / np.sqrt(n_vals + 1)
axes[1].semilogy(n_vals, strong_conv, 's-', color=color_y, linewidth=2, markersize=4, label='Strong: ‖Tₙx - Tx‖ → 0')
axes[1].axhline(y=0.01, color=color_bound, linestyle='--', alpha=0.5, label='Convergence threshold')
axes[1].set_xlabel('n (iteration)', fontsize=11)
axes[1].set_ylabel('‖Tₙx - Tx‖', fontsize=11)
axes[1].set_title('Strong Convergence (Medium)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, which='both')

# Case 3: Weak convergence (slowest)
weak_conv = 1 / (n_vals**0.5)
axes[2].semilogy(n_vals, weak_conv, '^-', color=color_x, linewidth=2, markersize=4, label='Weak: f(Tₙx - Tx) → 0')
axes[2].axhline(y=0.01, color=color_bound, linestyle='--', alpha=0.5, label='Convergence threshold')
axes[2].set_xlabel('n (iteration)', fontsize=11)
axes[2].set_ylabel('|f(Tₙx - Tx)|', fontsize=11)
axes[2].set_title('Weak Convergence (Slowest)', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
save_pdf('convergence_types.pdf')

# Figure 4: Operator Norm Definition
fig, ax = plt.subplots(figsize=(10, 7))

# Hide axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Operator Norm Definition', fontsize=16, fontweight='bold', ha='center')

# Box 1: Definition
box1 = Rectangle((0.2, 7), 9.6, 2, linewidth=2, edgecolor=color_map, facecolor='lightyellow', alpha=0.7)
ax.add_patch(box1)
ax.text(5, 8.5, '‖A‖_B = inf {M : ‖A(x)‖ ≤ M‖x‖, ∀x ∈ X}', fontsize=13, ha='center', fontweight='bold', family='monospace')
ax.text(5, 7.8, 'Norm of bounded linear operator A: X → Y', fontsize=11, ha='center', style='italic')

# Equivalent forms
y_pos = 6.5
ax.text(0.5, y_pos, 'Equivalent forms:', fontsize=12, fontweight='bold')
y_pos -= 0.6
ax.text(1, y_pos, '• ‖A‖_B = sup {‖A(x)‖ : ‖x‖ = 1}', fontsize=11, family='monospace')
y_pos -= 0.6
ax.text(1, y_pos, '• ‖A‖_B = sup {‖A(x)‖/‖x‖ : x ≠ 0}', fontsize=11, family='monospace')
y_pos -= 0.6
ax.text(1, y_pos, '• ‖A‖_B = sup {‖A(x)‖ : ‖x‖ ≤ 1}', fontsize=11, family='monospace')

# Properties box
y_pos -= 1
box2 = Rectangle((0.2, y_pos-1.8), 9.6, 1.8, linewidth=1.5, edgecolor=color_bound, facecolor='lightcyan', alpha=0.5)
ax.add_patch(box2)
ax.text(0.5, y_pos, 'Properties of operator norm:', fontsize=12, fontweight='bold')
y_pos -= 0.5
ax.text(1, y_pos, '1. ‖A(x)‖ ≤ ‖A‖_B · ‖x‖  for all x ∈ X', fontsize=10, family='monospace')
y_pos -= 0.4
ax.text(1, y_pos, '2. ‖A + B‖_B ≤ ‖A‖_B + ‖B‖_B  (triangle inequality)', fontsize=10, family='monospace')
y_pos -= 0.4
ax.text(1, y_pos, '3. ‖λA‖_B = |λ| · ‖A‖_B  for scalar λ', fontsize=10, family='monospace')
y_pos -= 0.4
ax.text(1, y_pos, '4. ‖A ∘ B‖_B ≤ ‖A‖_B · ‖B‖_B  (composition)', fontsize=10, family='monospace')

plt.tight_layout()
save_pdf('operator_norm_definition.pdf')

# Figure 5: Spaces of Operators - Hierarchical Structure
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.7, 'Hierarchy of Operator Spaces', fontsize=16, fontweight='bold', ha='center')

# Level 1: All operators
rect1 = Rectangle((1.5, 8), 7, 0.8, linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.5)
ax.add_patch(rect1)
ax.text(5, 8.4, 'All Linear Operators A: X → Y', fontsize=11, ha='center', fontweight='bold')

# Arrow
ax.arrow(5, 7.95, 0, -0.6, head_width=0.2, head_length=0.15, fc='black', ec='black')
ax.text(5.5, 7.6, 'Continuity', fontsize=9, style='italic')

# Level 2: Bounded/continuous
rect2 = Rectangle((1, 6.5), 8, 0.8, linewidth=2, edgecolor=color_map, facecolor='lightgreen', alpha=0.5)
ax.add_patch(rect2)
ax.text(5, 6.9, 'Bounded Linear Operators B(X, Y)', fontsize=11, ha='center', fontweight='bold')
ax.text(5, 6.55, '(continuous ⟺ bounded for linear operators)', fontsize=9, ha='center', style='italic')

# Arrow
ax.arrow(5, 6.45, 0, -0.6, head_width=0.2, head_length=0.15, fc='black', ec='black')
ax.text(5.5, 6.1, 'Completeness of Y', fontsize=9, style='italic')

# Level 3: Banach space
rect3 = Rectangle((1.5, 5), 7, 0.8, linewidth=2, edgecolor=color_bound, facecolor='#ffe6e6', alpha=0.7)
ax.add_patch(rect3)
ax.text(5, 5.4, 'B(X, Y) is a Banach Space', fontsize=11, ha='center', fontweight='bold')
ax.text(5, 5.05, 'if Y is a Banach space', fontsize=9, ha='center', style='italic')

# Properties box
y_pos = 4.4
ax.text(0.5, y_pos, 'Key Properties of B(X, Y):', fontsize=12, fontweight='bold')
y_pos -= 0.5
properties = [
    '• Norm: ‖A‖_B(X,Y) = sup {‖A(x)‖_Y : ‖x‖_X ≤ 1}',
    '• Linear space under operator addition and scalar multiplication',
    '• Normed space with complete metric if Y is Banach',
    '• Contains identity operator I and zero operator 0',
    '• Closed under composition (if codomain = domain)'
]

for prop in properties:
    ax.text(1, y_pos, prop, fontsize=9.5, family='monospace', va='top')
    y_pos -= 0.4

# Examples box
y_pos -= 0.3
box_ex = Rectangle((0.2, 0.3), 9.6, y_pos, linewidth=1, edgecolor='orange', facecolor='lightyellow', alpha=0.3)
ax.add_patch(box_ex)
ax.text(0.5, y_pos-0.1, 'Examples:', fontsize=11, fontweight='bold')
y_pos -= 0.35
examples = [
    '• B(ℝⁿ, ℝᵐ): n×m matrices with operator norm = largest singular value',
    '• B(C[a,b], C[a,b]): multiplication by continuous function bounded by max|f|',
    '• B(L²(Ω), L²(Ω)): integral operators with finite operator norm'
]
for ex in examples:
    ax.text(1, y_pos, ex, fontsize=9, family='monospace', va='top')
    y_pos -= 0.35

plt.tight_layout()
save_pdf('operator_spaces_hierarchy.pdf')

# Figure 6: Continuity and Boundedness Connection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Schematic of continuous operator
ax1.set_xlim(-1, 6)
ax1.set_ylim(-1, 6)
ax1.axis('off')

ax1.text(2.5, 5.5, 'Continuous Linear Operator', fontsize=13, fontweight='bold', ha='center')

# Domain circle
circle_X = Circle((1, 2.5), 0.8, fill=False, edgecolor=color_x, linewidth=2.5)
ax1.add_patch(circle_X)
ax1.text(1, 2.5, 'X', fontsize=12, ha='center', va='center', fontweight='bold')

# Codomain circle
circle_Y = Circle((4, 2.5), 0.8, fill=False, edgecolor=color_y, linewidth=2.5)
ax1.add_patch(circle_Y)
ax1.text(4, 2.5, 'Y', fontsize=12, ha='center', va='center', fontweight='bold')

# Operator arrow
arrow = FancyArrowPatch((1.7, 2.5), (3.3, 2.5), arrowstyle='->', mutation_scale=30, linewidth=2.5, color=color_map)
ax1.add_patch(arrow)
ax1.text(2.5, 3, 'A: continuous', fontsize=11, ha='center', style='italic', color=color_map)

# Properties box
rect = Rectangle((0.2, 0.2), 4.6, 1.5, linewidth=1.5, edgecolor=color_map, facecolor='lightgreen', alpha=0.3)
ax1.add_patch(rect)
ax1.text(2.5, 1.5, 'Continuous ⟹ Bounded', fontsize=11, ha='center', fontweight='bold')
ax1.text(2.5, 1.1, '∃M > 0: ‖A(x)‖ ≤ M‖x‖', fontsize=10, ha='center', family='monospace')
ax1.text(2.5, 0.6, 'xₙ → x ⟹ A(xₙ) → A(x)', fontsize=10, ha='center', family='monospace')

# Right: Schematic of bounded operator
ax2.set_xlim(-1, 6)
ax2.set_ylim(-1, 6)
ax2.axis('off')

ax2.text(2.5, 5.5, 'Bounded Linear Operator', fontsize=13, fontweight='bold', ha='center')

# Domain with ball
circle_X2 = Circle((1, 2.5), 0.8, fill=False, edgecolor=color_x, linewidth=2.5)
ax2.add_patch(circle_X2)
ball_X = Circle((1, 2.5), 0.5, fill=True, facecolor=color_x, alpha=0.2)
ax2.add_patch(ball_X)
ax2.text(1, 2.5, 'B(x,r)', fontsize=10, ha='center', va='center', fontweight='bold')

# Codomain with mapped ball
circle_Y2 = Circle((4, 2.5), 0.8, fill=False, edgecolor=color_y, linewidth=2.5)
ax2.add_patch(circle_Y2)
ball_Y = Circle((4, 2.5), 0.3, fill=True, facecolor=color_y, alpha=0.2)
ax2.add_patch(ball_Y)
ax2.text(4, 2.5, 'A(B)', fontsize=10, ha='center', va='center', fontweight='bold')

# Operator arrow
arrow2 = FancyArrowPatch((1.7, 2.5), (3.3, 2.5), arrowstyle='->', mutation_scale=30, linewidth=2.5, color=color_map)
ax2.add_patch(arrow2)
ax2.text(2.5, 3, 'A: bounded', fontsize=11, ha='center', style='italic', color=color_map)

# Properties box
rect2 = Rectangle((0.2, 0.2), 4.6, 1.5, linewidth=1.5, edgecolor=color_map, facecolor='lightgreen', alpha=0.3)
ax2.add_patch(rect2)
ax2.text(2.5, 1.5, 'Bounded ⟹ Continuous', fontsize=11, ha='center', fontweight='bold')
ax2.text(2.5, 1.1, 'Bounded set → bounded set', fontsize=10, ha='center', family='monospace')
ax2.text(2.5, 0.6, 'For linear: at 0 ⟹ everywhere', fontsize=10, ha='center', family='monospace')

plt.tight_layout()
save_pdf('continuity_boundedness_connection.pdf')

print("✓ All figures generated successfully!")
print("  - linear_vs_nonlinear.pdf")
print("  - bounded_vs_unbounded.pdf")
print("  - convergence_types.pdf")
print("  - operator_norm_definition.pdf")
print("  - operator_spaces_hierarchy.pdf")
print("  - continuity_boundedness_connection.pdf")
