#!/usr/bin/env python3
"""
Generate visualizations for Chapter 5: Fixed Point Theorems
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'nonexpansive': '#1f77b4',
    'asymptotic': '#ff7f0e',
    'schauder': '#2ca02c',
    'brouwer': '#d62728',
    'pseudo': '#9467bd',
}

# ============================================================================
# Figure 1: Hierarchy of Mapping Classes
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Hierarchy of Mapping Classes',
        fontsize=16, fontweight='bold', ha='center')

# Draw hierarchy boxes
boxes = [
    {'xy': (3, 7.5), 'label': 'Contractive\nMappings', 'color': '#e74c3c'},
    {'xy': (1, 5.5), 'label': 'Pseudocontractive', 'color': colors['pseudo']},
    {'xy': (3.5, 5.5), 'label': 'Nonexpansive', 'color': colors['nonexpansive']},
    {'xy': (6, 5.5), 'label': 'Weakly\nContractive', 'color': '#3498db'},
    {'xy': (3.5, 3.5), 'label': 'Asymptotically\nNonexpansive', 'color': colors['asymptotic']},
    {'xy': (1.5, 1.5), 'label': 'Quasi-\nnonexpansive', 'color': '#95a5a6'},
]

for box in boxes:
    fancy_box = FancyBboxPatch(box['xy'], 1.8, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=box['color'],
                               alpha=0.7, linewidth=2)
    ax.add_patch(fancy_box)
    ax.text(box['xy'][0] + 0.9, box['xy'][1] + 0.6, box['label'],
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Draw arrows showing hierarchy
arrow_pairs = [
    ((4, 7.5), (2.5, 6.7)),  # Contractive to Pseudocontractive
    ((4.3, 7.5), (4, 6.7)),   # Contractive to Nonexpansive
    ((4.6, 7.5), (6.8, 6.7)), # Contractive to Weakly Contractive
    ((4.1, 5.5), (4.3, 4.7)), # Nonexpansive to Asymptotically Nonexpansive
]

for start, end in arrow_pairs:
    arrow = FancyArrowPatch(start, end, arrowstyle='->',
                           mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)

# Add legend with fixed point results
legend_text = ("Key Results:\n"
              "• Contractive: Unique fixed point (Banach)\n"
              "• Nonexpansive: FPP in compact convex sets\n"
              "• Asymptotically Nonexpansive: Weak convergence\n"
              "• Pseudocontractive: Via accretive operators")
ax.text(0.5, 0.3, legend_text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='bottom', family='monospace')

plt.tight_layout()
plt.savefig('figures/mapping_hierarchy.pdf', bbox_inches='tight', dpi=300)
print("Saved: mapping_hierarchy.pdf")
plt.close()

# ============================================================================
# Figure 2: Fixed Point Properties - Conceptual Diagram
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Fixed Point Properties and Spaces',
        fontsize=16, fontweight='bold', ha='center')

# Left column: Space properties
left_boxes = [
    {'xy': (0.5, 7.5), 'label': 'Banach\nSpaces', 'color': '#3498db'},
    {'xy': (0.5, 5.5), 'label': 'Hilbert\nSpaces', 'color': '#2980b9'},
    {'xy': (0.5, 3.5), 'label': 'Locally\nConvex', 'color': '#1c3144'},
]

for box in left_boxes:
    fancy_box = FancyBboxPatch(box['xy'], 2, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=box['color'],
                               alpha=0.7, linewidth=2)
    ax.add_patch(fancy_box)
    ax.text(box['xy'][0] + 1, box['xy'][1] + 0.6, box['label'],
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Right column: Fixed point results
right_boxes = [
    {'xy': (4, 7.5), 'label': "Brouwer's FPT\n(Finite dim)", 'color': colors['brouwer']},
    {'xy': (7, 7.5), 'label': "Schauder's FPT\n(Compact convex)", 'color': colors['schauder']},
    {'xy': (4, 5.5), 'label': 'Schauder-\nTychonoff', 'color': '#16a085'},
    {'xy': (7, 5.5), 'label': 'Krasnoselski\nFPT', 'color': '#27ae60'},
    {'xy': (4, 3.5), 'label': 'Opial\'s Theorem\n(Asymptotic)', 'color': '#e67e22'},
    {'xy': (7, 3.5), 'label': 'Mann Iteration\nConvergence', 'color': '#e8daef'},
]

for box in right_boxes:
    fancy_box = FancyBboxPatch(box['xy'], 2.2, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=box['color'],
                               alpha=0.7, linewidth=2)
    ax.add_patch(fancy_box)
    ax.text(box['xy'][0] + 1.1, box['xy'][1] + 0.6, box['label'],
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Draw connecting arrows
for lbox in left_boxes:
    for rbox in right_boxes:
        if abs(lbox['xy'][1] - rbox['xy'][1]) < 0.3:  # Same vertical level
            start = (lbox['xy'][0] + 2, lbox['xy'][1] + 0.6)
            end = (rbox['xy'][0], rbox['xy'][1] + 0.6)
            arrow = FancyArrowPatch(start, end, arrowstyle='->',
                                   mutation_scale=15, linewidth=1.5,
                                   color='gray', alpha=0.6)
            ax.add_patch(arrow)

ax.text(5, 1, 'Relationships between spaces and their corresponding fixed point theorems',
        ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('figures/fpt_properties.pdf', bbox_inches='tight', dpi=300)
print("Saved: fpt_properties.pdf")
plt.close()

# ============================================================================
# Figure 3: Opial Condition Illustration
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Create a simple illustration of Opial's condition
t = np.linspace(0, 2*np.pi, 100)

# Fixed point at origin
ax.plot(0, 0, 'r*', markersize=20, label='Fixed point $x_0$')

# Weakly converging sequence
angles = np.linspace(0, 2*np.pi, 6)[:-1]
for i, angle in enumerate(angles):
    r = 2 - 0.3*i  # Decreasing radius
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    ax.plot(x, y, 'b.', markersize=12)
    ax.text(x+0.15, y+0.15, f'$x_{i}$', fontsize=10)

    # Draw spiral path
    if i < len(angles) - 1:
        next_angle = angles[i+1]
        next_r = 2 - 0.3*(i+1)
        next_x = next_r * np.cos(next_angle)
        next_y = next_r * np.sin(next_angle)
        ax.plot([x, next_x], [y, next_y], 'b--', alpha=0.5, linewidth=1)

# Draw a circle to show weak convergence region
circle = plt.Circle((0, 0), 2.5, color='green', fill=False,
                    linestyle='--', linewidth=2, alpha=0.5, label='Convergence region')
ax.add_patch(circle)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title("Opial's Condition:\n$\\liminf_{n\\to\\infty} \\|x_n - x\\| > \\liminf_{n\\to\\infty} \\|x_n - x_0\\|$ for $x \\neq x_0$",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('figures/opial_condition.pdf', bbox_inches='tight', dpi=300)
print("Saved: opial_condition.pdf")
plt.close()

# ============================================================================
# Figure 4: Asymptotic Regularity Concept
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Asymptotic regularity
n_values = np.arange(1, 21)
regular_values = 1.0 / n_values  # Asymptotically regular
nonregular_values = 0.3 * np.ones_like(n_values)  # Not asymptotically regular

ax1.plot(n_values, regular_values, 'g-o', linewidth=2.5, markersize=6, label='Asymptotically regular')
ax1.plot(n_values, nonregular_values, 'r--s', linewidth=2.5, markersize=6, label='Not asymptotically regular')
ax1.fill_between(n_values, 0, regular_values, alpha=0.2, color='green')
ax1.set_xlabel('Iteration $n$', fontsize=11)
ax1.set_ylabel('$\\|T^n x - T^{n+1} x\\|$', fontsize=11)
ax1.set_title('Asymptotic Regularity: $\\|T^n x - T^{n+1} x\\| \\to 0$ as $n \\to \\infty$',
             fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Right: Comparison of convergence rates
n_iter = np.arange(0, 31)
linear_conv = 0.9 ** n_iter  # Linear convergence
sublinear_conv = 1.0 / (n_iter + 1)  # Sublinear convergence
superlinear_conv = 0.5 ** (n_iter * 0.5)  # Superlinear convergence

ax2.semilogy(n_iter, linear_conv, 'b-o', linewidth=2.5, markersize=5, label='Linear (geometric)')
ax2.semilogy(n_iter, sublinear_conv, 'r-s', linewidth=2.5, markersize=5, label='Sublinear (1/n)')
ax2.semilogy(n_iter, superlinear_conv, 'g-^', linewidth=2.5, markersize=5, label='Superlinear')
ax2.set_xlabel('Iteration $n$', fontsize=11)
ax2.set_ylabel('Error $\\|x_n - x^*\\|$', fontsize=11)
ax2.set_title('Convergence Rates Comparison',
             fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/asymptotic_regularity.pdf', bbox_inches='tight', dpi=300)
print("Saved: asymptotic_regularity.pdf")
plt.close()

# ============================================================================
# Figure 5: Fixed Point Theorem Conditions Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 9))
ax.axis('off')

# Title
title_text = 'Fixed Point Theorems: Conditions and Conclusions'
ax.text(0.5, 0.95, title_text, ha='center', fontsize=16, fontweight='bold',
        transform=ax.transAxes)

# Create table data
theorems_data = [
    ['Theorem', 'Domain', 'Mapping Type', 'Key Condition', 'Conclusion'],
    ['Banach', 'Metric Space', 'Contraction', '$d(Tx,Ty) \\leq \\alpha d(x,y)$, $\\alpha < 1$', 'Unique fixed point'],
    ['Brouwer', '$\\mathbb{B}^n$', 'Continuous', 'Compact convex', 'Fixed point exists'],
    ['Schauder', 'Banach', 'Compact map', '$T(K) \\subset$ compact $\\subset K$', 'Fixed point exists'],
    ['Schauder-', 'Locally convex', 'Compact map', 'Compact convex set', 'Fixed point exists'],
    ['Tychonoff', 'topological', '', '', ''],
    ['Krasnoselski', 'Banach', '$T+S$ sum', '$T$ compact, $S$ contraction', 'Fixed point exists'],
    ['Goebel-Kirk', 'Banach', 'Asympt. nonexp.', 'Bounded convex set', 'Fixed point exists'],
]

# Create table
table = ax.table(cellText=theorems_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.15, 0.15, 0.28, 0.20])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(theorems_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')
        table[(i, j)].set_edgecolor('#95a5a6')

# Add footer
footer_text = ('Compact set property, convexity, and continuity are crucial ingredients.\n'
              'Different theorems apply to different geometric and analytical settings.')
ax.text(0.5, 0.02, footer_text, ha='center', fontsize=9, style='italic',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/theorem_conditions.pdf', bbox_inches='tight', dpi=300)
print("Saved: theorem_conditions.pdf")
plt.close()

# ============================================================================
# Figure 6: Convergence Behavior - Mann Iteration
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 7))

# Simulate Mann iteration convergence
np.random.seed(42)
x_fixed = 0.5

# Different lambda values for Mann iteration: x_{n+1} = (1-lambda)*x_n + lambda*T(x_n)
lambdas = [0.3, 0.5, 0.8, 1.0]  # 1.0 is standard Picard iteration

for lam in lambdas:
    x_vals = [0.0]  # Start at x_0 = 0
    for n in range(50):
        # Simulate T(x) = 0.7*x + 0.5
        T_x = 0.7 * x_vals[-1] + 0.5
        x_next = (1 - lam) * x_vals[-1] + lam * T_x
        x_vals.append(x_next)

    error = np.abs(np.array(x_vals) - x_fixed)
    ax.semilogy(range(len(x_vals)), error, 'o-', linewidth=2.5, markersize=5,
                label=f'$\\lambda = {lam}$', alpha=0.8)

ax.set_xlabel('Iteration $n$', fontsize=12)
ax.set_ylabel('Error $|x_n - x^*|$ (log scale)', fontsize=12)
ax.set_title('Mann Iteration: Effect of Step Size $\\lambda$\n$x_{n+1} = (1-\\lambda)x_n + \\lambda T(x_n)$',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/mann_iteration.pdf', bbox_inches='tight', dpi=300)
print("Saved: mann_iteration.pdf")
plt.close()

print("\nAll figures generated successfully!")
print("Output directory: figures/")
