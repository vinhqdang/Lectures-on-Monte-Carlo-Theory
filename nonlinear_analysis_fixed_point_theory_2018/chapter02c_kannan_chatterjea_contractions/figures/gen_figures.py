#!/usr/bin/env python3
"""
Generate figures for Chapter 2c: Kannan & Chatterjea Contractions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'banach': '#1f77b4',
    'kannan': '#ff7f0e',
    'chatterjea': '#2ca02c',
    'reich': '#d62728',
    'ciric': '#9467bd',
    'background': '#f0f0f0'
}

# Figure 1: Kannan's Discontinuous Contraction Example (Example 5.18)
fig, ax = plt.subplots(figsize=(10, 8))
x = np.array([0, 2, 2, 4], dtype=float)
y = np.array([0, 1/3, 1, 1], dtype=float)

ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color=colors['kannan'], label='T(x)')
ax.plot([0, 4], [0, 4], 'k--', linewidth=1.5, alpha=0.5, label='y = x (identity)')

# Mark the discontinuity
ax.plot([2], [1/3], 'o', markersize=12, markerfacecolor='white', markeredgewidth=2,
        markeredgecolor=colors['kannan'])
ax.plot([2], [1], 'o', markersize=10, color=colors['kannan'])

# Mark fixed point
ax.plot([1], [1], 'r*', markersize=25, label='Fixed point u*', zorder=5)

# Add annotations
ax.annotate('Discontinuity at x=2', xy=(2, 1/3), xytext=(2.5, 0.3),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.annotate('T is not continuous\nbut satisfies Kannan\'s condition',
            xy=(3, 0.8), fontsize=10, bbox=dict(boxstyle='round,pad=0.4',
            facecolor='lightblue', alpha=0.8))

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-0.2, 4.3)
ax.set_xlabel('x', fontsize=13, fontweight='bold')
ax.set_ylabel('T(x)', fontsize=13, fontweight='bold')
ax.set_title("Example 5.18: Kannan's Contraction (Discontinuous)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('figures/kannan_discontinuous_example.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Comparison of Contractive Conditions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Comparison of Contraction Conditions', fontsize=15, fontweight='bold')

# Banach Contraction
ax = axes[0, 0]
ax.text(0.5, 0.7, r'$d(Tx, Ty) \leq k \cdot d(x, y)$', fontsize=13, ha='center',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=colors['banach'], alpha=0.3))
ax.text(0.5, 0.5, 'Contraction constant k < 1', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.3, 'T is continuous', fontsize=10, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.5, 0.1, r'(Uniform convergence)', fontsize=9, ha='center',
        transform=ax.transAxes, color='gray')
ax.set_title('Banach (BC)', fontsize=12, fontweight='bold', color=colors['banach'])
ax.axis('off')

# Kannan Contraction
ax = axes[0, 1]
ax.text(0.5, 0.7, r'$d(Tx, Ty) \leq r[d(Tx, x) + d(Ty, y)]$', fontsize=12, ha='center',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=colors['kannan'], alpha=0.3))
ax.text(0.5, 0.5, r'Coefficient $r \in [0, \frac{1}{2})$', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.3, 'T is NOT necessarily continuous', fontsize=10, ha='center',
        transform=ax.transAxes, style='italic', color='red')
ax.text(0.5, 0.1, r'(Local convergence)', fontsize=9, ha='center',
        transform=ax.transAxes, color='gray')
ax.set_title('Kannan (KC)', fontsize=12, fontweight='bold', color=colors['kannan'])
ax.axis('off')

# Chatterjea Contraction
ax = axes[1, 0]
ax.text(0.5, 0.7, r'$d(Tx, Ty) \leq r[d(Tx, y) + d(Ty, x)]$', fontsize=12, ha='center',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=colors['chatterjea'], alpha=0.3))
ax.text(0.5, 0.5, r'Coefficient $r \in [0, \frac{1}{2})$', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.3, 'Symmetric form of Kannan', fontsize=10, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.5, 0.1, r'(Independent from BC)', fontsize=9, ha='center',
        transform=ax.transAxes, color='gray')
ax.set_title('Chatterjea (CHC)', fontsize=12, fontweight='bold', color=colors['chatterjea'])
ax.axis('off')

# Rhoades Independence
ax = axes[1, 1]
ax.text(0.5, 0.85, 'Rhoades (1977)', fontsize=12, ha='center',
        transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.65, 'BC, KC, and CHC are\nINDEPENDENT conditions', fontsize=11, ha='center',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        style='italic')
ax.text(0.5, 0.35, 'No one implies another', fontsize=10, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.15, 'But all guarantee unique fixed points\nin complete metric spaces', fontsize=9,
        ha='center', transform=ax.transAxes, color='green', fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('figures/contraction_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Independence Venn-like diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Draw three circles to represent independence
from matplotlib.patches import Circle
circles = [
    Circle((1, 1.5), 0.8, color=colors['banach'], alpha=0.3, label='Banach'),
    Circle((2.5, 1.5), 0.8, color=colors['kannan'], alpha=0.3, label='Kannan'),
    Circle((3.5, 0.5), 0.8, color=colors['chatterjea'], alpha=0.3, label='Chatterjea'),
]

for circle in circles:
    ax.add_patch(circle)

# Add labels
ax.text(1, 1.5, 'BC', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(2.5, 1.5, 'KC', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(3.5, 0.5, 'CHC', fontsize=14, fontweight='bold', ha='center', va='center')

# Add title and explanation
ax.text(2.5, 2.5, 'Independence of Contraction Types (Rhoades, 1977)',
        fontsize=13, fontweight='bold', ha='center')
ax.text(2.5, 2.1, 'No overlaps - each type admits mappings not in the others',
        fontsize=11, ha='center', style='italic', color='red')

ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 3)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig('figures/independence_diagram.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Iterative convergence comparison
fig, ax = plt.subplots(figsize=(11, 7))

# Simulate convergence for different contractions
n_iterations = 30
x0 = 0.1

# Banach: d(T^n x0, u) ≤ k^n d(x0, u)
k_banach = 0.7
convergence_banach = np.array([k_banach**n for n in range(n_iterations)])

# Kannan: More complex convergence
r_kannan = 0.4
# Approximated convergence for Kannan
convergence_kannan = np.array([0.9**n * 0.8**n for n in range(n_iterations)])

# Chatterjea: Similar to Kannan
r_chatterjea = 0.4
convergence_chatterjea = np.array([0.85**n * 0.8**n for n in range(n_iterations)])

# Plot
ax.semilogy(range(n_iterations), convergence_banach, 'o-', linewidth=2.5,
            markersize=6, color=colors['banach'], label=f'Banach (k={k_banach})')
ax.semilogy(range(n_iterations), convergence_kannan, 's-', linewidth=2.5,
            markersize=6, color=colors['kannan'], label=f'Kannan (r={r_kannan})')
ax.semilogy(range(n_iterations), convergence_chatterjea, '^-', linewidth=2.5,
            markersize=6, color=colors['chatterjea'], label=f'Chatterjea (r={r_chatterjea})')

ax.set_xlabel('Iteration n', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Error $d(T^n x_0, u^*)$ (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Convergence Behavior: Kannan vs Banach vs Chatterjea', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(1e-8, 1)
plt.tight_layout()
plt.savefig('figures/convergence_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Region where contractions hold
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Parameter Regions for Contraction Conditions', fontsize=14, fontweight='bold')

# Banach: k ∈ [0, 1)
ax = axes[0]
k_vals = np.linspace(0, 1, 100)
ax.fill_between(k_vals[k_vals < 1], 0, 1, alpha=0.3, color=colors['banach'])
ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Boundary')
ax.plot(k_vals[k_vals < 1], np.ones_like(k_vals[k_vals < 1]), 'o-',
        color=colors['banach'], markersize=4, linewidth=2)
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.1, 1.2)
ax.set_xlabel('k (contraction constant)', fontsize=11, fontweight='bold')
ax.set_title('Banach: k < 1', fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis='x')

# Kannan: r ∈ [0, 1/2)
ax = axes[1]
r_vals = np.linspace(0, 0.6, 100)
ax.fill_between(r_vals[r_vals < 0.5], 0, 1, alpha=0.3, color=colors['kannan'])
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Boundary')
ax.plot(r_vals[r_vals < 0.5], np.ones_like(r_vals[r_vals < 0.5]), 'o-',
        color=colors['kannan'], markersize=4, linewidth=2)
ax.set_xlim(-0.05, 0.6)
ax.set_ylim(-0.1, 1.2)
ax.set_xlabel(r'r (Kannan coefficient)', fontsize=11, fontweight='bold')
ax.set_title(r'Kannan: r < 1/2', fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis='x')

# Chatterjea: r ∈ [0, 1/2)
ax = axes[2]
r_vals = np.linspace(0, 0.6, 100)
ax.fill_between(r_vals[r_vals < 0.5], 0, 1, alpha=0.3, color=colors['chatterjea'])
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Boundary')
ax.plot(r_vals[r_vals < 0.5], np.ones_like(r_vals[r_vals < 0.5]), 'o-',
        color=colors['chatterjea'], markersize=4, linewidth=2)
ax.set_xlim(-0.05, 0.6)
ax.set_ylim(-0.1, 1.2)
ax.set_xlabel(r'r (Chatterjea coefficient)', fontsize=11, fontweight='bold')
ax.set_title(r'Chatterjea: r < 1/2', fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig('figures/parameter_regions.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Relationship to other generalized contractions
fig, ax = plt.subplots(figsize=(10, 8))

conditions = [
    'Banach\n(BC)',
    'Kannan\n(KC)',
    'Chatterjea\n(CHC)',
    'Meir-Keeler\n(MK)',
    'Reich\n(RC)',
    'Ciric\n(CRC)',
    'Zamfirescu\n(ZC)',
    'Hardy-Rogers\n(HRC)'
]

y_positions = [7, 6.5, 6, 5, 4, 3, 2, 1]
colors_list = [colors['banach'], colors['kannan'], colors['chatterjea'],
               colors['reich'], colors['reich'], colors['ciric'],
               colors['ciric'], colors['ciric']]

for i, (cond, y_pos, col) in enumerate(zip(conditions, y_positions, colors_list)):
    rect = FancyBboxPatch((0.1, y_pos-0.3), 3, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor=col, facecolor=col, alpha=0.3, linewidth=2)
    ax.add_patch(rect)
    ax.text(1.6, y_pos, cond, fontsize=11, ha='center', va='center', fontweight='bold')

# Add descriptions
descriptions = [
    'd(Tx,Ty) ≤ k·d(x,y)',
    'd(Tx,Ty) ≤ r[d(Tx,x)+d(Ty,y)]',
    'd(Tx,Ty) ≤ r[d(Tx,y)+d(Ty,x)]',
    'ε-δ localized Banach',
    'd(Tx,Ty) ≤ ad(x,y) + bd(Tx,x) + cd(Ty,y)',
    'd(Tx,Ty) ≤ a₁d(Tx,x) + a₂d(Ty,y) + a₃d(x,y) + ...',
    'max of distance terms',
    'Combination of multiple terms'
]

for i, (desc, y_pos) in enumerate(zip(descriptions, y_positions)):
    ax.text(3.3, y_pos, desc, fontsize=9, ha='left', va='center', style='italic')

ax.set_xlim(-0.2, 6.5)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Generalized Contraction Mapping Conditions', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/generalized_contractions.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Generated files:")
print("  - kannan_discontinuous_example.pdf")
print("  - contraction_comparison.pdf")
print("  - independence_diagram.pdf")
print("  - convergence_comparison.pdf")
print("  - parameter_regions.pdf")
print("  - generalized_contractions.pdf")
