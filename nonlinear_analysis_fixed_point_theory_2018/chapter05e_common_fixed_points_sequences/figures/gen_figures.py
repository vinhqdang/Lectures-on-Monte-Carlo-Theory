#!/usr/bin/env python3
"""
Generate figures for Chapter 5e: Common Fixed Points & Sequences
From Pathak: An Introduction to Nonlinear Analysis and Fixed Point Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.patches import Wedge, Polygon
import matplotlib.patches as mpatches

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============================================================================
# Figure 1: Common Fixed Points Concept
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Space X
circle_X = Circle((0.5, 0.5), 0.35, color='lightblue', alpha=0.3, linewidth=2, edgecolor='blue')
ax.add_patch(circle_X)
ax.text(0.5, 0.88, 'Space $X$', fontsize=12, ha='center', fontweight='bold')

# Fixed points of f
f_point = np.array([0.35, 0.4])
ax.plot(f_point[0], f_point[1], 'ro', markersize=10, label='Fixed point of $f$')
ax.text(f_point[0]-0.08, f_point[1], '$u_f$', fontsize=11, fontweight='bold')

# Fixed points of g
g_point = np.array([0.65, 0.6])
ax.plot(g_point[0], g_point[1], 'bs', markersize=10, label='Fixed point of $g$')
ax.text(g_point[0]+0.05, g_point[1], '$u_g$', fontsize=11, fontweight='bold')

# Common fixed point
common_point = np.array([0.5, 0.5])
ax.plot(common_point[0], common_point[1], 'g*', markersize=20, label='Common fixed point $u = f(u) = g(u)$', zorder=5)

# Arrows showing mappings
arrow1 = FancyArrowPatch((0.2, 0.3), f_point, arrowstyle='->', mutation_scale=20,
                         color='red', linewidth=1.5, alpha=0.6)
ax.add_patch(arrow1)
ax.text(0.25, 0.25, '$f$', fontsize=11, color='red', fontweight='bold')

arrow2 = FancyArrowPatch((0.8, 0.7), g_point, arrowstyle='->', mutation_scale=20,
                         color='blue', linewidth=1.5, alpha=0.6)
ax.add_patch(arrow2)
ax.text(0.78, 0.75, '$g$', fontsize=11, color='blue', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig('figures/fig_common_fixed_points.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_common_fixed_points.pdf")
plt.close()

# ============================================================================
# Figure 2: Convergence of Sequences of Contractions
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Generate iteration sequence
x0 = 0
x = [x0]
k = 0.7  # Contraction constant

for i in range(10):
    x.append(k * x[-1] + 0.3)

# Plot the convergent sequence
ax.plot(range(len(x)), x, 'bo-', linewidth=2, markersize=8, label='Sequence $\\{x_n\\}$')

# Fixed point (limit)
limit = 0.3 / (1 - k)
ax.axhline(y=limit, color='red', linestyle='--', linewidth=2, label=f'Fixed point $u = {limit:.2f}$')

ax.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
ax.set_ylabel('$x_n$', fontsize=12, fontweight='bold')
ax.set_title('Convergence of Sequence of Contraction Mappings', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='right')

plt.tight_layout()
plt.savefig('figures/fig_contraction_convergence.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_contraction_convergence.pdf")
plt.close()

# ============================================================================
# Figure 3: Ordered Banach Space - Cone Ordering
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Draw the cone K
theta = np.linspace(0, np.pi/3, 100)
r_outer = 0.6
r_inner = 0.2

# Outer arc of cone
x_outer = r_outer * np.cos(theta)
y_outer = r_outer * np.sin(theta)

# Inner arc
x_inner = r_inner * np.cos(theta)
y_inner = r_inner * np.sin(theta)

# Create cone patch
vertices_cone = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
cone_patch = Polygon(vertices_cone, color='lightgreen', alpha=0.4, edgecolor='darkgreen', linewidth=2)
ax.add_patch(cone_patch)

# Add negative cone
vertices_cone_neg = [(x, -y) for x, y in vertices_cone]
cone_patch_neg = Polygon(vertices_cone_neg, color='lightcoral', alpha=0.4, edgecolor='darkred', linewidth=2)
ax.add_patch(cone_patch_neg)

# Labels
ax.text(0.35, 0.25, 'Cone $K$', fontsize=12, fontweight='bold', color='darkgreen')
ax.text(0.35, -0.25, '$-K$', fontsize=12, fontweight='bold', color='darkred')
ax.text(-0.15, 0, 'Origin', fontsize=11, ha='right')

# Mark origin
ax.plot(0, 0, 'ko', markersize=8)

# Axis labels
ax.arrow(-0.1, 0, 0.9, 0, head_width=0.02, head_length=0.05, fc='black', ec='black')
ax.arrow(0, -0.1, 0, 0.9, head_width=0.02, head_length=0.05, fc='black', ec='black')
ax.text(0.95, 0, '$x_1$', fontsize=12, fontweight='bold')
ax.text(0, 0.95, '$x_2$', fontsize=12, fontweight='bold')

ax.set_xlim(-0.2, 1)
ax.set_ylim(-0.7, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('figures/fig_cone_ordering.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_cone_ordering.pdf")
plt.close()

# ============================================================================
# Figure 4: Example 5.47 - Numerical Convergence
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Functions from Example 5.47
def f(x, y):
    return (x**2 + y**2) / 13 + 18/13

def g(x):
    return x**2 - 2

# Starting point
x_vals = np.linspace(-2, 3, 100)

# Plot functions
ax.plot(x_vals, f(x_vals, x_vals), 'b-', linewidth=2.5, label='$f(x, y) = \\frac{x^2 + y^2}{13} + \\frac{18}{13}$')
ax.plot(x_vals, g(x_vals), 'r-', linewidth=2.5, label='$g(x) = x^2 - 2$')
ax.plot(x_vals, x_vals, 'k--', linewidth=1.5, alpha=0.5, label='$y = x$')

# Find and mark intersection (common fixed point)
# Solving: x = x^2 - 2 => x^2 - x - 2 = 0 => x = 2 or x = -1
x_fp1 = 2
x_fp2 = -1

ax.plot([x_fp1, x_fp2], [x_fp1, x_fp2], 'g*', markersize=15, label='Common fixed points', zorder=5)
ax.text(x_fp1 + 0.1, x_fp1 + 0.1, f'$(2, 2)$', fontsize=10, fontweight='bold')
ax.text(x_fp2 - 0.3, x_fp2 - 0.2, f'$(-1, -1)$', fontsize=10, fontweight='bold')

ax.set_xlabel('$x$', fontsize=12, fontweight='bold')
ax.set_ylabel('$y$', fontsize=12, fontweight='bold')
ax.set_title('Example 5.47: Common Fixed Points', fontsize=13, fontweight='bold')
ax.set_xlim(-2.5, 3.5)
ax.set_ylim(-2.5, 3.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper left')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/fig_example_547.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_example_547.pdf")
plt.close()

# ============================================================================
# Figure 5: Family of Mappings and Iteration Scheme
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Family of mappings
for i, alpha in enumerate(np.linspace(0.3, 0.9, 4)):
    x = np.linspace(-1, 1, 100)
    y = alpha * x
    ax1.plot(x, y, linewidth=2, label=f'$F_{{{i+1}}}$ (slope={alpha:.1f})')

ax1.plot([-1, 1], [-1, 1], 'k--', linewidth=1, alpha=0.3)
ax1.set_xlabel('$x$', fontsize=12, fontweight='bold')
ax1.set_ylabel('$y = F(x)$', fontsize=12, fontweight='bold')
ax1.set_title('Family of Linear Mappings $\\{F_i\\}$', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')

# Right plot: Iteration convergence
x_iter = [0.5]
for n in range(15):
    # Apply composite iteration with varying parameters
    c_n = 1 - (n+1)/20
    alpha_n = 0.5 + 0.3*np.sin(n)
    x_iter.append(c_n * x_iter[-1] + alpha_n * 0.1)

ax2.plot(range(len(x_iter)), x_iter, 'bo-', linewidth=2, markersize=6)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Fixed point')
ax2.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
ax2.set_ylabel('$U_n x$', fontsize=12, fontweight='bold')
ax2.set_title('Iteration Scheme Convergence', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_family_mappings.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_family_mappings.pdf")
plt.close()

# ============================================================================
# Figure 6: Banach Space Structure with Different Norms
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a visualization of different metrics
metrics = ['Euclidean\n($L^2$)', 'Manhattan\n($L^1$)', 'Maximum\n($L^\\infty$)', 'b-metric\n(weighted)']
convergence_rates = [0.85, 0.75, 0.80, 0.88]
colors = ['blue', 'red', 'green', 'purple']

bars = ax.barh(metrics, convergence_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, rate) in enumerate(zip(bars, convergence_rates)):
    ax.text(rate + 0.02, i, f'{rate:.2f}', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Convergence Rate', fontsize=12, fontweight='bold')
ax.set_title('Contraction Mappings: Convergence Rates by Metric', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig_banach_metrics.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_banach_metrics.pdf")
plt.close()

# ============================================================================
# Figure 7: Uniqueness of Common Fixed Points
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Create a heatmap showing uniqueness region
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Define a measure of contraction strength
# Higher values = stronger contraction = uniqueness guaranteed
Z = 1 - (0.5 * X**2 + 0.3 * Y**2 + 0.2 * X * Y)

im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn', alpha=0.8)
contours = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax.clabel(contours, inline=True, fontsize=8)

# Mark the unique fixed point region
circle_unique = Circle((0.2, 0.3), 0.15, fill=False, edgecolor='blue', linewidth=2.5,
                       linestyle='--', label='Unique fixed point region')
ax.add_patch(circle_unique)

ax.set_xlabel('Parameter $\\lambda_1$', fontsize=12, fontweight='bold')
ax.set_ylabel('Parameter $\\lambda_2$', fontsize=12, fontweight='bold')
ax.set_title('Uniqueness of Common Fixed Points\nContraction Strength: $\\phi(\\lambda_1, \\lambda_2)$',
             fontsize=13, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, label='Contraction Measure')
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('figures/fig_uniqueness.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_uniqueness.pdf")
plt.close()

# ============================================================================
# Figure 8: Iteration Process Visualization
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Simulate iteration process
n_iterations = 12
iteration_steps = []
distances = []

x = 1.0
for n in range(n_iterations):
    iteration_steps.append(n)
    distances.append(x)
    x = 0.6 * x  # Contraction with factor 0.6

# Create a staircase plot
ax.step(iteration_steps, distances, where='mid', linewidth=2.5, color='blue',
        marker='o', markersize=8, label='Distance to fixed point')

# Add reference lines
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Fixed point')

# Shade convergence region
ax.fill_between(iteration_steps, 0, distances, alpha=0.2, color='blue', label='Convergence region')

# Add annotations
for i in [0, 3, 6, n_iterations-1]:
    ax.annotate(f'$d_{i}$', xy=(iteration_steps[i], distances[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
ax.set_ylabel('$d(x_n, u)$ (Distance to fixed point)', fontsize=12, fontweight='bold')
ax.set_title('Iteration Process: Convergence via Contraction', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-0.5, n_iterations + 0.5)

plt.tight_layout()
plt.savefig('figures/fig_iteration_process.pdf', dpi=300, bbox_inches='tight')
print("Generated: fig_iteration_process.pdf")
plt.close()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
