#!/usr/bin/env python3
"""
Generate figures for Chapter 8e: Nonlinear Complementarity Problems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Figure 1: Cone and Dual Cone Illustration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Positive Orthant (Cone K)
theta = np.linspace(0, np.pi/2, 100)
x_cone = np.cos(theta)
y_cone = np.sin(theta)

ax1.fill_between(x_cone, y_cone, alpha=0.3, color='blue', label='Cone K')
ax1.plot(x_cone, y_cone, 'b-', linewidth=2)
ax1.arrow(0, 0, 1.2, 0, head_width=0.08, head_length=0.1, fc='black', ec='black')
ax1.arrow(0, 0, 0, 1.2, head_width=0.08, head_length=0.1, fc='black', ec='black')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Boundary of K')
ax1.set_xlim(-0.2, 1.4)
ax1.set_ylim(-0.2, 1.4)
ax1.set_aspect('equal')
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.set_title('Cone K (Positive Orthant)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Right: Dual Cone K*
ax2.fill_between(x_cone, y_cone, alpha=0.3, color='green', label='Dual Cone $K^*$')
ax2.plot(x_cone, y_cone, 'g-', linewidth=2)
ax2.arrow(0, 0, 1.2, 0, head_width=0.08, head_length=0.1, fc='black', ec='black')
ax2.arrow(0, 0, 0, 1.2, head_width=0.08, head_length=0.1, fc='black', ec='black')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Boundary of $K^*$')
ax2.set_xlim(-0.2, 1.4)
ax2.set_ylim(-0.2, 1.4)
ax2.set_aspect('equal')
ax2.set_xlabel('$f_1(x)$', fontsize=12)
ax2.set_ylabel('$f_2(x)$', fontsize=12)
ax2.set_title('Dual Cone $K^*$ (Positive Orthant)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/01_cone_dual_cone.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Complementarity Condition Visualization
fig, ax = plt.subplots(figsize=(10, 7))

# Create a visualization of the complementarity condition
# x in K, f(x) in K*, <x, f(x)> = 0
x_vals = np.linspace(0, 2, 100)

# Different f(x) scenarios
ax.plot(x_vals, np.exp(-x_vals), 'b-', linewidth=2.5, label='Example: $f(x) = e^{-x}$')
ax.fill_between(x_vals[x_vals <= 1], 0, np.exp(-x_vals[x_vals <= 1]),
                 alpha=0.2, color='blue', label='Region where $x \\in K$')

# Mark complementarity point
x_comp = 0.7
f_comp = np.exp(-x_comp)
ax.plot(x_comp, f_comp, 'ro', markersize=10, label=f'Complementarity point')
ax.axvline(x_comp, color='r', linestyle='--', alpha=0.5)
ax.axhline(f_comp, color='r', linestyle='--', alpha=0.5)

ax.set_xlim(-0.1, 2.2)
ax.set_ylim(-0.1, 1.5)
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$f(x)$', fontsize=13)
ax.set_title('Complementarity Condition: $\\langle x, f(x) \\rangle = 0$',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('figures/02_complementarity_condition.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Solution Existence - Convergence
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate iterative algorithm convergence
iterations = np.arange(0, 20)
error1 = 10 * np.exp(-0.3 * iterations)  # Exponential convergence
error2 = 10 * (0.7 ** iterations)  # Geometric convergence

ax.semilogy(iterations, error1, 'o-', linewidth=2.5, markersize=6,
            label='Strong Monotone Case', color='#2ca02c')
ax.semilogy(iterations, error2, 's-', linewidth=2.5, markersize=6,
            label='Lipschitz Case', color='#d62728')
ax.axhline(1e-6, color='gray', linestyle='--', alpha=0.5, label='Tolerance')

ax.set_xlabel('Iteration Number $n$', fontsize=12)
ax.set_ylabel('Error $\\|x_n - x^*\\|$ (log scale)', fontsize=12)
ax.set_title('Convergence to Solution of Complementarity Problem',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)
ax.set_xlim(-0.5, 19.5)

plt.tight_layout()
plt.savefig('figures/03_convergence.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Solution regions for different problem types
fig, ax = plt.subplots(figsize=(11, 8))

# Create a schematic showing different complementarity problem types
y_pos = [0, 1, 2, 3, 4]
problems = [
    'E.C.P: Explicit Complementarity\n$x_0 \\in K$: $f(x_0) \\in K^*$, $\\langle x_0, f(x_0) \\rangle = 0$',
    'I.C.P: Implicit Complementarity\n$x_0 \\in K$: $f(x_0) \\in K^*$, $g(x_0) \\in K$, $\\langle g(x_0), f(x_0) \\rangle = 0$',
    'S.E.C.P(f, g, K): Simultaneous Explicit\n$x_0 \\in K$: $f(x_0) \\in K^*$, $g(x_0) \\in K^*$',
    'S.I.C.P(f, g, K): Simultaneous Implicit\n$x_0 \\in K$: $f(x_0) \\in K^*$, $g(x_0) \\in K^*$',
    'Special Case: Single-valued complementarity'
]

colors_prob = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (y, prob, col) in enumerate(zip(y_pos, problems, colors_prob)):
    ax.barh(y, 1, height=0.6, color=col, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.text(0.5, y, prob, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.set_ylim(-0.5, 4.5)
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title('Complementarity Problem Types', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/04_problem_types.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Monotonicity Properties
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Strongly Monotone Mapping
x_vals = np.linspace(0, 2, 100)
# Strong monotone: fast growth in difference
f_smooth = 2 * x_vals ** 2
f_example = np.sin(x_vals) + 1.5 * x_vals

ax1.plot(x_vals, f_example, 'b-', linewidth=2.5, label='Strongly Monotone $f$')
ax1.fill_between(x_vals, 0, f_example, alpha=0.2, color='blue')

# Mark a pair of points showing strong monotonicity
x1, x2 = 0.5, 1.5
f_x1 = np.sin(x1) + 1.5 * x1
f_x2 = np.sin(x2) + 1.5 * x2

ax1.plot([x1, x2], [f_x1, f_x2], 'ro', markersize=8)
ax1.annotate('$(x_1, f(x_1))$', xy=(x1, f_x1), xytext=(x1-0.3, f_x1-0.5),
            fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='red'))
ax1.annotate('$(x_2, f(x_2))$', xy=(x2, f_x2), xytext=(x2+0.3, f_x2+0.3),
            fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='red'))

ax1.set_xlim(-0.2, 2.2)
ax1.set_ylim(-0.5, 4)
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$f(x)$', fontsize=12)
ax1.set_title('Strongly Monotone Mapping', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Right: n-Lipschitz Mapping
f_lipschitz = 1.2 * x_vals + 0.3

ax2.plot(x_vals, f_lipschitz, 'g-', linewidth=2.5, label='$n$-Lipschitz $f$')
ax2.fill_between(x_vals, 0, f_lipschitz, alpha=0.2, color='green')

# Show Lipschitz cone
x_mark = 1.0
f_mark = 1.2 * x_mark + 0.3
slope = 1.2

x_cone_left = np.linspace(max(0, x_mark - 0.5), x_mark, 50)
x_cone_right = np.linspace(x_mark, min(2, x_mark + 0.5), 50)

f_upper_left = f_mark - slope * (x_mark - x_cone_left)
f_upper_right = f_mark + slope * (x_cone_right - x_mark)

ax2.fill_between(x_cone_left, f_mark - slope * (x_mark - x_cone_left),
                 f_mark + slope * (x_mark - x_cone_left),
                 alpha=0.1, color='red', label='Lipschitz Cone')
ax2.fill_between(x_cone_right, f_mark - slope * (x_cone_right - x_mark),
                 f_mark + slope * (x_cone_right - x_mark),
                 alpha=0.1, color='red')

ax2.plot(x_mark, f_mark, 'ro', markersize=8)

ax2.set_xlim(-0.2, 2.2)
ax2.set_ylim(-0.5, 3.5)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$f(x)$', fontsize=12)
ax2.set_title('$n$-Lipschitz Mapping ($n$-Contraction)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('figures/05_monotonicity_properties.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Applications Framework
fig, ax = plt.subplots(figsize=(11, 8))

# Create a mind-map style diagram showing applications
applications = {
    'Complementarity\nProblems': {
        'x': 0.5, 'y': 0.5, 'color': '#d62728', 'size': 0.08
    },
    'Optimization': {
        'x': 0.15, 'y': 0.8, 'color': '#1f77b4', 'size': 0.06
    },
    'Engineering': {
        'x': 0.5, 'y': 0.85, 'color': '#2ca02c', 'size': 0.06
    },
    'Mechanics': {
        'x': 0.85, 'y': 0.8, 'color': '#ff7f0e', 'size': 0.06
    },
    'Variational\nProblems': {
        'x': 0.2, 'y': 0.2, 'color': '#9467bd', 'size': 0.06
    },
    'Economic\nEquilibrium': {
        'x': 0.8, 'y': 0.2, 'color': '#17becf', 'size': 0.06
    },
}

for app, props in applications.items():
    circle = Circle((props['x'], props['y']), props['size'],
                    color=props['color'], alpha=0.7, zorder=10)
    ax.add_patch(circle)
    ax.text(props['x'], props['y'], app, ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Draw connections to main node
    if app != 'Complementarity\nProblems':
        ax.plot([0.5, props['x']], [0.5, props['y']], 'k-', linewidth=1.5, alpha=0.3, zorder=1)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.05, 0.95)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Applications of Complementarity Problems', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/06_applications.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Files saved:")
print("  - figures/01_cone_dual_cone.pdf")
print("  - figures/02_complementarity_condition.pdf")
print("  - figures/03_convergence.pdf")
print("  - figures/04_problem_types.pdf")
print("  - figures/05_monotonicity_properties.pdf")
print("  - figures/06_applications.pdf")
