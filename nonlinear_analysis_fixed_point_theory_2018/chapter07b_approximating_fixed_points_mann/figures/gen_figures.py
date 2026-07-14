#!/usr/bin/env python3
"""
Generate figures for Chapter 9 Applications of Fixed Point Theorems
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Figure 1: Banach Space Contraction
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_aspect('equal')

# Draw a ball/domain
circle = Circle((5, 5), 3, fill=True, alpha=0.3, color=colors[0], edgecolor='black', linewidth=2)
ax.add_patch(circle)
ax.text(5, 5, r'$B_r \subset X$', fontsize=14, ha='center', va='center', weight='bold')

# Points
x_vals = np.array([3, 4.5, 6, 7.5])
y_vals = np.array([5, 5.8, 5, 4.2])
ax.scatter(x_vals, y_vals, s=200, c=colors[1], zorder=5, edgecolors='black', linewidth=2)

# Arrows showing contraction
for i in range(len(x_vals)-1):
    arrow = FancyArrowPatch((x_vals[i], y_vals[i]), (x_vals[i+1], y_vals[i+1]),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color=colors[2])
    ax.add_patch(arrow)

ax.text(2, 8.5, 'Banach Contraction Property', fontsize=16, weight='bold')
ax.text(2, 8, r'$\|Tx_1 - Tx_2\| \leq q\|x_1 - x_2\|, \quad q < 1$', fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('banach_contraction.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 2: Fixed Point Iteration Sequence
fig, ax = plt.subplots(figsize=(11, 6))

# Iteration counts
n_iter = 6
x_initial = 3
x_vals = np.array([x_initial])
for i in range(n_iter):
    x_vals = np.append(x_vals, x_vals[-1] * 0.6 + 2)

# Plot iteration sequence
x_positions = np.arange(len(x_vals))
ax.plot(x_positions, x_vals, 'o-', linewidth=2.5, markersize=10, color=colors[0], label='Iteration sequence')
ax.axhline(y=5, color=colors[2], linestyle='--', linewidth=2, label='Fixed point $x^* = 5$')

# Annotations
for i, (x, y) in enumerate(zip(x_positions, x_vals)):
    ax.text(x, y+0.3, f'$x_{{{i}}}$', fontsize=11, ha='center', weight='bold')

ax.set_xlabel('Iteration $n$', fontsize=13, weight='bold')
ax.set_ylabel('$x_n$', fontsize=13, weight='bold')
ax.set_title('Convergence of Fixed Point Iteration', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fixed_point_iteration.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 3: Integral Equation Domain
fig, ax = plt.subplots(figsize=(10, 7))

# Rectangle for domain
rect = Rectangle((0, 0), 10, 6, fill=True, alpha=0.2, color=colors[0], edgecolor='black', linewidth=2)
ax.add_patch(rect)

ax.text(5, 3, r'Domain: $[0, a] \times \mathbb{R}$', fontsize=14, ha='center', weight='bold')
ax.text(5, 2, r'Integral equation: $x(t) = u(t,x(t)) + \int_0^t p(t,s,x(s))ds$', fontsize=12, ha='center')
ax.text(5, 1, r'Kernels: $u(t,x)$, $p(t,s,x)$', fontsize=11, ha='center')

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 7)
ax.axis('off')
plt.tight_layout()
plt.savefig('integral_equation_domain.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 4: Convergence Rates Comparison
fig, ax = plt.subplots(figsize=(11, 7))

n_steps = 15
linear_rate = (0.8) ** np.arange(n_steps)
superlinear_rate = (0.8 ** 1.5) ** np.arange(n_steps)
quadratic_rate = (0.8 ** 2) ** np.arange(n_steps)

ax.semilogy(np.arange(n_steps), linear_rate, 'o-', linewidth=2.5, markersize=8,
           label='Linear convergence ($q=0.8$)', color=colors[0])
ax.semilogy(np.arange(n_steps), superlinear_rate, 's-', linewidth=2.5, markersize=8,
           label='Superlinear convergence', color=colors[1])
ax.semilogy(np.arange(n_steps), quadratic_rate, '^-', linewidth=2.5, markersize=8,
           label='Quadratic convergence', color=colors[2])

ax.set_xlabel('Iteration $n$', fontsize=13, weight='bold')
ax.set_ylabel('Error $\|x_n - x^*\|$', fontsize=13, weight='bold')
ax.set_title('Comparison of Convergence Rates', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('convergence_rates.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 5: Darbo Condition Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Illustration of measure of noncompactness
x = np.linspace(0, 10, 100)
y1 = 2 * np.sin(0.5 * x) + 5
y2 = 2 * np.sin(0.5 * x) + 3

ax.fill_between(x, y1, y2, alpha=0.3, color=colors[0], label='Set $\psi(X)$')
ax.plot(x, y1, linewidth=2, color=colors[0])
ax.plot(x, y2, linewidth=2, color=colors[0])

# Double arrows showing measure
ax.annotate('', xy=(1, 6.5), xytext=(1, 4.5),
            arrowprops=dict(arrowstyle='<->', color=colors[2], lw=2))
ax.text(0.3, 5.5, r'$\beta(X)$', fontsize=12, weight='bold', color=colors[2])

ax.text(5, 9, 'Darbo Condition: $\mu(Tx) \leq q\mu(x)$ for $q < 1$',
       fontsize=13, weight='bold', ha='center',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlim(-1, 11)
ax.set_ylim(2, 10)
ax.set_xlabel('Parameter', fontsize=12, weight='bold')
ax.set_ylabel('Measure of noncompactness', fontsize=12, weight='bold')
ax.set_title('Darbo Fixed Point Theorem', fontsize=14, weight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('darbo_condition.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 6: Application Framework
fig, ax = plt.subplots(figsize=(11, 8))

# Boxes for the framework
boxes = [
    (1, 7, 'Integral Equation\n$x(t) = (Tx)(t)$', colors[0]),
    (5, 7, 'Fixed Point\nTheorem', colors[1]),
    (9, 7, 'Solution\nExists', colors[2]),
    (3, 3.5, 'Volterra\nEquations', colors[3]),
    (7, 3.5, 'Functional\nEquations', colors[0]),
]

for x, y, text, color in boxes:
    rect = Rectangle((x-0.8, y-0.5), 1.6, 1, fill=True, facecolor=color, alpha=0.5,
                     edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=11, ha='center', va='center', weight='bold')

# Arrows
arrow_props = dict(arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
ax.annotate('', xy=(4.2, 7), xytext=(1.8, 7), arrowprops=arrow_props)
ax.annotate('', xy=(8.2, 7), xytext=(5.8, 7), arrowprops=arrow_props)
ax.annotate('', xy=(3.5, 4), xytext=(2, 6.5), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 4), xytext=(8, 6.5), arrowprops=arrow_props)

ax.set_xlim(0, 10)
ax.set_ylim(2, 9)
ax.axis('off')
ax.text(5, 0.5, 'Applications of Fixed Point Theorems', fontsize=14, ha='center', weight='bold')
plt.tight_layout()
plt.savefig('applications_framework.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("All figures generated successfully!")
print("Generated files:")
print("  - banach_contraction.pdf")
print("  - fixed_point_iteration.pdf")
print("  - integral_equation_domain.pdf")
print("  - convergence_rates.pdf")
print("  - darbo_condition.pdf")
print("  - applications_framework.pdf")
