"""
Generate figures for Chapter 4a: Monotone Operators and Iterative Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01'}

# Figure 1: Monotone Operator Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a simple 2D example showing monotonicity
x1 = np.array([1, 2])
x2 = np.array([3, 1])
Tx1 = np.array([1.5, 2.5])
Tx2 = np.array([3.5, 0.5])

# Plot points
ax.plot(*x1, 'o', markersize=12, color=colors['primary'], label='$x$', zorder=5)
ax.plot(*x2, 'o', markersize=12, color=colors['secondary'], label='$y$', zorder=5)
ax.plot(*Tx1, 's', markersize=12, color=colors['primary'], label='$Tx$', zorder=5)
ax.plot(*Tx2, 's', markersize=12, color=colors['secondary'], label='$Ty$', zorder=5)

# Draw arrows from points to their images
arrow1 = FancyArrowPatch(x1, Tx1, arrowstyle='->', mutation_scale=20,
                         linewidth=2, color=colors['primary'], alpha=0.6)
arrow2 = FancyArrowPatch(x2, Tx2, arrowstyle='->', mutation_scale=20,
                         linewidth=2, color=colors['secondary'], alpha=0.6)
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# Draw the vector (x-y) and (Tx-Ty)
arrow3 = FancyArrowPatch(x1, x2, arrowstyle='->', mutation_scale=15,
                         linewidth=1.5, color='gray', linestyle='--', alpha=0.7)
arrow4 = FancyArrowPatch(Tx1, Tx2, arrowstyle='->', mutation_scale=15,
                         linewidth=1.5, color='gray', linestyle='--', alpha=0.7)
ax.add_patch(arrow3)
ax.add_patch(arrow4)

# Annotations
ax.text(x1[0]-0.3, x1[1]+0.2, r'$x$', fontsize=14, fontweight='bold')
ax.text(x2[0]+0.1, x2[1]-0.3, r'$y$', fontsize=14, fontweight='bold')
ax.text(Tx1[0]-0.4, Tx1[1]+0.2, r'$Tx$', fontsize=14, fontweight='bold')
ax.text(Tx2[0]+0.1, Tx2[1]-0.3, r'$Ty$', fontsize=14, fontweight='bold')

# Add monotonicity condition
ax.text(0.5, -0.5, r'Monotone: $(Tx - Ty, x - y) \geq 0$',
        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-1.5, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('First coordinate', fontsize=11)
ax.set_ylabel('Second coordinate', fontsize=11)
ax.set_title('Visualization of Monotone Operator Property', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_monotone_operator.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_monotone_operator.pdf")
plt.close()

# Figure 2: Comparison of monotonicity types
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Weakly monotone
x = np.linspace(-2, 2, 100)
y = x**2
axes[0].plot(x, y, linewidth=2.5, color=colors['primary'])
axes[0].fill_between(x, y, alpha=0.2, color=colors['primary'])
axes[0].set_title('Weakly Monotone', fontsize=12, fontweight='bold')
axes[0].set_xlabel('$x$', fontsize=11)
axes[0].set_ylabel('$Tx$', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)

# Strictly monotone
y_strict = x**3
axes[1].plot(x, y_strict, linewidth=2.5, color=colors['secondary'])
axes[1].fill_between(x, y_strict, alpha=0.2, color=colors['secondary'])
axes[1].set_title('Strictly Monotone', fontsize=12, fontweight='bold')
axes[1].set_xlabel('$x$', fontsize=11)
axes[1].set_ylabel('$Tx$', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)

# Strongly monotone
y_strong = x + 0.5*x**2
axes[2].plot(x, y_strong, linewidth=2.5, color=colors['accent'])
axes[2].fill_between(x, y_strong, alpha=0.2, color=colors['accent'])
axes[2].set_title('Strongly Monotone', fontsize=12, fontweight='bold')
axes[2].set_xlabel('$x$', fontsize=11)
axes[2].set_ylabel('$Tx$', fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linewidth=0.5)
axes[2].axvline(x=0, color='k', linewidth=0.5)

fig.suptitle('Types of Monotonicity', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_monotonicity_types.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_monotonicity_types.pdf")
plt.close()

# Figure 3: Iterative convergence
fig, ax = plt.subplots(figsize=(10, 7))

# Simulate convergence of iterations
np.random.seed(42)
iterations = np.arange(0, 50)

# Different convergence rates
convergence_linear = 0.9 ** iterations
convergence_superlinear = 0.85 ** iterations * (1 + 0.1*np.sin(iterations/5))
convergence_fast = 0.7 ** (iterations**1.2)

ax.semilogy(iterations, convergence_linear, 'o-', label='Linear convergence',
            linewidth=2.5, markersize=6, color=colors['primary'])
ax.semilogy(iterations, convergence_superlinear, 's--', label='Superlinear convergence',
            linewidth=2.5, markersize=5, color=colors['secondary'])
ax.semilogy(iterations, convergence_fast, '^:', label='Fast convergence',
            linewidth=2.5, markersize=6, color=colors['accent'])

ax.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Error $\|x_n - x^*\|$', fontsize=12, fontweight='bold')
ax.set_title('Convergence Rates of Iterative Methods', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(1e-10, 1)
plt.tight_layout()
plt.savefig('fig_convergence_rates.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_convergence_rates.pdf")
plt.close()

# Figure 4: Fixed point iteration scheme
fig, ax = plt.subplots(figsize=(11, 8))

# Plot a function and the iteration path
x_vals = np.linspace(0, 2.5, 200)
f_vals = np.sqrt(x_vals)  # Function with fixed point at 1
id_vals = x_vals  # Identity line

ax.plot(x_vals, f_vals, linewidth=3, label='$y = \\sqrt{x}$ (operator $T$)',
        color=colors['primary'])
ax.plot(x_vals, id_vals, linewidth=3, label='$y = x$ (identity)',
        color='black', linestyle='--', alpha=0.7)

# Simulate fixed point iteration starting from x0 = 0.3
x_n = 0.3
path_x = [x_n]
path_y = [0]

for _ in range(20):
    y_n = np.sqrt(x_n)
    path_x.extend([x_n, x_n])
    path_y.extend([y_n, y_n])
    path_x.append(y_n)
    path_y.append(y_n)
    x_n = y_n
    if x_n > 2.3:
        break

ax.plot(path_x[:40], path_y[:40], 'r-', linewidth=1.5, alpha=0.6,
        label='Iteration path')

# Mark the fixed point
fixed_pt = 1.0
ax.plot(fixed_pt, fixed_pt, 'go', markersize=12, label='Fixed point $x^*$',
        zorder=5)

# Mark starting point
ax.plot(0.3, 0, 'r*', markersize=15, label='Starting point $x_0$', zorder=5)

ax.set_xlim(0, 2.5)
ax.set_ylim(0, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('$x$', fontsize=12, fontweight='bold')
ax.set_ylabel('$y$', fontsize=12, fontweight='bold')
ax.set_title('Fixed Point Iteration Method: Graphical Representation',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_fixed_point_iteration.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_fixed_point_iteration.pdf")
plt.close()

# Figure 5: Subdifferential of convex function
fig, ax = plt.subplots(figsize=(10, 7))

# Plot a convex function
x_vals = np.linspace(-3, 3, 200)
f_vals = x_vals**2 + 0.5*np.abs(x_vals) - 1

ax.plot(x_vals, f_vals, linewidth=3, color=colors['primary'], label='Convex function $f(x)$')

# At x=0, plot the subdifferential cone
x0 = 0
f_x0 = f_vals[100]

# Multiple subgradients at x=0
slopes = np.linspace(-1, 1, 5)
for slope in slopes:
    y_subgrad = f_x0 + slope * (x_vals - x0)
    ax.plot(x_vals, y_subgrad, '--', alpha=0.4, linewidth=1.5, color='gray')

# Highlight the subdifferential
ax.axvline(x=x0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
ax.fill_between([-0.3, 0.3], [f_x0-0.5]*2, [f_x0+0.5]*2, alpha=0.2, color='green')
ax.text(0.05, f_x0-1, r'$\partial f(0)$', fontsize=11, color='green', fontweight='bold')

ax.plot(x0, f_x0, 'go', markersize=10, label='Point $x_0 = 0$', zorder=5)

ax.set_xlim(-3, 3)
ax.set_ylim(-2, 10)
ax.set_xlabel('$x$', fontsize=12, fontweight='bold')
ax.set_ylabel('$f(x)$', fontsize=12, fontweight='bold')
ax.set_title('Subdifferential of a Convex Function', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_subdifferential.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_subdifferential.pdf")
plt.close()

# Figure 6: Maximal monotone operator property
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: Monotone but not maximal
theta = np.linspace(0, 2*np.pi, 50)
# A smaller set (not maximal)
x_circ = 1.5 * np.cos(theta)
y_circ = 1.5 * np.sin(theta)
axes[0].plot(x_circ, y_circ, 'o-', markersize=4, color=colors['primary'],
            linewidth=2, label='Graph of monotone $T_1$')
axes[0].fill(x_circ, y_circ, alpha=0.1, color=colors['primary'])

# Possible extension
x_ext = 2.5 * np.cos(theta)
y_ext = 2.5 * np.sin(theta)
axes[0].plot(x_ext, y_ext, 's--', markersize=3, color=colors['secondary'],
            linewidth=1.5, alpha=0.6, label='Possible extension')

axes[0].set_xlim(-3.5, 3.5)
axes[0].set_ylim(-3.5, 3.5)
axes[0].set_aspect('equal')
axes[0].set_xlabel(r'$x \in X$', fontsize=11)
axes[0].set_ylabel(r'$Tx \in X^*$', fontsize=11)
axes[0].set_title('Monotone but not Maximal', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right panel: Maximal monotone
theta2 = np.linspace(0, 2*np.pi, 100)
x_max = 2 * np.cos(theta2)
y_max = 2 * np.sin(theta2)
axes[1].plot(x_max, y_max, 'o-', markersize=4, color=colors['accent'],
            linewidth=2.5, label='Graph of maximal monotone $T_2$')
axes[1].fill(x_max, y_max, alpha=0.15, color=colors['accent'])

# Add notation for maximality
axes[1].text(0, 0, 'Maximal:\nNo proper\nextension', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontweight='bold')

axes[1].set_xlim(-3.5, 3.5)
axes[1].set_ylim(-3.5, 3.5)
axes[1].set_aspect('equal')
axes[1].set_xlabel(r'$x \in X$', fontsize=11)
axes[1].set_ylabel(r'$Tx \in X^*$', fontsize=11)
axes[1].set_title('Maximal Monotone', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Maximality of Monotone Operators', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('fig_maximal_monotone.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_maximal_monotone.pdf")
plt.close()

# Figure 7: Numerical example - solving an operator equation
fig, ax = plt.subplots(figsize=(10, 7))

# Simulate solving Tx = 0 using iterative method
n_iterations = np.arange(1, 26)
# Error decreases linearly in iterations (linear convergence)
error_linear = 10 * np.exp(-0.3 * n_iterations) + 0.01*np.random.randn(25)
error_linear = np.maximum(error_linear, 1e-8)

ax.semilogy(n_iterations, error_linear, 'o-', linewidth=2.5, markersize=7,
           color=colors['primary'], label=r'Iterative solution of $Tx = y$')

# Add reference lines for convergence rates
n_ref = np.arange(1, 26, 0.1)
ax.semilogy(n_ref, 10*0.9**n_ref, '--', linewidth=2, color=colors['secondary'],
           alpha=0.6, label='Linear convergence ($r=0.9$)')
ax.semilogy(n_ref, 10*0.7**n_ref, ':', linewidth=2, color=colors['accent'],
           alpha=0.6, label='Faster convergence ($r=0.7$)')

ax.set_xlabel('Iteration $n$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Residual $\|Tx_n - y\|$', fontsize=12, fontweight='bold')
ax.set_title('Numerical Example: Iterative Solution of Operator Equation',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(1e-8, 15)
plt.tight_layout()
plt.savefig('fig_numerical_convergence.pdf', dpi=300, bbox_inches='tight')
print("Created: fig_numerical_convergence.pdf")
plt.close()

print("\nAll figures generated successfully!")
