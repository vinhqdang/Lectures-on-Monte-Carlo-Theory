#!/usr/bin/env python3
"""
Generate figures for Chapter 5f: Approximation of General Mapping
Fixed Point Theorems in Ordered Banach Spaces and Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9

# Figure 1: Fixed Point Concept - Operator T mapping
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: Identity vs T
x = np.linspace(0, 3, 300)
y_identity = x
y_T = np.sin(x) + 0.5 * x

ax1.plot(x, y_identity, 'b-', linewidth=2, label='$y = x$ (identity)')
ax1.plot(x, y_T, 'r-', linewidth=2, label='$y = Tx$')

# Find intersection (fixed point)
idx = np.argmin(np.abs(y_T - y_identity))
x_fixed = x[idx]
y_fixed = y_identity[idx]

ax1.plot(x_fixed, y_fixed, 'go', markersize=10, label=f'Fixed point $x^*$')
ax1.vlines(x_fixed, 0, y_fixed, colors='g', linestyles='--', alpha=0.5)
ax1.hlines(y_fixed, 0, x_fixed, colors='g', linestyles='--', alpha=0.5)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('Fixed Point: $Tx^* = x^*$')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 3)
ax1.set_ylim(0, 3)

# Right: Iteration sequence
ax2.plot(x, y_identity, 'b-', linewidth=2, label='$y = x$')
ax2.plot(x, y_T, 'r-', linewidth=2, label='$y = Tx$')

# Show iteration sequence
x0 = 0.5
colors = plt.cm.rainbow(np.linspace(0, 1, 6))
for i, color in enumerate(colors[:5]):
    x1 = y_T[np.argmin(np.abs(x - x0))]
    ax2.plot([x0, x0], [x0, x1], color=color, linewidth=1.5, alpha=0.7)
    ax2.plot([x0, x1], [x1, x1], color=color, linewidth=1.5, alpha=0.7)
    ax2.plot(x0, x0, 'o', color=color, markersize=6)
    x0 = x1

ax2.plot(x_fixed, y_fixed, 'g*', markersize=15, label='$x^*$ (limit)')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_title('Iteration Sequence: $x_{n+1} = Tx_n \\to x^*$')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 3)

plt.tight_layout()
plt.savefig('fixed_point_concept.pdf', dpi=300, bbox_inches='tight')
print("Created: fixed_point_concept.pdf")
plt.close()

# Figure 2: Cone K in Banach space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: Cone in 2D
theta = np.linspace(0, np.pi/4, 100)
r_outer = 3
x_cone = r_outer * np.cos(theta)
y_cone = r_outer * np.sin(theta)

ax1.fill_between(x_cone, 0, y_cone, alpha=0.3, color='blue', label='Cone $K$')
ax1.plot(x_cone, y_cone, 'b-', linewidth=2)
ax1.plot([0, r_outer], [0, 0], 'b-', linewidth=2)

# Add points in and out of cone
points_in = [(0.5, 0.1), (1.0, 0.4), (1.5, 0.5)]
points_out = [(1.0, 1.5), (0.5, 0.8)]

for pt in points_in:
    ax1.plot(pt[0], pt[1], 'go', markersize=8)
ax1.text(2.0, 0.1, 'Points in $K$', fontsize=10, color='green')

for pt in points_out:
    ax1.plot(pt[0], pt[1], 'rx', markersize=10, markeredgewidth=2)
ax1.text(1.5, 1.5, 'Points outside $K$', fontsize=10, color='red')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Cone $K$ in Banach Space')
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-0.5, 2)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Right: Positive operator properties
ax2.text(0.5, 0.95, 'Positive Operator Properties', fontsize=12, fontweight='bold',
         ha='center', transform=ax2.transAxes)

properties = [
    '1. Positive: $T(K) \\subseteq K$ (maps cone to itself)',
    '2. Strictly Positive: $T(\\overline{K}) \\subset K$ (interior)',
    '3. Strongly Positive: $T(\\overline{K}) \\subseteq \\text{int}(K)$',
    '',
    'Fixed Point Properties:',
    '• Every positive linear operator has a characteristic vector',
    '• Fixed points exist under compactness conditions',
    '• Multiple fixed points possible (order-theoretic methods)',
]

y_pos = 0.85
for prop in properties:
    if prop:
        ax2.text(0.05, y_pos, prop, fontsize=10, transform=ax2.transAxes,
                verticalalignment='top', family='monospace')
    y_pos -= 0.08

ax2.axis('off')

plt.tight_layout()
plt.savefig('cone_positive_operator.pdf', dpi=300, bbox_inches='tight')
print("Created: cone_positive_operator.pdf")
plt.close()

# Figure 3: Non-uniqueness of fixed points
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-2*np.pi, 2*np.pi, 500)
y1 = x
y2 = x + np.sin(x)/2
y3 = x + np.cos(x)/2

ax.plot(x, y1, 'b-', linewidth=2.5, label='$y = x$ (identity)')
ax.plot(x, y2, 'r-', linewidth=2, label='$y = x + \\frac{\\sin x}{2}$ (isotone $T_1$)')
ax.plot(x, y3, 'g--', linewidth=2, label='$y = x + \\frac{\\cos x}{2}$ (non-isotone $T_2$)')

# Find intersections
idx2 = np.where(np.abs(y2 - y1) < 0.1)[0]
idx3 = np.where(np.abs(y3 - y1) < 0.1)[0]

for idx in idx2[::len(idx2)//3]:
    ax.plot(x[idx], y1[idx], 'ro', markersize=8)

for idx in idx3[::len(idx3)//3]:
    ax.plot(x[idx], y1[idx], 'g^', markersize=8)

ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title('Non-uniqueness of Fixed Points for Different Operators', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_ylim(-6, 6)

plt.tight_layout()
plt.savefig('nonunique_fixed_points.pdf', dpi=300, bbox_inches='tight')
print("Created: nonunique_fixed_points.pdf")
plt.close()

# Figure 4: Convergence of iterations (weak contraction)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: Standard contraction
n_iter = 20
x0 = 2.0
k = 0.7
x_vals_standard = [x0]
for i in range(n_iter):
    x_vals_standard.append(k * x_vals_standard[-1])

ax1.semilogy(range(len(x_vals_standard)), np.abs(x_vals_standard), 'bo-',
            linewidth=2, markersize=6, label=f'Standard contraction: $k={k}$')

# Weak contraction (slower decay)
def phi(r):
    return r / (1 + r)

x_vals_weak = [x0]
for i in range(n_iter):
    x_vals_weak.append(x_vals_weak[-1] - phi(x_vals_weak[-1]))

ax1.semilogy(range(len(x_vals_weak)), np.abs(np.array(x_vals_weak)), 'rs-',
            linewidth=2, markersize=6, label='Weak contraction')

ax1.set_xlabel('Iteration $n$', fontsize=11)
ax1.set_ylabel('$|x_n|$ (log scale)', fontsize=11)
ax1.set_title('Convergence Rates: Contraction vs Weak Contraction', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# Right: Operator inequality visualization
ax2.text(0.5, 0.95, 'Weakly Contractive Mapping', fontsize=12, fontweight='bold',
         ha='center', transform=ax2.transAxes)

conditions = [
    'Definition: For every x, y in X,',
    '',
    '||Tx - Ty|| <= ||x - y|| - psi(||x - y||)',
    '',
    'where psi: [0, infinity) -> [0, infinity) satisfies:',
    '• psi(0) = 0',
    '• psi continuous and nondecreasing',
    '• psi(t) > 0 for all t > 0',
    '• lim(t->inf) psi(t) = infinity',
    '',
    'Examples: psi(t) = at, psi(t) = ln(1+t)'
]

y_pos = 0.88
for cond in conditions:
    if cond:
        ax2.text(0.05, y_pos, cond, fontsize=9.5, transform=ax2.transAxes,
                verticalalignment='top', family='monospace')
        y_pos -= 0.07
    else:
        y_pos -= 0.03

ax2.axis('off')

plt.tight_layout()
plt.savefig('weak_contractions.pdf', dpi=300, bbox_inches='tight')
print("Created: weak_contractions.pdf")
plt.close()

# Figure 5: Annular region (sandwich theorem)
fig, ax = plt.subplots(figsize=(10, 8))

# Draw annular regions
circle_inner = Circle((0, 0), 1, fill=True, alpha=0.2, color='green', label='Inner sphere $\|x\| = r$')
circle_outer = Circle((0, 0), 2.5, fill=True, alpha=0.15, color='blue', label='Outer sphere $\|x\| = R$')

ax.add_patch(circle_outer)
ax.add_patch(circle_inner)

# Annular region (sandwich)
annulus = plt.Circle((0, 0), 2.5, fill=False, edgecolor='blue', linewidth=2.5)
ax.add_patch(annulus)
annulus2 = plt.Circle((0, 0), 1, fill=False, edgecolor='green', linewidth=2.5)
ax.add_patch(annulus2)

# Add some sample points in annular region
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
radii = 1.75
x_points = radii * np.cos(angles)
y_points = radii * np.sin(angles)

ax.plot(x_points, y_points, 'ro', markersize=8, label='Feasible region')

# Annotations
ax.text(0, 0, 'Fixed Point\n$x^* \\in D$', fontsize=11, ha='center', va='center',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.annotate('$r$', xy=(0.7, 0), fontsize=12, color='green', fontweight='bold')
ax.annotate('$R$', xy=(1.75, 1.75), fontsize=12, color='blue', fontweight='bold')

# Add boundary condition annotations
ax.text(3.2, 0, 'Boundary: $\|x\| = R$\n$(1+\\varepsilon)x \\not\\preceq Fx$',
       fontsize=10, va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(-3.5, -2, 'Boundary: $\|x\| = r$\n$Fx \\not\\preceq x$',
       fontsize=10, va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Annular Region Method (Sandwich Theorem)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('annular_region.pdf', dpi=300, bbox_inches='tight')
print("Created: annular_region.pdf")
plt.close()

# Figure 6: Spectral radius and eigenvalues
fig, ax = plt.subplots(figsize=(10, 8))

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='Unit circle $|\\lambda| = 1$')

# Eigenvalues for contraction case
lambda_vals_contract = np.array([0.3 + 0.2j, 0.5 + 0.1j, 0.4 - 0.15j, 0.6, 0.7 - 0.05j])
ax.plot(lambda_vals_contract.real, lambda_vals_contract.imag, 'bo', markersize=10, label='Spectral radius $< 1$ (contraction)')

# Eigenvalues for expansion case
lambda_vals_expand = np.array([1.3 + 0.2j, 1.5 + 0.1j, 1.4 - 0.15j, 1.6, 1.2 - 0.05j])
ax.plot(lambda_vals_expand.real, lambda_vals_expand.imag, 'rs', markersize=10, label='Spectral radius $> 1$ (expansion)')

# Spectral radius circle for contraction
theta_spec = np.linspace(0, 2*np.pi, 200)
r_contract = 0.7
ax.plot(r_contract * np.cos(theta_spec), r_contract * np.sin(theta_spec), 'b:',
       linewidth=2, label=f'$r(F) = {r_contract}$ (contraction region)')

ax.fill_between(r_contract * np.cos(theta_spec), r_contract * np.sin(theta_spec),
               alpha=0.1, color='blue')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel('$\\text{Re}(\\lambda)$', fontsize=12)
ax.set_ylabel('$\\text{Im}(\\lambda)$', fontsize=12)
ax.set_title('Spectral Radius and Fixed Point Existence', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('spectral_radius.pdf', dpi=300, bbox_inches='tight')
print("Created: spectral_radius.pdf")
plt.close()

# Figure 7: Partially ordered set properties
fig, ax = plt.subplots(figsize=(12, 6))

ax.text(0.5, 0.98, 'Fixed Point Theorems in Partially Ordered Sets',
       fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)

content = """
Key Concepts:

1. Partially Ordered Set (X, <=):
   • Reflexive: x <= x
   • Transitive: x <= y, y <= z => x <= z
   • Anti-symmetric: x <= y, y <= x => x = y

2. Monotone (Isotone) Mapping f: X -> X
   • Nondecreasing: x <= y => f(x) <= f(y)
   • Nonincreasing: x <= y => f(x) >= f(y)

3. Weakly Contractive Mapping:
   ||Tx - Ty|| <= ||x - y|| - psi(||x - y||)
   where psi(0) = 0, psi continuous, nondecreasing

4. Convergence Theorem:
   If x0 <= f(x0), then xn = f(xn-1) is nondecreasing
   and converges to fixed point x*

5. Uniqueness Conditions (SC1)-(SC3):
   (SC1) Every pair has a lower bound or upper bound
   (SC2) Sequences have comparable subsequences
   (SC3) f maps comparable elements to comparable elements
"""

ax.text(0.02, 0.92, content, fontsize=9.5, transform=ax.transAxes, verticalalignment='top',
       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.axis('off')

plt.tight_layout()
plt.savefig('partially_ordered_concepts.pdf', dpi=300, bbox_inches='tight')
print("Created: partially_ordered_concepts.pdf")
plt.close()

print("\nAll figures generated successfully!")
