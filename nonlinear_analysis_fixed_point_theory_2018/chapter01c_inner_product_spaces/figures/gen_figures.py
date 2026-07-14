#!/usr/bin/env python3
"""
Generate figures for Chapter 1c: Inner Product Spaces
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

output_dir = '.'

# =============================================================================
# Figure 1: Inner Product Definition (Cauchy-Schwarz Inequality)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Cauchy-Schwarz Inequality',
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

# Box content
box_text = r'$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$' + '\n\n'
box_text += 'Equality holds iff x and y are linearly dependent'

ax.text(0.5, 0.65, box_text, ha='center', va='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7),
        transform=ax.transAxes, family='monospace')

# Applications
apps = [
    r'Triangle inequality: $\|x+y\| \leq \|x\| + \|y\|$',
    r'Continuity of inner product',
    r'Orthogonality characterization',
    r'Best approximation theory'
]

y_pos = 0.45
for i, app in enumerate(apps):
    ax.text(0.1, y_pos - i*0.08, f'• {app}', fontsize=11,
            transform=ax.transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_cauchy_schwarz.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Orthogonality in Inner Product Spaces
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Draw two orthogonal vectors
origin = [0, 0]
v1 = np.array([3, 0])
v2 = np.array([0, 2.5])

# Draw vectors
ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
         fc='blue', ec='blue', linewidth=2, label=r'$x$')
ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
         fc='red', ec='red', linewidth=2, label=r'$y$')

# Draw right angle symbol
square_size = 0.4
square = mpatches.Rectangle((0, 0), square_size, square_size,
                            fill=False, edgecolor='black', linewidth=1)
ax.add_patch(square)

# Labels
ax.text(v1[0]/2, v1[1]-0.4, r'$x$', fontsize=14, color='blue', fontweight='bold')
ax.text(v2[0]-0.5, v2[1]/2, r'$y$', fontsize=14, color='red', fontweight='bold')
ax.text(0.7, 0.3, r'$\langle x,y \rangle = 0$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('Orthogonal Vectors in Inner Product Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_orthogonality.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: Hilbert Space Examples
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

ax.text(0.5, 0.95, 'Common Hilbert Spaces',
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

examples = [
    (r'$\mathbb{R}^n$', r'$\langle x,y \rangle = \sum_{i=1}^n x_i y_i$', 'lightblue'),
    (r'$\ell^2$', r'$\langle x,y \rangle = \sum_{i=1}^{\infty} x_i \overline{y_i}$', 'lightgreen'),
    (r'$L^2[a,b]$', r'$\langle f,g \rangle = \int_a^b f(t)\overline{g(t)} dt$', 'lightyellow'),
    (r'$\ell^2(\mathbb{N})$', r'Sequences with finite $\sum |x_i|^2$', 'lightcoral'),
]

y_start = 0.80
for i, (space, inner_prod, color) in enumerate(examples):
    y = y_start - i * 0.16

    # Space name
    ax.text(0.05, y, space, fontsize=13, fontweight='bold',
            transform=ax.transAxes, verticalalignment='center')

    # Inner product
    ax.text(0.25, y, inner_prod, fontsize=11,
            transform=ax.transAxes, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.6))

# Properties box
props_text = 'All Hilbert spaces are:\n• Complete Banach spaces with inner product\n• Separable (countable dense subset)\n• Possess orthonormal bases'
ax.text(0.5, 0.08, props_text, fontsize=10, ha='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_hilbert_examples.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: Parallelogram Law
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Vector diagram
ax1.set_xlim(-0.5, 4.5)
ax1.set_ylim(-0.5, 4)
ax1.set_aspect('equal')

x = np.array([3, 0])
y = np.array([1, 2.5])

# Draw vectors
ax1.arrow(0, 0, x[0], x[1], head_width=0.2, head_length=0.2,
         fc='blue', ec='blue', linewidth=2)
ax1.arrow(0, 0, y[0], y[1], head_width=0.2, head_length=0.2,
         fc='red', ec='red', linewidth=2)
ax1.arrow(x[0], x[1], y[0], y[1], head_width=0.2, head_length=0.2,
         fc='green', ec='green', linewidth=2, linestyle='--', alpha=0.7)

# Sum and difference
sum_vec = x + y
diff_vec = x - y

ax1.arrow(0, 0, sum_vec[0], sum_vec[1], head_width=0.2, head_length=0.2,
         fc='purple', ec='purple', linewidth=2, linestyle='--', alpha=0.7)

ax1.text(x[0]/2, x[1]-0.4, r'$x$', fontsize=12, color='blue', fontweight='bold')
ax1.text(y[0]-0.3, y[1]/2, r'$y$', fontsize=12, color='red', fontweight='bold')
ax1.text(sum_vec[0]-0.5, sum_vec[1]+0.2, r'$x+y$', fontsize=12, color='purple', fontweight='bold')

ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel(r'$x_1$', fontsize=12)
ax1.set_ylabel(r'$x_2$', fontsize=12)
ax1.set_title('Vector Sum and Difference', fontsize=13, fontweight='bold')

# Right: Parallelogram law formula
ax2.axis('off')
formula = r'$2(\|x\|^2 + \|y\|^2) = \|x+y\|^2 + \|x-y\|^2$'
ax2.text(0.5, 0.7, 'Parallelogram Law', fontsize=14, fontweight='bold',
         ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.5, formula, fontsize=13, ha='center', transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.7))

explanation = 'This law characterizes inner product spaces.\n' + \
              'A normed space is an inner product space\n' + \
              'if and only if the parallelogram law holds.'
ax2.text(0.5, 0.2, explanation, fontsize=11, ha='center', transform=ax2.transAxes,
         style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_parallelogram_law.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: Gram-Schmidt Orthogonalization Process
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.axis('off')

ax.text(0.5, 0.95, 'Gram-Schmidt Orthogonalization Process',
        ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)

steps = [
    (r'$e_1 = \frac{v_1}{\|v_1\|}$', 'Normalize first vector', 0.80),
    (r'$u_2 = v_2 - \langle v_2, e_1 \rangle e_1$', 'Orthogonalize second vector', 0.63),
    (r'$e_2 = \frac{u_2}{\|u_2\|}$', 'Normalize', 0.53),
    (r'$u_k = v_k - \sum_{j=1}^{k-1} \langle v_k, e_j \rangle e_j$', 'Orthogonalize k-th vector', 0.38),
    (r'$e_k = \frac{u_k}{\|u_k\|}$', 'Normalize', 0.28),
]

for formula, desc, y_pos in steps:
    ax.text(0.05, y_pos, formula, fontsize=12, transform=ax.transAxes,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.7))
    ax.text(0.55, y_pos, desc, fontsize=11, transform=ax.transAxes,
           verticalalignment='top', style='italic')

# Result box
result = r'Result: $\{e_1, e_2, \ldots, e_n\}$ is an orthonormal set'
ax.text(0.5, 0.08, result, fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.7),
       fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_gram_schmidt.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 6: Orthogonal Projection
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create subspace (line)
t = np.linspace(-2, 4, 100)
subspace_x = t * np.cos(np.pi/6)
subspace_y = t * np.sin(np.pi/6)

ax.plot(subspace_x, subspace_y, 'k-', linewidth=2, label='Subspace M')

# Point (vector)
point = np.array([2.5, 2.2])
ax.plot(point[0], point[1], 'ro', markersize=10, label='x')

# Projection
proj_scalar = np.dot(point, np.array([np.cos(np.pi/6), np.sin(np.pi/6)]))
proj = proj_scalar * np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
ax.plot(proj[0], proj[1], 'go', markersize=10, label='P_M(x)')

# Draw perpendicular line
ax.plot([point[0], proj[0]], [point[1], proj[1]], 'b--', linewidth=1.5, label='x - P_M(x)')

# Origin
ax.plot(0, 0, 'ko', markersize=8)

# Annotations
ax.annotate('x', point + 0.15, fontsize=13, fontweight='bold', color='red')
ax.annotate(r'$P_M(x)$', proj + np.array([-0.3, 0.15]), fontsize=13, fontweight='bold', color='green')
ax.annotate(r'$x - P_M(x)$', (point + proj)/2 + np.array([0.2, -0.2]), fontsize=11, style='italic')

ax.set_xlim(-2, 4)
ax.set_ylim(-2, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('Orthogonal Projection onto Subspace M', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_projection.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 7: Riesz Representation Theorem
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.axis('off')

ax.text(0.5, 0.95, "Riesz Representation Theorem",
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

# Main theorem
theorem = "For any bounded linear functional $f$ on a Hilbert space $H$,\n" + \
          "there exists a unique $y \\in H$ such that\n\n" + \
          r"$f(x) = \langle x, y \rangle$ for all $x \in H$"

ax.text(0.5, 0.75, theorem, fontsize=12, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8),
        verticalalignment='center')

# Properties
props = [
    r"$\|f\| = \|y\|$ (norm preservation)",
    r"The map $f \mapsto y$ is an isometric isomorphism $H^* \cong H$",
    r"$H^* = H$ (dual space equals the space itself)",
]

y_pos = 0.55
for i, prop in enumerate(props):
    ax.text(0.05, y_pos - i*0.08, f"• {prop}", fontsize=11,
           transform=ax.transAxes, verticalalignment='top')

# Significance box
sig = "Significance:\nThis theorem is fundamental to functional analysis.\nIt shows that Hilbert spaces are self-dual."

ax.text(0.5, 0.15, sig, fontsize=11, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8),
        style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_riesz_theorem.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 8: Convergence in Hilbert Space
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')

ax.text(0.5, 0.95, 'Convergence Concepts in Hilbert Spaces',
        ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)

concepts = [
    ('Strong Convergence', r'$\|x_n - x\| \to 0$', 'Norm convergence'),
    ('Weak Convergence', r'$\langle x_n, y \rangle \to \langle x, y \rangle$ for all $y \in H$', 'Weaker notion'),
    ('Weak-* Convergence', r'$f_n(x) \to f(x)$ for all $x \in H$', 'For functionals'),
]

y_positions = [0.78, 0.50, 0.25]

for i, (name, formula, desc) in enumerate(concepts):
    y = y_positions[i]

    # Name
    ax.text(0.05, y, name, fontsize=12, fontweight='bold',
           transform=ax.transAxes, verticalalignment='top')

    # Formula
    ax.text(0.35, y, formula, fontsize=11,
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.6))

    # Description
    ax.text(0.75, y, desc, fontsize=10,
           transform=ax.transAxes, verticalalignment='top', style='italic')

# Relationship
rel = r'Strong convergence $\Rightarrow$ Weak convergence, but converse is NOT true'
ax.text(0.5, 0.08, rel, fontsize=11, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.8),
       style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_convergence.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 9: Norm Induced by Inner Product
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Draw a circle (unit sphere in R^2)
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

ax.plot(circle[0], circle[1], 'b-', linewidth=2, label=r'Unit sphere: $\|x\| = 1$')

# Draw some points on the sphere
angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
for angle in angles:
    point = np.array([np.cos(angle), np.sin(angle)])
    ax.plot(point[0], point[1], 'ro', markersize=8)
    ax.arrow(0, 0, point[0]*0.9, point[1]*0.9, head_width=0.1, head_length=0.08,
            fc='red', ec='red', alpha=0.3)

# Add formula
formula = r'$\|x\| = \sqrt{\langle x, x \rangle}$'
ax.text(0, -1.5, formula, fontsize=13, ha='center',
       bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.8))

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('Norm Induced by Inner Product (Unit Sphere)', fontsize=14, fontweight='bold')

# Properties box
props = r'$\|x\|^2 = \langle x, x \rangle$' + '\n' + \
        r'Satisfies: positivity, homogeneity, triangle inequality'
ax.text(0, 1.45, props, fontsize=10, ha='center',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_unit_sphere.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 10: Complete Orthonormal System
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.axis('off')

ax.text(0.5, 0.95, 'Complete Orthonormal System (CONS) in Hilbert Space',
        ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)

# Definition box
definition = "An orthonormal set $\\{e_i\\}_{i \\in I}$ is complete if:\n\n" + \
             r"$\langle x, e_i \rangle = 0$ for all $i \Rightarrow x = 0$"

ax.text(0.5, 0.80, definition, fontsize=12, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7),
        verticalalignment='center')

# Parseval's equality
parseval = r"Parseval's Equality: $\|x\|^2 = \sum_{i \in I} |\langle x, e_i \rangle|^2$"
ax.text(0.5, 0.62, parseval, fontsize=12, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.7))

# Expansion formula
expansion = r"Fourier expansion: $x = \sum_{i \in I} \langle x, e_i \rangle e_i$"
ax.text(0.5, 0.50, expansion, fontsize=12, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.7))

# Examples
examples_title = "Examples of CONS:"
ax.text(0.05, 0.35, examples_title, fontsize=12, fontweight='bold',
       transform=ax.transAxes)

examples_list = [
    (r"$\mathbb{R}^n$", r"Standard basis: $\{e_1, e_2, \ldots, e_n\}$", 0.28),
    (r"$\ell^2$", r"$e_i = (0, \ldots, 1, \ldots, 0)$ with 1 in position $i$", 0.20),
    (r"$L^2[0, 2\pi]$", r"Trigonometric basis: $\{1, \cos(nx), \sin(nx)\}$", 0.12),
]

for space, basis, y in examples_list:
    ax.text(0.08, y, space, fontsize=11, transform=ax.transAxes, fontweight='bold')
    ax.text(0.25, y, basis, fontsize=10, transform=ax.transAxes, style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_orthonormal_system.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 11: Energy/Power Convergence Example
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Norms convergence
n = np.arange(1, 50)
norm_strong = 1 / n  # Strong convergence to 0
norm_weak = 1 / np.sqrt(n)  # Weak convergence to 0

ax1.semilogy(n, norm_strong, 'b-', linewidth=2, label=r'$\|x_n\|$ (strong)')
ax1.semilogy(n, norm_weak, 'r--', linewidth=2, label=r'$|\langle x_n, y \rangle|$ (weak)')
ax1.fill_between(n, 0, norm_strong, alpha=0.2, color='blue')
ax1.set_xlabel('n', fontsize=12)
ax1.set_ylabel('Magnitude (log scale)', fontsize=12)
ax1.set_title('Strong vs Weak Convergence', fontsize=13, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim(0, 50)

# Right: Example sequence
ax2.axis('off')
example_text = r"Example: $x_n = \frac{1}{n} \sin(nx)$ in $L^2[0, \pi]$" + "\n\n" + \
               "• Converges weakly to 0\n" + \
               "• Does NOT converge strongly to 0\n" + \
               "• Oscillations prevent strong convergence"

ax2.text(0.5, 0.5, example_text, fontsize=11, ha='center', transform=ax2.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8),
        verticalalignment='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_convergence_example.pdf', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 12: Applications Summary
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.axis('off')

ax.text(0.5, 0.97, 'Applications of Hilbert Space Theory',
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

applications = [
    ("Quantum Mechanics", "State space as Hilbert space (wave functions)", "lightyellow"),
    ("Signal Processing", "Fourier analysis, wavelets, time-frequency analysis", "lightgreen"),
    ("Partial Differential Equations", "Weak solutions, Sobolev spaces", "lightblue"),
    ("Optimization", "Convex optimization, projection methods", "lightcoral"),
    ("Approximation Theory", "Best approximation, least squares", "lightgray"),
    ("Machine Learning", "RKHS theory, kernel methods", "lightyellow"),
]

y_start = 0.87
for i, (app, desc, color) in enumerate(applications):
    y = y_start - i * 0.13

    # Application name
    ax.text(0.05, y, app, fontsize=12, fontweight='bold',
           transform=ax.transAxes, verticalalignment='center')

    # Description
    ax.text(0.3, y, desc, fontsize=10,
           transform=ax.transAxes, verticalalignment='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.6),
           style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_applications.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Generated files:")
import os
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.pdf'):
        print(f"  - {f}")
