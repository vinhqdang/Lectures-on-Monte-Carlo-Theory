#!/usr/bin/env python3
"""
Generate figures for Chapter 5c: Vector Spaces & Functional Analysis
Pathak - An Introduction to Nonlinear Analysis and Fixed Point Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01'}

# ============================================================================
# Figure 1: Hilbert Space Structure
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 9)
ax.axis('off')

# Title
ax.text(5, 8.5, 'Hilbert Space Structure', fontsize=16, fontweight='bold',
        ha='center', transform=ax.transData)

# Main box for Hilbert space
hilbert_box = FancyBboxPatch((0.5, 5), 9, 2.5, boxstyle="round,pad=0.1",
                             edgecolor=colors['primary'], facecolor='lightblue',
                             linewidth=2.5, alpha=0.3)
ax.add_patch(hilbert_box)
ax.text(5, 6.5, r'$\mathcal{H}$ = Hilbert Space', fontsize=13, ha='center', fontweight='bold')
ax.text(5, 5.9, r'Complete inner product space: $\langle \cdot, \cdot \rangle$',
        fontsize=11, ha='center', style='italic')

# Banach space box
banach_box = FancyBboxPatch((0.5, 2.5), 4, 2, boxstyle="round,pad=0.1",
                            edgecolor=colors['secondary'], facecolor='lightgreen',
                            linewidth=2, alpha=0.3)
ax.add_patch(banach_box)
ax.text(2.5, 3.8, 'Banach Space', fontsize=12, ha='center', fontweight='bold')
ax.text(2.5, 3.3, r'$\|\cdot\|$ norm', fontsize=10, ha='center')
ax.text(2.5, 2.8, 'Complete', fontsize=10, ha='center')

# Normed vector space box
normed_box = FancyBboxPatch((5.5, 2.5), 4, 2, boxstyle="round,pad=0.1",
                            edgecolor=colors['accent'], facecolor='lightyellow',
                            linewidth=2, alpha=0.3)
ax.add_patch(normed_box)
ax.text(7.5, 3.8, 'Inner Product Space', fontsize=12, ha='center', fontweight='bold')
ax.text(7.5, 3.3, r'$\langle x,y \rangle$', fontsize=10, ha='center')
ax.text(7.5, 2.8, r'$\|x\| = \sqrt{\langle x,x \rangle}$', fontsize=10, ha='center')

# Arrows showing relationships
arrow1 = FancyArrowPatch((5, 5), (2.5, 4.5), arrowstyle='->', mutation_scale=25,
                         linewidth=2, color='black')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((5, 5), (7.5, 4.5), arrowstyle='->', mutation_scale=25,
                         linewidth=2, color='black')
ax.add_patch(arrow2)

# Properties
y_pos = 1.8
ax.text(5, y_pos, 'Key Properties:', fontsize=12, ha='center', fontweight='bold')
y_pos -= 0.5
props = [
    r'$\mathcal{H}$ is a Banach space with $\|x\| = \sqrt{\langle x,x \rangle}$',
    r'Parallelogram law: $\|x+y\|^2 + \|x-y\|^2 = 2(\|x\|^2 + \|y\|^2)$',
    r'Cauchy-Schwarz: $|\langle x,y \rangle| \leq \|x\| \|y\|$'
]
for prop in props:
    ax.text(5, y_pos, prop, fontsize=9, ha='center')
    y_pos -= 0.4

plt.tight_layout()
plt.savefig('hilbert_space_structure.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: hilbert_space_structure.pdf")
plt.close()

# ============================================================================
# Figure 2: Banach Space Concepts
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Key Concepts in Banach Spaces', fontsize=14, fontweight='bold')

# Convergence
ax = axes[0, 0]
x = np.linspace(0, 10, 100)
y1 = np.exp(-0.3*x) * np.sin(x)
ax.plot(x, y1, linewidth=2.5, color=colors['primary'], label='Sequence $(x_n)$')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Limit $x$')
ax.fill_between(x, -0.1, 0.1, alpha=0.2, color='red')
ax.set_xlabel('$n$', fontsize=11)
ax.set_ylabel(r'$\|x_n - x\|$', fontsize=11)
ax.set_title('Strong Convergence in Norm', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Operator
ax = axes[0, 1]
ax.text(0.5, 0.8, 'Linear Operator', fontsize=12, ha='center', fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.6, r'$T: X \to Y$', fontsize=13, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.45, r'$T(\alpha x + \beta y) = \alpha T(x) + \beta T(y)$',
        fontsize=11, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(0.5, 0.25, 'Bounded if: $\|T\| = \sup_{\|x\|=1} \|T(x)\| < \infty$',
        fontsize=10, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax.axis('off')

# Dual space
ax = axes[1, 0]
ax.text(0.5, 0.85, "Dual Space $X^*$", fontsize=12, ha='center', fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.70, "Space of bounded linear functionals", fontsize=10, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.5, 0.55, r"$f: X \to \mathbb{R}$ or $\mathbb{C}$", fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.40, "Riesz Representation Theorem:", fontsize=10, ha='center',
        transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.25, r"For $f \in \mathcal{H}^*$, $\exists! y \in \mathcal{H}$:", fontsize=9, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.10, r"$f(x) = \langle x, y \rangle$ $\forall x \in \mathcal{H}$", fontsize=10, ha='center',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.axis('off')

# Weak convergence
ax = axes[1, 1]
ax.text(0.5, 0.85, "Weak Convergence", fontsize=12, ha='center', fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.70, r"$x_n \rightharpoonup x$ (weak)", fontsize=11, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.5, 0.55, r"$\lim_{n \to \infty} f(x_n) = f(x)$ for all $f \in X^*$", fontsize=10, ha='center',
        transform=ax.transAxes)
ax.text(0.5, 0.40, "vs.", fontsize=10, ha='center', transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.25, r"$x_n \to x$ (strong)", fontsize=11, ha='center',
        transform=ax.transAxes, style='italic')
ax.text(0.5, 0.10, r"$\lim_{n \to \infty} \|x_n - x\| = 0$", fontsize=10, ha='center',
        transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('banach_concepts.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: banach_concepts.pdf")
plt.close()

# ============================================================================
# Figure 3: Compact Operators
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Compact Operators', fontsize=13, fontweight='bold')

# Left: Visualization of compact mapping
ax = axes[0]
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# Bounded sequence
ax.plot(x_circle, y_circle, 'b-', linewidth=2.5, label='Unit ball in $X$')
ax.scatter([0.8], [0.6], color='red', s=100, zorder=5, label='Bounded sequence')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('Domain: Bounded Set', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Image under compact operator
ax = axes[1]
# Draw ellipse to represent compactification
ellipse = mpatches.Ellipse((0, 0), 1.5, 0.8, angle=30, alpha=0.3,
                           color=colors['primary'], label='Compact image')
ax.add_patch(ellipse)
theta_ellipse = np.linspace(0, 2*np.pi, 100)
x_ellipse = 0.75*np.cos(theta_ellipse)*np.cos(np.radians(30)) - 0.4*np.sin(theta_ellipse)*np.sin(np.radians(30))
y_ellipse = 0.75*np.cos(theta_ellipse)*np.sin(np.radians(30)) + 0.4*np.sin(theta_ellipse)*np.cos(np.radians(30))
ax.plot(x_ellipse, y_ellipse, 'b-', linewidth=2.5)
ax.scatter([0.5], [0.2], color='red', s=100, zorder=5, label='Convergent subsequence')
ax.set_xlim(-1, 1)
ax.set_ylim(-0.8, 0.8)
ax.set_aspect('equal')
ax.set_title(r'Image under $T$: Relatively Compact', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compact_operators.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: compact_operators.pdf")
plt.close()

# ============================================================================
# Figure 4: Spectral Theory Illustration
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Spectral Theory Basics', fontsize=13, fontweight='bold')

# Spectrum visualization
ax = axes[0]
# Eigenvalues on real axis
eigenvalues = np.array([-1.5, -0.5, 0.2, 1.2, 1.8])
ax.scatter(eigenvalues, np.zeros_like(eigenvalues), s=200, color=colors['primary'],
          marker='o', edgecolor='black', linewidth=2, zorder=5, label='Eigenvalues')

# Spectrum region
spectrum_x = np.linspace(-2, 2.5, 100)
spectrum_y = 0.3*np.sin(spectrum_x*2)
ax.fill_between(spectrum_x, -0.5, spectrum_y, alpha=0.2, color=colors['secondary'],
               label='Spectrum region')

ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlim(-2.5, 3)
ax.set_ylim(-1, 1)
ax.set_xlabel(r'$\lambda$ (Real part)', fontsize=11)
ax.set_title(r'Spectrum $\sigma(T)$', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Right: Power method convergence
ax = axes[1]
n_iter = 25
dominant_eigenvalue = 1.8
convergence = dominant_eigenvalue ** np.arange(n_iter)
convergence = convergence / convergence[0]  # Normalize

ax.semilogy(np.arange(n_iter), convergence, 'o-', linewidth=2.5, markersize=6,
           color=colors['secondary'], label='Power method')
ax.axhline(y=1e-10, color='red', linestyle='--', linewidth=2, alpha=0.7,
          label='Machine precision')
ax.set_xlabel('Iteration $k$', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Convergence to Dominant Eigenvalue', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('spectral_theory.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: spectral_theory.pdf")
plt.close()

# ============================================================================
# Figure 5: Fixed Point Property in Complete Spaces
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')

ax.text(5, 8, 'Fixed Point Theorems in Banach Spaces', fontsize=14, fontweight='bold',
       ha='center')

# Banach Contraction Mapping Theorem
box1 = FancyBboxPatch((0.2, 5.5), 4.5, 2, boxstyle="round,pad=0.1",
                      edgecolor=colors['primary'], facecolor='lightblue',
                      linewidth=2, alpha=0.3)
ax.add_patch(box1)
ax.text(2.45, 7.2, 'Banach Contraction', fontsize=11, ha='center', fontweight='bold')
ax.text(2.45, 6.8, 'Mapping Theorem', fontsize=11, ha='center', fontweight='bold')
ax.text(2.45, 6.3, r'$\|T(x) - T(y)\| \leq c\|x - y\|$', fontsize=9, ha='center')
ax.text(2.45, 5.9, r'$c < 1$ $\Rightarrow$ unique fixed point', fontsize=8, ha='center', style='italic')

# Brouwer Fixed Point Theorem
box2 = FancyBboxPatch((5.3, 5.5), 4.5, 2, boxstyle="round,pad=0.1",
                      edgecolor=colors['secondary'], facecolor='lightgreen',
                      linewidth=2, alpha=0.3)
ax.add_patch(box2)
ax.text(7.55, 7.2, 'Brouwer Fixed Point', fontsize=11, ha='center', fontweight='bold')
ax.text(7.55, 6.8, 'Theorem', fontsize=11, ha='center', fontweight='bold')
ax.text(7.55, 6.3, r'$T: B \to B$ continuous', fontsize=9, ha='center')
ax.text(7.55, 5.9, r'$B$ compact convex', fontsize=9, ha='center')

# Schauder Fixed Point Theorem
box3 = FancyBboxPatch((2.75, 3.2), 4.5, 2, boxstyle="round,pad=0.1",
                      edgecolor=colors['accent'], facecolor='lightyellow',
                      linewidth=2, alpha=0.3)
ax.add_patch(box3)
ax.text(5, 5, 'Schauder Fixed Point', fontsize=11, ha='center', fontweight='bold')
ax.text(5, 4.6, 'Theorem', fontsize=11, ha='center', fontweight='bold')
ax.text(5, 4.1, r'$T$ continuous, compact', fontsize=9, ha='center')
ax.text(5, 3.6, r'Self-map on compact convex', fontsize=8, ha='center', style='italic')

# Requirements section
ax.text(5, 2.6, 'Requirements for Fixed Points:', fontsize=10, ha='center', fontweight='bold')
reqs = [
    r'Completeness: $(X, \|\cdot\|)$ is Banach space',
    r'Continuity: $T: X \to X$ is continuous',
    r'Additional: Contraction, compactness, or convexity'
]
y_pos = 2.1
for req in reqs:
    ax.text(5, y_pos, req, fontsize=8, ha='center')
    y_pos -= 0.35

# Applications
ax.text(5, 1.1, 'Key Applications:', fontsize=9, ha='center', fontweight='bold')
ax.text(5, 0.65, 'Solvability of integral equations, existence of solutions to PDEs, iterative algorithms',
       fontsize=8, ha='center', style='italic')

plt.tight_layout()
plt.savefig('fixed_point_banach.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: fixed_point_banach.pdf")
plt.close()

# ============================================================================
# Figure 6: Numerical Example - Contraction Mapping
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Numerical Example: Contraction Mapping', fontsize=13, fontweight='bold')

# Define function and iterate
def f(x):
    """Contraction map: T(x) = 0.5*x + 0.3"""
    return 0.5 * x + 0.3

# Iterations
x0 = 1.5
iterations = 20
x_vals = [x0]
for i in range(iterations):
    x_vals.append(f(x_vals[-1]))

# Left plot: Convergence
ax = axes[0]
ax.plot(range(len(x_vals)), x_vals, 'o-', linewidth=2, markersize=5,
       color=colors['primary'], label='$x_n = T(x_{n-1})$')
fixed_point = 0.6  # Analytical: x = 0.5x + 0.3 => x = 0.6
ax.axhline(y=fixed_point, color='red', linestyle='--', linewidth=2,
          label=f'Fixed point: $x^* = {fixed_point}$')
ax.set_xlabel('Iteration $n$', fontsize=11)
ax.set_ylabel('$x_n$', fontsize=11)
ax.set_title('Convergence of Iterations', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right plot: Contraction factor
ax = axes[1]
errors = np.abs(np.array(x_vals) - fixed_point)
ax.semilogy(range(len(errors)), errors, 'o-', linewidth=2.5, markersize=6,
           color=colors['secondary'], label='$|x_n - x^*|$')

# Theoretical convergence rate
c = 0.5  # Contraction factor
theoretical = errors[0] * (c ** np.arange(len(errors)))
ax.semilogy(range(len(theoretical)), theoretical, '--', linewidth=2,
           color='gray', alpha=0.7, label=f'$|x_0 - x^*| \\cdot {c}^n$')

ax.set_xlabel('Iteration $n$', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11)
ax.set_title('Error Decay (Linear Convergence)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('contraction_mapping_example.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: contraction_mapping_example.pdf")
plt.close()

# ============================================================================
# Figure 7: Weak vs Strong Convergence Visualization
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 8))

# Create 2D illustration
# Strong convergence: sequence converges to a point
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 10)
ax.set_aspect('equal')

# Title
ax.text(5, 9.5, 'Weak vs. Strong Convergence in Hilbert Space', fontsize=13, fontweight='bold',
       ha='center')

# Strong convergence (left side)
center_s = np.array([2, 5])
for n in [20, 10, 5, 2, 1]:
    circle = plt.Circle(center_s, 0.3*np.sqrt(n)/np.sqrt(20), fill=False,
                       edgecolor=colors['primary'], linewidth=2-n/10)
    ax.add_patch(circle)
ax.scatter(*center_s, color=colors['primary'], s=300, marker='*', zorder=5, edgecolor='black', linewidth=2)
ax.text(2, 3.5, 'Strong: $x_n \\to x$\n$\\|x_n - x\\| \\to 0$', fontsize=10, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Weak convergence (right side)
center_w = np.array([8, 5])
# Oscillating sequence that doesn't converge strongly
for k, n in enumerate([1, 3, 5, 7, 10, 15]):
    angle = n * np.pi / 4
    x_pos = center_w[0] + np.cos(angle) * 0.8
    y_pos = center_w[1] + np.sin(angle) * 0.8
    ax.scatter(x_pos, y_pos, s=100*(21-n)/20, color=colors['secondary'],
              alpha=0.6, edgecolor='black', linewidth=1.5)
ax.scatter(*center_w, color='red', s=300, marker='*', zorder=5, edgecolor='black', linewidth=2)
ax.text(8, 3.5, 'Weak: $x_n \\rightharpoonup x$\nBounded oscillations', fontsize=10, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Add grid and axes
ax.axhline(y=5, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
ax.axvline(x=5, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
ax.set_xticks([])
ax.set_yticks([])

# Note at bottom
ax.text(5, 0.5, 'Strong convergence $\\Rightarrow$ Weak convergence, but not vice versa',
       fontsize=10, ha='center', style='italic',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('weak_vs_strong_convergence.pdf', dpi=300, bbox_inches='tight')
print("✓ Generated: weak_vs_strong_convergence.pdf")
plt.close()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
