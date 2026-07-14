#!/usr/bin/env python3
"""
Generate figures for Chapter 1b: Metric & Normed Spaces
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c', 'error': '#d62728'}

# Figure 1: Operator Norm Illustration
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# Unit ball in X
circle_x = plt.Circle((0, 0), 1, color=colors['primary'], fill=False, linewidth=2, label='Unit ball $B_X$')
ax.add_patch(circle_x)

# Mapped unit ball in Y (stretched)
circle_y = plt.Circle((1.5, 0), 1.5, color=colors['secondary'], fill=False, linewidth=2, linestyle='--', label='$A(B_X)$ (image of unit ball)')
ax.add_patch(circle_y)

# Add vectors
arrow1 = FancyArrowPatch((0, 0), (0.7, 0.7), arrowstyle='->', mutation_scale=20, linewidth=2, color=colors['primary'])
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((0, 0), (2.1, 0.9), arrowstyle='->', mutation_scale=20, linewidth=2, color=colors['secondary'])
ax.add_patch(arrow2)

ax.text(0.35, 0.9, '$x$', fontsize=12, color=colors['primary'])
ax.text(1.2, 1.2, '$Ax$', fontsize=12, color=colors['secondary'])
ax.text(0.8, -1.2, '$\\|Ax\\| \\leq M\\|x\\|$', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel('$X$', fontsize=11)
ax.set_ylabel('$Y$', fontsize=11)
ax.set_title('Bounded Linear Operator: Norm Control', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('operator_norm.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Convergence Hierarchy
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Boxes
boxes = [
    (1, 3, 'Uniform\nConvergence', colors['error']),
    (4, 3, 'Strong\nConvergence', colors['secondary']),
    (7, 3, 'Weak\nConvergence', colors['primary']),
]

for x, y, label, color in boxes:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=color, facecolor='white', linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((1.7, 3), (3.3, 3), arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((4.7, 3), (6.3, 3), arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
ax.add_patch(arrow2)

ax.text(2.5, 3.5, 'always', ha='center', fontsize=9, style='italic')
ax.text(5.5, 3.5, 'always', ha='center', fontsize=9, style='italic')

# Note
ax.text(5, 1, 'Converses do NOT hold in general\n(uniform ⇏ strong, strong ⇏ weak)',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_title('Hierarchy of Operator Convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('convergence_hierarchy.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Reflexivity Diagram
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 5)
ax.axis('off')

# Draw three layers
layer_y = [4, 2.5, 1]
labels = ['$X$', '$X^*$', '$X^{**}$']
colors_layers = [colors['primary'], colors['secondary'], colors['success']]

for i, (y, label, color) in enumerate(zip(layer_y, labels, colors_layers)):
    box = FancyBboxPatch((4, y-0.3), 2, 0.6, boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y, label, ha='center', va='center', fontsize=13, fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((5, 3.85), (5, 2.95), arrowstyle='<->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
ax.text(5.8, 3.4, 'duality pairing\n$(x, f) = f(x)$', fontsize=9)

arrow2 = FancyArrowPatch((5, 2.35), (5, 1.65), arrowstyle='<->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2)
ax.text(6.2, 2, 'natural embedding\n$\\varphi(x) = J_x$', fontsize=9)

# Reflexive vs non-reflexive
ax.text(1, 4, 'Reflexive:', fontsize=11, fontweight='bold', color=colors['success'])
ax.text(1, 3.5, '• $\\varphi$ is surjective\n• $J(X) = X^{**}$\n• $\\ell_p$ (1<p<∞)\n• Hilbert spaces', fontsize=9)

ax.text(1, 1.8, 'Non-Reflexive:', fontsize=11, fontweight='bold', color=colors['error'])
ax.text(1, 1.3, '• $\\varphi$ is not surjective\n• $\\ell_1, \\ell_\\infty, c_0, c$\n• $L_1(\\Omega), L_\\infty(\\Omega)$', fontsize=9)

ax.set_title('Natural Embedding and Reflexivity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reflexivity_diagram.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Weak Topology vs Norm Topology
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Norm topology (strong)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

circle1 = plt.Circle((0, 0), 0.5, color=colors['error'], fill=False, linewidth=2.5)
ax1.add_patch(circle1)
circle2 = plt.Circle((0, 0), 1, color=colors['error'], fill=False, linewidth=2, linestyle='--', alpha=0.6)
ax1.add_patch(circle2)
ax1.plot(0, 0, 'ko', markersize=8)
ax1.text(0.1, -0.4, 'Metric balls', fontsize=10, fontweight='bold')
ax1.text(0, -1.5, 'Norm topology\n(metric distance)', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Strong Topology', fontsize=12, fontweight='bold')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')

# Weak topology
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Draw weak neighborhoods (not metric balls)
rect1 = Rectangle((-0.3, -0.2), 0.6, 0.4, color=colors['primary'], fill=False, linewidth=2.5)
ax2.add_patch(rect1)
rect2 = Rectangle((-0.6, -0.4), 1.2, 0.8, color=colors['primary'], fill=False, linewidth=2, linestyle='--', alpha=0.6)
ax2.add_patch(rect2)
ax2.plot(0, 0, 'ko', markersize=8)
ax2.text(-0.1, 0.4, 'Level sets of\nfunctionals', fontsize=10, fontweight='bold')
ax2.text(0, -1.5, 'Weak topology\n(level sets of $f \\in X^*$)', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('Weak Topology', fontsize=12, fontweight='bold')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig('topology_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Hahn-Banach Theorem Illustration
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(-0.5, 10)
ax.set_ylim(-2, 4)

# Draw subspace and functional
ax.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
subspace_x = np.array([0, 8])
subspace_y = np.array([1, 3])
ax.plot(subspace_x, subspace_y, 'b-', linewidth=3, label='Subspace $Y$')
ax.plot([2, 6], [1.5, 2.5], 'ro', markersize=8)
ax.text(3, 1.2, '$f(y) = \\langle y, g \\rangle$', fontsize=10, color='red')

# Extended functional
extended_x = np.array([0, 8])
extended_y = np.array([0.5, 2.5])
ax.plot(extended_x, extended_y, 'g--', linewidth=2.5, label='Extension $F$')

# Level sets
for c in [-0.5, 0.5, 1.5, 2.5]:
    ax.axhline(y=c, color='green', alpha=0.2, linewidth=1)

# Hyperplane
ax.axhline(y=1.5, color='orange', linewidth=2.5, linestyle=':', label='Hyperplane (level set)')

ax.text(8.5, 3.5, '• $F(y) = f(y)$ for all $y \\in Y$\n• $\\|F\\|_* = \\|f\\|_*$\n• Preserves all properties',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 3.8)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=10)
ax.set_title('Hahn-Banach Theorem: Extension of Functionals', fontsize=13, fontweight='bold')
ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Functional Value')
plt.tight_layout()
plt.savefig('hahn_banach.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Compactness Relationships
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Compactness in Metric Spaces', fontsize=14, fontweight='bold', ha='center')

# Central box
main_box = FancyBboxPatch((3.5, 6.5), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2.5)
ax.add_patch(main_box)
ax.text(5, 7.1, 'Metric Space Compactness', ha='center', va='center', fontsize=11, fontweight='bold')

# Equivalences
equivalences = [
    (5, 5.5, 'Compact'),
    (2, 5.5, 'Every sequence has\nconvergent subsequence'),
    (8, 5.5, 'Totally bounded\n+ complete'),
]

for x, y, text in equivalences:
    if x == 5:
        box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8, boxstyle="round,pad=0.05",
                            edgecolor=colors['primary'], facecolor='white', linewidth=2)
    else:
        box = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1, boxstyle="round,pad=0.05",
                            edgecolor=colors['secondary'], facecolor='white', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9)

# Arrows from main box
arrow1 = FancyArrowPatch((5, 6.5), (5, 5.9), arrowstyle='<->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((4.2, 6.5), (2.5, 5.9), arrowstyle='<->', mutation_scale=20, linewidth=1.5, color='gray')
ax.add_patch(arrow2)
arrow3 = FancyArrowPatch((5.8, 6.5), (7.5, 5.9), arrowstyle='<->', mutation_scale=20, linewidth=1.5, color='gray')
ax.add_patch(arrow3)

# Properties
props = [
    (1.5, 3.5, '• Closed, bounded subset'),
    (5, 3.5, '• Continuous image'),
    (8.5, 3.5, '• Finite dimensional'),
]

for x, y, text in props:
    ax.text(x, y, text, fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Theorems
ax.text(5, 1.8, 'Key Theorems:', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 1.2, '• Heine-Borel: $\\mathbb{R}^n$ is compact iff closed and bounded',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
ax.text(5, 0.5, '• Kakutani: Unit ball in $X$ is weakly compact iff $X$ is reflexive',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

plt.tight_layout()
plt.savefig('compactness_relationships.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Example - Weak convergence in ℓ₂
fig, ax = plt.subplots(figsize=(11, 6))

# Sequence: e_n converges weakly to 0 but not strongly
n_values = np.arange(1, 21)
strong_norm = np.ones_like(n_values, dtype=float)  # ||e_n|| = 1 always
weak_decay = 1.0 / np.sqrt(n_values)  # Illustrative weak convergence pattern

ax.plot(n_values, strong_norm, 'o-', linewidth=2.5, markersize=8,
        label='$\\|x_n\\|_{\\ell_2} = 1$ (strong norm - no convergence)', color=colors['error'])
ax.plot(n_values, weak_decay, 's--', linewidth=2.5, markersize=7,
        label='$|\\langle x_n, f \\rangle| \\to 0$ (weak convergence)', color=colors['primary'])

ax.axhline(y=0, color='black', linewidth=0.5)
ax.fill_between(n_values, 0, weak_decay, alpha=0.2, color=colors['primary'])

ax.set_xlabel('$n$ (index)', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Example: Weak vs Strong Convergence in $\\ell_2$\n$x_n = (0, 0, \\ldots, 1, 0, \\ldots)$ (1 in $n$-th position)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig('weak_convergence_example.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All figures generated successfully!")
print("  - operator_norm.pdf")
print("  - convergence_hierarchy.pdf")
print("  - reflexivity_diagram.pdf")
print("  - topology_comparison.pdf")
print("  - hahn_banach.pdf")
print("  - compactness_relationships.pdf")
print("  - weak_convergence_example.pdf")
