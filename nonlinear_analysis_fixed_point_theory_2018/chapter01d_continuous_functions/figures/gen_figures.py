#!/usr/bin/env python3
"""
Generate figures for Pathak Chapter 1d: Continuous Functions
Includes visualizations of continuous functions, operators, and key theorems
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============================================================================
# Figure 1: Continuous Functions in Different Spaces
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Continuous Functions in Different Spaces', fontsize=14, fontweight='bold')

# Subplot 1: Continuous function
x = np.linspace(-2*np.pi, 2*np.pi, 300)
y = np.sin(x)
axes[0, 0].plot(x, y, 'b-', linewidth=2, label='f(x) = sin(x)')
axes[0, 0].fill_between(x, y, alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Continuous Function: $f: \\mathbb{R} \\to \\mathbb{R}$', fontsize=11)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].legend()

# Subplot 2: Discontinuous function
x1 = np.linspace(-1, 0, 100)
x2 = np.linspace(0, 1, 100)
axes[0, 1].plot(x1, -np.ones_like(x1), 'r-', linewidth=2, label='Step function')
axes[0, 1].plot(x2, np.ones_like(x2), 'r-', linewidth=2)
axes[0, 1].plot(0, -1, 'ro', markersize=8)
axes[0, 1].plot(0, 1, 'r^', markersize=8)
axes[0, 1].set_ylim(-1.5, 1.5)
axes[0, 1].set_title('Discontinuous Function', fontsize=11)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('f(x)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Subplot 3: Smooth continuous function
x = np.linspace(-2, 2, 300)
y = np.exp(-x**2)
axes[1, 0].plot(x, y, 'g-', linewidth=2, label='$f(x) = e^{-x^2}$')
axes[1, 0].fill_between(x, y, alpha=0.3, color='green')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Smooth Continuous Function', fontsize=11)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].legend()

# Subplot 4: Uniformly continuous function
x = np.linspace(-1, 1, 300)
y = np.sqrt(np.abs(x))
axes[1, 1].plot(x, y, 'purple', linewidth=2, label='$f(x) = \\sqrt{|x|}$')
axes[1, 1].fill_between(x, y, alpha=0.3, color='purple')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Uniformly Continuous Function', fontsize=11)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('f(x)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('fig_continuous_functions.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_continuous_functions.pdf")
plt.close()

# ============================================================================
# Figure 2: Nemytski Operator (Superposition Operator)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Nemytski Operator: Superposition of Functions', fontsize=14, fontweight='bold')

# Left: Function f(s,u)
x = np.linspace(-3, 3, 100)
s_vals = [0, 1, 2]
colors = ['blue', 'green', 'red']

for s, color in zip(s_vals, colors):
    y = s + 0.5*x**2
    axes[0].plot(x, y, color=color, linewidth=2, label=f's = {s}')

axes[0].set_title('Function $f(s, u)$ for different s', fontsize=11)
axes[0].set_xlabel('u')
axes[0].set_ylabel('f(s, u)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Right: Input function x(s)
s = np.linspace(0, 10, 200)
x_s = 2*np.sin(s/3) + 0.5*s
axes[1].plot(s, x_s, 'b-', linewidth=2, label='$x(s)$')
axes[1].fill_between(s, x_s, alpha=0.3)
axes[1].set_title('Input Function $x(s)$', fontsize=11)
axes[1].set_xlabel('s')
axes[1].set_ylabel('x(s)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_nemytski_operator.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_nemytski_operator.pdf")
plt.close()

# ============================================================================
# Figure 3: Caratheodory Conditions
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Carathéodory Conditions Illustrated', fontsize=14, fontweight='bold')

# Condition 1: f(s,·) is continuous for almost all s
s_vals = [0.2, 0.5, 0.8]
u = np.linspace(-3, 3, 200)
for s in s_vals:
    y = s*np.sin(u) + 0.1*u**2
    axes[0, 0].plot(u, y, linewidth=2, label=f's={s}')
axes[0, 0].set_title('Condition 1: $f(s, \\cdot)$ continuous for a.e. $s$', fontsize=11)
axes[0, 0].set_xlabel('u')
axes[0, 0].set_ylabel('f(s, u)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Condition 2: f(·,u) is measurable for all u
s = np.linspace(0, 10, 200)
u_vals = [-2, 0, 2]
for u in u_vals:
    y = np.sin(s) + 0.2*u*s
    axes[0, 1].plot(s, y, linewidth=2, label=f'u={u}')
axes[0, 1].set_title('Condition 2: $f(\\cdot, u)$ measurable for all $u$', fontsize=11)
axes[0, 1].set_xlabel('s')
axes[0, 1].set_ylabel('f(s, u)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Condition 3: Growth condition |f(s,u)| ≤ a(s) + b|u|^p
u = np.linspace(-3, 3, 200)
s_fixed = 1
f_val = s_fixed + 0.3*u**2
a_s = s_fixed + 0.5
b_u_p = 0.5*np.abs(u)**1.5
upper = a_s + b_u_p

axes[1, 0].fill_between(u, -upper, upper, alpha=0.2, color='gray', label='Allowed region')
axes[1, 0].plot(u, f_val, 'b-', linewidth=2, label='|f(s,u)|')
axes[1, 0].plot(u, upper, 'r--', linewidth=2, label='$a(s) + b|u|^p$')
axes[1, 0].plot(u, -upper, 'r--', linewidth=2)
axes[1, 0].set_title('Condition 3: Growth $|f(s,u)| \\leq a(s) + b|u|^p$', fontsize=11)
axes[1, 0].set_xlabel('u')
axes[1, 0].set_ylabel('f(s, u)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Summary: All conditions
ax = axes[1, 1]
ax.axis('off')
conditions_text = """
Carathéodory Function: $f: \\Omega \\times \\mathbb{R} \\to \\mathbb{R}$

Conditions:
(1) $f(s, \\cdot)$ is continuous for almost all $s \\in \\Omega$

(2) $f(\\cdot, u)$ is measurable for all $u \\in \\mathbb{R}$

(3) Growth: $|f(s,u)| \\leq a(s) + b|u|^p$
    where $a \\in L_q(\\Omega)$, $b \\geq 0$
    and $\\frac{1}{p} + \\frac{1}{q} = 1$

Result: Superposition operator $F$ is continuous
from $L_p(\\Omega)$ to $L_q(\\Omega)$
"""
ax.text(0.05, 0.95, conditions_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('fig_caratheodory_conditions.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_caratheodory_conditions.pdf")
plt.close()

# ============================================================================
# Figure 4: Function Spaces
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create a hierarchy of function spaces
spaces = {
    'L∞(Ω)': {'pos': (0.5, 0.9), 'width': 0.2, 'height': 0.08},
    'C(Ω)': {'pos': (0.2, 0.7), 'width': 0.15, 'height': 0.08},
    'W^{1,p}(Ω)': {'pos': (0.8, 0.7), 'width': 0.15, 'height': 0.08},
    'L^p(Ω)': {'pos': (0.5, 0.5), 'width': 0.15, 'height': 0.08},
    'L^1(Ω)': {'pos': (0.2, 0.3), 'width': 0.15, 'height': 0.08},
    'Measurable': {'pos': (0.8, 0.3), 'width': 0.2, 'height': 0.08},
}

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Draw boxes for each space
for space, info in spaces.items():
    x, y = info['pos']
    w, h = info['width'], info['height']
    fancy_box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.01",
                               edgecolor='blue', facecolor='lightblue',
                               linewidth=2)
    ax.add_patch(fancy_box)
    ax.text(x, y, space, ha='center', va='center', fontsize=11, fontweight='bold')

# Draw arrows showing inclusions
arrows = [
    ((0.5, 0.86), (0.2, 0.74)),  # L∞ → C
    ((0.5, 0.86), (0.8, 0.74)),  # L∞ → W^{1,p}
    ((0.2, 0.66), (0.5, 0.54)),  # C → L^p
    ((0.8, 0.66), (0.5, 0.54)),  # W^{1,p} → L^p
    ((0.5, 0.46), (0.2, 0.34)),  # L^p → L^1
    ((0.5, 0.46), (0.8, 0.34)),  # L^p → Measurable
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->',
                           mutation_scale=20, linewidth=1.5, color='darkblue')
    ax.add_patch(arrow)

# Add title and description
ax.text(0.5, 0.98, 'Hierarchy of Function Spaces', ha='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.12, 'Continuous embeddings and relationships between spaces\nwhere functions become continuous in various senses',
        ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('fig_function_spaces.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_function_spaces.pdf")
plt.close()

# ============================================================================
# Figure 5: Uniform Continuity
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Uniform Continuity vs Pointwise Continuity', fontsize=14, fontweight='bold')

# Uniformly continuous function
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
axes[0].plot(x, y, 'b-', linewidth=2.5)
axes[0].set_title('Uniformly Continuous Function\n(on compact set)', fontsize=11)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)
axes[0].text(np.pi, -0.5, 'For every ε>0, ∃δ>0 such that\n|x-y|<δ ⟹ |f(x)-f(y)|<ε\nfor ALL x,y',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Non-uniformly continuous function
x = np.linspace(0.01, 2, 200)
y = 1/x
axes[1].plot(x, y, 'r-', linewidth=2.5)
axes[1].set_ylim(0, 100)
axes[1].set_title('Non-Uniformly Continuous Function\n(near singularity)', fontsize=11)
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].grid(True, alpha=0.3)
axes[1].text(1, 50, 'Pointwise continuous but\nnot uniformly continuous\n(δ depends on point)',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig('fig_uniform_continuity.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_uniform_continuity.pdf")
plt.close()

# ============================================================================
# Figure 6: Continuity and Compactness
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Continuous Functions on Compact Sets', fontsize=14, fontweight='bold')

# Continuous function on compact set
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
axes[0].fill_between(x, 0, y, alpha=0.3, color='blue')
axes[0].plot(x, y, 'b-', linewidth=2.5, label='f continuous on [0, 2π]')
max_idx = np.argmax(y)
min_idx = np.argmin(y)
axes[0].plot(x[max_idx], y[max_idx], 'g*', markersize=15, label='Maximum')
axes[0].plot(x[min_idx], y[min_idx], 'r*', markersize=15, label='Minimum')
axes[0].set_title('Continuous on Compact Set\n(attains max and min)', fontsize=11)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Key results
ax = axes[1]
ax.axis('off')
results_text = """
Key Theorem: Continuous Image of Compact Set

If $K$ is compact and $f: K \\to \\mathbb{R}$ is continuous,
then $f(K)$ is compact.

Consequences:
• $f$ is bounded on $K$
• $f$ attains its maximum and minimum
• $f$ is uniformly continuous on $K$

Example:
$f(x) = \\sin(x)$ on $[0, 2\\pi]$:
• Bounded: $f(x) \\in [-1, 1]$
• Max: 1 (at $x = \\pi/2$)
• Min: -1 (at $x = 3\\pi/2$)
• Uniformly continuous
"""
ax.text(0.05, 0.95, results_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_compactness_continuity.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_compactness_continuity.pdf")
plt.close()

# ============================================================================
# Figure 7: Sobolev Spaces
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Sobolev Spaces: Functions with Weak Derivatives', fontsize=14, fontweight='bold')

# W^{0,p} = L^p
x = np.linspace(-3, 3, 200)
y = np.abs(x)
axes[0, 0].fill_between(x, y, alpha=0.3)
axes[0, 0].plot(x, y, 'b-', linewidth=2)
axes[0, 0].set_title('$W^{0,p}(\\Omega) = L^p(\\Omega)$', fontsize=11)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].grid(True, alpha=0.3)

# W^{1,1} - continuous
x = np.linspace(-3, 3, 200)
y = np.abs(x)
axes[0, 1].fill_between(x, y, alpha=0.3)
axes[0, 1].plot(x, y, 'g-', linewidth=2, label='f(x) = |x|')
axes[0, 1].plot(x, np.ones_like(x), 'r--', linewidth=1.5, label="f'(x) (weak)")
axes[0, 1].set_title('$W^{1,p}(\\Omega)$: Functions with $L^p$ weak derivatives', fontsize=11)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('f(x) or f\'(x)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Embeddings
x = np.linspace(-1, 1, 200)
y = x**2
axes[1, 0].fill_between(x, y, alpha=0.3, color='purple')
axes[1, 0].plot(x, y, 'purple', linewidth=2)
axes[1, 0].set_title('$W^{1,p}(\\Omega) \\subset C(\\Omega)$ under conditions', fontsize=11)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].grid(True, alpha=0.3)

# Sobolev space properties
ax = axes[1, 1]
ax.axis('off')
sobolev_text = """
Sobolev Space $W^{k,p}(\\Omega)$:

Definition:
$W^{k,p}(\\Omega) = \\{u \\in L^p(\\Omega) : D^\\alpha u \\in L^p(\\Omega)$
$\\text{ for all } |\\alpha| \\leq k\\}$

Norm:
$\\|u\\|_{k,p} = \\left(\\sum_{|\\alpha| \\leq k} \\int_\\Omega |D^\\alpha u|^p dx\\right)^{1/p}$

Properties:
• Complete normed space (Banach space)
• Hilbert space when p=2 ($H^k$)
• Sobolev embedding theorem relates
  regularity to integrability
"""
ax.text(0.05, 0.95, sobolev_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_sobolev_spaces.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_sobolev_spaces.pdf")
plt.close()

print("\nAll figures generated successfully!")
