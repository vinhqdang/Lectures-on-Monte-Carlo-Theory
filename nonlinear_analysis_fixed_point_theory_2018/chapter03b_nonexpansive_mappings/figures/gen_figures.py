#!/usr/bin/env python3
"""
Generate figures for Chapter 3b: Nonexpansive Mappings
Illustrates key concepts: nonexpansive mappings, Lipschitz continuity,
and asymptotic behavior of generalized nonexpansive sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-darkgrid')
colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'accent': '#2ca02c', 'error': '#d62728'}

# ============================================================================
# Figure 1: Nonexpansive vs Expansive Mappings
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Nonexpansive mapping example
ax = axes[0]
x = np.linspace(-2, 2, 100)
# f(x) = 0.7*x is nonexpansive (Lipschitz constant L=0.7 < 1)
y_nonexp = 0.7 * x

ax.plot(x, x, 'k--', linewidth=1, label='y = x (identity)', alpha=0.5)
ax.plot(x, y_nonexp, linewidth=2.5, color=colors['primary'], label='f(x) = 0.7x (nonexpansive)')
ax.fill_between(x, y_nonexp - 0.1, y_nonexp + 0.1, alpha=0.15, color=colors['primary'])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Nonexpansive Mapping: ||f(x) - f(y)|| ≤ L||x - y||, L < 1', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_aspect('equal')

# Expansive mapping example
ax = axes[1]
# g(x) = 1.3*x is expansive (Lipschitz constant L=1.3 > 1)
y_exp = 1.3 * x

ax.plot(x, x, 'k--', linewidth=1, label='y = x (identity)', alpha=0.5)
ax.plot(x, y_exp, linewidth=2.5, color=colors['error'], label='g(x) = 1.3x (expansive)')
ax.fill_between(x, y_exp - 0.1, y_exp + 0.1, alpha=0.15, color=colors['error'])
ax.set_xlim(-2, 2)
ax.set_ylim(-3, 3)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('g(x)', fontsize=11)
ax.set_title('Expansive Mapping: ||g(x) - g(y)|| > ||x - y||', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_nonexpansive_mapping.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_nonexpansive_mapping.pdf")

# ============================================================================
# Figure 2: Convergence of Nonexpansive Sequence
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate a generalized nonexpansive sequence
np.random.seed(42)
n_terms = 30
# Define sequence with ||x_{n+1} - x_n|| <= ||x_n - x_{n-1}||
x_seq = np.zeros(n_terms)
x_seq[0] = 2.0
x_seq[1] = 1.5

for i in range(2, n_terms):
    # Generate next term with decreasing differences
    decay = 0.85 ** (i / 5)  # exponential decay factor
    diff = (x_seq[i-1] - x_seq[i-2]) * decay
    x_seq[i] = x_seq[i-1] + diff

# Plot the sequence values
ax.plot(range(n_terms), x_seq, 'o-', linewidth=2, markersize=8,
        color=colors['primary'], label='Sequence {x_n}')
ax.axhline(y=x_seq[-1], color=colors['accent'], linestyle='--', linewidth=2,
           label=f'Limit point ≈ {x_seq[-1]:.3f}')

# Plot differences ||x_{n+1} - x_n||
diffs = np.abs(np.diff(x_seq))
ax2 = ax.twinx()
ax2.semilogy(range(1, n_terms), diffs, 's--', linewidth=2, markersize=6,
             color=colors['secondary'], label='||x_{n+1} - x_n|| (log scale)', alpha=0.7)

ax.set_xlabel('n (iteration)', fontsize=12, fontweight='bold')
ax.set_ylabel('x_n', fontsize=12, fontweight='bold', color=colors['primary'])
ax2.set_ylabel('||x_{n+1} - x_n|| (log scale)', fontsize=12, fontweight='bold',
               color=colors['secondary'])
ax.set_title('Asymptotic Behavior of Generalized Nonexpansive Sequence',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='y', labelcolor=colors['primary'])
ax2.tick_params(axis='y', labelcolor=colors['secondary'])

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_nonexpansive_convergence.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_nonexpansive_convergence.pdf")

# ============================================================================
# Figure 3: Lipschitz Constant Illustration
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Two points and their images
x_vals = np.array([0.5, 2.0])
y_vals = 0.8 * x_vals  # Nonexpansive with L = 0.8

# Plot function
x_full = np.linspace(-0.5, 2.5, 100)
y_full = 0.8 * x_full
ax.plot(x_full, y_full, linewidth=2.5, color=colors['primary'], label='f(x) = 0.8x')

# Plot points and vertical/horizontal lines to show distances
ax.plot(x_vals, np.zeros_like(x_vals), 'o', markersize=10, color=colors['error'], label='Points on x-axis')
ax.plot(x_vals, y_vals, 'o', markersize=10, color=colors['secondary'], label='Images on f(x)')

# Distance in x-space
ax.annotate('', xy=(x_vals[1], -0.15), xytext=(x_vals[0], -0.15),
            arrowprops=dict(arrowstyle='<->', color=colors['error'], lw=2))
ax.text((x_vals[0] + x_vals[1]) / 2, -0.30, f'||x - y|| = {x_vals[1] - x_vals[0]:.1f}',
        ha='center', fontsize=11, color=colors['error'], fontweight='bold')

# Distance in f(x)-space
ax.annotate('', xy=(x_vals[1] + 0.15, y_vals[1]), xytext=(x_vals[1] + 0.15, y_vals[0]),
            arrowprops=dict(arrowstyle='<->', color=colors['secondary'], lw=2))
ax.text(x_vals[1] + 0.35, (y_vals[0] + y_vals[1]) / 2, f'||f(x) - f(y)|| = {y_vals[1] - y_vals[0]:.2f}',
        ha='left', fontsize=11, color=colors['secondary'], fontweight='bold')

# Cone representing Lipschitz constant
slope_lines = [-0.8, 0.8]
for slope in slope_lines:
    x_cone = np.linspace(-0.5, 2.5, 50)
    y_cone = y_vals[0] + slope * (x_cone - x_vals[0])
    ax.plot(x_cone, y_cone, 'k:', linewidth=1, alpha=0.3)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.0)
ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
ax.set_title('Lipschitz Continuity: ||f(x) - f(y)|| ≤ L||x - y||',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper left')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/fig_lipschitz_constant.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_lipschitz_constant.pdf")

# ============================================================================
# Figure 4: Fixed Point Property
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Nonexpansive mapping that has a fixed point
x = np.linspace(0, 4, 200)
f_x = 0.6 * x + 0.5  # Has fixed point at x* = 1.25

ax.plot(x, x, 'k-', linewidth=2, label='y = x (diagonal)', alpha=0.7)
ax.plot(x, f_x, linewidth=2.5, color=colors['primary'], label='f(x) = 0.6x + 0.5')

# Fixed point
fixed_point = 0.5 / (1 - 0.6)  # Solution to x = 0.6x + 0.5
ax.plot(fixed_point, fixed_point, 'r*', markersize=20,
        label=f'Fixed point x* = {fixed_point:.2f}', zorder=5)

# Illustration of iteration
x0 = 3.0
x_iter = [x0]
for _ in range(8):
    x_new = 0.6 * x_iter[-1] + 0.5
    x_iter.append(x_new)

for i in range(len(x_iter) - 1):
    ax.plot([x_iter[i], x_iter[i]], [x_iter[i], 0.6 * x_iter[i] + 0.5],
            'b--', alpha=0.5, linewidth=1)
    ax.plot([x_iter[i], 0.6 * x_iter[i] + 0.5],
            [0.6 * x_iter[i] + 0.5, 0.6 * x_iter[i] + 0.5], 'b--', alpha=0.5, linewidth=1)
    ax.plot(x_iter[i], x_iter[i], 'bo', markersize=4, alpha=0.6)

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('f(x) or y', fontsize=12, fontweight='bold')
ax.set_title('Fixed Point Property of Nonexpansive Mappings',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/fig_fixed_point.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_fixed_point.pdf")

# ============================================================================
# Figure 5: Almost Nonexpansive Sequences
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate an almost nonexpansive sequence: ||x_{n+1} - x_n|| <= ||x_n - x_{n-1}|| + epsilon(n,j)
np.random.seed(42)
n_terms = 25

# Sequence 1: Perfect nonexpansive (epsilon = 0)
x_seq1 = np.zeros(n_terms)
x_seq1[0] = 2.0
x_seq1[1] = 1.5
for i in range(2, n_terms):
    decay = 0.82 ** (i / 4)
    diff = (x_seq1[i-1] - x_seq1[i-2]) * decay
    x_seq1[i] = x_seq1[i-1] + diff

# Sequence 2: Almost nonexpansive (with small perturbations)
x_seq2 = x_seq1.copy()
eps_noise = 0.03 * np.random.randn(n_terms)
x_seq2 = x_seq2 + np.cumsum(eps_noise)

ax.plot(range(n_terms), x_seq1, 'o-', linewidth=2.5, markersize=7,
        color=colors['primary'], label='Generalized nonexpansive (||x_{n+1} - x_n|| ≤ ||x_n - x_{n-1}||)')
ax.plot(range(n_terms), x_seq2, 's--', linewidth=2.5, markersize=6,
        color=colors['secondary'], label='Almost nonexpansive (with ε(i,j) > 0)', alpha=0.8)

# Error bands
diffs1 = np.abs(np.diff(x_seq1))
diffs2 = np.abs(np.diff(x_seq2))
ax.fill_between(range(1, n_terms), diffs1 + 0.02, diffs1 - 0.02,
                alpha=0.1, color=colors['primary'])

ax.set_xlabel('n (iteration)', fontsize=12, fontweight='bold')
ax.set_ylabel('x_n', fontsize=12, fontweight='bold')
ax.set_title('Generalized vs. Almost Nonexpansive Sequences',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('figures/fig_almost_nonexpansive.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_almost_nonexpansive.pdf")

# ============================================================================
# Figure 6: Asymptotic Behavior - Mean Convergence
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Generalized nonexpansive sequence
np.random.seed(42)
n_terms = 40
x_seq = np.zeros(n_terms)
x_seq[0] = 3.0
x_seq[1] = 2.5

for i in range(2, n_terms):
    decay = 0.85 ** (i / 6)
    diff = (x_seq[i-1] - x_seq[i-2]) * decay
    x_seq[i] = x_seq[i-1] + diff

# Mean convergence: (1/n) * sum(x_i) -> limit point
means = np.array([np.mean(x_seq[:i+1]) for i in range(n_terms)])
limit_point = x_seq[-1]

ax1.plot(range(n_terms), x_seq, 'o-', linewidth=2, markersize=6,
         color=colors['primary'], label='Sequence {x_n}')
ax1.axhline(y=limit_point, color=colors['accent'], linestyle='--', linewidth=2,
            label=f'Limit point ≈ {limit_point:.3f}')
ax1.set_xlabel('n', fontsize=11, fontweight='bold')
ax1.set_ylabel('x_n', fontsize=11, fontweight='bold')
ax1.set_title('Generalized Nonexpansive Sequence', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Mean convergence rate
ax2.plot(range(n_terms), means, 's-', linewidth=2.5, markersize=6,
         color=colors['secondary'], label='Mean: (1/n)Σx_i')
ax2.axhline(y=limit_point, color=colors['accent'], linestyle='--', linewidth=2,
            label=f'Limit ≈ {limit_point:.3f}')
ax2.fill_between(range(n_terms), means - 0.05, means + 0.05,
                  alpha=0.1, color=colors['secondary'])
ax2.set_xlabel('n', fontsize=11, fontweight='bold')
ax2.set_ylabel('(1/n)Σ x_i', fontsize=11, fontweight='bold')
ax2.set_title('Mean Convergence (Cesàro Averaging)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_asymptotic_behavior.pdf', bbox_inches='tight', dpi=300)
print("Generated: fig_asymptotic_behavior.pdf")

print("\nAll figures generated successfully!")
