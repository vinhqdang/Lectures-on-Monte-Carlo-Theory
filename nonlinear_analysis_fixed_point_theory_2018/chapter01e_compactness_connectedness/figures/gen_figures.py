#!/usr/bin/env python3
"""
Generate figures for Chapter 1e: Compactness & Connectedness
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality PDF output
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.5

# Color scheme
color_compact = '#1f77b4'
color_open = '#ff7f0e'
color_boundary = '#2ca02c'
color_interior = '#d62728'

# Figure 1: Compact vs Non-compact in R
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Compact: [0,1]
ax = axes[0]
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_aspect('equal')

# Draw interval [0,1]
ax.plot([0, 1], [0, 0], 'o-', color=color_compact, linewidth=3, markersize=8, label='$[0,1]$')
ax.text(0.5, -0.25, 'Compact: Closed and Bounded', ha='center', fontsize=11, weight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([])
ax.set_title('(a) Compact Set in $\\mathbb{R}$', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Non-compact: (0,1)
ax = axes[1]
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_aspect('equal')

# Draw open interval (0,1)
ax.plot([0, 1], [0, 0], 'o-', color=color_open, linewidth=3, markersize=8, markerfacecolor='white', label='$(0,1)$')
ax.text(0.5, -0.25, 'Not Compact: Open and Bounded', ha='center', fontsize=11, weight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([])
ax.set_title('(b) Non-compact Set in $\\mathbb{R}$', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/fig_compact_noncompact.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Open cover illustration
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compact case: [0,1] with finite subcover
ax = axes[0]
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')

# Draw interval
ax.plot([0, 1], [0, 0], 'o-', color=color_compact, linewidth=3, markersize=8)

# Draw open sets (finite subcover)
intervals = [(-0.2, 0.4, 'U1'), (0.3, 0.7, 'U2'), (0.6, 1.2, 'U3')]
colors_cover = ['#FF9999', '#99FF99', '#9999FF']
for i, (start, end, label) in enumerate(intervals):
    rect = FancyBboxPatch((start, -0.3), end-start, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor=colors_cover[i], facecolor=colors_cover[i],
                          alpha=0.3, linewidth=2)
    ax.add_patch(rect)
    ax.text((start+end)/2, 1, label, ha='center', fontsize=10, weight='bold')

ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.5, 1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([])
ax.set_title('(a) Compact Set: Finite Subcover Exists', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Non-compact case: (0,1) with infinite cover needed
ax = axes[1]
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')

# Draw open interval
ax.plot([0, 1], [0, 0], 'o-', color=color_open, linewidth=3, markersize=8, markerfacecolor='white')

# Draw many open sets showing infinite cover needed
n_sets = 8
for i in range(n_sets):
    center = (i + 0.5) / n_sets
    width = 0.15 / n_sets
    rect = FancyBboxPatch((center - width, -0.3), 2*width, 0.6,
                          boxstyle="round,pad=0.02",
                          edgecolor='gray', facecolor='gray',
                          alpha=0.2, linewidth=1)
    ax.add_patch(rect)

ax.text(0.5, 1, '...requires infinitely many...', ha='center', fontsize=10, style='italic')

ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.5, 1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([])
ax.set_title('(b) Non-compact Set: Infinite Cover Needed', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/fig_open_covers.pdf', bbox_inches='tight')
plt.close()

# Figure 3: Sequentially Compact illustration
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

# Draw unit circle (compact set)
circle = Circle((0, 0), 1, fill=False, edgecolor=color_compact, linewidth=3)
ax.add_patch(circle)

# Draw a sequence on the circle
angles = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(angles)
y_circle = np.sin(angles)
ax.plot(x_circle, y_circle, color=color_compact, linewidth=2.5, alpha=0.5)

# Mark some points of the sequence
n_points = 15
sequence_angles = np.linspace(0, 2*np.pi * 0.8, n_points)
for i, angle in enumerate(sequence_angles):
    x = np.cos(angle)
    y = np.sin(angle)
    ax.plot(x, y, 'o', color=color_interior, markersize=6, alpha=0.7)
    if i % 3 == 0:
        ax.text(x*1.2, y*1.2, f'$x_{i}$', ha='center', va='center', fontsize=9)

# Show convergent subsequence path
subseq_angles = np.linspace(0, np.pi/2, 5)
x_subseq = np.cos(subseq_angles)
y_subseq = np.sin(subseq_angles)
ax.plot(x_subseq, y_subseq, '--', color=color_boundary, linewidth=2, alpha=0.7, label='Convergent subsequence')

# Limit point
ax.plot(1, 0, '*', color=color_boundary, markersize=20, label='Limit point')

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x$', fontsize=11)
ax.set_ylabel('$y$', fontsize=11)
ax.set_title('Sequentially Compact: Every Sequence has Convergent Subsequence', fontsize=12, weight='bold')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_sequentially_compact.pdf', bbox_inches='tight')
plt.close()

# Figure 4: Weak compactness illustration
fig, ax = plt.subplots(figsize=(10, 6))

# Create a visualization of weak vs strong topology
from matplotlib.patches import Ellipse

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# Draw strong topology ball (small, tight)
strong_ball = Circle((0, 0), 0.7, fill=False, edgecolor=color_compact, linewidth=2.5, linestyle='-', label='Strong topology ball')
ax.add_patch(strong_ball)

# Draw weak topology ball (large, elliptical)
weak_ball = Ellipse((0, 0), 2.5, 1.8, fill=False, edgecolor=color_open, linewidth=2.5, linestyle='--', label='Weak topology ball')
ax.add_patch(weak_ball)

# Add text annotations
ax.text(0, 0.9, 'Strong\ntopology', ha='center', va='center', fontsize=10, weight='bold', color=color_compact)
ax.text(0, 1.3, 'Weak\ntopology', ha='center', va='center', fontsize=10, weight='bold', color=color_open)

# Add points
np.random.seed(42)
n_points = 20
for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    r = 0.5 + 0.1 * np.random.randn()
    x = r * np.cos(angle)
    y = 0.8 * r * np.sin(angle)
    ax.plot(x, y, 'o', color=color_interior, markersize=5, alpha=0.6)

ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.grid(True, alpha=0.2)
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.set_title('Weak Compactness: Weaker Topology => Larger Compact Sets', fontsize=12, weight='bold')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_weak_compactness.pdf', bbox_inches='tight')
plt.close()

# Figure 5: Continuity and compactness
fig, ax = plt.subplots(figsize=(10, 5))

# Draw a continuous function on compact domain
x_domain = np.linspace(0, 1, 300)
y_func = np.sin(5 * np.pi * x_domain) + 0.3 * np.cos(10 * np.pi * x_domain) + 1

ax.fill_between(x_domain, y_func - 0.2, y_func + 0.2, alpha=0.2, color=color_compact, label='Compact set + continuous map')
ax.plot(x_domain, y_func, linewidth=3, color=color_compact, label='$f: [0,1] \\to \\mathbb{R}$ (continuous)')

# Mark domain and range
ax.axvline(0, color=color_boundary, linestyle=':', alpha=0.7)
ax.axvline(1, color=color_boundary, linestyle=':', alpha=0.7)
y_min = y_func.min()
y_max = y_func.max()
ax.axhline(y_min, color=color_boundary, linestyle=':', alpha=0.7)
ax.axhline(y_max, color=color_boundary, linestyle=':', alpha=0.7)

# Add annotations
ax.text(0.5, y_min - 0.3, 'Compact domain: [0,1]', ha='center', fontsize=10, weight='bold', color=color_boundary)
ax.text(1.15, (y_min + y_max) / 2, f'Image:\n[{y_min:.1f}, {y_max:.1f}]', ha='left', fontsize=10, weight='bold', color=color_boundary)

ax.set_xlim(-0.1, 1.4)
ax.set_ylim(y_min - 0.5, y_max + 0.5)
ax.set_xlabel('$x$ (Domain)', fontsize=11)
ax.set_ylabel('$f(x)$ (Range)', fontsize=11)
ax.set_title('Continuous Image of Compact Set is Compact', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_continuous_compact.pdf', bbox_inches='tight')
plt.close()

# Figure 6: Compactness properties diagram
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Properties of Compact Sets', ha='center', fontsize=14, weight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Central property
props = [
    ('Closed and\nBounded', 'Heine-Borel Theorem\n(Complete Metric Space)', 1, 7.5),
    ('Sequential\nCompactness', 'Every sequence has\nconvergent subsequence', 4.5, 7.5),
    ('Finite Cover\nProperty', 'Every open cover has\nfinite subcover', 8, 7.5),
    ('Complete', 'Every bounded sequence\nhas convergent subseq.', 1, 4.5),
    ('Separable', 'Contains countable\ndense subset', 4.5, 4.5),
    ('Continuous Image', 'Image under continuous map\nis also compact', 8, 4.5),
]

for label, desc, x, y in props:
    # Draw box
    box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.4, boxstyle="round,pad=0.1",
                         edgecolor=color_compact, facecolor='lightyellow',
                         alpha=0.8, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.3, label, ha='center', va='center', fontsize=9, weight='bold')
    ax.text(x, y-0.4, desc, ha='center', va='center', fontsize=7.5, style='italic')

# Add central connection
ax.plot([5, 5], [7, 6.5], 'k-', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('figures/fig_compactness_properties.pdf', bbox_inches='tight')
plt.close()

# Figure 7: Mazur's theorem visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

# Draw a convex set (example: convex hull)
vertices = np.array([[0, 0], [1.5, -0.5], [2, 1], [1, 1.2], [0, 0.8]])
from matplotlib.patches import Polygon
polygon = Polygon(vertices, fill=True, alpha=0.2, edgecolor=color_compact,
                  facecolor='lightblue', linewidth=2.5, label='Closed convex hull of compact set')
ax.add_patch(polygon)

# Plot some points in the convex hull
np.random.seed(42)
for i in range(15):
    weights = np.random.dirichlet(np.ones(len(vertices)))
    point = np.average(vertices, axis=0, weights=weights)
    ax.plot(point[0], point[1], 'o', color=color_interior, markersize=6, alpha=0.7)

# Plot extreme points
ax.plot(vertices[:, 0], vertices[:, 1], 'o', color=color_boundary, markersize=10,
        label='Extreme points', zorder=5)

ax.set_xticks([-1, 0, 1, 2])
ax.set_yticks([-1, 0, 1])
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.set_title("Mazur's Theorem: Closed Convex Hull of Compact Set is Compact", fontsize=12, weight='bold')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig_mazur_theorem.pdf', bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Generated files:")
print("  - figures/fig_compact_noncompact.pdf")
print("  - figures/fig_open_covers.pdf")
print("  - figures/fig_sequentially_compact.pdf")
print("  - figures/fig_weak_compactness.pdf")
print("  - figures/fig_continuous_compact.pdf")
print("  - figures/fig_compactness_properties.pdf")
print("  - figures/fig_mazur_theorem.pdf")
