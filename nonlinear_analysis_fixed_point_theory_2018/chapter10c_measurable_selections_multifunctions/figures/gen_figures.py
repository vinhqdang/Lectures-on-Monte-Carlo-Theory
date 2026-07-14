#!/usr/bin/env python3
"""
Generate figures for Chapter 10c: Measurable Selections of Multifunctions
Visualization of key concepts from Pathak's book on Nonlinear Analysis and Fixed Point Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')

def set_theme(fig, ax):
    """Apply consistent theme to all figures"""
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    fig.patch.set_facecolor('white')

def save_pdf(filename):
    """Save current figure as PDF"""
    plt.savefig(f'figures/{filename}', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# ============================================================================
# Figure 1: Measurable Space Concept
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Draw the universal set Ω
omega_circle = Circle((0.5, 0.5), 0.4, color='lightblue', ec='blue', linewidth=2, alpha=0.5, zorder=1)
ax.add_patch(omega_circle)
ax.text(0.5, 0.92, r'$\Omega$ (Universal Set)', ha='center', fontsize=14, fontweight='bold')

# Draw some measurable sets
set_a = Polygon([[0.25, 0.45], [0.35, 0.25], [0.45, 0.35], [0.35, 0.55]],
                color='red', alpha=0.3, ec='red', linewidth=1.5, zorder=2)
ax.add_patch(set_a)
ax.text(0.35, 0.4, 'A', ha='center', fontsize=12, fontweight='bold')

set_b = Polygon([[0.55, 0.35], [0.65, 0.25], [0.75, 0.35], [0.65, 0.55]],
                color='green', alpha=0.3, ec='green', linewidth=1.5, zorder=2)
ax.add_patch(set_b)
ax.text(0.65, 0.4, 'B', ha='center', fontsize=12, fontweight='bold')

# Add σ-algebra text
sigma_text = (r"$\mathcal{A} = \sigma$-algebra" + "\n"
              r"Properties: $\emptyset, \Omega \in \mathcal{A}$" + "\n"
              r"Closed under complements, countable unions")
ax.text(0.5, 0.05, sigma_text, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
set_theme(fig, ax)
save_pdf('measurable_space.pdf')

# ============================================================================
# Figure 2: Multifunction Concept
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Domain X
x_circle = Circle((0.2, 0.5), 0.15, color='lightblue', ec='blue', linewidth=2, alpha=0.3, zorder=1)
ax.add_patch(x_circle)
ax.text(0.2, 0.75, r'Domain $X$', ha='center', fontsize=12, fontweight='bold')

# Codomain Y
y_circle = Circle((0.8, 0.5), 0.15, color='lightgreen', ec='green', linewidth=2, alpha=0.3, zorder=1)
ax.add_patch(y_circle)
ax.text(0.8, 0.75, r'Codomain $Y$', ha='center', fontsize=12, fontweight='bold')

# Points in X
x_points = [0.2, 0.2, 0.2]
y_points_x = [0.35, 0.5, 0.65]
for i, (xp, yp) in enumerate(zip(x_points, y_points_x)):
    ax.plot(xp, yp, 'bo', markersize=8)
    ax.text(xp - 0.08, yp, f'$x_{i+1}$', fontsize=10)

# Points in Y
y_points = [0.8, 0.8, 0.8, 0.8]
x_points_y = [0.35, 0.45, 0.55, 0.65]
for i, (xp, yp) in enumerate(zip(x_points_y, y_points)):
    ax.plot(xp, yp, 'go', markersize=8)
    ax.text(xp, yp - 0.08, f'$y_{i+1}$', fontsize=9, ha='center')

# Arrows showing multifunction mappings
ax.annotate('', xy=(0.73, 0.55), xytext=(0.27, 0.35),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
ax.annotate('', xy=(0.77, 0.50), xytext=(0.27, 0.50),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
ax.annotate('', xy=(0.77, 0.45), xytext=(0.27, 0.50),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
ax.annotate('', xy=(0.73, 0.45), xytext=(0.27, 0.65),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))

# Title and explanation
ax.text(0.5, 0.95, 'Multifunction: $F: X \\to \\mathcal{P}(Y)$',
        ha='center', fontsize=13, fontweight='bold')
ax.text(0.5, 0.15, 'Each point maps to a subset (possibly empty or multiple points)',
        ha='center', fontsize=11, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
set_theme(fig, ax)
save_pdf('multifunction.pdf')

# ============================================================================
# Figure 3: Selection Function Illustration
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Domain X
x_circle = Circle((0.2, 0.5), 0.15, color='lightblue', ec='blue', linewidth=2, alpha=0.3, zorder=1)
ax.add_patch(x_circle)
ax.text(0.2, 0.75, r'$X$', ha='center', fontsize=12, fontweight='bold')

# Codomain Y
y_circle = Circle((0.8, 0.5), 0.15, color='lightgreen', ec='green', linewidth=2, alpha=0.3, zorder=1)
ax.add_patch(y_circle)
ax.text(0.8, 0.75, r'$Y$', ha='center', fontsize=12, fontweight='bold')

# Points in X
x_points = [0.2, 0.2, 0.2]
y_points_x = [0.35, 0.5, 0.65]
for i, (xp, yp) in enumerate(zip(x_points, y_points_x)):
    ax.plot(xp, yp, 'bo', markersize=8, zorder=3)
    ax.text(xp - 0.08, yp, f'$x_{i+1}$', fontsize=10)

# Image sets (shown as intervals)
colors = ['red', 'orange', 'purple']
y_centers = [0.35, 0.5, 0.65]
for i, (yc, color) in enumerate(zip(y_centers, colors)):
    # Draw interval F(x_i)
    rect = Rectangle((0.65, yc - 0.08), 0.2, 0.16, fill=True,
                     facecolor=color, edgecolor=color, alpha=0.3, zorder=2)
    ax.add_patch(rect)
    ax.text(0.75, yc, f'$F(x_{i+1})$', fontsize=10, ha='center', fontweight='bold')

# Selection function - draw lines for selected points
selection_y = [0.32, 0.5, 0.62]
for i, (xp, yp, sy) in enumerate(zip(x_points, y_points_x, selection_y)):
    ax.plot([xp, 0.8], [yp, sy], 'g--', linewidth=2, alpha=0.7, zorder=2)
    ax.plot(0.8, sy, 'go', markersize=7)

ax.text(0.5, 0.15, 'Selection $s: X \\to Y$ where $s(x) \\in F(x)$ for all $x \\in X$',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
set_theme(fig, ax)
save_pdf('selection_function.pdf')

# ============================================================================
# Figure 4: Measurability of Multifunction
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Open set in Y
ax1.text(0.5, 0.95, 'Multifunction Measurability', ha='center', fontsize=13, fontweight='bold',
         transform=ax1.transAxes)

omega_circle = Circle((0.5, 0.5), 0.35, color='lightblue', ec='blue', linewidth=2, alpha=0.2, zorder=1)
ax1.add_patch(omega_circle)

# Draw some measurable sets in domain
for i in range(3):
    angle_start = i * 120
    rect = Polygon([[0.5, 0.5],
                    [0.5 + 0.3*np.cos(np.radians(angle_start)),
                     0.5 + 0.3*np.sin(np.radians(angle_start))],
                    [0.5 + 0.25*np.cos(np.radians(angle_start + 30)),
                     0.5 + 0.25*np.sin(np.radians(angle_start + 30))]],
                   color=['red', 'green', 'blue'][i], alpha=0.3, ec=['red', 'green', 'blue'][i],
                   linewidth=1.5, zorder=2)
    ax1.add_patch(rect)

ax1.text(0.5, 0.05, r'$F^{-1}(U) \in \mathcal{A}$ for open $U \subseteq Y$',
         ha='center', fontsize=11, transform=ax1.transAxes, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.axis('off')

# Right plot: Measurable selection condition
ax2.text(0.5, 0.95, 'Measurable Selection Condition', ha='center', fontsize=13, fontweight='bold',
         transform=ax2.transAxes)

# Show condition: graph of selection
x_vals = np.linspace(0.1, 0.9, 5)
y_vals = 0.2 + 0.5 * np.sin(x_vals * np.pi)

ax2.plot(x_vals, y_vals, 'go-', linewidth=2, markersize=8, label='Selection $s(x)$')
ax2.fill_between(x_vals, y_vals - 0.1, y_vals + 0.1, alpha=0.2, color='green',
                 label='Envelope of $F(x)$')
ax2.plot(x_vals, y_vals - 0.1, 'b--', linewidth=1.5, alpha=0.7)
ax2.plot(x_vals, y_vals + 0.1, 'b--', linewidth=1.5, alpha=0.7)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('$x$', fontsize=11)
ax2.set_ylabel('$s(x)$', fontsize=11)
ax2.set_title('Graph of Measurable Selection', fontsize=11)

set_theme(fig, ax2)
plt.tight_layout()
save_pdf('measurability_multifunction.pdf')

# ============================================================================
# Figure 5: Filipov Selection Theorem Illustration
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.text(0.5, 0.95, "Filipov's Selection Theorem", ha='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

# Draw conditions
conditions = [
    (0.5, 0.85, "1. $F: \\Omega \\times X \\to \\mathcal{P}(Y)$ measurable"),
    (0.5, 0.75, "2. $F(\\omega, \\cdot)$ has closed values for all $\\omega$"),
    (0.5, 0.65, "3. For $\\sigma$-finite measure space $(\\Omega, \\mathcal{A}, \\mu)$"),
]

colors_cond = ['lightblue', 'lightgreen', 'lightyellow']
for i, (x, y, text) in enumerate(conditions):
    ax.text(x, y, text, ha='center', fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors_cond[i], alpha=0.7))

# Arrow pointing to conclusion
ax.annotate('', xy=(0.5, 0.52), xytext=(0.5, 0.58),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='red'), xycoords='axes fraction')

# Conclusion
conclusion = (r"$\Rightarrow$ There exists a measurable selection $s: \Omega \to Y$" + "\n" +
              r"such that $s(\omega) \in F(\omega)$ for a.e. $\omega \in \Omega$")
ax.text(0.5, 0.42, conclusion, ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, linewidth=2))

# Application note
app_note = ("Applications: Existence of solutions to differential inclusions,\n"
            "integral equations, and optimal control problems")
ax.text(0.5, 0.1, app_note, ha='center', fontsize=10, style='italic',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
set_theme(fig, ax)
save_pdf('filipov_selection_theorem.pdf')

# ============================================================================
# Figure 6: Bochner Integral Concept
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.text(0.5, 0.95, "Bochner Integral in Banach Spaces", ha='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

# Timeline of simple functions
t_vals = np.linspace(0, 1, 100)
y_simple = np.where(t_vals < 0.33, 0.3, np.where(t_vals < 0.67, 0.6, 0.8))
y_actual = 0.3 + 0.4 * np.sin(t_vals * np.pi)

ax.plot(t_vals, y_simple, 'g-', linewidth=2.5, label='Simple function approximation')
ax.fill_between(t_vals, 0, y_simple, alpha=0.2, color='green')
ax.plot(t_vals, y_actual, 'b--', linewidth=2, label='Measurable function $u(t)$')
ax.plot(t_vals, y_actual, 'bo', markersize=4)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('$t$ (time domain)', fontsize=11, fontweight='bold')
ax.set_ylabel('Value in Banach space', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add definition box
def_box = (r"$\int_E u \, d\mu = \lim_{n \to \infty} \int_E u_n \, d\mu$" + "\n" +
           r"where $\{u_n\}$ is a sequence of simple functions")
ax.text(0.5, 0.1, def_box, ha='center', fontsize=11, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, linewidth=1.5))

set_theme(fig, ax)
plt.tight_layout()
save_pdf('bochner_integral.pdf')

# ============================================================================
# Figure 7: Measure Space Structure
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a flowchart-like diagram
ax.text(0.5, 0.95, "Measure Space: $(\\Omega, \\mathcal{A}, \\mu)$",
        ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

boxes = [
    (0.15, 0.80, r"$\Omega$ = Sample space", 'lightblue'),
    (0.50, 0.80, r"$\mathcal{A}$ = $\sigma$-algebra", 'lightgreen'),
    (0.85, 0.80, r"$\mu$ = Measure", 'lightyellow'),
    (0.25, 0.60, "Closure properties:\n$\emptyset, \Omega \in \mathcal{A}$\nComplements\nCountable unions", 'wheat'),
    (0.75, 0.60, "Countably additive:\n$\\mu(\\bigcup A_i) = \\sum \\mu(A_i)$", 'wheat'),
    (0.5, 0.35, "Measurable sets $A \in \mathcal{A}$", 'lightcyan'),
    (0.5, 0.15, "Measurable functions $f: \Omega \\to \\mathbb{R}$\n$f^{-1}(U) \in \mathcal{A}$ for open $U$", 'lightyellow'),
]

for x, y, text, color in boxes:
    bbox_props = dict(boxstyle='round', facecolor=color, alpha=0.7, linewidth=1.5)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=bbox_props)

# Draw arrows
arrow_pairs = [
    ((0.5, 0.75), (0.5, 0.45)),
    ((0.3, 0.55), (0.45, 0.40)),
    ((0.7, 0.55), (0.55, 0.40)),
]

for start, end in arrow_pairs:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
                xycoords='axes fraction')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
set_theme(fig, ax)
save_pdf('measure_space_structure.pdf')

print("All figures generated successfully!")
print("Generated files:")
print("  - measurable_space.pdf")
print("  - multifunction.pdf")
print("  - selection_function.pdf")
print("  - measurability_multifunction.pdf")
print("  - filipov_selection_theorem.pdf")
print("  - bochner_integral.pdf")
print("  - measure_space_structure.pdf")
