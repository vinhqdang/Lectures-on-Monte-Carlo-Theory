#!/usr/bin/env python3
"""
Generate figures for Chapter 8: Applications of Monotone Operator Theory
Extracted from Pathak "An Introduction to Nonlinear Analysis and Fixed Point Theory"
Pages 641-664 covering Integral Equations and Applications
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import os

# Setup
script_dir = Path(__file__).parent
pdf_path = "/home/user/Lectures-on-Monte-Carlo-Theory/nonlinear_analysis_fixed_point_theory_2018/An Introduction to Nonlinear Analysis and Fixed Point Theory 2018.pdf"

# Open PDF
doc = fitz.open(pdf_path)

# Pages 641-664 (0-indexed: 640-663)
# Key pages to extract diagrams
key_pages = {
    640: "page_641_hammerstein_integral",   # Integral Equations intro
    642: "page_643_hammerstein_operator",   # Hammerstein Operator
    643: "page_644_angle_bounded",          # Angle Bounded Operators
    654: "page_655_generalized_hammerstein" # Generalized Hammerstein
}

# Extract key theorem pages as figures
for pdf_idx, fig_name in key_pages.items():
    page = doc[pdf_idx]
    # Crop to content area (exclude header/footer)
    rect = fitz.Rect(0.1*page.rect.width, 0.15*page.rect.height,
                     0.95*page.rect.width, 0.95*page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
    output_path = script_dir / f"{fig_name}.pdf"
    pix.save(f"/tmp/{fig_name}.png")
    os.system(f"convert /tmp/{fig_name}.png -quality 95 {output_path}")

doc.close()

# Create mathematical illustrations
plt.style.use('default')

# Figure 1: Hammerstein Integral Equation Diagram
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.text(0.5, 0.9, r'Hammerstein Integral Equation',
        fontsize=14, weight='bold', ha='center', transform=ax.transAxes)

# Show the functional spaces and operators
ax.text(0.05, 0.75, r'$\Omega \subseteq \mathbb{R}^n$ measurable', fontsize=11, transform=ax.transAxes)
ax.text(0.05, 0.65, r'$K: \Omega \times \Omega \to \mathbb{R}$ kernel function', fontsize=11, transform=ax.transAxes)
ax.text(0.05, 0.55, r'$f: \Omega \times \mathbb{R} \to \mathbb{R}$ Carathéodory', fontsize=11, transform=ax.transAxes)

# Equation
eq_text = r'$x(s) + \int_{\Omega} k(s,t) f(t, x(t)) dt = y(s)$'
ax.text(0.5, 0.35, eq_text, fontsize=13, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        transform=ax.transAxes)

# Operators
ax.text(0.05, 0.15, r'$[Kx](s) = \int_{\Omega} k(s,t)x(t)dt$ (linear operator)',
        fontsize=11, transform=ax.transAxes)
ax.text(0.05, 0.05, r'$[N_f x](s) = f(s, x(s))$ (Nemytskii operator)',
        fontsize=11, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(script_dir / 'hammerstein_equation.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Operator Equation Reduction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Original equation
ax1.text(0.5, 0.9, 'Integral Equation', fontsize=12, weight='bold',
         ha='center', transform=ax1.transAxes)
ax1.add_patch(patches.FancyBboxPatch((0.05, 0.65), 0.9, 0.2,
              boxstyle="round,pad=0.01", transform=ax1.transAxes,
              edgecolor='blue', facecolor='lightblue', alpha=0.3))
ax1.text(0.5, 0.75, r'$x(s) + \int_{\Omega} k(s,t) f(t, x(t)) dt = y(s)$',
         fontsize=11, ha='center', transform=ax1.transAxes)

ax1.arrow(0.5, 0.6, 0, -0.15, head_width=0.1, head_length=0.05,
         fc='black', ec='black', transform=ax1.transAxes)
ax1.text(0.65, 0.5, 'Reduction to\nOperator Form', fontsize=10,
         transform=ax1.transAxes, style='italic')

ax1.add_patch(patches.FancyBboxPatch((0.05, 0.15), 0.9, 0.2,
              boxstyle="round,pad=0.01", transform=ax1.transAxes,
              edgecolor='green', facecolor='lightgreen', alpha=0.3))
ax1.text(0.5, 0.25, r'$x + K N_f x = y$',
         fontsize=12, ha='center', transform=ax1.transAxes, weight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Operator equations
ax2.text(0.5, 0.9, 'Operator Equations', fontsize=12, weight='bold',
         ha='center', transform=ax2.transAxes)

equations = [
    r'Linear: $[Kx](s) = \int_{\Omega} k(s,t)x(t)dt$',
    r'Nonlinear: $[N_f x](s) = f(s, x(s))$',
    r'Composed: $x + K N_f x = y$'
]

y_pos = 0.75
for eq in equations:
    ax2.text(0.05, y_pos, eq, fontsize=11, transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    y_pos -= 0.2

ax2.text(0.05, 0.1, 'Solve using operator theoretic methods:',
         fontsize=10, weight='bold', transform=ax2.transAxes, style='italic')
ax2.text(0.05, 0.02, 'monotone operators, fixed points, Banach spaces',
         fontsize=9, transform=ax2.transAxes)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
plt.savefig(script_dir / 'operator_reduction.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Angle Bounded Operators
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.text(0.5, 0.95, 'Angle Bounded Linear Operators', fontsize=13, weight='bold',
        ha='center', transform=ax.transAxes)

definition = [
    r'Definition: $K: X \to X^*$ is angle bounded with constant $c \geq 0$ if:',
    r'',
    r'$|(Kx_1, x_2) - (Kx_2, x_1)| \leq 2c\sqrt{(Kx_1,x_1)(Kx_2,x_2)}$'
]

y_pos = 0.85
for line in definition:
    if line:
        ax.text(0.5, y_pos, line, fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.6),
               transform=ax.transAxes)
    y_pos -= 0.12

properties = [
    r'Properties:',
    r'$\bullet$ Monotone: $(Kx,x) \geq 0$ for all $x \in X$',
    r'$\bullet$ Symmetric: $K^* = K$',
    r'$\bullet$ Range in sector: numerical range in a sector',
    r'$\bullet$ Important for: variational methods, PDE theory'
]

y_pos = 0.55
for prop in properties:
    fontweight = 'bold' if prop == 'Properties:' else 'normal'
    ax.text(0.05, y_pos, prop, fontsize=10, weight=fontweight,
           transform=ax.transAxes)
    y_pos -= 0.1

examples = [
    r'Examples:',
    r'$\bullet$ Linear differential operators',
    r'$\bullet$ Convolution operators',
    r'$\bullet$ Integral operators with symmetric kernels'
]

y_pos = 0.25
for ex in examples:
    fontweight = 'bold' if ex == 'Examples:' else 'normal'
    ax.text(0.05, y_pos, ex, fontsize=10, weight=fontweight,
           transform=ax.transAxes)
    y_pos -= 0.08

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(script_dir / 'angle_bounded.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Urysohn's Equation
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.text(0.5, 0.9, "Urysohn's Integral Equation", fontsize=13, weight='bold',
        ha='center', transform=ax.transAxes)

eq_text = r"$x(s) + \int_{\Omega} k(s, t, x(t)) dt = y(s)$"
ax.text(0.5, 0.75, eq_text, fontsize=12, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
       transform=ax.transAxes)

ax.text(0.05, 0.6, r'Generalization: kernel depends on solution', fontsize=11,
       transform=ax.transAxes, style='italic')
ax.text(0.05, 0.5, r'$k = k(s, t, x)$ (not just $k(s,t)$)', fontsize=11,
       transform=ax.transAxes)

features = [
    r'Key Features:',
    r'$\bullet$ Nonlinear integral equation',
    r'$\bullet$ Kernel depends on solution value',
    r'$\bullet$ Reduces to Hammerstein for $k = k(s,t) \cdot f(t,x(t))$',
    r'$\bullet$ Applications: physics, mechanics, biology'
]

y_pos = 0.4
for feat in features:
    fontweight = 'bold' if feat == 'Key Features:' else 'normal'
    ax.text(0.05, y_pos, feat, fontsize=10, weight=fontweight,
           transform=ax.transAxes)
    y_pos -= 0.08

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(script_dir / 'urysohn_equation.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Monotone Operators Framework
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

ax.text(0.5, 0.95, 'Monotone Operators Framework', fontsize=13, weight='bold',
        ha='center', transform=ax.transAxes)

framework = [
    ('Hilbert Space $\mathcal{H}$', 0.85),
    (r'$\downarrow$', 0.78),
    ('Linear Operator $K: \mathcal{H} \to \mathcal{H}$', 0.71),
    ('Strongly Monotone', 0.68),
    (r'$\downarrow$', 0.61),
    ('Nonlinear Operator $N_f: \mathcal{H} \to \mathcal{H}$', 0.54),
    ('Monotone, Bounded', 0.51),
    (r'$\downarrow$', 0.44),
    (r'Composite: $x + K N_f x = y$', 0.37),
    (r'$\downarrow$', 0.30),
    (r'Theorem 8.8: Unique Solution', 0.23),
    (r'Solution $x$ depends continuously on $y$', 0.16),
]

for text, y_pos in framework:
    if text.startswith(r'$\downarrow'):
        ax.text(0.5, y_pos, text, fontsize=14, ha='center',
               transform=ax.transAxes, weight='bold', color='red')
    elif 'Theorem' in text or 'Solution' in text:
        ax.text(0.5, y_pos, text, fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
               transform=ax.transAxes, weight='bold')
    else:
        ax.text(0.5, y_pos, text, fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(script_dir / 'monotone_framework.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Figure generation complete!")
print("Generated figures in:", script_dir)
