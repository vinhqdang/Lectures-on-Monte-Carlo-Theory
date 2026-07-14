#!/usr/bin/env python3
"""
Generate visualization figures for Chapter 2a: Fixed Point Concepts and Differential Calculus
Pathak: Introduction to Nonlinear Analysis and Fixed Point Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

def save_figure(fig, filename):
    """Save figure as PDF and PNG"""
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{filename}.png', format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)

# Figure 1: Gâteaux and Fréchet Derivatives - Intuition
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Gâteaux vs Fréchet Derivatives: Directional vs Full Differentiability',
             fontsize=14, fontweight='bold')

# Left: Gâteaux derivative (directional)
ax = axes[0]
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_title('Gâteaux Derivative (Directional)', fontweight='bold')

# Plot a curve
x = np.linspace(-1.5, 1.5, 100)
y = 0.5 * x**2 + 0.5
ax.plot(x, y, 'b-', linewidth=2.5, label='F(x)')

# Point of interest
ax.plot(0, 0.5, 'ro', markersize=10, label='Point x')

# Direction vector
ax.arrow(0, 0.5, 0.6, 0, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
ax.text(0.3, 0.1, 'Direction h', fontsize=11, ha='center', color='green', fontweight='bold')

# Line along direction
t_vals = np.linspace(0, 1, 50)
curve_dir = 0.5 + 0.5 * (0.6)**2 * t_vals
ax.plot(0.6 * t_vals, curve_dir, 'g--', linewidth=2, alpha=0.7, label='Gâteaux path')

ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('F(x)', fontsize=11)

# Right: Fréchet derivative (full neighborhoods)
ax = axes[1]
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_title('Fréchet Derivative (Full Differentiability)', fontweight='bold')

# Plot surface representation
x = np.linspace(-1.5, 1.5, 100)
y = 0.5 * x**2 + 0.5
ax.plot(x, y, 'b-', linewidth=2.5, label='F(x)')

# Point of interest
ax.plot(0, 0.5, 'ro', markersize=10, label='Point x')

# Tangent line (linear approximation)
tangent_x = np.linspace(-1.5, 1.5, 50)
tangent_y = np.full_like(tangent_x, 0.5)  # Since derivative at 0 is 0
ax.plot(tangent_x, tangent_y, 'r--', linewidth=2, label='Fréchet approximation')

# Highlight neighborhood
circle = Circle((0, 0.5), 0.8, fill=False, edgecolor='red', linestyle=':', linewidth=2)
ax.add_patch(circle)
ax.text(-0.6, 0.5, 'Neighborhood\nof x', fontsize=10, color='red', fontweight='bold')

ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('F(x)', fontsize=11)

plt.tight_layout()
save_figure(fig, 'fig_derivatives_intuition')

# Figure 2: Mean Value Theorem illustration
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Mean Value Theorem for Gâteaux Derivatives', fontsize=14, fontweight='bold')

# Function
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x) + 0.2*x

ax.plot(x, y, 'b-', linewidth=3, label='f(x + h) with f(x) = x + sin(x)')
ax.fill_between(x, y, alpha=0.1, color='blue')

# Points
x_start, x_end = np.pi/2, 3*np.pi/2
y_start = np.sin(x_start) + 0.2*x_start
y_end = np.sin(x_end) + 0.2*x_end

ax.plot([x_start, x_end], [y_start, y_end], 'ro', markersize=10, label='Endpoints')

# Secant line
x_secant = np.array([x_start, x_end])
y_secant = np.array([y_start, y_end])
slope = (y_end - y_start) / (x_end - x_start)
ax.plot(x_secant, y_secant, 'g--', linewidth=2.5, label=f'Secant line (slope={slope:.3f})')

# Find point where tangent is parallel to secant
x_middle = np.linspace(x_start, x_end, 100)
deriv = np.cos(x_middle) + 0.2
idx = np.argmin(np.abs(deriv - slope))
x_tau = x_middle[idx]
y_tau = np.sin(x_tau) + 0.2*x_tau

# Tangent line at τ
x_tangent = np.linspace(x_start, x_end, 100)
y_tangent = y_tau + slope * (x_tangent - x_tau)
ax.plot(x_tangent, y_tangent, 'r-', linewidth=2.5, label=f"Tangent at τ ∈ (x, x+h)")
ax.plot(x_tau, y_tau, 'r*', markersize=25)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_title('f(x+h) - f(x) = df(x+τh, h) for some τ ∈ (0,1)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
save_figure(fig, 'fig_mean_value_theorem')

# Figure 3: Norm spaces - Convexity comparison
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle('Unit Balls in Different Normed Spaces', fontsize=14, fontweight='bold')

# L2 norm (strictly convex)
ax = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 200)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
ax.plot(x_circle, y_circle, 'b-', linewidth=2.5)
ax.fill(x_circle, y_circle, alpha=0.1, color='blue')
ax.plot([0.7071, -0.7071], [0.7071, 0.7071], 'ro-', markersize=10, linewidth=2)
ax.plot([0], [1], 'go', markersize=10, label='Midpoint (strictly inside)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L² norm (Strictly Convex)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# L1 norm (not strictly convex)
ax = axes[0, 1]
l1_x = [1, 0, -1, 0, 1]
l1_y = [0, 1, 0, -1, 0]
ax.plot(l1_x, l1_y, 'b-', linewidth=2.5)
ax.fill(l1_x, l1_y, alpha=0.1, color='blue')
ax.plot([0.5, -0.5], [0.5, 0.5], 'ro-', markersize=10, linewidth=2)
ax.plot([0], [0.5], 'go', markersize=10, label='Midpoint (on boundary)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L¹ norm (Not Strictly Convex)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# L∞ norm (not strictly convex)
ax = axes[1, 0]
linf_x = [1, 1, -1, -1, 1]
linf_y = [1, -1, -1, 1, 1]
ax.plot(linf_x, linf_y, 'b-', linewidth=2.5)
ax.fill(linf_x, linf_y, alpha=0.1, color='blue')
ax.plot([0.5, -0.5], [0.5, 0.5], 'ro-', markersize=10, linewidth=2)
ax.plot([0], [0.5], 'go', markersize=10, label='Midpoint (on boundary)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('L∞ norm (Not Strictly Convex)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# C[0,1] space (not strictly convex)
ax = axes[1, 1]
ax.text(0.5, 0.7, 'C[0,1]: Space of continuous functions', ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.5, 'Unit ball: {f : ||f||∞ ≤ 1}', ha='center', fontsize=10,
        transform=ax.transAxes)
ax.text(0.5, 0.3, 'Example: f(t) = t and g(t) = -t\n' +
        'Both have ||f||∞ = ||g||∞ = 1\n' +
        'Midpoint: (f+g)/2 = 0 (strictly inside)', ha='center', fontsize=10,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
ax.text(0.5, 0.05, 'Not strictly convex', ha='center', fontsize=11, fontweight='bold', color='red',
        transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
save_figure(fig, 'fig_unit_balls')

# Figure 4: Hammerstein Operator
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle('Hammerstein Operator: [Fx](s) = ∫₀¹ k(s,t)f(t,x(t))dt',
             fontsize=13, fontweight='bold')

# Illustrate the kernel
s_vals = np.linspace(0, 1, 50)
t_vals = np.linspace(0, 1, 50)
S, T = np.meshgrid(s_vals, t_vals)

# Example kernel: k(s,t) = st
K = S * T

im = ax.contourf(S, T, K, levels=20, cmap='viridis')
ax.set_xlabel('t (integration variable)', fontsize=12, fontweight='bold')
ax.set_ylabel('s (output variable)', fontsize=12, fontweight='bold')
ax.set_title('Kernel k(s,t) = s·t (Example)', fontsize=12, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Kernel value k(s,t)', fontsize=11, fontweight='bold')

# Add annotations
ax.text(0.3, 0.85, 'Input space', fontsize=11, color='white', fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax.text(0.05, 0.5, 'Output\nspace', fontsize=11, color='white', fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5), rotation=90)

plt.tight_layout()
save_figure(fig, 'fig_hammerstein_operator')

# Figure 5: Chain Rule Composition
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle('Chain Rule for Derivatives: H = F ∘ G', fontsize=14, fontweight='bold')

# Draw spaces and operators
spaces = ['X', 'Y', 'Z']
positions = [0, 1, 2]

for pos, space in zip(positions, spaces):
    circle = Circle((pos, 0), 0.3, fill=False, edgecolor='black', linewidth=2.5)
    ax.add_patch(circle)
    ax.text(pos, 0, space, ha='center', va='center', fontsize=14, fontweight='bold')

# Draw operator arrows
ax.annotate('', xy=(0.7, 0), xytext=(0.3, 0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['primary']))
ax.text(0.5, 0.2, 'G', fontsize=12, fontweight='bold', ha='center', color=colors['primary'])

ax.annotate('', xy=(1.7, 0), xytext=(1.3, 0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['secondary']))
ax.text(1.5, 0.2, 'F', fontsize=12, fontweight='bold', ha='center', color=colors['secondary'])

# Composite operator
ax.annotate('', xy=(1.7, -0.6), xytext=(0.3, -0.6),
            arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['danger'], linestyle='--'))
ax.text(1.0, -0.8, 'H = F ∘ G', fontsize=12, fontweight='bold', ha='center', color=colors['danger'])

# Derivatives
ax.text(0.5, -1.3, "G'(x): X → Y", fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(1.5, -1.3, "F'(G(x)): Y → Z", fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(1.0, -2.0, "H'(x) = F'(G(x)) ∘ G'(x): X → Z", fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-2.5, 0.8)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
save_figure(fig, 'fig_chain_rule')

# Figure 6: Subdifferential illustration
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Subdifferential: Supporting Hyperplanes to Convex Functions', fontsize=14, fontweight='bold')

# Left: Smooth function (differential)
ax = axes[0]
x = np.linspace(-2, 2, 200)
f_x = np.abs(x)  # Absolute value function
ax.plot(x, f_x, 'b-', linewidth=3, label='f(x) = |x|')

x0 = 1.0
f_x0 = np.abs(x0)
ax.plot(x0, f_x0, 'ro', markersize=10, label='Point x₀')

# Supporting hyperplanes at x0
slopes = np.linspace(-1, 1, 5)
for slope in slopes:
    y_tangent = f_x0 + slope * (x - x0)
    ax.plot(x, y_tangent, 'g--', alpha=0.4, linewidth=1)

# Highlight one
slope_sel = 0.5
y_tang = f_x0 + slope_sel * (x - x0)
ax.plot(x, y_tang, 'r-', linewidth=2, label=f'Supporting hyperplane (slope=0.5)')

ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 3)
ax.set_xlabel('x', fontsize=11, fontweight='bold')
ax.set_ylabel('f(x)', fontsize=11, fontweight='bold')
ax.set_title('Non-smooth Point: Subdifferential has multiple slopes', fontweight='bold')
ax.legend(fontsize=10, loc='upper center')
ax.grid(True, alpha=0.3)

# Right: Text explanation
ax = axes[1]
ax.text(0.5, 0.95, 'Subdifferential Definition', ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

explanation = """At a point x₀, the subdifferential ∂f(x₀) is the set
of all vectors x* such that:

f(x) ≥ f(x₀) + ⟨x*, x - x₀⟩  ∀x ∈ X

Key Properties:
• For smooth functions: ∂f(x) = {f'(x)} (singleton)
• For non-smooth functions: ∂f(x) is a set
• For convex functions: always non-empty
• Generalization of gradient to non-smooth functions

Example: f(x) = |x| at x₀ = 0
∂f(0) = {z : |z| ≤ 1} = [-1, 1]
"""

ax.text(0.05, 0.7, explanation, ha='left', va='top', fontsize=10, family='monospace',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.axis('off')

plt.tight_layout()
save_figure(fig, 'fig_subdifferential')

# Figure 7: Operator properties hierarchy
fig, ax = plt.subplots(figsize=(11, 8))
fig.suptitle('Differential Calculus Concept Hierarchy', fontsize=14, fontweight='bold')

# Create a hierarchy diagram
y_pos = 8
x_center = 5

# Top level
rect1 = FancyBboxPatch((4.2, y_pos-0.5), 1.6, 0.8, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightblue', linewidth=2)
ax.add_patch(rect1)
ax.text(x_center, y_pos-0.1, 'Operators\nF: X → Y', ha='center', va='center', fontsize=11, fontweight='bold')

# Second level
y_pos -= 1.5
# Gâteaux
rect2a = FancyBboxPatch((1.5, y_pos-0.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightgreen', linewidth=2)
ax.add_patch(rect2a)
ax.text(2.75, y_pos-0.1, 'Gâteaux\nDifferentiable', ha='center', va='center', fontsize=10, fontweight='bold')

# Fréchet
rect2b = FancyBboxPatch((6.5, y_pos-0.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightcoral', linewidth=2)
ax.add_patch(rect2b)
ax.text(7.75, y_pos-0.1, 'Fréchet\nDifferentiable', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows down
ax.annotate('', xy=(2.75, y_pos-0.5), xytext=(4, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(7.75, y_pos-0.5), xytext=(6, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2))

# Third level
y_pos -= 1.5
# Properties of Gâteaux
prop_gateau = """• Directional derivative
• May not be continuous
• Used in variational
  analysis
• Examples: 3.1, 3.2"""
ax.text(2.75, y_pos+0.3, prop_gateau, ha='center', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Properties of Fréchet
prop_frechet = """• Full differentiability
• Implies continuity
• Stronger condition
• Standard analysis
  framework"""
ax.text(7.75, y_pos+0.3, prop_frechet, ha='center', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

# Bottom: Applications
y_pos -= 2.5
rect3 = FancyBboxPatch((1.5, y_pos-0.8), 7, 1.0, boxstyle="round,pad=0.1",
                       edgecolor='darkblue', facecolor='lightyellow', linewidth=2.5)
ax.add_patch(rect3)
ax.text(5, y_pos-0.2, 'Applications: Subdifferentials, Monotone Operators, Fixed Point Theorems, Variational Methods',
        ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 10)
ax.set_ylim(-1, 9)
ax.axis('off')

plt.tight_layout()
save_figure(fig, 'fig_concept_hierarchy')

print("✓ All figures generated successfully!")
print("  - fig_derivatives_intuition.pdf/png")
print("  - fig_mean_value_theorem.pdf/png")
print("  - fig_unit_balls.pdf/png")
print("  - fig_hammerstein_operator.pdf/png")
print("  - fig_chain_rule.pdf/png")
print("  - fig_subdifferential.pdf/png")
print("  - fig_concept_hierarchy.pdf/png")
