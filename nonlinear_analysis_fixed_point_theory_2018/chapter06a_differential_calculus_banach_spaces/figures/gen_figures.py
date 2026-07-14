#!/usr/bin/env python3
"""
Generate figures for Chapter 6a: Differential Calculus in Banach Spaces
Illustrations of topological degree, homotopy, and related concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'primary': '#1f77b4', 'accent': '#ff7f0e', 'good': '#2ca02c', 'bad': '#d62728'}

def figure_brouwer_degree_intuition():
    """Illustration of Brouwer degree concept - winding number."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: Domain and range
    ax = axes[0]
    circle1 = Circle((0.5, 0.5), 0.3, fill=False, edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(circle1)
    ax.text(0.5, 0.5, r'$\Omega$', fontsize=14, ha='center', va='center', weight='bold')
    ax.arrow(0.8, 0.5, 0.15, 0, head_width=0.03, head_length=0.02, fc='black', ec='black')
    circle2 = Circle((1.2, 0.5), 0.25, fill=False, edgecolor=colors['accent'], linewidth=2)
    ax.add_patch(circle2)
    ax.text(1.2, 0.5, r'$f(\Omega)$', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(0.65, 0.35, r'$f: \Omega \to \mathbb{R}^n$', fontsize=11, ha='center')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Mapping from Domain to Range', fontsize=12, weight='bold')

    # Middle: Winding number
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 100)
    # Simple curve around origin
    x = 0.3 * np.cos(theta) + 0.5
    y = 0.3 * np.sin(theta) + 0.5
    ax.plot(x, y, color=colors['primary'], linewidth=2, label='Boundary path')
    ax.plot([0.5], [0.5], 'o', color=colors['accent'], markersize=8, label='Point y')
    ax.text(0.5, 0.2, r'$\text{deg}(f, \Omega, y) = 1$', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Winding Number / Degree', fontsize=12, weight='bold')

    # Right: Multiple windings
    ax = axes[2]
    theta = np.linspace(0, 4*np.pi, 200)
    x = 0.3 * np.cos(theta) + 0.5
    y = 0.3 * np.sin(theta) + 0.5
    ax.plot(x, y, color=colors['good'], linewidth=2, label='2 loops around y')
    ax.plot([0.5], [0.5], 'o', color=colors['accent'], markersize=8, label='Point y')
    ax.text(0.5, 0.2, r'$\text{deg}(f, \Omega, y) = 2$', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Higher Degree', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('figures/degree_intuition.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: figures/degree_intuition.pdf")

def figure_homotopy_concept():
    """Illustration of homotopy between mappings."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Time t=0
    ax = axes[0]
    circle1 = Circle((0.3, 0.5), 0.25, fill=False, edgecolor=colors['primary'], linewidth=2.5)
    ax.add_patch(circle1)
    circle2 = Circle((0.7, 0.5), 0.25, fill=False, edgecolor=colors['bad'], linewidth=2.5)
    ax.add_patch(circle2)
    ax.text(0.3, 0.5, r'$f_0$', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(0.7, 0.5, r'$f_0(\Omega)$', fontsize=11, ha='center', va='center', weight='bold')
    ax.text(0.5, 0.15, r'$t = 0$', fontsize=12, ha='center', weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Initial Map', fontsize=12, weight='bold')

    # Intermediate t=0.5
    ax = axes[1]
    circle1 = Circle((0.3, 0.5), 0.25, fill=False, edgecolor=colors['primary'],
                     linewidth=2.5, linestyle='--', alpha=0.6)
    ax.add_patch(circle1)
    circle2 = Circle((0.7, 0.5), 0.25, fill=False, edgecolor=colors['bad'],
                     linewidth=2.5, linestyle='--', alpha=0.6)
    ax.add_patch(circle2)
    circle3 = Circle((0.5, 0.6), 0.28, fill=False, edgecolor='purple', linewidth=2.5)
    ax.add_patch(circle3)
    ax.text(0.5, 0.6, r'$H(t,\cdot)$', fontsize=12, ha='center', va='center', weight='bold')
    ax.text(0.5, 0.15, r'$t = 0.5$', fontsize=12, ha='center', weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Homotopy (Deformation)', fontsize=12, weight='bold')

    # Final t=1
    ax = axes[2]
    circle1 = Circle((0.3, 0.5), 0.25, fill=False, edgecolor=colors['primary'], linewidth=2.5)
    ax.add_patch(circle1)
    circle2 = Circle((0.7, 0.5), 0.25, fill=False, edgecolor=colors['good'], linewidth=2.5)
    ax.add_patch(circle2)
    ax.text(0.3, 0.5, r'$f_1$', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(0.7, 0.5, r'$f_1(\Omega)$', fontsize=11, ha='center', va='center', weight='bold')
    ax.text(0.5, 0.15, r'$t = 1$', fontsize=12, ha='center', weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Final Map', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('homotopy_concept.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: homotopy_concept.pdf")

def figure_fixed_point_illustration():
    """Illustration of fixed point problem."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Graphical illustration
    ax = axes[0]
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, 'k--', linewidth=2, label=r'$y = x$ (identity)')
    y_curve = 0.3 + 0.4 * np.sin(2 * np.pi * x)
    ax.plot(x, y_curve, color=colors['primary'], linewidth=2.5, label=r'$y = f(x)$')

    # Intersection point (fixed point)
    fixed_point_idx = np.argmin(np.abs(y_curve - x))
    x_star = x[fixed_point_idx]
    y_star = x_star
    ax.plot(x_star, y_star, 'o', color=colors['accent'], markersize=10,
            label=r'Fixed point: $x^* = f(x^*)$', zorder=5)

    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Graphical Method: Fixed Point', fontsize=12, weight='bold')
    ax.set_aspect('equal')

    # Right: Schauder fixed point theorem illustration
    ax = axes[1]
    # Draw domain
    polygon = patches.Polygon([[0.1, 0.2], [0.9, 0.1], [0.8, 0.9], [0.2, 0.8]],
                              closed=True, fill=True, edgecolor=colors['primary'],
                              facecolor='lightblue', alpha=0.3, linewidth=2)
    ax.add_patch(polygon)

    # Some sample points in domain
    np.random.seed(42)
    sample_x = np.random.uniform(0.2, 0.8, 8)
    sample_y = np.random.uniform(0.2, 0.8, 8)
    ax.scatter(sample_x, sample_y, s=80, c=colors['primary'], zorder=5)

    # Fixed point
    ax.plot(0.5, 0.5, '*', color=colors['accent'], markersize=20,
            label=r'Fixed point $x^*$', zorder=6)

    ax.text(0.5, 0.05, r'Compact convex set $K$', fontsize=11, ha='center', weight='bold')
    ax.text(0.5, 0.98, r'Continuous map $f: K \to K$', fontsize=11, ha='center', weight='bold')
    ax.text(0.5, -0.15, r'$\Rightarrow \exists x^* \in K: x^* = f(x^*)$', fontsize=12, ha='center',
            weight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Schauder's Fixed Point Theorem", fontsize=12, weight='bold')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('fixed_point_illustration.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fixed_point_illustration.pdf")

def figure_leray_schauder_degree():
    """Illustration of Leray-Schauder degree in infinite dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Compact perturbation
    ax = axes[0]
    ax.text(0.5, 0.85, r'Leray-Schauder Degree', fontsize=13, ha='center', weight='bold')
    ax.text(0.5, 0.75, r'For map: $\Phi = I - F$ where $F$ is compact',
            fontsize=11, ha='center', style='italic')

    # Draw space representation
    rect = patches.Rectangle((0.1, 0.35), 0.8, 0.35, fill=True,
                            edgecolor=colors['primary'], facecolor='lightblue',
                            alpha=0.2, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.52, r'Banach Space $X$', fontsize=11, ha='center', weight='bold')

    # Components
    ax.text(0.15, 0.45, r'$I$', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['primary']))
    ax.text(0.5, 0.45, r'$-$', fontsize=14, ha='center', weight='bold')
    ax.text(0.85, 0.45, r'$F$', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['accent']))

    ax.text(0.5, 0.15, r'Maps to finite dimensional subspace', fontsize=10, ha='center')
    ax.text(0.5, 0.05, r'for degree computation', fontsize=10, ha='center', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Leray-Schauder Construction', fontsize=12, weight='bold')

    # Right: Properties
    ax = axes[1]
    props = [
        r'$\mathbf{1.}$ Existence of Solutions',
        r'If $\deg(I - F, \Omega, y) \neq 0$, then',
        r'$\exists x \in \Omega: x - Fx = y$',
        r'',
        r'$\mathbf{2.}$ Homotopy Invariance',
        r'Degree is invariant under continuous deformations',
        r'that don\'t change boundary behavior',
        r'',
        r'$\mathbf{3.}$ Infinite Dimensional Extension',
        r'Extends Brouwer degree to Banach spaces',
        r'with compact perturbations',
    ]

    y_pos = 0.95
    for prop in props:
        if prop == '':
            y_pos -= 0.05
        else:
            ax.text(0.05, y_pos, prop, fontsize=10, ha='left', va='top', family='monospace')
            y_pos -= 0.08

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Key Properties', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('leray_schauder_degree.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: leray_schauder_degree.pdf")

def figure_degree_properties():
    """Illustration of properties of degree."""
    fig = plt.figure(figsize=(12, 8))

    props = [
        ('Homotopy Invariance',
         r'$H: [0,1] \times \Omega \to \mathbb{R}^n$ continuous,',
         r'$H(t,x) \neq y$ on $\partial\Omega$ for all $t$',
         r'$\Rightarrow \deg(H(t,\cdot), \Omega, y)$ is constant in $t$'),

        ('Existence Property',
         r'$\deg(f, \Omega, y) \neq 0$',
         r'$\Rightarrow \exists x \in \Omega: f(x) = y$',
         r'(nonzero degree implies solution exists)'),

        ('Domain Additivity',
         r'$\Omega_1 \cap \Omega_2 = \emptyset$, both open in $\Omega$',
         r'$y \notin f(\overline{\Omega} \setminus (\Omega_1 \cup \Omega_2))$',
         r'$\Rightarrow \deg(f, \Omega, y) = \deg(f, \Omega_1, y) + \deg(f, \Omega_2, y)$'),

        ('Excision Property',
         r'$C \subseteq \Omega$ closed, $y \notin f(C)$',
         r'$\Rightarrow \deg(f, \Omega, y) = \deg(f, \Omega \setminus C, y)$',
         r'(degree unchanged by removing regions not containing solution)'),
    ]

    colors_list = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    for idx, (title, line1, line2, line3) in enumerate(props):
        ax = plt.subplot(2, 2, idx + 1)

        # Background box
        rect = patches.Rectangle((0.02, 0.05), 0.96, 0.90, fill=True,
                                edgecolor=colors['primary'], facecolor=colors_list[idx],
                                alpha=0.3, linewidth=2)
        ax.add_patch(rect)

        # Title
        ax.text(0.5, 0.90, title, fontsize=12, ha='center', va='top', weight='bold')

        # Content
        ax.text(0.5, 0.75, line1, fontsize=9.5, ha='center', va='top', family='monospace')
        ax.text(0.5, 0.60, line2, fontsize=9.5, ha='center', va='top', family='monospace')
        ax.text(0.5, 0.45, line3, fontsize=9.5, ha='center', va='top', family='monospace')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Fundamental Properties of the Degree', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('degree_properties.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: degree_properties.pdf")

def figure_application_periodic_ode():
    """Illustration of application to periodic ODE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ODE system
    ax = axes[0]
    ax.text(0.5, 0.95, 'Periodic ODE Application', fontsize=13, ha='center', weight='bold')

    content = [
        r"System: $\mathbf{x}'(t) = \mathbf{f}(t, \mathbf{x}(t))$",
        r"",
        r"Seek: $T$-periodic solutions",
        r"$\mathbf{x}(t+T) = \mathbf{x}(t)$",
        r"",
        r"Construct operator $S$:",
        r"$S[\mathbf{x}](t) = \mathbf{x}_0 + \int_0^t \mathbf{f}(s, \mathbf{x}(s)) ds$",
        r"",
        r"Fixed point of $S$ gives periodic solution",
    ]

    y_pos = 0.85
    for line in content:
        if line:
            ax.text(0.05, y_pos, line, fontsize=10, ha='left', va='top', family='monospace')
        y_pos -= 0.08

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Right: Degree application
    ax = axes[1]
    ax.text(0.5, 0.95, 'Degree Theory Application', fontsize=13, ha='center', weight='bold')

    content = [
        r"Use degree $\deg(I - S, B, 0)$",
        r"",
        r"Key step: Show $\deg(I - S, B, 0) \neq 0$",
        r"",
        r"Then by existence property:",
        r"$\exists \mathbf{x}^* \in B: \mathbf{x}^* = S[\mathbf{x}^*]$",
        r"",
        r"This $\mathbf{x}^*$ is a periodic solution",
        r"",
        r"Can verify existence without solving explicitly",
    ]

    y_pos = 0.85
    for line in content:
        if line:
            ax.text(0.05, y_pos, line, fontsize=10, ha='left', va='top', family='monospace')
        y_pos -= 0.08

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('application_periodic_ode.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: application_periodic_ode.pdf")

def main():
    """Generate all figures."""
    print("Generating figures for Chapter 6a...")
    figure_brouwer_degree_intuition()
    figure_homotopy_concept()
    figure_fixed_point_illustration()
    figure_leray_schauder_degree()
    figure_degree_properties()
    figure_application_periodic_ode()
    print("\nAll figures generated successfully!")

if __name__ == '__main__':
    main()
