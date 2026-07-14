#!/usr/bin/env python3
"""
Generate figures for Chapter 10d: Multifunction Integral & Inclusions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# Set default font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def fig_multifunction_example():
    """Illustration of a multifunction F: X -> 2^Y"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Domain X
    ax.text(0.1, 0.8, r'Domain $X$', fontsize=12, fontweight='bold')
    x_points = np.array([0.15, 0.25, 0.35])
    ax.scatter(x_points, [0.7, 0.7, 0.7], s=100, color='blue', zorder=3)
    ax.text(0.15, 0.62, r'$x_1$', fontsize=11, ha='center')
    ax.text(0.25, 0.62, r'$x_2$', fontsize=11, ha='center')
    ax.text(0.35, 0.62, r'$x_3$', fontsize=11, ha='center')

    # Codomain Y
    ax.text(0.65, 0.8, r'Codomain $Y$', fontsize=12, fontweight='bold')
    y_points = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    ax.scatter(y_points, [0.7, 0.7, 0.7, 0.7, 0.7, 0.7], s=100, color='red', zorder=3)
    ax.text(np.mean(y_points[:2]), 0.62, r'$F(x_1)$', fontsize=10, ha='center', color='darkgreen', fontweight='bold')
    ax.text(np.mean(y_points[2:4]), 0.62, r'$F(x_2)$', fontsize=10, ha='center', color='darkgreen', fontweight='bold')
    ax.text(np.mean(y_points[4:]), 0.62, r'$F(x_3)$', fontsize=10, ha='center', color='darkgreen', fontweight='bold')

    # Arrows showing multivalued nature
    # F(x1) -> {y1, y2}
    ax.annotate('', xy=(0.7, 0.7), xytext=(0.15, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))
    ax.annotate('', xy=(0.6, 0.7), xytext=(0.15, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))

    # F(x2) -> {y3, y4}
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.25, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))
    ax.annotate('', xy=(0.4, 0.7), xytext=(0.25, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))

    # F(x3) -> {y5, y6}
    ax.annotate('', xy=(0.3, 0.7), xytext=(0.35, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))
    ax.annotate('', xy=(0.2, 0.7), xytext=(0.35, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))

    ax.text(0.5, 0.05, r'Multifunction $F: X \to 2^Y$ (set-valued map)',
            fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('multifunction_example.pdf', dpi=300, bbox_inches='tight')
    print("Generated: multifunction_example.pdf")
    plt.close()


def fig_selection_of_multifunction():
    """Illustration of selections of a multifunction"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Domain
    ax.text(0.1, 0.85, r'Domain $X = \mathbb{R}$', fontsize=12, fontweight='bold')
    x_vals = np.linspace(0.1, 0.4, 4)
    ax.scatter(x_vals, [0.75]*4, s=80, color='blue', zorder=3)

    # Codomain
    ax.text(0.65, 0.85, r'Sets in Codomain $Y = \mathbb{R}$', fontsize=12, fontweight='bold')

    # Draw some sets and selections
    y_positions = [0.7, 0.5, 0.3, 0.15]

    for i, (x_val, y_pos) in enumerate(zip(x_vals, y_positions)):
        # Draw multifunction value as an interval
        x_interval = [0.62, 0.82]
        height = 0.08
        rect = Rectangle((x_interval[0], y_pos - height/2),
                        x_interval[1] - x_interval[0], height,
                        linewidth=2, edgecolor='red', facecolor='pink', alpha=0.3)
        ax.add_patch(rect)

        # Draw selection as a point
        selection_y = y_pos - 0.02 + np.random.uniform(-0.03, 0.03)
        ax.scatter([0.72], [selection_y], s=50, color='green', marker='*', zorder=4)

        # Connect x to set and selection
        ax.annotate('', xy=(0.62, y_pos), xytext=(x_val, 0.75),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))
        ax.annotate('', xy=(0.72, selection_y), xytext=(x_val, 0.75),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.7))

    ax.text(0.5, 0.05, r'Selection $f: X \to Y$ where $f(x) \in F(x)$',
            fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('selection_of_multifunction.pdf', dpi=300, bbox_inches='tight')
    print("Generated: selection_of_multifunction.pdf")
    plt.close()


def fig_integral_multifunction():
    """Illustration of integral of a multifunction"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Parameter space
    ax.text(0.05, 0.95, r'Parameter space $\Omega$', fontsize=11, fontweight='bold')
    ax.add_patch(Rectangle((0.05, 0.75), 0.25, 0.15, linewidth=2,
                           edgecolor='blue', facecolor='lightblue', alpha=0.3))
    ax.text(0.175, 0.825, r'$\Omega$', fontsize=12)

    # Multifunction values
    ax.text(0.4, 0.95, r'Multifunction $F: \Omega \to 2^X$ (closed, convex)',
            fontsize=11, fontweight='bold')

    for i in range(4):
        y_pos = 0.85 - i*0.15
        omega_val = 0.1 + i*0.05
        # Draw set
        ellipse_width = 0.12
        ellipse_height = 0.06
        rect = Rectangle((0.4, y_pos - ellipse_height/2), ellipse_width, ellipse_height,
                        linewidth=1.5, edgecolor='red', facecolor='mistyrose', alpha=0.5)
        ax.add_patch(rect)
        ax.text(0.46, y_pos, r'$F(\omega_{})$'.format(i+1), fontsize=10)

        # Arrow from Omega to F(omega)
        ax.annotate('', xy=(0.4, y_pos), xytext=(0.175, 0.75 + np.random.uniform(-0.05, 0.05)),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))

    # Integral result
    ax.text(0.7, 0.95, r'Integral $\int_\Omega F(\omega) d\mu(\omega)$',
            fontsize=11, fontweight='bold')
    rect_result = Rectangle((0.7, 0.5), 0.25, 0.3, linewidth=2.5,
                           edgecolor='darkgreen', facecolor='lightgreen', alpha=0.3)
    ax.add_patch(rect_result)
    ax.text(0.825, 0.65, r'Convex set in $X$', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Arrows showing aggregation
    for i in range(4):
        y_pos = 0.85 - i*0.15
        ax.annotate('', xy=(0.7, 0.65), xytext=(0.52, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='darkgreen', alpha=0.6))

    ax.text(0.5, 0.1, r'Aggregation of set-valued functions: Aumann integral',
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('integral_multifunction.pdf', dpi=300, bbox_inches='tight')
    print("Generated: integral_multifunction.pdf")
    plt.close()


def fig_differential_inclusion():
    """Illustration of differential inclusion"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Problem statement
    ax.text(0.5, 0.95, r"Differential Inclusion Problem",
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ODE comparison
    ax.text(0.05, 0.85, r"Ordinary Differential Equation (ODE):", fontsize=11, fontweight='bold')
    ax.text(0.1, 0.80, r"$\frac{dx}{dt} = f(t, x(t))$", fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # DI
    ax.text(0.55, 0.85, r"Differential Inclusion (DI):", fontsize=11, fontweight='bold')
    ax.text(0.6, 0.80, r"$\frac{dx}{dt} \in F(t, x(t))$", fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Trajectory illustration
    ax.text(0.05, 0.70, "Single trajectory (ODE):", fontsize=10, fontweight='bold')
    t_vals = np.linspace(0, 1, 50)
    x_single = t_vals**2
    ax.plot(t_vals, 0.5 + 0.15*x_single, 'b-', linewidth=2.5, label='Unique solution')
    ax.scatter([0], [0.5], s=50, color='blue', zorder=3)

    ax.text(0.55, 0.70, "Solution set (DI):", fontsize=10, fontweight='bold')
    # Multiple possible trajectories
    for shift in np.linspace(-0.1, 0.1, 5):
        x_traj = t_vals**2 + shift
        ax.plot(t_vals, 0.5 + 0.15*x_traj, 'r-', linewidth=1.5, alpha=0.6)
    ax.scatter([0], [0.5], s=50, color='red', zorder=3)
    ax.text(0.8, 0.55, r'Multiple solutions', fontsize=9, ha='center', style='italic')

    # Key properties
    ax.text(0.05, 0.32, "Key Differences:", fontsize=11, fontweight='bold')
    properties = [
        r"• ODE: Single unique solution (under Lipschitz conditions)",
        r"• DI: Multiple possible solutions (solution set)",
        r"• DI: Includes uncertainties, delays, switching controls",
        r"• DI: Applications: optimal control, delay systems",
    ]

    for i, prop in enumerate(properties):
        ax.text(0.08, 0.28 - i*0.04, prop, fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('differential_inclusion.pdf', dpi=300, bbox_inches='tight')
    print("Generated: differential_inclusion.pdf")
    plt.close()


def fig_measurability_multifunction():
    """Illustration of measurability concepts for multifunctions"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Closed-valued multifunction
    ax[0].text(0.5, 0.95, "Closed-valued Multifunction", fontsize=11, fontweight='bold',
              ha='center', transform=ax[0].transAxes)
    ax[0].text(0.05, 0.85, r"$F(\omega)$ is closed for each $\omega \in \Omega$",
              fontsize=10, transform=ax[0].transAxes)

    for i in range(3):
        y_pos = 0.75 - i*0.25
        x_center = 0.3
        # Draw closed set (filled)
        circle = Circle((x_center, y_pos), 0.08, fill=True, edgecolor='red',
                       facecolor='lightcoral', linewidth=1.5, transform=ax[0].transAxes)
        ax[0].add_patch(circle)
        ax[0].text(x_center, y_pos, f'$F_{i+1}$', ha='center', va='center',
                  fontsize=9, transform=ax[0].transAxes)

    ax[0].text(0.5, 0.15, "Graph: $\\text{Gr}(F) = \\{(\omega, x): x \\in F(\omega)\\}$\nis closed in $\Omega \\times X$",
              fontsize=10, ha='center', transform=ax[0].transAxes,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].axis('off')

    # Right: Measurability types
    ax[1].text(0.5, 0.95, "Measurability Types", fontsize=11, fontweight='bold',
              ha='center', transform=ax[1].transAxes)

    types = [
        (r"Strong Measurability", r"$F$ is measurable\n(preimage of open sets)", 0.8),
        (r"Weak Measurability", r"Selections are measurable", 0.55),
        (r"$\mathcal{A}$-measurability", r"Closed graph and\nmeasurable projections", 0.30),
    ]

    for i, (title, desc, y_pos) in enumerate(types):
        ax[1].text(0.1, y_pos, title, fontsize=10, fontweight='bold',
                  transform=ax[1].transAxes)
        ax[1].text(0.15, y_pos - 0.08, desc, fontsize=9, style='italic',
                  transform=ax[1].transAxes, va='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig('measurability_multifunction.pdf', dpi=300, bbox_inches='tight')
    print("Generated: measurability_multifunction.pdf")
    plt.close()


def fig_existence_solutions():
    """Illustration of existence of solutions for inclusions"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Framework
    ax.text(0.5, 0.95, "Existence Theorems for Inclusions", fontsize=13, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Conditions
    y_pos = 0.88
    conditions = [
        (r"Condition 1: Compactness", r"Compact domain $\Omega$ or compact feasible set", 0.82),
        (r"Condition 2: Convexity", r"$F(\omega)$ is convex for all $\omega$", 0.72),
        (r"Condition 3: Measurability", r"$F$ has measurable selections", 0.62),
        (r"Condition 4: Growth condition", r"$|F(\omega)| \leq g(\omega)$ with $g$ integrable", 0.52),
    ]

    for title, desc, y in conditions:
        ax.text(0.05, y, title, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        ax.text(0.25, y, desc, fontsize=9, style='italic')

    # Main theorem
    ax.text(0.5, 0.42, r"Main Theorem (Existence Result)", fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.text(0.5, 0.35, r"If conditions 1-4 hold, then the inclusion", fontsize=10, ha='center')
    ax.text(0.5, 0.30, r"$\frac{dx}{dt} \in F(t, x(t))$", fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.text(0.5, 0.23, r"has at least one solution", fontsize=10, ha='center', style='italic')

    # Application areas
    ax.text(0.05, 0.15, "Applications:", fontsize=10, fontweight='bold')
    apps = [
        r"• Optimal control problems with constraints",
        r"• Delay differential equations",
        r"• Nonlinear partial differential equations",
        r"• Stochastic systems with uncertainties",
    ]
    for i, app in enumerate(apps):
        ax.text(0.08, 0.11 - i*0.03, app, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('existence_solutions.pdf', dpi=300, bbox_inches='tight')
    print("Generated: existence_solutions.pdf")
    plt.close()


def main():
    """Generate all figures"""
    print("Generating figures for Chapter 10d...")

    fig_multifunction_example()
    fig_selection_of_multifunction()
    fig_integral_multifunction()
    fig_differential_inclusion()
    fig_measurability_multifunction()
    fig_existence_solutions()

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
