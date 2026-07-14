#!/usr/bin/env python3
"""
Generate figures for Variational Inequality Problems
Chapter 8d: Mappings Associated with Variational Inequality
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Polygon, Circle, Arc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'main': '#1f77b4',
    'highlight': '#ff7f0e',
    'accent': '#2ca02c',
    'text': '#1a1a1a',
}

def save_figure(fig, filename):
    """Save figure to PDF"""
    filepath = f"{filename}"
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)

def fig_projection_operator():
    """Figure 7.1: Projection Operator"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Draw convex set K
    circle = Circle((2.5, 2.5), 1.0, fill=True, alpha=0.2,
                    edgecolor=colors['main'], linewidth=2, color=colors['main'])
    ax.add_patch(circle)
    ax.text(2.5, 2.5, r'$K$', fontsize=14, ha='center', va='center',
            weight='bold', color=colors['text'])

    # Draw point z outside K
    z_x, z_y = 0.5, 2.5
    ax.plot(z_x, z_y, 'o', markersize=8, color=colors['highlight'])
    ax.text(z_x - 0.2, z_y - 0.25, r'$z$', fontsize=12, color=colors['highlight'])

    # Draw projection P_K(z)
    proj_x, proj_y = 1.5, 2.5
    ax.plot(proj_x, proj_y, 's', markersize=8, color=colors['accent'])
    ax.text(proj_x, proj_y + 0.25, r'$P_K(z)$', fontsize=12, color=colors['accent'])

    # Draw line from z to P_K(z)
    ax.arrow(z_x + 0.1, z_y, proj_x - z_x - 0.2, 0,
            head_width=0.15, head_length=0.1, fc=colors['text'], ec=colors['text'])

    # Draw angle theta
    angle_arc = Arc((proj_x, proj_y), 0.5, 0.5, angle=0, theta1=140, theta2=180,
                    linewidth=1.5, color=colors['text'])
    ax.add_patch(angle_arc)
    ax.text(proj_x - 0.4, proj_y + 0.15, r'$\theta$', fontsize=11, color=colors['text'])

    # Draw perpendicular indication
    perp_size = 0.15
    perp_corner = patches.Rectangle((proj_x - perp_size, proj_y - perp_size),
                                    perp_size, perp_size,
                                    fill=False, edgecolor=colors['text'], linewidth=1)
    ax.add_patch(perp_corner)

    # Labels and formatting
    ax.text(2.5, 0.3, r'$P_K(z)$ is the closest point in $K$ to $z$',
            fontsize=11, ha='center', style='italic', color=colors['text'])
    ax.text(2.5, -0.2, r'$(z - P_K(z))^T(x - P_K(z)) \geq 0$ for all $x \in K$',
            fontsize=10, ha='center', style='italic', color=colors['text'], family='monospace')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    save_figure(fig, 'fig_projection_operator.pdf')

def fig_normal_cone():
    """Figure 7.2: Normal Cone and Variational Inequality"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Draw convex set K (polygon)
    K_vertices = np.array([[0.5, 1], [2, 0.5], [3, 2], [2.5, 3], [1, 2.8], [0.5, 2.5]])
    polygon_K = Polygon(K_vertices, alpha=0.25, edgecolor=colors['main'],
                       facecolor=colors['main'], linewidth=2.5)
    ax.add_patch(polygon_K)
    ax.text(1.5, 2, r'$K$', fontsize=14, ha='center', va='center',
           weight='bold', color=colors['text'])

    # Draw point x on boundary of K
    x_point = np.array([1.5, 3])
    ax.plot(x_point[0], x_point[1], 'o', markersize=10, color=colors['accent'], zorder=10)
    ax.text(x_point[0] - 0.25, x_point[1] + 0.25, r'$\bar{x}$', fontsize=12,
           color=colors['accent'], weight='bold')

    # Draw normal cone vectors
    normal_vectors = [
        np.array([0, 0.6]),      # upward normal
        np.array([0.3, 0.4]),    # upper-right normal
        np.array([-0.3, 0.4])    # upper-left normal
    ]

    for i, n_vec in enumerate(normal_vectors):
        # Draw vector
        ax.arrow(x_point[0], x_point[1], n_vec[0] * 0.8, n_vec[1] * 0.8,
                head_width=0.12, head_length=0.08, fc=colors['main'],
                ec=colors['main'], alpha=0.7, linewidth=1.5)

    ax.text(x_point[0] + 0.3, x_point[1] + 1.2, r'$N_K(\bar{x})$', fontsize=11,
           color=colors['main'], style='italic')

    # Draw F(x) vector
    f_vec = np.array([-0.8, 0.2])
    ax.arrow(x_point[0], x_point[1], f_vec[0], f_vec[1],
            head_width=0.12, head_length=0.08, fc=colors['highlight'],
            ec=colors['highlight'], linewidth=2)
    ax.text(x_point[0] - 1.1, x_point[1] + 0.3, r'$-F(\bar{x})$', fontsize=11,
           color=colors['highlight'], weight='bold')

    # Add equation
    eqn_text = (r'VIP: $(x - \bar{x})^T F(\bar{x}) \geq 0$ for all $x \in K$'
                '\n' + r'Equivalently: $0 \in F(\bar{x}) + N_K(\bar{x})$')
    ax.text(1.5, 0, eqn_text, fontsize=10, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           family='monospace')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.8, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    save_figure(fig, 'fig_normal_cone.pdf')

def fig_fixed_point_equivalence():
    """Figure showing equivalence between VIP and fixed point problem"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Left box: Variational Inequality Problem
    vip_box = patches.FancyBboxPatch((0.1, 2), 2.2, 1.5,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=colors['main'], facecolor='lightblue',
                                     linewidth=2, alpha=0.7)
    ax.add_patch(vip_box)
    ax.text(1.2, 3.2, 'Variational Inequality', fontsize=11, ha='center', weight='bold')
    ax.text(1.2, 2.8, r'Find $\bar{x} \in K$ such that', fontsize=9, ha='center')
    ax.text(1.2, 2.4, r'$(x-\bar{x})^T F(\bar{x}) \geq 0$', fontsize=9, ha='center', family='monospace')
    ax.text(1.2, 2.05, r'$\forall x \in K$', fontsize=9, ha='center')

    # Right box: Fixed Point Problem
    fp_box = patches.FancyBboxPatch((5.7, 2), 2.2, 1.5,
                                    boxstyle="round,pad=0.1",
                                    edgecolor=colors['accent'], facecolor='lightgreen',
                                    linewidth=2, alpha=0.7)
    ax.add_patch(fp_box)
    ax.text(6.8, 3.2, 'Fixed Point Problem', fontsize=11, ha='center', weight='bold')
    ax.text(6.8, 2.8, r'Find $\bar{x} \in K$ such that', fontsize=9, ha='center')
    ax.text(6.8, 2.4, r'$\bar{x} = P_K(\bar{x} - \alpha F(\bar{x}))$', fontsize=9, ha='center', family='monospace')
    ax.text(6.8, 2.05, r'$\alpha > 0$', fontsize=9, ha='center')

    # Arrow between boxes
    arrow = FancyArrowPatch((2.35, 2.75), (5.65, 2.75),
                           arrowstyle='<->', mutation_scale=25,
                           linewidth=2, color=colors['text'])
    ax.add_patch(arrow)
    ax.text(4, 3.05, 'Equivalent', fontsize=10, ha='center', style='italic', weight='bold')

    # Bottom: Normal equation form
    normal_box = patches.FancyBboxPatch((1, 0.2), 5.8, 1.2,
                                        boxstyle="round,pad=0.1",
                                        edgecolor=colors['highlight'], facecolor='lightyellow',
                                        linewidth=2, alpha=0.7)
    ax.add_patch(normal_box)
    ax.text(4, 1.15, 'Normal Equation Form', fontsize=11, ha='center', weight='bold')
    ax.text(4, 0.7, r'$0 \in F(\bar{x}) + N_K(\bar{x})$', fontsize=10, ha='center', family='monospace')

    ax.set_xlim(-0.2, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')

    save_figure(fig, 'fig_equivalence.pdf')

def fig_projection_property():
    """Figure showing projection property: characterization of projection"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Projection definition
    ax1.text(0.5, 0.95, 'Projection Operator Definition', fontsize=12, ha='center',
            weight='bold', transform=ax1.transAxes)

    definitions = [
        r'For $z \in \mathbb{R}^n$ and closed convex set $K$:',
        '',
        r'$P_K(z) = \arg\min_{y \in K} \|z - y\|_2^2$',
        '',
        'Properties:',
        r'1. Non-expansive: $\|P_K(x) - P_K(y)\| \leq \|x - y\|$',
        r'2. Idempotent: $P_K(P_K(z)) = P_K(z)$',
        r'3. Characterization:',
        r'   $x^* = P_K(z)$ iff $(z-x^*)^T(y-x^*) \leq 0$ for all $y \in K$'
    ]

    y_pos = 0.85
    for defn in definitions:
        ax1.text(0.05, y_pos, defn, fontsize=9, transform=ax1.transAxes,
                family='monospace' if r'\|' in defn else 'sans-serif',
                color=colors['text'])
        y_pos -= 0.08

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: Lemma visualization
    ax2.text(0.5, 0.95, 'Projection Lemma (Lemma 7.1)', fontsize=12, ha='center',
            weight='bold', transform=ax2.transAxes)

    lemma_text = [
        r'For nonempty closed convex $K \subseteq \mathbb{R}^n$:',
        '',
        r'$(x - P_K(z))^T(P_K(z) - z) \geq 0$',
        '',
        r'for all $x \in K$ and $z \in \mathbb{R}^n$',
        '',
        'Interpretation:',
        r'The vector from $z$ to $P_K(z)$ makes',
        r'an obtuse angle with any direction from',
        r'$P_K(z)$ into $K$.'
    ]

    y_pos = 0.85
    for txt in lemma_text:
        ax2.text(0.05, y_pos, txt, fontsize=9, transform=ax2.transAxes,
                family='monospace' if r'(' in txt and r'^T' in txt else 'sans-serif',
                color=colors['text'])
        y_pos -= 0.07

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    fig.tight_layout()
    save_figure(fig, 'fig_projection_properties.pdf')

def fig_algorithm_convergence():
    """Figure showing convergence behavior of projection algorithm"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Simulate convergence
    np.random.seed(42)
    x0 = np.array([0.1, 0.2])
    iterations = 20
    trajectory = [x0]

    for i in range(iterations):
        x_curr = trajectory[-1]
        # Simulate convergence to solution at [0.7, 0.6]
        x_next = 0.8 * x_curr + 0.2 * np.array([0.7, 0.6])
        trajectory.append(x_next)

    trajectory = np.array(trajectory)

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors['main'],
           markersize=5, linewidth=1.5, alpha=0.7)

    # Mark solution
    solution = np.array([0.7, 0.6])
    ax.plot(solution[0], solution[1], '*', markersize=20, color=colors['accent'],
           label=r'Solution $\bar{x}^*$')

    # Add convex set K
    K_circle = Circle(solution, 0.35, fill=True, alpha=0.1,
                     edgecolor=colors['main'], linewidth=2, color=colors['main'])
    ax.add_patch(K_circle)
    ax.text(solution[0] + 0.05, solution[1] - 0.45, r'$K$', fontsize=12, weight='bold')

    # Mark start point
    ax.plot(trajectory[0, 0], trajectory[0, 1], 's', markersize=8,
           color=colors['highlight'], label=r'Initial point $x_0$')

    # Add iteration labels for first few
    for i in [0, 5, 10, 15]:
        ax.annotate(f'$x_{i}$', xy=trajectory[i], xytext=(5, 5),
                   textcoords='offset points', fontsize=9, color=colors['text'])

    # Convergence info
    info_text = (
        r'Projected Gradient Algorithm:' + '\n'
        r'$x_{n+1} = P_K(x_n - \alpha \nabla f(x_n))$' + '\n'
        r'Converges to solution $\bar{x}^*$'
    )
    ax.text(0.05, 0.8, info_text, fontsize=10, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           family='monospace')

    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title('Convergence of Projection-Based Algorithm', fontsize=12, weight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    save_figure(fig, 'fig_algorithm_convergence.pdf')

def fig_normal_fixed_point_mapping():
    """Figure showing NFP mapping formulation"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Title and sections
    ax.text(0.5, 0.95, 'Normal-Fixed Point (NFP) Equation', fontsize=13, ha='center',
           weight='bold', transform=ax.transAxes)

    # VIP formulation
    ax.text(0.05, 0.88, 'Problem Formulation:', fontsize=11, weight='bold',
           transform=ax.transAxes, color=colors['main'])
    formulation = [
        r'Given: $K \subseteq \mathbb{R}^n$ closed convex, $F: K \to \mathbb{R}^n$',
        r'Find: $\bar{x} \in K$ such that $(x - \bar{x})^T F(\bar{x}) \geq 0$ for all $x \in K$'
    ]
    y_pos = 0.82
    for line in formulation:
        ax.text(0.08, y_pos, line, fontsize=9, transform=ax.transAxes,
               family='monospace')
        y_pos -= 0.05

    # Reformulation
    ax.text(0.05, 0.70, 'Reformulation as Fixed Point:', fontsize=11, weight='bold',
           transform=ax.transAxes, color=colors['accent'])
    reformulation = [
        r'$\Psi_\alpha(x) = f(P_K((1+\alpha)x - \alpha P_K(x - \alpha f(x))))$',
        r'$+ \alpha(P_K(x - \alpha f(x)) - P_K(x)) = 0$',
        '',
        r'Properties of $\Psi_\alpha$:',
        r'• Combines projection and fixed point formulations',
        r'• Allows use of fixed point iteration theory',
        r'• Parameter $\alpha > 0$ controls stepsize'
    ]
    y_pos = 0.64
    for line in reformulation:
        ax.text(0.08, y_pos, line, fontsize=9, transform=ax.transAxes,
               family='monospace' if r'P_K' in line else 'sans-serif')
        y_pos -= 0.045

    # Key relationships
    ax.text(0.05, 0.30, 'Key Relationships:', fontsize=11, weight='bold',
           transform=ax.transAxes, color=colors['highlight'])
    relationships = [
        r'1. $\bar{x}$ solves VIP $\Leftrightarrow$ $\Psi_\alpha(\bar{x}) = 0$',
        r'2. $\bar{x}$ is fixed point $\Leftrightarrow$ $\bar{x} - \frac{1}{\alpha}f(\bar{x})$ solves eq',
        r'3. Normal equation: $0 \in F(\bar{x}) + N_K(\bar{x})$'
    ]
    y_pos = 0.24
    for line in relationships:
        ax.text(0.08, y_pos, line, fontsize=9, transform=ax.transAxes,
               family='monospace')
        y_pos -= 0.05

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    save_figure(fig, 'fig_nfp_mapping.pdf')

def main():
    """Generate all figures"""
    print("Generating figures for Variational Inequality Problems...")
    print("=" * 60)

    fig_projection_operator()
    fig_normal_cone()
    fig_fixed_point_equivalence()
    fig_projection_property()
    fig_algorithm_convergence()
    fig_normal_fixed_point_mapping()

    print("=" * 60)
    print("All figures generated successfully!")

if __name__ == '__main__':
    main()
