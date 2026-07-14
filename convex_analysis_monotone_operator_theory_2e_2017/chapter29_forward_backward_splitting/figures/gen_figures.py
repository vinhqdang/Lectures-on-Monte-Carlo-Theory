#!/usr/bin/env python3
"""
Generate figures for Chapter 29: Projection Operators
Visualization of projection concepts, properties, and algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set up plotting parameters
plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE = (8, 6)
DPI = 150

# Color scheme
COLOR_SET = '#1f77b4'
COLOR_POINT = '#ff7f0e'
COLOR_PROJECTION = '#2ca02c'
COLOR_VECTOR = '#d62728'
COLOR_NORMAL = '#9467bd'

# ===== Figure 1: Basic Projection Concept =====
def fig_projection_concept():
    """Visualization of basic projection onto a convex set"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Draw a circle (convex set C)
    circle = Circle((0, 0), 1.0, fill=False, edgecolor=COLOR_SET, linewidth=2.5, label='Convex set C')
    ax.add_patch(circle)

    # Point x outside the set
    x_point = np.array([1.8, 0.8])
    ax.plot(x_point[0], x_point[1], 'o', markersize=10, color=COLOR_POINT, label='Point x')

    # Projection of x onto C
    proj_x = x_point / np.linalg.norm(x_point)  # Unit vector in direction of x
    ax.plot(proj_x[0], proj_x[1], 's', markersize=10, color=COLOR_PROJECTION, label='P_C(x)')

    # Draw vector from x to projection
    ax.arrow(proj_x[0], proj_x[1], x_point[0]-proj_x[0], x_point[1]-proj_x[1],
             head_width=0.1, head_length=0.1, fc=COLOR_VECTOR, ec=COLOR_VECTOR,
             linewidth=1.5, alpha=0.7, label='x - P_C(x)')

    # Draw radius to projection point
    ax.plot([0, proj_x[0]], [0, proj_x[1]], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlim(-1.5, 2.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Projection onto a Convex Set', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_projection_concept.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 2: Projection onto Affine Subspace =====
def fig_projection_affine():
    """Projection onto a line (affine subspace)"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Draw affine subspace (line): y = 0.5*x + 0.2
    x_line = np.linspace(-2, 2, 100)
    y_line = 0.5 * x_line + 0.2
    ax.plot(x_line, y_line, color=COLOR_SET, linewidth=2.5, label='Affine subspace C')

    # Point x
    x_point = np.array([1.5, 1.5])
    ax.plot(x_point[0], x_point[1], 'o', markersize=10, color=COLOR_POINT, label='Point x')

    # Normal direction to the line
    # Line has slope 0.5, so normal has slope -2
    normal_dir = np.array([-1, 2])
    normal_dir = normal_dir / np.linalg.norm(normal_dir)

    # Find projection point
    t = ((x_point[0] - 0) * 2 + (x_point[1] - 0.2) * 1) / (1 + 4)  # parameter for projection
    proj_point = np.array([t, 0.5*t + 0.2])
    ax.plot(proj_point[0], proj_point[1], 's', markersize=10, color=COLOR_PROJECTION, label='P_C(x)')

    # Normal vector
    ax.arrow(proj_point[0], proj_point[1],
             normal_dir[0]*0.5, normal_dir[1]*0.5,
             head_width=0.08, head_length=0.08, fc=COLOR_NORMAL, ec=COLOR_NORMAL,
             linewidth=2, label='Normal to C')

    # Draw perpendicular from x to C
    ax.plot([x_point[0], proj_point[0]], [x_point[1], proj_point[1]],
            color=COLOR_VECTOR, linewidth=1.5, linestyle='--', alpha=0.7, label='x - P_C(x)')

    ax.set_xlim(-2, 2.5)
    ax.set_ylim(-1, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Projection onto Affine Subspace', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_projection_affine.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 3: Projection onto Half-Space =====
def fig_projection_halfspace():
    """Projection onto a half-space"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Half-space: {x : <x, u> ≤ η} where u = (1,1)/√2, η = 0.5
    u = np.array([1, 1]) / np.sqrt(2)
    eta = 0.5

    # Boundary line: <x, u> = η
    x_range = np.linspace(-2, 2, 100)
    # u[0]*x + u[1]*y = eta => y = (eta - u[0]*x) / u[1]
    y_boundary = (eta - u[0]*x_range) / u[1]
    ax.plot(x_range, y_boundary, color=COLOR_SET, linewidth=2.5, label='Boundary: <x,u> = η')

    # Fill half-space
    x_fill = np.linspace(-2, 2, 100)
    y_fill = (eta - u[0]*x_fill) / u[1]
    ax.fill_between(x_fill, y_fill, -2, alpha=0.2, color=COLOR_SET, label='Half-space C')

    # Point x outside the set
    x_point = np.array([1.0, 1.0])
    ax.plot(x_point[0], x_point[1], 'o', markersize=10, color=COLOR_POINT, label='Point x')

    # Projection: x - <(x,u)-η>+ * u / ||u||²
    inner_prod = np.dot(x_point, u)
    if inner_prod > eta:
        proj_point = x_point - (inner_prod - eta) * u
    else:
        proj_point = x_point

    ax.plot(proj_point[0], proj_point[1], 's', markersize=10, color=COLOR_PROJECTION, label='P_C(x)')

    # Normal vector u
    ax.arrow(0, 0, u[0]*0.8, u[1]*0.8, head_width=0.1, head_length=0.1,
             fc=COLOR_NORMAL, ec=COLOR_NORMAL, linewidth=2, label='Normal u')

    # Line from x to projection
    ax.plot([x_point[0], proj_point[0]], [x_point[1], proj_point[1]],
            color=COLOR_VECTOR, linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Projection onto Half-Space', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_projection_halfspace.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 4: Projection onto Simplex =====
def fig_projection_simplex():
    """Projection onto probability simplex"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Simplex: {x ∈ ℝ³ : x_i ≥ 0, sum(x) = 1} projected to 2D
    # Using barycentric coordinates
    simplex_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    simplex = Polygon(simplex_vertices, fill=False, edgecolor=COLOR_SET, linewidth=2.5, label='Probability Simplex')
    ax.add_patch(simplex)

    # Fill simplex
    ax.fill(simplex_vertices[:, 0], simplex_vertices[:, 1], alpha=0.1, color=COLOR_SET)

    # Mark vertices
    ax.plot(simplex_vertices[:, 0], simplex_vertices[:, 1], 'ko', markersize=6)
    ax.text(0, -0.1, 'e₁', fontsize=10, ha='center')
    ax.text(1, -0.1, 'e₂', fontsize=10, ha='center')
    ax.text(0.5, np.sqrt(3)/2+0.1, 'e₃', fontsize=10, ha='center')

    # Point x outside simplex
    x_point = np.array([0.4, 0.8])
    ax.plot(x_point[0], x_point[1], 'o', markersize=10, color=COLOR_POINT, label='Point x')

    # Approximate projection onto simplex (for illustration)
    # Using soft-thresholding concept
    proj_point = np.array([0.35, 0.45])
    ax.plot(proj_point[0], proj_point[1], 's', markersize=10, color=COLOR_PROJECTION, label='P_Δ(x)')

    # Draw connection
    ax.plot([x_point[0], proj_point[0]], [x_point[1], proj_point[1]],
            color=COLOR_VECTOR, linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Projection onto Probability Simplex', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_projection_simplex.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 5: Firm Nonexpansiveness =====
def fig_nonexpansiveness():
    """Illustration of firm nonexpansiveness: ||P_C(x) - P_C(y)||² ≤ <x-y | P_C(x)-P_C(y)>"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Draw circle (convex set)
    circle = Circle((0, 0), 1.0, fill=False, edgecolor=COLOR_SET, linewidth=2.5, label='Convex set C')
    ax.add_patch(circle)

    # Two points x and y
    x = np.array([1.8, 0.6])
    y = np.array([0.8, -1.5])

    # Their projections
    proj_x = x / np.linalg.norm(x)
    proj_y = y / np.linalg.norm(y)

    # Plot points and projections
    ax.plot(x[0], x[1], 'o', markersize=10, color=COLOR_POINT, label='x')
    ax.plot(y[0], y[1], 'o', markersize=10, color=COLOR_POINT)
    ax.plot(proj_x[0], proj_x[1], 's', markersize=10, color=COLOR_PROJECTION, label='P_C(x)')
    ax.plot(proj_y[0], proj_y[1], 's', markersize=10, color=COLOR_PROJECTION)

    # Add labels
    ax.text(x[0]+0.1, x[1]+0.1, 'x', fontsize=11, fontweight='bold')
    ax.text(y[0]-0.2, y[1]-0.2, 'y', fontsize=11, fontweight='bold')
    ax.text(proj_x[0]+0.1, proj_x[1]+0.1, 'P_C(x)', fontsize=11, fontweight='bold')
    ax.text(proj_y[0]-0.2, proj_y[1]-0.2, 'P_C(y)', fontsize=11, fontweight='bold')

    # Draw vectors from projections to original points
    ax.arrow(proj_x[0], proj_x[1], x[0]-proj_x[0], x[1]-proj_x[1],
             head_width=0.08, head_length=0.08, fc=COLOR_NORMAL, ec=COLOR_NORMAL,
             linewidth=1.5, alpha=0.6)
    ax.arrow(proj_y[0], proj_y[1], y[0]-proj_y[0], y[1]-proj_y[1],
             head_width=0.08, head_length=0.08, fc=COLOR_NORMAL, ec=COLOR_NORMAL,
             linewidth=1.5, alpha=0.6)

    ax.set_xlim(-2, 2.2)
    ax.set_ylim(-2, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Firm Nonexpansiveness of Projections', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_nonexpansiveness.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 6: Algorithm Convergence =====
def fig_algorithm_convergence():
    """Illustration of subgradient projection algorithm convergence"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Simulate algorithm iterations
    np.random.seed(42)
    n_iter = 15

    # Starting point
    x = np.array([2.0, 2.0])
    trajectory = [x.copy()]

    # Simulate descent towards constraint set
    for i in range(n_iter):
        # Move towards origin (constraint set center)
        grad = x / np.linalg.norm(x)  # Gradient direction
        x = x - 0.25 * grad
        trajectory.append(x.copy())

    trajectory = np.array(trajectory)

    # Draw constraint set (ball)
    circle = Circle((0, 0), 0.3, fill=False, edgecolor=COLOR_SET, linewidth=2.5, label='Feasible set C')
    ax.add_patch(circle)

    # Draw trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', markersize=4,
            color=COLOR_POINT, linewidth=1.5, alpha=0.7, label='Iteration sequence')

    # Highlight start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=12, color=COLOR_POINT, label='x₀')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', markersize=12, color=COLOR_PROJECTION, label='x_n')

    # Add iteration numbers
    for i in [0, 5, 10, 14]:
        ax.text(trajectory[i, 0]+0.1, trajectory[i, 1]+0.1, f'{i}', fontsize=9)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Subgradient Projection Algorithm Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_algorithm_convergence.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 7: Error vs Iteration =====
def fig_convergence_rate():
    """Plot showing convergence rate of projection algorithms"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    n_iter = 50

    # Different convergence rates
    k = np.arange(1, n_iter+1)

    # Linear convergence
    linear = 0.9 ** k

    # Sublinear convergence
    sublinear = 1 / np.sqrt(k)

    # Exponential convergence
    exponential = 0.5 ** (k/5)

    ax.semilogy(k, linear, 'o-', linewidth=2, markersize=4, label='Linear: ρⁿ (ρ=0.9)', color='#1f77b4')
    ax.semilogy(k, sublinear, 's-', linewidth=2, markersize=4, label='Sublinear: 1/√n', color='#ff7f0e')
    ax.semilogy(k, exponential, '^-', linewidth=2, markersize=4, label='Faster: 0.5^(n/5)', color='#2ca02c')

    ax.set_xlabel('Iteration n', fontsize=12)
    ax.set_ylabel('Error ||xₙ - x*||', fontsize=12)
    ax.set_title('Convergence Rates of Projection Algorithms', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, n_iter)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_convergence_rate.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# ===== Figure 8: Box Constraint Projection =====
def fig_projection_box():
    """Projection onto a box constraint"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Box constraint [a, b] × [c, d]
    a, b = -1, 1
    c, d = -0.5, 1.5

    box = Rectangle((a, c), b-a, d-c, fill=False, edgecolor=COLOR_SET, linewidth=2.5, label='Box [a,b]×[c,d]')
    ax.add_patch(box)

    # Fill box
    ax.fill([a, b, b, a], [c, c, d, d], alpha=0.1, color=COLOR_SET)

    # Sample points outside and inside
    np.random.seed(42)
    n_points = 6

    for i in range(n_points):
        x = np.array([2.5 - i*0.8, 2.0 - i*0.5])

        # Projection onto box
        proj_x = np.array([np.clip(x[0], a, b), np.clip(x[1], c, d)])

        # Plot
        ax.plot(x[0], x[1], 'o', markersize=8, color=COLOR_POINT, alpha=0.6)
        ax.plot(proj_x[0], proj_x[1], 's', markersize=8, color=COLOR_PROJECTION, alpha=0.6)
        ax.plot([x[0], proj_x[0]], [x[1], proj_x[1]], '--', linewidth=1, color=COLOR_VECTOR, alpha=0.5)

    # Add one highlighted example
    x_ex = np.array([2.5, 2.0])
    proj_x_ex = np.array([np.clip(x_ex[0], a, b), np.clip(x_ex[1], c, d)])
    ax.plot(x_ex[0], x_ex[1], 'o', markersize=12, color=COLOR_POINT, label='x', zorder=5)
    ax.plot(proj_x_ex[0], proj_x_ex[1], 's', markersize=12, color=COLOR_PROJECTION, label='P_Box(x)', zorder=5)

    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Projection onto Box Constraints', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/user/Lectures-on-Monte-Carlo-Theory/convex_analysis_monotone_operator_theory_2e_2017/chapter29_forward_backward_splitting/figures/fig_projection_box.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.close()


# Main execution
if __name__ == '__main__':
    print("Generating figures for Chapter 29: Projection Operators...")

    fig_projection_concept()
    print("✓ Projection concept")

    fig_projection_affine()
    print("✓ Projection onto affine subspace")

    fig_projection_halfspace()
    print("✓ Projection onto half-space")

    fig_projection_simplex()
    print("✓ Projection onto simplex")

    fig_nonexpansiveness()
    print("✓ Firm nonexpansiveness")

    fig_algorithm_convergence()
    print("✓ Algorithm convergence")

    fig_convergence_rate()
    print("✓ Convergence rates")

    fig_projection_box()
    print("✓ Box constraint projection")

    print("\nAll figures generated successfully!")
