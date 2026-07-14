#!/usr/bin/env python3
"""
Generate figures for Chapter 30: Best Approximation Algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style for consistency
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

def setup_figure(width=6, height=4):
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    return fig, ax

# ============================================================================
# Figure 30.1: Method of Alternating Projections
# ============================================================================
def figure_alternating_projections():
    """Visualize the method of alternating projections."""
    fig, ax = setup_figure(6, 5)

    # Create two ellipses representing convex sets C1 and C2
    ellipse1 = Ellipse((0.3, 0.4), 0.6, 0.4, angle=25,
                       linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ellipse2 = Ellipse((0.55, 0.35), 0.5, 0.35, angle=-15,
                       linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.3)

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    # Initial point x0
    x0 = np.array([0.85, 0.7])
    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='$x_0$')
    ax.text(x0[0]+0.03, x0[1]+0.03, '$x_0$', fontsize=12, fontweight='bold')

    # Simulate alternating projections
    # C1 is centered around (0.3, 0.4), C2 around (0.55, 0.35)
    c1_center = np.array([0.3, 0.4])
    c2_center = np.array([0.55, 0.35])

    # Simulate iterates
    x = x0.copy()
    points = [x.copy()]

    for i in range(6):
        # Project onto C1
        direction = c1_center - x
        x = x + 0.5 * direction
        points.append(x.copy())

        # Project onto C2
        direction = c2_center - x
        x = x + 0.4 * direction
        points.append(x.copy())

    points = np.array(points)

    # Plot trajectory
    ax.plot(points[:, 0], points[:, 1], 'g--', alpha=0.5, linewidth=1.5, label='Iterates')
    ax.plot(points[:, 0], points[:, 1], 'go', markersize=4, alpha=0.6)

    # Limit point (intersection approximation)
    x_inf = points[-1]
    ax.plot(x_inf[0], x_inf[1], 'r*', markersize=15, label='$x_\infty$')
    ax.text(x_inf[0]-0.05, x_inf[1]-0.08, '$x_\infty$', fontsize=12, fontweight='bold')

    # Best approximation point (if it existed in intersection)
    p = np.array([0.4, 0.365])
    ax.plot(p[0], p[1], 'bs', markersize=8, label='$p$ (best approx.)')
    ax.text(p[0]-0.08, p[1]+0.05, '$p$', fontsize=11, fontweight='bold')

    # Labels for sets
    ax.text(0.15, 0.55, '$C_1$', fontsize=13, color='blue', fontweight='bold')
    ax.text(0.65, 0.25, '$C_2$', fontsize=13, color='red', fontweight='bold')

    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 0.85)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Fig. 30.1: Alternating Projections Method\n$x_{n+1} = P_{C_2} P_{C_1} x_n$',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_alternating_projections.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_alternating_projections.pdf")

# ============================================================================
# Figure: Convergence of Halper's Algorithm
# ============================================================================
def figure_halpers_convergence():
    """Plot convergence of Halper's algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Simulate algorithm convergence
    n = np.arange(0, 50)

    # Distance to fixed point decays exponentially
    lambda_values = [0.3, 0.5, 0.7, 0.9]
    colors = ['blue', 'green', 'orange', 'red']

    for lam, color in zip(lambda_values, colors):
        distance = (1 - lam) ** n
        ax1.semilogy(n, distance, 'o-', label=f'$\\lambda = {lam}$',
                    color=color, markersize=4, alpha=0.7)

    ax1.set_xlabel('Iteration $n$', fontsize=11)
    ax1.set_ylabel('Distance to Fixed Point', fontsize=11)
    ax1.set_title("Halper's Algorithm: Convergence Rate", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 50)

    # Iterate sequence in 2D
    np.random.seed(42)
    x = np.array([1.0, 1.0])
    T_center = np.array([0.0, 0.0])

    iterates = [x.copy()]
    for i in range(40):
        x = 0.6 * x + 0.4 * T_center
        iterates.append(x.copy())

    iterates = np.array(iterates)

    ax2.plot(iterates[:, 0], iterates[:, 1], 'b-', alpha=0.5, linewidth=1.5)
    ax2.plot(iterates[:, 0], iterates[:, 1], 'bo', markersize=4, alpha=0.6)
    ax2.plot(iterates[0, 0], iterates[0, 1], 'go', markersize=8, label='$x_0$')
    ax2.plot(T_center[0], T_center[1], 'r*', markersize=15, label='Fixed Point')

    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.set_title("Halper's Algorithm: Iterate Sequence", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fig_halpers_convergence.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_halpers_convergence.pdf")

# ============================================================================
# Figure: Dykstra's Algorithm Convergence
# ============================================================================
def figure_dykstra_convergence():
    """Visualize Dykstra's algorithm convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left: Convergence of error
    n = np.arange(0, 100)
    error_dykstra = np.exp(-0.08 * n)
    error_cyclic = np.exp(-0.04 * n)

    ax1.semilogy(n, error_dykstra, 'b-o', label='Dykstra', markersize=4, alpha=0.7, linewidth=2)
    ax1.semilogy(n, error_cyclic, 'r--s', label='Cyclic Projection', markersize=4, alpha=0.7, linewidth=2)
    ax1.set_xlabel('Iteration $n$', fontsize=11)
    ax1.set_ylabel('Error $\\|x_n - P_C x_0\\|$', fontsize=11)
    ax1.set_title("Dykstra vs. Cyclic Projection", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 100)

    # Right: Projection onto intersection of two convex sets
    # Two circles
    circle1 = Circle((0.3, 0.5), 0.3, fill=False, edgecolor='blue', linewidth=2, label='$C_1$')
    circle2 = Circle((0.7, 0.5), 0.3, fill=False, edgecolor='red', linewidth=2, label='$C_2$')

    ax2.add_patch(circle1)
    ax2.add_patch(circle2)

    # Starting point
    x0 = np.array([0.95, 0.7])
    ax2.plot(x0[0], x0[1], 'ko', markersize=8, label='$x_0$')
    ax2.text(x0[0]+0.02, x0[1]+0.03, '$x_0$', fontsize=11, fontweight='bold')

    # Simulate Dykstra iterations
    c1_center = np.array([0.3, 0.5])
    c2_center = np.array([0.7, 0.5])
    r1, r2 = 0.3, 0.3

    x = x0.copy()
    p = np.array([0.5, 0.5])

    iterates = [x.copy()]
    for i in range(8):
        # Project onto C1
        direction = c1_center - x
        dist = np.linalg.norm(direction)
        if dist > r1:
            x = c1_center + r1 * direction / dist
        iterates.append(x.copy())

        # Project onto C2
        direction = c2_center - x
        dist = np.linalg.norm(direction)
        if dist > r2:
            x = c2_center + r2 * direction / dist
        iterates.append(x.copy())

    iterates = np.array(iterates)

    ax2.plot(iterates[:, 0], iterates[:, 1], 'g--', alpha=0.5, linewidth=1.5)
    ax2.plot(iterates[:, 0], iterates[:, 1], 'go', markersize=4, alpha=0.6)
    ax2.plot(p[0], p[1], 'r*', markersize=15, label='$P_C x_0$')

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0.05, 0.95)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.set_title("Dykstra's Algorithm: Projection onto $C_1 \\cap C_2$",
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_dykstra_convergence.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_dykstra_convergence.pdf")

# ============================================================================
# Figure: Haugazeau's Algorithm
# ============================================================================
def figure_haugazeau_algorithm():
    """Visualize Haugazeau's algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left: Multiple sets intersection
    # Create three circles representing three convex sets
    sets = [
        Circle((0.35, 0.5), 0.25, fill=False, edgecolor='blue', linewidth=2.5, alpha=0.8),
        Circle((0.65, 0.6), 0.25, fill=False, edgecolor='red', linewidth=2.5, alpha=0.8),
        Circle((0.5, 0.25), 0.22, fill=False, edgecolor='green', linewidth=2.5, alpha=0.8),
    ]

    labels = ['$C_1$', '$C_2$', '$C_3$']
    label_positions = [(0.25, 0.65), (0.75, 0.75), (0.55, 0.05)]
    colors = ['blue', 'red', 'green']

    for patch, label, pos, color in zip(sets, labels, label_positions, colors):
        ax1.add_patch(patch)
        ax1.text(pos[0], pos[1], label, fontsize=12, color=color, fontweight='bold')

    # Starting point
    x0 = np.array([0.9, 0.8])
    ax1.plot(x0[0], x0[1], 'ko', markersize=8, label='$x_0$')

    # Intersection region
    intersection_x = np.linspace(0.4, 0.6, 100)
    intersection_y = 0.45 + 0.1 * np.sin((intersection_x - 0.4) * np.pi)
    ax1.fill_between(intersection_x, intersection_y - 0.08, intersection_y + 0.08,
                     color='yellow', alpha=0.3, label='$C_1 \\cap C_2 \\cap C_3$')

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.0)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.set_title("Haugazeau's Algorithm: Multiple Sets", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # Right: Convergence behavior
    n = np.arange(0, 80)

    # Different convergence rates for different numbers of sets
    m_values = [2, 3, 4, 5]
    colors_rates = ['blue', 'green', 'orange', 'red']

    for m, color in zip(m_values, colors_rates):
        # Convergence rate depends on number of sets
        error = np.exp(-0.05 * m * n)
        ax2.semilogy(n, error, 'o-', label=f'$m = {m}$ sets',
                    color=color, markersize=3, alpha=0.7, linewidth=2)

    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('Error to Projection', fontsize=11)
    ax2.set_title("Haugazeau's Algorithm: Convergence", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 80)

    plt.tight_layout()
    plt.savefig('fig_haugazeau_algorithm.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_haugazeau_algorithm.pdf")

# ============================================================================
# Figure: Forward-Backward Splitting
# ============================================================================
def figure_forward_backward_splitting():
    """Visualize forward-backward splitting algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left: Problem setup (f + g optimization)
    x = np.linspace(-1, 3, 200)

    # f(x) = (x-1)^2 (convex)
    f = (x - 1) ** 2
    # g(x) = |x - 0.5| (non-smooth)
    g = np.abs(x - 0.5)
    # h(x) = f(x) + g(x)
    h = f + g

    ax1.plot(x, f, 'b-', label='$f(x)$ (smooth)', linewidth=2, alpha=0.7)
    ax1.plot(x, g, 'r-', label='$g(x)$ (non-smooth)', linewidth=2, alpha=0.7)
    ax1.plot(x, h, 'g-', label='$h(x) = f(x) + g(x)$', linewidth=2.5)

    # Minimum
    min_idx = np.argmin(h)
    min_x = x[min_idx]
    min_val = h[min_idx]
    ax1.plot(min_x, min_val, 'r*', markersize=15, label='Minimizer')

    ax1.set_xlabel('$x$', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Forward-Backward: Problem Setup', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-1, 3)

    # Right: Convergence trajectory
    x_traj = [2.5]
    for i in range(40):
        x_curr = x_traj[-1]
        # Gradient of f: 2(x-1)
        grad_f = 2 * (x_curr - 1)
        # Proximal operator of g (soft thresholding)
        x_next = np.sign(x_curr - 0.2 * grad_f) * np.maximum(np.abs(x_curr - 0.2 * grad_f) - 0.1, 0)
        x_traj.append(x_next)

    x_traj = np.array(x_traj)

    ax2.plot(x_traj, 'bo-', markersize=4, linewidth=1.5, label='Iterate sequence')
    ax2.axhline(y=min_x, color='r', linestyle='--', linewidth=2, label='Minimizer')
    ax2.set_xlabel('Iteration $n$', fontsize=11)
    ax2.set_ylabel('$x_n$', fontsize=11)
    ax2.set_title('Forward-Backward: Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_forward_backward.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_forward_backward.pdf")

# ============================================================================
# Figure: Proximal Point Algorithm
# ============================================================================
def figure_proximal_point():
    """Visualize proximal point algorithm."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Convergence comparison for proximal point
    n = np.arange(0, 100)

    # Proximal point: linear convergence
    error_proximal = 0.5 ** n

    # Gradient descent (with step size 0.1, for Lipschitz cont. grad)
    error_gradient = 0.8 ** n

    # Accelerated gradient
    error_accelerated = 0.6 ** n

    ax.semilogy(n, error_proximal, 'b-o', label='Proximal Point',
               markersize=4, linewidth=2, alpha=0.8)
    ax.semilogy(n, error_gradient, 'r-s', label='Gradient Descent',
               markersize=4, linewidth=2, alpha=0.8)
    ax.semilogy(n, error_accelerated, 'g-^', label='Accelerated Gradient',
               markersize=4, linewidth=2, alpha=0.8)

    ax.set_xlabel('Iteration $n$', fontsize=12)
    ax.set_ylabel('Error $\\|x_n - x^*\\|$', fontsize=12)
    ax.set_title('Comparison of Convergence Rates', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-12, 1)

    plt.tight_layout()
    plt.savefig('fig_proximal_point.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_proximal_point.pdf")

# ============================================================================
# Figure: Projection onto Convex Set Example
# ============================================================================
def figure_projection_example():
    """Example of projection onto different convex sets."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)

    # (1,1): Projection onto ball
    ax = axes[0, 0]
    circle = Circle((0, 0), 0.5, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(circle)
    x0 = np.array([0.9, 0.8])
    x_proj = np.array([0.5 * x0[0] / np.linalg.norm(x0), 0.5 * x0[1] / np.linalg.norm(x0)])
    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='$x$')
    ax.plot(x_proj[0], x_proj[1], 'r*', markersize=15, label='$P_C(x)$')
    ax.plot([x0[0], x_proj[0]], [x0[1], x_proj[1]], 'k--', alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Projection onto Ball', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # (1,2): Projection onto hyperplane
    ax = axes[0, 1]
    x_line = np.linspace(-1, 1, 100)
    y_line = 0.3 * np.ones_like(x_line)
    ax.plot(x_line, y_line, 'b-', linewidth=2, label='Hyperplane')
    x0 = np.array([0.6, 0.9])
    x_proj = np.array([0.6, 0.3])
    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='$x$')
    ax.plot(x_proj[0], x_proj[1], 'r*', markersize=15, label='$P_C(x)$')
    ax.plot([x0[0], x_proj[0]], [x0[1], x_proj[1]], 'k--', alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Projection onto Hyperplane', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # (2,1): Projection onto box
    ax = axes[1, 0]
    box = mpatches.Rectangle((-0.5, -0.3), 1, 0.6,
                             fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(box)
    x0 = np.array([0.8, 0.7])
    x_proj = np.array([0.5, 0.3])
    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='$x$')
    ax.plot(x_proj[0], x_proj[1], 'r*', markersize=15, label='$P_C(x)$')
    ax.plot([x0[0], x_proj[0]], [x0[1], x_proj[1]], 'k--', alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Projection onto Box', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # (2,2): Projection onto polyhedron (cone)
    ax = axes[1, 1]
    cone_x = np.array([-0.5, 0, 0.5])
    cone_y = np.array([-0.5, 0.6, -0.5])
    ax.fill(cone_x, cone_y, 'lightblue', alpha=0.3, edgecolor='blue', linewidth=2)
    x0 = np.array([0.7, 0.8])
    x_proj = np.array([0.35, 0.3])
    ax.plot(x0[0], x0[1], 'ko', markersize=8, label='$x$')
    ax.plot(x_proj[0], x_proj[1], 'r*', markersize=15, label='$P_C(x)$')
    ax.plot([x0[0], x_proj[0]], [x0[1], x_proj[1]], 'k--', alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Projection onto Cone', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Examples of Projection onto Convex Sets', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('fig_projections.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_projections.pdf")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for Chapter 30: Best Approximation Algorithms")
    print("=" * 70)

    figure_alternating_projections()
    figure_halpers_convergence()
    figure_dykstra_convergence()
    figure_haugazeau_algorithm()
    figure_forward_backward_splitting()
    figure_proximal_point()
    figure_projection_example()

    print("=" * 70)
    print("All figures generated successfully!")
