"""
Generate figures for Chapter 5: Fejér Monotonicity
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

# Set up matplotlib for PDF output
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

def set_dark_mode():
    """Configure dark-friendly colors"""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'

set_dark_mode()

# Figure 1: Fejér Monotone Sequence Concept
def fig_fejer_monotone_concept():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the closed convex set C
    theta = np.linspace(0, 2*np.pi, 100)
    C_x = 5 + 2.5*np.cos(theta)
    C_y = 5 + 2.5*np.sin(theta)
    ax.fill(C_x, C_y, color='lightblue', alpha=0.3, label='Closed convex set $C$')
    ax.plot(C_x, C_y, 'b-', linewidth=2)

    # Draw a sequence {x_n} approaching C
    t = np.linspace(0, 4*np.pi, 20)
    x_seq = 5 + (4 - t/(np.pi))*0.3*np.cos(t)
    y_seq = 5 + (4 - t/(np.pi))*0.3*np.sin(t)

    # Plot sequence points
    ax.scatter(x_seq, y_seq, c=np.arange(len(x_seq)), cmap='autumn', s=100, zorder=5, edgecolors='black', linewidth=1)

    # Draw connections
    for i in range(len(x_seq)-1):
        ax.arrow(x_seq[i], y_seq[i], x_seq[i+1]-x_seq[i], y_seq[i+1]-y_seq[i],
                head_width=0.15, head_length=0.1, fc='gray', ec='gray', alpha=0.5)

    # Mark some points
    for i in [0, 5, 10, 15]:
        ax.annotate(f'$x_{i}$', xy=(x_seq[i], y_seq[i]), xytext=(10, 10),
                   textcoords='offset points', fontsize=10, color='darkred')

    # Add distance illustration
    center = np.array([5, 5])
    ax.plot([x_seq[0], center[0]], [y_seq[0], center[1]], 'r--', linewidth=1.5, alpha=0.7, label='Distance to $C$')
    ax.plot([x_seq[5], center[0]], [y_seq[5], center[1]], 'r--', linewidth=1.5, alpha=0.7)

    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Fejér Monotone Sequence: $\\|x_n - x\\|$ decreases for all $x \\in C$', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)

    plt.tight_layout()
    plt.savefig('fejer_monotone_concept.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2: Projection onto Convex Sets
def fig_projection_algorithm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Single projection
    C1_x = np.linspace(1, 4, 100)
    C1_y = np.sqrt(np.maximum(0, 9 - (C1_x - 2.5)**2))
    ax1.fill_between(C1_x, 0, C1_y, color='lightcoral', alpha=0.3, label='Set $C$')
    ax1.plot(C1_x, C1_y, 'r-', linewidth=2)
    ax1.plot(C1_x, -C1_y, 'r-', linewidth=2)

    # Starting point
    x0 = np.array([1, 2.5])
    ax1.scatter(*x0, s=150, c='green', marker='o', zorder=5, label='$x_0$', edgecolors='black', linewidth=1.5)

    # Projection
    proj = np.array([2.5, 0])
    ax1.scatter(*proj, s=150, c='blue', marker='s', zorder=5, label='$P_C(x_0)$', edgecolors='black', linewidth=1.5)

    # Connection
    ax1.plot([x0[0], proj[0]], [x0[1], proj[1]], 'k--', linewidth=1.5, alpha=0.7)

    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Projection onto Convex Set $C$', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)

    # Right: Multiple projections
    C1_theta = np.linspace(0, np.pi, 50)
    C1_x = 2 + 1.5*np.cos(C1_theta)
    C1_y = 2 + 1.5*np.sin(C1_theta)
    ax2.fill_between(C1_x, 0, C1_y, color='lightcoral', alpha=0.3, label='$C_1$')
    ax2.plot(C1_x, C1_y, 'r-', linewidth=2)

    C2_x = np.array([0, 4, 4, 0, 0])
    C2_y = np.array([0, 0, 2, 2, 0])
    ax2.fill(C2_x, C2_y, color='lightblue', alpha=0.3, label='$C_2$')
    ax2.plot(C2_x, C2_y, 'b-', linewidth=2)

    # Alternating projections sequence
    x_seq = np.array([[0.5, 3], [1.5, 2.5], [2, 2], [2.2, 1.9]])
    for i, pt in enumerate(x_seq):
        ax2.scatter(*pt, s=100, c=f'C{i}', cmap='viridis', zorder=5, edgecolors='black', linewidth=1)
        ax2.text(pt[0]+0.15, pt[1]+0.15, f'$x_{i}$', fontsize=10)

    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('POCS: Alternating Projections', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)

    plt.tight_layout()
    plt.savefig('projection_algorithm.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3: Convergence behavior
def fig_convergence_behavior():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Strong convergence
    ax = axes[0, 0]
    n = np.arange(1, 101)
    x_n = 1.0 / n
    ax.semilogy(n, x_n, 'b-', linewidth=2, label='$d_C(x_n)$')
    ax.fill_between(n, 0, x_n, alpha=0.2, color='blue')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(a) Strong Convergence', fontsize=11, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=10)
    ax.set_ylabel('$d_C(x_n)$', fontsize=10)
    ax.legend(fontsize=10)

    # (b) Weak convergence
    ax = axes[0, 1]
    n = np.arange(1, 101)
    x_n = 1.0 / np.sqrt(n) * np.sin(n)
    ax.plot(n, x_n, 'r.-', linewidth=1, markersize=4, label='$(\\langle x_n, z \\rangle)_{n \\in \\mathbb{N}}$')
    ax.plot(n, 1.0/np.sqrt(n), 'r--', linewidth=1.5, alpha=0.7)
    ax.plot(n, -1.0/np.sqrt(n), 'r--', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b) Weak Convergence (oscillating)', fontsize=11, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=10)
    ax.set_ylabel('$\\langle x_n, z \\rangle$', fontsize=10)
    ax.legend(fontsize=10)

    # (c) Distance sequences
    ax = axes[1, 0]
    n = np.arange(1, 101)
    d_strong = 1.0 / n
    d_linear = np.exp(-0.1*n)
    d_sublinear = 1.0 / np.sqrt(n)
    ax.semilogy(n, d_strong, 'b-', linewidth=2, label='Sublinear: $1/n$')
    ax.semilogy(n, d_linear, 'r-', linewidth=2, label='Linear: $\\kappa^n$, $\\kappa=0.9$')
    ax.semilogy(n, d_sublinear, 'g-', linewidth=2, label='Sublinear: $1/\\sqrt{n}$')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(c) Convergence Rates', fontsize=11, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=10)
    ax.set_ylabel('$d_C(x_n)$', fontsize=10)
    ax.legend(fontsize=10)

    # (d) Krasnosel'skii-Mann iteration
    ax = axes[1, 1]
    n = np.arange(1, 101)
    for lam in [0.3, 0.5, 0.7, 0.9]:
        x_n = np.power(1 - lam, n)
        ax.semilogy(n, x_n, linewidth=2, label=f'$\\lambda = {lam}$')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(d) KM Iteration: Effect of $\\lambda$', fontsize=11, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=10)
    ax.set_ylabel('Error', fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('convergence_behavior.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4: Fixed point theorem illustration
def fig_fixed_point_iteration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Contraction mapping cobweb plot
    x = np.linspace(0, 1, 1000)
    T_x = 0.6*x + 0.1

    ax1.plot(x, x, 'k-', linewidth=2, label='$y = x$')
    ax1.plot(x, T_x, 'b-', linewidth=2, label='$y = T(x)$')

    # Cobweb diagram
    x0 = 0.1
    for i in range(8):
        y = T_x[int(x0*1000)]
        ax1.plot([x0, x0], [x0, y], 'r-', linewidth=1, alpha=0.6)
        ax1.plot([x0, y], [y, y], 'r-', linewidth=1, alpha=0.6)
        x0 = y

    # Fixed point
    x_star = 0.25  # Solve x = 0.6*x + 0.1
    ax1.scatter(x_star, x_star, s=200, c='red', marker='*', zorder=5,
               edgecolors='black', linewidth=1.5, label=f'Fixed point $x^*$')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_title('Contraction Mapping Fixed Point', fontsize=12, fontweight='bold')
    ax1.set_xlabel('$x_n$', fontsize=11)
    ax1.set_ylabel('$x_{n+1} = T(x_n)$', fontsize=11)

    # Right: Nonexpansive operator (harder to converge)
    x = np.linspace(0, 1, 1000)
    T_x = 0.99*x + 0.005

    ax2.plot(x, x, 'k-', linewidth=2, label='$y = x$')
    ax2.plot(x, T_x, 'g-', linewidth=2, label='$y = T(x)$ (nonexpansive)')

    # Cobweb diagram for nonexpansive
    x0 = 0.05
    for i in range(30):
        y = 0.99*x0 + 0.005
        ax2.plot([x0, x0], [x0, y], 'orange', linewidth=0.8, alpha=0.5)
        ax2.plot([x0, y], [y, y], 'orange', linewidth=0.8, alpha=0.5)
        x0 = y

    # Fixed point
    x_star = 0.5
    ax2.scatter(x_star, x_star, s=200, c='red', marker='*', zorder=5,
               edgecolors='black', linewidth=1.5, label=f'Fixed point $x^*$')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_title('Nonexpansive Operator (Slow Convergence)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('$x_n$', fontsize=11)
    ax2.set_ylabel('$x_{n+1} = T(x_n)$', fontsize=11)

    plt.tight_layout()
    plt.savefig('fixed_point_iteration.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5: Quasi-Fejér monotonicity
def fig_quasi_fejer():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Exact Fejér
    ax = axes[0]
    n = np.arange(1, 101)
    d_exact = 2.0 / n
    ax.semilogy(n, d_exact, 'b-', linewidth=2.5, label='Fejér: $d_C(x_n) = O(1/n)$')
    ax.fill_between(n, d_exact*0.5, d_exact*2, alpha=0.2, color='blue')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(a) Exact Fejér Monotonicity', fontsize=12, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=11)
    ax.set_ylabel('$d_C(x_n)$', fontsize=11)
    ax.legend(fontsize=11)

    # Right: Quasi-Fejér with error
    ax = axes[1]
    d_quasi = 2.0/n + 0.5*np.sin(n/10)/n
    ax.semilogy(n, d_quasi, 'r-', linewidth=2.5, label='Quasi-Fejér: with error $\\varepsilon_n$')
    ax.semilogy(n, 0.5*np.sin(n/10)/n, 'orange', linewidth=1.5, linestyle='--', label='Error sequence $\\varepsilon_n$')
    ax.fill_between(n, d_quasi*0.5, d_quasi*1.5, alpha=0.2, color='red')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(b) Quasi-Fejér Monotonicity (with errors)', fontsize=12, fontweight='bold')
    ax.set_xlabel('$n$', fontsize=11)
    ax.set_ylabel('$d_C(x_n)$', fontsize=11)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('quasi_fejer.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 6: Alternating projections
def fig_alternating_projections():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Case 1: Two lines
    ax = axes[0, 0]
    x = np.linspace(-2, 5, 100)

    y1 = 0.5*x + 1
    y2 = -x + 4

    ax.plot(x, y1, 'b-', linewidth=2.5, label='$C_1$ (line 1)')
    ax.plot(x, y2, 'r-', linewidth=2.5, label='$C_2$ (line 2)')

    x_int = 2
    y_int = 2
    ax.scatter(x_int, y_int, s=200, c='green', marker='*', zorder=5,
              edgecolors='black', linewidth=1.5, label='$C_1 \\cap C_2$')

    pts = [(1, 1.5), (0.75, 1.25), (0.93, 1.07), (0.97, 1.01)]
    colors = plt.cm.Spectral(np.linspace(0, 1, len(pts)))
    for i, (px, py) in enumerate(pts):
        ax.scatter(px, py, s=100, c=[colors[i]], zorder=5, edgecolors='black', linewidth=1)
        ax.text(px-0.3, py+0.2, f'$x_{i}$', fontsize=10)

    ax.set_xlim(-1, 4)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('(a) Alternating Projections: Two Lines', fontsize=11, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)

    # Case 2: Two circles
    ax = axes[0, 1]
    theta = np.linspace(0, 2*np.pi, 100)

    C1_x = 1 + 1.5*np.cos(theta)
    C1_y = 1 + 1.5*np.sin(theta)
    ax.plot(C1_x, C1_y, 'b-', linewidth=2.5, label='$C_1$')
    ax.fill(C1_x, C1_y, alpha=0.1, color='blue')

    C2_x = 2.5 + 1.5*np.cos(theta)
    C2_y = 1 + 1.5*np.sin(theta)
    ax.plot(C2_x, C2_y, 'r-', linewidth=2.5, label='$C_2$')
    ax.fill(C2_x, C2_y, alpha=0.1, color='red')

    ax.set_xlim(-1, 5)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('(b) Alternating Projections: Two Circles', fontsize=11, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)

    # Case 3: Convergence plot
    ax = axes[1, 0]
    n = np.arange(1, 201)
    d_lines = 1.0 / np.power(2, n/20)
    d_circles = 1.0 / np.sqrt(n)

    ax.semilogy(n, d_lines, 'b-', linewidth=2.5, label='Two affine subspaces')
    ax.semilogy(n, d_circles, 'r-', linewidth=2.5, label='Two general convex sets')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(c) Convergence Comparison', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration $n$', fontsize=10)
    ax.set_ylabel('Distance to solution', fontsize=10)
    ax.legend(fontsize=10)

    # Case 4: Multiple sets
    ax = axes[1, 1]
    for i, (cx, cy, col) in enumerate([(1, 2.5, 'blue'), (2.5, 2.5, 'red'), (2, 0.5, 'green')]):
        C_x = cx + 1.2*np.cos(theta)
        C_y = cy + 1.2*np.sin(theta)
        ax.plot(C_x, C_y, color=col, linewidth=2.5, label=f'$C_{i+1}$')
        ax.fill(C_x, C_y, alpha=0.1, color=col)

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('(d) Multiple Sets (Cyclic Projections)', fontsize=11, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)

    plt.tight_layout()
    plt.savefig('alternating_projections.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 7: Numerical example
def fig_numerical_example():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example: Finding intersection of two lines
    ax = axes[0, 0]
    x = np.linspace(-0.5, 3.5, 100)
    y1 = 3 - x
    y2 = 2*x

    ax.plot(x, y1, 'b-', linewidth=2.5, label='$x + y = 3$')
    ax.plot(x, y2, 'r-', linewidth=2.5, label='$2x - y = 0$')

    ax.scatter([1], [2], s=200, c='green', marker='*', zorder=5,
              edgecolors='black', linewidth=1.5, label='Solution $(1, 2)$')

    x0 = np.array([0.2, 2.8])
    points = [x0]
    for _ in range(10):
        t = (3 - points[-1][0] - points[-1][1]) / 2
        p1 = points[-1] + t*np.array([1, 1])/np.sqrt(2)
        points.append(p1)

        t = (2*points[-1][0] - points[-1][1]) / 5
        p2 = points[-1] + t*np.array([2, -1])/np.sqrt(5)
        points.append(p2)

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], 'go-', markersize=4, linewidth=1, alpha=0.6)
    ax.scatter(points[0, 0], points[0, 1], s=100, c='orange', marker='o', zorder=5,
              edgecolors='black', linewidth=1, label='Start $x_0$')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('(a) Alternating Projections on Lines', fontsize=11, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)

    # Convergence plot
    ax = axes[0, 1]
    distances = np.sqrt((points[:, 0] - 1)**2 + (points[:, 1] - 2)**2)
    ax.semilogy(range(len(distances)), distances, 'b-o', linewidth=2, markersize=5)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(b) Convergence: Distance to Solution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('$\\|x_n - x^*\\|$', fontsize=10)

    # KM iteration example
    ax = axes[1, 0]
    lambda_vals = [0.3, 0.5, 0.7, 0.9]
    x0 = 2.0
    fixed_point = 0.25

    for lam in lambda_vals:
        x_n = [x0]
        for _ in range(50):
            T_x = 0.6*x_n[-1] + 0.1
            x_new = (1 - lam)*x_n[-1] + lam*T_x
            x_n.append(x_new)

        errors = np.abs(np.array(x_n) - fixed_point)
        ax.semilogy(range(len(errors)), errors, linewidth=2, marker='o',
                   markersize=3, label=f'$\\lambda = {lam}$', alpha=0.8)

    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(c) KM Iteration: Different Stepsizes', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Error $|x_n - x^*|$', fontsize=10)
    ax.legend(fontsize=10)

    # Fejér monotonicity verification
    ax = axes[1, 1]
    C = 1.0
    x_seq = [2.0]
    dist_seq = [abs(2.0 - C)]

    for i in range(100):
        x_new = x_seq[-1] - 0.1*(x_seq[-1] - C)
        x_seq.append(x_new)
        dist_seq.append(abs(x_new - C))

    dist_seq = np.array(dist_seq)
    ax.semilogy(range(len(dist_seq)), dist_seq, 'b-', linewidth=2.5, label='$d_C(x_n)$')
    ax.fill_between(range(len(dist_seq)), 0, dist_seq, alpha=0.2, color='blue')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('(d) Fejér Monotone Sequence', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('Distance to $C$', fontsize=10)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('numerical_example.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
if __name__ == '__main__':
    print("Generating Figure 1: Fejér Monotone Concept...")
    fig_fejer_monotone_concept()

    print("Generating Figure 2: Projection Algorithm...")
    fig_projection_algorithm()

    print("Generating Figure 3: Convergence Behavior...")
    fig_convergence_behavior()

    print("Generating Figure 4: Fixed Point Iteration...")
    fig_fixed_point_iteration()

    print("Generating Figure 5: Quasi-Fejér Monotonicity...")
    fig_quasi_fejer()

    print("Generating Figure 6: Alternating Projections...")
    fig_alternating_projections()

    print("Generating Figure 7: Numerical Examples...")
    fig_numerical_example()

    print("\nAll figures generated successfully!")
