"""
Generate all figures for Chapter 4: Local Descent slides.
Uses matplotlib (backend='Agg') and pymupdf for PDF crops.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
import numpy as np
import os
import sys

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────
# Figure 1: Descent direction iteration illustration
# ─────────────────────────────────────────────────────────────
def fig_descent_iteration():
    fig, ax = plt.subplots(figsize=(7, 4))

    # Rosenbrock-like contours
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100*(Y - X**2)**2

    levels = [0.5, 2, 5, 15, 50, 150, 500]
    cs = ax.contour(X, Y, Z, levels=levels, cmap='viridis')

    # Simulated descent path
    path = np.array([
        [-1.5, 2.0],
        [-0.8, 0.4],
        [-0.1, 0.2],
        [0.4, 0.15],
        [0.8, 0.65],
        [1.0, 1.0],
    ])
    ax.plot(path[:, 0], path[:, 1], 'k-o', linewidth=1.5, markersize=5, label='descent path')
    for i, (px, py) in enumerate(path):
        ax.annotate(f'$\\mathbf{{x}}^{{({i+1})}}$', (px, py),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

    ax.plot(*path[-1], 'r*', markersize=12, label='local min')
    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_title('Descent Direction Iteration on Rosenbrock', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    plt.tight_layout()
    savefig("fig_descent_iteration.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 2: Step factor decay schemes
# ─────────────────────────────────────────────────────────────
def fig_step_factor_decay():
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(0, 30)
    alpha0 = 1.0
    gamma = 0.9
    beta = 0.1
    p = 0.5

    # Exponential decay
    exp_decay = gamma**k * alpha0
    ax.plot(k, exp_decay, 'b-o', markersize=4, label=r'Exp: $\alpha^{(k)}=\gamma^k\alpha^{(0)}$, $\gamma=0.9$')

    # Polynomial decay
    poly_decay = alpha0 / (1 + beta * k)**p
    ax.plot(k, poly_decay, 'r-s', markersize=4, label=r'Poly: $\alpha^{(k)}=\alpha^{(0)}/(1+\beta k)^p$, $\beta=0.1,p=0.5$')

    # Clamped decay
    alpha_min = 0.1
    clamped = np.maximum(alpha_min, gamma**k * alpha0)
    ax.plot(k, clamped, 'g-^', markersize=4, label=r'Clamped: $\max(\alpha_{\min}, \gamma^k\alpha^{(0)})$')

    ax.set_xlabel('Iteration $k$', fontsize=11)
    ax.set_ylabel('Step factor $\\alpha^{(k)}$', fontsize=11)
    ax.set_title('Step Factor Decay Schemes', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_step_factor_decay.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 3: Exact line search — 1D objective along direction
# ─────────────────────────────────────────────────────────────
def fig_exact_line_search():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    alpha = np.linspace(0, 4, 300)

    # Example: f(x + alpha*d) for a quadratic-like function
    phi = (alpha - 2.0)**2 + 0.5

    ax.plot(alpha, phi, 'b-', linewidth=2, label=r'$\phi(\alpha)=f(\mathbf{x}+\alpha\mathbf{d})$')
    ax.axvline(x=2.0, color='r', linestyle='--', linewidth=1.5, label=r'Exact min: $\alpha^*=2$')
    ax.scatter([2.0], [0.5], color='red', s=60, zorder=5)

    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$\phi(\alpha)$', fontsize=12)
    ax.set_title('Exact Line Search: Minimize Along Direction', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_exact_line_search.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 4: Sufficient decrease (first Wolfe condition) — Fig 4.1
# ─────────────────────────────────────────────────────────────
def fig_sufficient_decrease():
    fig, ax = plt.subplots(figsize=(7, 4))
    alpha = np.linspace(0, 8, 400)

    # f(x + alpha*d)  — a non-monotone function
    f_x = 5.0          # f(x) at alpha=0
    grad_phi0 = -1.5   # directional derivative at 0 (negative = descent)
    phi = f_x + 3.0 * np.sin(0.7 * alpha) * np.exp(-0.3 * alpha) + grad_phi0 * alpha * 0.4

    # Sufficient decrease line: f(x) + beta * alpha * grad_phi0
    beta = 1e-4
    suff = f_x + beta * alpha * grad_phi0

    # Tangent line: f(x) + alpha * grad_phi0
    tangent = f_x + alpha * grad_phi0

    ax.plot(alpha, phi, 'b-', linewidth=2.5, label=r'$f(\mathbf{x}+\alpha\mathbf{d})$')
    ax.plot(alpha, tangent, 'b--', linewidth=1.5, label=r'$f(\mathbf{x})+\alpha\nabla_\mathbf{d}f(\mathbf{x})$')
    ax.plot(alpha, suff, 'b:', linewidth=2.0, label=r'$f(\mathbf{x})+\beta\alpha\nabla_\mathbf{d}f(\mathbf{x})$')

    # Shade sufficient decrease region
    sufficient_mask = phi <= suff
    ax.fill_between(alpha, phi, suff, where=sufficient_mask,
                    alpha=0.2, color='green', label='sufficient decrease')

    ax.axhline(y=f_x, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title('Sufficient Decrease Condition (First Wolfe)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(f_x - 3, f_x + 1)
    ax.set_xlim(0, 8)
    plt.tight_layout()
    savefig("fig_sufficient_decrease.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 5: Backtracking line search on Rosenbrock — Fig 4.2
# ─────────────────────────────────────────────────────────────
def fig_backtracking_rosenbrock():
    fig, ax = plt.subplots(figsize=(6, 6))

    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 100*(X2 - X1**2)**2

    levels = np.logspace(-0.5, 3.5, 20)
    ax.contour(X1, X2, Z, levels=levels, cmap='YlOrRd', alpha=0.7)

    # Simulated backtracking path (approximate)
    pts = np.array([
        [-1.5, 1.5],
        [-0.6, 0.9],
        [0.0, 0.2],
        [0.5, 0.1],
        [0.75, 0.55],
        [0.9, 0.82],
        [0.97, 0.94],
        [1.0, 1.0],
    ])
    ax.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=1.8)
    ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=5)
    for i, (px, py) in enumerate(pts):
        ax.annotate(str(i+1), (px, py), fontsize=8,
                    textcoords='offset points', xytext=(5, 3))

    ax.plot(1, 1, 'r*', markersize=12, label='minimum $(1,1)$')
    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_title('Backtracking Line Search on Rosenbrock', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("fig_backtracking_rosenbrock.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 6: Curvature (second Wolfe) condition
# ─────────────────────────────────────────────────────────────
def fig_curvature_condition():
    fig, ax = plt.subplots(figsize=(7, 4))
    alpha = np.linspace(0, 6, 400)

    f_x = 5.0
    g0 = -2.0   # grad at alpha=0 (negative)
    phi = f_x + g0 * alpha + 0.8 * alpha**2 - 0.15 * alpha**3 + 0.008 * alpha**4

    # derivative (directional)
    dphi = g0 + 1.6 * alpha - 0.45 * alpha**2 + 0.032 * alpha**3

    beta1 = 1e-4
    sigma = 0.9

    suff_line = f_x + beta1 * alpha * g0
    curvature_bound = sigma * g0   # must have dphi >= sigma * g0 (weak Wolfe)

    ax.plot(alpha, phi, 'b-', linewidth=2, label=r'$\phi(\alpha)$')
    ax.plot(alpha, suff_line, 'g--', linewidth=1.5, label=r'Armijo: $f(\mathbf{x})+\beta\alpha g_0$')

    # Mark region where curvature is satisfied
    curv_ok = dphi >= curvature_bound
    ax.fill_between(alpha, ax.get_ylim()[0] if ax.get_ylim()[0] < phi.min() else phi.min()-0.5,
                    phi, where=curv_ok, alpha=0.15, color='orange', label='curvature satisfied')

    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$\phi(\alpha)$', fontsize=12)
    ax.set_title('Wolfe Conditions: Sufficient Decrease + Curvature', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_curvature_condition.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 7: Strong backtracking bracket & zoom (Figs 4.4, 4.5)
# ─────────────────────────────────────────────────────────────
def fig_strong_backtracking():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, title, pts_label in zip(axes,
                                    ['Bracket Phase', 'Zoom Phase'],
                                    [r'$\alpha^{(1)},\alpha^{(2)},\alpha^{(3)}$',
                                     r'$\alpha^{(3)},\alpha^{(4)},\alpha^{(5)}$']):
        alpha = np.linspace(0, 10, 500)
        f_x = 4.0
        g0 = -1.2
        phi = f_x + g0 * alpha + 0.5 * alpha**2 - 0.07 * alpha**3 + 0.003 * alpha**4

        beta = 1e-4
        sigma = 0.1
        suff = f_x + beta * alpha * g0

        ax.plot(alpha, phi, 'b-', linewidth=2, label=r'$f(\mathbf{x}+\alpha\mathbf{d})$')
        ax.plot(alpha, suff, 'b--', linewidth=1, label=r'$f(\mathbf{x})+\beta\alpha\nabla_\mathbf{d}f$')

        # color-code conditions
        dphi = g0 + alpha - 0.21 * alpha**2 + 0.012 * alpha**3
        cond1 = phi <= suff
        cond2 = np.abs(dphi) <= -sigma * g0
        cond3 = dphi >= 0

        ax.fill_between(alpha, phi.min()-0.5, phi, where=cond1 & ~cond2 & ~cond3,
                        alpha=0.2, color='blue', label=r'$f\leq$ suff (cond 1)')
        ax.fill_between(alpha, phi.min()-0.5, phi, where=cond1 & cond2,
                        alpha=0.3, color='green', label='Wolfe satisfied')
        ax.fill_between(alpha, phi.min()-0.5, phi, where=cond3,
                        alpha=0.15, color='red', label=r'$\nabla_\mathbf{d}f\geq 0$')

        ax.set_xlabel(r'$\alpha$', fontsize=11)
        ax.set_ylabel(r'$\phi(\alpha)$', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(phi.min()-0.3, f_x + 0.5)

    plt.suptitle('Strong Backtracking Line Search', fontsize=12)
    plt.tight_layout()
    savefig("fig_strong_backtracking.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 8: Trust region — circular regions on Rosenbrock
# ─────────────────────────────────────────────────────────────
def fig_trust_region():
    fig, ax = plt.subplots(figsize=(6, 6))

    x1 = np.linspace(-0.5, 1.5, 400)
    x2 = np.linspace(-0.2, 1.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 100*(X2 - X1**2)**2

    levels = np.logspace(-1, 3, 18)
    ax.contour(X1, X2, Z, levels=levels, cmap='YlOrRd', alpha=0.7)

    # Trust region centers and radii (simulated)
    centers = [
        (0.0, 0.5),
        (0.3, 0.1),
        (0.6, 0.35),
        (0.85, 0.72),
        (1.0, 1.0),
    ]
    radii = [0.35, 0.20, 0.18, 0.15, 0.08]

    for i, ((cx, cy), r) in enumerate(zip(centers, radii)):
        circle = plt.Circle((cx, cy), r, fill=False, color='blue', linewidth=1.5, linestyle='--')
        ax.add_patch(circle)
        ax.plot(cx, cy, 'b.', markersize=7)
        ax.annotate(f'$\\mathbf{{x}}^{{({i+1})}}$', (cx, cy),
                    textcoords='offset points', xytext=(5, 4), fontsize=8)

    # Descent path
    cpath = np.array(centers)
    ax.plot(cpath[:, 0], cpath[:, 1], 'k-', linewidth=1.2)
    ax.plot(1, 1, 'r*', markersize=12, label='minimum')

    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_title('Trust Region Method on Rosenbrock', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.2, 1.5)
    plt.tight_layout()
    savefig("fig_trust_region.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 9: Elliptical trust region (Fig 4.8)
# ─────────────────────────────────────────────────────────────
def fig_elliptical_trust_region():
    fig, ax = plt.subplots(figsize=(5, 5))

    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-2, 2, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1**2 + 5 * X2**2

    levels = [0.5, 1, 2, 4, 8]
    ax.contour(X1, X2, Z, levels=levels, cmap='Blues')

    # Circular trust region
    circle = plt.Circle((0.5, 0.3), 0.6, fill=False, color='blue',
                         linewidth=2, linestyle='--', label='Circular TR')
    ax.add_patch(circle)

    # Elliptical trust region aligned with gradient
    ellipse = Ellipse((0.5, 0.3), width=1.2, height=0.4, angle=30,
                       fill=False, color='red', linewidth=2, linestyle='-', label='Elliptical TR')
    ax.add_patch(ellipse)

    ax.plot(0.5, 0.3, 'ko', markersize=6, label='current $\\mathbf{x}$')
    ax.plot(0, 0, 'r*', markersize=10, label='minimum')

    # Red arrow showing elongated direction
    ax.annotate('', xy=(0.5+0.5, 0.3+0.17), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_title('Elliptical vs Circular Trust Region', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    plt.tight_layout()
    savefig("fig_elliptical_trust_region.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 10: Termination conditions comparison
# ─────────────────────────────────────────────────────────────
def fig_termination():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    k = np.arange(1, 31)

    # Simulated convergence data
    f_vals = 10 * np.exp(-0.3 * k) + 1.0
    grad_norms = 5 * np.exp(-0.25 * k)
    abs_improve = np.abs(np.diff(f_vals, prepend=f_vals[0] + 1))

    axes[0].semilogy(k, f_vals, 'b-o', markersize=4, label='$f(\\mathbf{x}^{(k)})$')
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='approx. minimum')
    axes[0].set_title('Objective Value', fontsize=11)
    axes[0].set_xlabel('Iteration $k$')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    eps_a = 0.05
    axes[1].semilogy(k, abs_improve, 'g-s', markersize=4, label='$|f^{(k)}-f^{(k+1)}|$')
    axes[1].axhline(y=eps_a, color='r', linestyle='--', label=f'$\\epsilon_a={eps_a}$')
    axes[1].set_title('Absolute Improvement', fontsize=11)
    axes[1].set_xlabel('Iteration $k$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    eps_g = 0.1
    axes[2].semilogy(k, grad_norms, 'm-^', markersize=4, label='$\\|\\nabla f(\\mathbf{x}^{(k)})\\|$')
    axes[2].axhline(y=eps_g, color='r', linestyle='--', label=f'$\\epsilon_g={eps_g}$')
    axes[2].set_title('Gradient Magnitude', fontsize=11)
    axes[2].set_xlabel('Iteration $k$')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Termination Condition Criteria', fontsize=12)
    plt.tight_layout()
    savefig("fig_termination.pdf")


# ─────────────────────────────────────────────────────────────
# Figure 11: Trust region improvement ratio illustration
# ─────────────────────────────────────────────────────────────
def fig_trust_region_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.linspace(-3, 3, 300)

    for ax, (a, b, title) in zip(axes, [
        (1.0, 0.5, 'Good model: $\\eta > \\eta_2$ (expand TR)'),
        (0.5, 2.0, 'Poor model: $\\eta < \\eta_1$ (shrink TR)'),
    ]):
        # Actual function
        f_actual = a * x**2 + 0.2 * x**4
        # Quadratic model
        f_model = b * x**2

        ax.plot(x, f_actual, 'b-', linewidth=2, label='actual $f$')
        ax.plot(x, f_model, 'r--', linewidth=2, label='quadratic model $\\hat{f}$')
        ax.axvline(0, color='gray', linewidth=0.8)

        # Trust region boundary
        delta = 1.2
        ax.axvspan(-delta, delta, alpha=0.1, color='green', label=f'TR $\\delta={delta}$')
        ax.axvline(-delta, color='green', linestyle=':', linewidth=1.5)
        ax.axvline( delta, color='green', linestyle=':', linewidth=1.5)

        ax.set_xlabel('step', fontsize=10)
        ax.set_ylabel('function value', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)

    plt.suptitle('Trust Region Improvement Ratio $\\eta = \\Delta f_{\\rm actual}/\\Delta f_{\\rm predicted}$', fontsize=11)
    plt.tight_layout()
    savefig("fig_trust_region_ratio.pdf")


# ─────────────────────────────────────────────────────────────
# Run all figure generators
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 4: Local Descent")
    fig_descent_iteration()
    fig_step_factor_decay()
    fig_exact_line_search()
    fig_sufficient_decrease()
    fig_backtracking_rosenbrock()
    fig_curvature_condition()
    fig_strong_backtracking()
    fig_trust_region()
    fig_elliptical_trust_region()
    fig_termination()
    fig_trust_region_ratio()
    print("All figures generated successfully.")
