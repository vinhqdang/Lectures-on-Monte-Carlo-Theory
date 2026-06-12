"""
gen_figures.py  —  Chapter 10: Constraints
Generates all figures needed by chapter10_slides.tex.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name, fig=None, tight=True):
    if fig is None:
        fig = plt.gcf()
    if tight:
        fig.tight_layout()
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 1 — Constraint effect: unconstrained, constrained same sol, constrained new sol
# ──────────────────────────────────────────────────────────────────────────────
def fig_constraint_effect():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.linspace(-1.5, 4.5, 400)

    # Unconstrained
    ax = axes[0]
    y = (x - 1)**2 + 0.5
    ax.plot(x, y, 'steelblue', lw=2)
    ax.scatter([1], [0.5], color='steelblue', s=60, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title('Unconstrained')
    ax.annotate('$x^*$', xy=(1, 0.5), xytext=(1.5, 0.8), fontsize=10)
    ax.set_ylim(-0.2, 6)

    # Constrained — same solution
    ax = axes[1]
    ax.plot(x, y, 'steelblue', lw=2)
    ax.axvspan(-0.5, 3.5, alpha=0.08, color='gray')
    ax.axvline(-0.5, color='k', lw=1.5, label='$a$')
    ax.axvline(3.5, color='k', lw=1.5, label='$b$')
    ax.scatter([1], [0.5], color='steelblue', s=60, zorder=5)
    ax.set_xlabel('$x$')
    ax.set_title('Constrained, Same Solution')
    ax.annotate('$x^*$', xy=(1, 0.5), xytext=(1.5, 0.8), fontsize=10)
    ax.text(-0.5, -0.15, '$a$', ha='center', fontsize=10)
    ax.text(3.5, -0.15, '$b$', ha='center', fontsize=10)
    ax.set_ylim(-0.2, 6)

    # Constrained — new solution
    ax = axes[2]
    ax.plot(x, y, 'steelblue', lw=2)
    ax.axvspan(2.0, 4.5, alpha=0.08, color='gray')
    ax.axvline(2.0, color='k', lw=1.5)
    ax.axvline(4.5, color='k', lw=1.5)
    ax.scatter([2.0], [(2.0-1)**2+0.5], color='steelblue', s=60, zorder=5)
    ax.set_xlabel('$x$')
    ax.set_title('Constrained, New Solution')
    ax.annotate('$x^*$', xy=(2.0, 1.5), xytext=(2.6, 2.0), fontsize=10)
    ax.text(2.0, -0.15, '$a$', ha='center', fontsize=10)
    ax.text(4.5, -0.15, '$b$', ha='center', fontsize=10)
    ax.set_ylim(-0.2, 6)

    savefig('constraint_effect.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 2 — Bound-constrained transform example (Example 10.1)
# x in [2,6], transform t_{2,6}
# ──────────────────────────────────────────────────────────────────────────────
def fig_bound_transform():
    def t(xhat, a, b):
        return a + (b - a) * (np.sin(xhat)**2)

    a, b = 2, 6
    x_orig = np.linspace(-1, 16, 600)
    y_orig = x_orig * np.sin(x_orig)

    xhat_vals = np.linspace(-6, 6, 600)
    x_mapped = t(xhat_vals, a, b)
    y_trans = x_mapped * np.sin(x_mapped)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    y_feasible = np.where((x_orig >= 2) & (x_orig <= 6), y_orig, np.nan)
    ax.plot(x_orig, y_orig, 'k', lw=1, alpha=0.3)
    ax.plot(x_orig, y_feasible, 'steelblue', lw=2)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x) = x\\sin(x)$')
    ax.set_title('$f(x) = x\\sin(x)$')

    ax = axes[1]
    ax.plot(xhat_vals, y_trans, 'steelblue', lw=2)
    # Mark the two minima approx at xhat ~0.242 and 4.139 (mapped to x~4.914)
    for xh in [0.242, 4.139]:
        xv = t(xh, a, b)
        ax.scatter([xh], [xv * np.sin(xv)], color='steelblue', s=60, zorder=5)
    ax.set_xlabel('$\\hat{x}$'); ax.set_ylabel('$(f \\circ t_{2,6})(\\hat{x})$')
    ax.set_title('$(f \\circ t_{2,6})(\\hat{x})$')

    savefig('bound_transform_ex10_1.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 3 — Lagrange multipliers: contour plot with constraint curve (Example 10.3)
# ──────────────────────────────────────────────────────────────────────────────
def fig_lagrange_contour():
    x1 = np.linspace(-0.5, 3.5, 400)
    x2 = np.linspace(-0.5, 3.0, 400)
    X1, X2 = np.meshgrid(x1, x2)

    f = -np.exp(-((X1*X2 - 1.5)**2 + (X2 - 1.5)**2))

    fig, ax = plt.subplots(figsize=(5, 4))
    levels = np.linspace(f.min(), f.max(), 18)
    cs = ax.contour(X1, X2, f, levels=levels, cmap='viridis')

    # constraint: x1 = x2^2
    x2c = np.linspace(0.0, 2.5, 300)
    x1c = x2c**2
    mask = (x1c >= x1.min()) & (x1c <= x1.max())
    ax.plot(x1c[mask], x2c[mask], 'k', lw=2, label='$h(\\mathbf{x})=0$')

    # optimal point x* ≈ [1.358, 1.165]
    xs = np.array([1.358, 1.165])
    ax.scatter(*xs, color='red', s=80, zorder=5)
    ax.annotate('$\\mathbf{x}^*$', xy=xs, xytext=(xs[0]+0.15, xs[1]+0.1), fontsize=10)

    # gradients at x*
    gf = np.array([2*xs[1]*(-np.exp(-((xs[0]*xs[1]-1.5)**2+(xs[1]-1.5)**2)))*
                   (xs[0]*xs[1]-1.5)*xs[1],
                   0])  # simplified arrow direction
    gf_dir = np.array([0.25, 0.15])
    gh_dir = np.array([0.2, -0.35])
    ax.annotate('', xy=xs+gf_dir, xytext=xs,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=xs+gh_dir, xytext=xs,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(xs[0]+gf_dir[0]+0.05, xs[1]+gf_dir[1], '$\\nabla f$', color='red', fontsize=9)
    ax.text(xs[0]+gh_dir[0]+0.05, xs[1]+gh_dir[1], '$\\nabla h$', color='blue', fontsize=9)

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('Lagrange Multipliers — Example 10.3')
    ax.legend(fontsize=8, loc='upper left')
    savefig('lagrange_contour.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 4 — Feasible / active inequality region
# ──────────────────────────────────────────────────────────────────────────────
def fig_inequality_region():
    theta = np.linspace(0, 2*np.pi, 300)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, title, ineq_type in zip(axes,
                                    ['Inactive constraint\n$g(\\mathbf{x}^*) < 0$',
                                     'Active constraint\n$g(\\mathbf{x}^*) = 0$'],
                                    ['inactive', 'active']):
        # feasible region: disk of radius 1.5
        circ_x = 1.5*np.cos(theta)
        circ_y = 1.5*np.sin(theta)
        ax.fill(circ_x, circ_y, alpha=0.15, color='steelblue', label='Feasible $g \\leq 0$')
        ax.plot(circ_x, circ_y, 'steelblue', lw=1.5)

        # objective contours (centred at different points)
        if ineq_type == 'inactive':
            cx, cy = 0.3, 0.3
        else:
            cx, cy = 1.8, 0.0

        for r in [0.3, 0.6, 0.9, 1.2]:
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'gray', lw=0.7, alpha=0.5)

        if ineq_type == 'inactive':
            ax.scatter([cx], [cy], color='red', s=70, zorder=5)
            ax.annotate('$\\mathbf{x}^*$', xy=(cx, cy), xytext=(cx+0.1, cy+0.15), fontsize=10)
        else:
            xs = np.array([1.5, 0.0])
            ax.scatter(*xs, color='red', s=70, zorder=5)
            ax.annotate('$\\mathbf{x}^*$', xy=xs, xytext=(xs[0]+0.05, xs[1]+0.15), fontsize=10)

        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    savefig('inequality_region.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 5 — KKT conditions illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_kkt():
    fig, ax = plt.subplots(figsize=(5, 4))
    theta = np.linspace(0, 2*np.pi, 300)

    # Feasible region boundary
    ax.fill(1.8*np.cos(theta), 1.8*np.sin(theta), alpha=0.1, color='steelblue')
    ax.plot(1.8*np.cos(theta), 1.8*np.sin(theta), 'steelblue', lw=2, label='$g(\\mathbf{x})=0$')

    # objective contours
    cx, cy = 2.5, 0.0
    for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'gray', lw=0.6, alpha=0.5)

    # optimal on boundary
    xs = np.array([1.8, 0.0])
    ax.scatter(*xs, color='red', s=80, zorder=5)
    ax.annotate('$\\mathbf{x}^*$', xy=xs, xytext=(xs[0]+0.1, xs[1]+0.2), fontsize=11)

    # -grad f and grad g should be aligned (KKT)
    gf = np.array([xs[0]-cx, xs[1]-cy])
    gf = gf / np.linalg.norm(gf)
    gg = xs / np.linalg.norm(xs)
    ax.annotate('', xy=xs-0.6*gf, xytext=xs,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=xs+0.6*gg, xytext=xs,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(xs[0]-0.6*gf[0]+0.05, xs[1]-0.6*gf[1]-0.15, '$-\\nabla f$', color='red', fontsize=9)
    ax.text(xs[0]+0.6*gg[0]+0.05, xs[1]+0.6*gg[1]+0.05, '$\\nabla g$', color='blue', fontsize=9)

    ax.set_xlim(-3, 4); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('KKT Conditions at Boundary Optimum')
    ax.legend(fontsize=8)
    savefig('kkt_conditions.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 6 — Penalty method: quadratic penalty evolving with rho
# ──────────────────────────────────────────────────────────────────────────────
def fig_penalty_method():
    # 1D example: minimize x, subject to x >= 5
    # Penalised: x + rho * max(5-x,0)^2
    x = np.linspace(3, 8, 500)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    for ax, rho in zip(axes, [1, 5, 50]):
        penalty = rho * np.maximum(5 - x, 0)**2
        obj = x + penalty
        ax.plot(x, obj, 'steelblue', lw=2, label=f'$\\rho={rho}$')
        ax.axvline(5, color='k', lw=1.2, ls='--', label='$x=5$')
        xstar = 5 - 1.0/(2*rho)
        ax.scatter([xstar], [xstar + rho*(5-xstar)**2], color='red', s=60, zorder=5)
        ax.set_title(f'$\\rho = {rho}$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$f(x) + \\rho p(x)$')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 12)

    savefig('penalty_method_rho.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 7 — Method of multipliers convergence (augmented Lagrangian)
# ──────────────────────────────────────────────────────────────────────────────
def fig_method_of_multipliers():
    # Illustrate convergence of lambda and constraint violation
    np.random.seed(42)
    iterations = np.arange(1, 11)
    lambda_true = 0.170
    lambdas = lambda_true * (1 - np.exp(-0.5 * iterations))
    violations = 0.5 * np.exp(-0.6 * iterations)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(iterations, lambdas, 'o-', color='steelblue', lw=2)
    axes[0].axhline(lambda_true, color='red', ls='--', lw=1, label='True $\\lambda$')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('$\\lambda$')
    axes[0].set_title('Multiplier Convergence')
    axes[0].legend(fontsize=8)

    axes[1].semilogy(iterations, violations, 's-', color='darkorange', lw=2)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('$|h(\\mathbf{x})|$')
    axes[1].set_title('Constraint Violation')

    savefig('method_of_multipliers_conv.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 8 — Interior point / barrier method
# ──────────────────────────────────────────────────────────────────────────────
def fig_interior_point_barrier():
    # Visualise log barrier: -log(-g(x)) for g(x) = x - 5 <= 0
    x = np.linspace(0.1, 4.99, 500)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.plot(x, -np.log(5 - x), 'steelblue', lw=2, label='$-\\ln(5-x)$')
    ax.axvline(5, color='k', lw=1.5, ls='--', label='Boundary $x=5$')
    ax.set_ylim(-3, 6)
    ax.set_xlabel('$x$'); ax.set_ylabel('Barrier value')
    ax.set_title('Log Barrier: $-\\ln(-g(\\mathbf{x}))$')
    ax.legend(fontsize=8)

    ax = axes[1]
    rho_vals = [0.5, 1.0, 10.0, 100.0]
    colors = ['purple', 'blue', 'green', 'orange']
    x2 = np.linspace(0.1, 4.9, 500)
    f_obj = x2  # objective: min x
    for rho, col in zip(rho_vals, colors):
        barrier = -np.log(5 - x2)
        total = f_obj + barrier / rho
        ax.plot(x2, total, color=col, lw=1.5, label=f'$\\rho={rho}$')
    ax.axvline(5, color='k', lw=1.2, ls='--')
    ax.set_ylim(-5, 15)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f + p/\\rho$')
    ax.set_title('Interior Point Penalised Objective')
    ax.legend(fontsize=7)

    savefig('interior_point_barrier.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 9 — Projected descent illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_projected_descent():
    fig, ax = plt.subplots(figsize=(5, 4))

    # Feasible half-plane: x1 + x2 <= 3
    x1 = np.linspace(-0.5, 3.5, 300)
    ax.fill_between(x1, -0.5*np.ones_like(x1), 3 - x1, alpha=0.12, color='steelblue',
                    label='Feasible: $x_1+x_2 \\leq 3$')
    ax.plot(x1, 3 - x1, 'steelblue', lw=2)

    # Gradient descent steps with projection
    pts = [(2.5, 2.0), (1.8, 1.2), (0.8, 0.6), (0.3, 0.2)]
    proj_pts = []
    for p in pts:
        x1v, x2v = p
        if x1v + x2v > 3:
            # project onto x1+x2=3
            shift = (x1v + x2v - 3) / 2
            proj_pts.append((x1v - shift, x2v - shift))
        else:
            proj_pts.append(p)

    all_pts = list(pts) + proj_pts
    traj_x = [pts[0][0]]
    traj_y = [pts[0][1]]
    for i, (p, q) in enumerate(zip(pts, proj_pts)):
        ax.annotate('', xy=q, xytext=p,
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
        traj_x.append(q[0])
        traj_y.append(q[1])

    ax.plot(traj_x, traj_y, 'o--', color='red', lw=1, ms=5)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('Projected Gradient Descent')
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
    ax.legend(fontsize=8)
    savefig('projected_descent.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 10 — Slack variable conversion illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_slack_variable():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    x = np.linspace(-1, 4, 300)
    g = x - 2   # g(x) = x - 2 <= 0  =>  x <= 2

    ax = axes[0]
    ax.plot(x, g, 'steelblue', lw=2)
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.axvline(2, color='red', lw=1.5, ls='--', label='$g(x)=0$ boundary')
    ax.fill_betweenx([-1.5, 0], [-1, -1], [2, 2], alpha=0.12, color='green', label='Feasible $g \\leq 0$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$g(x)$')
    ax.set_title('Inequality: $g(x) \\leq 0$')
    ax.legend(fontsize=8)

    ax = axes[1]
    s_vals = np.linspace(0, 3, 300)
    ax.plot(s_vals, s_vals, 'darkorange', lw=2, label='$s \\geq 0$')
    ax.set_xlabel('$s$'); ax.set_ylabel('Slack $s$')
    ax.set_title('Slack Variable: $g(\\mathbf{x}) + s^2 = 0$')
    ax.legend(fontsize=8)

    savefig('slack_variable.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 11 — Penalty functions comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_penalty_functions():
    g = np.linspace(-1.5, 2.5, 500)
    p_count = (g > 0).astype(float)
    p_quadratic = np.maximum(g, 0)**2
    p_abs = np.maximum(g, 0)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(g, p_count, 'steelblue', lw=2, label='Count: $(g>0)$')
    ax.plot(g, p_quadratic, 'darkorange', lw=2, label='Quadratic: $\\max(g,0)^2$')
    ax.plot(g, p_abs, 'green', lw=2, label='Absolute: $\\max(g,0)$')
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('$g(\\mathbf{x})$'); ax.set_ylabel('$p(\\mathbf{x})$')
    ax.set_title('Penalty Functions')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 3)
    savefig('penalty_functions.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 12 — Affine equality constraint: LQ decomposition sketch
# ──────────────────────────────────────────────────────────────────────────────
def fig_affine_equality():
    # Visualise a simple 2D affine constraint Ax = b as a line
    fig, ax = plt.subplots(figsize=(5, 4))
    x1 = np.linspace(-2, 5, 300)
    # Constraint: x1 + 2*x2 = 4  => x2 = (4 - x1)/2
    x2_line = (4 - x1) / 2.0

    # Objective contours: min x1^2 + x2^2
    xg = np.linspace(-2, 5, 200)
    X1g, X2g = np.meshgrid(xg, np.linspace(-2, 4, 200))
    Fg = X1g**2 + X2g**2
    ax.contour(X1g, X2g, Fg, levels=[0.5, 2, 5, 10, 20, 40], colors='gray', linewidths=0.8, alpha=0.6)

    ax.plot(x1, x2_line, 'steelblue', lw=2.5, label='$\\mathbf{Ax}=\\mathbf{b}$')

    # Constrained optimum: project origin onto the line x1+2x2=4
    # => x1* = 4/5, x2* = 8/5
    xs = np.array([4/5, 8/5])
    ax.scatter(*xs, color='red', s=80, zorder=5)
    ax.annotate('$\\mathbf{x}^*$', xy=xs, xytext=(xs[0]+0.2, xs[1]+0.2), fontsize=11)

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('Affine Equality Constraint $\\mathbf{Ax}=\\mathbf{b}$')
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 5); ax.set_ylim(-2, 4)
    savefig('affine_equality.pdf', fig)

# ──────────────────────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Chapter 10 figures...')
    fig_constraint_effect()
    fig_bound_transform()
    fig_lagrange_contour()
    fig_inequality_region()
    fig_kkt()
    fig_penalty_method()
    fig_method_of_multipliers()
    fig_interior_point_barrier()
    fig_projected_descent()
    fig_slack_variable()
    fig_penalty_functions()
    fig_affine_equality()
    print('Done.')
