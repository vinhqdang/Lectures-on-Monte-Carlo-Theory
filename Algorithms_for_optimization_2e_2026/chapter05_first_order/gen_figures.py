"""
gen_figures.py  –  Chapter 5: First-Order Methods
Generates all figures needed for chapter05_slides.tex.
Run with: conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Output directory (relative to this script)
OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helper: Rosenbrock function and gradient
# ─────────────────────────────────────────────
def rosenbrock(x, y, b=1):
    return (1 - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, b=1):
    gx = -2*(1 - x) - 4*b*x*(y - x**2)
    gy = 2*b*(y - x**2)
    return np.array([gx, gy])

def banana(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def banana_grad(x, y):
    gx = -2*(1-x) - 400*x*(y - x**2)
    gy = 200*(y - x**2)
    return np.array([gx, gy])


# ─────────────────────────────────────────────
# Fig 1: Gradient descent path on a quadratic
# ─────────────────────────────────────────────
def fig_gradient_descent():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: narrow quadratic (zigzag behaviour)
    A = np.array([[10, 0], [0, 1]])
    def f_quad(x, y):
        return 0.5 * (A[0, 0]*x**2 + A[1, 1]*y**2)

    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = f_quad(X, Y)

    ax = axes[0]
    ax.contour(X, Y, Z, levels=15, cmap="Blues")
    ax.set_title("Gradient Descent on Narrow Quadratic", fontsize=10)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")

    # Run gradient descent
    pt = np.array([2.5, 2.5])
    alpha = 0.09
    path = [pt.copy()]
    for _ in range(30):
        g = A @ pt
        pt = pt - alpha * g
        path.append(pt.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "r-o", markersize=3, linewidth=1.2, label="GD path")
    ax.plot(0, 0, "k*", markersize=10, label="minimum")
    ax.legend(fontsize=8)

    # Right: well-conditioned
    B = np.array([[2, 0], [0, 2]])
    Z2 = 0.5 * (B[0, 0]*X**2 + B[1, 1]*Y**2)
    ax2 = axes[1]
    ax2.contour(X, Y, Z2, levels=10, cmap="Blues")
    ax2.set_title("Gradient Descent on Isotropic Quadratic", fontsize=10)
    ax2.set_xlabel(r"$x_1$"); ax2.set_ylabel(r"$x_2$")

    pt2 = np.array([2.5, 2.5])
    alpha2 = 0.2
    path2 = [pt2.copy()]
    for _ in range(15):
        g2 = B @ pt2
        pt2 = pt2 - alpha2 * g2
        path2.append(pt2.copy())
    path2 = np.array(path2)
    ax2.plot(path2[:, 0], path2[:, 1], "r-o", markersize=3, linewidth=1.2, label="GD path")
    ax2.plot(0, 0, "k*", markersize=10, label="minimum")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gradient_descent_paths.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved gradient_descent_paths.pdf")


# ─────────────────────────────────────────────
# Fig 2: Conjugate gradient on quadratic
# ─────────────────────────────────────────────
def fig_conjugate_gradient():
    """Conjugate gradient converges in n=2 steps on a 2D quadratic."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_vec = np.array([1.0, 2.0])

    def f(x): return 0.5 * x @ A @ x + b_vec @ x
    def grad(x): return A @ x + b_vec

    # True min
    x_star = np.linalg.solve(A, -b_vec)

    x = np.linspace(-2, 2, 300)
    y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi])) for xi in x] for yi in y])

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(X, Y, Z, levels=20, cmap="Blues")
    ax.set_title("Conjugate Gradient on 2D Quadratic\n(converges in 2 steps)", fontsize=11)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")

    # CG iterations
    x0 = np.array([1.5, 1.5])
    g0 = grad(x0)
    d0 = -g0  # step 1
    # Exact line search on quadratic
    alpha0 = -(d0 @ grad(x0)) / (d0 @ A @ d0)
    x1 = x0 + alpha0 * d0

    g1 = grad(x1)
    # Fletcher-Reeves beta
    beta1 = (g1 @ g1) / (g0 @ g0)
    d1 = -g1 + beta1 * d0
    alpha1 = -(d1 @ grad(x1)) / (d1 @ A @ d1)
    x2 = x1 + alpha1 * d1

    pts = np.array([x0, x1, x2, x_star])
    ax.plot(pts[:, 0], pts[:, 1], "r-o", markersize=6, linewidth=1.8, label="CG path")
    for i, (px, py) in enumerate([(x0[0], x0[1]), (x1[0], x1[1]), (x2[0], x2[1])]):
        ax.annotate(f"$\\mathbf{{x}}^{{({i+1})}}$", (px, py),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.plot(*x_star, "k*", markersize=12, label="minimum")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "conjugate_gradient.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved conjugate_gradient.pdf")


# ─────────────────────────────────────────────
# Fig 3: Momentum vs gradient descent
# ─────────────────────────────────────────────
def fig_momentum():
    """Compare gradient descent with momentum on a narrow quadratic."""
    A = np.array([[8.0, 0.0], [0.0, 1.0]])
    x_star = np.zeros(2)

    def grad_f(x): return A @ x

    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (A[0, 0]*X**2 + A[1, 1]*Y**2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.contour(X, Y, Z, levels=12, cmap="Blues", alpha=0.6)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")

    # Gradient descent (no momentum)
    x0 = np.array([2.5, 2.5])
    alpha = 0.11
    pt = x0.copy()
    path_gd = [pt.copy()]
    for _ in range(25):
        pt = pt - alpha * grad_f(pt)
        path_gd.append(pt.copy())
    path_gd = np.array(path_gd)
    axes[0].plot(path_gd[:, 0], path_gd[:, 1], "r-o", markersize=3, linewidth=1.2)
    axes[0].plot(0, 0, "k*", markersize=10)
    axes[0].set_title("Gradient Descent", fontsize=11)

    # Momentum
    pt = x0.copy()
    v = np.zeros(2)
    beta = 0.8
    alpha_m = 0.06
    path_mom = [pt.copy()]
    for _ in range(25):
        v = beta * v - alpha_m * grad_f(pt)
        pt = pt + v
        path_mom.append(pt.copy())
    path_mom = np.array(path_mom)
    axes[1].plot(path_mom[:, 0], path_mom[:, 1], "g-o", markersize=3, linewidth=1.2)
    axes[1].plot(0, 0, "k*", markersize=10)
    axes[1].set_title("Gradient Descent with Momentum ($\\beta=0.8$)", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "momentum_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved momentum_comparison.pdf")


# ─────────────────────────────────────────────
# Fig 4: Nesterov momentum vs standard momentum
# ─────────────────────────────────────────────
def fig_nesterov():
    """Compare standard momentum with Nesterov momentum on Rosenbrock."""
    def f(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    def gradf(x):
        gx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        gy = 200*(x[1] - x[0]**2)
        return np.array([gx, gy])

    x = np.linspace(-1.5, 1.5, 300)
    y = np.linspace(-0.5, 2.0, 300)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100*(Y - X**2)**2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.contour(X, Y, np.log1p(Z), levels=20, cmap="Blues", alpha=0.6)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.plot(1, 1, "k*", markersize=10)

    n_iter = 300
    alpha = 0.001

    # Standard momentum
    x0 = np.array([-1.0, 1.0])
    pt = x0.copy(); v = np.zeros(2); beta = 0.9
    path_m = [pt.copy()]
    for _ in range(n_iter):
        v = beta * v - alpha * gradf(pt)
        pt = pt + v
        path_m.append(pt.copy())
    path_m = np.array(path_m)
    axes[0].plot(path_m[:, 0], path_m[:, 1], "r-", linewidth=0.8, alpha=0.8)
    axes[0].set_title("Standard Momentum", fontsize=11)

    # Nesterov momentum
    pt = x0.copy(); v = np.zeros(2)
    path_n = [pt.copy()]
    for _ in range(n_iter):
        v = beta * v - alpha * gradf(pt + beta * v)
        pt = pt + v
        path_n.append(pt.copy())
    path_n = np.array(path_n)
    axes[1].plot(path_n[:, 0], path_n[:, 1], "g-", linewidth=0.8, alpha=0.8)
    axes[1].set_title("Nesterov Momentum", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "nesterov_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved nesterov_comparison.pdf")


# ─────────────────────────────────────────────
# Fig 5: Adaptive methods comparison
# ─────────────────────────────────────────────
def fig_adaptive_methods():
    """Compare AdaGrad, RMSProp, Adam on a simple 2D problem."""
    def f(x): return x[0]**2 + 10*x[1]**2
    def gradf(x): return np.array([2*x[0], 20*x[1]])

    x_lin = np.linspace(-4, 4, 300)
    y_lin = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = X**2 + 10*Y**2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ["AdaGrad", "RMSProp", "Adam"]
    colors = ["purple", "orange", "blue"]

    x0 = np.array([3.5, 3.5])
    n_iter = 200
    eps = 1e-8

    paths = []

    # AdaGrad
    alpha = 0.5
    pt = x0.copy(); s = np.zeros(2)
    path = [pt.copy()]
    for _ in range(n_iter):
        g = gradf(pt)
        s += g**2
        pt = pt - alpha / (np.sqrt(s) + eps) * g
        path.append(pt.copy())
    paths.append(np.array(path))

    # RMSProp
    alpha = 0.1; gamma = 0.9
    pt = x0.copy(); s = np.zeros(2)
    path = [pt.copy()]
    for _ in range(n_iter):
        g = gradf(pt)
        s = gamma * s + (1 - gamma) * g**2
        pt = pt - alpha / (np.sqrt(s) + eps) * g
        path.append(pt.copy())
    paths.append(np.array(path))

    # Adam
    alpha = 0.3; gamma_v = 0.9; gamma_s = 0.999
    pt = x0.copy(); v = np.zeros(2); s = np.zeros(2)
    path = [pt.copy()]
    for k in range(1, n_iter + 1):
        g = gradf(pt)
        v = gamma_v * v + (1 - gamma_v) * g
        s = gamma_s * s + (1 - gamma_s) * g**2
        v_hat = v / (1 - gamma_v**k)
        s_hat = s / (1 - gamma_s**k)
        pt = pt - alpha * v_hat / (np.sqrt(s_hat) + eps)
        path.append(pt.copy())
    paths.append(np.array(path))

    for ax, path, title, col in zip(axes, paths, titles, colors):
        ax.contour(X, Y, Z, levels=15, cmap="Blues", alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], "-", color=col, linewidth=1.0, alpha=0.85)
        ax.plot(*x0, "rs", markersize=7)
        ax.plot(0, 0, "k*", markersize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "adaptive_methods.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved adaptive_methods.pdf")


# ─────────────────────────────────────────────
# Fig 6: Convergence curves comparison
# ─────────────────────────────────────────────
def fig_convergence_curves():
    """Loss vs iteration for GD, Momentum, Nesterov, AdaGrad, RMSProp, Adam."""
    def f(x): return x[0]**2 + 10*x[1]**2
    def gradf(x): return np.array([2*x[0], 20*x[1]])

    x0 = np.array([3.5, 3.5])
    f0 = f(x0)
    n_iter = 300
    eps = 1e-8

    results = {}

    # Gradient Descent
    pt = x0.copy()
    vals = [f(pt)]
    for _ in range(n_iter):
        pt = pt - 0.04 * gradf(pt)
        vals.append(f(pt))
    results["Gradient Descent"] = vals

    # Momentum
    pt = x0.copy(); v = np.zeros(2)
    vals = [f(pt)]
    for _ in range(n_iter):
        v = 0.9*v - 0.04*gradf(pt)
        pt = pt + v
        vals.append(f(pt))
    results["Momentum"] = vals

    # Nesterov
    pt = x0.copy(); v = np.zeros(2)
    vals = [f(pt)]
    for _ in range(n_iter):
        v = 0.9*v - 0.04*gradf(pt + 0.9*v)
        pt = pt + v
        vals.append(f(pt))
    results["Nesterov"] = vals

    # AdaGrad
    pt = x0.copy(); s = np.zeros(2)
    vals = [f(pt)]
    for _ in range(n_iter):
        g = gradf(pt)
        s += g**2
        pt = pt - 0.5/(np.sqrt(s)+eps)*g
        vals.append(f(pt))
    results["AdaGrad"] = vals

    # RMSProp
    pt = x0.copy(); s = np.zeros(2)
    vals = [f(pt)]
    for _ in range(n_iter):
        g = gradf(pt)
        s = 0.9*s + 0.1*g**2
        pt = pt - 0.1/(np.sqrt(s)+eps)*g
        vals.append(f(pt))
    results["RMSProp"] = vals

    # Adam
    pt = x0.copy(); v = np.zeros(2); s = np.zeros(2)
    vals = [f(pt)]
    for k in range(1, n_iter+1):
        g = gradf(pt)
        v = 0.9*v + 0.1*g
        s = 0.999*s + 0.001*g**2
        v_hat = v/(1-0.9**k)
        s_hat = s/(1-0.999**k)
        pt = pt - 0.3*v_hat/(np.sqrt(s_hat)+eps)
        vals.append(f(pt))
    results["Adam"] = vals

    fig, ax = plt.subplots(figsize=(8, 5))
    iters = np.arange(n_iter + 1)
    colors_map = {
        "Gradient Descent": "black",
        "Momentum": "blue",
        "Nesterov": "cyan",
        "AdaGrad": "purple",
        "RMSProp": "orange",
        "Adam": "red",
    }
    for name, vals in results.items():
        vals_arr = np.array(vals)
        vals_arr = np.maximum(vals_arr, 1e-15)
        ax.semilogy(iters, vals_arr, label=name, color=colors_map[name], linewidth=1.6)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$f(\mathbf{x}^{(k)})$", fontsize=12)
    ax.set_title("Convergence Comparison of First-Order Methods", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "convergence_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved convergence_comparison.pdf")


# ─────────────────────────────────────────────
# Fig 7: Hypergradient descent
# ─────────────────────────────────────────────
def fig_hypergradient():
    """Show hypergradient descent adapting the step size."""
    def f(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    def gradf(x):
        gx = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
        gy = 200*(x[1]-x[0]**2)
        return np.array([gx, gy])

    x_lin = np.linspace(-1.5, 1.5, 300)
    y_lin = np.linspace(-0.5, 2.0, 300)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = (1 - X)**2 + 100*(Y - X**2)**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: trajectories comparison
    ax = axes[0]
    ax.contour(X, Y, np.log1p(Z), levels=20, cmap="Blues", alpha=0.6)
    ax.plot(1, 1, "k*", markersize=12, label="minimum")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title("Hypergradient Descent\nvs Gradient Descent", fontsize=11)

    x0 = np.array([-0.5, 0.8])
    n_iter = 400
    mu = 1e-5

    # Standard gradient descent
    pt = x0.copy(); alpha = 0.001
    path_gd = [pt.copy()]
    for _ in range(n_iter):
        pt = pt - alpha * gradf(pt)
        path_gd.append(pt.copy())
    path_gd = np.array(path_gd)
    ax.plot(path_gd[:, 0], path_gd[:, 1], "r-", linewidth=0.7, alpha=0.8, label="GD")

    # Hypergradient descent
    pt = x0.copy(); alpha = 0.001; g_prev = np.zeros(2)
    path_hgd = [pt.copy()]
    alphas = [alpha]
    for _ in range(n_iter):
        g = gradf(pt)
        alpha = alpha + mu * (g @ g_prev)
        alpha = max(alpha, 1e-7)
        g_prev = g.copy()
        pt = pt - alpha * g
        path_hgd.append(pt.copy())
        alphas.append(alpha)
    path_hgd = np.array(path_hgd)
    ax.plot(path_hgd[:, 0], path_hgd[:, 1], "b-", linewidth=0.7, alpha=0.8, label="Hyper-GD")
    ax.legend(fontsize=9)

    # Right: step size evolution
    ax2 = axes[1]
    ax2.plot(alphas, color="navy", linewidth=1.5)
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel(r"Step factor $\alpha^{(k)}$", fontsize=11)
    ax2.set_title("Adaptive Step Size (Hypergradient)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "hypergradient.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved hypergradient.pdf")


# ─────────────────────────────────────────────
# Fig 8: Adadelta schematic
# ─────────────────────────────────────────────
def fig_adadelta():
    """Adadelta step factor comparison."""
    def f(x): return x[0]**2 + 10*x[1]**2
    def gradf(x): return np.array([2*x[0], 20*x[1]])

    x_lin = np.linspace(-4, 4, 300)
    y_lin = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = X**2 + 10*Y**2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(X, Y, Z, levels=15, cmap="Blues", alpha=0.5)

    x0 = np.array([3.5, 3.5])
    n_iter = 200
    eps = 1e-6

    # Adadelta
    gamma_s = 0.95; gamma_x = 0.95
    pt = x0.copy(); s = np.zeros(2); u = np.zeros(2)
    path = [pt.copy()]
    for _ in range(n_iter):
        g = gradf(pt)
        s = gamma_s*s + (1-gamma_s)*g**2
        dx = -(np.sqrt(u + eps) / np.sqrt(s + eps)) * g
        u = gamma_x*u + (1-gamma_x)*dx**2
        pt = pt + dx
        path.append(pt.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "m-", linewidth=1.5, label="Adadelta")
    ax.plot(*x0, "rs", markersize=7)
    ax.plot(0, 0, "k*", markersize=10)
    ax.set_title("Adadelta Optimizer", fontsize=12)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "adadelta.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved adadelta.pdf")


# ─────────────────────────────────────────────
# Fig 9: Algorithm overview diagram
# ─────────────────────────────────────────────
def fig_algorithm_tree():
    """Schematic tree of first-order methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(ax, cx, cy, text, color="#4472C4", fontsize=9, width=1.8, height=0.55):
        rect = plt.Rectangle((cx - width/2, cy - height/2), width, height,
                              color=color, zorder=3, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", zorder=4)
        return (cx, cy)

    def arrow(ax, p1, p2):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # Root
    r = box(ax, 5, 7.2, "First-Order Methods", color="#1F4E79", fontsize=10, width=2.6)

    # Level 1
    p1 = box(ax, 2, 5.8, "Gradient Descent", color="#2E75B6")
    p2 = box(ax, 5, 5.8, "+ Momentum", color="#2E75B6")
    p3 = box(ax, 8, 5.8, "Adaptive LR", color="#2E75B6")

    for p in [p1, p2, p3]:
        arrow(ax, (5, 6.94), p)

    # Level 2 - GD family
    a1 = box(ax, 1, 4.2, "Standard GD", color="#5B9BD5", width=1.6, fontsize=8)
    a2 = box(ax, 2.8, 4.2, "Conjugate\nGradient", color="#5B9BD5", width=1.6, fontsize=8)
    for a in [a1, a2]:
        arrow(ax, p1, a)

    # Level 2 - Momentum family
    b1 = box(ax, 4.2, 4.2, "Heavy Ball /\nMomentum", color="#70AD47", width=1.6, fontsize=8)
    b2 = box(ax, 6.0, 4.2, "Nesterov\nMomentum", color="#70AD47", width=1.6, fontsize=8)
    for b in [b1, b2]:
        arrow(ax, p2, b)

    # Level 2 - Adaptive family
    c1 = box(ax, 7.0, 4.2, "AdaGrad", color="#ED7D31", width=1.4, fontsize=8)
    c2 = box(ax, 8.5, 4.2, "RMSProp", color="#ED7D31", width=1.4, fontsize=8)
    for c in [c1, c2]:
        arrow(ax, p3, c)

    # Level 3
    d1 = box(ax, 4.2, 2.6, "Adadelta", color="#A9D18E", width=1.4, fontsize=8)
    d2 = box(ax, 6.0, 2.6, "Adam", color="#A9D18E", width=1.4, fontsize=8)
    arrow(ax, b2, d1)
    arrow(ax, c2, d2)

    # Hypergradient
    e1 = box(ax, 5.1, 1.1, "Hypergradient Descent", color="#FF0000", width=2.2, fontsize=8)
    arrow(ax, (5.1, 2.33), e1)

    ax.set_title("Taxonomy of First-Order Methods (Chapter 5)", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "algorithm_tree.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved algorithm_tree.pdf")


# ─────────────────────────────────────────────
# Fig 10: Step size effect on gradient descent
# ─────────────────────────────────────────────
def fig_step_size_effect():
    """Show effect of different step sizes on gradient descent."""
    def f(x): return x**4
    def df(x): return 4*x**3

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x_vals = np.linspace(-2, 2, 300)
    alphas = [0.01, 0.08, 0.3]
    titles = [r"$\alpha=0.01$ (too small)", r"$\alpha=0.08$ (good)", r"$\alpha=0.3$ (too large)"]
    x0 = 1.5

    for ax, alpha, title in zip(axes, alphas, titles):
        ax.plot(x_vals, f(x_vals), "b-", linewidth=2, label=r"$f(x)=x^4$")
        pt = x0
        path_x = [pt]
        path_y = [f(pt)]
        for _ in range(20):
            pt = pt - alpha * df(pt)
            path_x.append(pt)
            path_y.append(f(pt))
            if abs(pt) > 5:
                break
        ax.plot(path_x, path_y, "r-o", markersize=4, linewidth=1.2, label="GD iterates")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "step_size_effect.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved step_size_effect.pdf")


# ─────────────────────────────────────────────
# Fig 11: AdaGrad effective step factor decay
# ─────────────────────────────────────────────
def fig_adagrad_decay():
    """Show AdaGrad's effective step factor decay over iterations."""
    n_iters = 200
    alpha = 0.01
    eps = 1e-8

    # Simulate a 1D problem with constant gradient
    g = 1.0
    s = 0.0
    eff_steps = []
    for _ in range(n_iters):
        s += g**2
        eff_steps.append(alpha / (np.sqrt(s) + eps))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eff_steps, color="purple", linewidth=2, label="AdaGrad effective $\\alpha$")
    ax.axhline(y=alpha, color="gray", linestyle="--", alpha=0.7, label=f"Initial $\\alpha={alpha}$")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Effective Step Factor", fontsize=12)
    ax.set_title("AdaGrad: Monotonically Decreasing Effective Step Factor", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "adagrad_decay.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved adagrad_decay.pdf")


# ─────────────────────────────────────────────
# Fig 12: Bias correction in Adam
# ─────────────────────────────────────────────
def fig_adam_bias_correction():
    """Show bias correction effect in Adam."""
    n_iters = 50
    gamma_v = 0.9
    gamma_s = 0.999
    g = 1.0

    v = 0.0; s = 0.0
    v_raw = []; s_raw = []
    v_corrected = []; s_corrected = []

    for k in range(1, n_iters + 1):
        v = gamma_v * v + (1 - gamma_v) * g
        s = gamma_s * s + (1 - gamma_s) * g**2
        v_raw.append(v)
        s_raw.append(s)
        v_corrected.append(v / (1 - gamma_v**k))
        s_corrected.append(s / (1 - gamma_s**k))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    iters = np.arange(1, n_iters + 1)

    axes[0].plot(iters, v_raw, "r--", linewidth=2, label="Raw $v^{(k)}$")
    axes[0].plot(iters, v_corrected, "b-", linewidth=2, label="Corrected $\\hat{v}^{(k)}$")
    axes[0].axhline(g, color="gray", linestyle=":", label=f"True mean = {g}")
    axes[0].set_title("1st Moment (gradient estimate)", fontsize=11)
    axes[0].set_xlabel("Iteration"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, s_raw, "r--", linewidth=2, label="Raw $s^{(k)}$")
    axes[1].plot(iters, s_corrected, "b-", linewidth=2, label="Corrected $\\hat{s}^{(k)}$")
    axes[1].axhline(g**2, color="gray", linestyle=":", label=f"True variance = {g**2}")
    axes[1].set_title("2nd Moment (gradient variance estimate)", fontsize=11)
    axes[1].set_xlabel("Iteration"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Adam Bias Correction ($\\gamma_v=0.9$, $\\gamma_s=0.999$)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "adam_bias_correction.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved adam_bias_correction.pdf")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for Chapter 5: First-Order Methods...")
    fig_gradient_descent()
    fig_conjugate_gradient()
    fig_momentum()
    fig_nesterov()
    fig_adaptive_methods()
    fig_convergence_curves()
    fig_hypergradient()
    fig_adadelta()
    fig_algorithm_tree()
    fig_step_size_effect()
    fig_adagrad_decay()
    fig_adam_bias_correction()
    print("\nAll figures saved to:", OUTDIR)
