"""
gen_figures.py  --  Generate all figures for Chapter 17: Surrogate Models
Algorithms for Optimization, 2nd ed., Kochenderfer & Wheeler (2026)

Requires: matplotlib, numpy, scipy
Run:  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import linalg
import os

# Output directory
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Common style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.4,
    'figure.dpi': 150,
})

BLUE  = '#2196F3'
BLACK = '#111111'
RED   = '#E53935'
GREEN = '#43A047'


# ── Helper: pseudoinverse regression ───────────────────────────────────────
def regression(X, y, bases, lam=0.0):
    """Fit theta using basis functions and optional L2 regularisation."""
    B = np.array([[b(x) for b in bases] for x in X])
    if lam == 0.0:
        theta, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    else:
        A = B.T @ B + lam * np.eye(B.shape[1])
        theta = np.linalg.solve(A, B.T @ y)
    def model(x):
        return sum(theta[i] * bases[i](x) for i in range(len(theta)))
    return model


def poly_bases_1d(k):
    """Return list of 1-D polynomial basis functions x^0 ... x^k."""
    return [lambda x, p=p: x**p for p in range(k + 1)]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 -- Surrogate model concept
# ═══════════════════════════════════════════════════════════════════════════
def fig_surrogate_concept():
    np.random.seed(7)
    x_true = np.linspace(0, 2, 400)
    f_true = np.sin(3 * x_true) * np.exp(-0.4 * x_true) + 0.5

    # design points
    xd = np.array([0.2, 0.6, 1.0, 1.6])
    yd = np.sin(3 * xd) * np.exp(-0.4 * xd) + 0.5

    # surrogate: polynomial degree 3 fit
    bases = poly_bases_1d(3)
    model = regression(xd, yd, bases)
    y_hat = np.array([model(x) for x in x_true])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(x_true, f_true, color=BLACK, lw=1.8, label='true objective function')
    ax.plot(x_true, y_hat, color=BLUE, lw=1.6, linestyle='--', label='surrogate model')
    ax.scatter(xd, yd, color=BLACK, s=30, zorder=5, label='design points')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Surrogate Model Concept')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_surrogate_concept.pdf'))
    plt.close(fig)
    print('fig_surrogate_concept.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 -- Linear regression cases (m < n+1, m = n+1, non-independent, m > n+1)
# ═══════════════════════════════════════════════════════════════════════════
def fig_linear_regression_cases():
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))
    x_line = np.linspace(0, 3, 200)

    # m < n+1  (underdetermined: one point, line passing through it)
    ax = axes[0, 0]
    xd, yd = np.array([[1.5]]), np.array([1.2])
    theta = np.linalg.lstsq(np.column_stack([np.ones(1), xd[:, 0]]), yd, rcond=None)[0]
    ax.plot(x_line, theta[0] + theta[1] * x_line, color=BLUE)
    ax.scatter(xd[:, 0], yd, color=BLACK, s=30, zorder=5)
    ax.set_title('$m < n+1$'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(0, 3); ax.set_ylim(-0.5, 2.5)

    # m = n+1  (exactly determined)
    ax = axes[0, 1]
    xd2 = np.array([0.8, 2.2])
    yd2 = np.array([1.8, 0.6])
    theta2 = np.linalg.lstsq(np.column_stack([np.ones(2), xd2]), yd2, rcond=None)[0]
    ax.plot(x_line, theta2[0] + theta2[1] * x_line, color=BLUE)
    ax.scatter(xd2, yd2, color=BLACK, s=30, zorder=5)
    ax.set_title('$m = n+1$'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(0, 3); ax.set_ylim(-0.5, 2.5)

    # nonindependent points (two repeated x values)
    ax = axes[1, 0]
    xd3 = np.array([1.5, 1.5])
    yd3 = np.array([0.7, 1.8])
    theta3 = np.linalg.lstsq(np.column_stack([np.ones(2), xd3]), yd3, rcond=None)[0]
    ax.plot(x_line, theta3[0] + theta3[1] * x_line, color=BLUE)
    ax.scatter(xd3, yd3, color=BLACK, s=30, zorder=5)
    ax.set_title('nonindependent points'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(0, 3); ax.set_ylim(-0.5, 2.5)

    # m > n+1  (overdetermined)
    ax = axes[1, 1]
    xd4 = np.array([0.3, 0.8, 1.2, 1.8, 2.4, 2.8])
    yd4 = np.array([1.9, 1.4, 1.2, 0.9, 0.5, 0.3])
    theta4 = np.linalg.lstsq(np.column_stack([np.ones(6), xd4]), yd4, rcond=None)[0]
    ax.plot(x_line, theta4[0] + theta4[1] * x_line, color=BLUE)
    ax.scatter(xd4, yd4, color=BLACK, s=30, zorder=5)
    for xi, yi in zip(xd4, yd4):
        yhat = theta4[0] + theta4[1] * xi
        ax.plot([xi, xi], [yi, yhat], color=RED, lw=0.8, linestyle='-')
    ax.set_title('$m > n+1$'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(0, 3); ax.set_ylim(-0.5, 2.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_linear_regression_cases.pdf'))
    plt.close(fig)
    print('fig_linear_regression_cases.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 -- Polynomial is linear in higher-dimensional space
# ═══════════════════════════════════════════════════════════════════════════
def fig_polynomial_higher_dim():
    np.random.seed(3)
    xd = np.array([-3, -1, 0, 1, 2, 3.5])
    yd = xd**2 * 0.15 + np.random.randn(6) * 0.2

    x_fine = np.linspace(-4, 4, 300)
    bases = poly_bases_1d(2)
    model = regression(xd, yd, bases)
    y_fit = np.array([model(x) for x in x_fine])

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

    ax = axes[0]
    ax.plot(x_fine, y_fit, color=BLUE, lw=1.5)
    ax.scatter(xd, yd, color=BLACK, s=25, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('1-D polynomial fit')

    # 3-D lifted space (x, x^2, y)
    ax3 = axes[1]
    ax3.set_visible(False)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.scatter(xd, xd**2, yd, color=BLACK, s=25, zorder=5)
    # Plane in (x, x^2) basis
    xx = np.linspace(-4, 4, 20)
    xx2 = np.linspace(0, 16, 20)
    XX, XX2 = np.meshgrid(xx, xx2)
    # Retrieve theta
    B = np.array([[1, x, x**2] for x in xd])
    theta, _, _, _ = np.linalg.lstsq(B, yd, rcond=None)
    ZZ = theta[0] + theta[1] * XX + theta[2] * XX2
    ax3d.plot_surface(XX, XX2, ZZ, alpha=0.35, color=BLUE)
    ax3d.set_xlabel('$x$', fontsize=7); ax3d.set_ylabel('$x^2$', fontsize=7)
    ax3d.set_zlabel('$y$', fontsize=7)
    ax3d.set_title('lifted to $(x, x^2)$', fontsize=8)
    ax3d.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_polynomial_higher_dim.pdf'))
    plt.close(fig)
    print('fig_polynomial_higher_dim.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 -- Several radial basis functions
# ═══════════════════════════════════════════════════════════════════════════
def fig_radial_basis_functions():
    r = np.linspace(0.001, 2.5, 300)
    sigma = 1.0

    rbfs = [
        ('linear: $r$',                r,                        ),
        ('cubic: $r^3$',               r**3,                     ),
        ('thin plate spline: $r^2\\log r$', r**2 * np.log(r),   ),
        ('Gaussian: $e^{-r^2/2\\sigma^2}$', np.exp(-r**2 / (2*sigma**2)),),
        ('multiquadratic: $(r^2+\\sigma^2)^{1/2}$', np.sqrt(r**2 + sigma**2),),
        ('inv. multiquadratic: $(r^2+\\sigma^2)^{-1/2}$', 1/np.sqrt(r**2 + sigma**2),),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7, 4))
    for ax, (name, vals) in zip(axes.flatten(), rbfs):
        ax.plot(r, vals, color=BLACK, lw=1.4)
        ax.set_xlabel('$r$', fontsize=7)
        ax.set_ylabel('$\\psi$', fontsize=7)
        ax.set_title(name, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_radial_basis_functions.pdf'))
    plt.close(fig)
    print('fig_radial_basis_functions.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 -- Gaussian RBF fits with different bandwidths (noise-free)
# ═══════════════════════════════════════════════════════════════════════════
def fig_rbf_bandwidth_comparison():
    np.random.seed(0)
    xd = np.array([0.0, 0.2, 0.6, 0.9])
    yd = xd * np.sin(5 * xd)

    x_fine = np.linspace(0, 1, 400)
    f_true = x_fine * np.sin(5 * x_fine)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(x_fine, f_true, color=BLACK, lw=1.8, label='$x\\sin(5x)$')

    for c, label in [(2, '$\\psi=\\exp(-2r^2)$'),
                     (5, '$\\psi=\\exp(-5r^2)$'),
                     (10, '$\\psi=\\exp(-10r^2)$')]:
        bases = [lambda x, xi=xi, c=c: np.exp(-c * (x - xi)**2) for xi in xd]
        model = regression(xd, yd, bases)
        y_fit = np.array([model(x) for x in x_fine])
        ax.plot(x_fine, y_fit, lw=1.3, label=label)

    ax.scatter(xd, yd, color=BLACK, s=30, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('RBF Bandwidth Comparison (noise-free)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_rbf_bandwidth_comparison.pdf'))
    plt.close(fig)
    print('fig_rbf_bandwidth_comparison.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 -- L2 regularisation on noisy data
# ═══════════════════════════════════════════════════════════════════════════
def fig_regularisation():
    np.random.seed(42)
    m = 10
    xd = np.sort(np.random.rand(m))
    noise = np.random.randn(m) * 0.1
    yd = xd * np.sin(5 * xd) + noise

    x_fine = np.linspace(0, 1, 400)
    f_true = x_fine * np.sin(5 * x_fine)

    c = 5.0
    bases = [lambda x, xi=xi: np.exp(-c * (x - xi)**2) for xi in xd]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(x_fine, f_true, color=BLACK, lw=1.8, label='$x\\sin(5x)$')

    for lam, color, label in [(0.0, BLUE, '$\\lambda=0$'),
                               (0.1, RED,  '$\\lambda=0.1$'),
                               (0.5, GREEN,'$\\lambda=0.5$')]:
        model = regression(xd, yd, bases, lam=lam)
        y_fit = np.array([model(x) for x in x_fine])
        ax.plot(x_fine, y_fit, color=color, lw=1.3, label=label)

    ax.scatter(xd, yd, color=BLACK, s=25, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('$L_2$ Regularisation Effect')
    ax.legend(fontsize=7)
    ax.set_ylim(-2, 2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_regularisation.pdf'))
    plt.close(fig)
    print('fig_regularisation.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 -- Bias-variance trade-off (polynomial degree vs error)
# ═══════════════════════════════════════════════════════════════════════════
def fig_bias_variance():
    """
    Example 17.1 from book: f(x) = x/10 + sin(x)/4 + exp(-x^2)
    9 training points in [-4,4], generalisation evaluated in [-5,5].
    """
    def f_true(x):
        return x / 10 + np.sin(x) / 4 + np.exp(-x**2)

    x_train = np.linspace(-4, 4, 9)
    y_train = f_true(x_train)

    x_gen = np.linspace(-5, 5, 500)
    y_gen_true = f_true(x_gen)

    degrees = range(9)
    train_err = []
    gen_err   = []

    for k in degrees:
        bases = poly_bases_1d(k)
        model = regression(x_train, y_train, bases)
        y_pred_train = np.array([model(x) for x in x_train])
        y_pred_gen   = np.array([model(x) for x in x_gen])
        te = np.mean((y_train - y_pred_train)**2)
        ge = np.mean((y_gen_true - y_pred_gen)**2)
        train_err.append(te)
        gen_err.append(ge)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.semilogy(list(degrees), train_err, 'o-', color=BLUE, label='training error')
    ax.semilogy(list(degrees), gen_err,   's--', color=RED,  label='generalisation error')
    ax.set_xlabel('polynomial degree $k$')
    ax.set_ylabel('mean squared error')
    ax.set_title('Bias-Variance Trade-off')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_bias_variance.pdf'))
    plt.close(fig)
    print('fig_bias_variance.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 -- Polynomial fits for different degrees (3x3 grid, Example 17.1)
# ═══════════════════════════════════════════════════════════════════════════
def fig_poly_degree_comparison():
    def f_true(x):
        return x / 10 + np.sin(x) / 4 + np.exp(-x**2)

    x_train = np.linspace(-4, 4, 9)
    y_train = f_true(x_train)
    x_gen   = np.linspace(-5, 5, 400)
    y_gen   = f_true(x_gen)

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6), sharex=True, sharey=True)
    for ax, k in zip(axes.flatten(), range(9)):
        bases = poly_bases_1d(k)
        model = regression(x_train, y_train, bases)
        y_pred_train = np.array([model(x) for x in x_train])
        y_pred_gen   = np.array([model(x) for x in x_gen])
        te = np.mean((y_train - y_pred_train)**2)
        ge = np.mean((y_gen - y_pred_gen)**2)

        ax.plot(x_gen, y_gen, color=BLACK, lw=1.3)
        ax.plot(x_gen, y_pred_gen, color=BLUE, lw=1.1, linestyle='--')
        ax.scatter(x_train, y_train, color=BLACK, s=15, zorder=5)
        ax.set_xlim(-5, 5); ax.set_ylim(-1.2, 1.2)
        ax.set_title(f'$k={k}$', fontsize=8)
        ax.text(0.03, 0.03, f'$\\epsilon_{{train}}={te:.3f}$\n$\\epsilon_{{gen}}={ge:.3f}$',
                transform=ax.transAxes, fontsize=6, va='bottom')

    for ax in axes[2]:
        ax.set_xlabel('$x$', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('$y$', fontsize=7)

    fig.suptitle('Polynomial Surrogate: Degree vs Error', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_poly_degree_comparison.pdf'))
    plt.close(fig)
    print('fig_poly_degree_comparison.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 -- Holdout method schematic
# ═══════════════════════════════════════════════════════════════════════════
def fig_holdout_schematic():
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.axis('off')

    n = 30
    xs = np.linspace(0.02, 0.98, n)
    n_train = int(0.7 * n)
    colors_train = [BLUE] * n_train
    colors_test  = [RED]  * (n - n_train)
    colors = colors_train + colors_test

    for i, (x, c) in enumerate(zip(xs, colors)):
        ax.add_patch(plt.Circle((x, 0.6), 0.02, color=c, zorder=3))

    ax.annotate('', xy=(0.3, 0.2), xytext=(0.0, 0.2),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))
    ax.text(0.0, 0.1, 'train', color=BLUE, fontsize=8, ha='left')

    ax.annotate('', xy=(0.98, 0.2), xytext=(0.72, 0.2),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax.text(0.72, 0.1, 'test', color=RED, fontsize=8, ha='left')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title('Holdout Method', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label='train set'),
                        Patch(color=RED, label='test set')],
              loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_holdout_schematic.pdf'))
    plt.close(fig)
    print('fig_holdout_schematic.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 -- k-fold cross-validation schematic
# ═══════════════════════════════════════════════════════════════════════════
def fig_kfold_schematic():
    k = 5
    n = 20
    fig, axes = plt.subplots(k, 1, figsize=(5.5, 3.5))
    xs = np.arange(n)

    fold_size = n // k
    for fold_idx, ax in enumerate(axes):
        colors = []
        for i in range(n):
            if fold_idx * fold_size <= i < (fold_idx + 1) * fold_size:
                colors.append(RED)
            else:
                colors.append(BLUE)
        for i, c in enumerate(colors):
            ax.add_patch(plt.Rectangle((i * 0.05, 0.1), 0.045, 0.8,
                                        color=c, alpha=0.85))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(-0.04, 0.45, f'fold {fold_idx+1}', fontsize=7, ha='right', va='center')

    axes[0].set_title(f'{k}-fold Cross-Validation', fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label='train'), Patch(color=RED, label='validate')],
               loc='lower center', ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, 'fig_kfold_schematic.pdf'))
    plt.close(fig)
    print('fig_kfold_schematic.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 -- Cross-validation for hyperparameter lambda (Example 17.2)
# ═══════════════════════════════════════════════════════════════════════════
def fig_cv_lambda():
    np.random.seed(0)
    m = 10
    xd = np.random.rand(m)
    yd = np.sin(2 * xd) * np.cos(10 * xd) + np.random.randn(m) / 10.0

    def k_fold_cv(X, y, bases, lam, k=3):
        n = len(X)
        perm = np.random.permutation(n)
        fold_size = n // k
        errors = []
        for i in range(k):
            val_idx   = perm[i*fold_size:(i+1)*fold_size]
            train_idx = np.concatenate([perm[:i*fold_size], perm[(i+1)*fold_size:]])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model = regression(X_tr, y_tr, bases, lam=lam)
            preds = np.array([model(x) for x in X_val])
            errors.append(np.mean((y_val - preds)**2))
        return np.mean(errors)

    lambdas = np.logspace(-4, 2, 60)
    np.random.seed(1)  # fixed fold split
    c = 5.0
    bases = [lambda x, xi=xi: np.exp(-c * (x - xi)**2) for xi in xd]
    cv_errors = [k_fold_cv(xd, yd, bases, lam) for lam in lambdas]

    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.semilogx(lambdas, cv_errors, color=BLACK, lw=1.4)
    best_lam = lambdas[np.argmin(cv_errors)]
    ax.axvline(best_lam, color=RED, linestyle='--', lw=1.0, label=f'best $\\lambda \\approx {best_lam:.2f}$')
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('mean cross-validated MSE')
    ax.set_title('CV for Hyperparameter Selection')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_cv_lambda.pdf'))
    plt.close(fig)
    print('fig_cv_lambda.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 -- Bootstrap schematic (dots diagram)
# ═══════════════════════════════════════════════════════════════════════════
def fig_bootstrap_schematic():
    np.random.seed(5)
    n = 40
    fig, axes = plt.subplots(2, 1, figsize=(6, 2.4), gridspec_kw={'height_ratios': [1, 0.6]})

    # Original dataset
    ax = axes[0]
    ax.axis('off')
    xs = np.linspace(0.02, 0.98, n)
    for x in xs:
        ax.add_patch(plt.Circle((x, 0.7), 0.012, color=BLACK))

    # Bootstrap sample
    boot_idx = np.random.choice(n, size=n, replace=True)
    boot_xs  = np.sort(xs[boot_idx])
    for x in boot_xs:
        ax.add_patch(plt.Circle((x, 0.3), 0.012, color=BLUE))

    ax.text(-0.01, 0.7, 'original', fontsize=7, ha='right', va='center', color=BLACK)
    ax.text(-0.01, 0.3, 'bootstrap', fontsize=7, ha='right', va='center', color=BLUE)
    ax.set_xlim(-0.08, 1.02); ax.set_ylim(0, 1)
    ax.set_title('Bootstrap Sampling (with replacement)', fontsize=9)

    # Arrow diagram
    ax2 = axes[1]
    ax2.axis('off')
    ax2.annotate('train($\\bullet$)', xy=(0.33, 0.5), xytext=(0.01, 0.5),
                 fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.2))
    ax2.annotate('test($\\hat{f}, \\bullet$)', xy=(0.66, 0.5), xytext=(0.38, 0.5),
                 fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.2))
    ax2.annotate('generalisation error', xy=(0.99, 0.5), xytext=(0.68, 0.5),
                 fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.2))
    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_bootstrap_schematic.pdf'))
    plt.close(fig)
    print('fig_bootstrap_schematic.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 13 -- AIC/BIC model selection
# ═══════════════════════════════════════════════════════════════════════════
def fig_aic_bic():
    """Show AIC and BIC as a function of polynomial degree."""
    np.random.seed(7)
    m = 20
    xd = np.linspace(-3, 3, m)
    yd = np.sin(xd) + np.random.randn(m) * 0.3

    degrees = range(1, 10)
    aic_vals = []
    bic_vals = []

    for k in degrees:
        bases = poly_bases_1d(k)
        B = np.array([[b(x) for b in bases] for x in xd])
        theta, _, _, _ = np.linalg.lstsq(B, yd, rcond=None)
        resid = yd - B @ theta
        sse = np.sum(resid**2)
        n_params = k + 1
        n = m
        # AIC = 2k - 2*ln(L),  for Gaussian: -2*ln(L) ~ n*ln(SSE/n) + const
        aic = n * np.log(sse / n) + 2 * n_params
        bic = n * np.log(sse / n) + np.log(n) * n_params
        aic_vals.append(aic)
        bic_vals.append(bic)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(list(degrees), aic_vals, 'o-', color=BLUE, label='AIC')
    ax.plot(list(degrees), bic_vals, 's--', color=RED,  label='BIC')
    ax.set_xlabel('polynomial degree $k$')
    ax.set_ylabel('criterion value')
    ax.set_title('AIC and BIC for Model Selection')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_aic_bic.pdf'))
    plt.close(fig)
    print('fig_aic_bic.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 14 -- Multifidelity surrogate concept
# ═══════════════════════════════════════════════════════════════════════════
def fig_multifidelity():
    np.random.seed(2)
    x_fine = np.linspace(0, 1, 300)

    # High-fidelity: expensive, few samples
    f_high = lambda x: np.sin(4 * x) * np.exp(-x) + 0.3 * x
    xh = np.array([0.1, 0.35, 0.65, 0.9])
    yh = f_high(xh)

    # Low-fidelity: cheap, many samples (biased)
    f_low  = lambda x: 0.8 * np.sin(4 * x) * np.exp(-x) + 0.5 * x
    xl = np.linspace(0.05, 0.95, 15)
    yl = f_low(xl)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: both functions
    ax = axes[0]
    ax.plot(x_fine, f_high(x_fine), color=BLACK, lw=1.8, label='$f_h$ (high-fidelity)')
    ax.plot(x_fine, f_low(x_fine),  color=BLUE,  lw=1.4, linestyle='--', label='$f_\\ell$ (low-fidelity)')
    ax.scatter(xh, yh, color=BLACK, s=40, zorder=5, marker='*', label='$X_h$ samples')
    ax.scatter(xl, yl, color=BLUE,  s=15, zorder=4, alpha=0.7, label='$X_\\ell$ samples')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('High vs Low Fidelity Data')
    ax.legend(fontsize=6)

    # Right: multifidelity surrogate result
    ax = axes[1]
    # Simple linear correction: f_h ~ a0 + a1 * f_low_model
    bases_l = poly_bases_1d(4)
    model_l = regression(xl, yl, bases_l)
    y_low_at_xh = np.array([model_l(x) for x in xh])
    # Fit a0, a1
    A = np.column_stack([np.ones(len(xh)), y_low_at_xh])
    params, _, _, _ = np.linalg.lstsq(A, yh, rcond=None)
    a0, a1 = params

    y_low_fine = np.array([model_l(x) for x in x_fine])
    y_mf = a0 + a1 * y_low_fine

    ax.plot(x_fine, f_high(x_fine), color=BLACK, lw=1.8, label='$f_h$ (truth)')
    ax.plot(x_fine, y_mf, color=RED, lw=1.4, linestyle='--',
            label=f'multifidelity: $a_0={a0:.2f}, a_1={a1:.2f}$')
    ax.scatter(xh, yh, color=BLACK, s=40, zorder=5, marker='*')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('Multifidelity Surrogate')
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_multifidelity.pdf'))
    plt.close(fig)
    print('fig_multifidelity.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 15 -- Sinusoidal basis functions
# ═══════════════════════════════════════════════════════════════════════════
def fig_sinusoidal_bases():
    x = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for k in range(1, 5):
        ax.plot(x, np.sin(2 * np.pi * k * x), lw=1.2, label=f'$\\sin(2\\pi {k} x)$')
        ax.plot(x, np.cos(2 * np.pi * k * x), lw=1.0, linestyle='--',
                label=f'$\\cos(2\\pi {k} x)$', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('basis value')
    ax.set_title('Sinusoidal Basis Functions (Fourier)')
    ax.legend(fontsize=6, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_sinusoidal_bases.pdf'))
    plt.close(fig)
    print('fig_sinusoidal_bases.pdf  done')


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures for Chapter 17: Surrogate Models')
    print(f'Output directory: {FIG_DIR}')
    print('-' * 50)

    fig_surrogate_concept()
    fig_linear_regression_cases()
    fig_polynomial_higher_dim()
    fig_radial_basis_functions()
    fig_rbf_bandwidth_comparison()
    fig_regularisation()
    fig_bias_variance()
    fig_poly_degree_comparison()
    fig_holdout_schematic()
    fig_kfold_schematic()
    fig_cv_lambda()
    fig_bootstrap_schematic()
    fig_aic_bic()
    fig_multifidelity()
    fig_sinusoidal_bases()

    print('-' * 50)
    print('All figures generated successfully.')
