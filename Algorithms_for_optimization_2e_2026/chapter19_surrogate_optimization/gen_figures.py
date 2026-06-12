"""
gen_figures.py  –  Generate all figures for Chapter 19: Surrogate Optimization
Algorithms for Optimization, 2nd ed., 2026 – Kochenderfer & Wheeler

Run:
    conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')

OUTDIR = "figures"

# ──────────────────────────────────────────────
# Gaussian Process helpers
# ──────────────────────────────────────────────
def sq_exp_kernel(x1, x2, ell=1.0, sigma_f=1.0):
    """Squared-exponential (RBF) kernel."""
    r2 = (x1 - x2) ** 2
    return sigma_f**2 * np.exp(-r2 / (2 * ell**2))

def gp_predict(X_train, y_train, X_test, ell=1.0, sigma_f=1.0, sigma_n=1e-6):
    """Return GP posterior mean and std at X_test."""
    n = len(X_train)
    K = np.array([[sq_exp_kernel(a, b, ell, sigma_f) for b in X_train]
                  for a in X_train]) + sigma_n**2 * np.eye(n)
    k_star = np.array([[sq_exp_kernel(xs, a, ell, sigma_f) for a in X_train]
                       for xs in X_test])
    k_ss = np.array([sq_exp_kernel(xs, xs, ell, sigma_f) for xs in X_test])
    alpha = np.linalg.solve(K, y_train)
    mu = k_star @ alpha
    v = np.linalg.solve(K, k_star.T)
    var = k_ss - np.sum(k_star * v.T, axis=1)
    var = np.maximum(var, 0.0)
    return mu, np.sqrt(var)

# ──────────────────────────────────────────────
# True test function (multimodal, for illustration)
# ──────────────────────────────────────────────
def true_f(x):
    return np.sin(x) + 0.3 * np.cos(3 * x)

# ═══════════════════════════════════════════════════════════
# Figure 1: Prediction-Based Exploration (4 panels)
# ═══════════════════════════════════════════════════════════
def fig_prediction_based():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)

    # Initial sample sets (growing)
    X_sets = [
        np.array([-2.0, 0.5]),
        np.array([-2.0, 0.5, -0.5]),
        np.array([-2.0, 0.5, -0.5, 1.8]),
        np.array([-2.0, 0.5, -0.5, 1.8, 2.5]),
    ]
    for ax, X_samp in zip(axes, X_sets):
        y_samp = true_f(X_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)
        ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                        alpha=0.25, color='steelblue', label='confidence')
        ax.plot(x_test, y_true, 'k-', lw=1.5, label='true')
        ax.plot(x_test, mu, 'b-', lw=1.5, label='predicted')
        ax.scatter(X_samp, y_samp, color='black', s=30, zorder=5, label='fit pts')
        # next sample = argmin of predicted mean
        next_x = x_test[np.argmin(mu)]
        ax.axvline(next_x, color='red', linestyle='--', lw=1, alpha=0.7)
        ax.scatter([next_x], [true_f(np.array([next_x]))[0]],
                   color='red', marker='*', s=80, zorder=6, label='next')
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('$y$', fontsize=9)
    # Legend on last panel
    handles = [
        mpatches.Patch(color='steelblue', alpha=0.4, label='confidence region'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='true'),
        plt.Line2D([0], [0], color='b', lw=1.5, label='predicted'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
        plt.Line2D([0], [0], color='none', marker='*', markerfacecolor='r',
                   markersize=8, label='next sample'),
    ]
    axes[-1].legend(handles=handles, fontsize=6, loc='upper right')
    fig.suptitle('Prediction-Based Exploration', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_prediction_based.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_prediction_based.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 2: Error-Based Exploration (4 panels)
# ═══════════════════════════════════════════════════════════
def fig_error_based():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)

    X_sets = [
        np.array([-2.0, 0.5]),
        np.array([-2.0, 0.5, 2.5]),
        np.array([-2.0, 0.5, 2.5, -0.8]),
        np.array([-2.0, 0.5, 2.5, -0.8, 1.3]),
    ]
    for ax, X_samp in zip(axes, X_sets):
        y_samp = true_f(X_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)
        ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                        alpha=0.25, color='steelblue', label='confidence')
        ax.plot(x_test, y_true, 'k-', lw=1.5)
        ax.plot(x_test, mu, 'b-', lw=1.5)
        ax.scatter(X_samp, y_samp, color='black', s=30, zorder=5)
        # next sample = argmax of sigma
        next_x = x_test[np.argmax(sigma)]
        ax.scatter([next_x], [true_f(np.array([next_x]))[0]],
                   color='darkorange', marker='D', s=60, zorder=6, label='sampled')
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('$y$', fontsize=9)
    handles = [
        mpatches.Patch(color='steelblue', alpha=0.4, label='confidence region'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='true'),
        plt.Line2D([0], [0], color='b', lw=1.5, label='predicted'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
        plt.Line2D([0], [0], color='none', marker='D', markerfacecolor='darkorange',
                   markersize=6, label='sampled'),
    ]
    axes[-1].legend(handles=handles, fontsize=6, loc='upper right')
    fig.suptitle('Error-Based Exploration', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_error_based.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_error_based.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 3: Lower Confidence Bound (LCB) Exploration
# ═══════════════════════════════════════════════════════════
def fig_lcb():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)
    alpha = 1.0  # trade-off parameter

    X_sets = [
        np.array([-2.0, 0.5]),
        np.array([-2.0, 0.5, -1.5]),
        np.array([-2.0, 0.5, -1.5, 1.0]),
        np.array([-2.0, 0.5, -1.5, 1.0, 2.2]),
    ]
    for ax, X_samp in zip(axes, X_sets):
        y_samp = true_f(X_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)
        lcb = mu - alpha * sigma
        ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                        alpha=0.2, color='steelblue')
        ax.plot(x_test, y_true, 'k-', lw=1.5)
        ax.plot(x_test, mu, 'b-', lw=1.5)
        ax.plot(x_test, lcb, 'g--', lw=1.2, label='LCB')
        ax.scatter(X_samp, y_samp, color='black', s=30, zorder=5)
        next_x = x_test[np.argmin(lcb)]
        ax.scatter([next_x], [true_f(np.array([next_x]))[0]],
                   color='green', marker='*', s=80, zorder=6)
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-3.5, 3.0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('$y$', fontsize=9)
    handles = [
        mpatches.Patch(color='steelblue', alpha=0.3, label='confidence region'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='true'),
        plt.Line2D([0], [0], color='b', lw=1.5, label='predicted'),
        plt.Line2D([0], [0], color='g', lw=1.2, linestyle='--', label='LCB'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
        plt.Line2D([0], [0], color='none', marker='*', markerfacecolor='g',
                   markersize=8, label='sampled'),
    ]
    axes[-1].legend(handles=handles, fontsize=6, loc='upper right')
    fig.suptitle('Lower Confidence Bound (LCB) Exploration  ($\\alpha=1$)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_lcb.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_lcb.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 4: Probability of Improvement – single panel illustration
# ═══════════════════════════════════════════════════════════
def fig_prob_improvement():
    fig, ax = plt.subplots(figsize=(7, 4))
    x_test = np.linspace(-3, 3, 400)
    y_true = true_f(x_test)

    X_samp = np.array([-2.0, -0.8, 0.3, 1.5])
    y_samp = true_f(X_samp)
    y_min = np.min(y_samp)
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)

    # Fill confidence region
    ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.25, color='steelblue', label='confidence region')
    ax.plot(x_test, y_true, 'k-', lw=2, label='true')
    ax.plot(x_test, mu, 'b-', lw=2, label='predicted')
    ax.scatter(X_samp, y_samp, color='black', s=50, zorder=5, label='fit points')

    # Query point
    x_query = 2.2
    mu_q, sigma_q = gp_predict(X_samp, y_samp, np.array([x_query]), ell=0.8)
    mu_q, sigma_q = float(mu_q), float(sigma_q)

    # Shade probability of improvement region in the function space (vertical strip)
    y_range = np.linspace(-3, y_min, 200)
    ax.axhline(y_min, color='gray', linestyle='--', lw=1.2, label=f'$y_{{\\min}}={y_min:.2f}$')
    ax.axvline(x_query, color='purple', linestyle=':', lw=1.5)
    ax.fill_betweenx(y_range, x_query - 0.12, x_query + 0.12,
                     alpha=0.5, color='royalblue', label='prob. improvement')
    ax.annotate('best so far', xy=(-1.5, y_min), xytext=(-2.8, y_min + 0.6),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray')
    ax.annotate('probability of\nimprovement', xy=(x_query, y_min - 0.15),
                xytext=(x_query + 0.4, y_min - 0.7),
                arrowprops=dict(arrowstyle='->', color='royalblue'), fontsize=8, color='royalblue')
    ax.annotate('query point', xy=(x_query, -2.5), xytext=(x_query + 0.3, -2.8),
                arrowprops=dict(arrowstyle='->', color='purple'), fontsize=8, color='purple')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.2, 2.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Probability of Improvement', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_prob_improvement.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_prob_improvement.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 5: PoI progression (4 panels – top: GP, bottom: PoI curve)
# ═══════════════════════════════════════════════════════════
def fig_poi_progression():
    fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True)
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)

    X_sets = [
        np.array([-2.0, 0.5]),
        np.array([-2.0, 0.5, -1.2]),
        np.array([-2.0, 0.5, -1.2, 2.0]),
        np.array([-2.0, 0.5, -1.2, 2.0, 1.0]),
    ]
    for j, X_samp in enumerate(X_sets):
        y_samp = true_f(X_samp)
        y_min = np.min(y_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)
        poi = np.where(sigma > 0, norm.cdf((y_min - mu) / sigma), 0.0)

        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        ax_top.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                            alpha=0.25, color='steelblue')
        ax_top.plot(x_test, y_true, 'k-', lw=1.5)
        ax_top.plot(x_test, mu, 'b-', lw=1.5)
        ax_top.scatter(X_samp, y_samp, color='black', s=25, zorder=5)
        ax_top.axhline(y_min, color='gray', linestyle='--', lw=0.8)
        # mark the sampled query (last point in each expanding set)
        if j > 0:
            ax_top.scatter([X_samp[-1]], [y_samp[-1]], color='red', s=40, zorder=6)
        ax_top.set_ylim(-2.8, 2.8)
        ax_top.set_ylabel('$y$', fontsize=8)
        ax_top.tick_params(labelsize=7)
        ax_top.grid(True, alpha=0.2)

        ax_bot.plot(x_test, poi, 'r-', lw=1.5)
        ax_bot.set_ylim(0, 1.1)
        ax_bot.set_xlabel('$x$', fontsize=8)
        ax_bot.set_ylabel('$P(f(\\mathbf{x}) \\leq y_{\\min})$', fontsize=7)
        ax_bot.tick_params(labelsize=7)
        ax_bot.grid(True, alpha=0.2)

    axes[0, 0].set_title('Initial', fontsize=8)
    axes[0, 1].set_title('Iter 1', fontsize=8)
    axes[0, 2].set_title('Iter 2', fontsize=8)
    axes[0, 3].set_title('Iter 3', fontsize=8)

    handles = [
        mpatches.Patch(color='steelblue', alpha=0.3, label='confidence region'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='true'),
        plt.Line2D([0], [0], color='b', lw=1.5, label='predicted'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='r',
                   markersize=5, label='sampled'),
    ]
    fig.legend(handles=handles, fontsize=7, loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle('Probability of Improvement – Progression', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_poi_progression.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_poi_progression.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 6: Expected Improvement illustration
# ═══════════════════════════════════════════════════════════
def fig_expected_improvement():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)

    X_sets = [
        np.array([-2.0, 0.5]),
        np.array([-2.0, 0.5, -1.2]),
        np.array([-2.0, 0.5, -1.2, 2.0]),
        np.array([-2.0, 0.5, -1.2, 2.0, 1.0]),
    ]
    for j, (ax, X_samp) in enumerate(zip(axes, X_sets)):
        y_samp = true_f(X_samp)
        y_min = np.min(y_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)
        # EI formula (equation 19.12)
        z = np.where(sigma > 0, (y_min - mu) / sigma, 0.0)
        ei = np.where(sigma > 0,
                      (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z),
                      0.0)

        ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                        alpha=0.25, color='steelblue')
        ax.plot(x_test, y_true, 'k-', lw=1.5)
        ax.plot(x_test, mu, 'b-', lw=1.5)
        ax.scatter(X_samp, y_samp, color='black', s=25, zorder=5)
        if j > 0:
            ax.scatter([X_samp[-1]], [y_samp[-1]], color='red', s=40, zorder=6)
        next_x = x_test[np.argmax(ei)]
        ax.axvline(next_x, color='orange', linestyle='--', lw=1.2, alpha=0.8)
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-2.8, 2.8)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('$y$', fontsize=9)
    handles = [
        mpatches.Patch(color='steelblue', alpha=0.3, label='confidence region'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='true'),
        plt.Line2D([0], [0], color='b', lw=1.5, label='predicted'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='r',
                   markersize=5, label='sampled'),
    ]
    axes[-1].legend(handles=handles, fontsize=6, loc='upper right')
    fig.suptitle('Expected Improvement Exploration', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_expected_improvement.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_expected_improvement.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 7: SafeOpt – safety regions and minimizer/expander sets
# ═══════════════════════════════════════════════════════════
def fig_safeopt_regions():
    """Illustrate the safe set S, potential minimizers M, expanders E."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    x_test = np.linspace(-4, 4, 400)

    # Simple function for illustration
    def f_safe(x):
        return 0.5 * np.sin(1.5 * x) + 0.2 * x

    y_safe_thresh = 0.6
    X_samp = np.array([-1.0, 0.0, 0.5, 1.5])
    y_samp = f_safe(X_samp)
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.9, sigma_f=0.8)

    beta_sqrt = np.sqrt(3.0)
    u = mu + beta_sqrt * sigma   # upper confidence bound
    l = mu - beta_sqrt * sigma   # lower confidence bound

    # Safe set: u(x) <= y_max
    safe_mask = u <= y_safe_thresh

    ax = axes[0]
    # Shade safe region green
    ax.fill_between(x_test, -2.5, 2.5, where=safe_mask,
                    alpha=0.35, color='limegreen', label='estimated safe $\\mathcal{S}$')
    ax.fill_between(x_test, l, u, alpha=0.30, color='steelblue', label='confidence interval')
    ax.plot(x_test, f_safe(x_test), 'k-', lw=2, label='objective $f$')
    ax.axhline(y_safe_thresh, color='gray', linestyle='--', lw=1.5, label='safety threshold')
    ax.scatter(X_samp, y_samp, color='black', s=40, zorder=5, label='fit points')
    ax.set_ylabel('$y$', fontsize=10)
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_title('Safe Region $\\mathcal{S}$ (green) Predicted by GP', fontsize=10, fontweight='bold')

    # Safety probability approximation
    ax2 = axes[1]
    p_safe = norm.cdf((y_safe_thresh - mu) / np.maximum(sigma, 1e-8))
    p_thresh = 0.75
    ax2.plot(x_test, p_safe, 'g-', lw=2)
    ax2.axhline(p_thresh, color='gray', linestyle='--', lw=1.2, label=f'$P_{{\\rm safe}}={p_thresh}$')
    ax2.set_ylabel('safety probability', fontsize=9)
    ax2.set_xlabel('$x$', fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_safeopt_regions.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_safeopt_regions.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 8: SafeOpt – potential minimizers M and expanders E
# ═══════════════════════════════════════════════════════════
def fig_safeopt_sets():
    fig, ax = plt.subplots(figsize=(8, 4))
    x_test = np.linspace(-4, 4, 400)

    def f_safe(x):
        return 0.5 * np.sin(1.5 * x) + 0.2 * x

    y_safe_thresh = 0.6
    X_samp = np.array([-1.5, -0.5, 0.3, 1.0, 1.8])
    y_samp = f_safe(X_samp)
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.9, sigma_f=0.8)

    beta_sqrt = np.sqrt(3.0)
    u = mu + beta_sqrt * sigma
    l = mu - beta_sqrt * sigma

    safe_mask = u <= y_safe_thresh
    # Potential minimizers: safe AND l < min(u[safe])
    if np.any(safe_mask):
        best_u = np.min(u[safe_mask])
        min_mask = safe_mask & (l < best_u)
    else:
        min_mask = np.zeros_like(safe_mask)

    # For expanders: simplified – boundary of safe region
    exp_mask = safe_mask & ~min_mask
    exp_mask_trim = np.zeros_like(exp_mask)
    idx = np.where(exp_mask)[0]
    if len(idx) > 4:
        exp_mask_trim[idx[:2]] = True
        exp_mask_trim[idx[-2:]] = True

    ax.fill_between(x_test, -2.5, 2.5, where=safe_mask,
                    alpha=0.3, color='limegreen', label='safe $\\mathcal{S}$')
    ax.fill_between(x_test, -2.5, 2.5, where=min_mask,
                    alpha=0.5, color='hotpink', label='minimizers $\\mathcal{M}$')
    ax.fill_between(x_test, l, u, alpha=0.25, color='steelblue', label='confidence region')
    ax.plot(x_test, f_safe(x_test), 'k-', lw=2, label='objective function')
    ax.axhline(y_safe_thresh, color='gray', linestyle='--', lw=1.2, label='safety threshold')
    # best upper bound line
    if np.any(safe_mask):
        ax.axhline(best_u, color='red', linestyle='-.', lw=1.0, label='best upper bound')
    ax.scatter(X_samp, y_samp, color='black', s=40, zorder=5, label='fit points')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_title('SafeOpt: Safe Set, Potential Minimizers $\\mathcal{M}$ (pink), Expanders $\\mathcal{E}$', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_safeopt_sets.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_safeopt_sets.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 9: SafeOpt progression – 4 iterations
# ═══════════════════════════════════════════════════════════
def fig_safeopt_progression():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
    x_test = np.linspace(-4, 4, 300)

    def f_safe(x):
        return 0.6 * np.sin(1.2 * x) - 0.1 * x

    y_safe_thresh = 0.5
    # Start with one safe point; expand
    X_history = [np.array([0.0]),
                 np.array([0.0, -0.8]),
                 np.array([0.0, -0.8, 0.9]),
                 np.array([0.0, -0.8, 0.9, -1.6])]

    for j, (ax, X_samp) in enumerate(zip(axes, X_history)):
        y_samp = f_safe(X_samp)
        mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.9, sigma_f=0.8)
        beta_sqrt = np.sqrt(3.0)
        u = mu + beta_sqrt * sigma
        l = mu - beta_sqrt * sigma
        safe_mask = u <= y_safe_thresh
        if np.any(safe_mask):
            best_u = np.min(u[safe_mask])
            min_mask = safe_mask & (l < best_u)
        else:
            min_mask = np.zeros_like(safe_mask)

        ax.fill_between(x_test, -2.5, 2.5, where=safe_mask,
                        alpha=0.3, color='limegreen')
        ax.fill_between(x_test, -2.5, 2.5, where=min_mask,
                        alpha=0.5, color='hotpink')
        ax.fill_between(x_test, l, u, alpha=0.25, color='steelblue')
        ax.plot(x_test, f_safe(x_test), 'k-', lw=1.5)
        ax.axhline(y_safe_thresh, color='gray', linestyle='--', lw=1.0)
        ax.scatter(X_samp, y_samp, color='black', s=25, zorder=5)
        ax.set_xlabel('$x$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.2)
        ax.set_title(f'Iter {j+1}', fontsize=9)

    axes[0].set_ylabel('$y$', fontsize=9)
    handles = [
        mpatches.Patch(color='limegreen', alpha=0.4, label='safe $\\mathcal{S}$'),
        mpatches.Patch(color='hotpink', alpha=0.6, label='minimizers $\\mathcal{M}$'),
        mpatches.Patch(color='steelblue', alpha=0.3, label='confidence'),
        plt.Line2D([0], [0], color='k', lw=1.5, label='objective $f$'),
        plt.Line2D([0], [0], color='gray', linestyle='--', lw=1.0, label='safety threshold'),
        plt.Line2D([0], [0], color='none', marker='o', markerfacecolor='k',
                   markersize=5, label='fit points'),
    ]
    fig.legend(handles=handles, fontsize=7, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('SafeOpt Progression (1-D example)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_safeopt_progression.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_safeopt_progression.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 10: Exercise 19.3 – GP with points at -1 and 1
# ═══════════════════════════════════════════════════════════
def fig_exercise_gp():
    """GP with training points at x=-1 and x=1 (zero-mean, sq-exp kernel)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    x_test = np.linspace(-5, 5, 500)
    X_samp = np.array([-1.0, 1.0])
    f_true = lambda x: (x - 2)**2 / 40.0 - 0.5
    y_samp = f_true(X_samp)
    y_true = f_true(x_test)
    y_min = np.min(y_samp)

    # GP with sq-exp kernel exp(-r^2/2)
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=1.0, sigma_f=1.0)
    poi = np.where(sigma > 0, norm.cdf((y_min - mu) / sigma), 0.0)
    z = np.where(sigma > 0, (y_min - mu) / sigma, 0.0)
    ei = np.where(sigma > 0,
                  (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z),
                  0.0)
    ei = np.maximum(ei, 0.0)

    # Panel 1: GP
    ax = axes[0]
    ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.3, color='steelblue', label='uncertainty')
    ax.plot(x_test, mu, 'b-', lw=2, label='predicted mean')
    ax.scatter(X_samp, y_samp, color='black', s=50, zorder=5, label='fitted points')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)
    ax.set_xlim(-5, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title('Gaussian Process', fontsize=10, fontweight='bold')

    # Panel 2: PoI
    ax = axes[1]
    ax.plot(x_test, poi, 'b-', lw=2)
    xi_poi = x_test[np.argmax(poi)]
    ax.axvline(xi_poi, color='red', linestyle='--', lw=1.2,
               label=f'max at $x={xi_poi:.2f}$')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('probability of improvement', fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title('Probability of Improvement', fontsize=10, fontweight='bold')

    # Panel 3: EI
    ax = axes[2]
    ax.plot(x_test, ei, 'r-', lw=2)
    xi_ei = x_test[np.argmax(ei)]
    ax.axvline(xi_ei, color='blue', linestyle='--', lw=1.2,
               label=f'max at $x={xi_ei:.2f}$')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('expected improvement', fontsize=9)
    ax.set_xlim(-5, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title('Expected Improvement', fontsize=10, fontweight='bold')

    fig.suptitle('Exercise 19.3: $f(x)=(x-2)^2/40-0.5$, samples at $x=\\pm 1$',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_exercise_gp.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig_exercise_gp.pdf  |  max PoI at x={xi_poi:.2f}, max EI at x={xi_ei:.2f}")

# ═══════════════════════════════════════════════════════════
# Figure 11: GP Surrogate overview (intro diagram)
# ═══════════════════════════════════════════════════════════
def fig_gp_surrogate_overview():
    """Simple illustration: true function vs GP surrogate with uncertainty."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x_test = np.linspace(-3, 3, 300)

    def f(x): return np.sin(x) + 0.3 * np.cos(2.5 * x)

    X_samp = np.array([-2.5, -1.0, 0.2, 1.5, 2.8])
    y_samp = f(X_samp)
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.7)

    ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.25, color='steelblue', label='$\\pm 2\\sigma$ (confidence)')
    ax.plot(x_test, f(x_test), 'k-', lw=2, label='true $f(x)$')
    ax.plot(x_test, mu, 'b-', lw=2, label='GP mean $\\hat{\\mu}(x)$')
    ax.scatter(X_samp, y_samp, color='black', s=60, zorder=5, label='observations')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Gaussian Process Surrogate Model', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_gp_surrogate_overview.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_gp_surrogate_overview.pdf")

# ═══════════════════════════════════════════════════════════
# Figure 12: EI derivation diagram (normal PDF at query point)
# ═══════════════════════════════════════════════════════════
def fig_ei_derivation():
    """Show the local normal distribution at a query point and improvement region."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_test = np.linspace(-3, 3, 300)
    y_true = true_f(x_test)

    X_samp = np.array([-2.0, -0.5, 0.8, 1.8])
    y_samp = true_f(X_samp)
    y_min = float(np.min(y_samp))
    mu, sigma = gp_predict(X_samp, y_samp, x_test, ell=0.8)

    ax.fill_between(x_test, mu - 2*sigma, mu + 2*sigma,
                    alpha=0.2, color='steelblue')
    ax.plot(x_test, y_true, 'k-', lw=2, label='true')
    ax.plot(x_test, mu, 'b-', lw=2, label='predicted')
    ax.scatter(X_samp, y_samp, color='black', s=50, zorder=5, label='fit points')
    ax.axhline(y_min, color='gray', linestyle='--', lw=1.3,
               label=f'$y_{{\\min}} = {y_min:.2f}$')

    # Query point
    x_q_idx = np.argmin(mu - 1.5 * sigma)
    x_q = x_test[x_q_idx]
    mu_q = float(mu[x_q_idx])
    sig_q = float(sigma[x_q_idx])

    # Plot local normal distribution (rotated to be vertical at x_q)
    y_range = np.linspace(mu_q - 3*sig_q, mu_q + 3*sig_q, 200)
    pdf_vals = norm.pdf(y_range, mu_q, sig_q)
    pdf_scale = 0.4  # scale for display
    ax.plot(x_q + pdf_vals * pdf_scale, y_range, 'm-', lw=1.5, label='local dist.')
    # Shade improvement region
    y_imp = y_range[y_range < y_min]
    pdf_imp = norm.pdf(y_imp, mu_q, sig_q)
    ax.fill_betweenx(y_imp, x_q, x_q + pdf_imp * pdf_scale,
                     alpha=0.5, color='magenta', label='EI region')
    ax.axvline(x_q, color='purple', linestyle=':', lw=1.2)

    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_title('Expected Improvement: Local Distribution at Query Point', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_ei_derivation.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_ei_derivation.pdf")

# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    fig_gp_surrogate_overview()
    fig_prediction_based()
    fig_error_based()
    fig_lcb()
    fig_prob_improvement()
    fig_poi_progression()
    fig_expected_improvement()
    fig_ei_derivation()
    fig_safeopt_regions()
    fig_safeopt_sets()
    fig_safeopt_progression()
    fig_exercise_gp()

    print("\nAll figures generated successfully.")
