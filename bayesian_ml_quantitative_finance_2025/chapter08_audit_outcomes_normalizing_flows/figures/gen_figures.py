"""
gen_figures.py -- Generate all figures for Chapter 8 slides
  "Bayesian Audit Outcome Model Selection Using Normalizing Flows"
  (Mongwe, Mbuvha & Marwala, 2025, Chapter 8)

ALL numerical experiments in this script are original, illustrative TOY
examples built by the slide author to explain the chapter's concepts
(Bayesian evidence, the harmonic mean estimator and its instability, and
how a normalizing-flow-style importance density stabilizes it). They are
NOT the book's actual dataset, models or results. Where the book's own
numbers are quoted in the slides, they are typed in directly from the
page images and are not recomputed here.

Uses matplotlib with the Agg backend; saves every figure as a vector PDF
into figures/.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
import os

FIGDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name, **kw):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches='tight', **kw)
    plt.close()
    print(f"  saved {path}")

np.random.seed(0)

# palette (consistent across figures)
BLUE    = "#2166ac"
LBLUE   = "#abd9e9"
ORANGE  = "#d6604d"
LORANGE = "#fddbc7"
GRAY    = "#888888"
GREEN   = "#1a9850"
PURPLE  = "#762a83"


# ============================================================================
# Figure 1 -- Bayesian model comparison / Occam's razor (illustrative)
#   Left panel : classic "evidence over the space of possible datasets"
#                cartoon (MacKay-style) showing why a simple model can beat
#                a complex model even though the complex model can fit more
#                datasets (including the observed one) reasonably well.
#   Right panel: bar chart of two TOY models' evidence values, indicating
#                which model is preferred.
# ============================================================================
def fig_occams_razor():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # --- left: p(D|M) over the space of possible datasets ---
    ax = axes[0]
    d = np.linspace(-6, 6, 600)
    # Simple model: concentrates its predictive mass on a narrow range of
    # datasets, so it is very confident there -> tall, narrow curve.
    simple = 1.05 * np.exp(-0.5 * (d - 0.3) ** 2 / 0.55 ** 2)
    simple /= np.trapezoid(simple, d)
    # Complex model: spreads its predictive mass over many more possible
    # datasets (it *could* explain many things) -> shorter, wider curve.
    complex_ = 1.0 * np.exp(-0.5 * (d - 0.3) ** 2 / 2.6 ** 2)
    complex_ /= np.trapezoid(complex_, d)

    ax.plot(d, simple, color=BLUE, lw=2.2, label='Simple model $M_1$')
    ax.plot(d, complex_, color=ORANGE, lw=2.2, label='Complex model $M_2$')
    ax.fill_between(d, simple, color=BLUE, alpha=0.12)
    ax.fill_between(d, complex_, color=ORANGE, alpha=0.12)

    d_obs = 0.3
    ax.axvline(d_obs, color=GRAY, ls='--', lw=1.3)
    ax.text(d_obs + 0.15, ax.get_ylim()[1] * 0.92, 'observed\ndata $D$',
            fontsize=8, color=GRAY)

    ax.set_xlabel('space of possible datasets (schematic)', fontsize=8.5)
    ax.set_ylabel(r'$P(D\,|\,M)$', fontsize=9)
    ax.set_title("Why evidence penalizes complexity", fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left', frameon=False)
    ax.set_yticks([])

    # --- right: toy evidence bar chart ---
    ax2 = axes[1]
    models = ['Simple\n$M_1$', 'Complex\n$M_2$']
    # illustrative (made-up) evidence values consistent with the left panel:
    # both models can fit the observed data, but the simple model was more
    # "confident" about it, so it gets a higher evidence value here.
    evidences = [0.041, 0.017]
    colors = [BLUE, ORANGE]
    bars = ax2.bar(models, evidences, color=colors, width=0.55, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, evidences):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}",
                  ha='center', fontsize=9)
    ax2.annotate('preferred\n(higher evidence)', xy=(0, evidences[0]),
                 xytext=(0.15, 0.052), fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax2.set_ylabel(r'illustrative evidence $Z_M = P(D\,|\,M)$', fontsize=8.5)
    ax2.set_title("Toy evidence comparison", fontsize=9.5)
    ax2.set_ylim(0, 0.065)

    fig.suptitle("Bayesian model comparison via the evidence (illustrative toy diagram)",
                  fontsize=10.5, y=1.03)
    fig.tight_layout()
    savefig("fig1_occams_razor_evidence.pdf")


# ============================================================================
# Figure 2 -- Hand-worked toy example: 4 posterior samples, harmonic mean
#   Shows how one small likelihood value dominates the reciprocal average.
# ============================================================================
def fig_toy4_harmonic_mean():
    labels = [r'$\theta^{(1)}$', r'$\theta^{(2)}$', r'$\theta^{(3)}$', r'$\theta^{(4)}$']
    L = np.array([0.20, 0.15, 0.18, 0.01])   # made-up toy likelihood values
    inv_L = 1.0 / L
    hm_estimate = 1.0 / np.mean(inv_L)
    simple_mean = np.mean(L)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    ax = axes[0]
    bars = ax.bar(labels, L, color=[BLUE, BLUE, BLUE, ORANGE], edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, L):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha='center', fontsize=9)
    ax.set_ylabel(r'$L(\theta^{(i)})$', fontsize=9.5)
    ax.set_title("4 toy posterior samples", fontsize=9.5)
    ax.set_ylim(0, 0.26)
    ax.annotate('rare small\nlikelihood value', xy=(3, 0.01), xytext=(1.7, 0.14),
                fontsize=8, color=ORANGE,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))

    ax2 = axes[1]
    bars2 = ax2.bar(labels, inv_L, color=[BLUE, BLUE, BLUE, ORANGE], edgecolor='black', linewidth=0.6)
    for b, v in zip(bars2, inv_L):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha='center', fontsize=9)
    ax2.set_ylabel(r'$1/L(\theta^{(i)})$', fontsize=9.5)
    ax2.set_title(r"Reciprocals -- one term dominates the mean", fontsize=9.2)
    ax2.text(0.02, 0.92,
             rf"$\hat{{Z}}_{{HM}} = 1 / \mathrm{{mean}}(1/L) = {hm_estimate:.3f}$" + "\n"
             rf"(vs. simple mean of $L$ = {simple_mean:.3f})",
             transform=ax2.transAxes, fontsize=8.3, va='top',
             bbox=dict(boxstyle='round', fc='white', ec=GRAY, lw=0.6))

    fig.suptitle("Toy illustration: the harmonic mean identity by hand", fontsize=10.5, y=1.03)
    fig.tight_layout()
    savefig("fig2_toy4_harmonic_mean.pdf")
    print(f"  toy4 HM estimate = {hm_estimate:.4f}, simple mean L = {simple_mean:.4f}")


# ============================================================================
# Toy Bayesian logistic regression helper (used by figs 3 and 5)
# ============================================================================
def make_toy_audit_data(n=200, informative_extra=False, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)     # "debt-to-revenue" (standardised)
    x2 = rng.normal(0, 1, n)     # "current ratio"   (standardised)
    x3 = rng.normal(0, 1, n)     # "capex % total expenditure" (noise-like)
    x4 = rng.normal(0, 1, n)     # "net operating surplus margin %" (noise-like)
    b0, b1, b2 = -0.2, 1.1, -0.9
    logit = b0 + b1 * x1 + b2 * x2
    if informative_extra:
        logit += 0.0  # kept as pure noise features by construction
    p = expit(logit)
    y = rng.binomial(1, p)
    return x1, x2, x3, x4, y


def neg_log_posterior(w, X, y, alpha=10.0):
    logits = X @ w
    # numerically stable Bernoulli log-likelihood
    ll = np.sum(y * logits - np.logaddexp(0.0, logits))
    lp = -0.5 * np.sum(w ** 2) / alpha ** 2
    return -(ll + lp)


def log_likelihood(w, X, y):
    logits = X @ w
    return np.sum(y * logits - np.logaddexp(0.0, logits))


def log_prior(w, alpha=10.0):
    d = len(w)
    return -0.5 * np.sum(w ** 2) / alpha ** 2 - 0.5 * d * np.log(2 * np.pi * alpha ** 2)


# ============================================================================
# Figure 3 -- Toy simulation: naive harmonic mean estimator variance vs a
#   flow-stabilized (learnt harmonic mean) estimator, for a small 2-parameter
#   toy Bayesian logistic regression model, compared against a ground-truth
#   evidence obtained by brute-force grid quadrature (feasible in 2D).
# ============================================================================
def fig_harmonic_mean_variance():
    # --- toy 2-parameter logistic regression: intercept b, slope w ---
    rng = np.random.default_rng(7)
    n = 22  # kept small so the likelihood magnitude stays numerically tame
    x = rng.normal(0, 1, n)
    b_true, w_true = 0.4, 1.6
    p_true = expit(b_true + w_true * x)
    y = rng.binomial(1, p_true)
    alpha = 10.0  # matches the Gaussian prior std used in the book (alpha=10)

    def loglik_grid(B, W):
        # B, W broadcastable grids; returns log-likelihood surface
        logits = B[..., None] * 0 + B[..., None] + W[..., None] * x  # placeholder, replaced below
        return None

    # build grid
    b_grid = np.linspace(-3.0, 4.0, 260)
    w_grid = np.linspace(-2.0, 5.5, 260)
    BB, WW = np.meshgrid(b_grid, w_grid, indexing='ij')
    db = b_grid[1] - b_grid[0]
    dw = w_grid[1] - w_grid[0]

    logits = BB[..., None] + WW[..., None] * x[None, None, :]
    loglik = np.sum(y[None, None, :] * logits - np.logaddexp(0.0, logits), axis=-1)
    logprior = (-0.5 * (BB ** 2 + WW ** 2) / alpha ** 2
                - np.log(2 * np.pi * alpha ** 2))
    log_unnorm_post = loglik + logprior

    # ground-truth evidence via quadrature: Z = sum(exp(log_unnorm_post))*db*dw
    m = log_unnorm_post.max()
    Z_true = np.exp(m) * np.sum(np.exp(log_unnorm_post - m)) * db * dw
    log_Z_true = np.log(Z_true)

    # discretised posterior probabilities for sampling
    post_w = np.exp(log_unnorm_post - m)
    post_w /= post_w.sum()
    flat_w = post_w.ravel()
    idx_b, idx_w = np.unravel_index(np.arange(BB.size), BB.shape)

    post_mean_b = np.sum(BB * post_w)
    post_mean_w = np.sum(WW * post_w)
    post_var_b = np.sum((BB - post_mean_b) ** 2 * post_w)
    post_var_w = np.sum((WW - post_mean_w) ** 2 * post_w)
    post_cov_bw = np.sum((BB - post_mean_b) * (WW - post_mean_w) * post_w)
    Sigma_post = np.array([[post_var_b, post_cov_bw], [post_cov_bw, post_var_w]])

    # "learnt/flow" importance density g: a Gaussian *shrunk* towards the
    # posterior mean, standing in for a normalizing flow trained so that its
    # probability mass is safely contained within the posterior support.
    shrink = 0.6
    Sigma_g = shrink ** 2 * Sigma_post
    Sigma_g_inv = np.linalg.inv(Sigma_g)
    log_det_Sigma_g = np.linalg.slogdet(Sigma_g)[1]

    def log_g(b_s, w_s):
        db_ = b_s - post_mean_b
        dw_ = w_s - post_mean_w
        quad = (Sigma_g_inv[0, 0] * db_ ** 2 + 2 * Sigma_g_inv[0, 1] * db_ * dw_
                + Sigma_g_inv[1, 1] * dw_ ** 2)
        return -0.5 * quad - 0.5 * (log_det_Sigma_g + 2 * np.log(2 * np.pi))

    n_replicates = 300
    n_samples = 60
    log_Z_hm = np.empty(n_replicates)
    log_Z_flow = np.empty(n_replicates)

    rng2 = np.random.default_rng(123)
    for r in range(n_replicates):
        choice = rng2.choice(flat_w.size, size=n_samples, p=flat_w)
        bs = BB.ravel()[choice]
        ws = WW.ravel()[choice]
        ll = loglik = np.array([
            np.sum(y * (b_ + w_ * x) - np.logaddexp(0.0, b_ + w_ * x))
            for b_, w_ in zip(bs, ws)
        ])
        lp = -0.5 * (bs ** 2 + ws ** 2) / alpha ** 2 - np.log(2 * np.pi * alpha ** 2)

        # naive harmonic mean: rho_hat = mean(1/L) -> log(Z) = -log(mean(exp(-ll)))
        log_Z_hm[r] = -(logsumexp(-ll) - np.log(n_samples))

        # flow-stabilized generalized harmonic mean:
        # 1/Z = E_post[ g(theta) / (L(theta) pi(theta)) ]
        lg = log_g(bs, ws)
        log_ratio = lg - ll - lp
        log_Z_flow[r] = -(logsumexp(log_ratio) - np.log(n_samples))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    ax = axes[0]
    data_box = [log_Z_hm, log_Z_flow]
    bp = ax.boxplot(data_box, tick_labels=['Naive\nharmonic mean', 'Flow-stabilized\n(learnt HM)'],
                     patch_artist=True, widths=0.5, showfliers=True,
                     flierprops=dict(marker='.', markersize=3, alpha=0.4))
    for patch, c in zip(bp['boxes'], [ORANGE, GREEN]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    ax.axhline(log_Z_true, color=GRAY, ls='--', lw=1.4, label=f'true $\\log Z$ = {log_Z_true:.2f}\n(grid quadrature)')
    ax.set_ylabel(r'$\log \hat{Z}$ across 300 replicate estimates', fontsize=8.7)
    ax.set_title("Naive HM is wildly unstable", fontsize=9.5)
    ax.legend(fontsize=7.3, loc='lower left', frameon=False)

    ax2 = axes[1]
    ax2.hist(log_Z_hm, bins=40, color=ORANGE, alpha=0.55, density=True, label='Naive HM')
    ax2.hist(log_Z_flow, bins=40, color=GREEN, alpha=0.55, density=True, label='Flow-stabilized')
    ax2.axvline(log_Z_true, color=GRAY, ls='--', lw=1.4)
    ax2.set_xlabel(r'$\log \hat{Z}$', fontsize=9)
    ax2.set_ylabel('density (across replicates)', fontsize=8.7)
    ax2.set_title("Distribution of the evidence estimate", fontsize=9.5)
    ax2.legend(fontsize=7.5, frameon=False)

    fig.suptitle("Toy simulation: harmonic mean instability vs. flow-stabilized correction",
                  fontsize=10.3, y=1.03)
    fig.tight_layout()
    savefig("fig3_harmonic_mean_variance.pdf")

    print(f"  toy 2-D logistic regression: log Z_true = {log_Z_true:.3f}")
    print(f"  naive HM: mean={np.mean(log_Z_hm):.3f}, std={np.std(log_Z_hm):.3f}, "
          f"min={np.min(log_Z_hm):.3f}, max={np.max(log_Z_hm):.3f}")
    print(f"  flow HM : mean={np.mean(log_Z_flow):.3f}, std={np.std(log_Z_flow):.3f}, "
          f"min={np.min(log_Z_flow):.3f}, max={np.max(log_Z_flow):.3f}")


# ============================================================================
# Figure 4 -- Normalizing flow schematic: base distribution -> flow -> a
#   distribution whose probability mass is concentrated safely within the
#   posterior (needed for a low-variance harmonic-mean-type estimator).
# ============================================================================
def fig_normalizing_flow_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

    # panel 1: base distribution q(z), simple Gaussian
    ax = axes[0]
    xx = np.linspace(-4, 4, 300)
    yy = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(xx, yy)
    Z_base = np.exp(-0.5 * (X ** 2 + Y ** 2))
    ax.contourf(X, Y, Z_base, levels=12, cmap='Blues')
    ax.set_title(r"Base $q(z)$" + "\n(simple, e.g. standard Gaussian)", fontsize=8.7)
    ax.set_xticks([]); ax.set_yticks([])

    # arrow annotation between panel 1 and 2 (drawn via fig.text/annotate on axes)
    ax.annotate('', xy=(1.35, 0.5), xycoords='axes fraction',
                xytext=(1.02, 0.5), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6))
    ax.text(1.18, 0.58, r'$T$', transform=ax.transAxes, fontsize=11, ha='center')

    # panel 2: posterior (target), elongated/skewed toy shape
    ax = axes[1]
    mean = np.array([0.3, -0.2])
    cov = np.array([[1.4, 0.9], [0.9, 0.9]])
    inv = np.linalg.inv(cov)
    d = np.stack([X - mean[0], Y - mean[1]], axis=-1)
    quad = np.einsum('...i,ij,...j->...', d, inv, d)
    Z_post = np.exp(-0.5 * quad)
    ax.contourf(X, Y, Z_post, levels=12, cmap='Reds')
    ax.set_title(r"Target posterior $P(\theta\,|\,D)$" + "\n(unknown normalizing constant)", fontsize=8.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate('', xy=(1.35, 0.5), xycoords='axes fraction',
                xytext=(1.02, 0.5), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6))
    ax.text(1.18, 0.58, 'trained\nflow', transform=ax.transAxes, fontsize=8, ha='center')

    # panel 3: learned flow density (concentrated flow), overlaid with posterior contour
    ax = axes[2]
    cov_flow = 0.55 ** 2 * cov
    inv_flow = np.linalg.inv(cov_flow)
    quad_flow = np.einsum('...i,ij,...j->...', d, inv_flow, d)
    Z_flow = np.exp(-0.5 * quad_flow)
    ax.contour(X, Y, Z_post, levels=6, colors=ORANGE, linewidths=1.3, linestyles='-')
    ax.contourf(X, Y, Z_flow, levels=12, cmap='Blues', alpha=0.75)
    ax.contour(X, Y, Z_flow, levels=6, colors=BLUE, linewidths=1.0)
    ax.set_title("Learned flow density\n(mass kept inside the posterior)", fontsize=8.7)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], color=ORANGE, lw=1.5, label='Posterior (true)'),
               plt.Line2D([0], [0], color=BLUE, lw=1.5, label='Learned flow')]
    ax.legend(handles=handles, fontsize=7, loc='lower left', frameon=False)

    fig.suptitle(r"Normalizing flow $\theta = T(z)$: learning a target density concentrated in the posterior",
                 fontsize=10.2, y=1.06)
    fig.subplots_adjust(wspace=0.55)
    savefig("fig4_normalizing_flow_schematic.pdf")


# ============================================================================
# Figure 5 -- Toy running example: two competing Bayesian logistic
#   regression models on a synthetic audit-outcome-like dataset. Evidence is
#   approximated via a Laplace (Gaussian) approximation at the MAP estimate
#   -- a standard, simple alternative way to approximate the evidence,
#   useful here purely to get concrete illustrative numbers.
# ============================================================================
def fig_toy_running_example():
    x1, x2, x3, x4, y = make_toy_audit_data(n=220, seed=1)
    n = len(y)
    alpha = 10.0

    # Model A: 2 informative covariates (+ intercept) => d = 3
    X_A = np.column_stack([np.ones(n), x1, x2])
    # Model B: adds two noise-like covariates => d = 5
    X_B = np.column_stack([np.ones(n), x1, x2, x3, x4])

    def laplace_log_evidence(X):
        d = X.shape[1]
        w0 = np.zeros(d)
        res = minimize(neg_log_posterior, w0, args=(X, y, alpha), method='BFGS')
        w_map = res.x
        # numerical Hessian of the negative log posterior at the MAP
        eps = 1e-4
        H = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                e_i = np.zeros(d); e_i[i] = eps
                e_j = np.zeros(d); e_j[j] = eps
                f_pp = neg_log_posterior(w_map + e_i + e_j, X, y, alpha)
                f_pm = neg_log_posterior(w_map + e_i - e_j, X, y, alpha)
                f_mp = neg_log_posterior(w_map - e_i + e_j, X, y, alpha)
                f_mm = neg_log_posterior(w_map - e_i - e_j, X, y, alpha)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
        sign, logdetH = np.linalg.slogdet(H)
        neg_log_post_at_map = neg_log_posterior(w_map, X, y, alpha)
        log_Z = -neg_log_post_at_map + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdetH
        return log_Z, w_map, d

    logZ_A, w_A, d_A = laplace_log_evidence(X_A)
    logZ_B, w_B, d_B = laplace_log_evidence(X_B)

    # simple train/"test" AUC via a fresh synthetic holdout sample, same generative process
    x1t, x2t, x3t, x4t, yt = make_toy_audit_data(n=150, seed=99)
    def auc_score(w, Xt):
        scores = Xt @ w
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = yt.sum(); n_neg = len(yt) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        return (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Xt_A = np.column_stack([np.ones(len(yt)), x1t, x2t])
    Xt_B = np.column_stack([np.ones(len(yt)), x1t, x2t, x3t, x4t])
    auc_A = auc_score(w_A, Xt_A)
    auc_B = auc_score(w_B, Xt_B)

    log_bayes_factor_AB = logZ_A - logZ_B

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    names = ['Model A\n(2 covariates,\n$d=3$)', 'Model B\n(4 covariates,\n$d=5$)']
    vals = [logZ_A, logZ_B]
    colors = [BLUE, ORANGE]
    bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.0 if v > 0 else -3.0),
                 f"{v:.2f}", ha='center', fontsize=9)
    ax.set_ylabel(r'illustrative $\log \hat Z_M$ (Laplace approx.)', fontsize=8.7)
    ax.set_title("Toy running example: log-evidence", fontsize=9.5)

    ax2 = axes[1]
    bars2 = ax2.bar(names, [auc_A, auc_B], color=colors, width=0.5, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars2, [auc_A, auc_B]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
    ax2.set_ylim(0.5, 1.0)
    ax2.axhline(0.5, color=GRAY, ls='--', lw=1.0)
    ax2.set_ylabel('AUC on synthetic held-out data', fontsize=8.7)
    ax2.set_title("Toy running example: predictive AUC", fontsize=9.5)

    fig.suptitle(r"Toy running example -- our own synthetic data, NOT the book's dataset/results",
                 fontsize=9.8, y=1.04)
    fig.tight_layout()
    savefig("fig5_toy_running_example.pdf")

    print(f"  toy running example: logZ_A={logZ_A:.3f} (d={d_A}), logZ_B={logZ_B:.3f} (d={d_B}), "
          f"log BF(A vs B)={log_bayes_factor_AB:.3f}")
    print(f"  toy running example: AUC_A={auc_A:.4f}, AUC_B={auc_B:.4f}")


# ============================================================================
if __name__ == "__main__":
    print("Generating Chapter 8 figures...")
    fig_occams_razor()
    fig_toy4_harmonic_mean()
    fig_harmonic_mean_variance()
    fig_normalizing_flow_schematic()
    fig_toy_running_example()
    print("Done.")
