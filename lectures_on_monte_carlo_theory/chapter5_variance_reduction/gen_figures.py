"""
gen_figures.py  –  Generate all figures for Chapter 5 slides.
Run with:  conda run -n py313 python3 gen_figures.py
All PDFs are saved to ./figures/
"""
import os, math, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, name + '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  saved {name}.pdf')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 1  Stratified vs CMC – variance illustration for estimating pi
# ──────────────────────────────────────────────────────────────────────────────
def fig_stratified_sampling():
    """Compare Stratified and CMC estimators for pi."""
    R_values = [10, 20, 50, 100, 200, 500, 1000]
    n_trials  = 2000
    true_pi   = math.pi

    cmc_var, str_var = [], []
    for R in R_values:
        cmc_ests = []
        str_ests = []
        for _ in range(n_trials):
            # CMC
            U1 = np.random.uniform(0,1,R); U2 = np.random.uniform(0,1,R)
            cmc_ests.append(4*np.mean(U1**2+U2**2 <= 1))
            # Stratified (m=R strata on U1)
            j   = np.arange(R)
            u1s = (j + np.random.uniform(0,1,R)) / R
            u2s = np.random.uniform(0,1,R)
            str_ests.append(4*np.mean(u1s**2+u2s**2 <= 1))
        cmc_var.append(np.var(cmc_ests))
        str_var.append(np.var(str_ests))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.loglog(R_values, cmc_var, 'o-', label='CMC', color='steelblue')
    ax.loglog(R_values, str_var, 's--', label='Stratified (m=R)', color='tomato')
    ax.set_xlabel('Number of replications $R$')
    ax.set_ylabel('Empirical variance of $\\hat{Y}_R$')
    ax.set_title('Variance Comparison: CMC vs Stratified Sampling ($\\pi$ estimation)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    save('fig_stratified_vs_cmc')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 2  Stratified sampling – optimal allocation illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_stratified_allocation():
    """Show optimal vs proportional allocation."""
    m = 5
    strata = np.arange(1, m+1)
    sigma  = np.array([1.0, 0.8, 1.5, 0.5, 1.2])   # within-stratum std
    p      = np.ones(m) / m                          # equal stratum sizes

    opt_alloc  = p * sigma / np.sum(p * sigma)
    prop_alloc = p

    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(m)
    w = 0.35
    ax.bar(x - w/2, prop_alloc, w, label='Proportional', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, opt_alloc,  w, label='Optimal',      color='tomato',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Stratum {i}' for i in strata])
    ax.set_ylabel('Fraction of replications $R_j / R$')
    ax.set_title('Proportional vs Optimal Allocation')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save('fig_stratified_allocation')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 3  Antithetic variates – negative correlation demonstration
# ──────────────────────────────────────────────────────────────────────────────
def fig_antithetic():
    """Scatter of U vs 1-U showing negative correlation."""
    n = 500
    U  = np.random.uniform(0, 1, n)
    Up = 1 - U
    g  = np.exp(U);  gp = np.exp(Up)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(U, Up, s=6, alpha=0.5, color='steelblue')
    axes[0].set_xlabel('$U$'); axes[0].set_ylabel("$U' = 1 - U$")
    axes[0].set_title('Antithetic pair $(U, 1-U)$')

    axes[1].scatter(g, gp, s=6, alpha=0.5, color='tomato')
    axes[1].set_xlabel('$g(U) = e^U$'); axes[1].set_ylabel("$g(1-U) = e^{1-U}$")
    axes[1].set_title('Payoff pairs (negative correlation)')

    rho = np.corrcoef(g, gp)[0,1]
    axes[1].set_title(f'Payoff pairs ($\\rho = {rho:.3f}$)')
    save('fig_antithetic')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 4  Control variates – regression-based coefficient
# ──────────────────────────────────────────────────────────────────────────────
def fig_control_variates():
    """Show Y vs X scatter and regression line (optimal c)."""
    n  = 300
    U1 = np.random.uniform(0,1,n); U2 = np.random.uniform(0,1,n)
    Y  = 4*(U1**2 + U2**2 <= 1).astype(float)   # estimating pi
    X  = (U1 + U2 <= 1).astype(float)            # control variate, EX=0.5

    cov_xy = np.cov(X, Y, ddof=1)
    c_opt  = -cov_xy[0,1] / cov_xy[0,0]
    x_line = np.linspace(X.min(), X.max(), 50)
    y_line = np.mean(Y) + c_opt * (x_line - 0.5)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(X, Y, s=8, alpha=0.3, color='steelblue', label='Samples')
    ax.plot(x_line, y_line, 'r-', lw=2, label=f'Regression ($c^*={c_opt:.2f}$)')
    ax.axvline(0.5, ls='--', color='gray', lw=1, label='$\\mathbb{E}X = 0.5$')
    ax.set_xlabel('Control variate $X$')
    ax.set_ylabel('Response $Y$')
    ax.set_title('Control Variate Regression ($\\pi$ estimation)')
    ax.legend()
    save('fig_control_variates')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 5  Importance sampling – proposal distributions for quarter-circle
# ──────────────────────────────────────────────────────────────────────────────
def fig_IS_proposals():
    """Plot k(x)f(x) = 4*sqrt(1-x^2) and candidate proposal densities."""
    x = np.linspace(0, 1, 300)
    k = 4 * np.sqrt(1 - x**2)
    f1 = 2 * x                    # proposal f_1(x)=2x
    f2 = (4 - 2*x) / 3            # proposal f_2(x)=(4-2x)/3

    # normalise so they're true densities (already are)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, k,  'k-',  lw=2,  label='$k(x)=4\\sqrt{1-x^2}$')
    ax.plot(x, f1, 'b-',  lw=1.5,label='$\\tilde{f}_1(x)=2x$')
    ax.plot(x, f2, 'r-',  lw=1.5,label='$\\tilde{f}_2(x)=(4-2x)/3$')
    ax.set_xlabel('$x$'); ax.set_ylabel('Density / integrand')
    ax.set_title('Importance Sampling Proposal Densities')
    ax.legend(); ax.grid(alpha=0.3)
    save('fig_IS_proposals')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 6  IS variance reduction for P(X>4), X~N(0,1)
# ──────────────────────────────────────────────────────────────────────────────
def fig_IS_rare_event():
    """Show CMC vs IS estimators for P(Z>4)."""
    R = 2000
    n_rep = 500
    true_I = 1 - stats.norm.cdf(4)

    cmc_ests = [4 * np.mean(np.random.standard_normal(R) > 4) for _ in range(n_rep)]
    # IS with proposal N(4,1)
    theta = 4.0
    Z_tilde = np.random.normal(theta, 1, (n_rep, R))
    Y_IS = (Z_tilde > 4) * np.exp(-theta * Z_tilde + theta**2/2)
    is_ests = Y_IS.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cmc_ests, bins=30, color='steelblue', alpha=0.7, label='CMC')
    axes[0].axvline(true_I, color='red', lw=2, label='True $I$')
    axes[0].set_title('CMC Estimates of $P(Z>4)$')
    axes[0].legend(); axes[0].set_xlabel('Estimate')

    axes[1].hist(is_ests, bins=30, color='tomato', alpha=0.7, label='IS ($\\theta=4$)')
    axes[1].axvline(true_I, color='red', lw=2, label='True $I$')
    axes[1].set_title('IS Estimates of $P(Z>4)$')
    axes[1].legend(); axes[1].set_xlabel('Estimate')
    save('fig_IS_rare_event')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 7  Brownian motion sample paths
# ──────────────────────────────────────────────────────────────────────────────
def fig_brownian_motion():
    """Simulate 5 Brownian motion paths."""
    n = 1000
    t = np.linspace(0, 1, n+1)
    dt = 1.0 / n

    fig, ax = plt.subplots(figsize=(8,4))
    for i in range(5):
        Z = np.random.standard_normal(n)
        B = np.concatenate([[0], np.cumsum(np.sqrt(dt) * Z)])
        ax.plot(t, B, lw=0.8, alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$t$'); ax.set_ylabel('$B(t)$')
    ax.set_title('Sample Paths of Standard Brownian Motion')
    ax.grid(alpha=0.2)
    save('fig_brownian_motion')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 8  Geometric Brownian motion (GBM) – stock price paths
# ──────────────────────────────────────────────────────────────────────────────
def fig_geometric_brownian():
    """Simulate 5 GBM paths (risk-neutral)."""
    S0, r, sigma, T = 100, 0.05, 0.25, 1.0
    mu_star = r - sigma**2 / 2
    n = 252
    dt = T / n
    t  = np.linspace(0, T, n+1)

    fig, ax = plt.subplots(figsize=(8,4))
    for _ in range(8):
        Z = np.random.standard_normal(n)
        log_ret = (mu_star * dt + sigma * np.sqrt(dt) * Z)
        S = S0 * np.exp(np.concatenate([[0], np.cumsum(log_ret)]))
        ax.plot(t, S, lw=0.8, alpha=0.7)
    ax.axhline(100, color='k', lw=1, ls='--', label='$S(0)=100$')
    ax.set_xlabel('$t$'); ax.set_ylabel('$S(t)$')
    ax.set_title(f'Geometric Brownian Motion Paths ($r={r}$, $\\sigma={sigma}$)')
    ax.legend(); ax.grid(alpha=0.2)
    save('fig_geometric_brownian')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 9  European call option: CMC vs IS variance comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig_european_option_IS():
    """Scatter of CMC and IS estimates for European call as K varies."""
    S0, r, sigma = 100, 0.05, 0.25
    mu_star = r - sigma**2 / 2
    R = 10000
    K_vals = [80, 90, 100, 110, 120, 130, 140, 150]

    cmc_errs, is_errs = [], []
    for K in K_vals:
        # CMC
        Z = np.random.standard_normal(R)
        S1 = S0 * np.exp(mu_star + sigma * Z)
        cmc_errs.append(np.std(np.exp(-r) * np.maximum(S1 - K, 0)) / np.sqrt(R))
        # IS: shift proposal to N(x*, 1) where x* ≈ (log(K/S0)-mu*)/sigma
        x_star = (np.log(K/S0) - mu_star) / sigma + 0.5
        Zt = np.random.normal(x_star, 1, R)
        S1t = S0 * np.exp(mu_star + sigma * Zt)
        lr  = np.exp(-x_star * Zt + x_star**2 / 2)
        Y_IS = np.exp(-r) * np.maximum(S1t - K, 0) * lr
        is_errs.append(np.std(Y_IS) / np.sqrt(R))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.semilogy(K_vals, cmc_errs, 'o-', label='CMC Std Error', color='steelblue')
    ax.semilogy(K_vals, is_errs,  's--',label='IS Std Error',  color='tomato')
    ax.set_xlabel('Strike price $K$')
    ax.set_ylabel('Standard error (log scale)')
    ax.set_title('European Call Option: CMC vs IS Standard Error')
    ax.legend(); ax.grid(alpha=0.3)
    save('fig_european_option_IS')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 10  Cross-entropy method – convergence of theta_t
# ──────────────────────────────────────────────────────────────────────────────
def fig_CE_convergence():
    """Show CE method convergence: gamma_t and theta_t vs iteration."""
    # Estimate P(X>4) for X~N(0,1) using CE with N(theta,1) proposals
    np.random.seed(0)
    M = 10000
    rho = 0.05
    alpha_smooth = 0.4
    theta = 0.0
    gamma_target = 4.0

    gammas  = []
    thetas  = []
    for t in range(50):
        X = np.random.normal(theta, 1, M)
        h = X  # performance function h(X)=X
        w = np.exp(-theta*X + theta**2/2)  # likelihood ratio to original
        # update gamma: (1-rho) quantile of h
        gamma_t = np.quantile(h, 1 - rho)
        gammas.append(gamma_t)
        # update theta: weighted mean of X where h>=gamma_t
        mask = h >= gamma_t
        if mask.sum() > 0:
            new_theta = np.sum(w[mask] * X[mask]) / np.sum(w[mask])
        else:
            new_theta = theta
        theta = alpha_smooth * new_theta + (1 - alpha_smooth) * theta
        thetas.append(theta)
        if gamma_t >= gamma_target:
            break

    iters = np.arange(1, len(gammas)+1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iters, gammas, 'o-', color='steelblue')
    axes[0].axhline(gamma_target, color='red', ls='--', label=f'Target $\\gamma={gamma_target}$')
    axes[0].set_xlabel('Iteration $t$'); axes[0].set_ylabel('$\\gamma_t$')
    axes[0].set_title('CE Method: Level sequence $\\gamma_t$')
    axes[0].legend()

    axes[1].plot(iters, thetas, 'o-', color='tomato')
    axes[1].axhline(4.2256, color='red', ls='--', label='Theoretical $\\theta^*=4.2256$')
    axes[1].set_xlabel('Iteration $t$'); axes[1].set_ylabel("$\\theta'_t$")
    axes[1].set_title("CE Method: Parameter sequence $\\theta'_t$")
    axes[1].legend()
    save('fig_CE_convergence')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 11  Variance comparison summary for estimating pi (Table 5.11)
# ──────────────────────────────────────────────────────────────────────────────
def fig_variance_summary():
    """Bar chart comparing variances of different methods for estimating pi."""
    methods = ['CMC\n(4.1.6)', 'CV\n(5.1.24)', 'CMC\n(4.1.7)', 'CV\n(5.1.24b)',
               'CRN\n(5.1.20)', 'IS\n(5.1.16)', 'Anti\n(5.1.16)', 'CV\n(5.1.24c)',
               'Str m=10\n(5.1.8)', 'Str m=20\n(5.1.8)']
    # Var * 10^4 for R=10^4 samples (from Table 5.11)
    variances = [2.6968, 1.959, 0.797, 0.6527, 0.2885, 0.224, 0.219, 0.1119, 0.0112, 0.00296]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(methods)))
    bars = ax.bar(methods, variances, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('$\\mathrm{Var}(\\hat{Y}_{10^4}) \\times 10^4$')
    ax.set_title('Variance Comparison for Estimating $\\pi$ (various methods)')
    ax.set_yscale('log')
    for bar, v in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.grid(axis='y', alpha=0.3)
    save('fig_variance_summary')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 12  Conditional Monte Carlo illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_conditional_mc():
    """Compare CMC vs conditional estimator for pi."""
    R_vals = np.logspace(1, 4, 20, dtype=int)
    n_trials = 1000

    cmc_var, cond_var = [], []
    for R in R_vals:
        cmc_ests, cond_ests = [], []
        for _ in range(n_trials):
            U1 = np.random.uniform(0,1,R)
            U2 = np.random.uniform(0,1,R)
            cmc_ests.append(4*np.mean(U1**2+U2**2 <= 1))
            # Conditional: c(x) = 4*sqrt(1-x^2), X=U1
            cond_ests.append(4*np.mean(np.sqrt(1 - U1**2)))
        cmc_var.append(np.var(cmc_ests))
        cond_var.append(np.var(cond_ests))

    fig, ax = plt.subplots(figsize=(7,4))
    ax.loglog(R_vals, cmc_var,  'o-', label='CMC', color='steelblue')
    ax.loglog(R_vals, cond_var, 's--',label='Conditional MC', color='tomato')
    ax.set_xlabel('$R$'); ax.set_ylabel('Empirical variance')
    ax.set_title('Conditional MC vs CMC for $\\pi$ estimation')
    ax.legend(); ax.grid(alpha=0.3)
    save('fig_conditional_mc')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 13  Black-Scholes formula illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_black_scholes():
    """European call option price vs strike (Black-Scholes vs MC)."""
    from scipy.stats import norm
    S0, r, sigma, T = 100, 0.05, 0.25, 1.0
    K_vals = np.linspace(70, 150, 50)
    mu_star = r - sigma**2/2

    # Black-Scholes
    d1 = (np.log(S0/K_vals) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K_vals*np.exp(-r*T)*norm.cdf(d2)

    # Monte Carlo
    R = 50000
    Z  = np.random.standard_normal(R)
    S1 = S0 * np.exp(mu_star + sigma * Z)
    mc_price = np.array([np.exp(-r)*np.mean(np.maximum(S1-K, 0)) for K in K_vals])

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(K_vals, bs_price, 'k-',  lw=2, label='Black-Scholes')
    ax.plot(K_vals, mc_price, 'r--', lw=1.5, label='MC ($R=50000$)', alpha=0.8)
    ax.set_xlabel('Strike $K$'); ax.set_ylabel('Option price $I$')
    ax.set_title('European Call Option Price: Black-Scholes vs Monte Carlo')
    ax.legend(); ax.grid(alpha=0.3)
    save('fig_black_scholes')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 14  Antithetic variates: Asian option paths
# ──────────────────────────────────────────────────────────────────────────────
def fig_asian_option_antithetic():
    """Show one original path and its antithetic for Asian call."""
    S0, r, sigma = 100, 0.05, 0.25
    mu_star = r - sigma**2/2
    n = 10
    t_vals = np.arange(1, n+1) / n

    np.random.seed(7)
    Z   = np.random.standard_normal(n)
    Zp  = -Z
    B   = np.cumsum(Z)  / np.sqrt(n)
    Bp  = np.cumsum(Zp) / np.sqrt(n)
    S   = S0 * np.exp(mu_star * t_vals + sigma * B)
    Sp  = S0 * np.exp(mu_star * t_vals + sigma * Bp)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(np.concatenate([[0], t_vals]), np.concatenate([[S0], S]),
            'b-o', ms=5, label='Original path $S(t)$')
    ax.plot(np.concatenate([[0], t_vals]), np.concatenate([[S0], Sp]),
            'r--s', ms=5, label="Antithetic path $S'(t)$")
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xlabel('$t$'); ax.set_ylabel('$S(t)$')
    ax.set_title('Asian Option: Original vs Antithetic Paths ($n=10$)')
    ax.legend(); ax.grid(alpha=0.3)
    save('fig_asian_option_antithetic')

# ──────────────────────────────────────────────────────────────────────────────
# Fig 15  Self-normalised IS weights illustration
# ──────────────────────────────────────────────────────────────────────────────
def fig_snIS_weights():
    """Histogram of IS normalised weights for N(0,1) vs N(4,1)."""
    n = 5000
    # Sample from N(4,1), compute weights w = f_0/f_tilde
    theta = 4.0
    X = np.random.normal(theta, 1, n)
    w = np.exp(-theta * X + theta**2 / 2)    # f_0(x)/f_tilde(x)
    w_norm = w / w.sum()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(X, bins=40, density=True, color='steelblue', alpha=0.7,
                 label='Samples from $N(4,1)$')
    x_plot = np.linspace(-3, 9, 200)
    axes[0].plot(x_plot, stats.norm.pdf(x_plot, 0, 1), 'k-', label='$N(0,1)$ target')
    axes[0].plot(x_plot, stats.norm.pdf(x_plot, 4, 1), 'r--', label='$N(4,1)$ proposal')
    axes[0].legend(fontsize=8); axes[0].set_title('IS Proposal vs Target')

    axes[1].hist(w_norm, bins=50, color='tomato', alpha=0.7)
    axes[1].set_xlabel('Normalised weight $w\'(X_i)$')
    axes[1].set_title('Distribution of IS Normalised Weights')
    axes[1].set_ylabel('Frequency')
    save('fig_snIS_weights')

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures for Chapter 5...')
    fig_stratified_sampling()
    fig_stratified_allocation()
    fig_antithetic()
    fig_control_variates()
    fig_IS_proposals()
    fig_IS_rare_event()
    fig_brownian_motion()
    fig_geometric_brownian()
    fig_european_option_IS()
    fig_CE_convergence()
    fig_variance_summary()
    fig_conditional_mc()
    fig_black_scholes()
    fig_asian_option_antithetic()
    fig_snIS_weights()
    print('Done. All figures saved to', OUTDIR)
