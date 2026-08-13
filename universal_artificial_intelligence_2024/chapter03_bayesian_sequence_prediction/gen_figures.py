"""
Figure generator for Chapter 3 -- Bayesian Sequence Prediction
(An Introduction to Universal Artificial Intelligence, Hutter/Quarel/Catt 2024)

Run with:
    conda run -n py313 python3 gen_figures.py

All figures are saved as PDF into ./figures/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10.5,
    'figure.dpi': 150,
})

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Figure 1: Bayes-Laplace weather / jacket example (Example 3.5.3, 3.5.7)
# ----------------------------------------------------------------------
def fig_weather_convergence():
    theta_true = 0.9
    n = 14
    # First three days forced 'warm' to match the book's worked example exactly,
    # remaining days sampled i.i.d. Bernoulli(theta_true).
    forced = [1, 1, 1]
    rest = RNG.binomial(1, theta_true, size=n - len(forced))
    x = np.array(forced + list(rest))  # 1 = warm, 0 = cold

    t_axis = np.arange(1, n + 1)
    n_warm_before = np.concatenate([[0], np.cumsum(x)[:-1]])  # #warm among x_{<t}
    theta_hat = (n_warm_before + 1) / (t_axis + 1)  # xi(x_t = warm | x_<t)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = ['#2f6f4f' if v >= 0.75 else '#b5442d' for v in theta_hat]
    ax.bar(t_axis, theta_hat, color=colors, width=0.6, zorder=3,
           label=r'$\hat\theta_t=\xi(x_t{=}\mathrm{warm}\,|\,x_{<t})$')
    ax.axhline(0.75, color='k', linestyle='--', linewidth=1.4, zorder=2,
               label=r'critical threshold $\theta_{\rm crit}=3/4$')
    for i, v in enumerate(x):
        marker = 'warm' if v == 1 else 'cold'
        mcolor = '#d99a00' if v == 1 else '#3a6ea5'
        ax.text(t_axis[i], 1.03, marker, ha='center', va='bottom', fontsize=8,
                 color=mcolor, rotation=60)
    ax.set_xlabel('day $t$')
    ax.set_ylabel(r'predicted probability of warm weather')
    ax.set_ylim(0, 1.3)
    ax.set_xticks(t_axis)
    ax.set_title('Bayes-Laplace predictor: jacket decision over time')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(0.5, -0.34,
            'green bars = "no jacket" decision, red bars = "take jacket" decision',
            transform=ax.transAxes, ha='center', fontsize=9.5, color='#444')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'weather_convergence.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: KL-divergence / distance convergence (Sec 3.2-3.3)
# ----------------------------------------------------------------------
def make_model_class(n_models=99):
    """Countable class M of Bernoulli(theta) models on a fine grid, uniform prior."""
    thetas = np.arange(1, n_models + 1) / (n_models + 1)  # in (0,1)
    log_w = np.full(n_models, -np.log(n_models))
    return thetas, log_w


def log_bernoulli(theta, x):
    """log P_theta(x) for x in {0,1} (1=warm/heads)."""
    return np.where(x == 1, np.log(theta), np.log1p(-theta))


def simulate_distances(theta_true=0.7, n=200, n_models=99, seed=0):
    rng = np.random.default_rng(seed)
    thetas, log_w0 = make_model_class(n_models)
    x = rng.binomial(1, theta_true, size=n)

    log_w = log_w0.copy()
    d_t = np.zeros(n)      # instantaneous KL(mu(.|x<t) || xi(.|x<t))
    xi_pred = np.zeros(n)  # xi(x_t=1|x_<t)

    for t in range(n):
        # normalize posterior weights (log-space)
        m = log_w.max()
        w = np.exp(log_w - m)
        w /= w.sum()
        xi1 = np.sum(w * thetas)          # xi(warm | x_<t)
        xi1 = np.clip(xi1, 1e-12, 1 - 1e-12)
        xi_pred[t] = xi1

        mu1 = theta_true
        d_t[t] = mu1 * np.log(mu1 / xi1) + (1 - mu1) * np.log((1 - mu1) / (1 - xi1))

        # update posterior with observed symbol x[t]
        log_w = log_w + log_bernoulli(thetas, x[t])

    return d_t, xi_pred, thetas, log_w0


def fig_distance_convergence():
    n = 200
    d_t, xi_pred, thetas, log_w0 = simulate_distances(theta_true=0.7, n=n, seed=1)
    D_n = np.cumsum(d_t)

    # w_mu: prior weight on the model closest to true theta=0.7
    n_models = len(thetas)
    idx = np.argmin(np.abs(thetas - 0.7))
    w_mu = np.exp(log_w0[idx])
    bound = np.log(1.0 / w_mu)

    t_axis = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    ax.plot(t_axis, d_t, color='#3a6ea5', linewidth=1.3)
    ax.set_xlabel('time step $t$')
    ax.set_ylabel(r'instantaneous $d_t(x_{<t})$')
    ax.set_title(r'KL distance $d_t \to 0$')
    ax.set_yscale('log')

    ax = axes[1]
    ax.plot(t_axis, D_n, color='#b5442d', linewidth=1.6, label=r'$D_n=\sum_{t\leq n} d_t$')
    ax.axhline(bound, color='k', linestyle='--', linewidth=1.3,
               label=r'bound $\ln w_\mu^{-1}=%.2f$' % bound)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'cumulative KL $D_n$')
    ax.set_title('Generalized Solomonoff bound (Thm 3.2.5)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, bound * 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'distance_convergence.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Solomonoff prior decay w_nu = 2^{-K(nu)}  (Sec 3.7.2, Def 3.7.2)
# ----------------------------------------------------------------------
def fig_prior_decay():
    K = np.arange(0, 17)
    w = 2.0 ** (-K.astype(float))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(K, w, color='#4a7c59', width=0.6, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel(r'Kolmogorov complexity $K(\nu)$ [bits]')
    ax.set_ylabel(r'Solomonoff weight $w_\nu^U = 2^{-K(\nu)}$')
    ax.set_title('Simplicity bias of the Solomonoff prior')
    ax.set_xticks(K[::2])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'prior_decay.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: Martingale Z_t = xi(x_1:t)/mu(x_1:t)  (Sec 3.9, Example 3.9.4)
# ----------------------------------------------------------------------
def simulate_martingale(theta_true=0.7, n=400, n_models=99, seed=0):
    rng = np.random.default_rng(seed)
    thetas, log_w0 = make_model_class(n_models)
    x = rng.binomial(1, theta_true, size=n)

    log_w = log_w0.copy()
    log_Z = np.zeros(n)  # log[ xi(x_1:t) / mu(x_1:t) ]
    running = 0.0
    for t in range(n):
        m = log_w.max()
        w = np.exp(log_w - m)
        w /= w.sum()
        xi1 = np.clip(np.sum(w * thetas), 1e-12, 1 - 1e-12)
        mu1 = theta_true
        p_xi = xi1 if x[t] == 1 else (1 - xi1)
        p_mu = mu1 if x[t] == 1 else (1 - mu1)
        running += np.log(p_xi) - np.log(p_mu)
        log_Z[t] = running
        log_w = log_w + log_bernoulli(thetas, x[t])
    return log_Z


def fig_martingale():
    n = 400
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    n_paths = 6
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, n_paths))
    for i in range(n_paths):
        log_Z = simulate_martingale(theta_true=0.7, n=n, seed=100 + i)
        Z = np.exp(log_Z)
        ax.plot(np.arange(1, n + 1), Z, color=colors[i], linewidth=1.2, alpha=0.9)
    ax.set_xlabel('time step $t$')
    ax.set_ylabel(r'$Z_t=\xi(x_{1:t})/\mu(x_{1:t})$')
    ax.set_title(r'Supermartingale $Z_t$: convergence to finite $Z_\infty$')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'martingale_paths.pdf'), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5: Pareto optimality of xi  (Sec 3.6, Definition 3.6.1)
# ----------------------------------------------------------------------
def fig_pareto():
    rng = np.random.default_rng(7)
    n_pts = 260
    # random candidate predictors: loss under environment nu_1 vs nu_2
    L1 = rng.uniform(0.15, 1.0, n_pts)
    L2 = 0.9 / (L1 + 0.25) - 0.15 + rng.normal(0, 0.10, n_pts)
    L2 = np.clip(L2, 0.05, 1.4)

    # Pareto frontier (non-dominated points): sort by L1, sweep for running min L2
    order = np.argsort(L1)
    L1s, L2s = L1[order], L2[order]
    frontier_mask = np.zeros(n_pts, dtype=bool)
    best = np.inf
    for i in range(n_pts):
        if L2s[i] < best:
            best = L2s[i]
            frontier_mask[i] = True

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(L1[~order[frontier_mask]] if False else L1, L2, s=22, color='#a9a9a9',
               alpha=0.55, label='other predictors $\\rho$', zorder=2)
    ax.plot(L1s[frontier_mask], L2s[frontier_mask], color='#b5442d', linewidth=2.0,
            marker='o', markersize=4, zorder=3, label='Pareto frontier')

    # highlight xi as a specific frontier point
    xi_idx = np.where(frontier_mask)[0][len(np.where(frontier_mask)[0]) // 2]
    ax.scatter([L1s[xi_idx]], [L2s[xi_idx]], s=140, color='#2f6f4f', marker='*',
               zorder=4, label=r'Bayes mixture $\xi$')

    ax.set_xlabel(r'Loss$_{1:n}^{\nu_1}(\rho)$')
    ax.set_ylabel(r'Loss$_{1:n}^{\nu_2}(\rho)$')
    ax.set_title(r'Pareto optimality of $\xi$ (Def. 3.6.1)')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'pareto.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_weather_convergence()
    fig_distance_convergence()
    fig_prior_decay()
    fig_martingale()
    fig_pareto()
    print('All figures written to', FIGDIR)
