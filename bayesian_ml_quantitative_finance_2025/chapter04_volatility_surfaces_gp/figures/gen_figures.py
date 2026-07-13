#!/usr/bin/env python3
"""
gen_figures.py

Generates two ILLUSTRATIVE TOY figures for the Chapter 4 slide deck
("Learning Equity Volatility Surfaces Using Sparse Gaussian Processes").

These are NOT reproductions of any figure in the book. They are original,
self-contained toy simulations built purely to give the student visual
intuition for:

  (a) what an implied volatility "smile" looks like versus the flat
      constant-volatility assumption of Black-Scholes, and

  (b) how a sparse Gaussian process (a GP trained using a small number of
      "inducing points" rather than the full dataset) fits noisy
      smile-shaped data, producing a posterior mean curve plus an
      uncertainty band that widens away from the data / inducing points.

Both figures are saved as vector PDF files in this directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# Shared plot style
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})


# ========================================================================
# FIGURE (a): Toy implied volatility "smile"
# ========================================================================
def true_smile(K, atm=100.0):
    """
    A purely illustrative (TOY, not calibrated to any real market) implied
    volatility curve as a function of strike K, exhibiting the classic
    downward-sloping "skew" plus curvature ("smile") seen in equity index
    options: higher implied vol for low strikes (crash protection demand),
    a minimum near-the-money, and a mild upturn for high strikes.
    """
    m = (K - atm) / atm  # moneyness-like coordinate
    return 0.18 + 0.55 * m**2 - 0.28 * m


def make_smile_figure(path):
    strikes_fine = np.linspace(70, 130, 400)
    iv_fine = true_smile(strikes_fine)

    # A handful of "market-quote-like" points with small noise
    strikes_obs = np.array([75, 82, 88, 94, 100, 106, 112, 118, 125])
    iv_obs = true_smile(strikes_obs) + rng.normal(0, 0.006, size=strikes_obs.shape)

    flat_bs = np.full_like(strikes_fine, true_smile(np.array([100.0]))[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(strikes_fine, flat_bs, color='gray', linestyle='--', lw=1.8,
             label='Black--Scholes: constant volatility (flat)')
    ax.plot(strikes_fine, iv_fine, color='#1f77b4', lw=2.2,
             label='True (toy) implied volatility smile')
    ax.scatter(strikes_obs, iv_obs, color='#d62728', zorder=5, s=45,
               label='Synthetic "market" quotes (toy)')

    ax.set_xlabel('Strike $K$')
    ax.set_ylabel('Implied volatility $\\sigma_{\\mathrm{imp}}(K)$')
    ax.set_title('Toy illustration: implied volatility smile vs.\nBlack--Scholes flat volatility')
    ax.legend(loc='upper center', fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'Wrote {path}')


# ========================================================================
# FIGURE (b): Toy sparse Gaussian process fit to smile-like data
# ========================================================================
def squared_exponential_kernel(x1, x2, lengthscale, sigma_f):
    """
    k(x, x') = sigma_f^2 * exp( -0.5 * (x - x')^2 / lengthscale^2 )

    x1: shape (n,), x2: shape (m,)  ->  returns (n, m) kernel matrix
    """
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    sqdist = (x1 - x2) ** 2
    return sigma_f**2 * np.exp(-0.5 * sqdist / lengthscale**2)


def sparse_gp_dtc_predict(x_train, y_train, x_inducing, x_star,
                           lengthscale, sigma_f, sigma_noise):
    """
    Sparse Gaussian process regression using the Deterministic Training
    Conditional (DTC) approximation with a fixed, small set of inducing
    points Z = x_inducing (Quinonero-Candela & Rasmussen, 2005).

    This is the textbook mechanism behind "sparse GPs": instead of
    inverting the full n x n kernel matrix (cost O(n^3)), we summarise the
    training data through m << n inducing points and only ever invert
    m x m matrices (cost O(n m^2)).

    Returns: (mean at x_star, std-dev at x_star)
    """
    n = len(x_train)
    m = len(x_inducing)

    Kmm = squared_exponential_kernel(x_inducing, x_inducing, lengthscale, sigma_f)
    Kmm += 1e-6 * np.eye(m)  # jitter for numerical stability
    Knm = squared_exponential_kernel(x_train, x_inducing, lengthscale, sigma_f)
    Kmn = Knm.T
    Ksm = squared_exponential_kernel(x_star, x_inducing, lengthscale, sigma_f)
    Kss_diag = sigma_f**2 * np.ones(len(x_star))  # k(x*,x*) for SE kernel

    Kmm_inv = np.linalg.inv(Kmm)

    # Sigma = Kmm + (1/sigma_noise^2) * Kmn * Knm      (m x m)
    Sigma = Kmm + (Kmn @ Knm) / sigma_noise**2
    Sigma_inv = np.linalg.inv(Sigma)

    # Posterior mean:  Ksm * Sigma^{-1} * Kmn * y / sigma_noise^2
    mean = (Ksm @ Sigma_inv @ (Kmn @ y_train)) / sigma_noise**2

    # Posterior variance (DTC):
    #   k(x*,x*) - Ksm Kmm^-1 Ksm^T + Ksm Sigma^-1 Ksm^T
    var_reduction = np.einsum('ij,jk,ik->i', Ksm, Kmm_inv, Ksm)
    var_dtc_term = np.einsum('ij,jk,ik->i', Ksm, Sigma_inv, Ksm)
    var = Kss_diag - var_reduction + var_dtc_term
    var = np.clip(var, 1e-10, None)

    return mean, np.sqrt(var)


def make_sparse_gp_figure(path):
    # --- synthetic smile-shaped data (toy) ---------------------------------
    x_true = np.linspace(70, 130, 400)
    y_true = true_smile(x_true)

    n_train = 24
    x_train = np.sort(rng.uniform(72, 128, size=n_train))
    noise_sd = 0.012
    y_train = true_smile(x_train) + rng.normal(0, noise_sd, size=n_train)

    # --- a HANDFUL of inducing points (the "sparse" part of sparse GP) -----
    x_inducing = np.array([76.0, 88.0, 100.0, 112.0, 124.0])

    # --- GP hyperparameters (fixed here for the toy illustration; in
    #     practice they are found by maximizing the log-marginal likelihood,
    #     Eq. (4.7) in the book) -------------------------------------------
    lengthscale = 16.0
    sigma_f = 0.20
    sigma_noise = noise_sd

    x_star = np.linspace(65, 135, 300)
    mean_star, std_star = sparse_gp_dtc_predict(
        x_train, y_train, x_inducing, x_star, lengthscale, sigma_f, sigma_noise)

    # inducing point y-values from the posterior mean (for plotting only)
    mean_induce, _ = sparse_gp_dtc_predict(
        x_train, y_train, x_inducing, x_inducing, lengthscale, sigma_f, sigma_noise)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    ax.plot(x_true, y_true, color='black', lw=1.6, linestyle='--',
             label='True underlying smile (toy, unknown to the GP)')
    ax.scatter(x_train, y_train, color='#7f7f7f', s=22, alpha=0.75, zorder=4,
               label=f'Noisy synthetic training points ($n={n_train}$)')

    ax.plot(x_star, mean_star, color='#1f77b4', lw=2.2,
             label='Sparse GP posterior mean')
    ax.fill_between(x_star, mean_star - 2 * std_star, mean_star + 2 * std_star,
                     color='#1f77b4', alpha=0.20,
                     label='Sparse GP $\\pm 2$ std. dev. band')

    ax.scatter(x_inducing, mean_induce, color='#d62728', marker='D', s=90,
               zorder=6, edgecolor='black', linewidth=0.6,
               label=f'Inducing points ($m={len(x_inducing)}$)')

    ax.set_xlabel('Strike $K$ (toy "moneyness" axis)')
    ax.set_ylabel('Implied volatility $\\sigma_{\\mathrm{imp}}(K)$')
    ax.set_title('Toy sparse GP: fitting a volatility smile with\n'
                  f'{len(x_inducing)} inducing points instead of {n_train} data points')
    ax.legend(loc='upper center', fontsize=7.8, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'Wrote {path}')

    # Print a small numeric summary useful for the slides (hand-checkable)
    print("\n--- Toy sparse GP numeric summary (for slide text) ---")
    for xi, mi, si in zip([70, 100, 130],
                           *sparse_gp_dtc_predict(x_train, y_train, x_inducing,
                                                   np.array([70, 100, 130]),
                                                   lengthscale, sigma_f, sigma_noise)):
        pass
    m70, s70 = sparse_gp_dtc_predict(x_train, y_train, x_inducing, np.array([70.0]),
                                      lengthscale, sigma_f, sigma_noise)
    m100, s100 = sparse_gp_dtc_predict(x_train, y_train, x_inducing, np.array([100.0]),
                                        lengthscale, sigma_f, sigma_noise)
    m130, s130 = sparse_gp_dtc_predict(x_train, y_train, x_inducing, np.array([130.0]),
                                        lengthscale, sigma_f, sigma_noise)
    print(f"K=70  (near edge):  mean={m70[0]:.4f}, std={s70[0]:.4f}")
    print(f"K=100 (ATM, dense): mean={m100[0]:.4f}, std={s100[0]:.4f}")
    print(f"K=130 (near edge):  mean={m130[0]:.4f}, std={s130[0]:.4f}")


# ========================================================================
# Toy 3-point GP worked example (for the "by hand" numeric slide)
# ========================================================================
def three_point_worked_example():
    """
    A tiny 3-training-point, 1 test-point EXACT GP calculation (no sparsity),
    used on the "derive the posterior by hand" slide. Printed here so the
    numbers quoted on the slide are reproducible and exact.
    """
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.9, 0.0, -0.9])  # roughly y = -x with a bit of curvature
    lengthscale = 1.0
    sigma_f = 1.0
    sigma_noise = 0.1

    K = squared_exponential_kernel(x, x, lengthscale, sigma_f) + sigma_noise**2 * np.eye(3)
    x_star = np.array([0.5])
    k_star = squared_exponential_kernel(x_star, x, lengthscale, sigma_f)  # (1,3)
    k_starstar = squared_exponential_kernel(x_star, x_star, lengthscale, sigma_f)

    K_inv = np.linalg.inv(K)
    mean = k_star @ K_inv @ y
    var = k_starstar - k_star @ K_inv @ k_star.T

    print("\n--- 3-point worked GP example (exact, no sparsity) ---")
    print("K (with noise) =\n", np.round(K, 4))
    print("k_* (test-to-train) =", np.round(k_star, 4))
    print("k_** (test-to-test) =", np.round(k_starstar, 4))
    print(f"Posterior mean at x*=0.5:     {mean[0]:.4f}")
    print(f"Posterior variance at x*=0.5: {var[0,0]:.4f}")
    print(f"Posterior std at x*=0.5:      {np.sqrt(var[0,0]):.4f}")


if __name__ == '__main__':
    make_smile_figure('toy_smile.pdf')
    make_sparse_gp_figure('toy_sparse_gp.pdf')
    three_point_worked_example()
