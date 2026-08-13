#!/usr/bin/env python3
"""
Generate figures for Chapter 2.3 (Statistical Inference and Estimation) slides.
An Introduction to Universal Artificial Intelligence, Hutter et al., 2024.

Run with: conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

OUT = "figures"

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Figure 1: Convergence of the MLE (running sample mean) to the true theta
# ---------------------------------------------------------------------------
def fig_mle_convergence():
    theta_true = 0.7
    n_max = 500
    x = (np.random.rand(n_max) < theta_true).astype(float)
    running_mean = np.cumsum(x) / np.arange(1, n_max + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(np.arange(1, n_max + 1), running_mean, color="#1f77b4", lw=1.6,
            label=r"$\widehat{\theta}_{ML}(n) = \frac{1}{n}\sum_{i=1}^n x_i$")
    ax.axhline(theta_true, color="#d62728", lw=1.8, ls="--",
               label=r"true $\theta = 0.7$")
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel(r"MLE estimate $\widehat{\theta}_{ML}$")
    ax.set_title("MLE of a Bernoulli parameter converges as $n$ grows")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mle_convergence.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Log-likelihood curve for a Bernoulli sample, showing the MLE
# ---------------------------------------------------------------------------
def fig_loglikelihood():
    n = 8
    s = 6  # number of 1's observed, e.g. x = 1,0,1,1,0,1,1,1
    theta = np.linspace(0.001, 0.999, 400)
    ll = s * np.log(theta) + (n - s) * np.log(1 - theta)
    theta_hat = s / n

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(theta, ll, color="#1f77b4", lw=2)
    ax.axvline(theta_hat, color="#d62728", ls="--", lw=1.8,
               label=r"$\widehat{\theta}_{ML} = %.3f$" % theta_hat)
    ll_max = s * np.log(theta_hat) + (n - s) * np.log(1 - theta_hat)
    ax.plot([theta_hat], [ll_max], "o", color="#d62728", ms=7, zorder=5)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\ln L(\theta)$")
    ax.set_title(r"Log-likelihood for $n=8$ flips, 6 heads ($x_i=1$)")
    ax.legend(loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loglikelihood_bernoulli.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: MSE = Variance + Bias^2 decomposition, comparing two estimators
#           T_n = (1/n) sum x_i           (unbiased)
#           T_n' = (1/n) sum x_i + 1/n      (biased, from Example 2.3.6)
# ---------------------------------------------------------------------------
def fig_mse_decomposition():
    n = 10
    theta_grid = np.linspace(0.01, 0.99, 200)

    # Unbiased estimator T_n: Var = theta(1-theta)/n, Bias = 0
    var_T = theta_grid * (1 - theta_grid) / n
    bias_T = np.zeros_like(theta_grid)
    mse_T = var_T + bias_T**2

    # Biased estimator T_n' = T_n + 1/n : same variance, Bias = 1/n
    var_Tp = var_T.copy()
    bias_Tp = np.full_like(theta_grid, 1.0 / n)
    mse_Tp = var_Tp + bias_Tp**2

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(theta_grid, mse_T, color="#1f77b4", lw=2,
            label=r"$T_n=\frac{1}{n}\sum x_i$ (unbiased)")
    ax.plot(theta_grid, mse_Tp, color="#d62728", lw=2, ls="--",
            label=r"$T_n'=T_n+\frac{1}{n}$ (biased)")
    ax.set_xlabel(r"true parameter $\theta$")
    ax.set_ylabel(r"$\mathrm{MSE}_\theta[T_n]$")
    ax.set_title(r"MSE of two estimators of $\theta$ ($n=10$ samples)")
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mse_decomposition.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Bias-Variance-MSE bar breakdown for the two estimators at theta=0.5
# ---------------------------------------------------------------------------
def fig_bias_variance_bars():
    n = 10
    theta = 0.5
    var = theta * (1 - theta) / n
    bias_T = 0.0
    bias_Tp = 1.0 / n
    mse_T = var + bias_T**2
    mse_Tp = var + bias_Tp**2

    labels = [r"$T_n$" + "\n(unbiased)", r"$T_n'=T_n+\frac{1}{n}$" + "\n(biased)"]
    variances = [var, var]
    bias_sq = [bias_T**2, bias_Tp**2]

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = np.arange(2)
    width = 0.5
    b1 = ax.bar(x, variances, width, label=r"$\mathrm{Var}_\theta[T_n]$", color="#1f77b4")
    b2 = ax.bar(x, bias_sq, width, bottom=variances, label=r"$\mathrm{Bias}(T_n)^2$", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("contribution to MSE")
    ax.set_title(r"MSE decomposition at $\theta=0.5$, $n=10$")
    for i, (v, bsq) in enumerate(zip(variances, bias_sq)):
        ax.text(i, v + bsq + 0.0015, f"MSE={v+bsq:.4f}", ha="center", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bias_variance_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_mle_convergence()
    fig_loglikelihood()
    fig_mse_decomposition()
    fig_bias_variance_bars()
    print("All figures written to", OUT)
