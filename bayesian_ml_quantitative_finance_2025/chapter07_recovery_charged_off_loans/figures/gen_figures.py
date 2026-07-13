#!/usr/bin/env python3
"""
gen_figures.py
--------------
Generates ILLUSTRATIVE figures for the Chapter 7 Beamer slides
("Bayesian Detection of Recovery on Charged-Off Loan Accounts").

Every figure produced here is a *toy simulation written by the slide
author*, used purely for pedagogical illustration. None of these numbers
are taken from the book's Lending Club dataset or the book's reported
results -- those are summarised qualitatively (and with the specific
numbers visible in the printed tables/figures) directly in the slides.

Outputs (all vector PDF, saved into this figures/ directory):
    fig_recovery_curves.pdf   -- toy "recovery amount over time" curves
    fig_mcmc_diagnostics.pdf  -- toy MALA trace plot + ESS comparison
    fig_toy_posterior.pdf     -- toy MALA posterior samples + summaries
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# Shared style
# -----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

OUTDIR = "."


# =========================================================================
# FIGURE (a): Toy "recovery amount over time" curves for synthetic
# charged-off loan accounts.
# =========================================================================
def make_recovery_curves():
    months = np.linspace(0, 24, 200)

    # Each fictional account has a different long-run recovery "ceiling"
    # (as a fraction of the exposure at default) and a different speed at
    # which it approaches that ceiling -- a simple saturating (logistic-
    # like) growth curve plus a little observation noise, purely for
    # illustration.
    accounts = [
        {"name": "Account A (never engages)",      "ceiling": 0.03, "rate": 0.35},
        {"name": "Account B (slow partial payer)",  "ceiling": 0.18, "rate": 0.20},
        {"name": "Account C (settles quickly)",     "ceiling": 0.45, "rate": 0.55},
        {"name": "Account D (legal action works)",  "ceiling": 0.30, "rate": 0.12},
        {"name": "Account E (near-full recovery)",  "ceiling": 0.68, "rate": 0.30},
        {"name": "Account F (short-lived promise)", "ceiling": 0.10, "rate": 0.45},
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for acc in accounts:
        curve = acc["ceiling"] * (1 - np.exp(-acc["rate"] * months / 3.0))
        noise = rng.normal(0, 0.006, size=months.shape)
        noisy_curve = np.clip(curve + np.cumsum(noise) * 0.15, 0, None)
        ax.plot(months, noisy_curve, lw=2, label=acc["name"])

    ax.axhline(0.10, color="black", ls=":", lw=1.2)
    ax.text(0.3, 0.115, "10% recovery threshold (toy classification target)",
            fontsize=7.5, style="italic")
    ax.set_xlabel("Months since charge-off (illustrative)")
    ax.set_ylabel("Cumulative recovery\n(fraction of exposure at default)")
    ax.set_title("Illustrative simulation: recovery trajectories of 6 fictional\ncharged-off loan accounts (author's toy example, not book data)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_recovery_curves.pdf")
    plt.close(fig)


# =========================================================================
# Toy Bayesian logistic regression + from-scratch MALA sampler
# used by both remaining figures.
# =========================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_toy_loan_dataset():
    """6 fictional charged-off accounts with 2 standardised features:
    days-since-default and outstanding-balance, plus a binary
    'recovered more than 10%?' outcome -- mirrors the book's binary
    target definition in Sect. 7.4.2, but this data is entirely
    invented for illustration."""
    days_since_default = np.array([30, 400, 90, 250, 600, 150.0])
    outstanding_balance = np.array([500, 8000, 1200, 4000, 12000, 2500.0])
    recovered_gt_10pct = np.array([1, 0, 1, 0, 0, 1.0])  # toy labels

    # standardise features for well-behaved sampling
    x1 = (days_since_default - days_since_default.mean()) / days_since_default.std()
    x2 = (outstanding_balance - outstanding_balance.mean()) / outstanding_balance.std()
    X = np.column_stack([np.ones_like(x1), x1, x2])  # intercept, days, balance
    y = recovered_gt_10pct
    return X, y, days_since_default, outstanding_balance, recovered_gt_10pct


def grad_log_posterior(w, X, y, prior_var):
    p = sigmoid(X @ w)
    grad_loglik = X.T @ (y - p)
    grad_logprior = -w / prior_var
    return grad_loglik + grad_logprior


def log_posterior(w, X, y, prior_var):
    p = np.clip(sigmoid(X @ w), 1e-9, 1 - 1e-9)
    loglik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    logprior = -0.5 * np.sum(w ** 2) / prior_var
    return loglik + logprior


def run_mala(X, y, n_iter=12000, burn_in=2000, step=0.35, prior_var=5.0, seed=0):
    """A minimal, from-scratch Metropolis-Adjusted Langevin Algorithm
    (MALA) sampler for the toy Bayesian logistic regression posterior."""
    local_rng = np.random.default_rng(seed)
    dim = X.shape[1]
    w = np.zeros(dim)
    samples = np.zeros((n_iter, dim))
    n_accept = 0

    for t in range(n_iter):
        grad = grad_log_posterior(w, X, y, prior_var)
        # Langevin proposal: drift (half step-size^2 * gradient) + Gaussian noise
        w_prop = w + 0.5 * step ** 2 * grad + step * local_rng.normal(size=dim)

        grad_prop = grad_log_posterior(w_prop, X, y, prior_var)

        # log proposal densities q(w -> w_prop) and q(w_prop -> w)
        fwd = w_prop - w - 0.5 * step ** 2 * grad
        bwd = w - w_prop - 0.5 * step ** 2 * grad_prop
        log_q_fwd = -np.sum(fwd ** 2) / (2 * step ** 2)
        log_q_bwd = -np.sum(bwd ** 2) / (2 * step ** 2)

        log_alpha = (log_posterior(w_prop, X, y, prior_var) + log_q_bwd
                     - log_posterior(w, X, y, prior_var) - log_q_fwd)

        if np.log(local_rng.uniform()) < log_alpha:
            w = w_prop
            n_accept += 1

        samples[t] = w

    accept_rate = n_accept / n_iter
    return samples[burn_in:], accept_rate


def effective_sample_size(x, max_lag=None):
    """Simple autocorrelation-based ESS estimate (Geyer initial positive
    sequence idea): ESS = N / (1 + 2 * sum of positive-paired
    autocorrelations)."""
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    xc = x - x.mean()
    var = np.dot(xc, xc) / n
    if var == 0:
        return float(n)

    acf = np.zeros(max_lag)
    for k in range(max_lag):
        acf[k] = np.dot(xc[:n - k], xc[k:]) / n / var

    total = 0.0
    k = 1
    while k + 1 < max_lag:
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum < 0:
            break
        total += pair_sum
        k += 2

    ess = n / (1.0 + 2.0 * total)
    return max(1.0, min(ess, float(n)))


# =========================================================================
# FIGURE (b): Toy MCMC diagnostic plot -- trace plot + ESS comparison
# for the toy Bayesian logistic regression, sampled with the from-scratch
# MALA sampler above.
# =========================================================================
def make_mcmc_diagnostics():
    X, y, *_ = make_toy_loan_dataset()
    # step size chosen so the acceptance rate lands near the ~60% target
    # the book uses for tuning MALA in Sect. 7.4.2
    samples, accept_rate = run_mala(X, y, n_iter=12000, burn_in=2000, step=1.6)

    param_names = ["intercept", r"$w_{\mathrm{days}}$", r"$w_{\mathrm{balance}}$"]
    ess_vals = [effective_sample_size(samples[:, j]) for j in range(samples.shape[1])]
    n_kept = samples.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Trace plot (first 3000 kept iterations, for readability)
    show = min(3000, n_kept)
    for j, name in enumerate(param_names):
        axes[0].plot(samples[:show, j], lw=0.7, alpha=0.85, label=name)
    axes[0].set_xlabel("MALA iteration (post burn-in)")
    axes[0].set_ylabel("Parameter value")
    axes[0].set_title(f"Trace plot: from-scratch MALA\n(toy 2-feature logistic model, "
                       f"accept rate $\\approx${accept_rate:.2f})")
    axes[0].legend(loc="upper right", fontsize=7.5)
    axes[0].grid(alpha=0.3)

    # ESS bar comparison: raw kept samples vs. effective sample size
    bar_x = np.arange(len(param_names))
    width = 0.35
    axes[1].bar(bar_x - width / 2, [n_kept] * len(param_names), width,
                label="Raw kept samples", color="#9ecae1")
    axes[1].bar(bar_x + width / 2, ess_vals, width,
                label="Effective sample size (ESS)", color="#3182bd")
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels(param_names)
    axes[1].set_ylabel("Count")
    axes[1].set_title("ESS vs. raw sample count\n(illustrative MALA run -- author's toy simulation)")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Toy MCMC diagnostics (not the book's Lending Club results)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_mcmc_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    return samples, ess_vals, accept_rate


# =========================================================================
# FIGURE (toy posterior summary): posterior samples + ARD-style
# importance summary for the toy running example.
# =========================================================================
def make_toy_posterior(samples):
    param_names = ["intercept", "days-since-default", "outstanding-balance"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) scatter of posterior draws in (w_days, w_balance) plane
    axes[0].scatter(samples[::5, 1], samples[::5, 2], s=6, alpha=0.25, color="#31a354")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_xlabel(r"$w_{\mathrm{days\,since\,default}}$")
    axes[0].set_ylabel(r"$w_{\mathrm{outstanding\,balance}}$")
    axes[0].set_title("Toy posterior samples\n(from-scratch MALA, 6 fictional accounts)")
    axes[0].grid(alpha=0.3)

    # (b) posterior mean +/- 1 s.d. per coefficient: crude ARD-style
    # "importance" summary -- larger posterior spread/mean magnitude
    # suggests a more influential feature, echoing (in miniature) the
    # alpha_i relevance-ranking idea used by BLR-ARD in the book.
    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    bar_x = np.arange(len(param_names))
    axes[1].bar(bar_x, means, yerr=stds, capsize=5, color="#756bb1", alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels(param_names, rotation=15)
    axes[1].set_ylabel("Posterior mean $\\pm$ 1 s.d.")
    axes[1].set_title("Toy posterior summaries\n(illustrative ARD-style importance)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Toy running example: posterior of a Bayesian logistic model for\n"
                  "'recovered $>$10%?' fit to 6 fictional charged-off accounts",
                  fontsize=9.5, y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_posterior.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_recovery_curves()
    samples, ess_vals, accept_rate = make_mcmc_diagnostics()
    make_toy_posterior(samples)
    print("Accept rate:", accept_rate)
    print("ESS per parameter:", ess_vals)
    print("Posterior means:", samples.mean(axis=0))
    print("Posterior stds:", samples.std(axis=0))
    print("All figures written to", OUTDIR)
