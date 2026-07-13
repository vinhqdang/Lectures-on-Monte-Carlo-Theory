#!/usr/bin/env python3
"""
gen_figures.py
--------------
Reproduces every figure used in the Chapter 11 Beamer slides:

    "Shadow and Adaptive Hamiltonian Monte Carlo Methods for
     Calibrating the Nelson and Siegel Model"
     (Mongwe, Mbuvha & Marwala, 2025, Chapter 11)

This is a from-scratch, illustrative TOY simulation written by the slide
author -- it is NOT code taken from the book. It is used only to build
intuition about the Nelson-Siegel (NS) model and about how Hamiltonian
Monte Carlo (HMC) style samplers can calibrate it in a Bayesian way.

Figures produced (saved as vector PDF into this folder):
    fig_ns_loadings.pdf     -- the three NS factor loading curves
    fig_toy_calibration.pdf -- toy synthetic yield curve + from-scratch
                               HMC-recovered fit vs. the true curve
    fig_toy_posteriors.pdf  -- posterior histograms of the 4 recovered
                               NS parameters from the toy HMC run

Run with: python3 gen_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. The Nelson-Siegel model
# ----------------------------------------------------------------------
# Yield curve (Eq. 11.2 in the book):
#   y(tau) = beta0
#            + beta1 * (1 - exp(-lam*tau)) / (lam*tau)
#            + beta2 * ( (1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau) )
#
# where beta0 = level (long-term factor, loading -> 1),
#       beta1 = slope  (short-term factor, loading -> 1 as tau->0, -> 0 as tau->inf),
#       beta2 = curvature (medium-term "hump" factor, loading -> 0 at both ends),
#       lam   = decay parameter controlling where the hump/loading peaks.


def loadings(tau, lam):
    """Return (level, slope, curvature) factor loadings at maturities tau."""
    tau = np.asarray(tau, dtype=float)
    lt = lam * tau
    level = np.ones_like(tau)
    slope = (1.0 - np.exp(-lt)) / lt
    curvature = slope - np.exp(-lt)
    return level, slope, curvature


def ns_yield(tau, beta0, beta1, beta2, lam):
    """Nelson-Siegel yield y(tau) (Eq. 11.2), tau in years, output in decimal."""
    level, slope, curvature = loadings(tau, lam)
    return beta0 * level + beta1 * slope + beta2 * curvature


# ----------------------------------------------------------------------
# Figure (a): the three factor loading curves
# ----------------------------------------------------------------------
lam_plot = 1.3  # same decay parameter used in the book's simulated example
tau_grid = np.linspace(1e-3, 20, 800)
level, slope, curvature = loadings(tau_grid, lam_plot)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot(tau_grid, level, lw=2.4, color="#1b4f72", label=r"Level: loading on $\beta_0$ (=1)")
ax.plot(tau_grid, slope, lw=2.4, color="#c0392b", label=r"Slope: loading on $\beta_1$")
ax.plot(tau_grid, curvature, lw=2.4, color="#1e8449", label=r"Curvature: loading on $\beta_2$")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Maturity " + r"$\tau$" + " (years)")
ax.set_ylabel("Factor loading")
ax.set_title(r"Nelson--Siegel factor loadings ($\lambda = 1.3$)")
ax.legend(loc="upper right", fontsize=9, frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_ns_loadings.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# 2. TOY illustrative example: simulate data, then calibrate with a
#    from-scratch Hamiltonian Monte Carlo sampler.
#
#    NOTE: this is the slide author's own illustrative toy simulation,
#    written independently of the book's code. We reuse the SAME true
#    parameter values quoted in the book's simulated experiment
#    (beta0=0.04, beta1=-0.05, beta2=0.2, lambda=1.3; see book Sec. 11.4.1)
#    purely so the toy example can be sanity-checked against the book's
#    qualitative story, but the noise draw, priors, and HMC sampler here
#    are all constructed independently for this slide deck.
# ----------------------------------------------------------------------
true_beta0, true_beta1, true_beta2, true_lam = 0.04, -0.05, 0.20, 1.30

maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
true_yields_pct = 100.0 * ns_yield(maturities, true_beta0, true_beta1, true_beta2, true_lam)

noise_std_pct = 0.05  # 5 basis points of i.i.d. Gaussian observation noise
observed_yields_pct = true_yields_pct + rng.normal(0.0, noise_std_pct, size=maturities.shape)

# --- Bayesian model -----------------------------------------------------
# Parametrize w = (z0, beta1, beta2, z3) where beta0 = exp(z0), lam = exp(z3)
# so that beta0 > 0 and lam > 0 automatically (this implements the book's
# LogNormal(beta0), LogNormal(lambda) priors as Normal priors on z0, z3).
#
#   Likelihood:  y_obs(tau_i) ~ Normal( y_NS(tau_i; w), sigma_obs^2 )
#   Priors:      z0  ~ Normal(log(0.03), 0.7^2)      [LogNormal(beta0)]
#                beta1 ~ Normal(0, 0.5^2)             [Gaussian(beta1)]
#                beta2 ~ Normal(0, 0.5^2)             [Gaussian(beta2)]
#                z3  ~ Normal(log(1.0), 0.7^2)        [LogNormal(lambda)]
#
# sigma_obs is treated as known (=noise_std_pct) to keep the toy model
# to exactly the 4 NS parameters, matching Table 11.1 of the book.

sigma_obs = noise_std_pct
prior_mu_z0, prior_sd_z0 = np.log(0.03), 0.7
prior_sd_beta = 0.5
prior_mu_z3, prior_sd_z3 = np.log(1.0), 0.7


def unpack(w):
    z0, b1, b2, z3 = w
    return np.exp(z0), b1, b2, np.exp(z3)


def neg_log_posterior(w):
    """U(w): potential energy = -log p(w | data) up to a constant."""
    z0, b1, b2, z3 = w
    b0, _, _, lam = unpack(w)
    model = ns_yield(maturities, b0, b1, b2, lam) * 100.0
    resid = observed_yields_pct - model
    nll = 0.5 * np.sum(resid**2) / sigma_obs**2
    log_prior = (
        -0.5 * ((z0 - prior_mu_z0) / prior_sd_z0) ** 2
        - 0.5 * (b1 / prior_sd_beta) ** 2
        - 0.5 * (b2 / prior_sd_beta) ** 2
        - 0.5 * ((z3 - prior_mu_z3) / prior_sd_z3) ** 2
    )
    return nll - log_prior


def grad_U(w, h=1e-6):
    """Numerical (central-difference) gradient of U(w)."""
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        g[i] = (neg_log_posterior(wp) - neg_log_posterior(wm)) / (2 * h)
    return g


def leapfrog(w, p, eps, L, grad_U):
    """Standard leapfrog integrator (Eq. 11.4 of the book), mass matrix M = I."""
    w = w.copy()
    p = p - 0.5 * eps * grad_U(w)
    for step in range(L):
        w = w + eps * p
        if step != L - 1:
            p = p - eps * grad_U(w)
    p = p - 0.5 * eps * grad_U(w)
    return w, -p


def hmc_sample(w0, n_samples, n_burnin, eps, L, seed=0):
    """Plain from-scratch HMC (Eqs. 11.3-11.5 of the book) with M = I."""
    local_rng = np.random.default_rng(seed)
    w = w0.copy()
    samples = np.zeros((n_samples, len(w0)))
    n_accept = 0
    n_total = n_samples + n_burnin
    for it in range(n_total):
        p0 = local_rng.normal(size=w.shape)
        H0 = neg_log_posterior(w) + 0.5 * np.sum(p0**2)
        w_new, p_new = leapfrog(w, p0, eps, L, grad_U)
        H1 = neg_log_posterior(w_new) + 0.5 * np.sum(p_new**2)
        log_alpha = -(H1 - H0)
        if np.log(local_rng.uniform()) < log_alpha:
            w = w_new
            n_accept += 1
        if it >= n_burnin:
            samples[it - n_burnin] = w
    accept_rate = n_accept / n_total
    return samples, accept_rate


# Initialize at a deliberately "wrong" point to show recovery
w0 = np.array([np.log(0.02), 0.0, 0.0, np.log(0.8)])
n_burnin, n_samples = 1000, 4000
eps, L = 0.02, 25  # step size, trajectory length (book fixes L=25 for HMC/S2HMC)

samples, accept_rate = hmc_sample(w0, n_samples, n_burnin, eps, L, seed=1)

beta0_samples = np.exp(samples[:, 0])
beta1_samples = samples[:, 1]
beta2_samples = samples[:, 2]
lam_samples = np.exp(samples[:, 3])

post_mean = np.array([
    beta0_samples.mean(), beta1_samples.mean(),
    beta2_samples.mean(), lam_samples.mean(),
])
post_std = np.array([
    beta0_samples.std(), beta1_samples.std(),
    beta2_samples.std(), lam_samples.std(),
])

print("=== Toy from-scratch HMC calibration of the Nelson-Siegel model ===")
print(f"Leapfrog step size eps = {eps}, trajectory length L = {L}")
print(f"HMC acceptance rate    = {accept_rate:.3f}")
print(f"{'param':>8s} {'true':>10s} {'post.mean':>10s} {'post.std':>10s}")
names = ["beta0", "beta1", "beta2", "lambda"]
truths = [true_beta0, true_beta1, true_beta2, true_lam]
for nm, tr, m, s in zip(names, truths, post_mean, post_std):
    print(f"{nm:>8s} {tr:10.4f} {m:10.4f} {s:10.4f}")

fitted_beta0, fitted_beta1, fitted_beta2, fitted_lam = post_mean

tau_fine = np.linspace(0.05, 30, 400)
true_curve_pct = 100.0 * ns_yield(tau_fine, true_beta0, true_beta1, true_beta2, true_lam)
fitted_curve_pct = 100.0 * ns_yield(tau_fine, fitted_beta0, fitted_beta1, fitted_beta2, fitted_lam)

rmse_bp = 100.0 * np.sqrt(np.mean(
    (ns_yield(maturities, true_beta0, true_beta1, true_beta2, true_lam)
     - ns_yield(maturities, fitted_beta0, fitted_beta1, fitted_beta2, fitted_lam)) ** 2
))
print(f"RMSE(true fit vs. HMC-fitted fit) on the 11 tenors = {rmse_bp:.4f} bp")

# ----------------------------------------------------------------------
# Figure (b): true vs. noisy synthetic data vs. HMC-fitted curve
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(tau_fine, true_curve_pct, color="#1b4f72", lw=2.2,
        label="True NS curve (author's toy simulation)")
ax.plot(tau_fine, fitted_curve_pct, color="#c0392b", lw=2.0, ls="--",
        label="Posterior-mean fit (from-scratch HMC)")
ax.scatter(maturities, observed_yields_pct, color="#1e8449", zorder=5, s=45,
           marker="x", label="Noisy synthetic observations")
ax.set_xlabel("Maturity " + r"$\tau$" + " (years)")
ax.set_ylabel("Yield (%)")
ax.set_title("Toy illustration: recovering NS parameters with HMC")
ax.legend(loc="best", fontsize=9, frameon=True)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_toy_calibration.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Figure (c): posterior histograms for the 4 recovered parameters
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
param_samples = [beta0_samples, beta1_samples, beta2_samples, lam_samples]
labels = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\lambda$"]
for ax, samp, lab, tr in zip(axes.ravel(), param_samples, labels, truths):
    ax.hist(samp, bins=40, color="#2e86c1", alpha=0.85)
    ax.axvline(tr, color="#c0392b", lw=2.0, ls="--", label="true value")
    ax.set_title(f"Toy HMC posterior: {lab}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig_toy_posteriors.pdf")
plt.close(fig)

print("\nSaved: fig_ns_loadings.pdf, fig_toy_calibration.pdf, fig_toy_posteriors.pdf")
