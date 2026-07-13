#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 13 (A Bayesian Investment Analyst on the JSE)
slides.

All simulations here are ILLUSTRATIVE TOY EXAMPLES written from scratch by the
slide author for pedagogical purposes. They use a tiny fictional set of
"sell-side analyst report" feature vectors (sentiment score, EPS revision,
price-target change) and a from-scratch Bayesian logistic regression (BLR)
posterior. They are NOT reproductions of the book's real Bloomberg/FTSE-JSE
sell-side analyst dataset (29,935 reports, Jan 2004 - Jun 2018) -- those
results (Tables 13.1-13.5, Figs 13.1-13.12) are only summarised qualitatively
(and quoted numerically only where explicitly visible in the book's own
tables) in the slides themselves.

Outputs (all vector PDF, saved into this directory):
  1. fig_toy_analyst_predictions.pdf - bar chart of predicted P(correct
                                        direction) for 5 fictional analyst
                                        reports (part a)
  2. fig_toy_mcmc_compare.pdf         - MALA vs HMC vs S2HMC trace plots and
                                        ESS / mixing comparison on the toy
                                        BLR-ARD posterior (part b)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(13)

OUTDIR = "."

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ======================================================================
# Toy "sell-side analyst report" dataset (5 fictional reports)
# ======================================================================
# Features are the *conviction magnitude* of each signal -- i.e. how strong
# the report's call is, regardless of whether it is bullish or bearish
# (already standardised to roughly [0,1], mirroring the min-max scaling the
# book applies to the real Bloomberg features):
#   x1 = |sentiment score|       (report-text tone strength, 0 = neutral)
#   x2 = |EPS revision|          (|fractional change| in forecast EPS)
#   x3 = |price-target change|   (|fractional change| in the target price)
# Target y: toy "bidirectional accuracy" indicator (1 = stock later moved in
# the direction the report implied, 0 = it did not) -- mirroring the book's
# bidirectional-accuracy target (Sect. 13.4.1), but entirely fictional data.
# The toy story: reports with *stronger conviction* (larger-magnitude
# sentiment/EPS-revision/target-change, whichever direction) are more often
# vindicated than wishy-washy, near-zero-conviction "Hold" reports.

REPORT_NAMES = ["Alpha Capital", "Beta Securities", "Gamma Research",
                "Delta Equities", "Epsilon Analytics"]
RATINGS = ["Buy", "Sell", "Hold", "Buy", "Hold"]
X_RAW = np.array([
    [0.80, 0.060, 0.120],   # Alpha Capital    - strong bullish "Buy" call
    [0.60, 0.040, 0.080],   # Beta Securities  - strong bearish "Sell" call
    [0.10, 0.005, 0.010],   # Gamma Research   - wishy-washy "Hold" call
    [0.50, 0.030, 0.070],   # Delta Equities   - moderate bullish "Buy" call
    [0.20, 0.005, 0.015],   # Epsilon Analytics- weak-conviction "Hold" call
])
Y = np.array([1, 1, 0, 1, 0])  # toy "bidirectional accuracy" outcome
N_OBS, D_FEAT = X_RAW.shape
X = np.hstack([np.ones((N_OBS, 1)), X_RAW])  # prepend bias -> 4 columns
DIM = X.shape[1]

FEATURE_NAMES = ["bias", "|sentiment|", "|EPS revision|", "|price-target chg.|"]
# Fixed ARD prior variances alpha_j (the book jointly *infers* these via
# MCMC; here we fix them at illustrative values to keep the toy example
# self-contained and small).
ALPHA_PRIOR_VAR = np.array([4.0, 4.0, 4.0, 4.0])

RATING_COLOR = {"Buy": "#2E8B57", "Hold": "#DAA520", "Sell": "#B22222"}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


_GRAD_CALLS = [0]


def log_posterior(w):
    """Un-normalised log posterior ln p(w|D): Bernoulli log-lik + Gaussian
    ARD log-prior, cf. Eq. (13.1)-(13.2)."""
    z = X @ w
    p = sigmoid(z)
    eps = 1e-12
    ll = np.sum(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
    lp = -0.5 * np.sum(w ** 2 / ALPHA_PRIOR_VAR)
    return ll + lp


def grad_log_posterior(w):
    """Analytic gradient of the log posterior (derived on the slides):
    grad_w l(D|w) = sum_i (y_i - p_i) x_i ,  grad_w log-prior = -w / alpha."""
    _GRAD_CALLS[0] += 1
    z = X @ w
    p = sigmoid(z)
    grad_ll = X.T @ (Y - p)
    grad_lp = -w / ALPHA_PRIOR_VAR
    return grad_ll + grad_lp


def reset_grad_counter():
    _GRAD_CALLS[0] = 0


# ---------------- MALA sampler (from scratch, Eq. 13.3 + MH correction) ----

def run_mala(n_iter=4000, step=0.4, seed=1):
    rng = np.random.default_rng(seed)
    w = np.zeros(DIM)
    lp = log_posterior(w)
    grad = grad_log_posterior(w)
    chain = np.zeros((n_iter, DIM))
    n_accept = 0
    for m in range(n_iter):
        mean_fwd = w + 0.5 * step * grad
        prop = mean_fwd + np.sqrt(step) * rng.standard_normal(DIM)
        lp_prop = log_posterior(prop)
        grad_prop = grad_log_posterior(prop)
        mean_bwd = prop + 0.5 * step * grad_prop
        log_q_fwd = -np.sum((prop - mean_fwd) ** 2) / (2 * step)
        log_q_bwd = -np.sum((w - mean_bwd) ** 2) / (2 * step)
        log_alpha = (lp_prop + log_q_bwd) - (lp + log_q_fwd)
        if np.log(rng.uniform()) < log_alpha:
            w, lp, grad = prop, lp_prop, grad_prop
            n_accept += 1
        chain[m] = w
    print(f"[MALA]  acceptance rate: {n_accept/n_iter:.3f}")
    return chain


# ---------------- HMC sampler (from scratch, Eqs. 13.4-13.6) ---------------

def run_hmc(n_iter=1500, eps=0.30, L=15, seed=2):
    rng = np.random.default_rng(seed)
    w = np.zeros(DIM)
    chain = np.zeros((n_iter, DIM))
    n_accept = 0
    for m in range(n_iter):
        p0 = rng.standard_normal(DIM)
        w_new, p = w.copy(), p0.copy()
        grad = grad_log_posterior(w_new)
        for _ in range(L):
            p = p + 0.5 * eps * grad
            w_new = w_new + eps * p
            grad = grad_log_posterior(w_new)
            p = p + 0.5 * eps * grad
        H0 = -log_posterior(w) + 0.5 * np.sum(p0 ** 2)
        H1 = -log_posterior(w_new) + 0.5 * np.sum(p ** 2)
        if np.log(rng.uniform()) < (H0 - H1):
            w = w_new
            n_accept += 1
        chain[m] = w
    print(f"[HMC]   acceptance rate: {n_accept/n_iter:.3f}")
    return chain


# ---------------- S2HMC sampler (from scratch, Eqs. 13.9-13.14) ------------
# Simplified illustrative version: mass matrix M = I, potential
# U(w) = -log p(w|D). The pre-/post-processing maps (13.11)-(13.14) are
# solved by a few fixed-point iterations, and the accept/reject step uses
# the 4th-order separable shadow Hamiltonian (13.10) in place of the true
# Hamiltonian -- this is the mechanism that lets S2HMC conserve energy
# better than plain leapfrog/HMC and so produce less auto-correlated draws.

def U_grad(w):
    """U_w = grad of the potential U(w) = -log posterior."""
    return -grad_log_posterior(w)


def shadow_forward_map(w, p, eps, n_fp=3):
    """Solve (13.11)-(13.12) for (what, phat) by fixed-point iteration."""
    phat = p.copy()
    for _ in range(n_fp):
        phat = p - (eps / 24.0) * (U_grad(w + eps * phat) - U_grad(w - eps * phat))
    what = w + (eps ** 2 / 24.0) * (U_grad(w + eps * phat) + U_grad(w - eps * phat))
    return what, phat


def shadow_backward_map(what, phat, eps):
    """Apply (13.13)-(13.14) to map back off the shadow manifold."""
    w = what - (eps ** 2 / 24.0) * (U_grad(what + eps * phat) + U_grad(what - eps * phat))
    p = phat + (eps / 24.0) * (U_grad(what + eps * phat) - U_grad(what - eps * phat))
    return w, p


def shadow_hamiltonian(w, p, eps):
    """4th-order separable shadow Hamiltonian, Eq. (13.10), with M = I."""
    Uw = U_grad(w)
    return (-log_posterior(w)) + 0.5 * np.sum(p ** 2) + (eps ** 2 / 24.0) * np.sum(Uw ** 2)


def run_s2hmc(n_iter=1500, eps=0.30, L=15, seed=3):
    rng = np.random.default_rng(seed)
    w = np.zeros(DIM)
    chain = np.zeros((n_iter, DIM))
    n_accept = 0
    for m in range(n_iter):
        p0 = rng.standard_normal(DIM)
        what, phat = shadow_forward_map(w, p0, eps)
        H0 = shadow_hamiltonian(w, p0, eps)
        w_l, p_l = what.copy(), phat.copy()
        grad = grad_log_posterior(w_l)
        for _ in range(L):
            p_l = p_l + 0.5 * eps * grad
            w_l = w_l + eps * p_l
            grad = grad_log_posterior(w_l)
            p_l = p_l + 0.5 * eps * grad
        w_new, p_new = shadow_backward_map(w_l, p_l, eps)
        H1 = shadow_hamiltonian(w_new, p_new, eps)
        if np.log(rng.uniform()) < (H0 - H1):
            w = w_new
            n_accept += 1
        chain[m] = w
    print(f"[S2HMC] acceptance rate: {n_accept/n_iter:.3f}")
    return chain


# ---------------- Effective sample size (Geyer initial positive seq.) ------

def effective_sample_size(x, max_lag=300):
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if var < 1e-12:
        return float(n)
    max_lag = min(max_lag, n - 2)
    acf = np.array([np.sum(x[:n - k] * x[k:]) / (n * var) for k in range(1, max_lag)])
    s = 0.0
    for k in range(0, len(acf) - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        s += pair
    return n / (1 + 2 * s)


# ======================================================================
# FIGURE (a): toy predicted probabilities for the 5 fictional reports
# ======================================================================

def make_prediction_figure(w_post_mean):
    p_hat = sigmoid(X @ w_post_mean)
    colors = [RATING_COLOR[r] for r in RATINGS]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(REPORT_NAMES, p_hat, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="decision threshold = 0.5")
    for bar, y_true, rating in zip(bars, Y, RATINGS):
        ax.annotate(f"y={y_true}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                    ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r"posterior mean $\hat p_i = \mathbb{E}[\sigma(\mathbf{w}^\top x_i)\mid D]$")
    ax.set_title("Toy BLR-ARD "'analyst'": predicted P(bidirectional accuracy)\n"
                 "(illustrative toy example -- not the book's JSE data)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=RATING_COLOR[r], edgecolor="black", label=f"{r} rating")
               for r in ["Buy", "Hold", "Sell"]]
    handles.append(plt.Line2D([0], [0], color="gray", ls="--", label="threshold = 0.5"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xticks(range(len(REPORT_NAMES)))
    ax.set_xticklabels(REPORT_NAMES, rotation=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_analyst_predictions.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_analyst_predictions.pdf")


# ======================================================================
# FIGURE (b): MALA vs HMC vs S2HMC toy mixing / ESS comparison
# ======================================================================

def make_mcmc_compare_figure(mala_chain, hmc_chain, s2hmc_chain, grad_calls):
    burn = {"MALA": len(mala_chain) // 4, "HMC": len(hmc_chain) // 4,
            "S2HMC": len(s2hmc_chain) // 4}
    chains = {"MALA": mala_chain, "HMC": hmc_chain, "S2HMC": s2hmc_chain}
    colors = {"MALA": "steelblue", "HMC": "darkorange", "S2HMC": "seagreen"}

    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1.0])

    # --- panel 1: trace of the "sentiment" coefficient w_1 ---
    ax0 = fig.add_subplot(gs[0])
    for name, chain in chains.items():
        b = burn[name]
        ax0.plot(chain[b:b + 400, 1], color=colors[name], lw=0.9, alpha=0.85, label=name)
    ax0.set_xlabel("iteration (post burn-in)")
    ax0.set_ylabel(r"$w_{\mathrm{sentiment}}$")
    ax0.set_title("Trace: sentiment coefficient")
    ax0.legend(fontsize=8)

    # --- panel 2: autocorrelation of w_1 ---
    ax1 = fig.add_subplot(gs[1])
    for name, chain in chains.items():
        b = burn[name]
        x = chain[b:, 1] - np.mean(chain[b:, 1])
        var = np.var(x)
        max_lag = 60
        acf = np.array([1.0 if k == 0 else
                         np.sum(x[:-k] * x[k:]) / (len(x) * var) for k in range(max_lag)])
        ax1.plot(acf, "o-", ms=2.5, lw=1, color=colors[name], label=name)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("lag")
    ax1.set_ylabel("autocorrelation")
    ax1.set_title("Mixing: autocorrelation decay")
    ax1.legend(fontsize=8)

    # --- panel 3: ESS per gradient evaluation (mixing efficiency) ---
    ax2 = fig.add_subplot(gs[2])
    ess_vals, ess_per_grad = [], []
    names = ["MALA", "HMC", "S2HMC"]
    for name in names:
        b = burn[name]
        ess = effective_sample_size(chains[name][b:, 1])
        ess_vals.append(ess)
        ess_per_grad.append(ess / grad_calls[name])
    bars = ax2.bar(names, ess_per_grad, color=[colors[n] for n in names], edgecolor="black")
    ax2.set_ylabel("ESS per gradient evaluation")
    ax2.set_title("Sampling efficiency")
    for bar, ess in zip(bars, ess_vals):
        ax2.annotate(f"ESS={ess:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle("Toy example: MALA vs HMC vs S2HMC on the BLR-ARD "'analyst'" posterior\n"
                  "(illustrative simulation -- not the book's JSE experiments)", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_mcmc_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_mcmc_compare.pdf")
    return dict(zip(names, ess_vals)), dict(zip(names, ess_per_grad))


if __name__ == "__main__":
    reset_grad_counter()
    mala_chain = run_mala()
    grad_calls_mala = _GRAD_CALLS[0]

    reset_grad_counter()
    hmc_chain = run_hmc()
    grad_calls_hmc = _GRAD_CALLS[0]

    reset_grad_counter()
    s2hmc_chain = run_s2hmc()
    grad_calls_s2hmc = _GRAD_CALLS[0]

    grad_calls = {"MALA": grad_calls_mala, "HMC": grad_calls_hmc, "S2HMC": grad_calls_s2hmc}

    # Posterior mean of w from the (best-mixing) MALA chain, post burn-in,
    # used to generate the toy predicted-probability bar chart.
    burn_mala = len(mala_chain) // 4
    w_post_mean = mala_chain[burn_mala:].mean(axis=0)
    print("Toy posterior mean w (bias, sentiment, EPS rev., price-target chg.):",
          np.round(w_post_mean, 3))

    make_prediction_figure(w_post_mean)
    ess, ess_per_grad = make_mcmc_compare_figure(mala_chain, hmc_chain, s2hmc_chain, grad_calls)

    print("\n--- Toy MCMC comparison summary (illustrative, NOT the book's numbers) ---")
    for name in ["MALA", "HMC", "S2HMC"]:
        print(f"{name:6s}: grad calls={grad_calls[name]:6d}, ESS={ess[name]:7.1f}, "
              f"ESS/grad-eval={ess_per_grad[name]:.4f}")
