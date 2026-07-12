"""
gen_figures.py
Generates the figures used in Chapter 4 (Post-hoc testing and decision making
with e-values) slides.

Figure 1 (fig_posthoc_validity.pdf):
    Illustrates Definition 4.1 / Proposition 4.3.  Under the null hypothesis
    (fair coin, theta = 0.5), we repeatedly draw batches of coin tosses and
    compute, for each replicate:
        - a (continuous, approximately uniform) p-value from a one-sided
          normal-approximation test,
        - a likelihood-ratio e-value for the simple alternative theta = 0.6.
    We then track the running (cumulative) average of
        R_P = 1 / P            (the worst-case post-hoc "risk" for p-values;
                                 this is sup_alpha 1{P<=alpha}/alpha)
        R_E = E * 1{E >= 1}     (the worst-case post-hoc "risk" for e-values;
                                 this is sup_alpha 1{E>=1/alpha}/alpha)
    Theory (Markov's inequality, since E^P[E] <= 1) guarantees
        E^P[R_E] <= E^P[E] <= 1,
    so the e-value running average must stabilize at or below 1.  By
    contrast, R_P = 1/P has an infinite mean when P ~ Uniform(0,1), so its
    running average is unstable and keeps jumping upward -- this is exactly
    why choosing your significance level *after* seeing a p-value voids its
    validity, while the analogous operation with e-values remains valid.

Figure 2 (fig_optional_continuation.pdf):
    Illustrates Section 4.3 (optional continuation).  Under the null
    (theta = 0.5), we run a sequence of "batches" of coin tosses, and at each
    batch we can either:
        (a) multiply in a fresh likelihood-ratio e-value for theta = 0.6 and
            monitor the running product (an e-process); or
        (b) naively compute a fresh p-value for each batch and reject as soon
            as any batch p-value dips below 0.05 (repeated, uncorrected
            "peeking").
    We plot, as a function of the number of looks (batches examined so far),
    the empirical probability of having (falsely) rejected the true null by
    that point.  The e-process, thanks to Ville's inequality, stays
    controlled at level alpha for *all* looks simultaneously, while the
    naive repeated p-value peeking strategy inflates well beyond alpha the
    more batches / looks are allowed.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

rng = np.random.default_rng(2024)

# ----------------------------------------------------------------------
# Shared coin-fairness setup: H0: theta = 0.5 (fair coin), simple
# alternative used to build the likelihood-ratio e-value: theta = 0.6.
# ----------------------------------------------------------------------
THETA0 = 0.5
THETA1 = 0.6
LR_HEAD = THETA1 / THETA0          # = 1.2
LR_TAIL = (1 - THETA1) / (1 - THETA0)  # = 0.8


def lr_evalue(heads, n):
    """Likelihood-ratio e-value for theta1 vs theta0 given `heads` out of n."""
    tails = n - heads
    return (LR_HEAD ** heads) * (LR_TAIL ** tails)


def normal_pvalue(heads, n):
    """One-sided continuous p-value via normal approximation to Binomial(n,0.5)."""
    mean = n * THETA0
    sd = np.sqrt(n * THETA0 * (1 - THETA0))
    z = (heads - mean) / sd
    # one-sided p-value for testing theta0 vs theta > theta0
    p = 0.5 * erfc(z / np.sqrt(2.0))
    return np.clip(p, 1e-300, 1.0)


# ========================================================================
# FIGURE 1: post-hoc validity -- e-values vs. p-values
# ========================================================================
n_toss = 30           # tosses per replicate
M = 200_000           # number of independent replicates

heads = rng.binomial(n_toss, THETA0, size=M)
E = lr_evalue(heads, n_toss)
P = normal_pvalue(heads, n_toss)

R_P = 1.0 / P                    # sup_alpha 1{P<=alpha}/alpha
R_E = E * (E >= 1.0)             # sup_alpha 1{E>=1/alpha}/alpha

running_mean_P = np.cumsum(R_P) / np.arange(1, M + 1)
running_mean_E = np.cumsum(R_E) / np.arange(1, M + 1)

fig, ax = plt.subplots(figsize=(9, 5.2))
xs = np.arange(1, M + 1)
ax.plot(xs, running_mean_P, color="#c0392b", lw=1.3,
        label="p-value risk: running avg of sup_a 1{P<=a}/a = 1/P")
ax.plot(xs, running_mean_E, color="#2471a3", lw=1.6,
        label="e-value risk: running avg of sup_a 1{E>=1/a}/a = E . 1{E>=1}")
ax.axhline(1.0, color="black", ls="--", lw=1.0, label="guarantee: bounded by 1 (Markov, since E[E]<=1)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of simulated replicates (under H0: θ = 0.5)")
ax.set_ylabel("running average of post-hoc risk")
ax.set_title("Post-hoc validity: e-values stay bounded, p-values do not\n"
             "(coin fairness test, H0: θ=0.5 vs LR alternative θ=0.6)")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig("fig_posthoc_validity.pdf")
plt.close(fig)

# ========================================================================
# FIGURE 2: optional continuation -- e-process (Ville) vs. naive peeking
# ========================================================================
n_batches = 20
batch_size = 10
n_runs = 20_000
alpha = 0.05
threshold = 1.0 / alpha  # = 20

# heads_per_batch[run, k] ~ Binomial(batch_size, 0.5) under H0
heads_per_batch = rng.binomial(batch_size, THETA0, size=(n_runs, n_batches))

e_batch = lr_evalue(heads_per_batch, batch_size)          # (n_runs, n_batches)
p_batch = normal_pvalue(heads_per_batch, batch_size)      # (n_runs, n_batches)

# cumulative e-process (product of fresh e-values across batches/looks)
e_process = np.cumprod(e_batch, axis=1)

# has the e-process ever crossed the threshold 1/alpha by look k?
ever_crossed_e = np.cumsum(e_process >= threshold, axis=1) > 0
# has naive repeated peeking ever seen a batch p-value below alpha by look k?
ever_crossed_p = np.cumsum(p_batch < alpha, axis=1) > 0

reject_rate_e = ever_crossed_e.mean(axis=0)
reject_rate_p = ever_crossed_p.mean(axis=0)

looks = np.arange(1, n_batches + 1)

fig2, ax2 = plt.subplots(figsize=(9, 5.2))
ax2.plot(looks, reject_rate_p, "o-", color="#c0392b", lw=1.6, ms=4,
          label="naive repeated peeking with p-values\n(reject at first batch with p < 0.05)")
ax2.plot(looks, reject_rate_e, "s-", color="#2471a3", lw=1.8, ms=4,
          label="e-process monitoring (Ville)\n(reject at first look with E(k) ≥ 1/α)")
ax2.axhline(alpha, color="black", ls="--", lw=1.2, label="nominal level α = 0.05")
ax2.set_xlabel("number of looks / batches examined so far (each batch = 10 fresh tosses)")
ax2.set_ylabel("empirical false-rejection probability under H0")
ax2.set_title("Optional continuation: e-processes control error under repeated looks,\n"
              "naive p-value peeking does not")
ax2.set_ylim(0, max(reject_rate_p.max(), reject_rate_e.max()) * 1.15)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("fig_optional_continuation.pdf")
plt.close(fig2)

print("Final running-average post-hoc risk (p-value, should be unstable/large):", running_mean_P[-1])
print("Final running-average post-hoc risk (e-value, should be <= 1):", running_mean_E[-1])
print("Final false-rejection rate, naive p-value peeking over", n_batches, "looks:", reject_rate_p[-1])
print("Final false-rejection rate, e-process (Ville) over", n_batches, "looks:", reject_rate_e[-1])
print("Saved fig_posthoc_validity.pdf and fig_optional_continuation.pdf")
