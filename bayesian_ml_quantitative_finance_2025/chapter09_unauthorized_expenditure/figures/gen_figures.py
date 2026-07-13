#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 9 (Bayesian Detection of Unauthorized
Expenditure Using Langevin and Hamiltonian Monte Carlo) slides.

ALL simulations here -- the toy government-transactions dataset, the toy
Bayesian logistic regression fit, and the MH / MALA / HMC / Magnetic-HMC
samplers -- are ILLUSTRATIVE examples written FROM SCRATCH by the slide
author for pedagogical purposes. They are NOT reproductions of the book's
real South African municipal-finance dataset or its real results; those are
only summarized qualitatively (and quoted numerically only where explicitly
shown in the book's own tables/figures, e.g. Table 9.1, Table 9.2, Figs.
9.1-9.5) in the slides themselves.

Outputs (all vector PDF, saved into this directory):
  1. fig_hmc_vs_mhmc_schematic.pdf  - HMC (straight/elliptical) vs Magnetic
                                      HMC (curving) trajectories on a toy 2D
                                      quadratic potential.
  2. fig_toy_data_scatter.pdf       - the 6 toy "government transactions"
                                      plotted in feature space.
  3. fig_mh_vs_mala_toy.pdf         - trace plots + autocorrelation for a
                                      from-scratch MH sampler vs a
                                      from-scratch MALA sampler on the same
                                      toy 2D logistic-regression posterior.
  4. fig_toy_posterior_summary.pdf  - posterior histograms of the 3
                                      regression weights from the toy MALA
                                      fit (full model, all 3 parameters).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(7)

OUTDIR = "."

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ======================================================================
# TOY DATASET: 6 fictional government transactions
# ======================================================================
# Columns: description, department, amount (ZAR), budget-variance (%),
# label (0 = authorized, 1 = unauthorized)
TOY_TRANSACTIONS = [
    ("Roads resurfacing contract",        "Infrastructure",      850_000,  12,  0),
    ("Emergency IT systems upgrade",       "IT Services",       4_200_000, 145,  1),
    ("Community hall maintenance",         "Community Services",  310_000,   5,  0),
    ("Bulk water pipeline repair",         "Water & Sanitation", 2_600_000,  88,  1),
    ("Office stationery supply",           "Finance",              190_000,  -3,  0),
    ("Unbudgeted road-signage project",    "Roads & Transport",  3_100_000, 110,  1),
]

amount = np.array([t[2] for t in TOY_TRANSACTIONS], dtype=float) / 1000.0  # in R'000
bvar = np.array([t[3] for t in TOY_TRANSACTIONS], dtype=float)            # % over/under budget
y = np.array([t[4] for t in TOY_TRANSACTIONS], dtype=float)

# Standardize features (zero mean, unit variance) for numerically stable sampling
x1 = (amount - amount.mean()) / amount.std()
x2 = (bvar - bvar.mean()) / bvar.std()
X = np.column_stack([np.ones_like(x1), x1, x2])  # design matrix with intercept
N = len(y)

TAU = 2.0  # Gaussian prior std on each weight (weakly informative ARD-style prior)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_posterior_full(w):
    """3-parameter toy logistic regression log-posterior: w = (w0, w1, w2)."""
    z = X @ w
    # numerically stable log-likelihood
    ll = np.sum(y * (-np.logaddexp(0, -z)) + (1 - y) * (-np.logaddexp(0, z)))
    lp = -0.5 * np.sum((w / TAU) ** 2)
    return ll + lp


def grad_log_posterior_full(w):
    z = X @ w
    p = sigmoid(z)
    grad_ll = X.T @ (y - p)
    grad_lp = -w / TAU ** 2
    return grad_ll + grad_lp


def make_toy_data_figure():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colors = ["seagreen" if yi == 0 else "crimson" for yi in y]
    for i, t in enumerate(TOY_TRANSACTIONS):
        ax.scatter(amount[i], bvar[i], color=colors[i], s=140, zorder=5,
                   edgecolor="black", linewidth=0.6)
        ax.annotate(f"T{i+1}", (amount[i], bvar[i]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Transaction amount (R'000)")
    ax.set_ylabel("Budget-variance (%)")
    ax.set_title("Toy dataset: 6 fictional municipal transactions\n"
                  "(green = authorized, red = unauthorized) -- illustrative only")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",
                          markeredgecolor="black", markersize=10, label="authorized (y=0)"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                          markeredgecolor="black", markersize=10, label="unauthorized (y=1)")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_data_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_data_scatter.pdf")


# ======================================================================
# FROM-SCRATCH MALA sampler, full 3-parameter model
# ======================================================================

def run_mala_full(n_iter=8000, step=0.8, w0=None, seed=1):
    rng = np.random.default_rng(seed)
    if w0 is None:
        w0 = np.zeros(3)
    w = w0.copy()
    lp = log_posterior_full(w)
    grad = grad_log_posterior_full(w)
    chain = np.zeros((n_iter, 3))
    n_accept = 0
    for m in range(n_iter):
        mean_fwd = w + 0.5 * step * grad
        prop = mean_fwd + np.sqrt(step) * rng.standard_normal(3)
        lp_prop = log_posterior_full(prop)
        grad_prop = grad_log_posterior_full(prop)
        mean_bwd = prop + 0.5 * step * grad_prop
        log_q_fwd = -np.sum((prop - mean_fwd) ** 2) / (2 * step)
        log_q_bwd = -np.sum((w - mean_bwd) ** 2) / (2 * step)
        log_alpha = (lp_prop + log_q_bwd) - (lp + log_q_fwd)
        if np.log(rng.uniform()) < log_alpha:
            w, lp, grad = prop, lp_prop, grad_prop
            n_accept += 1
        chain[m] = w
    acc_rate = n_accept / n_iter
    print(f"[full 3-param MALA] acceptance rate: {acc_rate:.3f}")
    return chain


def make_posterior_summary_figure(chain, burn=1500):
    post = chain[burn:]
    names = [r"$w_0$ (intercept)", r"$w_1$ (amount)", r"$w_2$ (budget-variance \%)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    means, stds = [], []
    for i in range(3):
        axes[i].hist(post[:, i], bins=40, color="steelblue", alpha=0.85)
        m, s = post[:, i].mean(), post[:, i].std()
        means.append(m)
        stds.append(s)
        axes[i].axvline(m, color="crimson", lw=2, label=f"mean={m:.2f}")
        axes[i].axvline(0, color="gray", ls="--", lw=1)
        axes[i].set_title(names[i], fontsize=10)
        axes[i].legend(fontsize=8)
    fig.suptitle("Toy example: posterior weights from from-scratch MALA\n"
                  "(3-parameter Bayesian logistic regression, illustrative only)", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_posterior_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_posterior_summary.pdf")
    return means, stds


# ======================================================================
# 2-PARAMETER SUB-POSTERIOR for the MH-vs-MALA toy comparison
# (intercept fixed at its posterior mean from the full MALA fit)
# ======================================================================

def make_2d_logposterior(w0_fixed):
    def logpost2(w12):
        w = np.array([w0_fixed, w12[0], w12[1]])
        return log_posterior_full(w)

    def grad2(w12):
        w = np.array([w0_fixed, w12[0], w12[1]])
        g = grad_log_posterior_full(w)
        return g[1:]

    return logpost2, grad2


def run_mh_2d(logpost, n_iter=8000, prop_sd=0.35, theta0=None, seed=11):
    rng = np.random.default_rng(seed)
    if theta0 is None:
        theta0 = np.zeros(2)
    theta = theta0.copy()
    lp = logpost(theta)
    chain = np.zeros((n_iter, 2))
    n_accept = 0
    for m in range(n_iter):
        prop = theta + prop_sd * rng.standard_normal(2)
        lp_prop = logpost(prop)
        if np.log(rng.uniform()) < (lp_prop - lp):
            theta, lp = prop, lp_prop
            n_accept += 1
        chain[m] = theta
    print(f"[toy 2D MH]   acceptance rate: {n_accept/n_iter:.3f}")
    return chain


def run_mala_2d(logpost, grad, n_iter=8000, step=0.12, theta0=None, seed=12):
    rng = np.random.default_rng(seed)
    if theta0 is None:
        theta0 = np.zeros(2)
    theta = theta0.copy()
    lp = logpost(theta)
    g = grad(theta)
    chain = np.zeros((n_iter, 2))
    n_accept = 0
    for m in range(n_iter):
        mean_fwd = theta + 0.5 * step * g
        prop = mean_fwd + np.sqrt(step) * rng.standard_normal(2)
        lp_prop = logpost(prop)
        g_prop = grad(prop)
        mean_bwd = prop + 0.5 * step * g_prop
        log_q_fwd = -np.sum((prop - mean_fwd) ** 2) / (2 * step)
        log_q_bwd = -np.sum((theta - mean_bwd) ** 2) / (2 * step)
        log_alpha = (lp_prop + log_q_bwd) - (lp + log_q_fwd)
        if np.log(rng.uniform()) < log_alpha:
            theta, lp, g = prop, lp_prop, g_prop
            n_accept += 1
        chain[m] = theta
    print(f"[toy 2D MALA] acceptance rate: {n_accept/n_iter:.3f}")
    return chain


def autocorr(x, max_lag=60):
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    return np.array([1.0 if lag == 0 else np.sum(x[:-lag] * x[lag:]) / denom
                      for lag in range(max_lag)])


def make_mh_vs_mala_figure(mh_chain, mala_chain, burn=1500):
    mh_post = mh_chain[burn:]
    mala_post = mala_chain[burn:]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))

    axes[0, 0].plot(mh_chain[:, 1], color="seagreen", lw=0.6)
    axes[0, 0].axvline(burn, color="gray", ls="--", lw=1)
    axes[0, 0].set_title(r"MH trace: $w_2$ (budget-variance coef.)")
    axes[0, 0].set_xlabel("iteration")

    axes[0, 1].plot(mala_chain[:, 1], color="darkorange", lw=0.6)
    axes[0, 1].axvline(burn, color="gray", ls="--", lw=1)
    axes[0, 1].set_title(r"MALA trace: $w_2$ (budget-variance coef.)")
    axes[0, 1].set_xlabel("iteration")

    ac_mh = autocorr(mh_post[:, 1])
    ac_mala = autocorr(mala_post[:, 1])
    axes[1, 0].plot(ac_mh, "o-", ms=3, color="seagreen", label="MH")
    axes[1, 0].plot(ac_mala, "o-", ms=3, color="darkorange", label="MALA")
    axes[1, 0].axhline(0, color="gray", lw=0.8)
    axes[1, 0].set_xlabel("lag")
    axes[1, 0].set_ylabel(r"autocorrelation of $w_2$")
    axes[1, 0].set_title("Autocorrelation: MALA decays faster than MH")
    axes[1, 0].legend(fontsize=8.5)

    axes[1, 1].scatter(mh_post[::4, 0], mh_post[::4, 1], s=6, alpha=0.35,
                       color="seagreen", label="MH samples")
    axes[1, 1].scatter(mala_post[::4, 0], mala_post[::4, 1], s=6, alpha=0.35,
                       color="darkorange", label="MALA samples")
    axes[1, 1].set_xlabel(r"$w_1$ (amount coef.)")
    axes[1, 1].set_ylabel(r"$w_2$ (budget-variance coef.)")
    axes[1, 1].set_title("Posterior samples (2D toy sub-posterior)")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Toy example: from-scratch MH vs MALA on the same 2D\n"
                 "logistic-regression sub-posterior (illustrative only)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_mh_vs_mala_toy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_mh_vs_mala_toy.pdf")

    ess_mh = len(mh_post) / (1 + 2 * np.sum(ac_mh[1:20]))
    ess_mala = len(mala_post) / (1 + 2 * np.sum(ac_mala[1:20]))
    print(f"[toy 2D] rough ESS (w2): MH={ess_mh:.1f}  MALA={ess_mala:.1f}")


# ======================================================================
# FIGURE: HMC (straight/elliptical) vs Magnetic HMC (curving) schematic
# ======================================================================
# Toy 2D quadratic potential U(w) = 0.5 * w^T A w  (A anisotropic + correlated,
# playing the role of the negative-log-posterior "landscape").
A_MAT = np.array([[3.0, 1.0],
                   [1.0, 1.0]])


def grad_U(w):
    return A_MAT @ w


def leapfrog_hmc(w0, p0, eps, L):
    w, p = w0.copy(), p0.copy()
    traj = [w.copy()]
    for _ in range(L):
        p = p - 0.5 * eps * grad_U(w)
        w = w + eps * p
        p = p - 0.5 * eps * grad_U(w)
        traj.append(w.copy())
    return np.array(traj)


def leapfrog_mhmc(w0, p0, eps, L, G):
    """Illustrative symmetric splitting integrator for the magnetic dynamics
    dw/dt = p ,  dp/dt = -grad U(w) + G p   (Eq. 9.7 with M = I).
    G is skew-symmetric, representing the 'magnetic field' term."""
    w, p = w0.copy(), p0.copy()
    traj = [w.copy()]
    for _ in range(L):
        force = -grad_U(w) + G @ p
        p = p + 0.5 * eps * force
        w = w + eps * p
        force = -grad_U(w) + G @ p
        p = p + 0.5 * eps * force
        traj.append(w.copy())
    return np.array(traj)


def make_hmc_vs_mhmc_schematic():
    w0 = np.array([-1.8, 1.6])
    p0 = np.array([0.9, 0.2])
    eps, L = 0.09, 60
    g = 2.2
    G = np.array([[0.0, g], [-g, 0.0]])  # skew-symmetric "magnetic field" matrix

    traj_hmc = leapfrog_hmc(w0, p0, eps, L)
    traj_mhmc = leapfrog_mhmc(w0, p0, eps, L, G)

    # contours of the toy potential
    xs = np.linspace(-3, 3, 200)
    ys = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = 0.5 * (A_MAT[0, 0] * XX ** 2 + 2 * A_MAT[0, 1] * XX * YY + A_MAT[1, 1] * YY ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    for ax, traj, title in zip(
        axes, [traj_hmc, traj_mhmc],
        ["Standard HMC:\nforce field only -- smooth, direct sweeps",
         "Magnetic HMC:\n+ sideways force $\\mathbf{G p}$ -- curving, spiraling path"],
    ):
        ax.contour(XX, YY, ZZ, levels=12, cmap="Greys", alpha=0.6, linewidths=0.8)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="darkorange", lw=1.8)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="green", zorder=5, s=90, label="start")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="crimson", zorder=5, s=90, label="end")
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        ax.set_title(title, fontsize=10.5)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Schematic: same toy potential, same starting momentum --\n"
                 "the magnetic term bends the trajectory sideways (illustrative)", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_hmc_vs_mhmc_schematic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_hmc_vs_mhmc_schematic.pdf")


if __name__ == "__main__":
    make_toy_data_figure()
    make_hmc_vs_mhmc_schematic()

    full_chain = run_mala_full()
    means, stds = make_posterior_summary_figure(full_chain)

    logpost2, grad2 = make_2d_logposterior(w0_fixed=means[0])
    mh_chain = run_mh_2d(logpost2)
    mala_chain = run_mala_2d(logpost2, grad2)
    make_mh_vs_mala_figure(mh_chain, mala_chain)

    print("\n--- Toy posterior summary (illustrative, NOT the book's numbers) ---")
    for nm, m, s in zip(["w0 (intercept)", "w1 (amount)", "w2 (budget-variance %)"], means, stds):
        print(f"  {nm}: mean={m:.3f}, std={s:.3f}")
    # illustrative predicted probability of "unauthorized" for a new transaction
    new_amount = 2_000.0  # R'000
    new_bvar = 60.0       # %
    x1_new = (new_amount - amount.mean()) / amount.std()
    x2_new = (new_bvar - bvar.mean()) / bvar.std()
    z_new = means[0] + means[1] * x1_new + means[2] * x2_new
    p_new = sigmoid(z_new)
    print(f"  Illustrative new transaction (R2,000,000 spend, +60% over budget): "
          f"P(unauthorized) = {p_new:.3f}")
