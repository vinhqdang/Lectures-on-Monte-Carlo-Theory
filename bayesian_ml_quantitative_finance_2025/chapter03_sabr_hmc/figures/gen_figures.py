#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 3 (SABR model & Hamiltonian Monte Carlo) slides.

All simulations here are ILLUSTRATIVE TOY EXAMPLES written from scratch by the
slide author for pedagogical purposes. They are NOT reproductions of the
book's real (ZAR swaption) results -- those are only quoted qualitatively (and
numerically only where explicitly seen in the book's tables/figures) in the
slides themselves.

Outputs (all vector PDF, saved into this directory):
  1. fig_leapfrog_schematic.pdf   - marble-in-a-well + phase-space leapfrog picture
  2. fig_toy_sabr_data.pdf        - synthetic "SABR-like" swaption skew data
  3. fig_mala_toy.pdf             - MALA trace + posterior histograms on toy data
  4. fig_hmc_toy.pdf              - HMC trace + posterior histograms on toy data
  5. fig_mala_vs_hmc_autocorr.pdf - autocorrelation comparison MALA vs HMC
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

OUTDIR = "."

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ======================================================================
# FIGURE 1: Leapfrog trajectory schematic (marble in a well + phase space)
# ======================================================================


def make_leapfrog_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- Panel (a): marble rolling in a 1D quadratic potential well ----
    ax = axes[0]
    xs = np.linspace(-2.5, 2.5, 400)
    U = 0.5 * xs ** 2
    ax.plot(xs, U, color="steelblue", lw=2.5, label=r"$U(w)=\frac{1}{2}w^2$ (potential = $-\log$ posterior)")

    # Simulate a short leapfrog trajectory with eps=0.3, L=8, starting w0=-2, p0=0.3
    eps, L = 0.3, 8
    w, p = -2.0, 0.3

    def Uprime(w):
        return w

    traj_w = [w]
    for _ in range(L):
        p = p - 0.5 * eps * Uprime(w)
        w = w + eps * p
        p = p - 0.5 * eps * Uprime(w)
        traj_w.append(w)
    traj_w = np.array(traj_w)
    traj_U = 0.5 * traj_w ** 2

    ax.plot(traj_w, traj_U, "o-", color="darkorange", ms=7, lw=1.5,
             label="leapfrog steps (marble position)")
    for i, (wx, uy) in enumerate(zip(traj_w, traj_U)):
        ax.annotate(str(i), (wx, uy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="darkorange")
    ax.scatter([traj_w[0]], [traj_U[0]], color="green", zorder=5, s=90, label="start ($m=0$)")
    ax.scatter([traj_w[-1]], [traj_U[-1]], color="crimson", zorder=5, s=90, label="after $L=8$ steps")
    ax.set_xlabel(r"parameter $w$")
    ax.set_ylabel(r"potential energy $U(w)$")
    ax.set_title("Marble rolling on the negative log-posterior surface")
    ax.legend(fontsize=7.5, loc="upper center")

    # ---- Panel (b): phase space (w, p) trajectory: near-ellipse ----
    ax2 = axes[1]
    w, p = -2.0, 0.3
    traj_w2, traj_p2 = [w], [p]
    for _ in range(L):
        p = p - 0.5 * eps * Uprime(w)
        w = w + eps * p
        p = p - 0.5 * eps * Uprime(w)
        traj_w2.append(w)
        traj_p2.append(p)
    traj_w2 = np.array(traj_w2)
    traj_p2 = np.array(traj_p2)

    # true continuous-time orbit (exact harmonic oscillator ellipse) for reference
    theta = np.linspace(0, 2 * np.pi, 400)
    r = np.sqrt(traj_w2[0] ** 2 + traj_p2[0] ** 2)
    ax2.plot(r * np.cos(theta), r * np.sin(theta), "--", color="gray", lw=1,
              label="exact Hamiltonian flow (continuous time)")
    ax2.plot(traj_w2, traj_p2, "o-", color="darkorange", ms=7, lw=1.5,
              label="leapfrog trajectory (discrete steps)")
    for i, (wx, py) in enumerate(zip(traj_w2, traj_p2)):
        ax2.annotate(str(i), (wx, py), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color="darkorange")
    ax2.scatter([traj_w2[0]], [traj_p2[0]], color="green", zorder=5, s=90, label="start")
    ax2.scatter([traj_w2[-1]], [traj_p2[-1]], color="crimson", zorder=5, s=90, label="end")
    ax2.set_xlabel(r"parameter $w$")
    ax2.set_ylabel(r"momentum $p$")
    ax2.set_title("Phase space $(w,p)$: leapfrog nearly\nconserves the Hamiltonian")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Schematic: one HMC trajectory built from $L$ leapfrog steps", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_leapfrog_schematic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_leapfrog_schematic.pdf")


# ======================================================================
# Toy "SABR-like" model: infer (alpha, rho) from synthetic swaption vols
# ======================================================================
#
# We use the SAME closed-form normal-volatility expansion structure that
# appears in the book (Hagan et al. 2002 normal vol expansion), but we fix
# beta and nu at illustrative values and only treat (alpha, rho) as the
# two unknown parameters to keep this a clean, from-scratch TOY example.

F_FWD = 0.07      # toy forward rate (7%)
BETA = 0.5        # elasticity, fixed as in the book's experiments
NU = 0.8          # vol-of-vol, fixed (toy value)
T_MAT = 1.0       # toy maturity
TRUE_ALPHA = 0.03
TRUE_RHO = -0.35
OBS_NOISE_SD = 8.0  # noise in basis points on the normal vol

# Diagonal preconditioner (plays the role of the M^{-1} matrix in the book's
# HMC/S2HMC update equations): alpha and rho live on very different natural
# scales (~0.01 vs ~1), so an isotropic step size is badly behaved. This is
# a standard, from-scratch fix -- NOT part of the book's reported results.
PRECOND = np.array([8e-8, 4e-3])  # M^{-1} diagonal
MASS = 1.0 / PRECOND               # M diagonal


def sabr_normal_vol(k, alpha, rho, nu=NU, beta=BETA, f=F_FWD, t=T_MAT):
    """Hagan et al. (2002)-style normal-volatility expansion (as used for
    sigma_B(f,k) in the book, Eqs. 3.9-3.15), returned in basis points."""
    k = np.asarray(k, dtype=float)
    fk_avg = np.sqrt(f * k)
    # A term
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(
            np.isclose(f, k),
            alpha * f ** beta,
            (1 - beta) * (f - k) / (f ** (1 - beta) - k ** (1 - beta)) * alpha,
        )
    B = -beta * (2 - beta) * alpha ** 2 / (24 * fk_avg ** (2 - 2 * beta))
    C = rho * alpha * nu * beta / (4 * fk_avg ** (1 - beta))
    D = (2 - 3 * rho ** 2) / 6.0 * (nu ** 2) / 4.0
    z = (nu / alpha) * (k - f)
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1 + 2 * rho * z + z ** 2) + z + rho) / (1 + rho))
        zx_ratio = np.where(np.isclose(z, 0.0), 1.0, z / xz)
    sigma = A * zx_ratio * (1 + (B + C + D) * t)
    return sigma * 1e4  # convert to basis points


# Synthetic strikes spanning a skew, and synthetic noisy observations
STRIKES = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11])
TRUE_VOLS = sabr_normal_vol(STRIKES, TRUE_ALPHA, TRUE_RHO)
OBS_VOLS = TRUE_VOLS + np.random.normal(0, OBS_NOISE_SD, size=STRIKES.shape)


def make_toy_data_figure():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(STRIKES * 100, TRUE_VOLS, "-", color="steelblue", lw=2,
             label=fr"true skew ($\alpha={TRUE_ALPHA}$, $\rho={TRUE_RHO}$)")
    ax.scatter(STRIKES * 100, OBS_VOLS, color="crimson", zorder=5,
                label="synthetic noisy observations")
    ax.set_xlabel("Strike (%)")
    ax.set_ylabel("Normal volatility (bps)")
    ax.set_title("Toy synthetic \"swaption-like\" volatility skew\n(illustrative only -- not the book's data)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_toy_sabr_data.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_sabr_data.pdf")


# ---------------- Toy log-posterior on (alpha, rho) --------------------

PRIOR_ALPHA_MEAN, PRIOR_ALPHA_SD = 0.02, 0.02
PRIOR_RHO_MEAN, PRIOR_RHO_SD = 0.0, 0.5


def log_posterior(theta):
    alpha, rho = theta
    if alpha <= 1e-5 or abs(rho) >= 0.999:
        return -np.inf
    model = sabr_normal_vol(STRIKES, alpha, rho)
    ll = -0.5 * np.sum(((OBS_VOLS - model) / OBS_NOISE_SD) ** 2)
    lp_alpha = -0.5 * ((alpha - PRIOR_ALPHA_MEAN) / PRIOR_ALPHA_SD) ** 2
    lp_rho = -0.5 * ((rho - PRIOR_RHO_MEAN) / PRIOR_RHO_SD) ** 2
    return ll + lp_alpha + lp_rho


def grad_log_posterior(theta, h=1e-5):
    """Finite-difference gradient (from scratch, no autodiff library)."""
    grad = np.zeros(2)
    for i in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        lp_p, lp_m = log_posterior(tp), log_posterior(tm)
        if np.isinf(lp_p) or np.isinf(lp_m):
            grad[i] = 0.0
        else:
            grad[i] = (lp_p - lp_m) / (2 * h)
    return grad


# ---------------- MALA sampler (from scratch) --------------------------

def run_mala(n_iter=6000, base_step=6.0, theta0=None, seed=1):
    """Preconditioned MALA: proposal covariance is base_step * PRECOND,
    drift is 0.5 * base_step * PRECOND * grad(log posterior)."""
    rng = np.random.default_rng(seed)
    if theta0 is None:
        theta0 = np.array([0.025, -0.2])
    theta = theta0.copy()
    lp = log_posterior(theta)
    grad = grad_log_posterior(theta)
    chain = np.zeros((n_iter, 2))
    n_accept = 0
    step_vec = base_step * PRECOND
    for m in range(n_iter):
        mean_fwd = theta + 0.5 * step_vec * grad
        prop = mean_fwd + np.sqrt(step_vec) * rng.standard_normal(2)
        lp_prop = log_posterior(prop)
        if np.isinf(lp_prop):
            chain[m] = theta
            continue
        grad_prop = grad_log_posterior(prop)
        mean_bwd = prop + 0.5 * step_vec * grad_prop
        # log q(theta -> prop) and q(prop -> theta) for the Gaussian proposal
        log_q_fwd = -np.sum((prop - mean_fwd) ** 2 / step_vec) / 2
        log_q_bwd = -np.sum((theta - mean_bwd) ** 2 / step_vec) / 2
        log_alpha = (lp_prop + log_q_bwd) - (lp + log_q_fwd)
        if np.log(rng.uniform()) < log_alpha:
            theta, lp, grad = prop, lp_prop, grad_prop
            n_accept += 1
        chain[m] = theta
    print(f"MALA acceptance rate: {n_accept / n_iter:.3f}")
    return chain


# ---------------- HMC sampler (from scratch) ----------------------------

def run_hmc(n_iter=1500, eps=1.0, L=25, theta0=None, seed=2):
    """HMC with a diagonal mass matrix M = diag(MASS) so that momentum and
    step sizes are sensible for both alpha (~0.01 scale) and rho (~1 scale)."""
    rng = np.random.default_rng(seed)
    if theta0 is None:
        theta0 = np.array([0.025, -0.2])
    theta = theta0.copy()
    chain = np.zeros((n_iter, 2))
    n_accept = 0
    for m in range(n_iter):
        p0 = rng.standard_normal(2) * np.sqrt(MASS)
        w, p = theta.copy(), p0.copy()
        grad = grad_log_posterior(w)
        for _ in range(L):
            p = p + 0.5 * eps * grad
            w = w + eps * PRECOND * p
            grad = grad_log_posterior(w)
            p = p + 0.5 * eps * grad
        lp_theta = log_posterior(theta)
        lp_w = log_posterior(w)
        H0 = -lp_theta + 0.5 * np.sum(p0 ** 2 * PRECOND)
        H1 = -lp_w + 0.5 * np.sum(p ** 2 * PRECOND)
        if np.log(rng.uniform()) < (H0 - H1):
            theta = w
            n_accept += 1
        chain[m] = theta
    print(f"HMC acceptance rate: {n_accept / n_iter:.3f}")
    return chain


def make_sampler_figure(chain, name, fname, burn=None):
    if burn is None:
        burn = len(chain) // 5
    post = chain[burn:]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))

    axes[0, 0].plot(chain[:, 0], color="steelblue", lw=0.6)
    axes[0, 0].axvline(burn, color="gray", ls="--", lw=1)
    axes[0, 0].set_title(fr"{name} trace: $\alpha$")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel(r"$\alpha$")

    axes[0, 1].hist(post[:, 0], bins=40, color="steelblue", alpha=0.85)
    axes[0, 1].axvline(TRUE_ALPHA, color="crimson", lw=2, label=f"true $\\alpha={TRUE_ALPHA}$")
    axes[0, 1].axvline(post[:, 0].mean(), color="black", ls="--", lw=1.5,
                        label=f"post. mean={post[:, 0].mean():.4f}")
    axes[0, 1].set_title(fr"{name} posterior: $\alpha$")
    axes[0, 1].legend(fontsize=7.5)

    axes[1, 0].plot(chain[:, 1], color="darkorange", lw=0.6)
    axes[1, 0].axvline(burn, color="gray", ls="--", lw=1)
    axes[1, 0].set_title(fr"{name} trace: $\rho$")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel(r"$\rho$")

    axes[1, 1].hist(post[:, 1], bins=40, color="darkorange", alpha=0.85)
    axes[1, 1].axvline(TRUE_RHO, color="crimson", lw=2, label=f"true $\\rho={TRUE_RHO}$")
    axes[1, 1].axvline(post[:, 1].mean(), color="black", ls="--", lw=1.5,
                        label=f"post. mean={post[:, 1].mean():.4f}")
    axes[1, 1].set_title(fr"{name} posterior: $\rho$")
    axes[1, 1].legend(fontsize=7.5)

    fig.suptitle(f"Toy example: {name} applied to synthetic SABR-like data", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")
    return post


def autocorr(x, max_lag=60):
    x = x - np.mean(x)
    denom = np.sum(x ** 2)
    if denom == 0:
        denom = 1.0
    result = np.array([1.0 if lag == 0 else
                        np.sum(x[:-lag] * x[lag:]) / denom
                        for lag in range(max_lag)])
    return result


def make_autocorr_figure(mala_chain, hmc_chain):
    burn_m = len(mala_chain) // 5
    burn_h = len(hmc_chain) // 5
    ac_mala = autocorr(mala_chain[burn_m:, 0])
    ac_hmc = autocorr(hmc_chain[burn_h:, 0])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(ac_mala, "o-", ms=3, color="steelblue", label="MALA")
    ax.plot(ac_hmc, "o-", ms=3, color="darkorange", label=f"HMC ($L=25$)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel(r"autocorrelation of $\alpha$ chain")
    ax.set_title("Toy example: HMC mixes faster than MALA\n(autocorrelation decays quicker)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig_mala_vs_hmc_autocorr.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_mala_vs_hmc_autocorr.pdf")


if __name__ == "__main__":
    make_leapfrog_schematic()
    make_toy_data_figure()

    mala_chain = run_mala()
    hmc_chain = run_hmc()

    mala_post = make_sampler_figure(mala_chain, "MALA", "fig_mala_toy.pdf")
    hmc_post = make_sampler_figure(hmc_chain, "HMC", "fig_hmc_toy.pdf")

    make_autocorr_figure(mala_chain, hmc_chain)

    print("\n--- Toy posterior summary (illustrative, NOT the book's numbers) ---")
    print(f"True parameters:   alpha={TRUE_ALPHA}, rho={TRUE_RHO}")
    print(f"MALA posterior:    alpha={mala_post[:,0].mean():.4f} (+/-{mala_post[:,0].std():.4f}), "
          f"rho={mala_post[:,1].mean():.4f} (+/-{mala_post[:,1].std():.4f})")
    print(f"HMC  posterior:    alpha={hmc_post[:,0].mean():.4f} (+/-{hmc_post[:,0].std():.4f}), "
          f"rho={hmc_post[:,1].mean():.4f} (+/-{hmc_post[:,1].std():.4f})")
