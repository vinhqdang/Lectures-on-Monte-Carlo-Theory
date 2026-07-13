#!/usr/bin/env python3
"""
gen_figures.py
==============
Self-contained figure generator for Chapter 12 (Static and Dynamic Nested
Sampling for Yield Curve Model Selection) slides.

Every figure here is an ILLUSTRATIVE / TOY simulation written from scratch by
the lecture-note author to build intuition about nested sampling.  None of
these figures are reproductions of the book's own figures (Figs. 12.1-12.14):
those show the book's actual dynesty runs on the Nelson-Siegel models and
real/simulated yield-curve data; the figures below use a tiny hand-rolled
rejection-sampling nested sampler on toy problems, purely for teaching.

Outputs (PDF, vector) are written to the current directory:
  1. ns_schematic.pdf            - shrinking prior-volume schematic + L(X) integral
  2. toy_ns_run.pdf              - from-scratch toy nested sampler on a 2D Gaussian
  3. static_vs_dynamic.pdf       - illustrative live-point allocation comparison
  4. toy_ns_model_comparison.pdf - toy "subset Nelson-Siegel" evidence comparison

Run:  python3 gen_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20250713)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
})


# =============================================================================
# FIGURE 1: Nested sampling schematic -- shrinking prior volume & L(X) integral
# =============================================================================
def fig_schematic():
    fig = plt.figure(figsize=(10, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)

    # --- Left panel: shrinking contours of live points in 2D parameter space
    ax = fig.add_subplot(gs[0])
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-0.5 * (X**2 + Y**2))  # toy unimodal Gaussian likelihood

    levels = [0.03, 0.15, 0.4, 0.7, 0.92]
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=1.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"L={v:.2f}")

    # live points: 3 generations shown shrinking around the mode
    rng_local = np.random.default_rng(1)
    for k, (radius, colour, lbl) in enumerate([
        (2.6, "0.55", "iteration 0 (K live points, wide)"),
        (1.5, "tab:orange", "iteration $\\approx$ 40 (shrunk)"),
        (0.7, "tab:red", "iteration $\\approx$ 90 (near mode)"),
    ]):
        theta = rng_local.uniform(0, 2 * np.pi, 14)
        r = radius * np.sqrt(rng_local.uniform(0.55, 1.0, 14))
        ax.scatter(r * np.cos(theta), r * np.sin(theta), s=18, color=colour,
                   zorder=5, label=lbl)

    # mark one worst point being discarded and replaced
    ax.scatter([2.3], [1.0], marker="x", s=90, color="black", zorder=6)
    ax.annotate("worst live point\n(lowest $L$) discarded", xy=(2.3, 1.0),
                xytext=(0.7, 2.55), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.scatter([1.7], [-0.6], marker="*", s=130, color="tab:blue", zorder=6)
    ax.annotate("new point sampled\nfrom prior with $L>L_{\\rm worst}$",
                xy=(1.7, -0.6), xytext=(-2.9, -2.6), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel(r"parameter $\theta_1$")
    ax.set_ylabel(r"parameter $\theta_2$")
    ax.set_title("Live points shrink onto high-likelihood region")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)

    # --- Right panel: L(X) vs X, the 1D transformed integral
    ax2 = fig.add_subplot(gs[1])
    Xgrid = np.linspace(1e-4, 1, 400)
    # toy L(X): monotone decreasing function of prior volume X (Skilling's trick)
    Lx = np.exp(-6.0 * Xgrid**0.6)
    ax2.plot(Xgrid, Lx, color="tab:blue", lw=2)
    ax2.fill_between(Xgrid, 0, Lx, color="tab:blue", alpha=0.18)

    # mark geometric shrinkage ticks X_i = t^i with t = K/(K+1), K=20
    K = 20
    t = K / (K + 1)
    Xi = t ** np.arange(0, 46, 5)
    for xi in Xi:
        Li = np.exp(-6.0 * xi**0.6)
        ax2.plot([xi, xi], [0, Li], color="0.4", lw=0.7, ls="--")
        ax2.plot([xi], [Li], "o", color="tab:red", ms=3)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"prior volume $X = \int_{L(\theta) > L^*} \pi(\theta)\,d\theta$")
    ax2.set_ylabel(r"likelihood $L(X)$")
    ax2.set_title(r"Evidence $Z=\int_0^1 L(X)\,dX$ (shaded area)")
    ax2.text(0.45, 0.75,
             r"$X$ shrinks geometrically:" "\n" r"$X_i \approx \left(\frac{K}{K+1}\right)^{i}$",
             fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    fig.suptitle("Nested sampling: turning a high-dimensional integral into a 1D integral\n"
                 "(illustrative schematic, not from the book)", fontsize=10.5, y=1.03)
    fig.savefig("ns_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Toy from-scratch nested sampler (rejection-sampling based) used by Figs 2 & 4
# =============================================================================
def toy_nested_sampling(log_likelihood, bounds, n_live=50, n_iter=600,
                        max_reject=20000, seed=0):
    """
    A bare-bones, from-scratch static nested sampling routine following
    Algorithm 12.1 exactly: sample K live points from the (uniform) prior,
    repeatedly discard the worst-likelihood point, increment the evidence,
    and replace it by rejection sampling a new point from the prior that
    satisfies the current likelihood constraint.

    This is a TOY / ILLUSTRATIVE implementation (rejection sampling from the
    *full* prior box, not from a shrinking ellipsoid as in production
    packages like dynesty/MultiNest) -- adequate for low dimensions only.

    bounds: list of (lo, hi) tuples, one per parameter dimension.
    Returns dict with dead point history, running evidence estimates, etc.
    """
    rng_l = np.random.default_rng(seed)
    ndim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    prior_vol_total = np.prod(hi - lo)

    def sample_prior(n):
        return lo + rng_l.uniform(size=(n, ndim)) * (hi - lo)

    live_theta = sample_prior(n_live)
    live_logl = np.array([log_likelihood(th) for th in live_theta])

    logZ = -np.inf
    logX_prev = 0.0  # ln(X_0) = ln(1) = 0
    dead_theta, dead_logl, dead_logX = [], [], []
    logZ_history = []
    live_count_history = []
    t = n_live / (n_live + 1.0)  # expected shrinkage factor per iteration

    for i in range(1, n_iter + 1):
        worst = np.argmin(live_logl)
        logl_worst = live_logl[worst]
        theta_worst = live_theta[worst].copy()

        logX_i = logX_prev + np.log(t)          # ln X_i = ln X_{i-1} + ln t
        # trapezoidal weight in X-space: dX_i = (X_{i-1} - X_{i+1})/2 approx by X_{i-1}-X_i
        dX = np.exp(logX_prev) - np.exp(logX_i)
        logw = np.log(max(dX, 1e-300)) + logl_worst
        logZ = np.logaddexp(logZ, logw)

        dead_theta.append(theta_worst)
        dead_logl.append(logl_worst)
        dead_logX.append(logX_i)
        logZ_history.append(logZ)
        live_count_history.append(n_live)

        # replace worst point via rejection sampling from the prior,
        # subject to the likelihood constraint L(theta_new) > L_worst
        n_reject = 0
        while True:
            cand = sample_prior(1)[0]
            cand_logl = log_likelihood(cand)
            n_reject += 1
            if cand_logl > logl_worst or n_reject > max_reject:
                break
        live_theta[worst] = cand
        live_logl[worst] = cand_logl
        logX_prev = logX_i

    # final contribution: remaining live points share the leftover volume
    logX_final = logX_prev
    remaining_X = np.exp(logX_final)
    mean_live_l = np.mean(live_logl)
    logZ_live = np.log(remaining_X + 1e-300) + mean_live_l
    logZ = np.logaddexp(logZ, logZ_live)

    return dict(
        dead_theta=np.array(dead_theta), dead_logl=np.array(dead_logl),
        dead_logX=np.array(dead_logX), logZ_history=np.array(logZ_history),
        logZ=logZ, live_theta=live_theta, live_logl=live_logl,
    )


# =============================================================================
# FIGURE 2: Toy nested-sampling run on a simple 2D Gaussian likelihood
# =============================================================================
def fig_toy_run():
    sigma = 1.0
    def loglike(theta):
        r2 = np.sum(theta**2)
        return -0.5 * r2 / sigma**2 - np.log(2 * np.pi * sigma**2)

    bounds = [(-5, 5), (-5, 5)]
    n_live = 60
    res = toy_nested_sampling(loglike, bounds, n_live=n_live, n_iter=500, seed=3)

    # analytic evidence: uniform prior density 1/(10*10)=0.01 times integral
    # of the (normalised) Gaussian likelihood over R^2, clipped to the box
    # (box is wide relative to sigma, so integral over R^2 is an excellent approx)
    prior_density = 1.0 / ((bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]))
    logZ_true = np.log(prior_density)  # integral of full 2D normalised Gaussian over R^2 is 1

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # (a) dead + live points in parameter space, coloured by discovery order
    ax = axes[0]
    order = np.arange(len(res["dead_theta"]))
    sca = ax.scatter(res["dead_theta"][:, 0], res["dead_theta"][:, 1],
                     c=order, cmap="viridis", s=10, label="dead points")
    ax.scatter(res["live_theta"][:, 0], res["live_theta"][:, 1],
              color="red", marker="*", s=70, label="final live points", zorder=5)
    plt.colorbar(sca, ax=ax, label="iteration (discovery order)")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title("(a) Toy sampler: dead \\& live points")
    ax.legend(fontsize=7, loc="upper right")

    # (b) likelihood of discarded point vs iteration (shrinking prior volume)
    ax = axes[1]
    ax.plot(-res["dead_logX"], res["dead_logl"], color="tab:blue", lw=1.6)
    ax.set_xlabel(r"$-\ln X$  (compression of prior volume)")
    ax.set_ylabel(r"$\ln L$ of discarded point")
    ax.set_title("(b) Likelihood climbs as $X$ shrinks")

    # (c) running evidence estimate converging to the true value
    ax = axes[2]
    iters = np.arange(1, len(res["logZ_history"]) + 1)
    ax.plot(iters, res["logZ_history"], color="tab:purple", lw=1.8,
           label="toy nested-sampling estimate")
    ax.axhline(logZ_true, color="black", ls="--", lw=1.3,
              label=fr"analytic $\ln Z = {logZ_true:.3f}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"running $\ln \hat Z$")
    ax.set_title("(c) Evidence estimate converges")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("From-scratch TOY nested sampler on a 2D Gaussian likelihood "
                 f"($K={n_live}$ live points) -- illustrative simulation only",
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig("toy_ns_run.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[toy_ns_run] final log Z estimate = {res['logZ']:.4f}, "
         f"analytic log Z = {logZ_true:.4f}")


# =============================================================================
# FIGURE 3: Illustrative static vs dynamic live-point allocation
# =============================================================================
def fig_static_vs_dynamic():
    # Purely schematic curves (shapes inspired by the qualitative behaviour of
    # static vs dynamic nested sampling live-point counts, NOT digitised from
    # the book's own figures) used only to build intuition.
    nlogX = np.linspace(0, 26, 400)

    # static: constant K, then falls to zero once samples run out at termination
    K_static = 500
    static_curve = np.where(nlogX < 16.5, K_static,
                            K_static * np.exp(-(nlogX - 16.5) / 1.7))

    # dynamic: starts with a small exploratory K, then ramps up sharply where
    # the importance weight (posterior mass) peaks, then decays
    base = 300
    peak_centre, peak_width, peak_height = 16.5, 2.4, 2600
    bump = peak_height * np.exp(-0.5 * ((nlogX - peak_centre) / peak_width) ** 2)
    dynamic_curve = base + bump
    dynamic_curve = np.where(nlogX > peak_centre,
                             base + bump * np.exp(-(nlogX - peak_centre) / 3.0),
                             dynamic_curve)

    importance = np.exp(-0.5 * ((nlogX - peak_centre) / 1.6) ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(nlogX, static_curve, color="tab:blue", lw=2, label="static NS ($K$ constant)")
    axes[0].plot(nlogX, dynamic_curve, color="tab:red", lw=2, label="dynamic NS (adaptive $K$)")
    axes[0].set_ylabel("number of live points")
    axes[0].set_title("Illustrative comparison: live-point allocation "
                      "(schematic, not digitised from the book)")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(nlogX, importance, color="0.3", lw=1.8)
    axes[1].fill_between(nlogX, 0, importance, color="0.7", alpha=0.5)
    axes[1].set_ylabel("importance\nweight (a.u.)")
    axes[1].set_xlabel(r"$-\ln X$ (progress of the run)")

    fig.tight_layout()
    fig.savefig("static_vs_dynamic.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE 4: Toy "subset Nelson-Siegel" evidence comparison (running example)
# =============================================================================
def nelson_siegel_loadings(tau, lam):
    x1 = (1 - np.exp(-lam * tau)) / (lam * tau)
    x2 = x1 - np.exp(-lam * tau)
    return x1, x2


def fig_toy_model_comparison():
    # --- synthetic yield curve, generated from the FULL (level+slope+curvature)
    # Nelson-Siegel model with known parameters -- purely a toy dataset built
    # for this lecture, distinct from the book's own simulated/real data.
    tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    lam_true = 0.7
    beta0_true, beta1_true, beta2_true = 0.035, -0.022, 0.018
    x1, x2 = nelson_siegel_loadings(tau, lam_true)
    y_true = beta0_true + beta1_true * x1 + beta2_true * x2
    noise_sigma = 0.0006
    y_obs = y_true + rng.normal(0, noise_sigma, size=tau.shape)

    # Model A: "level + slope" toy subset (beta2 fixed at 0), lambda fixed
    # Model B: "level + slope + curvature" toy subset (full model), lambda fixed
    def make_loglike(ndim):
        def loglike(theta):
            if ndim == 2:
                b0, b1 = theta
                pred = b0 + b1 * x1
            else:
                b0, b1, b2 = theta
                pred = b0 + b1 * x1 + b2 * x2
            resid = y_obs - pred
            n = len(y_obs)
            return -0.5 * np.sum(resid**2) / noise_sigma**2 \
                  - n * np.log(np.sqrt(2 * np.pi) * noise_sigma)
        return loglike

    bounds_A = [(-0.1, 0.1), (-0.1, 0.1)]
    bounds_B = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]

    res_A = toy_nested_sampling(make_loglike(2), bounds_A, n_live=80, n_iter=400, seed=11)
    res_B = toy_nested_sampling(make_loglike(3), bounds_B, n_live=80, n_iter=500, seed=12)

    logZ_A, logZ_B = res_A["logZ"], res_B["logZ"]
    logBF = logZ_B - logZ_A

    # best-fit curves (posterior-weighted mean of live points, crude toy estimate)
    def weighted_mean(res):
        w = np.exp(res["dead_logl"] - res["dead_logl"].max())
        w /= w.sum()
        return np.average(res["dead_theta"], axis=0, weights=w)

    theta_A = weighted_mean(res_A)
    theta_B = weighted_mean(res_B)
    pred_A = theta_A[0] + theta_A[1] * x1
    pred_B = theta_B[0] + theta_B[1] * x1 + theta_B[2] * x2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.scatter(tau, y_obs, color="black", marker="x", s=40, label="toy synthetic data", zorder=5)
    ax.plot(tau, pred_A, color="tab:blue", lw=2, ls="--",
           label=f"Model A: level+slope (best fit)")
    ax.plot(tau, pred_B, color="tab:red", lw=2,
           label=f"Model B: level+slope+curvature (best fit)")
    ax.set_xlabel(r"maturity $\tau$ (years)")
    ax.set_ylabel(r"yield $y_t(\tau)$")
    ax.set_title("(a) Toy synthetic yield curve \\& model fits")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    bars = ax.bar(["Model A\n(level+slope)", "Model B\n(level+slope+curv.)"],
                 [logZ_A, logZ_B], color=["tab:blue", "tab:red"])
    for b, v in zip(bars, [logZ_A, logZ_B]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
               ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.set_ylabel(r"toy log-evidence $\ln \hat Z$")
    ax.set_title(f"(b) Toy Bayesian evidence comparison\n"
                f"$\\ln BF_{{B,A}} = {logBF:.2f}$ (favours "
                f"{'B' if logBF>0 else 'A'})")

    fig.tight_layout()
    fig.savefig("toy_ns_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[toy_ns_model_comparison] logZ_A={logZ_A:.4f}  logZ_B={logZ_B:.4f}  "
         f"logBF(B vs A)={logBF:.4f}")
    print(f"[toy_ns_model_comparison] theta_A (b0,b1) = {theta_A}")
    print(f"[toy_ns_model_comparison] theta_B (b0,b1,b2) = {theta_B}")
    print(f"[toy_ns_model_comparison] true (b0,b1,b2) = "
         f"{beta0_true, beta1_true, beta2_true}")


if __name__ == "__main__":
    fig_schematic()
    fig_toy_run()
    fig_static_vs_dynamic()
    fig_toy_model_comparison()
    print("All figures written to:", __file__.rsplit("/", 1)[0])
