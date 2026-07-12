#!/usr/bin/env python3
"""
gen_figures.py
Generates all figures for Chapter 3 (Efficiency: E-values under the
alternative hypothesis) slides, based on:
  Ramdas & Wang, "Hypothesis Testing with E-Values" (arXiv:2410.23614), Ch. 3.

Figures produced (all saved as vector PDF in this directory):
  fig_gaussian_epower.pdf   -- e-power E^Q[log E_theta] vs assumed parameter
                               theta, for several true mean shifts mu
                               (Example 3.13 / 3.21 / Theorem 3.20).
  fig_growth_paths.pdf      -- growth of log(product of e-values) over
                               repeated coin flips: log-optimal vs a
                               suboptimal (naive) e-variable.
  fig_coin_epower_bar.pdf   -- bar chart: per-flip e-power (= KL divergence
                               for the optimal choice) of the log-optimal
                               e-variable vs a naive e-variable in the coin
                               example (theta_0 = 0.5, theta_1 = 0.7).

Run: python3 gen_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20240614)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Figure 1: e-power of the Gaussian LR e-variable as a function of the
# assumed parameter theta, for several true mean shifts mu.
#
# Setting (Example 3.13/3.21): P = N(0,1), Q = N(mu,1), n = 1 observation.
# E(theta) = exp(theta X - theta^2/2) is an e-variable for P for every theta.
# Its e-power under Q is  E^Q[log E(theta)] = theta*mu - theta^2/2.
# This is maximized at theta = mu (Theorem 3.20, log-optimality), with the
# maximal value mu^2/2 = KL(N(mu,1) || N(0,1)).
# ---------------------------------------------------------------------------
theta = np.linspace(-0.5, 3.0, 400)

fig, ax = plt.subplots(figsize=(7.0, 4.6))

mus = [0.8, 1.5, 2.2]
colors = ["#3b6ea5", "#c1666b", "#5a8f5a"]
for mu, c in zip(mus, colors):
    epower = theta * mu - theta**2 / 2
    ax.plot(theta, epower, color=c, lw=2.2, label=rf"true shift $\mu={mu}$")
    # mark the optimum theta = mu
    opt_val = mu**2 / 2
    ax.plot([mu], [opt_val], "o", color=c, ms=7, zorder=5)

# oracle envelope: the KL curve mu^2/2, traced as theta = mu (i.e. the
# log-optimal choice), shown as the upper envelope of all curves
mu_env = np.linspace(0, 3.0, 200)
ax.plot(mu_env, mu_env**2 / 2, "k--", lw=1.6,
        label=r"envelope: $\mathrm{KL}(\mathbb{Q}_\mu\Vert\mathbb{P})=\mu^2/2$"
              r" (attained at $\theta=\mu$)")

ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel(r"assumed parameter $\theta$ used to build $E(\theta)=\exp(\theta X-\theta^2/2)$")
ax.set_ylabel(r"e-power $\mathbb{E}^{\mathbb{Q}_\mu}[\log E(\theta)] = \theta\mu-\theta^2/2$")
ax.set_title("E-power of the Gaussian likelihood-ratio e-variable")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_gaussian_epower.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: growth of the log-e-process for repeated coin flips.
#
# Running example: H0: theta0 = 0.5 (fair coin) vs H1: theta1 = 0.7.
# Log-optimal e-variable at flip i:  E_i = (0.7/0.5)^{X_i} (0.3/0.5)^{1-X_i}
# "Naive" e-variable, built around a too-conservative guess theta=0.6:
#         E_i' = (0.6/0.5)^{X_i} (0.4/0.5)^{1-X_i}
# Data X_i are truly iid Bernoulli(0.7) (the alternative is correct).
# We plot the running sum of log E_i (i.e. log of the product M_t), whose
# slope is the e-power = KL divergence for the true rate under each choice.
# ---------------------------------------------------------------------------
theta0 = 0.5
theta1_true = 0.7
theta_naive = 0.6
n_steps = 400
n_paths = 2000

X = rng.binomial(1, theta1_true, size=(n_paths, n_steps))  # truly Bernoulli(0.7)

logE_opt = X * np.log(theta1_true / theta0) + (1 - X) * np.log((1 - theta1_true) / (1 - theta0))
logE_naive = X * np.log(theta_naive / theta0) + (1 - X) * np.log((1 - theta_naive) / (1 - theta0))

cum_opt = np.cumsum(logE_opt, axis=1)
cum_naive = np.cumsum(logE_naive, axis=1)

mean_opt = cum_opt.mean(axis=0)
mean_naive = cum_naive.mean(axis=0)

# theoretical slopes = KL divergences (per-flip e-power)
kl_opt = theta1_true * np.log(theta1_true / theta0) + (1 - theta1_true) * np.log((1 - theta1_true) / (1 - theta0))
kl_naive = theta1_true * np.log(theta_naive / theta0) + (1 - theta1_true) * np.log((1 - theta_naive) / (1 - theta0))

t = np.arange(1, n_steps + 1)

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# a handful of individual sample paths, faint
for k in range(25):
    ax.plot(t, cum_opt[k], color="#3b6ea5", lw=0.5, alpha=0.15)
    ax.plot(t, cum_naive[k], color="#c1666b", lw=0.5, alpha=0.15)

ax.plot(t, mean_opt, color="#3b6ea5", lw=2.5,
        label=rf"log-optimal $E(\theta=0.7)$: slope $\approx {kl_opt:.4f}$")
ax.plot(t, mean_naive, color="#c1666b", lw=2.5,
        label=rf"naive $E(\theta=0.6)$: slope $\approx {kl_naive:.4f}$")
ax.plot(t, kl_opt * t, color="#3b6ea5", ls="--", lw=1.2)
ax.plot(t, kl_naive * t, color="#c1666b", ls="--", lw=1.2)

ax.set_xlabel(r"number of coin flips $t$")
ax.set_ylabel(r"$\log M_t = \sum_{s=1}^t \log E_s$ (averaged over 2000 runs)")
ax.set_title(r"Growth rate under $\mathbb{Q}=\mathrm{Bernoulli}(0.7)$: log-optimal vs.\ naive e-variable")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_growth_paths.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: bar chart summarizing the per-flip e-power (numerically) for the
# coin example, comparing the log-optimal e-variable (= KL(Bernoulli(0.7) ||
# Bernoulli(0.5))) against the naive e-variable and the "wrong-direction"
# choice theta = 0.5 (i.e. the constant e-variable E=1, e-power = 0).
# ---------------------------------------------------------------------------
labels = [r"naive" "\n" r"$\theta=0.6$",
          r"log-optimal" "\n" r"$\theta=0.7$ (LR)",
          r"overshoot" "\n" r"$\theta=0.9$"]

theta_choices = [0.6, 0.7, 0.9]
epowers = []
for th in theta_choices:
    kl = theta1_true * np.log(th / theta0) + (1 - theta1_true) * np.log((1 - th) / (1 - theta0))
    epowers.append(kl)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
bar_colors = ["#c1666b", "#3b6ea5", "#c9a13b"]
bars = ax.bar(labels, epowers, color=bar_colors, width=0.55)
for b, v in zip(bars, epowers):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
            ha="center", va="bottom", fontsize=11)

ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel(r"e-power per flip $\mathbb{E}^{\mathbb{Q}}[\log E(\theta)]$ (nats)")
ax.set_title(r"Coin example: $H_0{:}\ \theta=0.5$ vs.\ true $\theta=0.7$")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("fig_coin_epower_bar.pdf")
plt.close(fig)

print("KL(Bernoulli(0.7) || Bernoulli(0.5)) =", kl_opt, "nats")
print("naive e-power (theta=0.6)            =", kl_naive, "nats")
print("Saved: fig_gaussian_epower.pdf, fig_growth_paths.pdf, fig_coin_epower_bar.pdf")
