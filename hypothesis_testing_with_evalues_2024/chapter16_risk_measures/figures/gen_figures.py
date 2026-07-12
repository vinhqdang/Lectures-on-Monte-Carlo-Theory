"""
Generate figures for Chapter 16: E-Values and Risk Measures.

Figure 1 (fig1_var_es.pdf): A loss density with VaR_95% and ES_95% marked,
    with the 5% tail region shaded, illustrating the plain-language
    definitions of Value-at-Risk and Expected Shortfall.

Figure 2 (fig2_eprocess.pdf): The e-process (running product) M_t built from
    the monotone backtest e-statistic for a quantile (Example 16.9 in the
    book), applied to the 10-day running numerical example used throughout
    the slides, testing H0: VaR_0.95(F) <= 3.

Both figures are saved as vector PDFs in this directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

# ---------------------------------------------------------------------------
# Figure 1: VaR and ES on a loss density
# ---------------------------------------------------------------------------

beta = 0.95
mu, sigma = 0.0, 1.0
dist = stats.norm(mu, sigma)

var_95 = dist.ppf(beta)                       # Value-at-Risk at level beta
# Expected Shortfall for a Normal loss: ES_beta = mu + sigma * phi(z_beta) / (1-beta)
z_beta = dist.ppf(beta)
es_95 = mu + sigma * stats.norm.pdf(z_beta) / (1 - beta)

x = np.linspace(mu - 4 * sigma, mu + 4.5 * sigma, 2000)
pdf = dist.pdf(x)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(x, pdf, color="black", lw=1.8, label="loss density $f(x)$")

# Shade the tail region x >= VaR_95 (the worst 5% of outcomes)
tail_x = x[x >= var_95]
tail_pdf = pdf[x >= var_95]
ax.fill_between(tail_x, tail_pdf, color="firebrick", alpha=0.35,
                label=r"worst $5\%$ of outcomes ($X \geq \mathrm{VaR}_{0.95}$)")

# VaR line
ax.axvline(var_95, color="firebrick", lw=2, linestyle="--")
ax.annotate(rf"$\mathrm{{VaR}}_{{0.95}} = {var_95:.2f}$",
            xy=(var_95, dist.pdf(var_95)), xytext=(var_95 + 0.15, 0.30),
            fontsize=11, color="firebrick",
            arrowprops=dict(arrowstyle="->", color="firebrick"))

# ES line (mean of the shaded tail)
ax.axvline(es_95, color="navy", lw=2, linestyle=":")
ax.annotate(rf"$\mathrm{{ES}}_{{0.95}} = {es_95:.2f}$" + "\n(average loss in tail)",
            xy=(es_95, 0.02), xytext=(es_95 + 0.25, 0.14),
            fontsize=11, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy"))

ax.set_xlabel("loss $X$")
ax.set_ylabel("density")
ax.set_title(r"Value-at-Risk and Expected Shortfall at $\beta = 0.95$")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_ylim(0, 0.45)
fig.tight_layout()
fig.savefig("fig1_var_es.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: e-process for backtesting a quantile (Example 16.9 / Eq. 16.15)
# ---------------------------------------------------------------------------

# The 10-day running example used throughout the slides (losses, $ thousands)
losses = np.array([1.2, -0.8, 2.5, 0.3, -1.5, 4.1, 0.9, -0.2, 3.6, 7.8])

beta_test = 0.95
r = 3.0          # candidate threshold: H0: VaR_0.95(F) <= r
lam = 0.5        # fixed betting fraction lambda_t = 0.5

# Monotone backtest e-statistic for the beta-quantile (Example 16.9):
#   e(x, r) = 1/(1-beta) * 1{x > r}
E_t = (1.0 / (1 - beta_test)) * (losses > r).astype(float)

# e-process via testing by betting (Eq. 16.15): M_t = prod_{s<=t} ((1-lam)+lam*E_s)
factors = (1 - lam) + lam * E_t
M_t = np.cumprod(factors)

days = np.arange(1, len(losses) + 1)

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.4]})

# Top panel: the raw daily losses with the threshold r marked
colors = ["firebrick" if x > r else "steelblue" for x in losses]
ax1.bar(days, losses, color=colors, width=0.6)
ax1.axhline(r, color="black", lw=1.2, linestyle="--", label=f"candidate threshold $r={r:.0f}$")
ax1.set_ylabel("daily loss $X_t$")
ax1.legend(loc="upper left", fontsize=9, frameon=False)
ax1.set_title(r"Backtesting $H_0: \mathrm{VaR}_{0.95}(F) \leq r$ with a monotone backtest e-statistic")

# Bottom panel: the e-process M_t
ax2.plot(days, M_t, marker="o", color="darkgreen", lw=1.8)
ax2.axhline(1.0, color="gray", lw=1.2, linestyle=":", label=r"$M_t=1$ (no evidence)")
ax2.set_xlabel("day $t$")
ax2.set_ylabel(r"e-process $M_t$")
ax2.set_xticks(days)
ax2.legend(loc="upper left", fontsize=9, frameon=False)
for xt, yt in zip(days, M_t):
    ax2.annotate(f"{yt:.2f}", (xt, yt), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=7.5)

fig2.tight_layout()
fig2.savefig("fig2_eprocess.pdf")
plt.close(fig2)

print("VaR_0.95 (Normal example):", var_95)
print("ES_0.95  (Normal example):", es_95)
print("Sorted losses:", np.sort(losses))
print("E_t:", E_t)
print("M_t:", M_t)
print("Saved fig1_var_es.pdf and fig2_eprocess.pdf")
