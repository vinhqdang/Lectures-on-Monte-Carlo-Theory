"""
Generate all figures for Chapter 2.2 (Measure Theory and Probability Theory) slides.
Run with: conda run -n py313 python3 gen_figures.py
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures/"

plt.rcParams.update({
    "font.size": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ---------------------------------------------------------------------------
# Figure 2.3 style: pdf p_X and cdf F_X of the uniform measure on [0,1]
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
x = np.linspace(-1, 2, 2000)
pX = np.where((x >= 0) & (x <= 1), 1.0, 0.0)
FX = np.clip(x, 0, 1)
FX = np.where(x < 0, 0.0, FX)
FX = np.where(x > 1, 1.0, FX)
ax.plot(x, pX, color="black", lw=2, label=r"$p_X$")
ax.plot(x, FX, color="gray", lw=2, ls="--", label=r"$F_X$")
ax.set_xlabel("$x$")
ax.set_ylabel("value of distribution")
ax.set_ylim(-0.1, 1.6)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT + "fig_uniform_pdf_cdf.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2.4 style: several Gaussians / mixture with the same mean, increasing variance
# ---------------------------------------------------------------------------
def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

fig, ax = plt.subplots(figsize=(6.6, 4.4))
x = np.linspace(-5, 5, 2000)
ax.plot(x, gauss(x, 0, 0.5), color="black", lw=2, label=r"$\mathcal{N}(0,(1/2)^2)$")
ax.plot(x, gauss(x, 0, 1.0), color="gray", lw=1.8, ls="--", label=r"$\mathcal{N}(0,1)$")
ax.plot(x, gauss(x, 0, 2.0), color="black", lw=1.5, ls=":", label=r"$\mathcal{N}(0,2^2)$")
mix = 0.5 * gauss(x, -3, 2) + 0.5 * gauss(x, 3, 2)
ax.plot(x, mix, color="dimgray", lw=1.5, ls="-.", label=r"$\mathcal{N}(-3,2^2){+}\mathcal{N}(3,2^2)$")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "fig_gaussian_variance.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2.5 style: secant line sits above a convex function f(x) = (x-1)^2
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.6))
x = np.linspace(-1, 3, 400)
f = (x - 1) ** 2
ax.plot(x, f, color="gray", lw=2)
P = (0, 1)
Q = (2.5, 2.25)
ax.plot([P[0], Q[0]], [P[1], Q[1]], color="black", lw=1.8, ls="--", label=r"$(x-1)^2$" + "\n" + r"secant")
ax.scatter(*P, color="black", zorder=5)
ax.scatter(*Q, color="black", zorder=5)
ax.annotate("$P$", P, textcoords="offset points", xytext=(-14, 4))
ax.annotate("$Q$", Q, textcoords="offset points", xytext=(6, 2))
ax.set_xlim(-1, 3)
ax.set_ylim(0, 3)
fig.tight_layout()
fig.savefig(OUT + "fig_convex_secant.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2.6 style: convex function f(x)=(x-1)^2 with tangent lines below it
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.6))
x = np.linspace(-1, 3, 400)
f = (x - 1) ** 2
ax.plot(x, f, color="black", lw=2.2)
for x0 in [0, 1, 2]:
    slope = 2 * (x0 - 1)
    y0 = (x0 - 1) ** 2
    tangent = y0 + slope * (x - x0)
    ax.plot(x, tangent, color="gray", lw=1.2)
    ax.scatter([x0], [y0], color="black", zorder=5)
ax.set_xlim(-1, 3)
ax.set_ylim(0, 3)
fig.tight_layout()
fig.savefig(OUT + "fig_convex_tangent.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2.7 style: P(|S_n/n - theta| >= eps) exactly, via the Binomial pmf
# ---------------------------------------------------------------------------
theta = 0.4
eps = 0.1
ns = np.arange(1, 201)
probs = []
for n in ns:
    p = 0.0
    for k in range(0, n + 1):
        if abs(k / n - theta) >= eps:
            p += math.comb(n, k) * (theta ** k) * ((1 - theta) ** (n - k))
    probs.append(p)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.fill_between(ns, probs, color="gray", alpha=0.3)
ax.plot(ns, probs, color="black", lw=1.3)
ax.set_xlabel("$n$")
ax.set_ylabel(r"$\mathrm{P}(|\frac{1}{n}S_n-\theta|\geq\varepsilon)$")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT + "fig_convergence_prob.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Extra (not in book, supports Python worked example): Markov vs Chebyshev vs
# exact tail probability for the sum of n fair dice.
# ---------------------------------------------------------------------------
n_dice = 20
mu = 3.5 * n_dice
var = n_dice * (35.0 / 12.0)  # Var of one die = E[X^2]-E[X]^2 = 91/6 - 12.25 = 35/12
ks = np.arange(int(mu), 6 * n_dice + 1)

# exact P(S >= k) via dynamic-programming convolution of one die's pmf
pmf1 = np.array([0] + [1 / 6.0] * 6)  # index 1..6
dist = np.array([1.0])
for _ in range(n_dice):
    newdist = np.zeros(len(dist) + 6)
    for face in range(1, 7):
        newdist[face:face + len(dist)] += dist * (1 / 6.0)
    dist = newdist
# dist[i] = P(sum of n_dice dice == i); indices below n_dice are exactly 0.
exact_tail = []
for k in ks:
    idx = int(k)
    exact_tail.append(dist[idx:].sum() if idx < len(dist) else 0.0)

markov_bound = np.minimum(1.0, mu / ks)
cheb_bound = np.minimum(1.0, var / (ks - mu) ** 2) if True else None
cheb_bound = np.array([min(1.0, var / max(k - mu, 1e-9) ** 2) for k in ks])

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(ks, exact_tail, color="black", lw=2, label="exact $P(S \\geq k)$")
ax.plot(ks, markov_bound, color="gray", lw=1.6, ls="--", label="Markov bound")
ax.plot(ks, cheb_bound, color="dimgray", lw=1.6, ls=":", label="Chebyshev bound (one-sided x2)")
ax.set_xlabel("$k$")
ax.set_ylabel("probability / bound")
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "fig_dice_bounds.pdf")
plt.close(fig)

print("All figures written to", OUT)
