"""
gen_figures.py

Generates ILLUSTRATIVE TOY figures for Chapter 5 slides:
  "Analyzing South African Equity Option Prices Using Normalizing Flows"
  (Mongwe, Mbuvha & Marwala, 2025, Chapter 5)

These are self-contained toy simulations built purely for teaching intuition.
They are NOT the JSE All Share Index market data used in the book, and the
numbers here should never be confused with the book's Tables 5.1-5.5 or
Figures 5.1-5.5.

Outputs (PDF, vector):
  figures/option_price_and_rnd.pdf   -- toy option prices vs strike, and the
                                         numerically-differentiated
                                         risk-neutral density recovered
                                         from them (Sec. 5.2 illustration)
  figures/normalizing_flow_toy.pdf   -- a toy 1D normalizing flow, built from
                                         scratch in numpy, mapping a standard
                                         Gaussian base density into a
                                         skewed / fat-tailed target density
                                         (Sec. 5.3 illustration)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# -----------------------------------------------------------------------
# Common toy market parameters
# -----------------------------------------------------------------------
r = 0.05        # toy continuously-compounded rate
tau = 0.25      # toy time to expiry (in years)
S0 = 100.0      # toy spot
F = S0 * np.exp(r * tau)   # toy forward price (illustrative)

# -----------------------------------------------------------------------
# FIGURE (a): toy option prices -> numerically differentiated risk-neutral
# density (Breeden-Litzenberger-style second-derivative recovery)
# -----------------------------------------------------------------------
# We DEFINE a "true" toy risk-neutral density on S_T as a mixture of two
# lognormals (to create mild skew/kurtosis, purely illustrative), then
# PRICE toy call options against it by numerical integration, then RECOVER
# the density purely from the option prices via finite differences -- this
# is the illustration of Sec. 5.2's logic ("you can back the density out
# purely from how option prices change with strike").

def lognormal_pdf(s, mu, sigma):
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    mask = s > 0
    out[mask] = (1.0 / (s[mask] * sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-(np.log(s[mask]) - mu) ** 2 / (2 * sigma ** 2))
    return out

# Mixture parameters chosen so that E[S_T] approx F (risk-neutral condition)
w1, w2 = 0.7, 0.3
sigma1, sigma2 = 0.15, 0.35
mu1 = np.log(F) - 0.5 * sigma1 ** 2
mu2 = np.log(F * 0.92) - 0.5 * sigma2 ** 2  # shifted, fatter component -> left skew/fat tail

def true_density(s):
    return w1 * lognormal_pdf(s, mu1, sigma1) + w2 * lognormal_pdf(s, mu2, sigma2)

# Numerical integration grid for S_T (fine grid for accurate pricing)
S_grid = np.linspace(1.0, 260.0, 20000)
dS = S_grid[1] - S_grid[0]
q_true_vals = true_density(S_grid)
q_true_vals /= np.trapezoid(q_true_vals, S_grid)  # renormalize for numerical safety

def call_price(K):
    payoff = np.maximum(S_grid - K, 0.0)
    integrand = payoff * q_true_vals
    price = np.exp(-r * tau) * np.trapezoid(integrand, S_grid)
    return price

# Strike grid for plotting the price curve and recovering the density
K_grid = np.linspace(60.0, 150.0, 181)
h = K_grid[1] - K_grid[0]
C_vals = np.array([call_price(K) for K in K_grid])

# Central finite-difference second derivative -> Breeden-Litzenberger recovery:
#   q(K) = exp(r*tau) * d^2 C / d K^2
q_implied = np.full_like(C_vals, np.nan)
q_implied[1:-1] = np.exp(r * tau) * (C_vals[2:] - 2 * C_vals[1:-1] + C_vals[:-2]) / h ** 2

# A small "by-hand" toy example table (5 strikes), independent, clean numbers
K_hand = np.array([95.0, 100.0, 105.0])
C_hand = np.array([call_price(k) for k in K_hand])
print("Toy 'by-hand' example strikes and prices (for slide table):")
for k, c in zip(K_hand, C_hand):
    print(f"  K={k:.1f}  C(K)={c:.4f}")
fd2_hand = np.exp(r * tau) * (C_hand[2] - 2 * C_hand[1] + C_hand[0]) / (5.0 ** 2)
print(f"  finite-difference second derivative estimate of q(100): {fd2_hand:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(K_grid, C_vals, color='#1b5e91', lw=2)
ax.scatter(K_hand, C_hand, color='#d6604d', zorder=5, label='toy by-hand strikes')
ax.set_xlabel('Strike $K$')
ax.set_ylabel('Toy call price $f_w(K)$')
ax.set_title('(a) Toy option prices vs. strike')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(S_grid, q_true_vals, color='#238b45', lw=2, label='true toy density $q(S_T)$')
ax.plot(K_grid, q_implied, 'o', ms=3, color='#d6604d',
        label='recovered via finite differences')
ax.set_xlim(50, 170)
ax.set_xlabel('$S_T$ / Strike $K$')
ax.set_ylabel('density')
ax.set_title('(b) Risk-neutral density: true vs. recovered')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle('Illustrative TOY simulation (author-generated, not JSE market data)',
             fontsize=9, style='italic', color='gray')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('option_price_and_rnd.pdf')
plt.close(fig)

# -----------------------------------------------------------------------
# FIGURE (b): a toy 1D normalizing flow built from scratch in numpy.
# Base: standard Gaussian z ~ N(0,1).
# Layer 1 (affine):        u = a*z + b
# Layer 2 (sinh-arcsinh):  x = sinh( (asinh(u) + eps) / delta )
#
# Both layers are monotonically increasing and hence invertible, so the
# composed map T(z) = x is invertible and the change-of-variables formula
#     p_X(x) = p_Z(z) * |dz/dx| = p_Z(z) / |dx/dz|
# applies, with dx/dz obtained by the chain rule through the two layers.
# eps != 0 introduces skew, delta < 1 introduces heavier-than-Gaussian tails.
# -----------------------------------------------------------------------

a, b = 1.0, 0.0          # affine layer parameters
eps, delta = 1.0, 0.6    # sinh-arcsinh layer parameters (skew, tail weight)

def flow_forward(z):
    u = a * z + b
    x = np.sinh((np.arcsinh(u) + eps) / delta)
    return x

def flow_forward_with_jacobian(z):
    # Layer 1: affine
    u = a * z + b
    du_dz = a

    # Layer 2: sinh-arcsinh
    x = np.sinh((np.arcsinh(u) + eps) / delta)
    dx_du = np.cosh((np.arcsinh(u) + eps) / delta) / (delta * np.sqrt(1.0 + u ** 2))

    dx_dz = dx_du * du_dz
    return x, dx_dz

def standard_normal_pdf(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)

# Parametric evaluation of the pushforward density on a grid of z
z_grid = np.linspace(-4.5, 4.5, 4000)
x_of_z, dxdz = flow_forward_with_jacobian(z_grid)
p_z = standard_normal_pdf(z_grid)
p_x = p_z / np.abs(dxdz)   # change-of-variables formula, applied at each z

# Sample-based verification: draw z samples, push forward, histogram
N = 200_000
z_samples = rng.standard_normal(N)
x_samples = flow_forward(z_samples)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(z_grid, p_z, color='#1b5e91', lw=2)
ax.fill_between(z_grid, p_z, alpha=0.15, color='#1b5e91')
ax.set_xlabel('$z$')
ax.set_ylabel('$p_Z(z)$')
ax.set_title('(a) Base density: standard Gaussian')
ax.grid(alpha=0.3)

ax = axes[1]
# Compute the histogram over the FULL sample range (so density=True
# normalizes correctly, including the probability mass in the tail beyond
# the plotting window), then only display bins inside the plotting window.
hist_counts, hist_edges = np.histogram(x_samples, bins=400, range=(-4, 60), density=True)
hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
in_view = hist_centers <= 10
ax.bar(hist_centers[in_view], hist_counts[in_view],
       width=(hist_edges[1] - hist_edges[0]),
       color='#a6bddb', alpha=0.7, label='histogram of flow samples $x=T(z)$')
order = np.argsort(x_of_z)
ax.plot(x_of_z[order], p_x[order], color='#d6604d', lw=2,
        label=r'$p_X(x)=p_Z(z)\,|dz/dx|$ (analytic)')
ax.set_xlim(-4, 10)
ax.set_xlabel('$x$')
ax.set_ylabel('$p_X(x)$')
ax.set_title('(b) Transformed density: skewed / fat-tailed')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

fig.suptitle('Illustrative TOY normalizing flow (author-generated, from scratch in numpy)',
             fontsize=9, style='italic', color='gray')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('normalizing_flow_toy.pdf')
plt.close(fig)

print("Saved figures/option_price_and_rnd.pdf and figures/normalizing_flow_toy.pdf")
