"""
Generate figures for Chapter 10 (Bayesian Neural Network Inference of Motor
Insurance Claims) slides on:
Bayesian Machine Learning in Quantitative Finance (Mongwe, Mbuvha & Marwala, 2025)

Figures produced (all saved as vector PDF):
  1. fig_bnn_weight_uncertainty.pdf -- A small feed-forward neural network
                                       diagram in which every connection
                                       weight is drawn as a small Gaussian
                                       "blob" / error bar instead of a single
                                       fixed number, illustrating the core
                                       idea of a Bayesian neural network:
                                       weights are random variables with a
                                       posterior distribution, not fixed
                                       numbers.
  2. fig_toy_laplace_1d.pdf         -- OUR OWN illustrative toy simulation:
                                       a tiny 1-hidden-layer (2 hidden unit)
                                       tanh network is fit by gradient descent
                                       to 6 synthetic "driver age -> claim
                                       amount"-like points (a MAP / penalised
                                       least-squares fit).  The Hessian of the
                                       negative log-posterior is approximated
                                       numerically at the MAP estimate, the
                                       posterior covariance is taken to be its
                                       inverse (the Laplace approximation),
                                       and the resulting predictive mean and
                                       uncertainty band are plotted, showing
                                       the band widening away from the
                                       training data.
  3. fig_laplace_1param_toy.pdf    -- A single-parameter toy example showing
                                       the Laplace approximation of a
                                       non-Gaussian posterior by a Gaussian
                                       matched at the mode, with the curvature
                                       (second derivative / scalar "Hessian")
                                       computed by hand.

All numbers in these figures come from a small synthetic toy dataset created
purely for illustration; they are NOT the real dataset or the real numerical
results reported in the book chapter (which used 3000 policies from a Spanish
motor-insurance portfolio and six different BNN architectures fit with the
Python `Laplace` package).
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

rng = np.random.default_rng(7)

# =============================================================================
# Figure 1: Small feed-forward network with weight *distributions* rather
# than fixed numbers -- the core visual idea of a Bayesian neural network.
# =============================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Layer x-positions
x_in, x_hid, x_out = 1.6, 5.0, 8.6

# Node y-positions
y_in = [5.6, 4.2, 2.8, 1.4]
y_hid = [5.2, 3.5, 1.8]
y_out = [3.5]

node_r = 0.38

def draw_layer(xs, ys, color, labels):
    for y, lab in zip(ys, labels):
        circ = mpatches.Circle((xs, y), node_r, facecolor=color,
                                edgecolor="black", zorder=5, linewidth=1.2)
        ax.add_patch(circ)
        ax.text(xs, y, lab, ha="center", va="center", fontsize=10, zorder=6)

in_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
hid_labels = [r"$h_1$", r"$h_2$", r"$h_3$"]
out_labels = [r"$y$"]

draw_layer(x_in, y_in, "#9FD8A0", in_labels)
draw_layer(x_hid, y_hid, "#9FB8E8", hid_labels)
draw_layer(x_out, y_out, "#F0A6A0", out_labels)

ax.text(x_in, 6.55, "Input\nlayer", ha="center", fontsize=11, color="#1a1a1a")
ax.text(x_hid, 6.55, "Hidden layer\n(weights = distributions)", ha="center", fontsize=11, color="#1a1a1a")
ax.text(x_out, 6.55, "Output\nlayer", ha="center", fontsize=11, color="#1a1a1a")

# Draw all edges input -> hidden, and store their midpoints
edge_mid_ih = []
for yi in y_in:
    for yh in y_hid:
        ax.plot([x_in + node_r, x_hid - node_r], [yi, yh],
                 color="gray", lw=0.8, alpha=0.55, zorder=1)
        edge_mid_ih.append(((x_in + x_hid) / 2, (yi + yh) / 2))

# Draw all edges hidden -> output
edge_mid_ho = []
for yh in y_hid:
    ax.plot([x_hid + node_r, x_out - node_r], [yh, y_out[0]],
             color="gray", lw=0.8, alpha=0.55, zorder=1)
    edge_mid_ho.append(((x_hid + x_out) / 2, (yh + y_out[0]) / 2))

# For a handful of representative edges, draw a small inset "bell curve" /
# error-bar glyph at the edge midpoint to show that the weight is a
# distribution, not a fixed number.
def draw_weight_glyph(ax, center, mu, sigma, width=0.85, height=0.6, color="#2F4C6B"):
    cx, cy = center
    t = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 60)
    dens = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    dens = dens / dens.max() * height
    t_scaled = cx + (t - mu) / (3.5 * sigma) * (width / 2)
    ax.fill_between(t_scaled, cy, cy + dens, color=color, alpha=0.55, zorder=4, lw=0)
    ax.plot(t_scaled, cy + dens, color=color, lw=1.0, zorder=4)
    ax.plot([cx, cx], [cy, cy], marker="|", color=color)

rep_idx_ih = [1, 4, 7, 9]
sigmas_ih = [0.35, 0.9, 0.5, 1.2]
mus_ih = [0.2, -0.4, 0.6, -0.1]
for idx, mu, sig in zip(rep_idx_ih, mus_ih, sigmas_ih):
    draw_weight_glyph(ax, edge_mid_ih[idx], mu, sig)

rep_idx_ho = [0, 2]
sigmas_ho = [0.4, 1.0]
mus_ho = [0.5, -0.3]
for idx, mu, sig in zip(rep_idx_ho, mus_ho, sigmas_ho):
    draw_weight_glyph(ax, edge_mid_ho[idx], mu, sig)

ax.text(1.0, 0.55,
        r"Each little bell curve is one weight $w_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$"
        "\nnarrow bump = confident (low variance);  wide bump = uncertain (high variance).",
        fontsize=9.5, color="#2F4C6B", ha="left", va="center")

ax.set_title("A Bayesian Neural Network: every weight is a distribution, not a number\n"
              "(illustrative diagram)", fontsize=12.5)

fig.tight_layout()
fig.savefig("fig_bnn_weight_uncertainty.pdf", bbox_inches="tight")
plt.close(fig)
print("Figure 1 written: fig_bnn_weight_uncertainty.pdf")

# =============================================================================
# Figure 2: Single-parameter toy Laplace approximation, worked "by hand".
# =============================================================================
# Toy negative log-posterior for a single parameter theta (e.g. one network
# weight), chosen to be non-Gaussian (a quartic bowl) so that the Laplace
# Gaussian approximation is visibly imperfect away from the mode, but exact
# in curvature at the mode.
def M(theta):
    return 0.5 * (theta - 1.0) ** 2 + 0.08 * (theta - 1.0) ** 4

def M_prime(theta):
    return (theta - 1.0) + 0.32 * (theta - 1.0) ** 3

def M_double_prime(theta):
    return 1.0 + 0.96 * (theta - 1.0) ** 2

# Step 1: MAP found by Newton's method (find where M'(theta) = 0)
theta = 3.0
for _ in range(50):
    theta = theta - M_prime(theta) / M_double_prime(theta)
theta_map = theta

# Step 2: Hessian (here just the scalar second derivative) at the MAP
A_scalar = M_double_prime(theta_map)

# Step 3: posterior variance = inverse Hessian
sigma2 = 1.0 / A_scalar
sigma = np.sqrt(sigma2)

theta_grid = np.linspace(-1.5, 4.0, 400)
true_unnorm = np.exp(-M(theta_grid))
true_posterior = true_unnorm / np.trapezoid(true_unnorm, theta_grid)
gauss_approx = np.exp(-0.5 * (theta_grid - theta_map) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(theta_grid, true_posterior, color="#C44E52", lw=2.5, label="True (toy) posterior")
ax.plot(theta_grid, gauss_approx, color="#4C72B0", lw=2.5, ls="--",
        label=fr"Laplace approx.: $\mathcal{{N}}({theta_map:.3f}, {sigma2:.3f})$")
ax.axvline(theta_map, color="black", lw=1, ls=":")
ax.annotate(fr"MAP: $\theta_{{MP}}={theta_map:.3f}$" "\n" fr"$A=M''(\theta_{{MP}})={A_scalar:.3f}$"
            "\n" fr"$\sigma^2=1/A={sigma2:.3f}$",
            xy=(theta_map, gauss_approx.max()), xytext=(theta_map + 0.9, gauss_approx.max() * 0.92),
            fontsize=9.5, color="#2F4C6B",
            arrowprops=dict(arrowstyle="->", color="#2F4C6B", lw=1))
ax.set_xlabel(r"$\theta$ (a single toy network weight)")
ax.set_ylabel("density")
ax.set_title("Laplace approximation, hand-worked scalar example\n"
              r"$M(\theta)=\frac{1}{2}(\theta-1)^2+0.08(\theta-1)^4$")
ax.legend(loc="upper left", fontsize=9.5, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("fig_laplace_1param_toy.pdf", bbox_inches="tight")
plt.close(fig)

print("\nFigure 3 (1-parameter hand example) summary:")
print(f"  theta_MP  = {theta_map:.6f}")
print(f"  A = M''(theta_MP) = {A_scalar:.6f}")
print(f"  posterior variance 1/A = {sigma2:.6f},  std = {sigma:.6f}")
print("Figure written: fig_laplace_1param_toy.pdf")

# =============================================================================
# Figure 3: Toy 1-D regression -- driver age -> claim amount, tiny BNN via
# Laplace approximation, predictive mean + uncertainty band.
# =============================================================================
# --- Toy "motor insurance claims" data: driver age (x) vs. claim amount (y) ---
# Purely illustrative synthetic numbers, NOT the book's real dataset.
x_data = np.array([22.0, 28.0, 35.0, 45.0, 55.0, 65.0])
y_data = np.array([4.8, 3.1, 2.0, 2.4, 3.4, 5.1])  # claim amount, in R'000, toy units

x_mean, x_std = x_data.mean(), x_data.std()
xn = (x_data - x_mean) / x_std  # normalise inputs for stable fitting

# --- Tiny network: 2 tanh hidden units plus a linear "skip" unit, 1 output ---
# f(x; theta) = v1*tanh(w1*x+b1) + v2*tanh(w2*x+b2) + m*x + c
# theta = [w1, b1, v1, w2, b2, v2, m, c]   (8 parameters)
# (the linear term m*x is what a single linear-output neuron with a fixed
#  identity activation would compute; keeping it makes the network's
#  extrapolation behaviour -- and hence its Laplace uncertainty band --
#  behave like the familiar "bowtie" widening of a Bayesian linear
#  regression away from the bulk of the training inputs, exactly as we
#  would also see from the *linear part* of a real trained BNN.)
def forward(theta, x):
    w1, b1, v1, w2, b2, v2, m, c = theta
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    return v1 * h1 + v2 * h2 + m * x + c

def forward_and_jac(theta, x):
    """Returns f(x;theta) and the analytic gradient d f / d theta at input x
    (x may be a scalar or a 1-D array; returns arrays broadcast over x)."""
    w1, b1, v1, w2, b2, v2, m, c = theta
    z1 = w1 * x + b1
    z2 = w2 * x + b2
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    f = v1 * h1 + v2 * h2 + m * x + c
    sech1_sq = 1.0 - h1 ** 2
    sech2_sq = 1.0 - h2 ** 2
    d_w1 = v1 * sech1_sq * x
    d_b1 = v1 * sech1_sq
    d_v1 = h1
    d_w2 = v2 * sech2_sq * x
    d_b2 = v2 * sech2_sq
    d_v2 = h2
    d_m = x * np.ones_like(f)
    d_c = np.ones_like(f)
    J = np.stack([d_w1, d_b1, d_v1, d_w2, d_b2, d_v2, d_m, d_c], axis=-1)
    return f, J

def neg_log_posterior(theta, x, y, alpha, beta):
    """M(theta) = alpha/2 * ||theta||^2 (Gaussian prior, precision alpha)
                 + beta/2 * sum (y - f(x;theta))^2 (Gaussian likelihood, precision beta)."""
    resid = y - forward(theta, x)
    return 0.5 * alpha * np.sum(theta ** 2) + 0.5 * beta * np.sum(resid ** 2)

def grad_neg_log_posterior(theta, x, y, alpha, beta):
    """Analytic gradient of the negative log posterior (chain rule through
    the tanh network)."""
    f, J = forward_and_jac(theta, x)
    resid = y - f
    # dM/dtheta = alpha*theta - beta * sum_n resid_n * df_n/dtheta
    return alpha * theta - beta * (resid[:, None] * J).sum(axis=0)

def numerical_hessian_from_grad(theta, x, y, alpha, beta, eps=1e-4):
    """Numerically approximate the Hessian A of the negative log posterior at
    theta by central-differencing the *analytic* gradient (more stable than
    finite-differencing the loss value twice)."""
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        gp = grad_neg_log_posterior(tp, x, y, alpha, beta)
        gm = grad_neg_log_posterior(tm, x, y, alpha, beta)
        H[:, i] = (gp - gm) / (2 * eps)
    return H

alpha_prior = 0.3   # prior precision on weights (Gaussian prior N(0, 1/alpha))
beta_noise = 8.0    # likelihood precision (1/noise-variance)

# --- Step 1: find the MAP estimate by gradient-based optimisation
#     (ordinary penalised / L2-regularised training of the network --
#     this is exactly "training as usual", just minimising M(theta)) ---
n_params = 8
rng2 = np.random.default_rng(9)
theta0 = rng2.normal(scale=1.0, size=n_params)
opt_result = minimize(
    neg_log_posterior, theta0, args=(xn, y_data, alpha_prior, beta_noise),
    jac=grad_neg_log_posterior, method="BFGS", options={"gtol": 1e-10, "maxiter": 5000},
)
theta_map7 = opt_result.x

# --- Step 2: numerically approximate the Hessian A of the negative log
#     posterior at the MAP estimate (by central-differencing the analytic
#     gradient) ---
A = numerical_hessian_from_grad(theta_map7, xn, y_data, alpha_prior, beta_noise)
# Symmetrise for numerical safety
A = 0.5 * (A + A.T)
assert np.all(np.linalg.eigvalsh(A) > 0), "Hessian at the MAP must be positive definite for the Laplace approximation to be valid"
A_reg = A

# --- Step 3: posterior covariance = inverse Hessian ---
Sigma = np.linalg.inv(A_reg)

# --- Step 4: propagate uncertainty through the network via a local
#     linearisation (delta method): Var[f(x)] ~= J(x) Sigma J(x)^T, plus
#     the observation noise variance 1/beta for the *predictive* band.
x_plot = np.linspace(-15, 90, 250)
xn_plot = (x_plot - x_mean) / x_std
mean_pred, J_plot = forward_and_jac(theta_map7, xn_plot)   # J_plot: (250, 7)
var_pred = np.einsum("ni,ij,nj->n", J_plot, Sigma, J_plot)
var_pred = np.clip(var_pred, 0.0, None)
std_pred = np.sqrt(var_pred)
std_pred_total = np.sqrt(var_pred + 1.0 / beta_noise)  # add observation noise

fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.fill_between(x_plot, mean_pred - 2 * std_pred_total, mean_pred + 2 * std_pred_total,
                 color="#4C72B0", alpha=0.15, label=r"$\pm 2$ s.d. predictive band")
ax.fill_between(x_plot, mean_pred - 2 * std_pred, mean_pred + 2 * std_pred,
                 color="#4C72B0", alpha=0.30, label=r"$\pm 2$ s.d. weight-uncertainty band")
ax.plot(x_plot, mean_pred, color="#2F4C6B", lw=2.5, label="Predictive mean $f(x;\\theta_{MP})$")
ax.scatter(x_data, y_data, color="#C44E52", s=60, zorder=5, label="Toy training data (6 points)")
ax.set_xlabel("Driver age (years) -- toy input")
ax.set_ylabel("Claim amount (R'000) -- toy output")
ax.set_title("Toy 1-D BNN regression via the Laplace approximation\n"
              "(illustrative simulation, not the book's real data)", fontsize=12)
ax.legend(loc="upper right", fontsize=8.5, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("fig_toy_laplace_1d.pdf", bbox_inches="tight")
plt.close(fig)

print("\nFigure 2 (toy 1-D BNN regression) summary:")
print(f"  Training data: x = {x_data.tolist()}")
print(f"                 y = {y_data.tolist()}")
print(f"  theta_MP (w1,b1,v1,w2,b2,v2,m,c) = {np.round(theta_map7, 4).tolist()}")
print(f"  Band widens away from data: std at x=35 (near data) = "
      f"{std_pred_total[np.argmin(np.abs(x_plot-35))]:.4f}, "
      f"std at x=90 (far from data) = {std_pred_total[np.argmin(np.abs(x_plot-90))]:.4f}, "
      f"std at x=-15 (far from data) = {std_pred_total[np.argmin(np.abs(x_plot+15))]:.4f}")
print("Figure written: fig_toy_laplace_1d.pdf")

print("\nAll figures written: fig_bnn_weight_uncertainty.pdf, "
      "fig_laplace_1param_toy.pdf, fig_toy_laplace_1d.pdf")
