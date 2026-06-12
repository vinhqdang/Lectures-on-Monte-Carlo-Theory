"""Generate all figures for Chapter 4 slides (Simulation Output Analysis)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(31415926)

# ---------------------------------------------------------------
# Fig 1: CLT convergence -- sample mean of Bernoulli(p=pi/4)
# distribution approaches normal as R increases
# ---------------------------------------------------------------
p_true = np.pi / 4
R_vals = [10, 50, 200, 1000]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
N_rep = 5000
for ax, R in zip(axes, R_vals):
    samples = rng.binomial(1, p_true, size=(N_rep, R)).mean(axis=1)
    sigma = np.sqrt(p_true * (1 - p_true) / R)
    x = np.linspace(samples.min(), samples.max(), 200)
    ax.hist(samples, bins=40, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', label='Simulation')
    ax.plot(x, stats.norm.pdf(x, p_true, sigma), 'r-', lw=2,
            label=r'$\mathcal{N}(I, \sigma_Y^2/R)$')
    ax.axvline(p_true, color='k', ls='--', lw=1.2, label=f'$I=\\pi/4$')
    ax.set_title(f'$R={R}$', fontsize=11)
    ax.set_xlabel(r'$\hat{Y}_R$', fontsize=10)
    if R == 10:
        ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle(r'CLT: Distribution of $\hat{Y}_R$ for estimating $I=\pi/4$', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_clt_convergence.pdf', dpi=150)
plt.close()
print("fig_clt_convergence.pdf done")

# ---------------------------------------------------------------
# Fig 2: Hit-or-miss estimator (Fig 4.1 in book)
# Estimating int_0^1 k(x) dx using HoM with R=7 pairs
# ---------------------------------------------------------------
def k_func(x):
    return (np.exp(x) - 1) / (np.e - 1)

rng2 = np.random.default_rng(42)
# Use same example as book: R=7 pairs -> U_1,...,U_14
U = np.array([0.62, 0.15, 0.47, 0.11, 0.83, 0.75, 0.33,
              0.23, 0.78, 0.44, 0.65, 0.19, 0.71, 0.85])
# pairs: (U_{2j-1}, U_{2j}) for j=1..7
pairs = [(U[2*j], U[2*j+1]) for j in range(7)]

fig, ax = plt.subplots(figsize=(7, 4))
x_plot = np.linspace(0, 1, 300)
ax.fill_between(x_plot, 0, k_func(x_plot), alpha=0.12, color='steelblue',
                label=r'$k(x)$')
ax.plot(x_plot, k_func(x_plot), 'steelblue', lw=2, label=r'$k(x)=\frac{e^x-1}{e-1}$')
for j, (u1, u2) in enumerate(pairs):
    kval = k_func(u1)
    hit = u2 < kval
    color = 'green' if hit else 'red'
    marker = 'o' if hit else 'x'
    label = ('Hit ($Y=1$)' if hit else 'Miss ($Y=0$)') if j == 0 or (hit and j < 3) else ''
    ax.scatter(u1, u2, color=color, marker=marker, s=80, zorder=5)
ax.scatter([], [], color='green', marker='o', s=80, label='Hit ($Y=1$)')
ax.scatter([], [], color='red', marker='x', s=80, label='Miss ($Y=0$)')
ax.set_xlabel('$U_1$', fontsize=12)
ax.set_ylabel('$U_2$', fontsize=12)
ax.set_title('Hit-or-Miss estimator for $\\int_0^1 k(x)\\,dx$', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_hit_or_miss.pdf', dpi=150)
plt.close()
print("fig_hit_or_miss.pdf done")

# ---------------------------------------------------------------
# Fig 3: Sample mean convergence to pi (Examples 4.1.6 and 4.1.7)
# Two estimators: indicator (Ex 4.1.6) and sqrt (Ex 4.1.7)
# ---------------------------------------------------------------
Rs = np.array([2**k for k in range(4, 16)])
n_trials = 30

fig, ax = plt.subplots(figsize=(8, 4.5))
rng3 = np.random.default_rng(100)
for trial in range(n_trials):
    errs_ind = []
    errs_sqrt = []
    for R in Rs:
        u = rng3.random(2 * R).reshape(R, 2)
        y_ind = 4 * ((u[:, 0]**2 + u[:, 1]**2) <= 1).astype(float)
        y_sqrt = 4 * np.sqrt(1 - rng3.random(R)**2)
        errs_ind.append(abs(y_ind.mean() - np.pi))
        errs_sqrt.append(abs(y_sqrt.mean() - np.pi))
    lw = 0.4
    ax.loglog(Rs, errs_ind,  color='steelblue', lw=lw, alpha=0.5)
    ax.loglog(Rs, errs_sqrt, color='darkorange', lw=lw, alpha=0.5)

# Reference lines
sigma_ind  = np.sqrt(np.pi * (1 - np.pi / 4))  # ~ sqrt(2.6968/4 * 16)
sigma_sqrt = np.sqrt(0.797)
ref_x = Rs.astype(float)
ax.loglog(ref_x, 1.96 * sigma_ind / np.sqrt(ref_x), 'b-',
          lw=2, label=r'$1.96\sigma_{\rm ind}/\sqrt{R}$ (Ex 4.1.6)')
ax.loglog(ref_x, 1.96 * sigma_sqrt / np.sqrt(ref_x), '-',
          color='darkorange', lw=2, label=r'$1.96\sigma_{\rm sqrt}/\sqrt{R}$ (Ex 4.1.7)')
ax.loglog(ref_x, 2 / np.sqrt(ref_x), 'k--', lw=1.5, label=r'$O(R^{-1/2})$')
ax.set_xlabel('$R$ (replications)', fontsize=12)
ax.set_ylabel(r'$|\hat{Y}_R - \pi|$', fontsize=12)
ax.set_title(r'Convergence of two CMC estimators for $\pi$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_pi_convergence.pdf', dpi=150)
plt.close()
print("fig_pi_convergence.pdf done")

# ---------------------------------------------------------------
# Fig 4: Confidence interval coverage (1D)
# ---------------------------------------------------------------
R = 500
n_ci = 200
I_true = np.pi / 4
sigma_true = np.sqrt(I_true * (1 - I_true))

rng4 = np.random.default_rng(7)
covered = 0
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(n_ci):
    Y = rng4.binomial(1, I_true, R)
    y_bar = Y.mean()
    s = Y.std(ddof=1)
    lo = y_bar - 1.96 * s / np.sqrt(R)
    hi = y_bar + 1.96 * s / np.sqrt(R)
    c = (lo <= I_true <= hi)
    covered += c
    color = 'steelblue' if c else 'red'
    ax.plot([lo, hi], [i, i], color=color, lw=0.8, alpha=0.7)
ax.axvline(I_true, color='k', lw=2, ls='--', label=f'$I = \\pi/4 \\approx {I_true:.4f}$')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Replication index', fontsize=12)
ax.set_title(f'95% CI coverage: {covered}/{n_ci} = {100*covered/n_ci:.1f}% (R={R})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_ci_coverage.pdf', dpi=150)
plt.close()
print("fig_ci_coverage.pdf done")

# ---------------------------------------------------------------
# Fig 5: Bivariate normal confidence regions (Fig 4.2 in book)
# Sigma = [[1,2],[2,6]], alpha=0.05
# ---------------------------------------------------------------
Sigma = np.array([[1., 2.], [2., 6.]])
mu = np.array([0., 0.])
alpha = 0.05
d = 2

# Sample 5000 points
rng5 = np.random.default_rng(12345)
L = np.linalg.cholesky(Sigma)
Z = rng5.standard_normal((5000, 2))
X = (L @ Z.T).T

# Ellipsoid
q_chi2 = chi2.ppf(1 - alpha, df=d)  # 5.991
eigvals, eigvecs = np.linalg.eigh(Sigma)
# eigvals sorted ascending; we want descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

theta = np.linspace(0, 2 * np.pi, 300)
ell_axes = np.sqrt(q_chi2 * eigvals)
ell = eigvecs @ (np.array([ell_axes[0] * np.cos(theta),
                            ell_axes[1] * np.sin(theta)]))

# Parallelogram
Sigma_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
z_par = stats.norm.ppf((np.sqrt(0.95) + 1) / 2)
corners_h = np.array([[-z_par, -z_par], [-z_par, z_par],
                       [z_par, z_par], [z_par, -z_par], [-z_par, -z_par]])
par_corners = (Sigma_half @ corners_h.T).T

# Rectangle: sigma_1=1, sigma_2=sqrt(6)
sigma1 = np.sqrt(Sigma[0, 0])
sigma2 = np.sqrt(Sigma[1, 1])
z_rec = stats.norm.ppf((0.95**(1/2) + 1) / 2)
c1 = sigma1 * z_rec
c2 = sigma2 * z_rec

fig, ax = plt.subplots(figsize=(6, 7))
ax.scatter(X[:, 0], X[:, 1], s=2, color='gray', alpha=0.3, label='Samples')
ax.plot(ell[0], ell[1], 'k-', lw=2.0, label='Confidence ellipsoid')
# principal axes
for i in range(2):
    v = eigvecs[:, i] * ell_axes[i]
    ax.annotate('', xy=v, xytext=-v,
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.plot(par_corners[:, 0], par_corners[:, 1], '-',
        color='brown', lw=2.0, label='Confidence parallelogram')
rect_x = [-c1, c1, c1, -c1, -c1]
rect_y = [-c2, -c2, c2, c2, -c2]
ax.plot(rect_x, rect_y, 'b-', lw=1.5, label='Rectangular region')
ax.set_xlabel('$X_1$', fontsize=12)
ax.set_ylabel('$X_2$', fontsize=12)
ax.set_title(r'95% confidence regions for $\mathbf{X}\sim\mathcal{N}(\mathbf{0},\mathbf{\Sigma})$'
             '\n' + r'$\mathbf{\Sigma}=[[1,2],[2,6]]$', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5); ax.set_ylim(-8, 8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(f'{OUT}/fig_conf_regions_2d.pdf', dpi=150)
plt.close()
print("fig_conf_regions_2d.pdf done")

# ---------------------------------------------------------------
# Fig 6: Delta method -- k(y) = y^2, estimating k(I) = I^2
# Show distribution of k(Y_R) and CLT approximation
# ---------------------------------------------------------------
I_val = 0.5
sigma_Y = 0.3
R_dm = 200
N_dm = 10000

rng6 = np.random.default_rng(99)
Y_samples = rng6.normal(I_val, sigma_Y, (N_dm, R_dm))
Y_bar = Y_samples.mean(axis=1)
Z_bar = Y_bar**2  # k(Y_bar), estimating k(I) = I^2 = 0.25

# asymptotic variance: zeta^2 = k'(I)^2 * sigma_Y^2 = (2I)^2 * sigma_Y^2
zeta2 = (2 * I_val)**2 * sigma_Y**2
z_mean = I_val**2
z_std_asymp = np.sqrt(zeta2 / R_dm)

x_plot = np.linspace(Z_bar.min(), Z_bar.max(), 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(Z_bar, bins=60, density=True, alpha=0.6, color='steelblue',
        edgecolor='white', label=r'$k(\hat{Y}_R) = \hat{Y}_R^2$')
ax.plot(x_plot, stats.norm.pdf(x_plot, z_mean, z_std_asymp), 'r-', lw=2.5,
        label=r'$\mathcal{N}(k(I),\,\varsigma^2/R)$ (delta method)')
ax.axvline(z_mean, color='k', ls='--', lw=1.5, label=f'$k(I)=I^2={z_mean}$')
ax.set_xlabel(r'$k(\hat{Y}_R)$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(r'Delta Method: $k(y)=y^2$, $I=0.5$, $R=200$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_delta_method.pdf', dpi=150)
plt.close()
print("fig_delta_method.pdf done")

# ---------------------------------------------------------------
# Fig 7: MSE = Variance + Bias^2 decomposition illustration
# ---------------------------------------------------------------
R_arr = np.arange(10, 500, 5).astype(float)
# Example: biased estimator with beta = 0.2 (asymptotic bias),
# asymptotic variance zeta^2 = 1.0
beta2 = 0.04   # beta^2
zeta2 = 1.0

var_part = zeta2 / R_arr
bias_part = beta2 / R_arr**2
mse = var_part + bias_part

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(R_arr, mse, 'k-', lw=2.5, label=r'MSE $\approx \varsigma^2/R + \beta^2/R^2$')
ax.loglog(R_arr, var_part, 'b--', lw=1.8, label=r'Variance $\varsigma^2/R$')
ax.loglog(R_arr, bias_part, 'r:', lw=1.8, label=r'Bias$^2$: $\beta^2/R^2$')
ax.set_xlabel('$R$ (replications)', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('Bias-Variance Decomposition of MSE', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_mse_decomp.pdf', dpi=150)
plt.close()
print("fig_mse_decomp.pdf done")

# ---------------------------------------------------------------
# Fig 8: Quantile estimation -- order statistics
# Distribution: Exponential(1), p=0.9 quantile
# ---------------------------------------------------------------
p_q = 0.9
q_true = -np.log(1 - p_q)  # = log(10) approx 2.303

R_q = 200
rng8 = np.random.default_rng(55)
X_q = rng8.exponential(1.0, R_q)
X_sorted = np.sort(X_q)

q_hat = X_sorted[int(np.floor(p_q * R_q))]

# Asymptotic distribution
f_qp = np.exp(-q_true)  # density at quantile
zeta_q = np.sqrt(p_q * (1 - p_q)) / f_qp

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Left: histogram of samples with quantile
ax = axes[0]
ax.hist(X_q, bins=30, density=True, color='steelblue', alpha=0.6, edgecolor='white')
x_exp = np.linspace(0, 6, 200)
ax.plot(x_exp, np.exp(-x_exp), 'r-', lw=2, label=r'$f(x)=e^{-x}$')
ax.axvline(q_true, color='k', ls='--', lw=1.5, label=f'$q_{{0.9}}={q_true:.3f}$')
ax.axvline(q_hat, color='darkorange', ls='-', lw=1.5, label=f'$\\hat{{q}}_p={q_hat:.3f}$')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Exp(1) sample ($R={R_q}$), 0.9-quantile', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: distribution of q_hat over many replications
rng8b = np.random.default_rng(66)
n_rep_q = 3000
q_hats = []
for _ in range(n_rep_q):
    X_rep = rng8b.exponential(1.0, R_q)
    q_hats.append(np.sort(X_rep)[int(np.floor(p_q * R_q))])
q_hats = np.array(q_hats)

ax = axes[1]
x_qh = np.linspace(q_hats.min(), q_hats.max(), 200)
ax.hist(q_hats, bins=50, density=True, color='steelblue', alpha=0.6, edgecolor='white',
        label='Simulation of $\\hat{q}_p$')
ax.plot(x_qh, stats.norm.pdf(x_qh, q_true, zeta_q / np.sqrt(R_q)), 'r-', lw=2.5,
        label=r'CLT: $\mathcal{N}(q_p, p(1-p)/(Rf^2(q_p)))$')
ax.axvline(q_true, color='k', ls='--', lw=1.5, label=f'$q_p={q_true:.3f}$')
ax.set_xlabel(r'$\hat{q}_p$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(r'CLT for sample quantile $\hat{q}_p$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_quantile_estimation.pdf', dpi=150)
plt.close()
print("fig_quantile_estimation.pdf done")

# ---------------------------------------------------------------
# Fig 9: Chebyshev vs Chernoff vs CLT comparison (Fig 4.4 in book)
# p=1/2, compare probability bounds for |S_R - R/2| >= R/4
# ---------------------------------------------------------------
R_arr2 = np.arange(1, 41, 1).astype(float)
# Chebyshev: 4/R
cheby = 4 / R_arr2
# Chernoff: 2*exp(-R/24) for eps=1/2
chernoff = 2 * np.exp(-R_arr2 / 24)
# CLT approximation: sqrt(8/(pi*R)) * exp(-R/8)
clt_approx = np.sqrt(8 / (np.pi * R_arr2)) * np.exp(-R_arr2 / 8)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(R_arr2, cheby, 'b-', lw=2.0, label=r'Chebyshev: $4/R$')
ax.plot(R_arr2, chernoff, 'g--', lw=2.0, label=r'Chernoff: $2\exp(-R/24)$')
ax.plot(R_arr2, clt_approx, 'r:', lw=2.0,
        label=r'CLT: $\sqrt{8/(\pi R)}\exp(-R/8)$')
ax.set_xlabel('$R$', fontsize=12)
ax.set_ylabel(r'$\mathbb{P}(|B_1+\cdots+B_R - R/2| \geq R/4)$', fontsize=11)
ax.set_title(r'Comparison: Chebyshev, Chernoff, CLT bounds ($p=1/2$, $\varepsilon=1/2$)', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, 40); ax.set_ylim(-0.05, 4.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_chernoff_comparison.pdf', dpi=150)
plt.close()
print("fig_chernoff_comparison.pdf done")

# ---------------------------------------------------------------
# Fig 10: Required R as function of epsilon for estimating pi
# (Fig 4.5 in book): absolute vs relative error
# ---------------------------------------------------------------
delta = 0.01
z_val = stats.norm.ppf(1 - delta / 2)  # z_{1-delta/2} for delta=0.01 -> 2.576
VarY = 16 * (np.pi / 4) * (1 - np.pi / 4)  # Var Y for indicator estimator

eps_arr = np.linspace(0.2, 1.0, 200)

R_abs = z_val**2 * VarY / eps_arr**2
R_rel1 = 3 * np.log(2 / delta) / (eps_arr**2 * np.pi)  # Chernoff for relative
R_rel2 = z_val**2 * VarY / (eps_arr**2 * np.pi**2)     # CLT for relative

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(eps_arr, R_abs,  'r-',  lw=2.0, label=r'$R^{\rm abs}$ (absolute, CLT)')
ax.plot(eps_arr, R_rel1, 'g--', lw=2.0, label=r'$R^{{\rm rel},1}$ (relative, Chernoff)')
ax.plot(eps_arr, R_rel2, 'b:',  lw=2.0, label=r'$R^{{\rm rel},2}$ (relative, CLT)')
ax.set_xlabel(r'$\varepsilon$', fontsize=13)
ax.set_ylabel('Required $R$', fontsize=12)
ax.set_title(r'Required replications to estimate $\pi$ ($\delta=0.01$)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1300)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_required_R_pi.pdf', dpi=150)
plt.close()
print("fig_required_R_pi.pdf done")

# ---------------------------------------------------------------
# Fig 11: Kernel density estimation of 1/f(q_p)
# ---------------------------------------------------------------
rng11 = np.random.default_rng(77)
R_kde = 500
X_kde = rng11.exponential(1.0, R_kde)
X_kde_sorted = np.sort(X_kde)

p_kde = 0.75
q_true_kde = -np.log(1 - p_kde)  # true 0.75-quantile of Exp(1) = log(4)

# Estimate 1/f(q_p) using order-statistic bandwidth
m_vals = [5, 15, 30]
idx_p = int(np.floor(p_kde * R_kde))

fig, ax = plt.subplots(figsize=(7, 4))
x_show = np.linspace(0, 3, 300)
ax.plot(x_show, np.exp(-x_show), 'k-', lw=2, label=r'True $f(x)=e^{-x}$', zorder=5)
ax.axvline(q_true_kde, color='k', ls='--', lw=1.2, alpha=0.7)
colors = ['steelblue', 'darkorange', 'green']
for m, col in zip(m_vals, colors):
    # estimate 1/f(q_p) using Proposition 4.3.4 kernel estimator
    lo = max(0, idx_p - m)
    hi = min(R_kde - 1, idx_p + m)
    q_hat_1_f = R_kde * (X_kde_sorted[hi] - X_kde_sorted[lo]) / (2 * m)
    f_est = 1 / q_hat_1_f
    ax.axhline(f_est, color=col, ls=':', lw=1.5,
               label=f'$\\hat{{f}}(\\hat{{q}}_p)$ with $m={m}$: {f_est:.3f}')
ax.axhline(np.exp(-q_true_kde), color='k', ls='-', lw=1.0, alpha=0.5,
           label=f'True $f(q_p)={np.exp(-q_true_kde):.3f}$')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(r'Estimation of $1/f(q_p)$ via order statistics, $p=0.75$', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 3); ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_kde_quantile.pdf', dpi=150)
plt.close()
print("fig_kde_quantile.pdf done")

print(f"\nAll figures saved to: {OUT}")
