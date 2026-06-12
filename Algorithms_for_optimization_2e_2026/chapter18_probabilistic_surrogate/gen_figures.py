"""
gen_figures.py  –  Chapter 18: Probabilistic Surrogate Models
Generates all figures needed for chapter18_slides.tex.
Run with:  conda run -n py313 python3 gen_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 – Multivariate Gaussians with different covariance matrices (Fig 18.1)
# ═══════════════════════════════════════════════════════════════════════════════
def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    coeff = (2*np.pi)**(-n/2) * det**(-0.5)
    diff = pos - mu
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, Sigma_inv, diff)
    return coeff * np.exp(exponent)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
mu = np.array([0.0, 0.0])
covs = [
    np.array([[1, 0], [0, 1]]),
    np.array([[3, 0], [0, 0.5]]),
    np.array([[1, 0.9], [0.9, 1]]),
]
titles = [
    r'$\Sigma=[[1,0],[0,1]]$',
    r'$\Sigma=[[3,0],[0,0.5]]$',
    r'$\Sigma=[[1,0.9],[0.9,1]]$',
]
for ax, cov, title in zip(axes, covs, titles):
    Z = multivariate_gaussian(pos, mu, cov)
    ax.contour(X, Y, Z, levels=6, cmap='viridis')
    ax.set_xlabel(r'$x_1$', fontsize=9)
    ax.set_ylabel(r'$x_2$', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
plt.tight_layout()
savefig("fig_gaussian_contours.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Kernel functions and sampled functions (Fig 18.3)
# ═══════════════════════════════════════════════════════════════════════════════
def gp_samples(kernel, x, n_samples=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    K = np.array([[kernel(xi, xj) for xj in x] for xi in x])
    K += 1e-8 * np.eye(len(x))
    L = np.linalg.cholesky(K)
    return [L @ rng.standard_normal(len(x)) for _ in range(n_samples)]

x = np.linspace(-5, 5, 300)

kernels = {
    r'Constant: $c_f^2$':           lambda a, b: 1.0,
    r'Linear: $\sum c_d^2 x_d x_d^{\prime}$': lambda a, b: a*b + 1,
    r'Polynomial: $(x^T x^{\prime}+c_p^2)^p$': lambda a, b: (a*b + 1)**2,
    r'Exponential: $\exp(-r)$':    lambda a, b: np.exp(-abs(a-b)),
    r'$\gamma$-Exp: $\exp(-(r/\ell)^\gamma)$':
                                   lambda a, b: np.exp(-(abs(a-b)/1.0)**1.5),
    r'Squared Exp: $\exp(-r^2/2\ell^2)$':
                                   lambda a, b: np.exp(-0.5*(a-b)**2),
}

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
rng = np.random.default_rng(0)
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
for ax, (name, k) in zip(axes.flat, kernels.items()):
    samples = gp_samples(k, x, n_samples=5, rng=rng)
    for s, c in zip(samples, colors):
        ax.plot(x, s, color=c, linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.3, linestyle='--')
    ax.set_title(name, fontsize=8)
    ax.set_xlim(-5, 5); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('$x$', fontsize=8); ax.tick_params(labelsize=7)
plt.suptitle('Functions sampled from GPs with different kernels', fontsize=10)
plt.tight_layout()
savefig("fig_kernel_samples.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 – GP prediction with squared exponential kernel (Fig 18.5)
# ═══════════════════════════════════════════════════════════════════════════════
def sq_exp_kernel(x1, x2, ell=1.0, sigma_f=1.0):
    return sigma_f**2 * np.exp(-0.5 * ((x1 - x2) / ell)**2)

def gp_predict(X_train, y_train, X_pred, kernel, noise=0.0):
    m = len(X_train)
    n = len(X_pred)
    K_XX = np.array([[kernel(X_train[i], X_train[j])
                      for j in range(m)] for i in range(m)])
    K_XX += (noise + 1e-8) * np.eye(m)
    K_Xs = np.array([[kernel(X_pred[i], X_train[j])
                      for j in range(m)] for i in range(n)])
    K_ss = np.array([[kernel(X_pred[i], X_pred[j])
                      for j in range(n)] for i in range(n)])
    K_inv = np.linalg.inv(K_XX)
    mu = K_Xs @ K_inv @ y_train
    cov = K_ss - K_Xs @ K_inv @ K_Xs.T
    var = np.diag(cov)
    return mu, np.maximum(var, 0)

# True function
f_true = lambda x: np.sin(x) + 0.5*np.sin(2*x)

X_train = np.array([-3.0, -1.5, 0.5, 2.0])
y_train = f_true(X_train)
X_pred  = np.linspace(-5, 5, 300)
y_true  = f_true(X_pred)

kernel = lambda a, b: sq_exp_kernel(a, b, ell=1.0)
mu, var = gp_predict(X_train, y_train, X_pred, kernel, noise=0.0)
std = np.sqrt(var)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(X_pred, mu - 1.96*std, mu + 1.96*std,
                alpha=0.35, color='steelblue', label='95% confidence interval')
ax.plot(X_pred, y_true, 'k-', linewidth=1.2, label='true objective function')
ax.plot(X_pred, mu, '-', color='steelblue', linewidth=1.5, label='predicted mean')
ax.scatter(X_train, y_train, color='black', zorder=5, marker='.', s=60, label='fit points')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('GP with Squared Exponential Kernel (noise-free)', fontsize=10)
plt.tight_layout()
savefig("fig_gp_prediction.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 – GP with gradient observations vs without (Fig 18.6)
# ═══════════════════════════════════════════════════════════════════════════════
def kff(x1, x2, ell=1.0):
    return np.exp(-0.5*(x1-x2)**2/ell**2)

def kvf(x, xp, ell=1.0):
    return -(x - xp)/ell**2 * kff(x, xp, ell)

def kfv(x, xp, ell=1.0):
    return (xp - x)/ell**2 * kff(x, xp, ell)

def kvv(x, xp, ell=1.0):
    d = x - xp
    return (1.0/ell**2 - d**2/ell**4) * kff(x, xp, ell)

f_true2 = lambda x: np.sin(x)
df_true2 = lambda x: np.cos(x)

X_obs = np.array([-2.0, 0.0, 2.0])
y_obs = f_true2(X_obs)
dy_obs = df_true2(X_obs)
X_star = np.linspace(-4, 4, 300)
ell = 1.0

# --- Without gradients ---
mu_no, var_no = gp_predict(X_obs, y_obs, X_star, lambda a, b: kff(a, b, ell))
std_no = np.sqrt(var_no)

# --- With gradients ---
m = len(X_obs)
n = len(X_star)
# Build joint covariance [K_ff  K_fv; K_vf  K_vv]
K_ff_train = np.array([[kff(X_obs[i],X_obs[j],ell) for j in range(m)] for i in range(m)])
K_vf_train = np.array([[kvf(X_obs[i],X_obs[j],ell) for j in range(m)] for i in range(m)])
K_fv_train = K_vf_train.T
K_vv_train = np.array([[kvv(X_obs[i],X_obs[j],ell) for j in range(m)] for i in range(m)])
K_train = np.block([[K_ff_train, K_fv_train],
                    [K_vf_train, K_vv_train]]) + 1e-8*np.eye(2*m)

K_star_f = np.array([[kff(X_star[i],X_obs[j],ell) for j in range(m)] for i in range(n)])
K_star_v = np.array([[kfv(X_star[i],X_obs[j],ell) for j in range(m)] for i in range(n)])
K_star_obs = np.hstack([K_star_f, K_star_v])
K_ss_diag = np.array([kff(xi, xi, ell) for xi in X_star])

y_combined = np.concatenate([y_obs, dy_obs])
K_inv = np.linalg.inv(K_train)
mu_grad = K_star_obs @ K_inv @ y_combined
var_grad = K_ss_diag - np.sum((K_star_obs @ K_inv) * K_star_obs, axis=1)
std_grad = np.sqrt(np.maximum(var_grad, 0))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(X_star, mu_no - 1.96*std_no, mu_no + 1.96*std_no,
                alpha=0.35, color='steelblue', label='95% CI without gradient')
ax.fill_between(X_star, mu_grad - 1.96*std_grad, mu_grad + 1.96*std_grad,
                alpha=0.4, color='salmon', label='95% CI with gradient')
ax.plot(X_star, f_true2(X_star), 'k-', linewidth=1.2, label='true function')
ax.plot(X_star, mu_no,   '-', color='steelblue', linewidth=1.2, label='mean w/o gradient')
ax.plot(X_star, mu_grad, '-', color='red',       linewidth=1.2, label='mean with gradient')
ax.scatter(X_obs, y_obs, color='black', zorder=5, s=50, marker='.')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('GP with and without Gradient Observations', fontsize=10)
ax.legend(fontsize=7, ncol=2, loc='upper right')
plt.tight_layout()
savefig("fig_gp_gradient.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 – Noisy GP (Fig 18.7)
# ═══════════════════════════════════════════════════════════════════════════════
np.random.seed(7)
X_noisy = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5])
noise_var = 0.3
y_noisy = f_true(X_noisy) + np.random.normal(0, np.sqrt(noise_var), len(X_noisy))

mu_n, var_n = gp_predict(X_noisy, y_noisy, X_pred, kernel, noise=noise_var)
std_n = np.sqrt(var_n)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(X_pred, mu_n - 1.96*std_n, mu_n + 1.96*std_n,
                alpha=0.35, color='steelblue', label='95% confidence region')
ax.plot(X_pred, f_true(X_pred), 'k-', linewidth=1.2, label='true objective function')
ax.plot(X_pred, mu_n, '-', color='steelblue', linewidth=1.5, label='predicted mean')
ax.scatter(X_noisy, y_noisy, color='black', zorder=5, marker='.', s=60, label='fit points')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.legend(fontsize=8)
ax.set_title('Noisy GP with Squared Exponential Kernel', fontsize=10)
plt.tight_layout()
savefig("fig_gp_noisy.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 – MLE log-likelihood landscape (illustrative)
# ═══════════════════════════════════════════════════════════════════════════════
def log_marginal_likelihood(X_train, y_train, ell, sigma_f=1.0, noise=1e-4):
    m = len(X_train)
    K = np.array([[sigma_f**2 * np.exp(-0.5*((X_train[i]-X_train[j])/ell)**2)
                   for j in range(m)] for i in range(m)])
    K += noise * np.eye(m)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    log_lik = (-0.5 * y_train @ alpha
               - np.sum(np.log(np.diag(L)))
               - 0.5 * m * np.log(2*np.pi))
    return log_lik

X_fit = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
y_fit = f_true(X_fit)

ells = np.linspace(0.1, 4.0, 200)
lmls = [log_marginal_likelihood(X_fit, y_fit, e) for e in ells]
best_ell = ells[np.argmax(lmls)]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(ells, lmls, 'b-', linewidth=1.5)
ax.axvline(best_ell, color='red', linestyle='--',
           label=f'optimal $\\ell = {best_ell:.2f}$')
ax.set_xlabel('length-scale $\\ell$')
ax.set_ylabel('log marginal likelihood')
ax.set_title('MLE Fitting: Log Marginal Likelihood vs. $\\ell$', fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
savefig("fig_mle_loglik.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 – 2D GP contour plot (Fig 18.4 style)
# ═══════════════════════════════════════════════════════════════════════════════
def sq_exp_2d(x1, x2, ell=1.0):
    return np.exp(-0.5 * np.sum((np.array(x1) - np.array(x2))**2) / ell**2)

# Build GP prediction on 2D grid
rng2 = np.random.default_rng(1)
X2_train = rng2.uniform(-2, 2, (8, 2))
y2_train = np.sin(X2_train[:,0]) * np.cos(X2_train[:,1])

gx = np.linspace(-2.5, 2.5, 50)
gy = np.linspace(-2.5, 2.5, 50)
GX, GY = np.meshgrid(gx, gy)
X2_pred = np.column_stack([GX.ravel(), GY.ravel()])

m2 = len(X2_train)
n2 = len(X2_pred)
K2_train = np.array([[sq_exp_2d(X2_train[i], X2_train[j])
                      for j in range(m2)] for i in range(m2)]) + 1e-6*np.eye(m2)
K2_cross  = np.array([[sq_exp_2d(X2_pred[i], X2_train[j])
                       for j in range(m2)] for i in range(n2)])
mu2 = K2_cross @ np.linalg.solve(K2_train, y2_train)
MU2 = mu2.reshape(GX.shape)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
# Mean
im0 = axes[0].contourf(GX, GY, MU2, levels=20, cmap='RdYlBu_r')
axes[0].scatter(X2_train[:,0], X2_train[:,1], c='k', s=40, zorder=5)
axes[0].set_title('GP Posterior Mean (2D)', fontsize=10)
axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
plt.colorbar(im0, ax=axes[0])

# Variance
var2_diag = np.array([sq_exp_2d(X2_pred[i], X2_pred[i]) -
                      K2_cross[i] @ np.linalg.solve(K2_train, K2_cross[i])
                      for i in range(n2)])
VAR2 = var2_diag.reshape(GX.shape)
im1 = axes[1].contourf(GX, GY, np.sqrt(np.maximum(VAR2,0)), levels=20, cmap='Blues')
axes[1].scatter(X2_train[:,0], X2_train[:,1], c='k', s=40, zorder=5)
axes[1].set_title('GP Posterior Std Dev (2D)', fontsize=10)
axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
savefig("fig_gp_2d.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 – Conditional Gaussian example (Example 18.1 style)
# ═══════════════════════════════════════════════════════════════════════════════
mu_joint = np.array([0.0, 0.0])
Sigma_joint = np.array([[2.0, 1.0], [1.0, 1.5]])
# Marginals: a ~ N(0,2), b ~ N(0,1.5)
# Conditional: a|b=1 ~ N(mu_a|b, Sigma_a|b)
b_val = 1.0
A = Sigma_joint[0,0]; B = Sigma_joint[1,1]; C = Sigma_joint[0,1]
mu_cond = mu_joint[0] + C / B * (b_val - mu_joint[1])
sigma_cond = A - C**2 / B

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
xa = np.linspace(-4, 4, 300)
# Marginal of a
from scipy.stats import norm
axes[0].plot(xa, norm.pdf(xa, 0, np.sqrt(A)), 'b-', linewidth=2, label='marginal $a$')
axes[0].fill_between(xa, norm.pdf(xa, 0, np.sqrt(A)), alpha=0.2, color='blue')
axes[0].set_xlabel('$a$'); axes[0].set_ylabel('density')
axes[0].set_title('Marginal Distribution of $a$', fontsize=10)
axes[0].legend()

# Conditional
axes[1].plot(xa, norm.pdf(xa, mu_cond, np.sqrt(sigma_cond)), 'r-', linewidth=2,
             label=f'$a|b={b_val}$')
axes[1].fill_between(xa, norm.pdf(xa, mu_cond, np.sqrt(sigma_cond)), alpha=0.2, color='red')
axes[1].set_xlabel('$a$'); axes[1].set_ylabel('density')
axes[1].set_title(f'Conditional Distribution $a\\mid b={b_val}$', fontsize=10)
axes[1].legend()
plt.tight_layout()
savefig("fig_conditional_gaussian.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 – Algorithm 18.1 / 18.2 illustration: kernel matrix + GP structure
# ═══════════════════════════════════════════════════════════════════════════════
X_demo = np.array([-2., -1., 0., 1., 2.])
K_demo = np.array([[sq_exp_kernel(X_demo[i], X_demo[j])
                    for j in range(5)] for i in range(5)])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
im = axes[0].imshow(K_demo, cmap='Blues', vmin=0, vmax=1)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels([f'$x_{i+1}$' for i in range(5)])
axes[0].set_yticks(range(5)); axes[0].set_yticklabels([f'$x_{i+1}$' for i in range(5)])
axes[0].set_title('Kernel (Covariance) Matrix $K(X,X)$', fontsize=10)
plt.colorbar(im, ax=axes[0])
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{K_demo[i,j]:.2f}', ha='center', va='center', fontsize=7)

# GP prior samples
rng3 = np.random.default_rng(5)
xp = np.linspace(-4, 4, 200)
K_prior = np.array([[sq_exp_kernel(xp[i], xp[j]) for j in range(len(xp))] for i in range(len(xp))])
K_prior += 1e-8*np.eye(len(xp))
L_prior = np.linalg.cholesky(K_prior)
colors5 = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
for c in colors5:
    s = L_prior @ rng3.standard_normal(len(xp))
    axes[1].plot(xp, s, color=c, linewidth=0.9, alpha=0.8)
axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$f(x)$')
axes[1].set_title('GP Prior Samples (Squared Exp Kernel)', fontsize=10)
plt.tight_layout()
savefig("fig_kernel_matrix.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 – Exercise 18.3: sin(x)/(x^2+1) with gradient GP
# ═══════════════════════════════════════════════════════════════════════════════
f_ex3 = lambda x: np.sin(x) / (x**2 + 1)
df_ex3 = lambda x: ((x**2+1)*np.cos(x) - 2*x*np.sin(x)) / (x**2+1)**2

X_ex3 = np.array([-5., -2.5, 0., 2.5, 5.])
y_ex3 = f_ex3(X_ex3)
dy_ex3 = df_ex3(X_ex3)

X_star3 = np.linspace(-5, 5, 300)
ell3 = 1.0

# Without gradient
mu_e3, var_e3 = gp_predict(X_ex3, y_ex3, X_star3, lambda a, b: kff(a, b, ell3))
std_e3 = np.sqrt(var_e3)

# With gradient
m3 = len(X_ex3)
n3 = len(X_star3)
K_ff3 = np.array([[kff(X_ex3[i],X_ex3[j],ell3) for j in range(m3)] for i in range(m3)])
K_vf3 = np.array([[kvf(X_ex3[i],X_ex3[j],ell3) for j in range(m3)] for i in range(m3)])
K_vv3 = np.array([[kvv(X_ex3[i],X_ex3[j],ell3) for j in range(m3)] for i in range(m3)])
K_tr3 = np.block([[K_ff3, K_vf3.T],[K_vf3, K_vv3]]) + 1e-8*np.eye(2*m3)

Ksf3 = np.array([[kff(X_star3[i],X_ex3[j],ell3) for j in range(m3)] for i in range(n3)])
Ksv3 = np.array([[kfv(X_star3[i],X_ex3[j],ell3) for j in range(m3)] for i in range(n3)])
Ks3 = np.hstack([Ksf3, Ksv3])
Kss3 = np.array([kff(xi, xi, ell3) for xi in X_star3])

yc3 = np.concatenate([y_ex3, dy_ex3])
Ki3 = np.linalg.inv(K_tr3)
mu_g3 = Ks3 @ Ki3 @ yc3
var_g3 = Kss3 - np.sum((Ks3 @ Ki3) * Ks3, axis=1)
std_g3 = np.sqrt(np.maximum(var_g3, 0))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(X_star3, mu_e3-1.96*std_e3, mu_e3+1.96*std_e3,
                alpha=0.35, color='steelblue', label='95% CI without derivative')
ax.fill_between(X_star3, mu_g3-1.96*std_g3, mu_g3+1.96*std_g3,
                alpha=0.4, color='salmon', label='95% CI with derivative')
ax.plot(X_star3, f_ex3(X_star3), 'k-', lw=1.3, label='true function')
ax.plot(X_star3, mu_e3, '-', color='steelblue', lw=1.2, label='mean w/o deriv.')
ax.plot(X_star3, mu_g3, '-', color='red', lw=1.2, label='mean w/ deriv.')
ax.scatter(X_ex3, y_ex3, c='k', s=40, zorder=5, label='fit points')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(r'Ex. 18.3: $f(x)=\sin(x)/(x^2+1)$ with/without derivative info', fontsize=9)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
savefig("fig_exercise3.pdf")


print("\nAll figures generated successfully.")
