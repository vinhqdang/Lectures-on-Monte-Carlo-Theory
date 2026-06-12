"""
gen_figures.py  —  Chapter 21: Uncertainty Propagation
Generates all figures needed for chapter21_slides.tex.
Requires: numpy, scipy, matplotlib (conda env py313)
Run: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, integrate, linalg
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

import os
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.6,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Monte Carlo convergence — sample mean/variance vs m
# ─────────────────────────────────────────────────────────────────────────────
def fig_mc_convergence():
    np.random.seed(42)
    # f(z) = sin(z), z ~ N(0,1); true mean = 0, true var ~ ?
    # Use f(z) = exp(-z^2/2), z ~ Uniform(-3,3)
    # Actually use f(z) = z^2 + z, z ~ N(0,1): true mean=1, true var=3
    true_mu = 1.0   # E[z^2 + z] = 1
    true_var = 3.0  # Var[z^2+z] = Var[z^2]+Var[z] = 2+1

    ms = np.logspace(1, 4, 50).astype(int)
    mu_hats = []
    var_hats = []
    for m in ms:
        z = np.random.randn(m)
        fz = z**2 + z
        mu_hats.append(np.mean(fz))
        var_hats.append(np.mean(fz**2) - np.mean(fz)**2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].semilogx(ms, mu_hats, 'steelblue', label=r'$\hat{\mu}$')
    axes[0].axhline(true_mu, color='black', linestyle='--', label='true')
    axes[0].set_xlabel('Number of samples $m$')
    axes[0].set_ylabel(r'$\hat{\mu}$')
    axes[0].set_title('Sample Mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(ms, var_hats, 'coral', label=r'$\hat{\nu}$')
    axes[1].axhline(true_var, color='black', linestyle='--', label='true')
    axes[1].set_xlabel('Number of samples $m$')
    axes[1].set_ylabel(r'$\hat{\nu}$')
    axes[1].set_title('Sample Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r'Monte Carlo: $f(z)=z^2+z$, $z\sim\mathcal{N}(0,1)$', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'mc_convergence.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved mc_convergence.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Taylor approximation — Example 21.1
# f(x,z) = sin(x+z1)*cos(x+z2), z1~N(0,0.1), z2~N(0,0.2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_taylor_approx():
    xs = np.linspace(-np.pi, np.pi, 400)

    # True mean/variance by Monte Carlo
    np.random.seed(0)
    m = 50000
    z1 = np.random.randn(m) * np.sqrt(0.1)
    z2 = np.random.randn(m) * np.sqrt(0.2)

    true_mu = np.array([np.mean(np.sin(x + z1) * np.cos(x + z2)) for x in xs])
    true_var = np.array([np.mean((np.sin(x + z1) * np.cos(x + z2))**2)
                         - np.mean(np.sin(x + z1) * np.cos(x + z2))**2 for x in xs])

    # First-order Taylor: mu = f(mu_z) = sin(x)cos(x)
    #   nu = (df/dz1)^2 * nu1 + (df/dz2)^2 * nu2  at z=(0,0)
    mu1 = np.sin(xs) * np.cos(xs)
    # df/dz1|z=0 = cos(x)cos(x), df/dz2|z=0 = -sin(x)sin(x)
    nu1 = (np.cos(xs)**2)**2 * 0.1 + (np.sin(xs)**2)**2 * 0.2

    # Second-order Taylor:
    # mu += 0.5*(d2f/dz1^2*nu1 + d2f/dz2^2*nu2) at z=0
    # d2f/dz1^2 = -sin(x)cos(x), d2f/dz2^2 = -sin(x)cos(x)
    mu2 = mu1 + 0.5 * (-np.sin(xs) * np.cos(xs) * 0.1
                        - np.sin(xs) * np.cos(xs) * 0.2)
    # nu2 (second-order): nu1 + 0.5*sum_{ij}(d2f/dzidj)^2*ni*nj
    nu2 = (nu1
           + 0.5 * (np.sin(xs)**2 * np.cos(xs)**2 * 0.1**2)       # (d2f/dz1^2)^2 * nu1^2 / ... simplified
           + 0.5 * (np.sin(xs)**2 * np.cos(xs)**2 * 0.2**2)
           + 0.5 * 2 * (np.cos(xs)**2 * np.sin(xs)**2 * 0.1 * 0.2))
    # Using exact formula: nu += 0.5*sum_ij (d2f/dzi dzj)^2 nu_i nu_j
    # d2f/dz1dz2 = -cos(x+z1)sin(x+z2) -> at z=0: -cos(x)sin(x)
    nu2_correct = (np.cos(xs)**4 * 0.1
                   + np.sin(xs)**4 * 0.2
                   + 0.5 * (np.sin(xs)**2 * np.cos(xs)**2 * 0.1**2)
                   + 0.5 * (np.sin(xs)**2 * np.cos(xs)**2 * 0.2**2)
                   + 0.5 * 2 * np.cos(xs)**2 * np.sin(xs)**2 * 0.1 * 0.2)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(xs, true_mu, 'black', lw=2, label='true')
    axes[0].plot(xs, mu1, color='#1f77b4', lw=1.4, linestyle='--', label='1st-order Taylor')
    axes[0].plot(xs, mu2, color='#d62728', lw=1.4, linestyle='-.', label='2nd-order Taylor')
    axes[0].set_ylabel('mean')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, true_var, 'black', lw=2, label='true')
    axes[1].plot(xs, nu1, color='#1f77b4', lw=1.4, linestyle='--', label='1st-order Taylor')
    axes[1].plot(xs, nu2_correct, color='#d62728', lw=1.4, linestyle='-.', label='2nd-order Taylor')
    axes[1].set_ylabel('variance')
    axes[1].set_xlabel('$x$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r'Taylor Approx: $f(x,z)=\sin(x+z_1)\cos(x+z_2)$', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'taylor_approx.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved taylor_approx.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Orthogonal basis functions (Legendre, Laguerre, Hermite) — Fig 21.1
# ─────────────────────────────────────────────────────────────────────────────
def legendre_basis(i, z):
    """Legendre polynomial b_i(z) on [-1,1], b_1=1."""
    from scipy.special import legendre as sp_legendre
    if i == 1:
        return np.ones_like(z)
    # Standard Legendre P_{i-1}(z)
    return sp_legendre(i - 1)(z)


def laguerre_basis(i, z):
    """Laguerre polynomial b_i(z) on [0,inf), b_1=1."""
    from scipy.special import laguerre as sp_laguerre
    if i == 1:
        return np.ones_like(z)
    return sp_laguerre(i - 1)(z)


def hermite_basis(i, z):
    """Probabilist's Hermite polynomial b_i(z), b_1=1."""
    from scipy.special import hermitenorm
    if i == 1:
        return np.ones_like(z)
    return hermitenorm(i - 1)(z)


def fig_orthogonal_bases():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Legendre on [-1,1]
    z_leg = np.linspace(-1, 1, 300)
    for i in range(1, 7):
        axes[0].plot(z_leg, legendre_basis(i, z_leg), color=colors[i-1],
                     label=f'$b_{i}$')
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1.3, 1.3)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].set_xlabel('$z$')
    axes[0].set_title('Legendre')

    # Laguerre on [0,15]
    z_lag = np.linspace(0, 15, 400)
    for i in range(1, 7):
        vals = laguerre_basis(i, z_lag)
        axes[1].plot(z_lag, np.clip(vals, -22, 22), color=colors[i-1],
                     label=f'$b_{i}$')
    axes[1].set_xlim(0, 15)
    axes[1].set_ylim(-22, 22)
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_xlabel('$z$')
    axes[1].set_title('Laguerre')

    # Hermite on [-4,4]
    z_her = np.linspace(-4, 4, 400)
    for i in range(1, 7):
        vals = hermite_basis(i, z_her)
        axes[2].plot(z_her, np.clip(vals, -13, 13), color=colors[i-1],
                     label=f'$b_{i}$')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-13, 13)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].set_xlabel('$z$')
    axes[2].set_title('Hermite')

    # Shared legend on last axis
    axes[2].legend(loc='upper left', fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'orthogonal_bases.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved orthogonal_bases.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Polynomial chaos Example 21.2 — Hermite fit of unknown function
# f(x,z) = 1 - exp(-(x+z-1)^2) - 2*exp(-(x+z-3)^2), z ~ N(0,1)
# ─────────────────────────────────────────────────────────────────────────────
def f_ex2(x, z):
    return 1 - np.exp(-(x + z - 1)**2) - 2 * np.exp(-(x + z - 3)**2)


def poly_chaos_hermite_fit(x_val, z_samples, k=4):
    """Fit k Hermite basis polynomials to f evaluated at z_samples for given x."""
    from scipy.special import hermitenorm
    # Build Vandermonde-like matrix
    B = np.column_stack([hermitenorm(i)(z_samples) for i in range(k)])
    y = f_ex2(x_val, z_samples)
    theta, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    return theta


def hermite_mean_var(theta):
    """With orthonormal Hermite polynomials under N(0,1):
       mu = theta[0], var = sum_{i>=1} theta[i]^2 * norm_i
       For probabilist's Hermite: <He_m, He_n>_{N(0,1)} = m! * delta_{mn}
    """
    # Coefficients from least-squares are NOT normalized
    # With weight N(0,1): integral He_i^2(z) * N(0,1) dz = i!
    norms = np.array([factorial(i) for i in range(len(theta))])
    mu = theta[0]  # since b1=He_0=1, integral b1^2 p dz = 0! = 1 -> mu=theta[0]
    var = sum(theta[i]**2 * norms[i] for i in range(1, len(theta)))
    return mu, var


def fig_poly_chaos_hermite():
    np.random.seed(7)
    xs = np.linspace(-2, 6, 300)
    k = 4  # 3rd-order: basis He_0,...,He_3

    # True noise-free f(x,0)
    f_nf = f_ex2(xs, 0.0)

    # True expected value (by quadrature)
    true_mu = np.array([integrate.quad(lambda z: f_ex2(x, z) * stats.norm.pdf(z),
                                       -6, 6)[0] for x in xs])

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    sample_counts = [10, 30, 50]

    for ax, m in zip(axes, sample_counts):
        # Bootstrap to get CI
        n_boot = 200
        mu_boots = np.zeros((n_boot, len(xs)))
        for b in range(n_boot):
            z_s = np.random.randn(m)
            for j, x in enumerate(xs):
                theta = poly_chaos_hermite_fit(x, z_s, k=k)
                mu_b, _ = hermite_mean_var(theta)
                mu_boots[b, j] = mu_b

        mu_mean = mu_boots.mean(axis=0)
        mu_lo = np.percentile(mu_boots, 2.5, axis=0)
        mu_hi = np.percentile(mu_boots, 97.5, axis=0)

        ax.fill_between(xs, mu_lo, mu_hi, alpha=0.3, color='steelblue', label='95% interval')
        ax.plot(xs, mu_mean, color='steelblue', lw=1.5, label='mean')
        ax.plot(xs, f_nf, color='gray', lw=1, linestyle='--', label='noise-free')
        ax.plot(xs, true_mu, 'black', lw=1.5, label='exact')
        ax.set_ylabel(r'$\mathbb{E}[f|x]$')
        ax.set_ylim(-1.5, 1.5)
        ax.text(0.02, 0.85, f'{m} samples', transform=ax.transAxes, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('$x$')
    axes[0].legend(loc='upper right', fontsize=7)
    fig.suptitle('Polynomial Chaos (Hermite, 3rd order) — Example 21.2', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'poly_chaos_hermite.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved poly_chaos_hermite.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Legendre fit — Example 21.3
# f(z) = sin(pi*z), z ~ Uniform[-1,1]
# ─────────────────────────────────────────────────────────────────────────────
def fig_legendre_fit():
    from scipy.special import legendre as sp_legendre

    z_samples = np.array([-1.0, -0.2, 0.3, 0.7, 0.9])
    y_samples = np.sin(np.pi * z_samples)

    z_plot = np.linspace(-1, 1, 400)
    f_true = np.sin(np.pi * z_plot)

    colors_deg = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_plot, f_true, 'black', lw=2, label='true')
    ax.scatter(z_samples, y_samples, color='black', zorder=5, s=30)

    results = []
    for deg in range(1, 6):
        # Build Legendre Vandermonde
        B = np.column_stack([sp_legendre(j)(z_samples) for j in range(deg)])
        theta, _, _, _ = np.linalg.lstsq(B, y_samples, rcond=None)
        f_hat = np.column_stack([sp_legendre(j)(z_plot) for j in range(deg)]) @ theta
        # Mean: theta[0] (since Legendre P0 = 1)
        mu = theta[0]
        # Variance: sum_{i>=1} theta[i]^2 * integral P_{i}(z)^2 * (1/2) dz over [-1,1]
        # integral P_i^2 (1/2) dz = (1/2) * 2/(2i+1) = 1/(2i+1)
        var = sum(theta[j]**2 / (2*j + 1) for j in range(1, deg))
        results.append((deg, mu, var))
        ax.plot(z_plot, f_hat, color=colors_deg[deg-1], lw=1.3,
                label=f'$i={deg}$, $\\hat{{\\mu}}={mu:+.3f}$, $\\hat{{\\nu}}={var:.3f}$')

    ax.set_xlabel('$z$')
    ax.set_ylabel('$f(z)$')
    ax.set_title(r'Legendre Fit: $f(z)=\sin(\pi z)$, $z\sim\mathrm{Uniform}[-1,1]$')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'legendre_fit.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved legendre_fit.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Bayesian Monte Carlo vs Sample Mean — Example 21.6
# f(x,z) = sin(x+z1)*cos(x+z2), z1~N(0,1), z2~N(0,0.5)
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_kernel(z1, z2, w):
    """Gaussian kernel k(z1,z2) = exp(-0.5 * sum((z1-z2)^2/w^2))"""
    diff = z1 - z2
    return np.exp(-0.5 * np.sum((diff / w)**2))


def build_K(Z, w):
    n = len(Z)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = gaussian_kernel(Z[i], Z[j], w)
    return K


def bayesian_mc(Z, y, w, mu_z, Sigma_z):
    """Bayesian Monte Carlo with Gaussian kernel."""
    W = np.diag(w**2)
    K_mat = build_K(Z, w)
    invK = np.linalg.inv(K_mat + 1e-8 * np.eye(len(Z)))

    # q_i = |W^{-1} Sigma_z + I|^{-1/2} * exp(-0.5*(mu_z-z_i)^T (Sigma_z+W)^{-1} (mu_z-z_i))
    WinvSz_I = np.linalg.inv(W) @ Sigma_z + np.eye(W.shape[0])
    det_factor = np.linalg.det(WinvSz_I)

    q = np.zeros(len(Z))
    SzW_inv = np.linalg.inv(Sigma_z + W)
    for i, z in enumerate(Z):
        diff = mu_z - z
        q[i] = np.exp(-0.5 * diff @ SzW_inv @ diff)
    q *= det_factor**(-0.5)

    mu_est = q @ invK @ y
    nu_est = np.linalg.det(2 * np.linalg.inv(W) @ Sigma_z + np.eye(W.shape[0]))**(-0.5) \
             - q @ invK @ q
    return mu_est, nu_est


def fig_bayesian_mc():
    np.random.seed(42)
    xs = np.linspace(-2.5, 2.5, 200)
    mu_z = np.array([0.0, 0.0])
    Sigma_z = np.diag([1.0, 0.5])
    w = np.array([1.0, 1.0])
    m_per_x = 10

    # True expected value by Monte Carlo
    n_mc = 30000
    z1_mc = np.random.randn(n_mc)
    z2_mc = np.random.randn(n_mc) * np.sqrt(0.5)
    true_mu = np.array([np.mean(np.sin(x + z1_mc) * np.cos(x + z2_mc)) for x in xs])

    bmc_mu = np.zeros(len(xs))
    samp_mu = np.zeros(len(xs))

    for idx, x in enumerate(xs):
        Z_pts = np.random.multivariate_normal(mu_z, Sigma_z, m_per_x)
        y_pts = np.sin(x + Z_pts[:, 0]) * np.cos(x + Z_pts[:, 1])
        samp_mu[idx] = np.mean(y_pts)
        try:
            mu_b, _ = bayesian_mc(Z_pts, y_pts, w, mu_z, Sigma_z)
            bmc_mu[idx] = mu_b
        except Exception:
            bmc_mu[idx] = samp_mu[idx]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, true_mu, 'black', lw=2, label='true')
    ax.plot(xs, samp_mu, color='#d62728', lw=1, alpha=0.8, label='sample mean')
    ax.plot(xs, bmc_mu, color='steelblue', lw=1.5, label='Bayesian MC')
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$\mathbb{E}[f\,|\,x]$')
    ax.set_title(r'Bayesian MC vs Sample Mean ($m=10$ per point)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'bayesian_mc.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved bayesian_mc.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Method comparison summary — variance of mean estimate vs m
# ─────────────────────────────────────────────────────────────────────────────
def fig_method_comparison():
    """Show how different methods compare in terms of approximation quality."""
    np.random.seed(0)
    xs = np.linspace(-np.pi, np.pi, 300)

    # f(x,z) = sin(x+z1)cos(x+z2), z1~N(0,0.1), z2~N(0,0.2)
    n_mc = 20000
    z1 = np.random.randn(n_mc) * np.sqrt(0.1)
    z2 = np.random.randn(n_mc) * np.sqrt(0.2)
    true_mu = np.array([np.mean(np.sin(x + z1) * np.cos(x + z2)) for x in xs])

    # 1st-order Taylor
    mu1 = np.sin(xs) * np.cos(xs)

    # 2nd-order Taylor (from Ex 21.1)
    mu2 = mu1 + 0.5 * (-np.sin(xs) * np.cos(xs) * 0.1
                        - np.sin(xs) * np.cos(xs) * 0.2)

    # Monte Carlo (m=50)
    np.random.seed(5)
    z1_s = np.random.randn(50) * np.sqrt(0.1)
    z2_s = np.random.randn(50) * np.sqrt(0.2)
    mu_mc50 = np.array([np.mean(np.sin(x + z1_s) * np.cos(x + z2_s)) for x in xs])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(xs, true_mu, 'black', lw=2, label='True')
    ax.plot(xs, mu_mc50, color='coral', lw=1, alpha=0.8, label='MC ($m=50$)')
    ax.plot(xs, mu1, color='#1f77b4', lw=1.5, linestyle='--', label='Taylor 1st')
    ax.plot(xs, mu2, color='#d62728', lw=1.5, linestyle='-.', label='Taylor 2nd')
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$\hat{\mu}(x)$')
    ax.set_title(r'Method Comparison: $\hat{\mu}$ for $f(x,z)=\sin(x+z_1)\cos(x+z_2)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'method_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved method_comparison.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Stieltjes algorithm illustration — truncated Gaussian (Ex 21.4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_stieltjes_fit():
    """Legendre polynomials via Stieltjes for truncated Gaussian on [2,5]."""
    from scipy.special import legendre as sp_legendre

    # Truncated Gaussian: N(3,1) on [2,5]
    a, b = 2.0, 5.0
    norm_const = stats.norm.cdf(b, 3, 1) - stats.norm.cdf(a, 3, 1)

    def p(z):
        return stats.norm.pdf(z, 3, 1) / norm_const

    z_samples = np.array([2.1, 2.5, 3.3, 3.9, 4.7])
    y_samples = np.sin(np.pi * z_samples)
    z_plot = np.linspace(a, b, 400)
    f_true = np.sin(np.pi * z_plot)

    # Build orthogonal basis via Stieltjes (3-term recurrence)
    def stieltjes_basis(n_basis, p_func, domain, n_quad=200):
        """Build n_basis orthogonal polynomials using Stieltjes algorithm."""
        zq = np.linspace(domain[0], domain[1], n_quad)
        wq = p_func(zq) * (domain[1] - domain[0]) / n_quad

        bases = [np.ones(n_quad)]  # b_1 = 1
        polys = [lambda z, _b=bases[0]: np.ones_like(z)]

        for _ in range(1, n_basis):
            b_curr = bases[-1]
            alpha = np.sum(zq * b_curr**2 * wq) / np.sum(b_curr**2 * wq)
            if len(bases) > 1:
                b_prev = bases[-2]
                beta = np.sum(b_curr**2 * wq) / np.sum(b_prev**2 * wq)
                b_next = (zq - alpha) * b_curr - beta * b_prev
            else:
                b_next = (zq - alpha) * b_curr
            bases.append(b_next)

        return bases, zq, wq

    colors_deg = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_plot, f_true, 'black', lw=2, label='true')
    ax.scatter(z_samples, y_samples, color='black', zorder=5, s=35)

    for deg in range(1, 6):
        bases_q, zq, wq = stieltjes_basis(deg, p, (a, b), n_quad=500)
        # Evaluate bases at sample points (interpolate)
        B_samp = np.zeros((len(z_samples), deg))
        B_plot = np.zeros((len(z_plot), deg))
        for j in range(deg):
            # Re-evaluate polynomial at sample points using recurrence
            bs_at_samples = np.ones(len(z_samples))
            bs_at_plot = np.ones(len(z_plot))
            prev_s = np.ones(len(z_samples))
            prev_p = np.ones(len(z_plot))
            prev2_s = np.zeros(len(z_samples))
            prev2_p = np.zeros(len(z_plot))
            # Rebuild recurrence for evaluation points
            bs_s_list = [np.ones(len(z_samples))]
            bs_pl_list = [np.ones(len(z_plot))]
            # Recompute alpha, beta from quadrature
            for ii in range(j):
                bc = bases_q[ii]
                alpha_i = np.sum(zq * bc**2 * wq) / np.sum(bc**2 * wq)
                if ii > 0:
                    bp = bases_q[ii-1]
                    beta_i = np.sum(bc**2 * wq) / np.sum(bp**2 * wq)
                    bs_s_list.append((z_samples - alpha_i) * bs_s_list[-1]
                                     - beta_i * bs_s_list[-2])
                    bs_pl_list.append((z_plot - alpha_i) * bs_pl_list[-1]
                                      - beta_i * bs_pl_list[-2])
                else:
                    bs_s_list.append((z_samples - alpha_i) * bs_s_list[-1])
                    bs_pl_list.append((z_plot - alpha_i) * bs_pl_list[-1])

            B_samp[:, j] = bs_s_list[j]
            B_plot[:, j] = bs_pl_list[j]

        theta, _, _, _ = np.linalg.lstsq(B_samp, y_samples, rcond=None)
        f_hat = B_plot @ theta
        mu = theta[0]
        var_val = sum(theta[jj]**2 * np.sum(bases_q[jj]**2 * wq)
                      for jj in range(1, deg))
        ax.plot(z_plot, f_hat, color=colors_deg[deg-1], lw=1.3,
                label=f'$i={deg}$, $\\hat{{\\mu}}={mu:+.3f}$, $\\hat{{\\nu}}={var_val:.3f}$')

    ax.set_xlabel('$z$')
    ax.set_ylabel('$f(z)$')
    ax.set_title(r'Stieltjes Fit: $f(z)=\sin(\pi z)$, Truncated Gaussian [2,5]')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(a, b)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'stieltjes_fit.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved stieltjes_fit.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Uncertainty propagation overview diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_overview():
    """Schematic showing how input uncertainty maps to output uncertainty."""
    np.random.seed(99)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Left: input distribution p(z)
    z = np.linspace(-3, 3, 300)
    axes[0].fill_between(z, stats.norm.pdf(z), alpha=0.4, color='steelblue')
    axes[0].plot(z, stats.norm.pdf(z), 'steelblue', lw=2)
    axes[0].set_xlabel('$z$')
    axes[0].set_ylabel('$p(z)$')
    axes[0].set_title('Input distribution $p(z)$')
    axes[0].set_yticks([])
    axes[0].grid(True, alpha=0.3)

    # Middle: function f(z) = z^3 / 5
    axes[1].plot(z, z**3 / 5, 'black', lw=2)
    axes[1].set_xlabel('$z$')
    axes[1].set_ylabel('$f(z)$')
    axes[1].set_title('Objective function $f(z)$')
    axes[1].grid(True, alpha=0.3)

    # Right: output distribution p(f(z))
    samples_z = np.random.randn(50000)
    samples_f = samples_z**3 / 5
    axes[2].hist(samples_f, bins=60, density=True, color='coral', alpha=0.7, edgecolor='none')
    axes[2].axvline(samples_f.mean(), color='black', lw=2, linestyle='--',
                    label=f'$\\hat{{\\mu}}={samples_f.mean():.2f}$')
    axes[2].set_xlabel('$f(z)$')
    axes[2].set_title('Output distribution')
    axes[2].legend(fontsize=9)
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Uncertainty Propagation: $p(z) \\to p(f(z))$', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'overview.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved overview.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Polynomial chaos for multivariate — basis function grid
# ─────────────────────────────────────────────────────────────────────────────
def fig_multivariate_basis():
    """Show multivariate basis b(z1,z2) = b_i(z1)*b_j(z2) as heatmaps."""
    from scipy.special import legendre as sp_legendre

    z = np.linspace(-1, 1, 60)
    Z1, Z2 = np.meshgrid(z, z)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    pairs = [(1,1),(2,1),(1,2),(2,2),(3,1),(1,3)]
    for ax, (i, j) in zip(axes.flat, pairs):
        bi = sp_legendre(i-1)(Z1)
        bj = sp_legendre(j-1)(Z2)
        bij = bi * bj
        im = ax.contourf(Z1, Z2, bij, levels=20, cmap='RdBu_r')
        ax.set_title(f'$b_{i}(z_1)\\cdot b_{j}(z_2)$', fontsize=9)
        ax.set_xlabel('$z_1$', fontsize=8)
        ax.set_ylabel('$z_2$', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Multivariate Polynomial Chaos Basis Functions (Legendre)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'multivariate_basis.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved multivariate_basis.pdf')


if __name__ == '__main__':
    print('Generating figures for Chapter 21: Uncertainty Propagation...')
    fig_overview()
    fig_mc_convergence()
    fig_taylor_approx()
    fig_orthogonal_bases()
    fig_poly_chaos_hermite()
    fig_legendre_fit()
    fig_stieltjes_fit()
    fig_bayesian_mc()
    fig_method_comparison()
    fig_multivariate_basis()
    print('All figures saved to', FIGDIR)
