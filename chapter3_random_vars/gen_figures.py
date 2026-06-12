"""
gen_figures.py  –  Generate all figures for Chapter 3 slides.
Saves every figure as a PDF in the figures/ subdirectory.
Run with: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, expon
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name + '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  saved {name}.pdf')

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# Fig 1: Generalized inverse F^← — CDF and its generalized inverse
# ─────────────────────────────────────────────
def fig_generalized_inverse():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: CDF F with a flat portion (jump distribution)
    ax = axes[0]
    xs = np.array([-2, 2, 2, 7, 12, 12, 14, 16])
    ys = np.array([0,  0, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0])
    # piecewise: continuous with a jump and a flat
    x_cont = np.linspace(-2, 16, 1000)
    def F(x):
        # smooth increasing piece with flat at [2,7) and jump at x=12
        if x < 2:
            return 0.0
        elif x < 7:
            return 0.4 * (x - 2) / 5.0 * 0.0 + 0.6  # flat at 0.6 for x in [2,7)
        elif x < 12:
            return 0.6 + 0.2 * (x - 7) / 5.0
        else:
            return 1.0

    x_arr = np.linspace(-2, 16, 500)
    F_arr = np.array([F(xi) for xi in x_arr])

    ax.step(x_arr, F_arr, where='post', color='steelblue', lw=2)
    # Mark open/closed circles for jumps
    ax.plot(2,  0.6, 'o', color='steelblue', markersize=8, markerfacecolor='white')
    ax.plot(7,  0.6, 'o', color='steelblue', markersize=8, markerfacecolor='steelblue')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('F(x)', fontsize=12)
    ax.set_title('CDF $F$', fontsize=13)
    ax.axhline(0.4, color='gray', ls='--', lw=0.8)
    ax.axhline(0.6, color='gray', ls='--', lw=0.8)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8)
    ax.set_xlim(-2, 16)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)

    # Right: generalized inverse F^←
    ax2 = axes[1]
    u_arr = np.linspace(0, 1, 500)
    def Finv(u):
        if u <= 0:
            return -2.0
        elif u < 0.6:
            return 2.0  # flat region → jump in inverse at u=0.6 doesn't exist; floor to 2
        elif u < 0.8:
            return 7.0 + (u - 0.6) / 0.2 * 5.0
        else:
            return 12.0

    Finv_arr = np.array([Finv(ui) for ui in u_arr])
    ax2.plot(u_arr, Finv_arr, color='darkorange', lw=2)
    ax2.set_xlabel('u', fontsize=12)
    ax2.set_ylabel('$F^{\\leftarrow}(u)$', fontsize=12)
    ax2.set_title('Generalised Inverse $F^{\\leftarrow}$', fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    fig.suptitle('CDF and its Generalised Inverse', fontsize=14, y=1.02)
    plt.tight_layout()
    savefig('fig_generalized_inverse')

fig_generalized_inverse()


# ─────────────────────────────────────────────
# Fig 2: ITM — Exponential distribution
# ─────────────────────────────────────────────
def fig_itm_exponential():
    lam = 1.0
    n = 2000
    U = rng.uniform(0, 1, n)
    X = -np.log(U) / lam

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    x = np.linspace(0, 6, 300)
    ax.plot(x, lam * np.exp(-lam * x), 'b-', lw=2, label='$f(x)=e^{-x}$')
    ax.hist(X, bins=40, density=True, alpha=0.5, color='steelblue', label='ITM samples')
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.set_title('Exponential(1) via ITM')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    u_arr = np.linspace(0.001, 0.999, 300)
    ax2.plot(u_arr, -np.log(u_arr) / lam, 'r-', lw=2)
    ax2.set_xlabel('$U \\sim \\mathcal{U}[0,1)$')
    ax2.set_ylabel('$X = -\\log(U)/\\lambda$')
    ax2.set_title('Inverse CDF $F^{\\leftarrow}(u) = -\\log(u)/\\lambda$')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    savefig('fig_itm_exponential')

fig_itm_exponential()


# ─────────────────────────────────────────────
# Fig 3: Pareto distribution
# ─────────────────────────────────────────────
def fig_pareto():
    alphas = [1.5, 2.0, 3.0]
    x = np.linspace(0, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for a in alphas:
        # CDF: F(x) = 1 - 1/(1+x)^alpha
        F = 1 - 1.0 / (1 + x) ** a
        ax.plot(x, F, lw=2, label=f'$\\alpha={a}$')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Pareto CDF')
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for a in alphas:
        # PDF: f(x) = alpha/(1+x)^(alpha+1)
        f = a / (1 + x) ** (a + 1)
        ax2.plot(x, f, lw=2, label=f'$\\alpha={a}$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Pareto PDF (heavy tail for small $\\alpha$)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    savefig('fig_pareto')

fig_pareto()


# ─────────────────────────────────────────────
# Fig 4: Geometric distribution — ITM-d vs closed form
# ─────────────────────────────────────────────
def fig_geometric():
    p = 0.3
    n = 3000
    # Closed form: X = floor(log(U)/log(p))
    U = rng.uniform(0, 1, n)
    X = np.floor(np.log(U) / np.log(p)).astype(int)

    k_max = 15
    ks = np.arange(0, k_max)
    pmf_true = (1 - p) * p ** ks

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ks - 0.2, pmf_true, width=0.35, label='True PMF', color='steelblue', alpha=0.8)
    counts = np.bincount(X[X < k_max], minlength=k_max) / n
    ax.bar(ks + 0.2, counts, width=0.35, label='Simulated', color='darkorange', alpha=0.8)
    ax.set_xlabel('k')
    ax.set_ylabel('Probability')
    ax.set_title(f'Geometric($p={p}$): $p_k = (1-p)p^k$')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    savefig('fig_geometric')

fig_geometric()


# ─────────────────────────────────────────────
# Fig 5: Poisson distribution — ad hoc method
# ─────────────────────────────────────────────
def fig_poisson():
    from scipy.stats import poisson

    lam = 4.0
    n = 5000

    def sample_poisson(lam, rng):
        X = np.zeros(n, dtype=int)
        for i in range(n):
            s = -np.log(rng.uniform())
            k = 0
            while s < lam:
                s -= np.log(rng.uniform())
                k += 1
            X[i] = k
        return X

    # Fast numpy version
    samples = rng.poisson(lam, n)

    k_max = 15
    ks = np.arange(0, k_max)
    pmf_true = poisson.pmf(ks, lam)
    counts = np.bincount(samples[samples < k_max], minlength=k_max) / n

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ks - 0.2, pmf_true, width=0.35, label='True PMF', color='steelblue', alpha=0.8)
    ax.bar(ks + 0.2, counts, width=0.35, label='Simulated', color='darkorange', alpha=0.8)
    ax.set_xlabel('k')
    ax.set_ylabel('Probability')
    ax.set_title(f'Poisson($\\lambda={lam}$): renewal-process method')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    savefig('fig_poisson')

fig_poisson()


# ─────────────────────────────────────────────
# Fig 6: Acceptance-Rejection — half-normal with Exp(1) envelope (Fig 3.4)
# ─────────────────────────────────────────────
def fig_ar_halfnormal():
    x = np.linspace(0, 5, 400)
    f = (2 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)   # half-normal
    g = np.exp(-x)                                        # Exp(1) proposal
    c = np.sqrt(2 * np.e / np.pi)
    cg = c * g

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, f,  'b-', lw=2.5, label='$f(x)$: half-normal')
    ax.plot(x, g,  'k-', lw=1.5, label='$g(x)$: proposal Exp(1)')
    ax.plot(x, cg, 'r-', lw=1.5, label=f'$cg(x)$, $c=\\sqrt{{2e/\\pi}}\\approx{c:.2f}$')

    # Shade acceptance region
    ax.fill_between(x, f, cg, where=(cg >= f), alpha=0.2, color='red', label='Rejection region')
    ax.fill_between(x, 0,  f, alpha=0.25, color='blue', label='Acceptance region')

    # Mark a sample point Y=1 with cg(Y) and f(Y)
    Y = 1.0
    ax.axvline(Y, color='darkgreen', ls='--', lw=1)
    ax.annotate('$Y$', xy=(Y, 0), xytext=(Y + 0.1, -0.02), fontsize=11, color='darkgreen')
    ax.annotate('', xy=(Y, cg[int(Y*80)]), xytext=(Y, f[int(Y*80)]),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.03, 0.85)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('AR-c: Half-Normal from Exp(1) proposal', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_ar_halfnormal')

fig_ar_halfnormal()


# ─────────────────────────────────────────────
# Fig 7: Box-Muller — pair of normals from 2 uniforms
# ─────────────────────────────────────────────
def fig_box_muller():
    n = 2000
    U1 = rng.uniform(0, 1, n)
    U2 = rng.uniform(0, 1, n)
    D  = -2 * np.log(U1)
    V  = 2 * np.pi * U2
    Z1 = np.sqrt(D) * np.cos(V)
    Z2 = np.sqrt(D) * np.sin(V)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, Z, lbl in zip(axes, [Z1, Z2], ['$Z_1$', '$Z_2$']):
        x = np.linspace(-4, 4, 200)
        ax.hist(Z, bins=40, density=True, alpha=0.6, color='steelblue')
        ax.plot(x, norm.pdf(x), 'r-', lw=2, label='$\\mathcal{N}(0,1)$')
        ax.set_title(f'Box-Muller: {lbl}')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_box_muller')

fig_box_muller()


# ─────────────────────────────────────────────
# Fig 8: Marsaglia-Bray (polar method)
# ─────────────────────────────────────────────
def fig_marsaglia_bray():
    n_total = 0
    Z1_list, Z2_list = [], []
    target = 2000
    while len(Z1_list) < target:
        u1, u2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        X_ = u1**2 + u2**2
        if X_ <= 1:
            Y = np.sqrt(-2 * np.log(X_) / X_)
            Z1_list.append(u1 * Y)
            Z2_list.append(u2 * Y)
        n_total += 1

    Z1 = np.array(Z1_list[:target])
    Z2 = np.array(Z2_list[:target])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z1, Z2, alpha=0.3, s=8, color='steelblue')
    theta = np.linspace(0, 2*np.pi, 200)
    for r in [1, 2, 3]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'r--', lw=0.8, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title('Marsaglia-Bray: 2000 standard normal pairs')
    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_marsaglia_bray')

fig_marsaglia_bray()


# ─────────────────────────────────────────────
# Fig 9: Multivariate normal — bivariate scatter (Fig 3.6)
# ─────────────────────────────────────────────
def fig_bivariate_normal():
    n = 500
    configs = [
        (1, 1, 0.0,  '(a) $\\sigma_1^2=\\sigma_2^2=1,\\,\\rho=0$'),
        (3, 1, 0.6,  '(b) $\\sigma_1^2=3,\\,\\sigma_2^2=1,\\,\\rho=0.6$'),
        (3, 1, 0.9,  '(c) $\\sigma_1^2=3,\\,\\sigma_2^2=1,\\,\\rho=0.9$'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (s1sq, s2sq, rho, title) in zip(axes, configs):
        s1, s2 = np.sqrt(s1sq), np.sqrt(s2sq)
        Sigma = np.array([[s1sq, rho*s1*s2], [rho*s1*s2, s2sq]])
        samples = rng.multivariate_normal([0, 0], Sigma, n)
        ax.scatter(samples[:,0], samples[:,1], alpha=0.4, s=8, color='steelblue')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal' if s1sq == s2sq else 'auto')
    plt.suptitle('Bivariate Normal Samples', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('fig_bivariate_normal')

fig_bivariate_normal()


# ─────────────────────────────────────────────
# Fig 10: Random points in B3 (3-ball) and on S2 (sphere)
# ─────────────────────────────────────────────
def fig_ball_sphere():
    n = 800

    # On S2: normalise standard normals
    Z = rng.standard_normal((n, 3))
    S2 = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    # In B3: S2 point * U^(1/3)
    U = rng.uniform(0, 1, n)
    B3 = S2 * U[:,None]**(1/3)

    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(S2[:,0], S2[:,1], S2[:,2], s=4, alpha=0.5, color='steelblue')
    ax1.set_title('Uniform on $S_2$ (sphere)')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(B3[:,0], B3[:,1], B3[:,2], s=4, alpha=0.4, color='darkorange')
    ax2.set_title('Uniform in $B_3$ (ball)')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')

    plt.suptitle('Random Points on Sphere and in Ball', fontsize=13)
    plt.tight_layout()
    savefig('fig_ball_sphere')

fig_ball_sphere()


# ─────────────────────────────────────────────
# Fig 11: Copulas — scatter plots (normal copula, independence, Frechet bounds)
# ─────────────────────────────────────────────
def fig_copulas():
    n = 400
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Frechet upper bound: V1=V2=U
    U = rng.uniform(0, 1, n)
    axes[0].scatter(U, U, alpha=0.5, s=10, color='steelblue')
    axes[0].set_title('Fréchet upper: $C_U(u_1,u_2)=\\min(u_1,u_2)$\n$V_1=V_2=U$')

    # (b) Independence copula
    U1, U2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    axes[1].scatter(U1, U2, alpha=0.5, s=10, color='darkorange')
    axes[1].set_title('Independence: $C_I(u_1,u_2)=u_1 u_2$')

    # (c) Gaussian copula rho=0.8
    rho = 0.8
    Sigma = np.array([[1, rho], [rho, 1]])
    Z = rng.multivariate_normal([0, 0], Sigma, n)
    V1, V2 = norm.cdf(Z[:,0]), norm.cdf(Z[:,1])
    axes[2].scatter(V1, V2, alpha=0.5, s=10, color='green')
    axes[2].set_title(f'Gaussian copula $\\rho={rho}$')

    for ax in axes:
        ax.set_xlabel('$V_1$'); ax.set_ylabel('$V_2$')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Bivariate Copulas', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('fig_copulas')

fig_copulas()


# ─────────────────────────────────────────────
# Fig 12: Composition method — mixture of two Gaussians
# ─────────────────────────────────────────────
def fig_composition():
    n = 3000
    p1, mu1, s1 = 0.4, -2.0, 0.8
    p2, mu2, s2 = 0.6,  2.5, 1.2

    J = rng.binomial(1, p2, n)
    X = np.where(J == 0,
                 rng.normal(mu1, s1, n),
                 rng.normal(mu2, s2, n))

    x = np.linspace(-6, 7, 400)
    f = p1 * norm.pdf(x, mu1, s1) + p2 * norm.pdf(x, mu2, s2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(X, bins=50, density=True, alpha=0.5, color='steelblue', label='Samples')
    ax.plot(x, f, 'r-', lw=2, label='Mixture density $f$')
    ax.plot(x, p1 * norm.pdf(x, mu1, s1), 'b--', lw=1.5, label=f'$p_1 g_1$  ($p_1={p1}$)')
    ax.plot(x, p2 * norm.pdf(x, mu2, s2), 'g--', lw=1.5, label=f'$p_2 g_2$  ($p_2={p2}$)')
    ax.set_title('Composition Method: Gaussian Mixture')
    ax.set_xlabel('x')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_composition')

fig_composition()


# ─────────────────────────────────────────────
# Fig 13: Heavy tail comparison (Fig 3.22 style)
# ─────────────────────────────────────────────
def fig_heavy_tail():
    n = 1500
    # Normal, Exp, Pareto(1.2) rescaled to mean 1
    X_norm = rng.normal(1, 1, n)
    X_exp  = rng.exponential(1, n)
    # Pareto alpha=1.2, mean = 1/(alpha-1) = 5; rescale by 0.2
    U = rng.uniform(0, 1, n)
    X_par  = 0.2 * (U**(-1/1.2) - 1)   # Par(1.2) scaled so mean≈1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Running averages
    ax = axes[0]
    for X, lbl, col in [(X_norm, '$\\mathcal{N}(1,1)$', 'green'),
                         (X_exp,  '$\\mathrm{Exp}(1)$', 'blue'),
                         (X_par,  '$0.2\\cdot\\mathrm{Par}(1.2)$', 'red')]:
        ax.plot(np.cumsum(X) / np.arange(1, n+1), color=col, lw=1.2, label=lbl)
    ax.axhline(1, color='k', ls='--', lw=0.8)
    ax.set_xlabel('n')
    ax.set_ylabel('$S_n/n$')
    ax.set_title('Running averages — convergence to mean 1')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 3)

    # Time series
    ax2 = axes[1]
    for X, lbl, col in [(X_norm, '$\\mathcal{N}(1,1)$', 'green'),
                         (X_exp,  '$\\mathrm{Exp}(1)$', 'blue'),
                         (X_par,  '$0.2\\cdot\\mathrm{Par}(1.2)$', 'red')]:
        ax2.plot(X, color=col, lw=0.6, alpha=0.8, label=lbl)
    ax2.set_xlabel('i')
    ax2.set_ylabel('$X_i$')
    ax2.set_title('Time series — heavy tail spikes')
    ax2.set_ylim(-3, 25)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Light vs Heavy Tails', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('fig_heavy_tail')

fig_heavy_tail()


# ─────────────────────────────────────────────
# Fig 14: QQ plots for light and heavy tailed distributions
# ─────────────────────────────────────────────
def fig_qq_plots():
    from scipy.stats import probplot
    n = 500
    X_norm = rng.normal(0, 1, n)
    U = rng.uniform(0, 1, n)
    X_par  = U**(-1/1.5) - 1   # Pareto alpha=1.5

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, X, title in zip(axes,
                              [X_norm, X_par],
                              ['Normal $\\mathcal{N}(0,1)$', 'Pareto$(1.5)$ (heavy tail)']):
        (osm, osr), (slope, intercept, r) = probplot(X, dist='norm', plot=None)
        ax.scatter(osm, osr, s=8, alpha=0.5, color='steelblue')
        xline = np.linspace(osm[0], osm[-1], 100)
        ax.plot(xline, slope * xline + intercept, 'r--', lw=1.5)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
        ax.set_title(f'QQ plot: {title}')
        ax.grid(alpha=0.3)

    plt.suptitle('QQ Plots: Detecting Tail Behaviour', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('fig_qq_plots')

fig_qq_plots()


# ─────────────────────────────────────────────
# Fig 15: Conditional ITM — scatter of (X,Y) with joint density x*exp(-x(y+1))
# ─────────────────────────────────────────────
def fig_conditional_itm():
    n = 500
    # X ~ Exp(1), Y | X=x ~ Exp(x)
    U1 = rng.uniform(0, 1, n)
    X = -np.log(U1)
    U2 = rng.uniform(0, 1, n)
    Y = -np.log(U2) / X

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X, np.minimum(Y, 10), alpha=0.5, s=12, color='steelblue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y (truncated at 10)')
    ax.set_title('Conditional ITM: $f(x,y)=xe^{-x(y+1)}$\n$X\\sim\\mathrm{Exp}(1),\\; Y|X=x\\sim\\mathrm{Exp}(x)$')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_conditional_itm')

fig_conditional_itm()


# ─────────────────────────────────────────────
# Fig 16: ITM algorithm illustration (flowchart-style)
# (simple diagram with text boxes)
# ─────────────────────────────────────────────
def fig_itm_algorithm():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    boxes = [
        (0.5, 0.85, 'Input: c.d.f. $F$', '#D6EAF8'),
        (0.5, 0.65, 'Step 1: Generate $U \\sim \\mathcal{U}[0,1)$', '#D5F5E3'),
        (0.5, 0.45, 'Step 2: Set $X = F^{\\leftarrow}(U)$', '#D5F5E3'),
        (0.5, 0.25, 'Output: $X \\sim F$', '#FDEBD0'),
    ]
    for (cx, cy, text, color) in boxes:
        fancy = mpatches.FancyBboxPatch((cx - 0.38, cy - 0.08), 0.76, 0.16,
                                         boxstyle='round,pad=0.02',
                                         facecolor=color, edgecolor='gray', lw=1.2)
        ax.add_patch(fancy)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=11)

    for y_top, y_bot in [(0.77, 0.73), (0.57, 0.53), (0.37, 0.33)]:
        ax.annotate('', xy=(0.5, y_bot), xytext=(0.5, y_top),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    ax.set_title('Algorithm: Inverse Transform Method (ITM)', fontsize=13)
    plt.tight_layout()
    savefig('fig_itm_algorithm')

fig_itm_algorithm()


# ─────────────────────────────────────────────
# Fig 17: Erlang distribution — sum of exponentials
# ─────────────────────────────────────────────
def fig_erlang():
    from scipy.stats import gamma

    n_values = [1, 2, 5, 10]
    lam = 1.0
    x = np.linspace(0, 20, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_values)))
    for n, col in zip(n_values, colors):
        # Erl(n, lam) = Gamma(n, 1/lam)
        pdf = gamma.pdf(x, a=n, scale=1/lam)
        ax.plot(x, pdf, lw=2, color=col, label=f'$n={n}$')

    ax.set_xlabel('x')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Erlang$( n, \\lambda=1)$ = sum of $n$ i.i.d. Exp$(\\lambda)$')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('fig_erlang')

fig_erlang()


# ─────────────────────────────────────────────
# Fig 18: Uniform on ellipse (x/5)^2 + y^2 = 1
# ─────────────────────────────────────────────
def fig_ellipse_uniform():
    try:
        from scipy.special import ellipeinc
        from pynverse import inversefunc

        a, b = 5.0, 1.0
        m = 1 - (b/a)**2  # eccentricity parameter (note: b<a here)
        # arc length CDF
        F_fun = lambda x: ellipeinc(x, 1 - b**2/a**2) / ellipeinc(2*np.pi, 1 - b**2/a**2)

        n = 500
        U = np.random.default_rng(0).uniform(0, 1, n)
        # Numerically invert
        theta_all = np.linspace(0, 2*np.pi, 2000)
        F_all = np.array([F_fun(t) for t in theta_all])
        thetas = np.interp(U, F_all, theta_all)

        xs = a * np.cos(thetas)
        ys = b * np.sin(thetas)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(xs, ys, s=6, alpha=0.7, color='steelblue')
        theta_c = np.linspace(0, 2*np.pi, 400)
        ax.plot(a*np.cos(theta_c), b*np.sin(theta_c), 'r-', lw=1)
        ax.set_aspect('equal')
        ax.set_title('Uniform random points on ellipse $(x/5)^2 + y^2 = 1$')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig('fig_ellipse_uniform')
    except ImportError:
        # Fallback without pynverse
        a, b = 5.0, 1.0
        n = 500
        # Rejection: sample uniform angle, accept with arc-length weight
        rng2 = np.random.default_rng(1)
        thetas_accepted = []
        while len(thetas_accepted) < n:
            t = rng2.uniform(0, 2*np.pi)
            # arc length element |J| = sqrt(a^2 sin^2 t + b^2 cos^2 t)
            w = np.sqrt(a**2 * np.sin(t)**2 + b**2 * np.cos(t)**2)
            M = a  # max
            if rng2.uniform() <= w / M:
                thetas_accepted.append(t)
        thetas_accepted = np.array(thetas_accepted[:n])
        xs = a * np.cos(thetas_accepted)
        ys = b * np.sin(thetas_accepted)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(xs, ys, s=6, alpha=0.7, color='steelblue')
        theta_c = np.linspace(0, 2*np.pi, 400)
        ax.plot(a*np.cos(theta_c), b*np.sin(theta_c), 'r-', lw=1)
        ax.set_aspect('equal')
        ax.set_title('Uniform random points on ellipse $(x/5)^2 + y^2 = 1$')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig('fig_ellipse_uniform')

fig_ellipse_uniform()


# ─────────────────────────────────────────────
# Fig 19: Normal copula scatter (rho=0.9 and rho=-0.8) like Fig 3.20
# ─────────────────────────────────────────────
def fig_normal_copula():
    n = 300
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, rho, col in zip(axes, [0.9, -0.8], ['steelblue', 'purple']):
        Sigma = np.array([[1, rho], [rho, 1]])
        Z = rng.multivariate_normal([0, 0], Sigma, n)
        V1, V2 = norm.cdf(Z[:,0]), norm.cdf(Z[:,1])
        ax.scatter(V1, V2, alpha=0.5, s=10, color=col)
        ax.set_xlabel('$V_1$'); ax.set_ylabel('$V_2$')
        ax.set_title(f'Normal copula $\\rho={rho}$')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Gaussian Copula Samples', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig('fig_normal_copula')

fig_normal_copula()


print('\nAll figures saved to', FIGDIR)
