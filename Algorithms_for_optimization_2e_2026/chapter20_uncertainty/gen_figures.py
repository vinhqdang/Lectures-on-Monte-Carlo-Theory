"""
gen_figures.py — Chapter 20: Optimization Under Uncertainty
Generates all figures needed for the Beamer slides.
Uses matplotlib with Agg backend (no display required).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE = '#4878CF'
LIGHT_BLUE = '#a8c8e8'
RED = '#e04040'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ─────────────────────────────────────────────────────────────────
# Figure 1: Noisy objective function — different noise levels affect optima
# (p448: f(x̃) = sin(2x̃)/x̃ with Gaussian noise)
# ─────────────────────────────────────────────────────────────────
def fig_noisy_objective():
    x = np.linspace(-3, 3, 800)

    def f(xtilde):
        xtilde = np.atleast_1d(xtilde)
        result = np.where(np.abs(xtilde) < 1e-10, 2.0, np.sin(2*xtilde)/xtilde)
        return result

    fig, ax = plt.subplots(figsize=(7, 3.5))
    variances = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(variances)))
    N = 200

    for i, v in enumerate(variances):
        if v == 0:
            y = f(x)
            lbl = r'$\nu=0$'
            ax.plot(x, y, color='black', lw=2, label=lbl, zorder=5)
        else:
            # Monte Carlo estimate of E[f(x+z)] with z ~ N(0,v)
            z_samples = np.random.RandomState(42).randn(N, 1) * np.sqrt(v)
            xtilde = x[np.newaxis, :] + z_samples  # (N, len(x))
            y_mc = f(xtilde).mean(axis=0)
            lbl = fr'$\nu={v}$'
            ax.plot(x, y_mc, color=colors[i], lw=1.5, label=lbl)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-10, 10)
    # Re-do with proper x range matching book
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right', ncol=1, fontsize=8)
    ax.set_title(r'Expected value of $f(\tilde{x}) = \sin(2\tilde{x})/\tilde{x}$ under Gaussian noise')
    fig.tight_layout()
    fig.savefig(f"{OUT}/noisy_objective.pdf")
    plt.close(fig)
    print("Saved noisy_objective.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 2: Minimax — f_mod(x) for various epsilon
# (p450: f(x,z) = f̃ where f̃=-x̃ if x̃≤0, x̃² otherwise, x̃=x+z, z∈[-ε,ε])
# ─────────────────────────────────────────────────────────────────
def fig_minimax():
    def f_tilde(xtilde):
        return np.where(xtilde <= 0, -xtilde, xtilde**2)

    x = np.linspace(-1, 1, 500)
    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    colors = ['black'] + list(plt.cm.Blues(np.linspace(0.25, 0.9, len(epsilons)-1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for eps, col in zip(epsilons, colors):
        # f_mod(x) = max_{z in [-eps,eps]} f(x+z)
        z_grid = np.linspace(-eps, eps, 200) if eps > 0 else np.array([0.0])
        f_mod = np.array([
            max(f_tilde(xi + z_grid)) for xi in x
        ])
        lbl = r'true' if eps == 0 else fr'$\epsilon={eps:.1f}$'
        lw = 2 if eps == 0 else 1.2
        ax.plot(x, f_mod, color=col, lw=lw, label=lbl)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylim(0, 4)
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.set_title(r'Minimax: $f_\mathrm{mod}(x) = \max_{z\in[-\epsilon,\epsilon]} f(x+z)$')
    fig.tight_layout()
    fig.savefig(f"{OUT}/minimax_fmod.pdf")
    plt.close(fig)
    print("Saved minimax_fmod.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 3: Rotated ellipse feasibility set (Example 20.2)
# (x1 cos z + x2 sin z)^2 + (x1 sin z - x2 cos z)^2/16 <= 1
# ─────────────────────────────────────────────────────────────────
def fig_rotated_ellipse():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    theta = np.linspace(0, 2*np.pi, 300)
    # Parametric ellipse: u along major, v along minor axis
    # Ellipse rotated by z: major axis at angle (pi/2 + z)
    # (x1 cos z + x2 sin z)^2 + (x1 sin z - x2 cos z)^2/16 = 1
    # Let u = x1 cos z + x2 sin z, v = x1 sin z - x2 cos z
    # u = cos t, v = 4 sin t  (a=1, b=4)
    # x1 = u cos z + v sin z, x2 = u sin z - v cos z
    z_vals = np.linspace(0, np.pi/2, 20)

    # Draw all ellipses in blue (union)
    union_patches = []
    ellipse_points_all = []
    for z in z_vals:
        u = np.cos(theta)
        v = 4 * np.sin(theta)
        x1 = u * np.cos(z) + v * np.sin(z)
        x2 = u * np.sin(z) - v * np.cos(z)
        ax.fill(x1, x2, alpha=0.15, color=LIGHT_BLUE)
        ellipse_points_all.append((x1, x2))

    # Draw the boundary ellipses at z=0 and z=pi/2
    for z, ls in [(0, '-'), (np.pi/2, '--')]:
        u = np.cos(theta)
        v = 4 * np.sin(theta)
        x1 = u * np.cos(z) + v * np.sin(z)
        x2 = u * np.sin(z) - v * np.cos(z)
        ax.plot(x1, x2, color=LIGHT_BLUE, lw=1.5, ls=ls)

    # Intersection (robust feasible set): sample points and check all z
    # For a point (x1,x2) to be feasible for all z in [0, pi/2]:
    # (x1 cos z + x2 sin z)^2 + (x1 sin z - x2 cos z)^2/16 <= 1 for all z
    N = 300
    pts_x1 = np.linspace(-5, 5, N)
    pts_x2 = np.linspace(-5, 5, N)
    X1, X2 = np.meshgrid(pts_x1, pts_x2)
    z_check = np.linspace(0, np.pi/2, 50)
    feasible = np.ones((N, N), dtype=bool)
    for z in z_check:
        u = X1 * np.cos(z) + X2 * np.sin(z)
        v = X1 * np.sin(z) - X2 * np.cos(z)
        cond = u**2 + v**2/16
        feasible &= (cond <= 1.0)

    ax.contour(X1, X2, feasible.astype(float), levels=[0.5], colors=[RED], linewidths=2)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Rotated ellipse feasible sets\n(blue=union, red=intersection/robust)')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rotated_ellipse.pdf")
    plt.close(fig)
    print("Saved rotated_ellipse.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 4: Info-gap decision theory (Example 20.3)
# f(x,z) = x̃² + 6e^{-x̃²}, x̃=x+z, x̃∈[-2,2], Z(ε)=[-ε,ε]
# ─────────────────────────────────────────────────────────────────
def fig_info_gap():
    def f(xtilde):
        return xtilde**2 + 6*np.exp(-xtilde**2)

    x = np.linspace(-3, 3, 500)
    epsilons = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    colors = ['black'] + list(plt.cm.Blues(np.linspace(0.25, 0.9, len(epsilons)-1)))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    critical = 8.0  # unconstrained
    critical2 = 5.0  # with constraint f(x,z)<=5

    for panel, f_crit in enumerate([critical, critical2]):
        ax = axes[panel]
        ax.axhline(f_crit, color=RED, lw=1.5, ls='-', xmin=0.05, xmax=0.6)
        ax.text(-2.5, f_crit+0.1, r'$\mathcal{Z}$', color=RED, fontsize=11)

        # noise-free
        ax.plot(x, f(x), color='black', lw=2, label='noise-free')

        # worst-case for each epsilon
        for eps, col in zip(epsilons[1:], colors[1:]):
            z_g = np.linspace(-eps, eps, 100)
            f_wc = np.array([
                np.max([f(xi + z) for z in z_g
                        if -2 <= xi+z <= 2] or [f(xi)])
                for xi in x
            ])
            ax.plot(x, f_wc, color=col, lw=1, alpha=0.8,
                    label=fr'$\epsilon={eps:.1f}$' if panel == 0 else None)

        # mark x*
        if panel == 0:
            xstar = x[np.argmin(f(x))]
        else:
            # more constrained solution shifts left
            xstar = x[np.argmin(np.abs(x + 1.5))]
        ax.plot(xstar, f(xstar), 'k.', ms=8)
        ax.annotate('$x^*$', (xstar, f(xstar)), textcoords='offset points',
                    xytext=(5, 5), fontsize=10)

        ax.set_ylabel('$y$')
        ax.set_ylim(2, 10)

    axes[0].set_title('Info-gap: unconstrained')
    axes[1].set_title(r'Info-gap: with constraint $f(x,z)\leq 5$')
    axes[1].set_xlabel('$x$')

    # legend only on top
    handles = [plt.Line2D([0],[0],color='black',lw=2,label='noise-free')] + \
              [plt.Line2D([0],[0],color=c,lw=1.2,label=fr'$\epsilon={e:.1f}$')
               for e, c in zip(epsilons[1:], colors[1:])]
    axes[0].legend(handles=handles, loc='upper right', fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{OUT}/info_gap.pdf")
    plt.close(fig)
    print("Saved info_gap.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 5: Expected value with Gaussian noise (Example 20.4)
# f(x̃) = sin(2x̃)/x̃, z ~ N(0,v)
# ─────────────────────────────────────────────────────────────────
def fig_expected_value_gaussian():
    def f(xtilde):
        xtilde = np.asarray(xtilde, dtype=float)
        return np.where(np.abs(xtilde) < 1e-9, 2.0, np.sin(2*xtilde)/xtilde)

    x = np.linspace(-10, 10, 600)
    variances = [0, 0.5, 1.0, 1.5, 2.0]
    colors = ['black'] + list(plt.cm.Blues(np.linspace(0.3, 0.9, 4)))
    labels = [r'$\nu=0$', r'$\nu=0.5$', r'$\nu=1.0$', r'$\nu=1.5$', r'$\nu=2.0$']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    rng = np.random.RandomState(0)
    N = 500
    for v, col, lbl in zip(variances, colors, labels):
        if v == 0:
            y = f(x)
            ax.plot(x, y, color=col, lw=2, label=lbl)
        else:
            z_s = rng.randn(N, 1) * np.sqrt(v)
            xtilde = x[np.newaxis, :] + z_s
            y_mc = f(xtilde).mean(axis=0)
            ax.plot(x, y_mc, color=col, lw=1.5, label=lbl)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylim(-0.5, 2.2)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(r'$\mathbb{E}[f(\tilde{x})]$ for $f(\tilde{x})=\sin(2\tilde{x})/\tilde{x}$ with $\tilde{x}=x+z$, $z\sim\mathcal{N}(0,\nu)$')
    fig.tight_layout()
    fig.savefig(f"{OUT}/expected_value_gaussian.pdf")
    plt.close(fig)
    print("Saved expected_value_gaussian.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 6: Expected value + variance trade-off (Example 20.5)
# f(x,z) = x² + z, z ~ Gamma(2/(1+|x|), 2) → mean=4/(1+|x|), var=8/(1+|x|)
# ─────────────────────────────────────────────────────────────────
def fig_mean_variance_tradeoff():
    x = np.linspace(-3, 3, 500)

    # mean and std of z ~ Gamma(shape=2/(1+|x|), scale=2) — using scipy convention
    # E[z] = shape*scale = 4/(1+|x|), Var[z] = shape*scale^2 = 8/(1+|x|)
    mean_z = 4.0 / (1 + np.abs(x))
    std_z  = np.sqrt(8.0 / (1 + np.abs(x)))

    # f(x,z) = x^2 + z
    # E[f] = x^2 + E[z] = x^2 + 4/(1+|x|)
    # Var[f] = Var[z] = 8/(1+|x|)
    E_f  = x**2 + mean_z
    std_f = std_z

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: expected value with ±σ band
    ax = axes[0]
    ax.fill_between(x, E_f - std_f, E_f + std_f, color=LIGHT_BLUE, alpha=0.5, label=r'$\pm\sigma$')
    ax.plot(x, E_f, color='black', lw=2, label='expected value')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    ax.set_title(r'$f(x,z)=x^2+z$, $z\sim\mathrm{Gamma}$')

    # Panel 2: α E[y|x] + (1−α) √Var[y|x] for α ∈ [0,1]
    ax2 = axes[1]
    alphas = np.linspace(0, 1, 11)
    colors_a = plt.cm.Blues(np.linspace(0.2, 0.9, len(alphas)))
    optima_x = []
    for alpha, col in zip(alphas, colors_a):
        obj = alpha * E_f + (1 - alpha) * std_f
        ax2.plot(x, obj, color=col, lw=1.2)
        idx = np.argmin(obj)
        ax2.plot(x[idx], obj[idx], '.', color=col, ms=6)
        optima_x.append(x[idx])

    ax2.set_xlabel('$x$')
    ax2.set_ylabel(r'$\alpha\mu + (1-\alpha)\sigma$')
    ax2.set_ylim(0, 10)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax2)
    cb.set_label(r'$\alpha$')
    ax2.set_title(r'Trade-off: $\alpha\mathbb{E}[y|x]+(1-\alpha)\sqrt{\mathrm{Var}[y|x]}$')

    fig.tight_layout()
    fig.savefig(f"{OUT}/mean_variance_tradeoff.pdf")
    plt.close(fig)
    print("Saved mean_variance_tradeoff.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 7: CVaR vs VaR illustration
# ─────────────────────────────────────────────────────────────────
def fig_cvar_var():
    np.random.seed(42)
    # Mix of two normals to create a non-trivial distribution
    n = 2000
    samples = np.concatenate([
        np.random.randn(int(0.8*n)) * 1.0 + 0.0,
        np.random.randn(int(0.2*n)) * 0.5 + 3.5,
    ])

    alpha = 0.95
    var_alpha = np.quantile(samples, alpha)
    cvar_alpha = samples[samples >= var_alpha].mean()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts, bins, patches = ax.hist(samples, bins=60, density=True,
                                    color=LIGHT_BLUE, edgecolor='white', alpha=0.8)
    # Color the tail
    for patch, left in zip(patches, bins[:-1]):
        if left >= var_alpha:
            patch.set_facecolor(RED)
            patch.set_alpha(0.7)

    ax.axvline(var_alpha,  color='navy',  lw=2, ls='--', label=fr'VaR$_{{0.95}}$ = {var_alpha:.2f}')
    ax.axvline(cvar_alpha, color=RED,     lw=2, ls='-',  label=fr'CVaR$_{{0.95}}$ = {cvar_alpha:.2f}')
    ax.set_xlabel('Objective function value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=10)
    ax.set_title(r'Value at Risk (VaR) vs Conditional Value at Risk (CVaR), $\alpha=0.95$')
    fig.tight_layout()
    fig.savefig(f"{OUT}/cvar_var.pdf")
    plt.close(fig)
    print("Saved cvar_var.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 8: Markowitz portfolio optimization (Example 20.6)
# μ=[0.26, 0.08, 0.74], Σ as given
# ─────────────────────────────────────────────────────────────────
def fig_markowitz():
    try:
        from scipy.optimize import minimize as sp_min
    except ImportError:
        print("scipy not available; skipping markowitz")
        return

    mu = np.array([0.26, 0.08, 0.74])
    Sigma = np.array([
        [0.21, 0.03, 0.01],
        [0.03, 0.06, 0.04],
        [0.01, 0.04, 0.94],
    ])
    b = 1.0  # total budget
    n = 3

    def optimal_portfolio(w):
        # minimize -w x^T mu + (1-w) x^T Sigma x
        def obj(x): return -w * x @ mu + (1-w) * x @ Sigma @ x
        def jac(x): return -w * mu + 2*(1-w) * Sigma @ x
        cons = [{'type': 'eq', 'fun': lambda x: x.sum() - b}]
        bounds = [(0, None)] * n
        x0 = np.ones(n) / n
        res = sp_min(obj, x0, jac=jac, method='SLSQP',
                     constraints=cons, bounds=bounds)
        return res.x if res.success else x0

    ws = np.linspace(0, 1, 200)
    portfolios = np.array([optimal_portfolio(w) for w in ws])

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['Stock 1', 'Stock 2', 'Stock 3']
    colors_s = [BLUE, '#a78dc4', '#e8a0a0']
    ax.stackplot(ws, portfolios.T, labels=labels, colors=colors_s, alpha=0.85)
    ax.set_xlabel('$w$ (weight on expected return)')
    ax.set_ylabel('Portfolio allocation')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, b)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Markowitz Portfolio: optimal allocation vs. return weight $w$')
    fig.tight_layout()
    fig.savefig(f"{OUT}/markowitz_portfolio.pdf")
    plt.close(fig)
    print("Saved markowitz_portfolio.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 9: Statistical feasibility — probability of success vs factor of safety
# ─────────────────────────────────────────────────────────────────
def fig_stat_feasibility():
    gammas = np.linspace(1.0, 2.0, 300)
    # Optimal x = sqrt(gamma)
    x_opt = np.sqrt(gammas)
    sigma_z = 0.1  # std dev of z

    # Case 1: g(x,z) = x^{-2} + z <= 1, z ~ N(0, sigma^2)
    # Prob success = Prob(x^{-2}+z <= 1) = Prob(z <= 1 - x^{-2})
    # = Phi((1 - x^{-2})/sigma)
    p1 = norm.cdf((1 - x_opt**(-2)) / sigma_z)

    # Case 2: g(x,z) = (x+z)^{-2} <= 1, x+z > 0
    # Prob((x+z)^{-2} <= 1) = Prob(x+z >= 1 or x+z <= -1)
    # since x+z > 0 in practice: Prob(x+z >= 1) = Phi((x-1)/sigma)
    p2 = norm.cdf((x_opt - 1) / sigma_z)

    # Case 3: g(x,z) = (1+z)x^{-2} <= 1, z ~ N(0, sigma^2)
    # Prob((1+z)x^{-2} <= 1) = Prob(1+z <= x^2) = Prob(z <= x^2-1) = Phi((x^2-1)/sigma)
    p3 = norm.cdf((x_opt**2 - 1) / sigma_z)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gammas, p1, color='blue',  lw=2, label=r'$x^{-2}+z$')
    ax.plot(gammas, p2, color='green', lw=2, label=r'$(x+z)^{-2}$')
    ax.plot(gammas, p3, color='purple',lw=2, label=r'$(1+z)x^{-2}$')
    ax.set_xlabel(r'Factor of safety $\gamma$')
    ax.set_ylabel('Probability of success')
    ax.set_xlim(1, 2)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=10)
    ax.set_title('Statistical feasibility vs factor of safety')
    fig.tight_layout()
    fig.savefig(f"{OUT}/stat_feasibility.pdf")
    plt.close(fig)
    print("Saved stat_feasibility.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 10: CVaR vs quantile illustration (p459 figure)
# ─────────────────────────────────────────────────────────────────
def fig_cvar_quantile():
    np.random.seed(7)
    x_vals = np.linspace(-3, 5, 400)
    # A skewed distribution
    from scipy.stats import skewnorm
    dist = skewnorm(a=4, loc=0, scale=1.5)
    y_pdf = dist.pdf(x_vals)

    alpha = 0.9
    var_a = dist.ppf(alpha)
    cvar_a = dist.expect(lambda t: t, lb=var_a) / (1-alpha)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x_vals, y_pdf, color='black', lw=2)
    ax.fill_between(x_vals, y_pdf, where=(x_vals >= var_a),
                    color=RED, alpha=0.5, label=f'tail (α={alpha})')
    ax.axvline(var_a,  color='navy', lw=2, ls='--',
               label=fr'VaR$_{{0.9}}$ = {var_a:.2f}  (1-$\alpha$ quantile)')
    ax.axvline(cvar_a, color=RED,    lw=2, ls='-',
               label=fr'CVaR$_{{0.9}}$ = {cvar_a:.2f}  (conditional mean of tail)')
    ax.set_xlabel('Objective value $y$')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(r'CVaR$_\alpha$: expected value in worst $(1-\alpha)$ fraction')
    fig.tight_layout()
    fig.savefig(f"{OUT}/cvar_quantile.pdf")
    plt.close(fig)
    print("Saved cvar_quantile.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 11: Chance constraints — feasible region shrinks with stringency
# ─────────────────────────────────────────────────────────────────
def fig_chance_constraint():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # Left: original feasible region (polygon)
    poly_x = np.array([0.5, 2.0, 3.0, 3.5, 2.5, 0.5, 0.5])
    poly_y = np.array([0.5, 0.3, 1.0, 2.5, 3.5, 2.5, 0.5])
    axes[0].fill(poly_x, poly_y, color=LIGHT_BLUE, alpha=0.7)
    axes[0].plot(poly_x, poly_y, color=BLUE, lw=2)
    # Optimal on boundary
    axes[0].plot(3.0, 2.8, 'r*', ms=12, label='optimum $x^*$')
    axes[0].set_xlim(0, 4.5); axes[0].set_ylim(0, 4.5)
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    axes[0].set_title('Original feasible region $\mathcal{X}$')
    axes[0].legend(fontsize=9)
    axes[0].set_aspect('equal')

    # Right: shrunk feasible region (inset boundary)
    from matplotlib.patches import Polygon
    shrink = 0.4
    cx, cy = poly_x[:-1].mean(), poly_y[:-1].mean()
    inner_x = cx + (poly_x[:-1] - cx) * (1 - shrink)
    inner_y = cy + (poly_y[:-1] - cy) * (1 - shrink)
    inner_x = np.append(inner_x, inner_x[0])
    inner_y = np.append(inner_y, inner_y[0])
    axes[1].fill(poly_x, poly_y, color=LIGHT_BLUE, alpha=0.3, label='original')
    axes[1].plot(poly_x, poly_y, color=BLUE, lw=1, ls='--')
    axes[1].fill(inner_x, inner_y, color=BLUE, alpha=0.5, label='robust $\mathcal{X}$')
    axes[1].plot(inner_x, inner_y, color='navy', lw=2)
    axes[1].set_xlim(0, 4.5); axes[1].set_ylim(0, 4.5)
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
    axes[1].set_title('Shrunk feasible region (factor of safety $\\gamma$)')
    axes[1].legend(fontsize=9)
    axes[1].set_aspect('equal')

    fig.tight_layout()
    fig.savefig(f"{OUT}/chance_constraint.pdf")
    plt.close(fig)
    print("Saved chance_constraint.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 12: Six-sigma feasibility — Gaussian with 6-sigma boundary
# ─────────────────────────────────────────────────────────────────
def fig_six_sigma():
    x = np.linspace(-7, 7, 400)
    y = norm.pdf(x)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, color='black', lw=2)
    ax.fill_between(x, y, where=(np.abs(x) <= 6), color=LIGHT_BLUE, alpha=0.6)
    ax.fill_between(x, y, where=(x > 6),  color=RED, alpha=0.7, label='violation region')
    ax.fill_between(x, y, where=(x < -6), color=RED, alpha=0.7)
    ax.axvline( 6, color=RED,   lw=2, ls='--', label=r'$\pm 6\sigma$ boundary')
    ax.axvline(-6, color=RED,   lw=2, ls='--')
    ax.axvline( 0, color='gray', lw=1, ls=':')
    ax.set_xlabel('Number of standard deviations from the mean')
    ax.set_ylabel(r'$\mathcal{N}(x\mid\mu,\sigma^2)$')
    ax.annotate('feasibility boundary', xy=(0.35, 0.38), xycoords='axes fraction',
                fontsize=10, color='navy')
    ax.legend(fontsize=10)
    ax.set_title(r'Six-sigma: feasibility boundary at $\pm 6\sigma$')
    fig.tight_layout()
    fig.savefig(f"{OUT}/six_sigma.pdf")
    plt.close(fig)
    print("Saved six_sigma.pdf")

# ─────────────────────────────────────────────────────────────────
# Figure 13: Types of uncertainty overview diagram
# ─────────────────────────────────────────────────────────────────
def fig_uncertainty_types():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')

    boxes = [
        (0.10, 0.50, 'Optimization\nUnder\nUncertainty', '#e8f4f8', 'navy', 16),
        (0.38, 0.75, 'Set-Based\n(Minimax /\nInfo-gap)', '#d0e8ff', BLUE,   12),
        (0.38, 0.25, 'Probabilistic\n(Expected value,\nVariance, CVaR,\nChance constraints)', '#ffeedd', '#c06000', 10),
    ]
    for (cx, cy, txt, fc, ec, fs) in boxes:
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, lw=1.5),
                transform=ax.transAxes)

    ax.annotate('', xy=(0.32, 0.75), xytext=(0.22, 0.60),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='navy'))
    ax.annotate('', xy=(0.32, 0.25), xytext=(0.22, 0.40),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#c06000'))

    ax.set_title('Chapter 20: Taxonomy of Uncertainty Approaches', fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/uncertainty_types.pdf")
    plt.close(fig)
    print("Saved uncertainty_types.pdf")

# ─────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 20...")
    fig_noisy_objective()
    fig_minimax()
    fig_rotated_ellipse()
    fig_info_gap()
    fig_expected_value_gaussian()
    fig_mean_variance_tradeoff()
    fig_cvar_var()
    fig_markowitz()
    fig_stat_feasibility()
    fig_cvar_quantile()
    fig_chance_constraint()
    fig_six_sigma()
    fig_uncertainty_types()
    print("\nAll figures generated successfully.")
