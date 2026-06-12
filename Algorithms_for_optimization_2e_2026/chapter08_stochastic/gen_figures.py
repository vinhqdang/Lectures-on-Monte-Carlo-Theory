"""
gen_figures.py  –  Generate all figures for Chapter 8 slides.
Uses matplotlib with Agg backend; saves PDFs to figures/.
Also crops selected diagrams from the book PDF via pymupdf.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os, sys

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name, **kw):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches='tight', **kw)
    plt.close()
    print(f"  saved {path}")

# ── palette ─────────────────────────────────────────────────────────────────
BLUE   = "#2166ac"
LBLUE  = "#abd9e9"
ORANGE = "#d6604d"
GRAY   = "#888888"

# ============================================================
# Fig 1 – Noisy descent: clean vs noisy gradient trajectory
# ============================================================
def fig_noisy_descent():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Quadratic bowl
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 2*Y**2

    for ax, noisy, title in zip(axes, [False, True],
                                ['Clean gradient descent',
                                 'Noisy gradient descent\n(stochastic perturbation)']):
        ax.contour(X, Y, Z, levels=10, colors='gray', linewidths=0.6, alpha=0.5)
        # Simulate trajectory
        pos = np.array([2.5, 2.0])
        traj = [pos.copy()]
        for _ in range(30):
            grad = np.array([2*pos[0], 4*pos[1]])
            if noisy:
                grad += np.random.randn(2) * 0.8
            pos = pos - 0.12 * grad
            traj.append(pos.copy())
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=BLUE, ms=3, lw=1.5)
        ax.plot(traj[0,0], traj[0,1], 's', color=ORANGE, ms=7, zorder=5, label='start')
        ax.plot(0, 0, '*', color='gold', ms=10, zorder=5, label='minimum')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.legend(fontsize=7)
        ax.set_aspect('equal')

    plt.suptitle('Effect of Noise on Gradient Descent', fontsize=10, y=1.01)
    plt.tight_layout()
    savefig("noisy_descent.pdf")

# ============================================================
# Fig 2 – MADS: positive spanning sets in 2D
# ============================================================
def fig_mads_spanning():
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    directions_sets = [
        [(1,0),(-1,1),(0,-1)],
        [(0,1),(1,-1),(-1,0)],
        [(1,1),(-1,0),(0,-1)],
        [(1,0),(0,1),(-1,-1)],
        [(1,0),(-1,0),(0,1)],
        [(1,1),(-1,0),(0,-1)],
        [(1,-1),(0,1),(-1,0)],
        [(0,1),(1,0),(-1,-1)],
    ]
    for ax, dirs in zip(axes.flat, directions_sets):
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        for d in dirs:
            ax.annotate('', xy=(d[0]*1.2, d[1]*1.2), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))
        ax.plot(0, 0, 'o', color='black', ms=4)
        ax.set_xticks([]); ax.set_yticks([])
        # Light blue border
        for spine in ax.spines.values():
            spine.set_edgecolor(LBLUE); spine.set_linewidth(2)
    plt.suptitle('Positive Spanning Sets for $\\mathbb{R}^2$ (Example 8.1)', fontsize=10)
    plt.tight_layout()
    savefig("mads_spanning_sets.pdf")

# ============================================================
# Fig 3 – MADS: search progression
# ============================================================
def fig_mads_search():
    np.random.seed(7)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-3, 3, 200); y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (X-0.5)**2 + 2*(Y+0.5)**2 + 0.3*np.sin(3*X)*np.cos(2*Y)
    ax.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.4)
    ax.contour(X, Y, Z, levels=15, colors='navy', linewidths=0.5, alpha=0.5)

    # Simulate MADS steps
    pos = np.array([-2.0, 2.0])
    alpha = 1.0
    pts = [pos.copy()]
    for _ in range(20):
        directions = [np.array([1,0]), np.array([-1,0]),
                      np.array([0,1]), np.array([0,-1]),
                      np.array([-1,-1])]
        improved = False
        for d in directions:
            cand = pos + alpha * d
            if (cand[0]-0.5)**2 + 2*(cand[1]+0.5)**2 < (pos[0]-0.5)**2 + 2*(pos[1]+0.5)**2:
                pos = cand; improved = True; break
        alpha = min(1.0, 4*alpha) if improved else alpha/4
        pts.append(pos.copy())
    pts = np.array(pts)
    ax.plot(pts[:,0], pts[:,1], '-o', color=ORANGE, ms=4, lw=1.5, label='MADS path')
    ax.plot(pts[0,0], pts[0,1], 's', color='green', ms=8, label='start')
    ax.plot(0.5, -0.5, '*', color='gold', ms=12, label='approx. min')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('MADS Search Progression', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("mads_search.pdf")

# ============================================================
# Fig 4 – Memory-efficient zeroth-order: stochastic gradient idea
# ============================================================
def fig_mezo():
    np.random.seed(0)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    x = np.linspace(-3, 3, 300)
    f = lambda t: t**2 + 0.5*np.sin(5*t)

    # Left: function + finite diff approximation
    ax = axes[0]
    ax.plot(x, f(x), color=BLUE, lw=2, label='$f(x)$')
    x0 = 1.5; eps = 0.5
    ax.axvline(x0, color='gray', lw=0.8, ls='--')
    ax.annotate('', xy=(x0+eps, f(x0+eps)), xytext=(x0-eps, f(x0-eps)),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))
    ax.plot([x0-eps, x0+eps], [f(x0-eps), f(x0+eps)], 'o', color=ORANGE, ms=6)
    ax.set_title('Finite-Difference Approx.\n$\\hat{g} = \\frac{f(x+\\epsilon z)-f(x-\\epsilon z)}{2\\epsilon}z$', fontsize=8)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.legend(fontsize=8)

    # Right: gradient walk
    ax = axes[1]
    pos = np.array([-2.5])
    traj = [float(pos)]
    for _ in range(40):
        z = np.random.randn()
        eps_v = 0.3
        g_hat = (f(pos+eps_v*z) - f(pos-eps_v*z)) / (2*eps_v) * z
        pos = pos - 0.1 * g_hat
        traj.append(float(pos))
    iters = np.arange(len(traj))
    ax.plot(iters, [f(t) for t in traj], color=BLUE, lw=1.5)
    ax.set_xlabel('Iteration'); ax.set_ylabel('$f(x^{(k)})$')
    ax.set_title('MeZO: Function value over iterations', fontsize=8)

    plt.suptitle('Memory-Efficient Zeroth-Order Optimization (MeZO)', fontsize=10, y=1.01)
    plt.tight_layout()
    savefig("mezo_illustration.pdf")

# ============================================================
# Fig 5 – Simulated Annealing: acceptance probability
# ============================================================
def fig_sa_acceptance():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Acceptance probability vs Delta_y
    dy = np.linspace(-2, 5, 300)
    ax = axes[0]
    for t, color in zip([0.5, 1.0, 2.0, 5.0], ['navy','blue','cornflowerblue','lightblue']):
        prob = np.where(dy < 0, 1.0, np.exp(-dy/t))
        ax.plot(dy, prob, color=color, lw=1.8, label=f'$t={t}$')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('$\\Delta y = f(x\')-f(x)$')
    ax.set_ylabel('Acceptance probability')
    ax.set_title('Metropolis acceptance $P = e^{-\\Delta y/t}$', fontsize=9)
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # Temperature schedules
    k = np.arange(1, 101)
    ax = axes[1]
    ax.plot(k, 10/k,       lw=2, label='Boltzmann: $t_0/\\ln k$', color=BLUE)
    ax.plot(k, 10*0.95**k, lw=2, label='Exponential: $t_0\\cdot\\gamma^k$', color=ORANGE)
    ax.plot(k, 10/k**0.5,  lw=2, label='Cauchy: $t_0/k$', color='green')
    ax.set_xlabel('Iteration $k$'); ax.set_ylabel('Temperature $t^{(k)}$')
    ax.set_title('Cooling Schedules', fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    savefig("sa_acceptance.pdf")

# ============================================================
# Fig 6 – SA: Effect of sigma and temperature (Ackley example)
# ============================================================
def fig_sa_ackley():
    """Simulate SA on Ackley function; show convergence bands."""
    def ackley(x):
        n = len(x)
        a, b, c = 20, 0.2, 2*np.pi
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(c*x))
        return -a*np.exp(-b*np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e

    np.random.seed(123)
    n_runs = 30; n_iter = 100
    x0 = np.array([15.0, 15.0])

    fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True)
    sigmas = [1, 5, 25]; t0s = [1, 10, 25]

    for ri, sigma in enumerate(sigmas):
        for ci, t_init in enumerate(t0s):
            ax = axes[ri][ci]
            all_vals = np.zeros((n_runs, n_iter+1))
            for r in range(n_runs):
                x = x0.copy(); y = ackley(x)
                all_vals[r, 0] = y
                for k in range(1, n_iter+1):
                    t = t_init / k
                    x_new = x + np.random.randn(2)*sigma
                    y_new = ackley(x_new)
                    dy = y_new - y
                    if dy < 0 or np.random.rand() < np.exp(-dy/max(t, 1e-10)):
                        x, y = x_new, y_new
                    all_vals[r, k] = y
            med = np.median(all_vals, axis=0)
            p5  = np.percentile(all_vals, 5, axis=0)
            p95 = np.percentile(all_vals, 95, axis=0)
            p25 = np.percentile(all_vals, 25, axis=0)
            p75 = np.percentile(all_vals, 75, axis=0)
            iters = np.arange(n_iter+1)
            ax.fill_between(iters, p5, p95, alpha=0.25, color=LBLUE)
            ax.fill_between(iters, p25, p75, alpha=0.4, color=LBLUE)
            ax.plot(iters, med, color='black', lw=1.2)
            ax.set_ylim(0, 32)
            if ri == 0:
                ax.set_title(f'$t^{{(1)}}={t_init}$', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'$\\sigma={sigma}$\n$y$', fontsize=8)
            if ri == 2:
                ax.set_xlabel('iteration', fontsize=7)

    plt.suptitle("SA on Ackley's Function: $\\sigma$ vs Temperature (Example 8.2)", fontsize=10)
    plt.tight_layout()
    savefig("sa_ackley.pdf")

# ============================================================
# Fig 7 – Cross-Entropy Method: distribution evolution (1D)
# ============================================================
def fig_cem_1d():
    """Illustrate CEM narrowing around minimum on a 1D multi-modal function."""
    np.random.seed(5)
    def f(x):
        return np.sin(3*x)*np.cos(x+0.3) + 0.05*x**2

    x_plot = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    mu, sigma = 0.0, 2.0
    m, m_elite = 50, 10

    for step, ax in enumerate(axes.flat):
        ax.plot(x_plot, f(x_plot), 'k-', lw=1.5, alpha=0.8)
        ax_twin = ax.twinx()
        # Draw proposal density
        x_dens = np.linspace(-5, 5, 200)
        from scipy.stats import norm
        dens = norm.pdf(x_dens, mu, sigma)
        ax_twin.fill_between(x_dens, 0, dens, alpha=0.35, color=LBLUE)
        ax_twin.set_ylim(0, norm.pdf(mu, mu, sigma)*3)
        ax_twin.set_yticks([])
        ax.set_xlim(-5, 5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Iter {step+1}\n$\\mu$={mu:.2f}, $\\sigma$={sigma:.2f}', fontsize=7)
        ax.set_xlabel('$x$')
        # CEM update
        samples = np.random.normal(mu, sigma, m)
        vals    = f(samples)
        order   = np.argsort(vals)  # ascending = best first
        elite   = samples[order[:m_elite]]
        mu    = elite.mean()
        sigma = elite.std() + 0.1  # smoothing

    plt.suptitle('Cross-Entropy Method: Distribution Evolution', fontsize=10)
    plt.tight_layout()
    savefig("cem_evolution.pdf")

# ============================================================
# Fig 8 – Proposal Distribution Descent (PDD)
# ============================================================
def fig_pdd():
    np.random.seed(9)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: Unimodal vs mixture proposal
    x = np.linspace(-6, 6, 500)
    from scipy.stats import norm
    unimodal = norm.pdf(x, 0, 2)
    mixture  = 0.5*norm.pdf(x, -3, 0.7) + 0.5*norm.pdf(x, 3, 0.7)
    ax = axes[0]
    ax.plot(x, unimodal, color=BLUE,   lw=2, label='Unimodal $\\mathcal{N}(0,4)$')
    ax.plot(x, mixture,  color=ORANGE, lw=2, label='Mixture: two modes')
    # Mark two minima
    f_bimodal = lambda t: -2*norm.pdf(t,-3,0.7) - 2*norm.pdf(t,3,0.7) + 0.3
    ax.fill_between(x, 0, mixture, alpha=0.2, color=ORANGE)
    ax.axvline(-3, color='green', lw=1, ls='--', label='minima')
    ax.axvline( 3, color='green', lw=1, ls='--')
    ax.set_xlabel('$x$'); ax.set_ylabel('density / $f(x)$')
    ax.set_title('Proposal Distributions', fontsize=9)
    ax.legend(fontsize=7); ax.set_ylim(bottom=0)

    # Right: gradient descent on log-likelihood
    ax = axes[1]
    mu_hist = [3.0]; v_hist = [2.0]
    mu, v = 3.0, 2.0
    alpha = 0.3
    def target_f(t):  # multi-modal minimization target
        return (t-1.5)**2 + np.sin(3*t)

    m = 50
    for _ in range(20):
        xs = np.random.normal(mu, np.sqrt(v), m)
        fs = target_f(xs)
        # grad wrt mu
        g_mu = np.mean(fs * (xs - mu) / v)
        g_v  = np.mean(fs * ((xs-mu)**2 - v) / (2*v**2))
        mu = mu - alpha * g_mu
        v  = max(0.01, v - alpha * g_v)
        mu_hist.append(mu); v_hist.append(v)

    ax.plot(mu_hist,  color=BLUE,   lw=1.8, label='$\\mu$ trajectory')
    ax.plot(v_hist,   color=ORANGE, lw=1.8, label='$\\nu$ (variance)')
    ax.axhline(1.5, color='gray', ls='--', lw=1, label='true min $x^*$')
    ax.set_xlabel('Iteration'); ax.legend(fontsize=8)
    ax.set_title('PDD Parameter Evolution', fontsize=9)

    plt.tight_layout()
    savefig("pdd_illustration.pdf")

# ============================================================
# Fig 9 – Information-Geometric Optimization (IGO)
# ============================================================
def fig_igo():
    """Show natural gradient update invariance via 2D Gaussian distribution."""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    x = np.linspace(-4, 4, 200); y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)

    # Banana-shaped objective
    def f_banana(X, Y): return (1-X)**2 + 100*(Y - X**2)**2

    Z = np.log1p(f_banana(X, Y))

    for ax, (mu, Sigma), title in zip(axes,
        [
         (np.array([0., 0.]), np.eye(2)*2.0),
         (np.array([-0.5, 0.3]), np.array([[1.2, 0.5],[0.5, 0.8]])),
         (np.array([0.8, 0.6]), np.array([[0.4, 0.1],[0.1, 0.3]])),
        ],
        ['Initial $\\mathcal{N}(\\mu, \\Sigma)$',
         'After 5 IGO steps',
         'After 15 IGO steps']):

        ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd', alpha=0.5)
        ax.contour(X, Y, Z, levels=15, colors='gray', linewidths=0.4, alpha=0.5)

        # Draw ellipse for Gaussian
        from matplotlib.patches import Ellipse
        import numpy.linalg as nla
        vals, vecs = nla.eigh(Sigma)
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        for nsig, alpha_e in [(1, 0.7), (2, 0.35)]:
            ell = Ellipse(xy=mu, width=2*nsig*np.sqrt(vals[0]),
                          height=2*nsig*np.sqrt(vals[1]),
                          angle=angle, edgecolor=BLUE, fc='none', lw=2, alpha=alpha_e)
            ax.add_patch(ell)
        ax.plot(*mu, 'o', color=BLUE, ms=6)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    plt.suptitle('Information-Geometric Optimization (IGO) on Banana Function', fontsize=10)
    plt.tight_layout()
    savefig("igo_illustration.pdf")

# ============================================================
# Fig 10 – CMA-ES: covariance evolution
# ============================================================
def fig_cmaes():
    """Show CMA-ES adapting covariance on rotated ellipsoidal function."""
    np.random.seed(1)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))

    x = np.linspace(-5, 5, 200); y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    # Rotated ellipsoid
    theta = np.pi/4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    def f_rot(X, Y):
        XV = np.stack([X.ravel(), Y.ravel()], axis=1) @ R
        return (XV[:,0]**2 + 10*XV[:,1]**2).reshape(X.shape)

    Z = f_rot(X, Y)

    from matplotlib.patches import Ellipse
    import numpy.linalg as nla

    steps = [
        (np.array([0.,0.]), 2.0, np.eye(2)),
        (np.array([0.5,0.5]), 1.5, R @ np.diag([1.5, 0.3]) @ R.T),
        (np.array([0.8,0.8]), 0.8, R @ np.diag([0.8, 0.1]) @ R.T),
        (np.array([0.95,0.95]), 0.3, R @ np.diag([0.15, 0.02]) @ R.T),
    ]

    for ax, (mu, sigma, C), title in zip(axes, steps,
                                          ['Iter 0','Iter 5','Iter 15','Iter 40']):
        ax.contour(X, Y, Z, levels=12, colors='gray', linewidths=0.5, alpha=0.6)
        cov = sigma**2 * C
        vals, vecs = nla.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        ell = Ellipse(xy=mu, width=2*np.sqrt(abs(vals[0])),
                      height=2*np.sqrt(abs(vals[1])),
                      angle=angle, edgecolor=BLUE, fc=LBLUE, alpha=0.4, lw=2)
        ax.add_patch(ell)
        ax.plot(*mu, 'o', color=ORANGE, ms=6)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x_1$')
        if ax == axes[0]: ax.set_ylabel('$x_2$')

    plt.suptitle('CMA-ES: Covariance Adaptation on Rotated Ellipsoid', fontsize=10)
    plt.tight_layout()
    savefig("cmaes_evolution.pdf")

# ============================================================
# Fig 11 – CMA-ES evolution paths
# ============================================================
def fig_cmaes_paths():
    np.random.seed(3)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-3, 3, 200); y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    theta = np.pi/3
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    def f_rot(X, Y):
        XV = np.stack([X.ravel(), Y.ravel()], axis=1) @ R
        return (XV[:,0]**2 + 8*XV[:,1]**2).reshape(X.shape)
    Z = f_rot(X, Y)
    ax.contour(X, Y, Z, levels=10, colors='gray', linewidths=0.6, alpha=0.5)

    # Simulate evolution paths
    sigma_path = [np.zeros(2)]
    cov_path   = [np.zeros(2)]
    mu = np.array([-2.0, 2.0])
    sigma = 1.5
    C = np.eye(2)
    for k in range(30):
        m = 10
        xs = np.random.multivariate_normal(mu, sigma**2*C, m)
        vals = [f_rot(np.array([[xi[0]]]), np.array([[xi[1]]]))[0,0] for xi in xs]
        idx = np.argsort(vals)[:5]
        elite = xs[idx]
        mu_new = elite.mean(axis=0)
        delta = (mu_new - mu)/sigma
        sigma_path.append(0.9*sigma_path[-1] + 0.3*delta)
        cov_path.append(0.9*cov_path[-1] + 0.3*delta)
        mu = mu_new
    sigma_path = np.array(sigma_path)
    ax.quiver(sigma_path[:-1,0], sigma_path[:-1,1],
              np.diff(sigma_path[:,0]), np.diff(sigma_path[:,1]),
              scale_units='xy', angles='xy', scale=1, color=ORANGE, alpha=0.7, width=0.006)
    ax.plot(sigma_path[:,0], sigma_path[:,1], '-', color=ORANGE, lw=1, label='$p_\\sigma$ path')
    ax.plot(-2, 2, 's', color='green', ms=8, label='start')
    ax.plot(0, 0, '*', color='gold', ms=12, label='minimum')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('CMA-ES Evolution Path $p_\\sigma$', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("cmaes_paths.pdf")

# ============================================================
# Fig 12 – Step size control in CMA-ES
# ============================================================
def fig_cmaes_stepsize():
    np.random.seed(7)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    k = np.arange(1, 101)
    # Simulate sigma evolution
    sigma_hist = [1.0]
    s = 1.0
    for _ in range(99):
        # random walk on path norm
        norm_ratio = 1.0 + 0.1*np.random.randn()
        s *= np.exp(0.2*(norm_ratio - 1))
        sigma_hist.append(s)

    axes[0].plot(k, sigma_hist, color=BLUE, lw=1.8)
    axes[0].axhline(1.0, color='gray', ls='--', lw=1, label='initial $\\sigma$')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('$\\sigma$')
    axes[0].set_title('Step-Size $\\sigma$ Adaptation', fontsize=9)
    axes[0].legend(fontsize=8)

    # Right: illustrate scale invariance
    scales = np.logspace(-1, 1, 50)
    ratio  = np.ones_like(scales)  # ideal: always finds optimum in same #evals
    axes[1].semilogx(scales, ratio + 0.05*np.random.randn(50), 'o-',
                     color=ORANGE, ms=4, lw=1.5, label='CMA-ES (scale inv.)')
    axes[1].semilogx(scales, scales/scales[25] + 0.1*np.random.randn(50), 's--',
                     color=BLUE,   ms=4, lw=1.5, label='Plain ES (not inv.)')
    axes[1].set_xlabel('Problem scale'); axes[1].set_ylabel('Relative iterations')
    axes[1].set_title('Scale Invariance of CMA-ES', fontsize=9)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    savefig("cmaes_stepsize.pdf")

# ============================================================
# Fig 13 – Method comparison on Rosenbrock
# ============================================================
def fig_method_comparison():
    from scipy.optimize import minimize
    np.random.seed(42)

    def rosenbrock(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    x0 = np.array([-1.5, 1.5])
    methods_data = {}

    # Noisy descent (SGD-like)
    pos = x0.copy(); noisy_hist = [rosenbrock(pos)]
    for _ in range(500):
        g = np.array([-2*(1-pos[0]) - 400*pos[0]*(pos[1]-pos[0]**2),
                       200*(pos[1]-pos[0]**2)])
        g += np.random.randn(2)*2
        pos = pos - 0.001*g
        noisy_hist.append(rosenbrock(pos))
    methods_data['Noisy descent'] = noisy_hist

    # Simulated annealing
    pos = x0.copy(); sa_hist = [rosenbrock(pos)]
    t = 5.0
    for k in range(1, 501):
        t = 5.0/(k**0.5)
        cand = pos + np.random.randn(2)*0.5
        dy = rosenbrock(cand) - rosenbrock(pos)
        if dy < 0 or np.random.rand() < np.exp(-dy/max(t,1e-10)):
            pos = cand
        sa_hist.append(rosenbrock(pos))
    methods_data['Simulated Annealing'] = sa_hist

    # CEM
    mu_c = x0.copy(); sigma_c = np.eye(2)*2; cem_hist = []
    for _ in range(25):
        xs = np.random.multivariate_normal(mu_c, sigma_c, 40)
        fs = [rosenbrock(x) for x in xs]
        order = np.argsort(fs)
        elite = xs[order[:8]]
        mu_c = elite.mean(axis=0)
        sigma_c = np.cov(elite.T) + np.eye(2)*0.01
        cem_hist.extend([rosenbrock(mu_c)]*20)
    methods_data['Cross-Entropy'] = cem_hist[:500]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE, ORANGE, 'green']
    for (name, hist), color in zip(methods_data.items(), colors):
        ax.semilogy(hist, label=name, color=color, lw=1.8)
    ax.set_xlabel('Function evaluations'); ax.set_ylabel('$f(x)$ (log scale)')
    ax.set_title('Method Comparison on Rosenbrock Function', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("method_comparison.pdf")

# ──────────────────────────────────────────────────────────────
# Book diagram crops via PyMuPDF
# ──────────────────────────────────────────────────────────────
def crop_book_figures():
    try:
        import fitz  # pymupdf
    except ImportError:
        print("  pymupdf not available – skipping PDF crops")
        return

    pdf_path = os.path.join(os.path.dirname(__file__), "..", "optimization_book.pdf")
    if not os.path.exists(pdf_path):
        print(f"  Book PDF not found at {pdf_path} – skipping crops")
        return

    doc = fitz.open(pdf_path)
    # Pages are 0-indexed in pymupdf; book pages listed are 1-indexed
    crops = [
        # (page_index_0based, rect_xywh_in_pt, output_name)
        # Page 137 (index 136): positive spanning sets figure (Example 8.1)
        (136, fitz.Rect(40, 40, 480, 330), "book_mads_spanning.pdf"),
        # Page 141 (index 140): SA Ackley example (distribution + temperature)
        (141, fitz.Rect(40, 200, 500, 580), "book_sa_ackley.pdf"),
        # Page 145 (index 144): SA flowchart + adaptive SA algorithm
        (144, fitz.Rect(380, 150, 660, 750), "book_sa_flowchart.pdf"),
        # Page 146 (index 145): CEM distribution evolution figure
        (145, fitz.Rect(40, 30, 500, 340), "book_cem_fig.pdf"),
        # Page 150 (index 149): proposal distribution descent figure
        (149, fitz.Rect(360, 30, 670, 270), "book_pdd_fig.pdf"),
        # Page 152 (index 151): IGO multivariate normal figure
        (151, fitz.Rect(360, 30, 680, 310), "book_igo_fig.pdf"),
        # Page 157 (index 156): CMA-ES flower function figure
        (156, fitz.Rect(35, 30, 680, 650), "book_cmaes_flower.pdf"),
    ]

    for page_idx, rect, outname in crops:
        if page_idx >= len(doc):
            print(f"  page {page_idx} out of range, skipping {outname}")
            continue
        page = doc[page_idx]
        mat  = fitz.Matrix(2, 2)  # 2x zoom for resolution
        clip = rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        out_path = os.path.join(FIGDIR, outname)
        pix.save(out_path.replace('.pdf', '.png'))
        # Convert PNG to PDF using matplotlib
        img = plt.imread(out_path.replace('.pdf', '.png'))
        fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
        ax.imshow(img); ax.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"  cropped {outname}")

    doc.close()

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating matplotlib figures...")
    fig_noisy_descent()
    fig_mads_spanning()
    fig_mads_search()
    fig_mezo()
    fig_sa_acceptance()
    fig_sa_ackley()
    fig_cem_1d()
    fig_pdd()
    fig_igo()
    fig_cmaes()
    fig_cmaes_paths()
    fig_cmaes_stepsize()
    fig_method_comparison()

    print("Cropping book PDF figures...")
    crop_book_figures()

    print("Done. All figures in:", FIGDIR)
