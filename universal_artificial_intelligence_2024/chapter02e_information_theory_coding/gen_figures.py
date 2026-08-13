"""
Figure generation for Chapter 2.5 "Information Theory and Coding" slides.
An Introduction to Universal Artificial Intelligence (Hutter, Quarel, Catt, 2024)

Run with:
  conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

FIGDIR = 'figures'

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# Figure 2.12: Entropy H(X) of a biased coin as a function of theta = P(X=H)
# ---------------------------------------------------------------------------
def fig_entropy_coin():
    theta = np.linspace(1e-6, 1 - 1e-6, 2000)
    H = theta * np.log2(1 / theta) + (1 - theta) * np.log2(1 / (1 - theta))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(theta, H, color='#1f4e79', lw=2.2)
    ax.set_xlabel(r'$P(X{=}x) = \theta$')
    ax.set_ylabel(r'$H(X)$  (bits)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/entropy_coin.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2.13: cross-section: KL(P||Q) and KL(Q||P) for biased-vs-fair coin,
# P = (theta, 1-theta) varying, Q = (0.5, 0.5) fixed.
# ---------------------------------------------------------------------------
def kl_bernoulli(p, q):
    """KL( Bern(p) || Bern(q) ) in bits, elementwise-safe."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(p > 0, p * np.log2(np.clip(p, 1e-300, 1) / np.clip(q, 1e-300, 1)), 0.0)
        term2 = np.where((1 - p) > 0, (1 - p) * np.log2(np.clip(1 - p, 1e-300, 1) / np.clip(1 - q, 1e-300, 1)), 0.0)
    out = term1 + term2
    # divergence to +inf where q hits 0/1 but p doesn't
    bad = ((q <= 0) & (p > 0)) | ((q >= 1) & (p < 1))
    out = np.where(bad, np.inf, out)
    return out


def fig_kl_cross_section():
    theta = np.linspace(0.002, 0.998, 900)
    KL_PQ = kl_bernoulli(theta, 0.5 * np.ones_like(theta))   # KL(P||Q), Q fixed fair coin
    KL_QP = kl_bernoulli(0.5 * np.ones_like(theta), theta)   # KL(Q||P)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(theta, KL_PQ, color='black', lw=2.0, label=r'$\mathrm{KL}(P\|Q)$')
    ax.plot(theta, np.clip(KL_QP, 0, 2.05), color='gray', lw=2.0, ls='--', label=r'$\mathrm{KL}(Q\|P)$')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('KL (bits)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kl_cross_section.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2.14: contour plot of KL(P||Q) for biased coins P=(p,1-p), Q=(q,1-q)
# ---------------------------------------------------------------------------
def fig_kl_contour():
    p = np.linspace(0.01, 0.99, 400)
    q = np.linspace(0.01, 0.99, 400)
    P, Q = np.meshgrid(p, q)
    KL = kl_bernoulli(P, Q)
    KL = np.clip(KL, 0, 1.2)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cf = ax.contourf(P, Q, KL, levels=30, cmap='Greys')
    cs = ax.contour(P, Q, KL, levels=[0.01, 0.1, 0.25, 0.5, 1.0], colors='black', linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
    ax.set_xlabel('$p$')
    ax.set_ylabel('$q$')
    ax.set_title(r'$\mathrm{KL}(P\|Q)$', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kl_contour.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2.16: mixture-of-Gaussians P, single Gaussian Q minimizing
# KL(P||Q) (moment matching -> covering) vs KL(Q||P) (mode seeking)
# ---------------------------------------------------------------------------
def gaussian_pdf(X, Y, mu, Sigma):
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    dx = X - mu[0]
    dy = Y - mu[1]
    quad = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))


def fig_gaussian_kl_asymmetry():
    # Mixture P: two well-separated 2D Gaussian components. (This uses a
    # larger mode separation than the book's own numbers so that the
    # mode-seeking behaviour of reverse KL is unambiguous -- with the book's
    # closer components the valley between modes is too shallow for the
    # single-mode solution to actually attain the lower KL(Q||P).)
    mu1, Sigma1 = np.array([0.0, 0.0]), np.array([[1.0, 0.3], [0.3, 1.0]])
    mu2, Sigma2 = np.array([5.0, 4.0]), np.array([[1.0, -0.3], [-0.3, 1.0]])
    w1, w2 = 0.5, 0.5

    xs = np.linspace(-5, 10, 360)
    ys = np.linspace(-5, 9, 360)
    X, Y = np.meshgrid(xs, ys)
    P = w1 * gaussian_pdf(X, Y, mu1, Sigma1) + w2 * gaussian_pdf(X, Y, mu2, Sigma2)

    # --- Q minimizing KL(P||Q): closed-form moment matching (mean-covering) ---
    mu_cov = w1 * mu1 + w2 * mu2
    Sigma_cov = (w1 * (Sigma1 + np.outer(mu1 - mu_cov, mu1 - mu_cov))
                 + w2 * (Sigma2 + np.outer(mu2 - mu_cov, mu2 - mu_cov)))
    Q_cov = gaussian_pdf(X, Y, mu_cov, Sigma_cov)

    # --- Q minimizing KL(Q||P): numerically, mode-seeking (locks onto one mode) ---
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    cell = dx * dy

    def unpack(theta):
        mu = theta[:2]
        l11, l21, l22 = theta[2], theta[3], theta[4]
        L = np.array([[np.exp(l11), 0.0], [l21, np.exp(l22)]])
        Sigma = L @ L.T
        return mu, Sigma

    def neg_obj(theta):
        mu, Sigma = unpack(theta)
        Qg = gaussian_pdf(X, Y, mu, Sigma)
        Qg = np.clip(Qg, 1e-300, None)
        Pg = np.clip(P, 1e-300, None)
        kl = np.sum(Qg * (np.log(Qg) - np.log(Pg))) * cell
        return kl

    best = None
    for start_mu in (mu1, mu2, mu_cov):
        theta0 = np.array([start_mu[0], start_mu[1], 0.0, 0.0, 0.0])
        res = minimize(neg_obj, theta0, method='Nelder-Mead',
                        options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 4000})
        if best is None or res.fun < best.fun:
            best = res
    mu_mode, Sigma_mode = unpack(best.x)
    Q_mode = gaussian_pdf(X, Y, mu_mode, Sigma_mode)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, Qg, title in zip(axes, (Q_cov, Q_mode),
                              (r'Minimize $\mathrm{KL}(P\|Q)$', r'Minimize $\mathrm{KL}(Q\|P)$')):
        ax.contourf(X, Y, P, levels=12, cmap='Greys', alpha=0.55)
        ax.contour(X, Y, Qg, levels=8, colors='black', linewidths=1.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-3, 8)
        ax.set_ylim(-3, 7)
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/gaussian_kl_asymmetry.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2.17: geometric intuition for the Kraft inequality.
# Prefix code C = {01, 100, 101, 11}: binary tree, mass 1 split in half at
# every node, leaves in bold sum to 3/4 <= 1.
# ---------------------------------------------------------------------------
def fig_kraft_tree():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axis('off')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 3.3)

    def node(x, y, label, bold=False):
        fc = 'white'
        r = 0.34
        circ = plt.Circle((x, y), r, facecolor=fc, edgecolor='black',
                           linewidth=2.2 if bold else 1.2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                 fontweight='bold' if bold else 'normal', zorder=4)

    def edge(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color='black', lw=1.3, zorder=1)

    # root
    node(2, 3, '1')
    # depth 1
    node(1, 2, '1/2'); node(3, 2, '1/2')
    edge(2, 3, 1, 2); edge(2, 3, 3, 2)
    # depth 2 left branch -> codeword "01" leaf (bold, prob 1/4), and continue right "0-"?
    # Tree layout for C={01,100,101,11}:
    #   root -> 0 -> "0" node(1/2) -> 1 -> leaf "01" (1/4)  [bold leaf]
    #   root -> 1 -> "1" node(1/2) -> 0 -> "10" node(1/4) -> 0 -> leaf "100" (1/8) [bold]
    #                                                      -> 1 -> leaf "101" (1/8) [bold]
    #                            -> 1 -> leaf "11" (1/4) [bold]
    node(1, 1, '1/4', bold=True)
    ax.text(1, 0.58, '01', ha='center', fontsize=9)
    edge(1, 2, 1, 1)

    node(2.6, 1, '1/4')
    edge(3, 2, 2.6, 1)
    node(3.6, 1, '1/4', bold=True)
    ax.text(3.6, 0.58, '11', ha='center', fontsize=9)
    edge(3, 2, 3.6, 1)

    node(2.1, 0, '1/8', bold=True)
    ax.text(2.1, -0.32, '100', ha='center', fontsize=9)
    edge(2.6, 1, 2.1, 0)
    node(3.1, 0, '1/8', bold=True)
    ax.text(3.1, -0.32, '101', ha='center', fontsize=9)
    edge(2.6, 1, 3.1, 0)

    ax.set_ylim(-0.7, 3.4)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kraft_tree.pdf')
    plt.close(fig)


if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    fig_entropy_coin()
    fig_kl_cross_section()
    fig_kl_contour()
    fig_gaussian_kl_asymmetry()
    fig_kraft_tree()
    print('All figures generated in', FIGDIR)
