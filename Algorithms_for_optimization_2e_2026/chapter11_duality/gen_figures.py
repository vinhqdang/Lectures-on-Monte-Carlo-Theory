"""
gen_figures.py  –  Chapter 11: Duality
Generates all figures for the Beamer slides using matplotlib (Agg backend).
Also crops select diagrams from the book PDF using pymupdf.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import os, sys

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name, fig=None, **kw):
    path = os.path.join(FIGDIR, name)
    (fig or plt).savefig(path, bbox_inches='tight', **kw)
    plt.close('all')
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Dual function as lower bound  (Example 11.1)
#   minimize sin(x),  s.t. x^2 <= 1
# ─────────────────────────────────────────────────────────────────────────────
def fig_dual_lower_bound():
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(-4.5, 4.5, 800)

    # objective
    ax.plot(x, np.sin(x), 'k', lw=2, label=r'$\sin(x)$  (objective)')

    # feasible region: x^2 <= 1  i.e. x in [-1,1]
    xfeas = np.linspace(-1, 1, 300)
    ax.fill_between(xfeas, -1.5, np.sin(xfeas), alpha=0.15, color='royalblue',
                    label='feasible region')

    # Lagrangian L(x,mu) = sin(x) + mu*(x^2-1) for various mu
    mus = np.arange(0, 2.25, 0.25)
    cmap = plt.cm.Purples
    colors = cmap(np.linspace(0.3, 0.95, len(mus)))
    for mu, col in zip(mus, colors):
        lbl = r'$\mu={:.2f}$'.format(mu) if mu in [0.0, 1.0, 2.0] else None
        ax.plot(x, np.sin(x) + mu*(x**2 - 1), color=col, lw=1, alpha=0.85,
                label=lbl)

    # optimal primal
    xstar = -1.0
    pstar = np.sin(xstar)
    ax.axvline(xstar, color='steelblue', ls='--', lw=1, alpha=0.6)
    ax.scatter([xstar], [pstar], color='steelblue', zorder=5, s=60)
    ax.annotate(r'$x^*=-1,\;p^*\approx{:.3f}$'.format(pstar),
                xy=(xstar, pstar), xytext=(-3.5, 3.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-1.6, 10.5)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
    ax.set_title(r'Dual Function as Lower Bound (Ex.\ 11.1)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    savefig("fig_dual_lower_bound.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Dual function  D(lambda)  (Example 11.2)
#   minimize x1+x2+x1*x2,  s.t. x1^2+x2^2 = 1
# ─────────────────────────────────────────────────────────────────────────────
def fig_dual_function_ex2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- top: contour + constraint circle ----
    ax = axes[0]
    xv = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(xv, xv)
    Z = X + Y + X*Y
    cs = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r', alpha=0.7)
    plt.colorbar(cs, ax=ax, shrink=0.8)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=2, label='$x_1^2+x_2^2=1$')

    # critical points
    pts = [(-1,0), (0,-1),
           ((np.sqrt(2)+1)/(np.sqrt(2)+2), (np.sqrt(2)+1)/(np.sqrt(2)+2)),
           ((np.sqrt(2)-1)/(np.sqrt(2)-2), (np.sqrt(2)-1)/(np.sqrt(2)-2))]
    vals = [p[0]+p[1]+p[0]*p[1] for p in pts]
    cols = ['green' if v == min(vals) else 'red' for v in vals]
    for p, c in zip(pts, cols):
        ax.scatter(*p, color=c, zorder=5, s=60)
    ax.set_title('Objective + Constraint')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ---- bottom: dual function D(lambda) ----
    ax = axes[1]
    def dual(lam):
        if lam < 0.5:
            return -np.inf
        # x1=x2=-1/(2*lam+1)
        u = -2/(2*lam+1)
        # D(lam) = -(1/(2*lam+1) + lam)
        return -(1/(2*lam+1) + lam)

    lams = np.linspace(0.5, 5, 400)
    Dvals = np.array([dual(l) for l in lams])
    ax.plot(lams, Dvals, 'b-', lw=2, label=r'$\mathcal{D}(\lambda)$')
    lstar = 0.5*(np.sqrt(2)-1)   # approx
    Dstar = dual(max(lstar, 0.5))
    ax.axvline(max(lstar, 0.5), ls='--', color='gray', lw=1)
    ax.scatter([max(lstar, 0.5)], [Dstar], color='red', zorder=5, s=60)
    ax.annotate(r'$\lambda^*$', xy=(max(lstar,0.5), Dstar),
                xytext=(max(lstar,0.5)+0.3, Dstar+0.5), fontsize=10)
    ax.set_title(r'Dual Function $\mathcal{D}(\lambda)$')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\mathcal{D}(\lambda)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig_dual_function_ex2.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Duality gap illustration  (weak vs strong duality)
# ─────────────────────────────────────────────────────────────────────────────
def fig_duality_gap():
    fig, ax = plt.subplots(figsize=(6, 4))
    # Schematic: primal value p* and dual value d*
    ax.axhline(0, color='k', lw=0.5)

    # draw a simple diagram
    p_star = 3.0; d_star = 2.0
    ax.annotate('', xy=(p_star, 0.5), xytext=(d_star, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((p_star+d_star)/2, 0.55, 'duality gap', ha='center',
            fontsize=10, color='red')
    ax.scatter([p_star], [0], s=80, color='steelblue', zorder=5)
    ax.scatter([d_star], [0], s=80, color='darkorange', zorder=5)
    ax.text(p_star, -0.07, r'$p^*$ (primal)', ha='center', fontsize=10,
            color='steelblue')
    ax.text(d_star, -0.07, r'$d^*$ (dual)', ha='center', fontsize=10,
            color='darkorange')

    # strong duality annotation
    ax.annotate('Strong duality:\n$d^* = p^*$', xy=(4.5, 0.5),
                fontsize=9, color='green',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.6))

    ax.set_xlim(0, 6); ax.set_ylim(-0.25, 1.0)
    ax.axis('off')
    ax.set_title('Weak vs. Strong Duality')
    savefig("fig_duality_gap.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Primal-dual interior point – residual convergence  (Example 11.3)
#   minimize x^2,  s.t. (x-3)^2 <= 1
# ─────────────────────────────────────────────────────────────────────────────
def fig_primal_dual_residual():
    """Simulate primal-dual interior point on minimize x^2 s.t. (x-3)^2<=1."""
    def f(x): return x**2
    def gradf(x): return np.array([2*x])
    def hessf(x): return np.array([[2.0]])
    def g(x): return (x-3)**2 - 1
    def gradg(x): return np.array([2*(x-3)])
    def hessg(x): return np.array([[2.0]])

    x = np.array([0.0]); mu = np.array([1.0]); rho = 1.0
    residuals = []
    gaps = []
    for _ in range(60):
        r_x = gradf(x[0]) + mu[0]*gradg(x[0])
        r_mu = -mu[0]*g(x[0]) - 1/rho
        res = np.sqrt(r_x**2 + r_mu**2)
        gap = -mu[0]*g(x[0])
        residuals.append(float(res)); gaps.append(float(gap))
        # Newton step
        J = np.array([[hessf(x[0])[0,0] + mu[0]*hessg(x[0])[0,0], gradg(x[0])[0]],
                      [-mu[0]*gradg(x[0])[0], -g(x[0])]])
        try:
            d = -np.linalg.solve(J, np.array([r_x[0], r_mu]))
        except np.linalg.LinAlgError:
            break
        alpha = 1.0
        for _ in range(20):
            xn = x[0] + alpha*d[0]; mun = mu[0] + alpha*d[1]
            if mun > 0 and g(xn) < 0:
                break
            alpha *= 0.5
        x = np.array([x[0]+alpha*d[0]])
        mu = np.array([max(1e-12, mu[0]+alpha*d[1])])
        rho *= 1.5

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(residuals, 'b-o', ms=3, label='Residual $\|\\mathbf{r}\|$')
    ax.semilogy(gaps, 'r--s', ms=3, label='Duality gap $m/\\rho$')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Value (log scale)')
    ax.set_title('Primal-Dual Interior Point Convergence')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    savefig("fig_primal_dual_convergence.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Dual ascent  –  update diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_dual_ascent_diagram():
    """Illustrate dual ascent: maximize D(mu) for minimize x^2+y^2 s.t. x+y=1."""
    # D(lambda) = -1/2 * lambda^2 + lambda  for equality constrained problem
    # Primal: min x^2+y^2, s.t. x+y=1  => x=y=0.5, p*=0.5
    # Dual: D(lambda) = min_{x,y} x^2+y^2+lambda*(x+y-1)
    #      = -lambda^2/2 - lambda  ... actually D(lam)= -lam^2/2 + lam? Let's compute:
    # grad=0: 2x+lam=0 => x=-lam/2; same y=-lam/2
    # D(lam) = 2*(lam/2)^2 + lam*(-lam-1) = lam^2/2 - lam^2 - lam = -lam^2/2 - lam
    # Maximize: d/dlam D = -lam - 1 = 0 => lam=-1 => D(-1)=-1/2+1=1/2 ✓
    lams = np.linspace(-3, 1, 400)
    D = -lams**2/2 - lams

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(lams, D, 'b-', lw=2)
    ax.scatter([-1], [0.5], color='red', zorder=5, s=80, label=r'$\lambda^*=-1$')
    ax.axvline(-1, ls='--', color='gray', lw=1)
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\mathcal{D}(\lambda)$')
    ax.set_title(r'Dual Function $\mathcal{D}(\lambda)$')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Dual ascent trace
    ax = axes[1]
    lam = np.array([-3.0])
    alpha = 0.3
    lam_hist = [float(lam)]
    for _ in range(12):
        # gradient of D w.r.t. lam: x*(lam) = -lam/2, y=-lam/2
        # grad_D = x+y-1 = -lam-1
        x_opt = -lam/2; y_opt = -lam/2
        grad_D = x_opt + y_opt - 1   # constraint violation
        lam = lam + alpha * grad_D
        lam_hist.append(float(lam))

    ax.plot(lam_hist, -np.array(lam_hist)**2/2 - np.array(lam_hist), 'ro-',
            ms=6, label='Dual ascent iterates')
    ax.plot(lams, D, 'b-', lw=1.5, alpha=0.7, label=r'$\mathcal{D}(\lambda)$')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\mathcal{D}(\lambda)$')
    ax.set_title('Dual Ascent Iterations')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig_dual_ascent.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: ADMM convergence  (Figure 11.2 from book – flower function)
#   minimize ||x1||^2,  s.t. ||x2||^2 <= 2  in ADMM form
#   We replicate the residual plot showing primal & dual residuals converging
# ─────────────────────────────────────────────────────────────────────────────
def fig_admm_convergence():
    """ADMM on minimize exp(x), s.t. (x-2)^2 <= 1  (Example 11.4)."""
    rho = 1.0; gamma = 1.5
    x1 = 0.0; x2 = 0.0; u = 0.0
    primal_res = []; dual_res = []

    for k in range(40):
        v = x2 - u
        # x1-update: argmin exp(x1) + rho/2*(x1-v)^2
        res = minimize_scalar(lambda x: np.exp(x) + 0.5*rho*(x-v)**2,
                              bounds=(-5, 10), method='bounded')
        x1_new = res.x
        # x2-update: argmin f2(x2) + lam*(x1'-x2) + rho/2*(x1'-x2)^2
        # f2 = indicator (x2-2)^2<=1, so project x1_new+u onto [1,3]
        x2_new = np.clip(x1_new + u, 1.0, 3.0)
        u_new = u + x1_new - x2_new
        r = abs(x1_new - x2_new)
        s = abs(rho*(x2_new - x2))
        primal_res.append(r); dual_res.append(s)
        x1 = x1_new; x2 = x2_new; u = u_new
        rho *= gamma

    fig, ax = plt.subplots(figsize=(6, 3.5))
    iters = np.arange(1, len(primal_res)+1)
    ax.semilogy(iters, primal_res, 'b-', lw=2, label=r'Primal residual $\|\mathbf{r}\|$')
    ax.semilogy(iters, dual_res, 'r--', lw=2, label=r'Dual residual $\|\mathbf{s}\|$')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Residual (log scale)')
    ax.set_title('ADMM Convergence (Example 11.4)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    savefig("fig_admm_convergence.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Soft thresholding operator S_kappa  (Figure 11.5 from book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_soft_threshold():
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-4, 4, 800)
    kappa = 2.0
    Sx = np.where(x < -kappa, x + kappa, np.where(np.abs(x) <= kappa, 0, x - kappa))
    ax.plot(x, Sx, 'steelblue', lw=2.5, label=r'$S_\kappa(x),\;\kappa=2$')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.plot([-kappa, kappa], [0, 0], 'ro', ms=6, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel(r'$S_\kappa(x)$')
    ax.set_title(r'Soft Thresholding Operator ($\kappa=2$)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    savefig("fig_soft_threshold.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Huber function  (Figure 11.6 from book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_huber():
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-4, 4, 800)
    huber = np.where(np.abs(x) <= 1, 0.5*x**2, np.abs(x) - 0.5)
    ax.plot(x, huber, 'steelblue', lw=2.5, label='huber$(x)$')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('huber$(x)$')
    ax.set_title('Huber Loss Function')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    savefig("fig_huber.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: ADMM applied to LASSO  – sparsity of solution
# ─────────────────────────────────────────────────────────────────────────────
def fig_lasso():
    np.random.seed(42)
    m, n = 30, 60
    A = np.random.randn(m, n)
    x_true = np.zeros(n); x_true[:5] = np.array([3., -2., 1.5, -1., 2.])
    b = A @ x_true + 0.1*np.random.randn(m)

    # ADMM lasso
    lam = 0.5; rho = 1.0
    x1 = np.zeros(n); x2 = np.zeros(n); u = np.zeros(n)

    AtA = A.T @ A
    Atb = A.T @ b

    def soft(v, kappa):
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)

    residuals = []
    for _ in range(80):
        x1 = np.linalg.solve(AtA + rho*np.eye(n), Atb + rho*(x2 - u))
        x2 = soft(x1 + u, lam/rho)
        u = u + x1 - x2
        residuals.append(np.linalg.norm(x1 - x2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.stem(x_true, linefmt='b-', markerfmt='bo', basefmt='k-', label='True $x$')
    ax.stem(x1, linefmt='r--', markerfmt='r^', basefmt='k-', label='ADMM LASSO')
    ax.set_xlabel('Index'); ax.set_ylabel('Coefficient')
    ax.set_title(f'LASSO: $\\lambda={lam}$'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(residuals, 'b-', lw=2)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Primal residual (log)')
    ax.set_title('ADMM-LASSO Convergence')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig_lasso.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Consensus ADMM  – points converging to mean
# ─────────────────────────────────────────────────────────────────────────────
def fig_consensus():
    """Consensus: minimize sum of L2 distances to k fixed target points."""
    np.random.seed(7)
    k = 6
    targets = np.random.randn(k, 2) * 1.5
    # analytical solution: mean
    mean_pt = targets.mean(axis=0)

    rho = 0.5; gamma = 1.3
    x1s = targets.copy()   # each x1^(i) starts at its target
    x2 = np.zeros(2)
    lams = np.zeros((k, 2))

    snapshots = [x2.copy()]
    for it in range(15):
        # x1^(i) update: argmin ||x1-t_i||^2 + lam^(i)T(x1-x2) + rho/2||x1-x2||^2
        for i in range(k):
            x1s[i] = (2*targets[i] + rho*(x2 - lams[i]/rho)) / (2 + rho)
        # x2 update: mean consensus step
        x2 = (x1s + lams/rho).mean(axis=0)
        # lam update
        lams += rho * (x1s - x2)
        rho *= gamma
        if it % 3 == 2:
            snapshots.append(x2.copy())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets[:, 0], targets[:, 1], s=80, color='k', zorder=5,
               label='Target points')
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(snapshots)))
    for i, (snap, col) in enumerate(zip(snapshots, colors)):
        ax.scatter(*snap, s=100, color=col, zorder=6,
                   label=f'$\\mathbf{{x}}_2$ iter {i*3}' if i < 3 else None)
    ax.scatter(*mean_pt, s=120, color='red', marker='*', zorder=7,
               label='Analytical mean')
    ax.set_title('Consensus ADMM: Converging to Mean')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    savefig("fig_consensus.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: ADMM algorithm schematic (block diagram)
# ─────────────────────────────────────────────────────────────────────────────
def fig_admm_block():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    def draw_box(ax, xy, w, h, text, color='lightblue'):
        x, y = xy
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0.05',
                                       fc=color, ec='steelblue', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=9, wrap=True)

    draw_box(ax, (0.05, 0.3), 0.22, 0.4, r'$\mathbf{x}_1^{(k+1)}=$'+'\n'+
             r'$\arg\min_{\mathbf{x}_1}\mathcal{L}_\rho$', 'lightblue')
    draw_box(ax, (0.38, 0.3), 0.22, 0.4, r'$\mathbf{x}_2^{(k+1)}=$'+'\n'+
             r'$\arg\min_{\mathbf{x}_2}\mathcal{L}_\rho$', 'lightyellow')
    draw_box(ax, (0.71, 0.3), 0.22, 0.4,
             r'$\boldsymbol{\lambda}^{(k+1)}=$'+'\n'+
             r'$\boldsymbol{\lambda}^{(k)}+\rho\,\mathbf{r}^{(k+1)}$', 'lightgreen')

    for x0, x1 in [(0.27, 0.38), (0.60, 0.71)]:
        ax.annotate('', xy=(x1, 0.5), xytext=(x0, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('ADMM Iteration Structure', fontsize=11)
    savefig("fig_admm_block.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Lagrangian duality – primal vs dual feasible region schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_lagrangian_overview():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')

    # Layered text diagram
    items = [
        (0.5, 0.88, r'Primal: $\min_{\mathbf{x}} f(\mathbf{x})$  s.t. $g_i(\mathbf{x})\leq 0,\;h_j(\mathbf{x})=0$', 12, 'lightcoral'),
        (0.5, 0.68, r'Lagrangian: $\mathcal{L}(\mathbf{x},\mu,\lambda) = f(\mathbf{x})+\sum_i\mu_i g_i(\mathbf{x})+\sum_j\lambda_j h_j(\mathbf{x})$', 10, 'lightyellow'),
        (0.5, 0.48, r'Dual function: $\mathcal{D}(\mu,\lambda) = \min_{\mathbf{x}}\mathcal{L}(\mathbf{x},\mu,\lambda)$', 10, 'lightblue'),
        (0.5, 0.28, r'Dual problem: $\max_{\mu\geq 0,\lambda} \mathcal{D}(\mu,\lambda)$', 10, 'lightgreen'),
        (0.5, 0.08, r'Weak duality: $d^* \leq p^*$   Strong duality: $d^* = p^*$ (convex + CQ)', 10, 'lavender'),
    ]
    for x, y, txt, fs, col in items:
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.4', fc=col, ec='gray', alpha=0.85),
                transform=ax.transAxes)
        if y > 0.1:
            ax.annotate('', xy=(x, y-0.12), xytext=(x, y-0.05),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=1.3, color='gray'))

    ax.set_title('Lagrangian Duality Framework', fontsize=12, pad=10)
    savefig("fig_lagrangian_overview.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: Proximal minimization  –  projection onto convex set
# ─────────────────────────────────────────────────────────────────────────────
def fig_proximal():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    configs = [
        (np.array([0.5, 2.0]),  'v outside $\\mathcal{X}$'),
        (np.array([1.0, 1.0]),  'v on boundary'),
        (np.array([1.5, 0.5]),  'v inside $\\mathcal{X}$'),
    ]
    for ax, (v, title) in zip(axes, configs):
        # Feasible set: x1 in [0,2], x2 in [0,2]  (rotated square for visual)
        rect = plt.Polygon([[0,1],[1,0],[2,1],[1,2]], closed=True,
                           fc='lightblue', ec='steelblue', lw=1.5, alpha=0.5)
        ax.add_patch(rect)
        # projection: closest point in set to v
        # parametrize: project v onto the rotated square
        from scipy.optimize import minimize as sci_min
        def dist(xy):
            px,py = xy
            # check inside: |px-1|+|py-1| <= 1
            return (px-v[0])**2 + (py-v[1])**2
        def constr(xy):
            return 1 - (abs(xy[0]-1)+abs(xy[1]-1))
        from scipy.optimize import LinearConstraint
        res = sci_min(dist, [1.0,1.0], method='SLSQP',
                      constraints={'type':'ineq','fun':constr})
        proj = res.x
        ax.scatter(*v, s=80, color='red', zorder=5, label='$\\mathbf{v}$')
        ax.scatter(*proj, s=80, color='steelblue', zorder=5,
                   label="$\\mathbf{x}_1'$")
        ax.plot([v[0],proj[0]], [v[1],proj[1]], 'k--', lw=1.2)
        ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.3, 2.3)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle('Proximal Minimization: Projection onto $\\mathcal{X}$', y=1.02)
    fig.tight_layout()
    savefig("fig_proximal.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 14: Method of multipliers  vs  dual ascent
# ─────────────────────────────────────────────────────────────────────────────
def fig_method_of_multipliers():
    """Augmented Lagrangian vs plain dual ascent convergence."""
    # minimize 0.5*x^2, s.t. x=1  =>  x*=1, lambda*=-1
    x_star = 1.0
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Dual ascent (gradient ascent on D(lam) = -1/2*(lam+1)^2 + const)
    lam = -5.0; alpha = 0.3
    da_err = []
    for _ in range(30):
        x = -lam    # argmin 0.5x^2 + lam*x = -lam  (wait: grad=x+lam=0 => x=-lam)
        # wait: constraint x=1, so h(x)=x-1
        # L=0.5x^2+lam*(x-1), grad_x=x+lam=0 => x=-lam
        # but primal problem minimize 0.5x^2 s.t. x=1 has x*=1
        # dual ascent: lam = lam + alpha*(x-1)  where x=-lam
        lam = lam + alpha * (-lam - 1)
        da_err.append(abs(-lam - 1))   # |x - 1|

    # Method of multipliers  rho=2
    lam2 = -5.0; rho = 2.0
    mom_err = []
    for _ in range(30):
        # argmin 0.5x^2 + lam2*(x-1) + rho/2*(x-1)^2
        # grad = x + lam2 + rho*(x-1) = 0 => x*(1+rho) = 1*rho - lam2
        x = (rho - lam2) / (1 + rho)
        lam2 = lam2 + rho*(x - 1)
        mom_err.append(abs(x - 1))

    ax.semilogy(da_err, 'b-o', ms=4, label='Dual Ascent')
    ax.semilogy(mom_err, 'r-s', ms=4, label='Method of Multipliers ($\\rho=2$)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('$|x - x^*|$ (log)')
    ax.set_title('Dual Ascent vs Method of Multipliers')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    savefig("fig_method_of_multipliers.pdf", fig)

# ─────────────────────────────────────────────────────────────────────────────
# Run all figure generators
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Chapter 11 figures...")
    fig_dual_lower_bound()
    fig_dual_function_ex2()
    fig_duality_gap()
    fig_primal_dual_residual()
    fig_dual_ascent_diagram()
    fig_admm_convergence()
    fig_soft_threshold()
    fig_huber()
    fig_lasso()
    fig_consensus()
    fig_admm_block()
    fig_lagrangian_overview()
    fig_proximal()
    fig_method_of_multipliers()
    print("All figures generated successfully.")
