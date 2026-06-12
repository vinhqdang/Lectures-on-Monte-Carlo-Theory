"""
gen_figures.py  –  Generate all figures for Chapter 6: Second-Order Methods
Requires: matplotlib, numpy, scipy, pymupdf (fitz)
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize
from scipy.linalg import solve

# ── output directory ───────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved {path}")


# ══════════════════════════════════════════════════════════════════════
# Figure 1 – First-order vs Second-order approximation (Fig 6.1)
# ══════════════════════════════════════════════════════════════════════
def fig_first_vs_second_order():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    x = np.linspace(-2.5, 2.0, 400)
    # A function with a local minimum not at the expansion point
    def f(x): return 0.3*x**4 - 0.5*x**3 - x**2 + 0.5

    xk = 1.2  # expansion point
    fxk = f(xk)

    # first derivative (numerical for simplicity)
    h = 1e-7
    fpk  = (f(xk+h) - f(xk-h)) / (2*h)
    fppk = (f(xk+h) - 2*f(xk) + f(xk-h)) / h**2

    # first-order approx
    q1 = fxk + fpk*(x - xk)
    # second-order approx
    q2 = fxk + fpk*(x - xk) + 0.5*fppk*(x - xk)**2

    for ax, q, title, color in zip(axes,
                                   [q1, q2],
                                   ["First-order approx.", "Second-order approx."],
                                   ["royalblue", "royalblue"]):
        ax.plot(x, f(x), 'k', lw=2, label="$f(x)$")
        ax.plot(x, q,    color=color, lw=2, ls='--', label=title)
        ax.axvline(xk, color='gray', lw=0.8, ls=':')
        ax.plot(xk, fxk, 'ko', ms=5)
        ax.set_xlim(-2.5, 2.0)
        ax.set_ylim(-2, 4)
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$f$", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_title(title, fontsize=10)

    fig.suptitle("First- vs. Second-Order Approximations", fontsize=12)
    plt.tight_layout()
    savefig("fig_first_vs_second_order.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 – Newton's method iterations on a 1D function
# ══════════════════════════════════════════════════════════════════════
def fig_newton_1d():
    fig, ax = plt.subplots(figsize=(6, 4))

    def f(x):  return x**4 - 3*x**3 + 2
    def fp(x): return 4*x**3 - 9*x**2
    def fpp(x): return 12*x**2 - 18*x

    x = np.linspace(-0.5, 3.2, 500)
    ax.plot(x, f(x), 'k', lw=2, label="$f(x)$")

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    xk = 2.5
    for i, c in enumerate(colors):
        xk_new = xk - fp(xk)/fpp(xk)
        # draw quadratic approx
        xq = np.linspace(xk - 0.8, xk + 0.8, 200)
        qx = f(xk) + fp(xk)*(xq - xk) + 0.5*fpp(xk)*(xq - xk)**2
        ax.plot(xq, qx, color=c, lw=1.2, ls='--', alpha=0.8)
        ax.plot(xk, f(xk), 'o', color=c, ms=6)
        ax.axvline(xk_new, color=c, lw=0.5, ls=':')
        xk = xk_new
        if abs(fp(xk)) < 1e-8:
            break

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.set_ylim(-5, 10)
    ax.set_title("Newton's Method – 1D Iterations", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("fig_newton_1d.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 – Newton's method: oscillation / overshoot / negative f''
# ══════════════════════════════════════════════════════════════════════
def fig_newton_behaviors():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Oscillation – flat region
    ax = axes[0]
    x = np.linspace(-3, 3, 400)
    f = lambda t: np.tanh(t)
    fp = lambda t: 1 - np.tanh(t)**2
    fpp = lambda t: -2*np.tanh(t)*(1 - np.tanh(t)**2)
    ax.plot(x, f(x), 'k', lw=2)
    xk = -2.0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for c in colors:
        xn = xk - fp(xk)/fpp(xk) if abs(fpp(xk)) > 1e-10 else xk - 0.1
        ax.annotate("", xy=(xn, f(xn)), xytext=(xk, f(xk)),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.plot(xk, f(xk), 'o', color=c, ms=5)
        xk = xn
    ax.set_title("Oscillation", fontsize=11)
    ax.set_xlabel("$x$"); ax.set_ylabel("$f$")

    # Overshoot – steep gradient
    ax = axes[1]
    f2 = lambda t: (t - 1)**2
    fp2 = lambda t: 2*(t - 1)
    fpp2 = lambda t: np.ones_like(np.atleast_1d(t)) * 2.0
    x2 = np.linspace(-1, 5, 400)
    ax.plot(x2, f2(x2), 'k', lw=2)
    xk2 = 4.5
    for c in colors[:3]:
        xn2 = xk2 - fp2(xk2)/fpp2(xk2)
        ax.annotate("", xy=(xn2, f2(xn2)), xytext=(xk2, f2(xk2)),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.plot(xk2, f2(xk2), 'o', color=c, ms=5)
        xk2 = xn2
    ax.set_title("Convergence (quadratic)", fontsize=11)
    ax.set_xlabel("$x$"); ax.set_ylabel("$f$")

    # Negative f'' – saddle / inflection
    ax = axes[2]
    f3 = lambda t: t**3 / 3
    fp3 = lambda t: t**2
    fpp3 = lambda t: 2*t
    x3 = np.linspace(-2, 2, 400)
    ax.plot(x3, f3(x3), 'k', lw=2)
    xk3 = -1.5
    for c in colors[:4]:
        d = fpp3(xk3)
        if abs(d) < 1e-8: break
        xn3 = xk3 - fp3(xk3)/d
        ax.annotate("", xy=(min(max(xn3, -2), 2), f3(min(max(xn3, -2), 2))),
                    xytext=(xk3, f3(xk3)),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.plot(xk3, f3(xk3), 'o', color=c, ms=5)
        xk3 = min(max(xn3, -1.99), 1.99)
    ax.set_title("Negative $f''$ region", fontsize=11)
    ax.set_xlabel("$x$"); ax.set_ylabel("$f$")

    fig.suptitle("Newton's Method Behaviors", fontsize=12)
    plt.tight_layout()
    savefig("fig_newton_behaviors.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 – Newton's method on Booth's function (Example 6.1)
# ══════════════════════════════════════════════════════════════════════
def fig_booth_newton():
    # Booth: f(x) = (x1 + 2x2 - 7)^2 + (2x1 + x2 - 5)^2, min at (1,3)
    def booth(x): return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    def grad_booth(x):
        return np.array([2*(x[0]+2*x[1]-7) + 4*(2*x[0]+x[1]-5),
                         4*(x[0]+2*x[1]-7) + 2*(2*x[0]+x[1]-5)])
    H = np.array([[10., 8.], [8., 10.]])

    x1 = np.linspace(-1, 12, 300)
    x2 = np.linspace(-1, 12, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 + 2*X2 - 7)**2 + (2*X1 + X2 - 5)**2

    fig, ax = plt.subplots(figsize=(5, 4.5))
    levels = [1, 5, 20, 50, 150, 400, 900]
    cs = ax.contour(X1, X2, Z, levels=levels, cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=7)

    # Trajectory: x0 = [9, 8]
    traj = [[9., 8.]]
    x = np.array([9., 8.])
    for _ in range(5):
        g = grad_booth(x)
        x = x - np.linalg.solve(H, g)
        traj.append(x.copy())
        if np.linalg.norm(g) < 1e-10:
            break

    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], 'ro-', ms=6, lw=1.5, label='Newton path')
    ax.plot(1, 3, 'k*', ms=12, label='Minimum $(1,3)$')
    ax.plot(9, 8, 'gs', ms=8, label='Start $(9,8)$')
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Newton's Method on Booth's Function", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("fig_booth_newton.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 5 – Secant method illustration (1D)
# ══════════════════════════════════════════════════════════════════════
def fig_secant_method():
    fig, ax = plt.subplots(figsize=(6, 4))

    def f(x):  return x**3 - 2*x - 5
    def fp(x): return 3*x**2 - 2

    x = np.linspace(1.5, 2.8, 400)
    ax.axhline(0, color='gray', lw=0.8)
    ax.plot(x, fp(x), 'k', lw=2, label="$f'(x)$")

    xprev, xcurr = 2.5, 2.3
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, 4))
    for c in colors:
        gp = fp(xprev)
        gc = fp(xcurr)
        denom = gc - gp
        if abs(denom) < 1e-12:
            break
        xnext = xcurr - (xcurr - xprev) / denom * gc
        # draw secant line of f'
        xs_line = np.array([xprev, xcurr])
        ys_line = np.array([gp, gc])
        ax.plot(xs_line, ys_line, '--', color=c, lw=1.5)
        ax.plot(xcurr, gc, 'o', color=c, ms=6)
        ax.axvline(xnext, color=c, lw=0.8, ls=':')
        xprev, xcurr = xcurr, min(max(xnext, 1.5), 2.8)

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$f'(x)$", fontsize=12)
    ax.set_title("Secant Method: Approximating $f''$ from $f'$", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("fig_secant_method.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 6 – Levenberg-Marquardt: damping interpolation sketch
# ══════════════════════════════════════════════════════════════════════
def fig_lm_damping():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    deltas = np.logspace(-2, 3, 300)

    # Conceptual: step norm as function of delta
    # Large delta => small step (gradient descent direction)
    # Small delta => Newton step

    def step_norm(d, H_eig_min=0.5, H_eig_max=3.0, g_norm=2.0):
        # ||( H + delta I)^{-1} g|| approximately
        # eigenvalues of (H + delta I) are eig_H + delta
        # rough estimate: g_norm / (eig_mean + delta)
        eig_mean = (H_eig_min + H_eig_max) / 2
        return g_norm / (eig_mean + d)

    ax.semilogx(deltas, step_norm(deltas), 'b', lw=2, label=r'$\|$step$\|$ (decreases with $\delta$)')
    ax.axhline(step_norm(0.0), color='green', ls='--', lw=1.5, label='Newton step size')
    ax.axhline(0, color='black', lw=0.5)

    ax.set_xlabel(r"Damping factor $\delta$", fontsize=12)
    ax.set_ylabel("Step norm", fontsize=12)
    ax.set_title("Levenberg-Marquardt: Effect of Damping", fontsize=11)
    ax.legend(fontsize=9)
    ax.annotate("Gradient\nDescent", xy=(100, step_norm(100)),
                xytext=(200, step_norm(100)+0.15),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')
    ax.annotate("Newton\nStep", xy=(0.02, step_norm(0.0)),
                xytext=(0.05, step_norm(0.0)+0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')
    plt.tight_layout()
    savefig("fig_lm_damping.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 7 – LM convergence: f(x) vs iterations (Wheeler's Ridge style)
# ══════════════════════════════════════════════════════════════════════
def fig_lm_convergence():
    """Simulate LM on a simple 2D function (Wheeler's Ridge style)."""
    # Wheeler's Ridge: f(x1,x2) = -exp(-(x1*x2 - a)^2 - (x2 - b)^2)
    a, b = 3, 2

    def f(x):
        return -np.exp(-(x[0]*x[1] - a)**2 - (x[1] - b)**2)

    def grad_f(x):
        u = x[0]*x[1] - a
        v = x[1] - b
        e = np.exp(-u**2 - v**2)
        g1 = 2*u*x[1]*e
        g2 = (2*u*x[0] + 2*v)*e
        return np.array([g1, g2])

    def hess_f(x):
        u = x[0]*x[1] - a
        v = x[1] - b
        e = np.exp(-u**2 - v**2)
        h11 = (-2*x[1]**2 + 4*u**2*x[1]**2)*e
        h12 = (-2*(x[0]*x[1]-a) - 2*x[1]*(x[0]) + 4*u**2*x[0]*x[1])*e
        h21 = h12
        h22 = (-2*(x[0]**2 + 1) + 4*(u*x[0]+v)**2)*e
        return np.array([[h11, h12], [h21, h22]])

    def lm_step(x, delta, gamma_acc=0.1, gamma_rej=10.0, eps=1e-6):
        H = hess_f(x)
        g = grad_f(x)
        d = np.diag(np.maximum(np.diag(H), eps))
        M = H + delta * d
        try:
            step = np.linalg.solve(M, g)
        except np.linalg.LinAlgError:
            step = g
        xnew = x - step
        if f(xnew) < f(x):
            return xnew, delta * gamma_acc
        else:
            return x, delta * gamma_rej

    # Run LM (with diagonal)
    x0 = np.array([2.5, 1.5])
    x = x0.copy()
    delta = 1.0
    hist_lm = [f(x)]
    for _ in range(30):
        x, delta = lm_step(x, delta)
        hist_lm.append(f(x))

    # Run LM (without diagonal, plain identity damping)
    x2 = x0.copy()
    delta2 = 1.0
    hist_lm2 = [f(x2)]
    for _ in range(30):
        H2 = hess_f(x2)
        g2 = grad_f(x2)
        M2 = H2 + delta2 * np.eye(2)
        try:
            step2 = np.linalg.solve(M2, g2)
        except np.linalg.LinAlgError:
            step2 = g2
        x2new = x2 - step2
        if f(x2new) < f(x2):
            x2, delta2 = x2new, delta2 * 0.1
        else:
            delta2 = delta2 * 10.0
        hist_lm2.append(f(x2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(hist_lm,  'k-',  lw=2, label='with diag(H)')
    ax.plot(hist_lm2, 'gray', lw=1.5, ls='--', label='plain $\\delta I$')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("$f(\\mathbf{x})$", fontsize=12)
    ax.set_title("LM Convergence", fontsize=11)
    ax.legend(fontsize=9)

    # Contour + trajectories
    ax2 = axes[1]
    x1v = np.linspace(0.5, 3.5, 200)
    x2v = np.linspace(0.5, 3.0, 200)
    X1, X2 = np.meshgrid(x1v, x2v)
    Z = -np.exp(-(X1*X2 - a)**2 - (X2 - b)**2)
    ax2.contour(X1, X2, Z, 15, cmap='viridis')

    # rerun to get full trajectory
    traj1, traj2 = [x0.copy()], [x0.copy()]
    xr, dr = x0.copy(), 1.0
    xr2, dr2 = x0.copy(), 1.0
    for _ in range(25):
        xr, dr = lm_step(xr, dr)
        traj1.append(xr.copy())
        H2 = hess_f(xr2); g2 = grad_f(xr2)
        M2 = H2 + dr2 * np.eye(2)
        try: step2 = np.linalg.solve(M2, g2)
        except: step2 = g2
        xr2new = xr2 - step2
        if f(xr2new) < f(xr2): xr2, dr2 = xr2new, dr2*0.1
        else: dr2 = dr2*10.0
        traj2.append(xr2.copy())

    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    ax2.plot(traj1[:,0], traj1[:,1], 'k+', ms=5, label='with diag')
    ax2.plot(traj2[:,0], traj2[:,1], 'ko', ms=4, label='w/o diag')
    ax2.set_xlabel("$x_1$", fontsize=12)
    ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_title("LM Trajectories on Wheeler's Ridge", fontsize=11)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    savefig("fig_lm_convergence.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 8 – LM for Sum of Squares: Jacobian/outer product illustration
# ══════════════════════════════════════════════════════════════════════
def fig_lm_least_squares():
    """Show LM fitting a curve: sum-of-squares residuals."""
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 20)
    y_true = np.sin(t)
    y_obs = y_true + 0.15 * np.random.randn(len(t))

    # Model: a*sin(b*t + c)
    def model(params, t):
        return params[0] * np.sin(params[1]*t + params[2])

    def residuals(params):
        return model(params, t) - y_obs

    from scipy.optimize import least_squares
    p0 = [0.8, 0.9, 0.1]
    result = least_squares(residuals, p0, method='lm')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.scatter(t, y_obs, s=20, c='gray', zorder=5, label='Observations')
    ax.plot(t, y_true, 'k--', lw=1.5, label='True $\\sin(t)$')
    ax.plot(t, model(result.x, t), 'b-', lw=2, label='LM fit')
    ax.plot(t, model(p0, t), 'r--', lw=1.5, label='Initial guess')
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("LM: Curve Fitting (Sum of Squares)", fontsize=11)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    # Show cost vs iteration (simulate manually)
    p = np.array(p0, dtype=float)
    costs = [0.5 * np.sum(residuals(p)**2)]
    delta = 0.1
    for _ in range(30):
        r = residuals(p)
        # finite-difference Jacobian
        J = np.zeros((len(t), 3))
        eps = 1e-6
        for j in range(3):
            dp = np.zeros(3); dp[j] = eps
            J[:, j] = (residuals(p + dp) - residuals(p - dp)) / (2*eps)
        Htilde = J.T @ J
        gtilde = J.T @ r
        d = np.diag(np.maximum(np.diag(Htilde), 1e-6))
        M = Htilde + delta * d
        try:
            step = np.linalg.solve(M, gtilde)
        except:
            step = gtilde
        pnew = p - step
        cnew = 0.5 * np.sum(residuals(pnew)**2)
        if cnew < costs[-1]:
            p, delta = pnew, delta * 0.1
        else:
            delta *= 10.0
        costs.append(0.5 * np.sum(residuals(p)**2))

    ax2.semilogy(costs, 'b-o', ms=3, lw=1.5)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Cost $\\frac{1}{2}\\|\\mathbf{r}\\|^2$", fontsize=12)
    ax2.set_title("LM: Cost Convergence", fontsize=11)
    plt.tight_layout()
    savefig("fig_lm_least_squares.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 9 – Quasi-Newton methods compared (Fig 6.6)
# ══════════════════════════════════════════════════════════════════════
def fig_quasi_newton_comparison():
    """Compare DFP, BFGS, L-BFGS on 2D Rosenbrock-style function."""
    # Extended Rosenbrock (2D for illustration)
    def rosenbrock(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    def grad_rosen(x):
        g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        g2 = 200*(x[1] - x[0]**2)
        return np.array([g1, g2])

    x0 = np.array([-1.5, 1.0])

    def line_search_wolfe(f, g, x, d):
        """Simple backtracking line search."""
        alpha = 1.0
        c1 = 1e-4
        fx = f(x)
        gd = g(x) @ d
        for _ in range(50):
            if f(x + alpha*d) <= fx + c1*alpha*gd:
                break
            alpha *= 0.5
        return alpha

    # BFGS
    def run_bfgs(f, g, x0, max_iter=100):
        x = x0.copy()
        n = len(x)
        Q = np.eye(n)  # inv Hessian approx
        fvals = [f(x)]
        for _ in range(max_iter):
            gk = g(x)
            if np.linalg.norm(gk) < 1e-8:
                break
            d = -Q @ gk
            alpha = line_search_wolfe(f, g, x, d)
            xnew = x + alpha*d
            gnew = g(xnew)
            delta = xnew - x
            gamma = gnew - gk
            dg = delta @ gamma
            if abs(dg) < 1e-10:
                x = xnew
                fvals.append(f(x))
                continue
            # BFGS update
            rho = 1.0 / dg
            A = np.eye(n) - rho * np.outer(delta, gamma)
            B = np.eye(n) - rho * np.outer(gamma, delta)
            Q = A @ Q @ B + rho * np.outer(delta, delta)
            x = xnew
            fvals.append(f(x))
        return fvals

    # DFP
    def run_dfp(f, g, x0, max_iter=100):
        x = x0.copy()
        n = len(x)
        Q = np.eye(n)
        fvals = [f(x)]
        for _ in range(max_iter):
            gk = g(x)
            if np.linalg.norm(gk) < 1e-8:
                break
            d = -Q @ gk
            alpha = line_search_wolfe(f, g, x, d)
            xnew = x + alpha*d
            gnew = g(xnew)
            delta = xnew - x
            gamma = gnew - gk
            dg = delta @ gamma
            if abs(dg) < 1e-10:
                x = xnew; fvals.append(f(x)); continue
            # DFP update
            Q = Q - (Q @ np.outer(gamma, gamma) @ Q) / (gamma @ Q @ gamma) \
                  + np.outer(delta, delta) / dg
            x = xnew
            fvals.append(f(x))
        return fvals

    # L-BFGS
    def run_lbfgs(f, g, x0, m=5, max_iter=100):
        x = x0.copy()
        deltas_hist, gammas_hist = [], []
        fvals = [f(x)]
        for _ in range(max_iter):
            gk = g(x)
            if np.linalg.norm(gk) < 1e-8:
                break
            # two-loop recursion
            q = gk.copy()
            alphas = []
            for i in range(len(deltas_hist)-1, -1, -1):
                d_i = deltas_hist[i]; y_i = gammas_hist[i]
                a_i = (d_i @ q) / (d_i @ y_i + 1e-15)
                q = q - a_i * y_i
                alphas.append(a_i)
            alphas.reverse()
            if deltas_hist:
                d_m = deltas_hist[-1]; y_m = gammas_hist[-1]
                z = (d_m @ y_m) / (y_m @ y_m + 1e-15) * q
            else:
                z = q.copy()
            for i in range(len(deltas_hist)):
                d_i = deltas_hist[i]; y_i = gammas_hist[i]
                beta = (y_i @ z) / (d_i @ y_i + 1e-15)
                z = z + d_i * (alphas[i] - beta)
            d = -z
            alpha = line_search_wolfe(f, g, x, d)
            xnew = x + alpha*d
            gnew = g(xnew)
            delta = xnew - x
            gamma = gnew - gk
            deltas_hist.append(delta)
            gammas_hist.append(gamma)
            if len(deltas_hist) > m:
                deltas_hist.pop(0); gammas_hist.pop(0)
            x = xnew
            fvals.append(f(x))
        return fvals

    bfgs_hist = run_bfgs(rosenbrock, grad_rosen, x0)
    dfp_hist  = run_dfp(rosenbrock, grad_rosen, x0)
    lbfgs5    = run_lbfgs(rosenbrock, grad_rosen, x0, m=5)
    lbfgs3    = run_lbfgs(rosenbrock, grad_rosen, x0, m=3)
    lbfgs1    = run_lbfgs(rosenbrock, grad_rosen, x0, m=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(bfgs_hist, 'k',  lw=2, label='BFGS')
    ax.semilogy(dfp_hist,  'k--', lw=2, label='DFP')
    ax.semilogy(lbfgs5, 'g-',  lw=1.5, label='L-BFGS ($m=5$)')
    ax.semilogy(lbfgs3, 'c-',  lw=1.5, label='L-BFGS ($m=3$)')
    ax.semilogy(lbfgs1, 'b-',  lw=1.5, label='L-BFGS ($m=1$)')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective function value", fontsize=12)
    ax.set_title("Quasi-Newton Methods: Rosenbrock (2D)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(15, max(len(bfgs_hist), len(dfp_hist), len(lbfgs5))))
    plt.tight_layout()
    savefig("fig_quasi_newton_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 10 – BFGS Hessian update illustration
# ══════════════════════════════════════════════════════════════════════
def fig_bfgs_update():
    """Visualize secant condition: H*delta = gamma."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 2D function: simple quadratic f(x) = x^T A x
    A = np.array([[3., 1.], [1., 2.]])
    x0 = np.array([2., 1.])
    g0 = 2 * A @ x0
    x1 = np.array([1., 0.5])
    g1 = 2 * A @ x1

    delta = x1 - x0
    gamma = g1 - g0

    ax.annotate("", xy=x1, xytext=x0,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate("", xy=x0 + 0.4*g0/np.linalg.norm(g0),
                xytext=x0,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate("", xy=x1 + 0.4*g1/np.linalg.norm(g1),
                xytext=x1,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.plot(*x0, 'bo', ms=8, label=r'$\mathbf{x}^{(k)}$')
    ax.plot(*x1, 'bs', ms=8, label=r'$\mathbf{x}^{(k+1)}$')

    ax.text(x0[0]-0.1, x0[1]-0.15, r'$\mathbf{x}^{(k)}$', fontsize=11)
    ax.text(x1[0]+0.05, x1[1]+0.05, r'$\mathbf{x}^{(k+1)}$', fontsize=11)
    ax.text(1.55, 0.85, r'$\boldsymbol{\delta} = \mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}$',
            color='blue', fontsize=10)

    ax.set_xlim(0.5, 2.8)
    ax.set_ylim(0.1, 1.8)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Secant Condition: $\\mathbf{H}\\boldsymbol{\\delta} = \\boldsymbol{\\gamma}$", fontsize=12)

    legend_elements = [
        mpatches.Patch(color='blue',  label=r'Step $\boldsymbol{\delta}$'),
        mpatches.Patch(color='red',   label=r'Gradient $\mathbf{g}^{(k)}$'),
        mpatches.Patch(color='green', label=r'Gradient $\mathbf{g}^{(k+1)}$'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    savefig("fig_bfgs_update.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 11 – SR1 vs BFGS Hessian update comparison
# ══════════════════════════════════════════════════════════════════════
def fig_sr1_vs_bfgs():
    """Show SR1 update formula visually."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Simple 2D rosenbrock
    def f(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def g(x):
        return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                          200*(x[1]-x[0]**2)])

    x1v = np.linspace(-2, 2, 200)
    x2v = np.linspace(-1, 3, 200)
    X1, X2 = np.meshgrid(x1v, x2v)
    Z = (1-X1)**2 + 100*(X2-X1**2)**2

    for ax, title in zip(axes, ['SR1', 'BFGS']):
        ax.contour(X1, X2, np.log(1+Z), 20, cmap='Blues_r')
        ax.plot(1, 1, 'r*', ms=12, label='Minimum')
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.set_title(f"{title} Update", fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle("SR1 vs BFGS: Rosenbrock Contours", fontsize=12)
    plt.tight_layout()
    savefig("fig_sr1_vs_bfgs.pdf")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Chapter 6: Second-Order Methods")
    fig_first_vs_second_order()
    fig_newton_1d()
    fig_newton_behaviors()
    fig_booth_newton()
    fig_secant_method()
    fig_lm_damping()
    fig_lm_convergence()
    fig_lm_least_squares()
    fig_quasi_newton_comparison()
    fig_bfgs_update()
    fig_sr1_vs_bfgs()
    print("All figures generated successfully.")
