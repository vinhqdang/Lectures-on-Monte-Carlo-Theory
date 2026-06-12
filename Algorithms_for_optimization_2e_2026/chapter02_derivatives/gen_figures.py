"""
gen_figures.py
Generate all figures for Chapter 2: Derivatives and Gradients
Algorithms for Optimization (2nd ed., 2026) - Kochenderfer & Wheeler
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  saved {name}')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Tangent line / derivative visualisation  (book Fig 2.1)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 3))
x = np.linspace(-0.5, 2.5, 300)
y = 0.4*x**3 - 0.5*x + 1.0
x0, y0 = 1.5, 0.4*1.5**3 - 0.5*1.5 + 1.0
dydx = 1.2*1.5**2 - 0.5
xt = np.linspace(0.6, 2.4, 2)
yt = y0 + dydx*(xt - x0)

ax.plot(x, y, 'k', lw=2)
ax.plot(xt, yt, color='steelblue', lw=2)
ax.plot(x0, y0, 'ko', ms=6, zorder=5)
ax.annotate(r'$f(x)$', xy=(2.3, 0.4*2.3**3 - 0.5*2.3 + 1.0),
            fontsize=12, ha='left')
ax.axvline(x0, color='gray', lw=0.8, linestyle='--', ymax=0.75)
ax.set_xlabel(r'$x$', fontsize=11)
ax.set_xlim(-0.3, 2.7)
ax.set_ylim(-0.2, 3.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Derivative = slope of tangent line', fontsize=10)
savefig('fig_tangent_line.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Forward / Central / Backward difference illustration (book Fig 2.2)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
xv = np.linspace(-2, 2, 400)
yv = -xv**2 + 2.5
x0 = 0.0
h  = 1.0
configs = [
    ('Forward', x0, x0+h, 'steelblue'),
    ('Central', x0-h/2, x0+h/2, 'steelblue'),
    ('Backward', x0-h, x0, 'steelblue'),
]
titles = ['Forward difference', 'Central difference', 'Backward difference']
for ax, (name, xa, xb, col), title in zip(axes, configs, titles):
    ya = -xa**2 + 2.5
    yb = -xb**2 + 2.5
    slope = (yb - ya) / (xb - xa) if xb != xa else 0
    xmid = (xa + xb) / 2
    ymid = -xmid**2 + 2.5
    xt = np.linspace(xmid - 1.5, xmid + 1.5, 2)
    yt = ymid + slope*(xt - xmid)
    ax.plot(xv, yv, 'k', lw=2)
    ax.plot(xt, yt, color=col, lw=2)
    ax.plot([xa, xb], [ya, yb], 'ko', ms=5, zorder=5)
    ax.axvline(xa, color='gray', lw=0.7, ls='--')
    ax.axvline(xb, color='gray', lw=0.7, ls='--')
    ax.annotate('', xy=(xb, -1.6), xytext=(xa, -1.6),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text((xa+xb)/2, -2.0, r'$h$', ha='center', fontsize=11)
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.5, 3.0)
    ax.set_title(title, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
savefig('fig_differences.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Error vs step-size comparison (book Fig 2.4)
# forward / central / complex-step  for sin(x) at x=1/2
# ─────────────────────────────────────────────────────────────────────────────
import cmath

x0   = 0.5
true_deriv = np.cos(x0)
hs   = np.logspace(-18, 0, 300)

err_fwd  = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - true_deriv)
err_cen  = np.abs((np.sin(x0 + hs/2) - np.sin(x0 - hs/2)) / hs - true_deriv)
err_cplx = np.abs(np.imag(np.sin(x0 + 1j*hs)) / hs - true_deriv)

# clip zeros for log plot
err_fwd  = np.clip(err_fwd, 1e-18, None)
err_cen  = np.clip(err_cen, 1e-18, None)
err_cplx = np.clip(err_cplx, 1e-18, None)

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(hs, err_cplx, color='green',  lw=1.5, label='complex')
ax.loglog(hs, err_fwd,  color='steelblue', lw=1.5, label='forward')
ax.loglog(hs, err_cen,  color='salmon', lw=1.5, label='central')
ax.set_xlabel('step size $h$', fontsize=11)
ax.set_ylabel('absolute relative error', fontsize=11)
ax.set_title(r'Derivative error for $\sin(x)$ at $x=0.5$', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(1e-18, 1e0)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
savefig('fig_error_comparison.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Gradient direction on a 2D contour (book Fig 2.3 style)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
X, Y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
Z = X**2 + 2*Y**2
cs = ax.contourf(X, Y, Z, levels=12, cmap='viridis', alpha=0.7)
ax.contour(X, Y, Z, levels=12, colors='white', linewidths=0.5, alpha=0.5)
plt.colorbar(cs, ax=ax, shrink=0.8)

# gradient at a point
x0, y0 = 1.0, 0.8
gx, gy = 2*x0, 4*y0
scale = 0.35
ax.annotate('', xy=(x0 + scale*gx, y0 + scale*gy), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
ax.plot(x0, y0, 'ro', ms=7, zorder=5)
ax.text(x0 + scale*gx + 0.05, y0 + scale*gy, r'$\nabla f(\mathbf{x})$',
        color='red', fontsize=12)
ax.set_xlabel(r'$x_1$', fontsize=12); ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title(r'Gradient points in direction of steepest ascent', fontsize=10)
plt.tight_layout()
savefig('fig_gradient_contour.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Regression gradient illustration (book Fig 2.7)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
fig, ax = plt.subplots(figsize=(5, 4))
# simple 2D bowl: f(x,y) = x^2 + y^2,  evaluate at centre x0=(0.5, 0.5)
x0 = np.array([0.5, 0.5])
m  = 20
delta = 0.3
X_samp = np.random.randn(m, 2)
X_samp /= np.linalg.norm(X_samp, axis=1, keepdims=True)
X_samp *= delta
f0 = np.sum(x0**2)
df = np.array([np.sum((x0 + dx)**2) - f0 for dx in X_samp])

# draw contours
Xg, Yg = np.meshgrid(np.linspace(-0.3, 1.3, 100), np.linspace(-0.3, 1.3, 100))
Zg = Xg**2 + Yg**2
ax.contour(Xg, Yg, Zg, levels=10, alpha=0.5)

# perturbation vectors coloured by delta f
sc = ax.scatter(x0[0]+X_samp[:,0], x0[1]+X_samp[:,1], c=df,
                cmap='RdYlGn_r', s=50, zorder=5)
plt.colorbar(sc, ax=ax, label=r'$\Delta f$', shrink=0.8)

# estimated gradient
g_est, *_ = np.linalg.lstsq(X_samp, df, rcond=None)
ax.annotate('', xy=(x0[0]+0.3*g_est[0], x0[1]+0.3*g_est[1]),
            xytext=(x0[0], x0[1]),
            arrowprops=dict(arrowstyle='->', color='navy', lw=2))
ax.plot(*x0, 'ks', ms=8, zorder=6)
ax.text(x0[0]+0.3*g_est[0]+0.03, x0[1]+0.3*g_est[1],
        r'$\hat{\mathbf{g}}$', color='navy', fontsize=13)
ax.set_xlabel(r'$x_1$', fontsize=12); ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('Regression gradient (m=20 perturbations)', fontsize=10)
plt.tight_layout()
savefig('fig_regression_gradient.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: SPSA illustration
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(7)
fig, ax = plt.subplots(figsize=(5, 4))
# same bowl, show +z and -z perturbation pairs
x0 = np.array([0.5, 0.5])
delta = 0.25
for _ in range(8):
    z = np.random.choice([-1, 1], size=2).astype(float)
    z /= np.linalg.norm(z)
    xp = x0 + delta*z
    xm = x0 - delta*z
    ax.plot([xm[0], xp[0]], [xm[1], xp[1]], 'b-', alpha=0.5, lw=1)
    ax.plot(*xp, 'b^', ms=6, alpha=0.7)
    ax.plot(*xm, 'bv', ms=6, alpha=0.7)

Xg, Yg = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
Zg = Xg**2 + Yg**2
ax.contour(Xg, Yg, Zg, levels=8, alpha=0.4)
ax.plot(*x0, 'rs', ms=9, zorder=5)
ax.set_xlabel(r'$x_1$', fontsize=12); ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_title('SPSA: symmetric perturbation pairs', fontsize=10)
plt.tight_layout()
savefig('fig_spsa.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Hessian curvature – eigenvalue illustration
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, (a, b), title in zip(axes,
        [(1, 1), (1, 5)],
        [r'$\mathbf{H}$ isotropic ($\lambda_1=\lambda_2$)',
         r'$\mathbf{H}$ anisotropic ($\lambda_1 \ll \lambda_2$)']):
    X, Y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    Z = a*X**2 + b*Y**2
    cs = ax.contourf(X, Y, Z, levels=12, cmap='plasma', alpha=0.8)
    ax.contour(X, Y, Z, levels=12, colors='white', linewidths=0.4, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$x_1$', fontsize=11); ax.set_ylabel(r'$x_2$', fontsize=11)
plt.tight_layout()
savefig('fig_hessian_curvature.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Dual-number forward accumulation trace for f(a,b)=ln(ab+max(a,2))
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.axis('off')
ax.set_facecolor('white')

# node positions  (label, x, y, shape)
nodes = {
    'b':   (1.0, 4.2),
    'a':   (1.0, 2.0),
    '2':   (1.0, 0.6),
    'x':   (3.2, 3.6),
    'max': (3.2, 1.3),
    '+':   (5.5, 2.5),
    'ln':  (7.5, 2.5),
}
# draw circles / rectangles
circle_nodes = {'x', 'max', '+', 'ln'}
for name, (nx, ny) in nodes.items():
    if name in circle_nodes:
        circ = plt.Circle((nx, ny), 0.4, color='lightblue', ec='navy', lw=1.5)
        ax.add_patch(circ)
        ax.text(nx, ny, name, ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(nx-0.1, ny, name, ha='center', va='center', fontsize=12, fontstyle='italic')

# edges
edges = [
    ('b',   'x'),
    ('a',   'x'),
    ('a',   'max'),
    ('2',   'max'),
    ('x',   '+'),
    ('max', '+'),
    ('+',   'ln'),
]
def arrow(ax, n1, n2, nodes, offset=0.4):
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    dx, dy = x2-x1, y2-y1
    length = np.sqrt(dx**2+dy**2)
    ux, uy = dx/length, dy/length
    ax.annotate('', xy=(x2 - offset*ux, y2 - offset*uy),
                xytext=(x1 + (0.15 if n1 not in circle_nodes else offset)*ux,
                        y1 + (0.15 if n1 not in circle_nodes else offset)*uy),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.2))

for e in edges:
    arrow(ax, e[0], e[1], nodes)

# intermediate labels
labels = {'x': r'$c_1$', 'max': r'$c_2$', '+': r'$c_3$', 'ln': r'$c_4$'}
offsets = {'x': (0.6, 0.3), 'max': (0.6, -0.3), '+': (0.6, 0.3), 'ln': (0.7, 0.0)}
for n, lbl in labels.items():
    nx, ny = nodes[n]
    ox, oy = offsets[n]
    ax.text(nx+ox, ny+oy, lbl, fontsize=10, color='darkgreen')

ax.set_title(r'Computational graph for $f(a,b)=\ln(ab+\max(a,2))$', fontsize=11)
plt.tight_layout()
savefig('fig_comp_graph.pdf')

print('All figures generated successfully.')
