"""
gen_figures.py
Generates all figures for Chapter 1 (Introduction) slides on
"The Krasnosel'skii-Mann Iterative Method" (Dong, Cho, He, Pardalos & Rassias,
SpringerBriefs in Optimization, 2022).

RUNNING EXAMPLE used throughout the chapter deck:
  H = R^2 with the usual dot product and Euclidean norm.
  C = closed unit disk {x in R^2 : ||x|| <= 1}.
  T = P_C, the metric projection onto C:
        T(x) = x            if ||x|| <= 1
        T(x) = x / ||x||    if ||x|| >  1
  T is nonexpansive (indeed firmly nonexpansive) but NOT a contraction.
  x0 = (3,4) = 5*(0.6,0.8); since (0.6,0.8) is already on the unit circle,
  the whole orbit stays on the ray {r*(0.6,0.8) : r >= 0}, and writing
  x_n = r_n*(0.6,0.8) collapses every iteration to a 1-D recursion in r_n.

A SECOND toy example, T_rot(x,y) = (-y,x) (rotation by 90 degrees), is used
to illustrate a nonexpansive map for which plain Picard iteration truly
fails to converge (it has the unique fixed point 0, but Picard iterates
cycle forever on a circle unless started at 0). This is the "rotation-like
nonexpansive example without a fixed point [reached by Picard]" mentioned
in the chapter.

Run with:  python3 gen_figures.py
Saves all figures as vector PDFs into this directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(2022)

# ---------------------------------------------------------------------------
# Running-example constants
# ---------------------------------------------------------------------------
X0 = np.array([3.0, 4.0])          # starting point of the running example
U_RAY = np.array([0.6, 0.8])       # unit vector defining the ray; ||U_RAY|| = 1
R0 = 5.0                            # X0 = R0 * U_RAY


def project_disk(x):
    """T = P_C: metric projection onto the closed unit disk."""
    norm = np.linalg.norm(x)
    if norm <= 1.0:
        return x.copy()
    return x / norm


def krasnoselskii_orbit_on_ray(r0, lam, n_steps):
    """
    Krasnoselskii iteration r_{n+1} = (1-lam) r_n + lam*1 restricted to the
    ray, valid exactly as long as r_n > 1 (so that T maps the point to the
    boundary point U_RAY exactly). Returns the sequence r_0, ..., r_{n_steps}.
    """
    rs = [r0]
    r = r0
    for _ in range(n_steps):
        r = (1 - lam) * r + lam * 1.0
        rs.append(r)
    return np.array(rs)


# Print the worked-example arithmetic verbatim, to check against the slides.
print("=== Running example: Krasnoselskii iteration (lambda=1/2), x0=(3,4) ===")
r_seq = krasnoselskii_orbit_on_ray(R0, 0.5, 6)
for n, r in enumerate(r_seq):
    print(f"  r_{n} = {r:.4f}")

# ---------------------------------------------------------------------------
# Figure 1: Picard may fail (T = -Id) vs. Krasnoselskii converges (T = P_C)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

# --- Left panel: Picard iteration on T(x) = -x, x0 = (3,4) ---
# x_{n+1} = -x_n: oscillates between (3,4) and (-3,-4) forever, never
# converging (its only fixed point is the origin, which is never reached
# unless x0 = 0).
ax = axes[0]
n_iters = 6
pts = [X0.copy()]
x = X0.copy()
for _ in range(n_iters):
    x = -x
    pts.append(x.copy())
pts = np.array(pts)

ax.axhline(0, color='gray', linewidth=0.6)
ax.axvline(0, color='gray', linewidth=0.6)
ax.plot(pts[:, 0], pts[:, 1], color='#C44E52', linewidth=1.2, zorder=2)
ax.scatter(pts[0::2, 0], pts[0::2, 1], color='#C44E52', marker='o', s=55,
           zorder=3, label='$x_0, x_2, x_4,\\dots = (3,4)$')
ax.scatter(pts[1::2, 0], pts[1::2, 1], color='#4C72B0', marker='s', s=55,
           zorder=3, label='$x_1, x_3, x_5,\\dots = (-3,-4)$')
ax.scatter([0], [0], color='black', marker='*', s=140, zorder=4,
           label='Fix($T$) $=\\{(0,0)\\}$ (never reached)')
for i in range(len(pts) - 1):
    ax.annotate('', xy=pts[i + 1], xytext=pts[i],
                arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8,
                                shrinkA=8, shrinkB=8))
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.set_aspect('equal')
ax.set_title('Picard iteration, $T(x)=-x$\n(nonexpansive, NOT a contraction):\nnever converges')
ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlabel('$x^{(1)}$')
ax.set_ylabel('$x^{(2)}$')

# --- Right panel: Krasnoselskii iteration on T = P_C, x0 = (3,4) ---
ax = axes[1]
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color='0.6', linewidth=1.0)
ax.fill(np.cos(theta), np.sin(theta), color='0.92', zorder=0)

lam = 0.5
n_iters = 6
km_pts = [X0.copy()]
x = X0.copy()
for _ in range(n_iters):
    x = (1 - lam) * x + lam * project_disk(x)
    km_pts.append(x.copy())
km_pts = np.array(km_pts)

ax.plot(km_pts[:, 0], km_pts[:, 1], color='#55A868', linewidth=1.2, zorder=2)
ax.scatter(km_pts[:, 0], km_pts[:, 1], color='#55A868', s=45, zorder=3)
for n, p in enumerate(km_pts):
    ax.annotate(f'$x_{n}$', p, textcoords='offset points', xytext=(6, 4),
                fontsize=8)
ax.scatter([0.6], [0.8], color='black', marker='*', s=140, zorder=4,
           label='fixed point $(0.6,0.8)$')
ax.axhline(0, color='gray', linewidth=0.4)
ax.axvline(0, color='gray', linewidth=0.4)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 5.5)
ax.set_aspect('equal')
ax.set_title('Krasnoselskii iteration ($\\lambda=1/2$), $T=P_C$\n'
              '$x_{n+1}=\\frac{1}{2} x_n+\\frac{1}{2} T(x_n)$: converges to Fix($T$)')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlabel('$x^{(1)}$')
ax.set_ylabel('$x^{(2)}$')

fig.suptitle('Running example: Picard can fail for nonexpansive maps, '
             'but averaging (Krasnoselskii) fixes it', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('fig_running_example.pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Picard vs. Krasnoselskii vs. Mann on the SAME simple problem
#           T_rot(x,y) = (-y,x): rotation by 90 degrees, Fix(T) = {(0,0)}.
#           Represented as multiplication by i in the complex plane.
# ---------------------------------------------------------------------------
z0 = complex(3.0, 4.0)
n_iters = 24


def picard_orbit(z0, n):
    zs = [z0]
    z = z0
    for _ in range(n):
        z = 1j * z          # T_rot
        zs.append(z)
    return np.array(zs)


def krasnoselskii_orbit(z0, lam, n):
    zs = [z0]
    z = z0
    for _ in range(n):
        z = (1 - lam) * z + lam * (1j * z)
        zs.append(z)
    return np.array(zs)


def mann_orbit(z0, lam_fn, n):
    zs = [z0]
    z = z0
    for k in range(n):
        lam = lam_fn(k)
        z = (1 - lam) * z + lam * (1j * z)
        zs.append(z)
    return np.array(zs)


picard_z = picard_orbit(z0, 8)                       # cycles with period 4
km_z = krasnoselskii_orbit(z0, 0.5, n_iters)          # constant lambda = 1/2
mann_z = mann_orbit(z0, lambda k: 1.0 / (k + 2), n_iters)   # lambda_k -> 0

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

# --- Left panel: trajectories in the plane ---
ax = axes[0]
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(5 * np.cos(theta), 5 * np.sin(theta), color='0.75', linewidth=0.8,
        linestyle=':', label='circle of radius 5 (Picard orbit lies here)')

ax.plot(picard_z.real, picard_z.imag, 'o-', color='#C44E52', linewidth=1.1,
        markersize=6, label='Picard ($\\lambda\\equiv 1$): cycles forever')
ax.plot(km_z.real, km_z.imag, 's-', color='#55A868', linewidth=1.0,
        markersize=3.5, label='Krasnoselskii ($\\lambda\\equiv 1/2$): spirals in')
ax.plot(mann_z.real, mann_z.imag, '^-', color='#4C72B0', linewidth=1.0,
        markersize=3.5, label='Mann ($\\lambda_n=1/(n{+}2)$): spirals in, slower')
ax.scatter([0], [0], color='black', marker='*', s=150, zorder=5,
           label='Fix($T$) $=\\{(0,0)\\}$')
ax.set_aspect('equal')
ax.set_xlabel('$x^{(1)}$')
ax.set_ylabel('$x^{(2)}$')
ax.set_title('Trajectories: $T(x,y)=(-y,x)$ (a $90^\\circ$ rotation)')
ax.legend(fontsize=7, loc='upper right')

# --- Right panel: distance to the fixed point vs iteration ---
ax = axes[1]
ns = np.arange(len(picard_z))
ax.plot(ns, np.abs(picard_z), 'o-', color='#C44E52', markersize=4,
        label='Picard: $\\|x_n\\|\\equiv 5$ (no decrease)')
ns2 = np.arange(len(km_z))
ax.plot(ns2, np.abs(km_z), 's-', color='#55A868', markersize=3,
        label='Krasnoselskii: geometric, rate $1/\\sqrt{2}\\approx0.707$')
ns3 = np.arange(len(mann_z))
ax.plot(ns3, np.abs(mann_z), '^-', color='#4C72B0', markersize=3,
        label='Mann: slower (diminishing step)')
ax.set_yscale('log')
ax.set_xlabel('iteration $n$')
ax.set_ylabel('$\\|x_n - x^*\\|$ (log scale), $x^*=(0,0)$')
ax.set_title('Distance to the fixed point')
ax.legend(fontsize=7.5, loc='upper right')

fig.suptitle('Same nonexpansive map, three iteration schemes: '
             'damping is what makes convergence possible', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('fig_iteration_comparison.pdf')
plt.close(fig)

print("\n=== Rotation example: Picard cycle (first 5 points) ===")
for n, z in enumerate(picard_z[:5]):
    print(f"  x_{n} = ({z.real:.3f}, {z.imag:.3f}),  ||x_{n}|| = {abs(z):.3f}")

# ---------------------------------------------------------------------------
# Figure 3: convergence rate as a function of the constant lambda
#           (rotation example): rate(lambda) = |1 - lambda + lambda*i|
#           minimized at lambda = 1/2 -- Krasnoselskii's original choice!
# ---------------------------------------------------------------------------
lams = np.linspace(0.001, 0.999, 400)
rate = np.sqrt((1 - lams) ** 2 + lams ** 2)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(lams, rate, color='#4C72B0', linewidth=1.8)
ax.axvline(0.5, color='#55A868', linestyle='--', linewidth=1.2,
           label='$\\lambda=1/2$ (Krasnoselskii): rate $=1/\\sqrt{2}\\approx0.707$ (minimal)')
ax.scatter([0.5], [1 / np.sqrt(2)], color='#55A868', zorder=5, s=60)
ax.axhline(1.0, color='#C44E52', linestyle=':', linewidth=1.2,
           label='rate $=1$ at $\\lambda\\to0,1$ (no contraction; $\\lambda=1$ is Picard)')
ax.set_xlabel('constant relaxation parameter $\\lambda\\in(0,1)$')
ax.set_ylabel('contraction rate of $T_\\lambda=(1-\\lambda)\\mathrm{Id}+\\lambda T$')
ax.set_title('Rotation example: averaging turns a non-contractive\n'
              'nonexpansive map into a genuine contraction')
ax.legend(fontsize=8, loc='upper center')
ax.set_ylim(0.65, 1.05)
fig.tight_layout()
fig.savefig('fig_lambda_rate.pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: projection P_C as a fixed-point / convex-feasibility problem
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 6.0))
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color='0.5', linewidth=1.2)
ax.fill(np.cos(theta), np.sin(theta), color='0.93', zorder=0,
        label='$C$ = closed unit disk = Fix($P_C$)')

exterior_pts = np.array([[3.0, 4.0], [-2.0, 1.5], [1.8, -2.4], [-1.0, -2.8]])
for i, p in enumerate(exterior_pts):
    proj = project_disk(p)
    ax.plot([p[0], proj[0]], [p[1], proj[1]], color='#4C72B0', linewidth=1.0,
            linestyle='--', zorder=1)
    ax.scatter(*p, color='#C44E52', s=45, zorder=3)
    ax.scatter(*proj, color='#55A868', s=45, zorder=3)

# Highlight the running example and its supporting hyperplane at (0.6,0.8),
# i.e. the variational-inequality characterization of the projection:
# <x0 - P_C(x0), y - P_C(x0)> <= 0 for all y in C.
p0, proj0 = X0, project_disk(X0)
normal = proj0  # outward normal at the boundary point equals the point itself
tangent_dir = np.array([-normal[1], normal[0]])
t = np.linspace(-1.4, 1.4, 2)
line = proj0[None, :] + t[:, None] * tangent_dir[None, :]
ax.plot(line[:, 0], line[:, 1], color='black', linewidth=1.0,
        label='supporting line at $P_C(x_0)$: the VIP halfspace boundary')

ax.scatter(*p0, color='#C44E52', s=90, zorder=4, marker='D',
           label='$x_0=(3,4)$ (running example)')
ax.scatter(*proj0, color='#55A868', s=90, zorder=4, marker='D',
           label='$P_C(x_0)=(0.6,0.8)=T(x_0)$')
ax.annotate('$x_0$', p0, textcoords='offset points', xytext=(8, 4), fontsize=10)
ax.annotate('$P_C(x_0)$', proj0, textcoords='offset points', xytext=(8, -14),
            fontsize=10)

ax.axhline(0, color='gray', linewidth=0.4)
ax.axvline(0, color='gray', linewidth=0.4)
ax.set_xlim(-3.5, 5.5)
ax.set_ylim(-3.5, 5.5)
ax.set_aspect('equal')
ax.set_title('$P_C(x_0)$ solves three equivalent problems:\n'
              'fixed point, convex feasibility, and variational inequality')
ax.legend(fontsize=7.2, loc='lower right')
fig.tight_layout()
fig.savefig('fig_projection_feasibility.pdf')
plt.close(fig)

print("\nAll figures written to the current directory:")
print("  fig_running_example.pdf")
print("  fig_iteration_comparison.pdf")
print("  fig_lambda_rate.pdf")
print("  fig_projection_feasibility.pdf")
