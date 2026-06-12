"""Generate all figures for Chapter 1 slides."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol, Halton, LatinHypercube
import os

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Fig 1: Simple Symmetric Random Walk (single trajectory)
# ---------------------------------------------------------------
rng = np.random.default_rng(31415)
N = 1000
steps = 1 - 2 * rng.integers(0, 2, N)
S = np.concatenate([[0], steps.cumsum()])

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(S, lw=0.9, color='steelblue')
ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Step $n$', fontsize=12)
ax.set_ylabel('$S_n$', fontsize=12)
ax.set_title('Simple Symmetric Random Walk ($N=1000$)', fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_ssrw_single.pdf', dpi=150)
plt.close()
print("fig_ssrw_single.pdf done")

# ---------------------------------------------------------------
# Fig 2: LIL envelope – multiple trajectories
# ---------------------------------------------------------------
rng2 = np.random.default_rng(0)
R_lil, N_lil = 80, 2**14
n_arr = np.arange(1, N_lil + 1)
envelope = np.sqrt(2 * n_arr * np.log(np.log(n_arr + 2)))

fig, ax = plt.subplots(figsize=(9, 4))
for _ in range(R_lil):
    s = np.concatenate([[0], (1 - 2 * rng2.integers(0, 2, N_lil)).cumsum()])
    ax.plot(s, lw=0.25, alpha=0.35, color='gray')
ax.plot(n_arr,  envelope, 'b-', lw=2.0, label=r'$\pm\sqrt{2n\log\log n}$')
ax.plot(n_arr, -envelope, 'b-', lw=2.0)
ax.plot(n_arr,  np.sqrt(n_arr), 'r--', lw=1.2, alpha=0.7, label=r'$\pm\sqrt{n}$')
ax.plot(n_arr, -np.sqrt(n_arr), 'r--', lw=1.2, alpha=0.7)
ax.set_xlabel('Step $n$', fontsize=12)
ax.set_ylabel('$S_n$', fontsize=12)
ax.set_title(f'Law of the Iterated Logarithm: {R_lil} trajectories, $N=2^{{14}}$', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_lil_envelope.pdf', dpi=150)
plt.close()
print("fig_lil_envelope.pdf done")

# ---------------------------------------------------------------
# Fig 3: Arcsine law histogram
# ---------------------------------------------------------------
rng3 = np.random.default_rng(42)
n_arc, R_arc = 500, 10_000
steps3 = 1 - 2 * rng3.integers(0, 2, (R_arc, 2 * n_arc))
S3 = np.hstack([np.zeros((R_arc, 1)), steps3.cumsum(axis=1)])
above = (S3[:, :-1] >= 0) | (S3[:, 1:] >= 0)
fracs = above.sum(axis=1) / (2 * n_arc)

x_arc = np.linspace(0.01, 0.99, 400)
pdf_arc = 1.0 / (np.pi * np.sqrt(x_arc * (1 - x_arc)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(fracs, bins=60, density=True, alpha=0.65, color='steelblue',
        edgecolor='white', label='Simulation')
ax.plot(x_arc, pdf_arc, 'r-', lw=2.2, label=r'Arcsine density $\frac{1}{\pi\sqrt{x(1-x)}}$')
ax.set_xlabel(r'$L^+_{2n}/(2n)$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Arcsine Law: simulation vs theory ($n={n_arc}$, $R={R_arc:,}$)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_arcsine_law.pdf', dpi=150)
plt.close()
print("fig_arcsine_law.pdf done")

# ---------------------------------------------------------------
# Fig 4: TSP – random cities + best random tour
# ---------------------------------------------------------------
rng4 = np.random.default_rng(7)
n_cities = 15
cities = rng4.random((n_cities, 2))

def tour_len(c, p):
    return np.linalg.norm(c[p] - c[np.roll(p, -1)], axis=1).sum()

best_len, best_tour = np.inf, None
for _ in range(20_000):
    p = rng4.permutation(n_cities)
    L = tour_len(cities, p)
    if L < best_len:
        best_len, best_tour = L, p.copy()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (tour, title) in zip(axes, [
        (np.arange(n_cities), 'Arbitrary order (no tour)'),
        (best_tour, f'Best of 20,000 random tours\n(length={best_len:.3f})')]):
    loop = np.append(tour, tour[0])
    ax.plot(cities[loop, 0], cities[loop, 1], 'b-o',
            markersize=7, lw=1.2, markerfacecolor='orange')
    for i, (x, y) in enumerate(cities):
        ax.text(x + 0.01, y + 0.01, str(i), fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
fig.suptitle('Travelling Salesman Problem – random search heuristic', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_tsp.pdf', dpi=150)
plt.close()
print("fig_tsp.pdf done")

# ---------------------------------------------------------------
# Fig 5: Pseudorandom vs Quasi-random points in [0,1]^2
# ---------------------------------------------------------------
n_pts = 512
rng5 = np.random.default_rng(0)
U_mc    = rng5.random((n_pts, 2))
U_sobol = Sobol(2, scramble=True, seed=0).random(n_pts)
U_lhs   = LatinHypercube(2, seed=0).random(n_pts)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, U, title in zip(axes,
        [U_mc, U_sobol, U_lhs],
        ['MC (PCG64)', "Sobol'", 'Latin Hypercube']):
    ax.scatter(U[:, 0], U[:, 1], s=4, alpha=0.6, color='steelblue')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$')
fig.suptitle(f'$n={n_pts}$ points in $[0,1]^2$: random vs low-discrepancy', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_mc_vs_qmc_scatter.pdf', dpi=150)
plt.close()
print("fig_mc_vs_qmc_scatter.pdf done")

# ---------------------------------------------------------------
# Fig 6: MC vs QMC convergence for pi estimation
# ---------------------------------------------------------------
Rs = [2**k for k in range(4, 15)]
mc_err, sob_err, hal_err, lhs_err = [], [], [], []
for R in Rs:
    rng6 = np.random.default_rng(0)
    def pi_from(U): return abs(4.0 * ((U**2).sum(1) <= 1).mean() - np.pi)
    mc_err.append( pi_from(rng6.random((R, 2))))
    sob_err.append(pi_from(Sobol(2, scramble=True, seed=0).random(R)))
    hal_err.append(pi_from(Halton(2, scramble=True, seed=0).random(R)))
    lhs_err.append(pi_from(LatinHypercube(2, seed=0).random(R)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(Rs, mc_err,  'b-o',  ms=5, label='MC (PCG64)')
ax.loglog(Rs, sob_err, 'r-s',  ms=5, label="Sobol'")
ax.loglog(Rs, hal_err, 'g-^',  ms=5, label='Halton')
ax.loglog(Rs, lhs_err, 'm-D',  ms=5, label='Latin Hypercube')
ref = [mc_err[0] * (Rs[0] / r)**0.5 for r in Rs]
ax.loglog(Rs, ref, 'k--', lw=1.5, label=r'$O(R^{-1/2})$')
ax.set_xlabel('$R$ (number of samples)', fontsize=12)
ax.set_ylabel(r'$|\hat{\pi}_R - \pi|$', fontsize=12)
ax.set_title(r'MC vs QMC convergence to $\pi$', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_convergence_pi.pdf', dpi=150)
plt.close()
print("fig_convergence_pi.pdf done")

# ---------------------------------------------------------------
# Fig 7: Structured appearance of QMC consecutive pairs
# ---------------------------------------------------------------
n_str = 1024
rng7  = np.random.default_rng(0)
U_mc2    = rng7.random((n_str, 2))
U_sobol2 = Sobol(2, scramble=True, seed=0).random(n_str)
U_hal2   = Halton(2, scramble=True, seed=0).random(n_str)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, U, title in zip(axes,
        [U_mc2, U_sobol2, U_hal2],
        ['MC (PCG64)', "Sobol'", 'Halton']):
    # scatter of (U[i,0], U[i,1]) for consecutive rows -- shows structure
    ax.scatter(U[:, 0], U[:, 1], s=3, alpha=0.5, color='steelblue')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$U_1^{(i)}$'); ax.set_ylabel('$U_2^{(i)}$')
fig.suptitle('Consecutive 2-D samples: structured vs random appearance', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_qmc_structure.pdf', dpi=150)
plt.close()
print("fig_qmc_structure.pdf done")

# ---------------------------------------------------------------
# Fig 8: 2D game convergence MC vs QMC
# ---------------------------------------------------------------
N1, N2 = 4, 4
TRUE_RHO = 3.0 / 8.0

def play_game(U_stream):
    i1, i2 = 2, 3
    for U1, U2 in U_stream:
        p = q = 0.25
        thr1, thr2 = 2*p, 4*p
        if U1 < thr1:
            i1 += 1 if U2 < 0.5 else -1
        elif U1 < thr2:
            i2 += 1 if U2 < 0.5 else -1
        if i1 <= 0 or i2 <= 0:   return 0
        if i1 >= N1 and i2 >= N2: return 1
    return 0

Rgame_list = [50, 100, 200, 500, 1000, 2000, 5000]
mc_game, sob_game, hal_game = [], [], []
rng8 = np.random.default_rng(123)
for R in Rgame_list:
    wins_mc  = sum(play_game(rng8.random((300, 2))) for _ in range(R))
    mc_game.append(abs(wins_mc / R - TRUE_RHO))
    wins_sob = sum(play_game(iter(Sobol(2, scramble=True, seed=r).random(300)))
                   for r in range(R))
    sob_game.append(abs(wins_sob / R - TRUE_RHO))
    wins_hal = sum(play_game(iter(Halton(2, scramble=True, seed=r).random(300)))
                   for r in range(R))
    hal_game.append(abs(wins_hal / R - TRUE_RHO))

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(Rgame_list, mc_game,  'b-o', ms=5, label='MC (PCG64)')
ax.loglog(Rgame_list, sob_game, 'r-s', ms=5, label="Sobol'")
ax.loglog(Rgame_list, hal_game, 'g-^', ms=5, label='Halton')
ref2 = [mc_game[0] * (Rgame_list[0] / r)**0.5 for r in Rgame_list]
ax.loglog(Rgame_list, ref2, 'k--', lw=1.5, label=r'$O(R^{-1/2})$')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('$R$ (number of games)', fontsize=12)
ax.set_ylabel(r'$|\hat{\rho} - 3/8|$', fontsize=12)
ax.set_title(r'2-D game: MC vs QMC convergence to $\rho(2,3)=3/8$', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_game_convergence.pdf', dpi=150)
plt.close()
print("fig_game_convergence.pdf done")

print("\nAll figures saved to:", OUT)
