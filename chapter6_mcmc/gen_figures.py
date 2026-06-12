"""
gen_figures.py  –  Chapter 6: Markov Chain Monte Carlo Methods
Generates all figures as PDF files in the figures/ subdirectory.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name + '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  saved {name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1  –  Simple 2-state Markov chain diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_two_state_chain():
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    # states
    circle0 = plt.Circle((1, 0.5), 0.35, color='steelblue', fill=True, alpha=0.3, linewidth=2, ec='steelblue')
    circle1 = plt.Circle((3, 0.5), 0.35, color='darkorange', fill=True, alpha=0.3, linewidth=2, ec='darkorange')
    ax.add_patch(circle0)
    ax.add_patch(circle1)
    ax.text(1, 0.5, r'$s_1$', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(3, 0.5, r'$s_2$', ha='center', va='center', fontsize=14, fontweight='bold')

    # arrows
    ax.annotate('', xy=(2.65, 0.65), xytext=(1.35, 0.65),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
    ax.annotate('', xy=(1.35, 0.35), xytext=(2.65, 0.35),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))
    ax.text(2.0, 0.85, r'$\alpha$', ha='center', fontsize=13, color='steelblue')
    ax.text(2.0, 0.12, r'$\beta$', ha='center', fontsize=13, color='darkorange')

    # self-loops
    ax.annotate('', xy=(0.75, 0.82), xytext=(0.65, 0.82),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                connectionstyle='arc3,rad=-2.5'))
    ax.annotate('', xy=(3.35, 0.82), xytext=(3.25, 0.82),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                connectionstyle='arc3,rad=-2.5'))
    ax.text(0.45, 1.1, r'$1-\alpha$', ha='center', fontsize=10, color='gray')
    ax.text(3.6, 1.1, r'$1-\beta$', ha='center', fontsize=10, color='gray')

    ax.set_title('Two-state Markov chain', fontsize=12)
    savefig('fig_two_state_chain')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  –  Metropolis acceptance ratio illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_metropolis_acceptance():
    np.random.seed(42)
    # Show a simple pi distribution (Gibbs/Boltzmann on 10 states)
    states = np.arange(10)
    energy = np.array([3, 1, 2, 0.5, 1.5, 0.8, 2.5, 1.2, 3.5, 1.8])
    pi = np.exp(-energy)
    pi /= pi.sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].bar(states, pi, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('State $s$', fontsize=12)
    axes[0].set_ylabel(r'$\pi_s$', fontsize=12)
    axes[0].set_title('Target distribution $\\pi$', fontsize=12)

    # Metropolis acceptance
    s_cur = 3
    s_prop = 7
    alpha = min(1, pi[s_prop] / pi[s_cur])
    axes[1].bar(states, pi, color='lightgray', edgecolor='white')
    axes[1].bar([s_cur, s_prop], [pi[s_cur], pi[s_prop]],
                color=['steelblue', 'darkorange'], edgecolor='white', alpha=0.9)
    axes[1].axhline(pi[s_cur], color='steelblue', linestyle='--', alpha=0.7, label=f'current $\\pi_s={pi[s_cur]:.3f}$')
    axes[1].axhline(pi[s_prop], color='darkorange', linestyle='--', alpha=0.7,
                    label=f'proposal $\\pi_{{s\'}}={pi[s_prop]:.3f}$')
    axes[1].set_xlabel('State $s$', fontsize=12)
    axes[1].set_title(
        r'Acceptance ratio $\alpha = \min(1, \pi_{s}/\pi_s) = $' + f'{alpha:.3f}',
        fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    savefig('fig_metropolis_acceptance')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  –  Metropolis-Hastings trace plot
# ─────────────────────────────────────────────────────────────────────────────
def fig_mh_trace():
    np.random.seed(0)
    # Target: mixture of two Gaussians
    def log_target(x):
        return np.log(0.4 * np.exp(-0.5*(x-(-2))**2) + 0.6 * np.exp(-0.5*(x-3)**2 / 4))

    N = 2000
    samples = np.zeros(N)
    samples[0] = 0.0
    accept = 0
    for i in range(1, N):
        prop = samples[i-1] + np.random.normal(0, 1.5)
        log_alpha = log_target(prop) - log_target(samples[i-1])
        if np.log(np.random.rand()) < log_alpha:
            samples[i] = prop
            accept += 1
        else:
            samples[i] = samples[i-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(samples, alpha=0.7, color='steelblue', lw=0.8)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('State', fontsize=12)
    axes[0].set_title(f'MH trace (acceptance rate {accept/N:.1%})', fontsize=12)

    x = np.linspace(-7, 9, 500)
    target = 0.4 * np.exp(-0.5*(x-(-2))**2) + 0.6 * np.exp(-0.5*(x-3)**2 / 4)
    target /= target.sum() * (x[1]-x[0])
    axes[1].hist(samples[200:], bins=60, density=True, color='steelblue', alpha=0.6, label='MCMC samples')
    axes[1].plot(x, target, 'r-', lw=2, label='Target $\\pi$')
    axes[1].set_xlabel('State', fontsize=12)
    axes[1].set_title('Sample histogram vs target', fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    savefig('fig_mh_trace')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  –  Graph coloring example (small graph, 3 colors)
# ─────────────────────────────────────────────────────────────────────────────
def fig_graph_coloring():
    G = nx.cycle_graph(6)
    G.add_edge(0, 3)
    pos = nx.spring_layout(G, seed=7)
    colors = ['red', 'blue', 'green', 'red', 'blue', 'green']

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    nx.draw(G, pos, ax=ax, node_color=colors, with_labels=True,
            node_size=600, font_color='white', font_weight='bold',
            edge_color='gray', width=2)
    ax.set_title('Proper 3-coloring of a graph', fontsize=12)
    savefig('fig_graph_coloring')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  –  Hard-core model on a grid
# ─────────────────────────────────────────────────────────────────────────────
def fig_hardcore_grid():
    n = 5
    # checkerboard independent set
    config = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                config[i, j] = 1

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for i in range(n):
        for j in range(n):
            color = 'steelblue' if config[i, j] == 1 else 'lightgray'
            circle = plt.Circle((j, n-1-i), 0.35, color=color, ec='black', lw=1.5)
            ax.add_patch(circle)
            if config[i, j] == 1:
                ax.text(j, n-1-i, '1', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=10)

    # draw edges
    for i in range(n):
        for j in range(n):
            for di, dj in [(0,1),(1,0)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < n and 0 <= nj < n:
                    ax.plot([j, nj], [n-1-i, n-1-ni], 'k-', lw=1, zorder=0)

    ax.set_xlim(-0.6, n-0.4)
    ax.set_ylim(-0.6, n-0.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Hard-core model: independent set on grid', fontsize=11)
    savefig('fig_hardcore_grid')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  –  Ising model sample (small grid)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ising_sample():
    np.random.seed(5)
    n = 20
    beta = 0.45

    # Gibbs sampling for Ising model
    config = np.random.choice([-1, 1], size=(n, n))
    for _ in range(5000):
        i, j = np.random.randint(0, n, 2)
        neighbors_sum = 0
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = (i+di)%n, (j+dj)%n
            neighbors_sum += config[ni, nj]
        h = beta * neighbors_sum
        prob_plus = np.exp(h) / (np.exp(h) + np.exp(-h))
        config[i, j] = 1 if np.random.rand() < prob_plus else -1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(config, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(f'Ising model sample ($\\beta={beta}$, {n}x{n} grid)', fontsize=11)
    ax.axis('off')
    savefig('fig_ising_sample')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7  –  CFTP schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_cftp_schematic():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-12, 1)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    # Time axis
    ax.annotate('', xy=(0.5, 0), xytext=(-12, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(0.5, -0.3, '$0$', ha='center', fontsize=12)
    for t in [-10, -6, -4, -2]:
        ax.text(t, -0.3, str(t), ha='center', fontsize=10)

    # Show two chains collapsing
    np.random.seed(2)
    times = np.array([-10, -8, -6, -4, -2, 0])
    upper = np.array([4.0, 3.5, 3.0, 2.5, 2.2, 2.0])
    lower = np.array([0.5, 0.8, 1.2, 1.6, 1.9, 2.0])

    ax.plot(times, upper, 'b-o', lw=2, ms=5, label='$\\xi_{\\max}$ chain')
    ax.plot(times, lower, 'r-o', lw=2, ms=5, label='$\\xi_{\\min}$ chain')
    ax.scatter([0], [2.0], color='green', s=150, zorder=5, label='Output $\\xi^* \\sim \\pi$')

    ax.axvline(x=-10, color='gray', linestyle='--', alpha=0.5)
    ax.text(-10, 4.3, '$-N_1$', ha='center', fontsize=10, color='gray')
    ax.axvline(x=-6, color='gray', linestyle='--', alpha=0.5)
    ax.text(-6, 4.3, '$-N_2$', ha='center', fontsize=10, color='gray')

    ax.annotate('Chains coalesce', xy=(0, 2.0), xytext=(-3.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=11, color='green')

    ax.set_title('Coupling from the Past (CFTP): schematic', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    savefig('fig_cftp_schematic')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8  –  FPRAS decomposition for independent sets
# ─────────────────────────────────────────────────────────────────────────────
def fig_fpras_decomposition():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Show z_G = product of ratios
    ax.text(1, 1.5, r'$z_G = z_0 \times \frac{z_1}{z_0} \times \frac{z_2}{z_1} \times \cdots \times \frac{z_m}{z_{m-1}}$',
            fontsize=14, ha='left')
    ax.text(1, 0.8, r'$= 2^M \cdot \varrho_1 \cdot \varrho_2 \cdots \varrho_m$',
            fontsize=14, ha='left', color='steelblue')
    ax.text(1, 0.2, r'where $\varrho_i = z_i / z_{i-1} \geq 1/2$ ensures efficient estimation',
            fontsize=11, ha='left', color='gray')
    savefig('fig_fpras_decomposition')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9  –  Motif finding: energy convergence
# ─────────────────────────────────────────────────────────────────────────────
def fig_motif_energy():
    np.random.seed(42)
    # Simulate decreasing energy (simplified)
    steps = np.arange(251)
    energy = 28 - 15 * (1 - np.exp(-steps / 40)) + np.random.normal(0, 0.4, 251)
    energy = np.maximum(energy, 12.5)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, energy, color='steelblue', lw=1.5, alpha=0.9)
    ax.axhline(y=12.62, color='red', linestyle='--', lw=1.5, label=r'$H(\xi^\bullet) = 12.62$')
    ax.set_xlabel('Gibbs step', fontsize=12)
    ax.set_ylabel(r'Energy $H(\xi^{\rm step})$', fontsize=12)
    ax.set_title('Motif finding: energy during Gibbs sampling', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig('fig_motif_energy')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10  –  Ergodic theorem illustration (MCMC estimator convergence)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ergodic_convergence():
    np.random.seed(7)
    # Two-state chain with alpha=0.25, beta=0.125
    alpha, beta = 0.25, 0.125
    true_I = alpha / (alpha + beta)  # = 2/3

    R_max = 3000
    X = np.zeros(R_max, dtype=int)
    X[0] = 0
    for i in range(1, R_max):
        if X[i-1] == 0:
            X[i] = 1 if np.random.rand() < alpha else 0
        else:
            X[i] = 0 if np.random.rand() < beta else 1

    running_mean = np.cumsum(X) / np.arange(1, R_max+1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(running_mean, color='steelblue', lw=1.5, label=r'$\hat{Y}_R = \frac{1}{R}\sum_{j=1}^R f(X_j)$')
    ax.axhline(true_I, color='red', linestyle='--', lw=2, label=f'$I = \\alpha/(\\alpha+\\beta) = {true_I:.3f}$')
    ax.set_xlabel('Steps $R$', fontsize=12)
    ax.set_ylabel(r'$\hat{Y}_R$', fontsize=12)
    ax.set_title('Ergodic theorem: MCMC estimator convergence', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, R_max)
    plt.tight_layout()
    savefig('fig_ergodic_convergence')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11  –  Asymptotic variance comparison: MC vs i.i.d.
# ─────────────────────────────────────────────────────────────────────────────
def fig_asymptotic_variance():
    alpha_vals = np.linspace(0.05, 0.95, 200)
    beta = 0.25

    # ratio zeta^2 / Var(Y^ind) = (2 - alpha - beta) / (alpha + beta)
    ratio = (2 - alpha_vals - beta) / (alpha_vals + beta)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(alpha_vals, ratio, 'steelblue', lw=2)
    ax.axhline(1, color='red', linestyle='--', lw=1.5, label='i.i.d. baseline')
    ax.fill_between(alpha_vals, ratio, 1,
                    where=ratio > 1, alpha=0.15, color='red', label='MC worse than i.i.d.')
    ax.fill_between(alpha_vals, ratio, 1,
                    where=ratio < 1, alpha=0.15, color='green', label='MC better than i.i.d.')
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$\varsigma^2 \,/\, \mathrm{Var}(Y^{\rm ind})$', fontsize=12)
    ax.set_title(r'Asymptotic variance ratio ($\beta=0.25$)', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig('fig_asymptotic_variance')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12  –  Hard-core model configurations at steps 2, 20, 3000
# ─────────────────────────────────────────────────────────────────────────────
def fig_hardcore_steps():
    np.random.seed(3)
    n = 8
    # Gibbs sampler for hard-core model
    def gibbs_hardcore_step(config, n):
        v = (np.random.randint(n), np.random.randint(n))
        i, j = v
        # Check neighbors
        has_neighbor_1 = False
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < n and 0 <= nj < n and config[ni, nj] == 1:
                has_neighbor_1 = True
                break
        if has_neighbor_1:
            config[i, j] = 0
        else:
            config[i, j] = np.random.randint(2)
        return config

    config = np.zeros((n, n), dtype=int)
    snapshots = {}
    step = 0
    for target_step in [2, 20, 3000]:
        while step < target_step:
            config = gibbs_hardcore_step(config, n)
            step += 1
        snapshots[target_step] = config.copy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (s, cfg) in zip(axes, snapshots.items()):
        for i in range(n):
            for j in range(n):
                color = 'steelblue' if cfg[i, j] == 1 else 'lightgray'
                circle = plt.Circle((j+1, n-i), 0.35, color=color, ec='black', lw=1)
                ax.add_patch(circle)
        for i in range(n):
            for j in range(n):
                for di, dj in [(0,1),(1,0)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < n and 0 <= nj < n:
                        ax.plot([j+1, nj+1], [n-i, n-ni], 'k-', lw=0.5, zorder=0)
        cnt = cfg.sum()
        ax.set_xlim(0.3, n+0.7)
        ax.set_ylim(0.3, n+0.7)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Step {s}\n({cnt} particles)', fontsize=11)
    plt.suptitle('Hard-core model: Gibbs sampler evolution', fontsize=12, y=1.02)
    plt.tight_layout()
    savefig('fig_hardcore_steps')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13  –  Regenerative structure of Markov chain
# ─────────────────────────────────────────────────────────────────────────────
def fig_regenerative():
    np.random.seed(11)
    n = 80
    X = np.zeros(n)
    X[0] = 0
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    for i in range(1, n):
        X[i] = 1 if (np.random.rand() < P[int(X[i-1]), 1]) else 0

    # Find returns to state 0
    returns = [0] + [i for i in range(1, n) if X[i] == 0 and X[i-1] != 0]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(range(n), X + 0.5, where='post', color='steelblue', lw=1.5)
    for r in returns:
        ax.axvline(x=r, color='red', linestyle='--', alpha=0.4, lw=1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['$s_1$', '$s_2$'])
    ax.set_xlabel('Step $n$', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('Regenerative structure: returns to $s_0$ define independent cycles', fontsize=12)

    # label some cycles
    for i in range(min(3, len(returns)-1)):
        mid = (returns[i] + returns[i+1]) / 2
        ax.annotate('', xy=(returns[i+1], -0.2), xytext=(returns[i], -0.2),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax.text(mid, -0.35, f'$\\eta_{i+1}$', ha='center', fontsize=10, color='gray')
    ax.set_ylim(-0.5, 2.5)
    plt.tight_layout()
    savefig('fig_regenerative')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14  –  Batch means confidence interval
# ─────────────────────────────────────────────────────────────────────────────
def fig_batch_means():
    np.random.seed(22)
    alpha, beta = 0.25, 0.125
    true_I = alpha / (alpha + beta)
    R = 3000
    X = np.zeros(R, dtype=int)
    for i in range(1, R):
        if X[i-1] == 0:
            X[i] = 1 if np.random.rand() < alpha else 0
        else:
            X[i] = 0 if np.random.rand() < beta else 1
    Y = X.astype(float)

    running = np.cumsum(Y) / np.arange(1, R+1)
    # batch means CI
    N = 100
    batch_size = N
    n_batches = R // batch_size
    batch_means = [Y[k*batch_size:(k+1)*batch_size].mean() for k in range(n_batches)]
    batch_var = np.var(batch_means, ddof=1)
    steps = np.arange(1, R+1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, running, 'steelblue', lw=1.5, label=r'$\hat{Y}_R$')
    # running CI based on batch means (approximate)
    se = np.sqrt(batch_var / n_batches)
    final_est = running[-1]
    ax.fill_between([R//2, R], [final_est-1.96*se, final_est-1.96*se],
                    [final_est+1.96*se, final_est+1.96*se],
                    alpha=0.3, color='orange', label='95% CI (batch means)')
    ax.axhline(true_I, color='red', linestyle='--', lw=2, label=f'True $I={true_I:.3f}$')
    ax.set_xlabel('Steps $R$', fontsize=12)
    ax.set_ylabel(r'$\hat{Y}_R$', fontsize=12)
    ax.set_title('MCMC estimator with batch means confidence interval', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig('fig_batch_means')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 15  –  Siegmund duality / stable simulation illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_siegmund_duality():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: partial order on state space
    ax = axes[0]
    states = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    pos = {(x,y): (x, y) for (x,y) in states}

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    # Draw edges (partial order)
    for (x,y) in states:
        if (x+1, y) in pos:
            ax.annotate('', xy=(x+1, y), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        if (x, y+1) in pos:
            ax.annotate('', xy=(x, y+1), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    for (x,y) in states:
        ax.scatter(x, y, s=300, color='steelblue', zorder=5)
        ax.text(x, y, f'({x},{y})', ha='center', va='center', fontsize=7, color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Partial order on $\\mathcal{S} = \\{0,1,2\\}^2$', fontsize=11)

    # Right: absorption probabilities
    ax = axes[1]
    k1_vals = np.arange(0, 6)
    k2_vals = np.arange(0, 6)
    # Toy absorption probabilities
    probs = np.zeros((6,6))
    for k1 in k1_vals:
        for k2 in k2_vals:
            probs[k1, k2] = (1 - 0.15*k1) * (1 - 0.15*k2)
            probs[k1, k2] = max(0, min(1, probs[k1, k2]))
    im = ax.imshow(probs, origin='lower', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('$k_1$', fontsize=12)
    ax.set_ylabel('$k_2$', fontsize=12)
    ax.set_title(r'$\hat{\pi}(\{(x,y)\}^\downarrow)$: absorption probabilities', fontsize=11)

    plt.tight_layout()
    savefig('fig_siegmund_duality')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 16 – Gibbs sampler: full conditional illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_gibbs_full_conditional():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Show Gibbs sampler updating one coordinate
    ax = axes[0]
    x_vals = np.linspace(-4, 4, 200)
    # Bivariate normal: fix x2 = 1, show conditional of x1
    rho = 0.7
    x2_fixed = 1.0
    cond_mean = rho * x2_fixed
    cond_std = np.sqrt(1 - rho**2)
    cond = np.exp(-0.5 * ((x_vals - cond_mean)/cond_std)**2) / (cond_std * np.sqrt(2*np.pi))
    ax.plot(x_vals, cond, 'steelblue', lw=2.5)
    ax.fill_between(x_vals, cond, alpha=0.2, color='steelblue')
    ax.axvline(cond_mean, color='red', lw=2, linestyle='--', label=f'$E[X_1|X_2={x2_fixed}]={cond_mean:.1f}$')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Full conditional $p(x_1 | x_2 = {x2_fixed})$\n(bivariate normal, $\\rho={rho}$)', fontsize=11)
    ax.legend(fontsize=9)

    # Gibbs sampler trace
    ax = axes[1]
    np.random.seed(1)
    n_steps = 200
    rho = 0.95
    X = np.zeros((n_steps, 2))
    X[0] = [-3, -3]
    for i in range(1, n_steps):
        # Update x1 | x2
        X[i, 0] = np.random.normal(rho * X[i-1, 1], np.sqrt(1 - rho**2))
        # Update x2 | x1
        X[i, 1] = np.random.normal(rho * X[i, 0], np.sqrt(1 - rho**2))
    ax.plot(X[:, 0], X[:, 1], 'o-', ms=2, lw=0.5, color='steelblue', alpha=0.6)
    ax.scatter(X[0, 0], X[0, 1], color='red', s=80, zorder=5, label='Start')
    ax.scatter(X[-1, 0], X[-1, 1], color='green', s=80, zorder=5, label='End')

    # Contours
    x1g = np.linspace(-4, 4, 100)
    x2g = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x1g, x2g)
    Z = np.exp(-0.5 / (1-rho**2) * (X1**2 - 2*rho*X1*X2 + X2**2))
    ax.contour(X1, X2, Z, levels=5, cmap='Reds', alpha=0.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'Gibbs sampler path ($\\rho={rho}$, high correlation)', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig('fig_gibbs_full_conditional')


# ─────────────────────────────────────────────────────────────────────────────
# Run all figure generators
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures for Chapter 6 ...')
    fig_two_state_chain()
    fig_metropolis_acceptance()
    fig_mh_trace()
    fig_graph_coloring()
    fig_hardcore_grid()
    fig_ising_sample()
    fig_cftp_schematic()
    fig_fpras_decomposition()
    fig_motif_energy()
    fig_ergodic_convergence()
    fig_asymptotic_variance()
    fig_hardcore_steps()
    fig_regenerative()
    fig_batch_means()
    fig_siegmund_duality()
    fig_gibbs_full_conditional()
    print('Done.')
