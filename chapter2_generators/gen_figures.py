"""
gen_figures.py — Generate all figures for Chapter 2: The Theory of Generators
Saves every figure as a PDF in figures/
Run with: conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from scipy.special import erfc
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name + '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  Saved {name}.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: LCG scatter plot (pairs)
# ─────────────────────────────────────────────────────────────────────────────
def fig_lcg_pairs():
    a, c, M = 137, 187, 256
    x = 1
    seq = []
    for _ in range(256):
        x = (a * x + c) % M
        seq.append(x / M)
    u = seq
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(u[:-1], u[1:], s=4, alpha=0.7, color='steelblue')
    axes[0].set_xlabel(r'$u_i$'); axes[0].set_ylabel(r'$u_{i+1}$')
    axes[0].set_title('LCG pairs $(u_i, u_{i+1})$\n$a=137, c=187, M=256$')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    # Triples projected
    axes[1].scatter(u[:-2], u[2:], s=4, alpha=0.7, color='tomato')
    axes[1].set_xlabel(r'$u_i$'); axes[1].set_ylabel(r'$u_{i+2}$')
    axes[1].set_title('LCG pairs $(u_i, u_{i+2})$')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    savefig('lcg_pairs')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: PCG64 scatter (uniform appearance)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pcg64_pairs():
    rng = np.random.default_rng(42)
    u = rng.random(256)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(u[:-1], u[1:], s=4, alpha=0.7, color='steelblue')
    axes[0].set_xlabel(r'$u_i$'); axes[0].set_ylabel(r'$u_{i+1}$')
    axes[0].set_title('PCG64 pairs $(u_i, u_{i+1})$, $n=256$')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[1].scatter(u[:-2], u[2:], s=4, alpha=0.7, color='tomato')
    axes[1].set_xlabel(r'$u_i$'); axes[1].set_ylabel(r'$u_{i+2}$')
    axes[1].set_title('PCG64 pairs $(u_i, u_{i+2})$')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    savefig('pcg64_pairs')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: PRNG period illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_prng_period():
    """Show state-space cycle: tail + period"""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis('off')
    # Draw a simple chain with a cycle
    # tail: s0 -> s1 -> s2 -> (cycle s3 -> s4 -> s5 -> s3)
    positions = [(0.1, 0.5), (0.25, 0.5), (0.4, 0.5),
                 (0.6, 0.7), (0.78, 0.5), (0.6, 0.3)]
    labels = [r'$s_0$', r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$', r'$s_5$']
    for (x, y), label in zip(positions, labels):
        circ = plt.Circle((x, y), 0.04, color='steelblue', zorder=5)
        ax.add_patch(circ)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color='white', zorder=6)
    # arrows for tail
    for i in range(2):
        x0, y0 = positions[i]; x1, y1 = positions[i+1]
        ax.annotate('', xy=(x1-0.045, y1), xytext=(x0+0.045, y0),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    # cycle arrows
    cycle = [3, 4, 5, 3]
    for i in range(3):
        x0, y0 = positions[cycle[i]]; x1, y1 = positions[cycle[i+1]]
        dx, dy = x1-x0, y1-y0
        norm = (dx**2+dy**2)**0.5
        ax.annotate('', xy=(x1 - 0.04*dx/norm, y1 - 0.04*dy/norm),
                    xytext=(x0 + 0.04*dx/norm, y0 + 0.04*dy/norm),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))
    # transition from tail to cycle
    x0, y0 = positions[2]; x1, y1 = positions[3]
    dx, dy = x1-x0, y1-y0; norm = (dx**2+dy**2)**0.5
    ax.annotate('', xy=(x1-0.04*dx/norm, y1-0.04*dy/norm),
                xytext=(x0+0.04*dx/norm, y0+0.04*dy/norm),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.175, 0.62, 'tail', fontsize=10, color='gray')
    ax.text(0.68, 0.85, 'period', fontsize=10, color='tomato')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('PRNG state sequence: tail + periodic cycle', fontsize=11)
    plt.tight_layout()
    savefig('prng_period')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Fisher-Yates shuffle illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_fisher_yates():
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5))
    states = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 5, 4],
        [1, 2, 4, 5, 3],
        [1, 3, 4, 5, 2],
    ]
    steps = ['Initial', 'Step i=5:\nswap pos 5,4', 'Step i=4:\nswap pos 4,3', 'Step i=3:\nswap pos 3,2']
    colors_base = ['#AED6F1'] * 5
    for ax, state, title in zip(axes, states, steps):
        for j, val in enumerate(state):
            rect = plt.Rectangle((j, 0), 1, 1, color='#AED6F1', ec='black')
            ax.add_patch(rect)
            ax.text(j + 0.5, 0.5, str(val), ha='center', va='center', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 5); ax.set_ylim(0, 1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    plt.suptitle('Fisher-Yates Algorithm: Shuffling $[1,2,3,4,5]$', fontsize=11)
    plt.tight_layout()
    savefig('fisher_yates')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: LCG period-length examples (a,c,M)=(2,5,13) and (1,5,13)
# ─────────────────────────────────────────────────────────────────────────────
def fig_lcg_sequences():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, (a, c, M, x0, title) in zip(axes, [
        (2, 5, 13, 1, r'$(a,c,M)=(2,5,13),\;x_0=1$, period 13'),
        (1, 5, 13, 1, r'$(a,c,M)=(1,5,13),\;x_0=1$, period 12'),
    ]):
        x = x0
        seq = [x]
        for _ in range(30):
            x = (a * x + c) % M
            seq.append(x)
            if x == x0 and len(seq) > 1:
                break
        ax.plot(seq, 'o-', markersize=4, color='steelblue')
        ax.set_xlabel('Step $n$'); ax.set_ylabel(r'$x_n$')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('lcg_sequences')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Birthday paradox probability
# ─────────────────────────────────────────────────────────────────────────────
def fig_birthday():
    r_vals = np.arange(1, 80)
    k = 365
    log_prob_unique = np.cumsum(np.log(np.maximum(1 - np.arange(0, 79)/k, 1e-300)))
    p_multiple = 1 - np.exp(log_prob_unique[r_vals - 1])
    approx = 1 - np.exp(-r_vals**2 / (2*k))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_vals, p_multiple, 'b-', label='Exact')
    ax.plot(r_vals, approx, 'r--', label=r'Approx $1-e^{-r^2/2k}$')
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.axvline(23, color='green', ls=':', lw=1, label=r'$r=23$ (p$\approx$0.507)')
    ax.set_xlabel('Number of people $r$')
    ax.set_ylabel('Probability of at least one shared birthday')
    ax.set_title('Birthday Problem: $k=365$ boxes')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('birthday_problem')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Three sets A, B, C of n=50 points from [0,1) — scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_sets_abc():
    rng = np.random.default_rng(0)
    n = 50
    A = rng.random(n)
    # Set B: nonlinear transform to make it non-uniform
    eps = 1e-7
    B_raw = np.exp(A - 1) - 1 - eps
    m_b = B_raw.min(); M_b = B_raw.max()
    B = (B_raw - m_b) / (M_b - m_b)
    # Set C: quasi-random (Halton-like with van der Corput)
    def vdc(n, base=2):
        seq = []
        for i in range(1, n+1):
            num, denom = 0, 1
            x = i
            while x > 0:
                denom *= base
                num += (x % base) / denom
                x //= base
            seq.append(num)
        return np.array(seq)
    C = vdc(n, base=2)

    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.scatter(A, np.full(n, 2), marker='o', s=30, color='blue',  label='Set A', alpha=0.8)
    ax.scatter(B, np.full(n, 1), marker='x', s=30, color='red',   label='Set B', alpha=0.8)
    ax.scatter(C, np.full(n, 0), marker='^', s=30, color='green', label='Set C', alpha=0.8)
    # partition shading
    partitions = [(0, 0.15), (0.35, 0.6), (0.8, 1.0)]
    for (lo, hi) in partitions:
        ax.axvspan(lo, hi, alpha=0.1, color='gray')
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['Set C', 'Set B', 'Set A'])
    ax.set_xlabel('Value'); ax.set_title(r'Three sets of $n=50$ points from $[0,1)$: A (blue), B (red), C (green)')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    savefig('sets_abc')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Empirical CDFs for sets A, B, C
# ─────────────────────────────────────────────────────────────────────────────
def fig_ecdf_abc():
    rng = np.random.default_rng(0)
    n = 50
    A = rng.random(n)
    eps = 1e-7
    B_raw = np.exp(A - 1) - 1 - eps
    m_b = B_raw.min(); M_b = B_raw.max()
    B = (B_raw - m_b) / (M_b - m_b)
    def vdc(n, base=2):
        seq = []
        for i in range(1, n+1):
            num, denom = 0, 1
            x = i
            while x > 0:
                denom *= base
                num += (x % base) / denom
                x //= base
            seq.append(num)
        return np.array(seq)
    C = vdc(n, base=2)

    t = np.linspace(0, 1, 500)

    def ecdf(data, t):
        return np.array([np.mean(data <= ti) for ti in t])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, data, label, color in zip(axes,
                                       [A, B, C],
                                       ['Set A', 'Set B', 'Set C'],
                                       ['blue', 'red', 'green']):
        ax.plot(t, ecdf(data, t), color=color, lw=1.5, label=f'Empirical c.d.f. {label}')
        ax.plot(t, t, 'k--', lw=1, label='c.d.f. of Uniform(0,1)')
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\hat{F}_n(t)$')
        ax.set_title(f'Empirical c.d.f. of {label}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('ecdf_abc')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: KS test illustration — D_n statistic
# ─────────────────────────────────────────────────────────────────────────────
def fig_ks_illustration():
    rng = np.random.default_rng(7)
    n = 20
    u = np.sort(rng.random(n))
    t = np.linspace(0, 1, 500)
    Fn = np.array([np.mean(u <= ti) for ti in t])
    Ft = t
    diff = np.abs(Fn - Ft)
    idx = np.argmax(diff)
    Dn = diff[idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(np.concatenate([[0], u, [1]]),
            np.concatenate([[0], np.arange(1, n+1)/n, [1]]),
            where='post', color='blue', lw=1.5, label=r'$\hat{F}_n(t)$')
    ax.plot(t, Ft, 'k--', lw=1.5, label=r'$F(t)=t$')
    ax.annotate('', xy=(t[idx], Ft[idx]), xytext=(t[idx], Fn[idx]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(t[idx]+0.03, (Ft[idx]+Fn[idx])/2, r'$D_n$', color='red', fontsize=12)
    ax.set_xlabel('$t$'); ax.set_ylabel('CDF')
    ax.set_title(f'KS statistic $D_n = \\sup|\\hat{{F}}_n(t) - t|$  ($n={n}$, $D_n={Dn:.3f}$)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('ks_illustration')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Chi-square test — partition histogram
# ─────────────────────────────────────────────────────────────────────────────
def fig_chisq_hist():
    rng = np.random.default_rng(0)
    n = 50
    A = rng.random(n)
    eps = 1e-7
    B_raw = np.exp(A - 1) - 1 - eps
    m_b = B_raw.min(); M_b = B_raw.max()
    B = (B_raw - m_b) / (M_b - m_b)
    def vdc(n, base=2):
        seq = []
        for i in range(1, n+1):
            num, denom = 0, 1
            x = i
            while x > 0:
                denom *= base
                num += (x % base) / denom
                x //= base
            seq.append(num)
        return np.array(seq)
    C = vdc(n, base=2)

    bins = [0, 0.15, 0.35, 0.6, 0.8, 1.0]
    expected = [0.15, 0.20, 0.25, 0.20, 0.20]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, data, label, color in zip(axes,
                                       [A, B, C],
                                       ['Set A', 'Set B', 'Set C'],
                                       ['steelblue', 'tomato', 'seagreen']):
        counts, _ = np.histogram(data, bins=bins)
        exp_counts = [e * n for e in expected]
        x = np.arange(len(bins)-1)
        ax.bar(x, counts, color=color, alpha=0.7, label='Observed')
        ax.bar(x, exp_counts, color='gray', alpha=0.4, label='Expected')
        chi2_val = sum((o-e)**2/e for o, e in zip(counts, exp_counts))
        ax.set_xticks(x)
        ax.set_xticklabels([f'$P_{i+1}$' for i in range(5)], fontsize=8)
        ax.set_title(f'{label}: $X^2={chi2_val:.2f}$', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')
    plt.suptitle(r'Chi-Square Test Observed vs Expected Counts ($n=50$, $k=5$ bins)', fontsize=10)
    plt.tight_layout()
    savefig('chisq_hist')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Balls and boxes schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_balls_boxes():
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')
    k = 5; r = 12
    rng = np.random.default_rng(1)
    assignments = rng.integers(0, k, size=r)
    box_contents = [[] for _ in range(k)]
    for ball, box in enumerate(assignments):
        box_contents[box].append(ball + 1)

    box_w, box_h = 1.4, 1.8
    gap = 0.3
    for i in range(k):
        x = i * (box_w + gap)
        rect = plt.Rectangle((x, 0), box_w, box_h, fc='#D6EAF8', ec='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, box_h + 0.15, f'Box {i+1}', ha='center', va='bottom', fontsize=9)
        for j, ball in enumerate(box_contents[i]):
            bx = x + 0.25 + (j % 2) * 0.7
            by = 0.3 + (j // 2) * 0.6
            circ = plt.Circle((bx, by), 0.2, color='#2E86C1', zorder=5)
            ax.add_patch(circ)
            ax.text(bx, by, str(ball), ha='center', va='center', fontsize=7, color='white', zorder=6)

    ax.set_xlim(-0.2, k * (box_w + gap))
    ax.set_ylim(-0.3, box_h + 0.5)
    ax.set_title(f'Balls-and-Boxes: $r={r}$ balls placed into $k={k}$ boxes', fontsize=11)
    plt.tight_layout()
    savefig('balls_boxes')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 12: Arcsine law density and CDF
# ─────────────────────────────────────────────────────────────────────────────
def fig_arcsine():
    w = np.linspace(0.01, 0.99, 500)
    density = 1 / (np.pi * np.sqrt(w * (1 - w)))
    cdf = (2/np.pi) * np.arcsin(np.sqrt(w))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(w, density, 'b-', lw=2)
    axes[0].set_xlabel('$w$'); axes[0].set_ylabel('Density')
    axes[0].set_title(r'Density: $\frac{1}{\pi\sqrt{w(1-w)}}$')
    axes[0].set_ylim(0, 5); axes[0].grid(alpha=0.3)

    axes[1].plot(w, cdf, 'r-', lw=2)
    axes[1].set_xlabel('$w$'); axes[1].set_ylabel('CDF')
    axes[1].set_title(r'Arcsine CDF: $\frac{2}{\pi}\arcsin\sqrt{w}$')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Arcsine Distribution (used in Random Walk tests)', fontsize=11)
    plt.tight_layout()
    savefig('arcsine_law')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 13: Random walk simulation
# ─────────────────────────────────────────────────────────────────────────────
def fig_random_walk():
    rng = np.random.default_rng(42)
    n = 200
    bits = rng.integers(0, 2, size=n)
    X = 2 * bits - 1
    S = np.concatenate([[0], np.cumsum(X)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k = np.arange(n+1)
    axes[0].plot(k, S, 'b-', lw=1)
    axes[0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0].fill_between(k, S, 0, where=(S > 0), alpha=0.2, color='green', label='Above zero')
    axes[0].fill_between(k, S, 0, where=(S < 0), alpha=0.2, color='red', label='Below zero')
    axes[0].set_xlabel('Step $k$'); axes[0].set_ylabel(r'$S_k$')
    axes[0].set_title('Symmetric Random Walk ($n=200$ steps)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Monobit test statistic illustration
    n_trials = 1000
    stats_vals = []
    for _ in range(n_trials):
        b = rng.integers(0, 2, size=200)
        sn = np.sum(2*b - 1)
        stats_vals.append(abs(sn) / np.sqrt(200))
    axes[1].hist(stats_vals, bins=30, density=True, alpha=0.7, color='steelblue', label='Simulated $T$')
    z = np.linspace(0, 4, 300)
    # half-normal density
    axes[1].plot(z, np.sqrt(2/np.pi) * np.exp(-z**2/2), 'r-', lw=2, label='Half-normal')
    axes[1].set_xlabel(r'$T=|S_n/\sqrt{n}|$')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Monobit Statistic')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig('random_walk')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 14: Second-level testing schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig_second_level():
    rng = np.random.default_rng(99)
    R = 200; n = 50
    # Good PRNG: p-values ~ Uniform(0,1)
    p_good = rng.random(R)
    # Bad PRNG: p-values skewed low
    p_bad = rng.beta(0.3, 1.5, size=R)
    # Quasi-random: p-values clustered near 1
    p_quasi = rng.beta(5, 0.5, size=R)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, pvals, label, color in zip(axes,
        [p_good, p_bad, p_quasi],
        ['Good PRNG (uniform p-values)', 'Bad PRNG (small p-values)', 'Quasi-random (large p-values)'],
        ['steelblue', 'tomato', 'seagreen']):
        ax.hist(pvals, bins=20, range=(0,1), density=True, color=color, alpha=0.75, edgecolor='black', lw=0.5)
        ax.axhline(1.0, color='black', ls='--', lw=1, label='Expected')
        ax.set_xlabel('$p$-value'); ax.set_ylabel('Density')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle(f'Second-Level Testing: Histograms of $R={R}$ $p$-values', fontsize=11)
    plt.tight_layout()
    savefig('second_level')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 15: RC4 KSA + PRGA illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_rc4_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    n = 8  # small n for illustration
    # KSA
    S = list(range(n))
    key = [3, 1, 4, 1, 5, 2, 6, 5]
    j = 0
    history_S = [S.copy()]
    for i in range(n):
        j = (j + S[i] + key[i % len(key)]) % n
        S[i], S[j] = S[j], S[i]
        history_S.append(S.copy())

    cmap = plt.cm.Blues
    ax = axes[0]
    ax.imshow([history_S[0], history_S[-1]], cmap=cmap, aspect='auto', vmin=0, vmax=n-1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Initial', 'After KSA'])
    ax.set_xticks(range(n)); ax.set_xticklabels([f'S[{i}]' for i in range(n)], fontsize=7)
    for idx, row in enumerate([history_S[0], history_S[-1]]):
        for col, val in enumerate(row):
            ax.text(col, idx, str(val), ha='center', va='center', fontsize=9,
                    color='white' if val > n//2 else 'black')
    ax.set_title('KSA: Initial vs Final State Array', fontsize=9)

    # PRGA - output bytes
    i_r, j_r = 0, 0
    outputs = []
    for _ in range(n):
        i_r = (i_r + 1) % n
        j_r = (j_r + S[i_r]) % n
        S[i_r], S[j_r] = S[j_r], S[i_r]
        t = (S[i_r] + S[j_r]) % n
        outputs.append(S[t])

    ax2 = axes[1]
    ax2.bar(range(len(outputs)), outputs, color='tomato', alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(outputs)))
    ax2.set_xticklabels([f'$x_{i+1}$' for i in range(len(outputs))], fontsize=9)
    ax2.set_ylabel('Output byte value'); ax2.set_ylim(0, n)
    ax2.set_title(f'PRGA: First {len(outputs)} output bytes (mod {n})', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('RC4 Algorithm: KSA + PRGA (illustrated with $n=8$)', fontsize=11)
    plt.tight_layout()
    savefig('rc4_schematic')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 16: LFSR shift register illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_lfsr():
    """Illustrate an LFSR with taps at positions 1,4 for a 4-bit register."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    n = 4  # register length
    # draw register cells
    for i in range(n):
        rect = plt.Rectangle((i * 1.5, 0.5), 1.2, 0.8, fc='#AED6F1', ec='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(i * 1.5 + 0.6, 0.9, f'$b_{{{i+1}}}$', ha='center', va='center', fontsize=12)

    # feedback arrow
    ax.annotate('', xy=(0, 0.5), xytext=(4.5 * 1.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=2,
                                connectionstyle='arc3,rad=-0.4'))
    # XOR symbol
    ax.text(-0.8, 0.5, r'$\oplus$', fontsize=16, color='tomato', ha='center', va='center')
    # tap indicators
    for tap in [0, 3]:
        ax.annotate('', xy=(-0.5, 0.5), xytext=(tap * 1.5 + 0.6, 0.5),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5, ls='dashed'))
    ax.text(0.6, 1.6, 'Taps: $a_1, a_4$', fontsize=10, color='tomato')
    ax.set_xlim(-1.5, 7); ax.set_ylim(-0.5, 2.5)
    ax.set_title('Linear Feedback Shift Register (LFSR)\n'
                 r'$x_i = (a_1 x_{i-1} \oplus a_2 x_{i-2} \oplus \cdots \oplus a_k x_{i-k})$',
                 fontsize=11)
    plt.tight_layout()
    savefig('lfsr_schematic')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 17: p-value distribution under H0 (continuous)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pvalue_dist():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(1)
    # Under H0: p-values are Uniform(0,1)
    pvals_h0 = rng.random(500)
    # Under H1: p-values tend to be small
    pvals_h1 = rng.beta(0.5, 3, size=500)
    for ax, pvals, title in zip(axes,
        [pvals_h0, pvals_h1],
        [r'Under $\mathcal{H}_0$ (true randomness): $p\sim\mathcal{U}(0,1)$',
         r'Under $\mathcal{H}_1$ (non-randomness): $p$ small']):
        ax.hist(pvals, bins=20, range=(0,1), density=True, color='steelblue', alpha=0.7, edgecolor='black', lw=0.5)
        ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Uniform density')
        ax.set_xlabel('$p$-value'); ax.set_ylabel('Density')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('$p$-value Distributions', fontsize=11)
    plt.tight_layout()
    savefig('pvalue_distribution')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 18: Normal distribution with shaded p-value tails
# ─────────────────────────────────────────────────────────────────────────────
def fig_normal_pvalue():
    fig, ax = plt.subplots(figsize=(7, 4))
    z = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(z)
    Tobs = 1.8
    ax.plot(z, y, 'b-', lw=2)
    ax.fill_between(z, y, where=(z >= Tobs), alpha=0.4, color='red', label=r'$p$-value region')
    ax.fill_between(z, y, where=(z <= -Tobs), alpha=0.4, color='red')
    ax.axvline(Tobs, color='red', ls='--', lw=1.5, label=f'$T(obs)={Tobs}$')
    ax.axvline(-Tobs, color='red', ls='--', lw=1.5)
    ax.text(-Tobs-0.1, 0.05, r'$-T(\mathrm{obs})$', ha='right', fontsize=9, color='red')
    ax.text(Tobs+0.1, 0.05, r'$T(\mathrm{obs})$', ha='left', fontsize=9, color='red')
    ax.set_xlabel('$z$'); ax.set_ylabel('Density')
    ax.set_title(r'Standard Normal: $p = \mathbb{P}(|Z| \geq T(\mathrm{obs}))$ (shaded area)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('normal_pvalue')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 19: Frequency pairs test illustration (2D grid)
# ─────────────────────────────────────────────────────────────────────────────
def fig_freq_pairs():
    rng = np.random.default_rng(0)
    n = 50
    L = 3
    A = rng.random(n)
    def vdc(n, base=2):
        seq = []
        for i in range(1, n+1):
            num, denom = 0, 1
            x = i
            while x > 0:
                denom *= base
                num += (x % base) / denom
                x //= base
            seq.append(num)
        return np.array(seq)
    C = vdc(n, base=2)

    pairs_A = [(int(A[2*i]*L), int(A[2*i+1]*L)) for i in range(n//2)]
    pairs_C = [(int(C[2*i]*L), int(C[2*i+1]*L)) for i in range(n//2)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, pairs, title in zip(axes, [pairs_A, pairs_C], ['Set A (random)', 'Set C (quasi-random)']):
        grid = np.zeros((L, L))
        for (s, t) in pairs:
            if 0 <= s < L and 0 <= t < L:
                grid[s, t] += 1
        im = ax.imshow(grid, cmap='Blues', vmin=0)
        for i in range(L):
            for j in range(L):
                ax.text(j, i, f'{grid[i,j]:.0f}', ha='center', va='center',
                        fontsize=12, color='black')
        ax.set_xticks(range(L)); ax.set_yticks(range(L))
        ax.set_xlabel('$t$ coordinate'); ax.set_ylabel('$s$ coordinate')
        ax.set_title(f'Frequency of Pairs Test: {title}\n($L={L}$, $r={n//2}$ pairs)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    savefig('freq_pairs_test')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 20: Collision count distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_collision_dist():
    # Show Poisson approximation for collision count
    lam = 4.0
    c_vals = np.arange(0, 15)
    poisson_pmf = stats.poisson.pmf(c_vals, lam)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(c_vals, poisson_pmf, color='steelblue', alpha=0.8, edgecolor='black', lw=0.5,
           label=r'Poisson($\lambda$) approx')
    ax.set_xlabel('Number of collisions $c$')
    ax.set_ylabel(r'$\mathbb{P}(C_{k,r}=c)$')
    ax.set_title(r'Collision Count: Poisson Approximation ($\lambda = r^2/2k$)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    # annotate lambda
    ax.text(0.65, 0.85, f'$\\lambda = {lam}$\n($r^2/2k \\to \\lambda$)',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    plt.tight_layout()
    savefig('collision_distribution')

# ─────────────────────────────────────────────────────────────────────────────
# Run all figure generators
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures for Chapter 2...')
    fig_lcg_pairs()
    fig_pcg64_pairs()
    fig_prng_period()
    fig_fisher_yates()
    fig_lcg_sequences()
    fig_birthday()
    fig_sets_abc()
    fig_ecdf_abc()
    fig_ks_illustration()
    fig_chisq_hist()
    fig_balls_boxes()
    fig_arcsine()
    fig_random_walk()
    fig_second_level()
    fig_rc4_schematic()
    fig_lfsr()
    fig_pvalue_dist()
    fig_normal_pvalue()
    fig_freq_pairs()
    fig_collision_dist()
    print('All figures generated successfully.')
