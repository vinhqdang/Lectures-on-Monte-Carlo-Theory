"""
gen_figures.py  --  Generate all figures for Chapter 8 slides.
Saves every figure as PDF in ./figures/
Run with:  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import poisson
import os

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGDIR, name + '.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')

# ──────────────────────────────────────────────────────────
# Fig 1:  Homogeneous Poisson Process sample path
# ──────────────────────────────────────────────────────────
def fig_homogeneous_poisson():
    rng = np.random.default_rng(42)
    lam = 1.0
    t_max = 8.0

    # Generate arrivals
    arrivals = []
    t = 0.0
    while t < t_max:
        t += rng.exponential(1/lam)
        if t < t_max:
            arrivals.append(t)

    arrivals = np.array(arrivals)
    # Build step function
    times = np.concatenate([[0], arrivals, [t_max]])
    counts = np.arange(len(times))
    counts[-1] = counts[-2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top: N(t) step function
    for i in range(len(arrivals)):
        x0 = arrivals[i-1] if i > 0 else 0
        x1 = arrivals[i]
        ax1.hlines(i, x0, x1, colors='steelblue', lw=2)
        ax1.plot(x1, i+1, 'o', color='steelblue', ms=5, zorder=5)
        ax1.plot(x1, i, 'o', color='white', ms=5, zorder=5,
                 markeredgecolor='steelblue')
    # last segment
    ax1.hlines(len(arrivals), arrivals[-1], t_max, colors='steelblue', lw=2)
    ax1.set_ylabel(r'$N(t)$')
    ax1.set_title('Homogeneous Poisson Process')

    # Bottom: inter-arrival times
    prev = 0
    for i, a in enumerate(arrivals):
        ax2.annotate('', xy=(a, 0.5), xytext=(prev, 0.5),
                     arrowprops=dict(arrowstyle='<->', color='coral', lw=1.5))
        ax2.text((prev + a)/2, 0.6, r'$\tau_{%d}$' % (i+1), ha='center', fontsize=8)
        ax2.plot(a, 0.5, 'k|', ms=10)
        prev = a
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Time $t$')
    ax2.set_xlim(0, t_max)

    plt.tight_layout()
    savefig('poisson_process')

# ──────────────────────────────────────────────────────────
# Fig 2:  Non-homogeneous Poisson Process (thinning)
# ──────────────────────────────────────────────────────────
def fig_nhpp_thinning():
    rng = np.random.default_rng(7)
    t_max = 24.0
    lam_max = 10.0

    def lam(t):
        return lam_max * (0.5 + 0.5 * np.sin(2 * np.pi * t / 24 - np.pi/2))

    # Generate homogeneous Poisson points
    t = 0.0
    h_arrivals = []
    while t < t_max:
        t += rng.exponential(1/lam_max)
        if t < t_max:
            h_arrivals.append(t)
    h_arrivals = np.array(h_arrivals)

    # Thin
    kept = []
    rejected = []
    for a in h_arrivals:
        p = lam(a) / lam_max
        if rng.uniform() < p:
            kept.append(a)
        else:
            rejected.append(a)

    t_grid = np.linspace(0, t_max, 500)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_grid, lam(t_grid), 'k-', lw=1.5, label=r'$\lambda(t)$')
    ax.axhline(lam_max, color='gray', ls='--', lw=1, label=r'$\lambda_{\max}$')
    ax.scatter(rejected, [0]*len(rejected), marker='x', color='gray', s=30, label='Rejected')
    ax.scatter(kept, [0]*len(kept), marker='o', color='steelblue', s=30, label='Accepted')
    ax.set_xlabel('Time $t$ (hours)')
    ax.set_ylabel(r'$\lambda(t)$')
    ax.set_title('Non-homogeneous Poisson Process (Thinning)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig('nhpp_thinning')

# ──────────────────────────────────────────────────────────
# Fig 3:  Poisson Point Process on [0,3]^2
# ──────────────────────────────────────────────────────────
def fig_poisson_2d():
    rng = np.random.default_rng(17)
    # Two intensity regions
    lam_high = 20
    lam_low  = 5
    grid = np.array([[0,1,2],[0,1,2]])  # 3x3 cells

    pts_x, pts_y = [], []
    for row in range(3):
        for col in range(3):
            # checkerboard: high if (row+col)%2==0
            lam = lam_high if (row + col) % 2 == 0 else lam_low
            n = rng.poisson(lam)
            x = rng.uniform(col, col+1, n)
            y = rng.uniform(row, row+1, n)
            pts_x.extend(x)
            pts_y.extend(y)

    fig, ax = plt.subplots(figsize=(4, 4))
    for row in range(3):
        for col in range(3):
            color = '#d0d0d0' if (row + col) % 2 == 0 else '#f5f5f5'
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))
    ax.scatter(pts_x, pts_y, s=8, color='navy')
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([0,1,2,3]); ax.set_yticks([0,1,2,3])
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title('Poisson Point Process on $[0,3]^2$')
    plt.tight_layout()
    savefig('poisson_2d')

# ──────────────────────────────────────────────────────────
# Fig 4:  M/M/1 queue simulations (4 panels)
# ──────────────────────────────────────────────────────────
def simulate_mm1(lam, mu, t_max, rng, L0=5):
    """Birth-and-death simulation of M/M/1 queue."""
    t = 0.0
    L = L0
    times = [0.0]
    states = [L0]
    while t < t_max:
        rate = lam + (mu if L > 0 else 0)
        if rate == 0:
            break
        dt = rng.exponential(1/rate)
        t += dt
        if t > t_max:
            break
        u = rng.uniform()
        if u < lam / rate:
            L += 1
        else:
            L = max(L - 1, 0)
        times.append(t)
        states.append(L)
    times.append(t_max)
    states.append(states[-1])
    return np.array(times), np.array(states)

def fig_mm1_simulations():
    rng = np.random.default_rng(0)
    mu = 1.0
    t_max = 1.5e4
    params = [(0.5, r'$\rho=1/2$', r'$\bar{l}=1$'),
              (8/9, r'$\rho=8/9$', r'$\bar{l}=8$'),
              (1.0, r'$\rho=1$ (critical)', None),
              (1.2, r'$\rho=1.2$ (unstable)', None)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (lam, title, theory) in zip(axes.flat, params):
        times, states = simulate_mm1(lam, mu, t_max, rng)
        ax.plot(times, states, lw=0.4, color='steelblue')
        if theory:
            lbar = lam / (mu - lam)
            ax.axhline(lbar, color='red', ls='--', lw=1.2, label=f'Theory $\\bar{{l}}={lbar:.0f}$')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Queue length $L(t)$')
    plt.suptitle('M/M/1 Queue Simulations ($\\mu=1$)', fontsize=11)
    plt.tight_layout()
    savefig('mm1_simulations')

# ──────────────────────────────────────────────────────────
# Fig 5:  Queueing system diagram (c servers, 1 queue)
# ──────────────────────────────────────────────────────────
def fig_queue_diagram():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_aspect('equal')

    # Arrow in
    ax.annotate('', xy=(2.0, 3.5), xytext=(0.3, 3.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(0.15, 3.8, 'Arrivals\nrate $\\lambda$', ha='center', fontsize=9)

    # Buffer queue boxes
    for i in range(5):
        rect = plt.Rectangle((2.1 + i*0.45, 3.0), 0.4, 1.0,
                              fill=True, facecolor='lightyellow', edgecolor='black', lw=1)
        ax.add_patch(rect)
    ax.text(4.35, 2.5, 'Buffer ($0 \\leq k \\leq \\infty$)', ha='center', fontsize=8)

    # Servers
    server_y = [5.5, 3.5, 1.5]
    labels = ['Server 1', 'Server 2', 'Server $c$']
    for y, lab in zip(server_y, labels):
        ax.annotate('', xy=(6.2, y), xytext=(4.5, 3.5),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
        rect = plt.Rectangle((6.2, y-0.4), 1.5, 0.8,
                              fill=True, facecolor='lightblue', edgecolor='black', lw=1)
        ax.add_patch(rect)
        ax.text(6.95, y, lab, ha='center', va='center', fontsize=8)

    ax.text(5.1, 3.5, '...', fontsize=14, va='center', ha='center')

    # Arrow out
    ax.annotate('', xy=(9.7, 3.5), xytext=(7.8, 3.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(9.85, 3.5, 'Departures', ha='left', fontsize=9, va='center')

    ax.text(4.35, 6.5, 'GI/G/c Queueing System', ha='center', fontsize=11, weight='bold')
    plt.tight_layout()
    savefig('queue_diagram')

# ──────────────────────────────────────────────────────────
# Fig 6:  On-off process
# ──────────────────────────────────────────────────────────
def fig_onoff_process():
    rng = np.random.default_rng(5)
    lam_on = 1/3   # mean 3
    lam_off = 1.0  # mean 1
    t_max = 15.0

    # Generate on-off sequence
    t = 0.0
    state = 1
    events = [0.0]
    states_at_events = [1]
    while t < t_max:
        if state == 1:
            dur = rng.exponential(1/lam_on)
        else:
            dur = rng.exponential(1/lam_off)
        t += dur
        if t > t_max:
            t = t_max
        events.append(t)
        state = 1 - state
        states_at_events.append(state)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    for i in range(len(events)-1):
        color = 'steelblue' if states_at_events[i] == 1 else 'salmon'
        label = 'ON' if (states_at_events[i] == 1 and i == 0) else \
                ('OFF' if (states_at_events[i] == 0 and i == 1) else None)
        ax.hlines(states_at_events[i], events[i], events[i+1],
                  colors=color, lw=4, label=label)
        if i < len(events)-1:
            ax.vlines(events[i+1], 0, 1, colors='gray', lw=0.5, ls=':')

    ax.set_ylim(-0.3, 1.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['OFF', 'ON'])
    ax.set_xlabel('Time $t$')
    ax.set_title('On-Off Process')
    handles = [mpatches.Patch(color='steelblue', label='ON'),
               mpatches.Patch(color='salmon', label='OFF')]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    plt.tight_layout()
    savefig('onoff_process')

# ──────────────────────────────────────────────────────────
# Fig 7:  GI/G/1 queue path (single server)
# ──────────────────────────────────────────────────────────
def fig_gig1_path():
    # Reproduce Table 8.1 from book
    tau = [1.0, 1.5, 4.5, 1.0, 1.0]
    S   = [2.5, 2.5, 0.5, 5.0, 1.0]

    A = np.cumsum(tau)  # [1, 2.5, 7, 8, 9]
    W = [0.0]
    D = [A[0] + S[0]]
    for i in range(1, len(A)):
        w = max(D[-1] - A[i], 0)
        W.append(w)
        D.append(A[i] + w + S[i])

    t_max = 15.0
    fig, ax = plt.subplots(figsize=(9, 3.5))

    for i in range(len(A)):
        # Service interval
        ax.hlines(1, A[i] + W[i], D[i], colors='steelblue', lw=6, alpha=0.6)
        # Waiting interval
        if W[i] > 0:
            ax.hlines(1, A[i], A[i] + W[i], colors='orange', lw=6, alpha=0.6)
        # Markers
        ax.plot(A[i], 1, 'kv', ms=7, zorder=5)  # arrival
        ax.plot(D[i], 1, 'r^', ms=7, zorder=5)  # departure
        # Labels
        ax.text(A[i], 0.6, f'$A_{i+1}$', ha='center', fontsize=8)
        ax.text(D[i], 1.3, f'$D_{i+1}$', ha='center', fontsize=8, color='red')
        if W[i] > 0:
            ax.text(A[i] + W[i]/2, 1.2, f'$W_{i+1}$', ha='center', fontsize=8, color='darkorange')

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([])
    ax.set_xlabel('Time')
    ax.set_title('GI/G/1 Queue: Single Server Path')
    service_patch = mpatches.Patch(color='steelblue', alpha=0.6, label='Service')
    wait_patch = mpatches.Patch(color='orange', alpha=0.6, label='Waiting')
    ax.legend(handles=[service_patch, wait_patch], fontsize=8)
    plt.tight_layout()
    savefig('gig1_path')

# ──────────────────────────────────────────────────────────
# Fig 8:  Slotted ALOHA simulation
# ──────────────────────────────────────────────────────────
def fig_slotted_aloha():
    rng = np.random.default_rng(11)
    lam = 0.31
    h = 0.1
    n_slots = 10000

    X = 0
    Xs = [0]
    throughputs = [0.0]
    success_count = 0
    for n in range(1, n_slots+1):
        A = rng.poisson(lam)
        if X == 0:
            Z = A  # only new arrivals
        else:
            Y = rng.binomial(X, h)
            Z = A + Y  # new + retransmissions
            # actually: Z = 1 if exactly one transmits
        # Count number transmitting
        transmitting = A + (rng.binomial(X, h) if X > 0 else 0)
        if transmitting == 1:
            success = 1
        else:
            success = 0
        success_count += success
        X_new = X + A - success
        X = max(X_new, 0)
        Xs.append(X)
        throughputs.append(success_count / n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(Xs, lw=0.5, color='steelblue')
    ax1.set_xlabel('Slot $n$')
    ax1.set_ylabel('Queue length $X_n$')
    ax1.set_title(f'Slotted ALOHA: $\\lambda={lam}$, $h={h}$')

    ax2.plot(throughputs, lw=0.8, color='steelblue')
    ax2.axhline(lam, color='red', ls='--', lw=1.2, label=f'$\\lambda={lam}$')
    ax2.set_xlabel('Slot $n$')
    ax2.set_ylabel('Local throughput $\\gamma_n$')
    ax2.set_title('Local Throughput')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    savefig('slotted_aloha')

# ──────────────────────────────────────────────────────────
# Fig 9:  TAVC for ergodic processes
# ──────────────────────────────────────────────────────────
def fig_tavc_illustration():
    rng = np.random.default_rng(13)
    lam = 0.5
    mu = 1.0
    rho = lam / mu
    tavc_theory = 2 * rho * (1 + rho) / (mu * (1 - rho)**4)

    # Simulate M/M/1 and estimate running mean of L
    def run_mm1_mean(lam, mu, t_max, rng):
        t = 0; L = 0; integral = 0; last_t = 0
        ts = []; means = []
        record_times = np.linspace(0.01, t_max, 500)
        ri = 0
        while t < t_max:
            rate = lam + (mu if L > 0 else 0)
            dt = rng.exponential(1/rate)
            new_t = min(t + dt, t_max)
            integral += L * (new_t - t)
            t = new_t
            if t >= t_max: break
            if rng.uniform() < lam / rate:
                L += 1
            else:
                L = max(L-1, 0)
            while ri < len(record_times) and record_times[ri] <= t:
                ts.append(record_times[ri])
                means.append(integral / record_times[ri])
                ri += 1
        return np.array(ts), np.array(means)

    ts, means = run_mm1_mean(lam, mu, 5000, rng)
    l_theory = rho / (1 - rho)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ts, means, lw=1, color='steelblue', label=r'$\hat{X}(t)$')
    ax.axhline(l_theory, color='red', ls='--', lw=1.5, label=f'Theory $\\bar{{l}}={l_theory:.2f}$')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel(r'Time-average $\hat{X}(t)$')
    ax.set_title(f'M/M/1 Running Mean ($\\rho={rho}$, theory TAVC = {tavc_theory:.2f})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig('tavc_illustration')

# ──────────────────────────────────────────────────────────
# Fig 10:  Regenerative process cycles
# ──────────────────────────────────────────────────────────
def fig_regenerative_cycles():
    rng = np.random.default_rng(3)
    # On-off: Exp(1/3) on, Exp(1) off
    t = 0; state = 1
    events = [0.0]; states_ev = [1]
    cycle_starts = [0.0]
    while t < 30:
        if state == 1:
            dur = rng.exponential(3)
        else:
            dur = rng.exponential(1)
        t += dur
        events.append(t)
        state = 1 - state
        states_ev.append(state)
        if state == 1:
            cycle_starts.append(t)
    events = np.array(events[:events.__len__()])

    fig, ax = plt.subplots(figsize=(9, 2.5))
    for i in range(len(events)-1):
        col = 'steelblue' if states_ev[i] == 1 else 'salmon'
        ax.hlines(states_ev[i], events[i], min(events[i+1], 30), lw=4, colors=col)

    for i, cs in enumerate(cycle_starts[:-1]):
        ce = cycle_starts[i+1]
        ax.annotate('', xy=(ce, -0.35), xytext=(cs, -0.35),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
        ax.text((cs+ce)/2, -0.55, f'$C_{i}$', ha='center', fontsize=9, color='purple')
        ax.axvline(cs, color='purple', ls=':', lw=0.8, ymin=0)

    ax.set_xlim(0, 30)
    ax.set_ylim(-0.7, 1.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['OFF', 'ON'])
    ax.set_xlabel('Time')
    ax.set_title('Regenerative Cycles in On-Off Process')
    plt.tight_layout()
    savefig('regenerative_cycles')

# ──────────────────────────────────────────────────────────
# Fig 11:  M/G/inf running estimates
# ──────────────────────────────────────────────────────────
def fig_mginf_estimates():
    rng = np.random.default_rng(99)
    lam = 5.0
    mu = 1.0  # ES=1, rho=5
    t_max = 1e4

    # Simulate M/G/inf (exponential service for simplicity)
    arrivals = []
    t = 0.0
    while t < t_max:
        t += rng.exponential(1/lam)
        if t < t_max:
            arrivals.append(t)

    service = rng.exponential(1/mu, len(arrivals))
    departures = np.array(arrivals) + service
    arrivals = np.array(arrivals)

    # Compute L(t) at record times
    record_times = np.linspace(1, t_max, 1000)
    Lhat_vals = []
    running_sum = 0.0
    for i, tr in enumerate(record_times):
        # Count tasks in system
        L_sum = np.sum(np.minimum(departures, tr) - np.minimum(arrivals, tr))
        Lhat_vals.append(L_sum / tr)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(record_times, Lhat_vals, lw=0.8, color='steelblue', label=r'$\hat{L}(t)$')
    ax.axhline(lam/mu, color='red', ls='--', lw=1.5, label=f'$\\rho = {lam/mu:.0f}$')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel(r'$\hat{L}(t)$')
    ax.set_title(r'M/M/$\infty$ Queue: Running Mean ($\lambda=5$, $\mu=1$)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig('mginf_estimates')

# ──────────────────────────────────────────────────────────
# Fig 12:  Little's Law illustration
# ──────────────────────────────────────────────────────────
def fig_littles_law():
    rng = np.random.default_rng(21)
    lam = 2.0; mu = 3.0; rho = lam/mu
    # Theory: L = lambda * W
    W_theory = 1/(mu - lam)
    L_theory = lam * W_theory

    t_max = 500
    # Simulate GI/G/1 (M/M/1 case)
    W_list = []
    w = 0.0
    for _ in range(2000):
        tau = rng.exponential(1/lam)
        s = rng.exponential(1/mu)
        w = max(w - tau, 0) + s
        W_list.append(w - s)  # actual wait

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].hist(W_list, bins=40, density=True, color='steelblue', alpha=0.7, label='Simulated $W$')
    x = np.linspace(0, max(W_list)*1.2, 200)
    axes[0].plot(x, (mu-lam)*np.exp(-(mu-lam)*x), 'r-', lw=2, label='Theory')
    axes[0].axvline(np.mean(W_list), color='orange', ls='--', label=f'$\\hat{{w}}={np.mean(W_list):.2f}$')
    axes[0].set_xlabel('Waiting time $W$')
    axes[0].set_title("Waiting Time Distribution (M/M/1)")
    axes[0].legend(fontsize=8)

    # Little's law: L = lambda * W
    rhos = np.linspace(0.05, 0.95, 50)
    Ls = rhos / (1 - rhos)
    Ws = 1 / (mu * (1 - rhos))
    axes[1].plot(rhos, Ls, 'b-', lw=2, label='$L = \\rho/(1-\\rho)$')
    axes[1].plot(rhos, lam * Ws, 'r--', lw=2, label="$\\lambda W$ (Little's)")
    axes[1].set_xlabel('Traffic intensity $\\rho$')
    axes[1].set_ylabel('Mean queue length $L$')
    axes[1].set_title("Little's Law $L = \\lambda W$")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    savefig('littles_law')

# ──────────────────────────────────────────────────────────
# Fig 13:  TAVC vs rho for M/M/1
# ──────────────────────────────────────────────────────────
def fig_tavc_vs_rho():
    mu = 1.0
    rhos = np.linspace(0.05, 0.95, 200)
    tavc = 2 * rhos * (1 + rhos) / (mu * (1 - rhos)**4)
    sacv = 2 * (1 + rhos) / (mu * rhos * (1 - rhos)**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.semilogy(rhos, tavc, 'b-', lw=2)
    ax1.set_xlabel(r'Traffic intensity $\rho$')
    ax1.set_ylabel(r'TAVC $\varsigma^2$')
    ax1.set_title(r'M/M/1: Asymptotic Variance $\varsigma^2$')
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogy(rhos, sacv, 'r-', lw=2)
    ax2.set_xlabel(r'Traffic intensity $\rho$')
    ax2.set_ylabel(r'SACV $= \varsigma^2/\bar{l}^2$')
    ax2.set_title('Squared Asymptotic Coefficient of Variation')
    ax2.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    savefig('tavc_vs_rho')

# ──────────────────────────────────────────────────────────
# Fig 14:  Required simulation length vs rho
# ──────────────────────────────────────────────────────────
def fig_run_length_vs_rho():
    rhos = np.linspace(0.1, 0.95, 200)
    mu = 1.0
    # SACV for M/M/1: 2(1+rho)/(mu*rho*(1-rho)^2)
    sacv = 2 * (1 + rhos) / (mu * rhos * (1 - rhos)**2)
    t_rel = 400 * sacv   # t_max = 400 * SACV for 10% relative error

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(rhos, t_rel, 'b-', lw=2)
    ax.set_xlabel(r'Traffic intensity $\rho$')
    ax.set_ylabel(r'Required $t_{\max}$ (log scale)')
    ax.set_title('Required Simulation Length for 10\\% Relative Error')
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.axvline(0.9, color='gray', ls=':', lw=1)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    savefig('run_length_vs_rho')

# ──────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures...')
    fig_homogeneous_poisson()
    fig_nhpp_thinning()
    fig_poisson_2d()
    fig_mm1_simulations()
    fig_queue_diagram()
    fig_onoff_process()
    fig_gig1_path()
    fig_slotted_aloha()
    fig_tavc_illustration()
    fig_regenerative_cycles()
    fig_mginf_estimates()
    fig_littles_law()
    fig_tavc_vs_rho()
    fig_run_length_vs_rho()
    print('All done.')
