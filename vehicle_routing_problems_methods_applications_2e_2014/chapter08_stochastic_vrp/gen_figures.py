"""
gen_figures.py  –  Generate all figures for Chapter 8: Stochastic VRP slides.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os
import fitz  # pymupdf

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: VRP taxonomy – deterministic vs stochastic
# ─────────────────────────────────────────────────────────────────────────────
def fig_vrp_taxonomy():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    boxes = [
        (0.5, 0.85, "Vehicle Routing Problem (VRP)", "#2c7bb6", "white"),
        (0.25, 0.55, "Deterministic VRP\n(complete information)", "#74add1", "white"),
        (0.75, 0.55, "Stochastic VRP (SVRP)\n(uncertain parameters)", "#d7191c", "white"),
        (0.10, 0.22, "CVRP\n(capacitated)", "#abd9e9", "black"),
        (0.35, 0.22, "VRPTW\n(time windows)", "#abd9e9", "black"),
        (0.60, 0.22, "Stochastic\nDemands", "#fdae61", "black"),
        (0.75, 0.22, "Stochastic\nCustomers", "#fdae61", "black"),
        (0.90, 0.22, "Stochastic\nTravel Times", "#fdae61", "black"),
    ]

    for (x, y, text, color, tc) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x-0.13, y-0.10), 0.26, 0.20,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', linewidth=1.2,
            transform=ax.transAxes))
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5, color=tc,
                fontweight='bold', transform=ax.transAxes)

    # arrows
    for (x1, y1, x2, y2) in [(0.5,0.75,0.25,0.65),(0.5,0.75,0.75,0.65),
                               (0.25,0.45,0.10,0.32),(0.25,0.45,0.35,0.32),
                               (0.75,0.45,0.60,0.32),(0.75,0.45,0.75,0.32),(0.75,0.45,0.90,0.32)]:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title("Chapter 8 – Stochastic VRP: Problem Taxonomy", fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_vrp_taxonomy.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_vrp_taxonomy.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: A priori route and recourse illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_apriori_recourse():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    depot = np.array([0, 0])
    customers = {
        1: np.array([2, 3]),
        2: np.array([4, 4]),
        3: np.array([5, 1]),
        4: np.array([3, -2]),
        5: np.array([1, -3]),
    }

    def draw_route(ax, route, title, skip=None, col='steelblue'):
        pts = [depot] + [customers[i] for i in route] + [depot]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-o', color=col, lw=2, zorder=2)
        ax.plot(*depot, 's', color='black', markersize=12, zorder=5)
        ax.text(depot[0]+0.15, depot[1]+0.15, 'Depot', fontsize=8)
        for i, c in customers.items():
            if skip and i in skip:
                ax.plot(*c, 'rx', markersize=12, markeredgewidth=2.5, zorder=5)
                ax.text(c[0]+0.15, c[1]+0.15, f'C{i}\n(absent)', fontsize=7.5, color='red')
            else:
                ax.plot(*c, 'o', color=col, markersize=9, zorder=5)
                ax.text(c[0]+0.15, c[1]+0.15, f'C{i}', fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_xlim(-1, 7); ax.set_ylim(-5, 6)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    draw_route(axes[0], [1,2,3,4,5], "A Priori Route\n(planned before realization)", col='#2c7bb6')
    draw_route(axes[1], [1,2,4,5], "Recourse: C3 absent → skip\n(route adjusted online)",
               skip={3}, col='#d7191c')
    # draw dashed line showing skip
    c3 = customers[3]
    axes[1].annotate("", xy=(customers[4][0], customers[4][1]),
                     xytext=(customers[2][0], customers[2][1]),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))

    fig.suptitle("A Priori Optimization and Recourse in SVRP", fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_apriori_recourse.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_apriori_recourse.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Network Flow formulation diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_network_flow():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    nodes = {
        0: (0.5, 0.85, 'Depot\n(node 0)', '#2c7bb6', 'white'),
        1: (0.15, 0.50, 'C1', '#74add1', 'black'),
        2: (0.35, 0.50, 'C2', '#74add1', 'black'),
        3: (0.50, 0.50, 'C3', '#74add1', 'black'),
        4: (0.65, 0.50, 'C4', '#74add1', 'black'),
        5: (0.85, 0.50, 'C5', '#74add1', 'black'),
    }

    for nid, (x, y, lbl, col, tc) in nodes.items():
        r = 0.07 if nid == 0 else 0.05
        circle = plt.Circle((x, y), r, color=col, ec='black', lw=1.5, transform=ax.transAxes, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8, color=tc,
                fontweight='bold', transform=ax.transAxes)

    edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,0)]
    for (u, v) in edges:
        xu, yu = nodes[u][0], nodes[u][1]
        xv, yv = nodes[v][0], nodes[v][1]
        ax.annotate("", xy=(xv, yv), xytext=(xu, yu),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                                   connectionstyle='arc3,rad=0.15'))

    ax.set_title("Network Flow Graph for SVRP\n"
                 r"$x_{ij} \in \{0,1\}$ = arc traversal; $y_i \geq 0$ = demand served at $i$",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_network_flow.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_network_flow.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Integer L-shaped algorithm convergence
# ─────────────────────────────────────────────────────────────────────────────
def fig_lshaped_convergence():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    np.random.seed(42)
    iters = np.arange(1, 31)
    lower = 120 + 30 * (1 - np.exp(-0.15 * iters)) + np.random.normal(0, 1.5, 30)
    upper = 200 - 50 * (1 - np.exp(-0.12 * iters)) + np.random.normal(0, 2, 30)
    lower = np.minimum.accumulate(upper[::-1])[::-1]
    upper = np.maximum.accumulate(lower)

    # smooth
    from scipy.ndimage import uniform_filter1d
    lower_s = uniform_filter1d(lower, 4)
    upper_s = uniform_filter1d(upper, 4)

    ax.fill_between(iters, lower_s, upper_s, alpha=0.25, color='steelblue', label='Gap region')
    ax.plot(iters, lower_s, 'b-o', markersize=4, lw=2, label='Lower bound (LP relaxation)')
    ax.plot(iters, upper_s, 'r-s', markersize=4, lw=2, label='Upper bound (incumbent)')

    ax.axhline(y=(lower_s[-1]+upper_s[-1])/2, color='green', lw=2, ls='--', label='Optimal value')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Objective value', fontsize=11)
    ax.set_title('Integer L-Shaped Algorithm: Bound Convergence', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_lshaped_convergence.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_lshaped_convergence.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Stochastic demands – recourse strategies
# ─────────────────────────────────────────────────────────────────────────────
def fig_stochastic_demands_recourse():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    depot = np.array([0, 0])
    customers_pos = {
        1: np.array([-2, 2]),
        2: np.array([0, 3]),
        3: np.array([2, 2]),
        4: np.array([2, -1]),
        5: np.array([-2, -2]),
    }
    titles = [
        "Preventive Restocking\n(return before overflow)",
        "Corrective Restocking\n(return when capacity exceeded)",
        "Ignore Overflow\n(penalty cost incurred)"
    ]
    colors = ['#2c7bb6', '#d7191c', '#756bb1']

    for idx, (ax, col, title) in enumerate(zip(axes, colors, titles)):
        route = [1, 2, 3, 4, 5]
        if idx == 0:
            # preventive: detour back after C2
            pts = [depot, customers_pos[1], customers_pos[2], depot, customers_pos[3],
                   customers_pos[4], customers_pos[5], depot]
        elif idx == 1:
            # corrective: return after C3 overflows
            pts = [depot, customers_pos[1], customers_pos[2], customers_pos[3], depot,
                   customers_pos[4], customers_pos[5], depot]
        else:
            pts = [depot] + [customers_pos[i] for i in route] + [depot]

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-o', color=col, lw=2, zorder=2)
        ax.plot(*depot, 's', color='black', markersize=12, zorder=5)
        ax.text(0.15, 0.15, 'D', fontsize=8, fontweight='bold')
        for i, c in customers_pos.items():
            ax.plot(*c, 'o', color=col, markersize=9, zorder=5)
            ax.text(c[0]+0.12, c[1]+0.12, f'C{i}', fontsize=8)
        if idx == 0:
            ax.annotate("restocking\ntrip", xy=(0, 1.5), fontsize=7.5, color='darkblue',
                        ha='center', style='italic')
        ax.set_title(title, fontsize=8.5, fontweight='bold')
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 4.5)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    fig.suptitle("Recourse Strategies for Stochastic Demands", fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_stochastic_demands_recourse.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_stochastic_demands_recourse.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Stochastic customers – probability of presence
# ─────────────────────────────────────────────────────────────────────────────
def fig_stochastic_customers():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    np.random.seed(7)
    n = 8
    pos = {0: (0, 0)}  # depot
    for i in range(1, n+1):
        pos[i] = (np.random.uniform(-4, 4), np.random.uniform(-4, 4))
    probs = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.8, 5: 0.3, 6: 0.6, 7: 0.95, 8: 0.4}

    for ax_idx, ax in enumerate(axes):
        ax.plot(*pos[0], 's', color='black', markersize=12, zorder=5)
        ax.text(pos[0][0]+0.2, pos[0][1]+0.2, 'Depot', fontsize=8, fontweight='bold')
        for i in range(1, n+1):
            p = probs[i]
            color = plt.cm.RdYlGn(p)
            ax.plot(*pos[i], 'o', color=color, markersize=10+5*p, zorder=5)
            ax.text(pos[i][0]+0.2, pos[i][1]+0.2, f'C{i}\np={p}', fontsize=7.5)
        ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # left: full a priori route
    route = [0, 7, 1, 4, 2, 6, 3, 8, 5, 0]
    xs = [pos[r][0] for r in route]; ys = [pos[r][1] for r in route]
    axes[0].plot(xs, ys, '--', color='steelblue', lw=1.5, alpha=0.6, zorder=1)
    axes[0].set_title("A Priori Route (all customers planned)\nNode size ∝ presence probability", fontsize=9)

    # right: realized route (only high-prob customers present)
    present = [i for i in range(1, n+1) if probs[i] >= 0.6]
    route2 = [0] + present + [0]
    xs2 = [pos[r][0] for r in route2]; ys2 = [pos[r][1] for r in route2]
    axes[1].plot(xs2, ys2, '-', color='#d7191c', lw=2, zorder=1)
    for i in range(1, n+1):
        if probs[i] < 0.6:
            axes[1].plot(*pos[i], 'rx', markersize=12, markeredgewidth=2.5, zorder=6)
    axes[1].set_title("Realized Route (low-prob customers absent)\nCross = customer not present", fontsize=9)

    fig.suptitle("Stochastic Customers: Presence Probability and Route Realization",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_stochastic_customers.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_stochastic_customers.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Stochastic travel times – time-window feasibility
# ─────────────────────────────────────────────────────────────────────────────
def fig_stochastic_travel_times():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: distribution of arrival time vs time window
    ax = axes[0]
    mu, sigma = 30, 5
    x = np.linspace(10, 55, 300)
    y = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    tw_early, tw_late = 22, 38
    ax.plot(x, y, 'b-', lw=2.5, label=f'Arrival time distribution\nμ={mu}, σ={sigma}')
    ax.axvline(tw_early, color='green', lw=2, ls='--', label=f'Time window [{tw_early}, {tw_late}]')
    ax.axvline(tw_late, color='green', lw=2, ls='--')
    ax.fill_between(x, y, where=(x >= tw_early) & (x <= tw_late),
                    alpha=0.3, color='green', label='Feasible region')
    ax.fill_between(x, y, where=(x < tw_early), alpha=0.3, color='red', label='Early arrival (wait)')
    ax.fill_between(x, y, where=(x > tw_late), alpha=0.3, color='orange', label='Late arrival (penalty)')
    ax.set_xlabel('Arrival time', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Arrival Time Distribution vs. Time Window', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: probability of feasibility as function of time window width
    ax2 = axes[1]
    widths = np.linspace(0, 30, 200)
    # P(arrival in [mu - w/2, mu + w/2])
    from scipy.stats import norm
    probs_feasible = norm.cdf(mu + widths/2, mu, sigma) - norm.cdf(mu - widths/2, mu, sigma)
    ax2.plot(widths, probs_feasible, 'b-', lw=2.5)
    ax2.axhline(0.95, color='red', ls='--', lw=1.8, label='95% service level')
    ax2.axhline(0.90, color='orange', ls='--', lw=1.8, label='90% service level')
    w95 = widths[np.argmin(np.abs(probs_feasible - 0.95))]
    ax2.axvline(w95, color='red', ls=':', lw=1.5)
    ax2.text(w95+0.5, 0.5, f'w={w95:.1f}\nfor 95%', fontsize=8, color='red')
    ax2.set_xlabel('Time window width', fontsize=10)
    ax2.set_ylabel('P(feasible arrival)', fontsize=10)
    ax2.set_title('Feasibility Probability vs. Time Window Width', fontsize=9.5, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Stochastic Travel Times: Time-Window Feasibility Analysis",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_stochastic_travel_times.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_stochastic_travel_times.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Demand distribution and chance constraint illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_chance_constraint():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: individual demand distributions
    ax = axes[0]
    x = np.linspace(0, 15, 300)
    params = [(4, 1.2, '#2c7bb6', 'C1: μ=4'), (6, 1.5, '#d7191c', 'C2: μ=6'),
              (3, 0.8, '#1a9641', 'C3: μ=3'), (5, 1.0, '#fdae61', 'C4: μ=5')]
    for mu, sig, col, lbl in params:
        y = np.exp(-0.5*((x-mu)/sig)**2) / (sig*np.sqrt(2*np.pi))
        ax.plot(x, y, color=col, lw=2, label=lbl)
    ax.axvline(10, color='black', lw=2.5, ls='--', label='Q=10 (vehicle capacity)')
    ax.set_xlabel('Demand', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Individual Customer Demand Distributions', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Right: total route demand distribution
    ax2 = axes[1]
    total_mu = 4 + 6 + 3 + 5  # = 18
    total_sig = np.sqrt(1.2**2 + 1.5**2 + 0.8**2 + 1.0**2)
    x2 = np.linspace(5, 35, 400)
    y2 = np.exp(-0.5*((x2-total_mu)/total_sig)**2) / (total_sig*np.sqrt(2*np.pi))
    ax2.plot(x2, y2, 'b-', lw=2.5, label=f'Total demand: μ={total_mu}, σ={total_sig:.2f}')
    Q = 20
    ax2.axvline(Q, color='red', lw=2.5, ls='--', label=f'Q={Q} (capacity)')
    from scipy.stats import norm as sp_norm
    p_exceed = 1 - sp_norm.cdf(Q, total_mu, total_sig)
    ax2.fill_between(x2, y2, where=(x2 > Q), alpha=0.4, color='red',
                     label=f'P(overflow)={p_exceed:.3f}')
    ax2.fill_between(x2, y2, where=(x2 <= Q), alpha=0.3, color='green',
                     label=f'P(feasible)={1-p_exceed:.3f}')
    ax2.set_xlabel('Total route demand', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Total Demand Distribution & Chance Constraint', fontsize=9.5, fontweight='bold')
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Stochastic Demands: Chance Constraints and Capacity Feasibility",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_chance_constraint.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_chance_constraint.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Reoptimization model – online decision timeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_reoptimization_timeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')

    events = [
        (0.05, "Plan\na priori\nroute", '#2c7bb6'),
        (0.22, "Depart\ndepot", '#2c7bb6'),
        (0.38, "Arrive C1\n(demand\nrevealed)", '#fdae61'),
        (0.52, "Reoptimize\n(online\ncorrection)", '#d7191c'),
        (0.67, "Arrive C2\n(absent!)\n→ skip", '#756bb1'),
        (0.82, "Return\nto depot", '#2c7bb6'),
    ]

    ax.axhline(y=0.5, xmin=0.03, xmax=0.97, color='gray', lw=2.5)
    for x, lbl, col in events:
        ax.plot(x, 0.5, 'o', color=col, markersize=16, zorder=5, transform=ax.transAxes)
        ax.text(x, 0.72, lbl, ha='center', va='bottom', fontsize=8,
                color=col, fontweight='bold', transform=ax.transAxes)
        ax.plot([x, x], [0.5, 0.45], color='gray', lw=1.5, transform=ax.transAxes)

    # arrow
    ax.annotate("", xy=(0.97, 0.5), xytext=(0.03, 0.5),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, 0.1, 'Time', ha='center', fontsize=10, transform=ax.transAxes)
    ax.set_title("Reoptimization Model: Online Decision Timeline for SVRP",
                 fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_reoptimization_timeline.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_reoptimization_timeline.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Benchmark comparison table (Table 8.1 from book)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark_table():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Data based on book p229 Table 8.1
    columns = ['Instance', 'N', '#Routes', 'RAM\n(opt)', 'B&P\n(Gendreau)', 'ILS', '% Gap\nILS/opt']
    data = [
        ['Set A-n32', 32, 3, '2000', '2021', '2034', '1.70%'],
        ['Set A-n33', 33, 3, '1560', '1563', '1581', '1.35%'],
        ['Set B-n34', 34, 3, '1780', '1780', '1803', '1.29%'],
        ['Set B-n41', 41, 4, '2560', '2571', '2598', '1.48%'],
        ['Set B-n78', 78, 8, '8024', '8091', '8156', '1.65%'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c7bb6')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#e8f4f8')
        else:
            cell.set_facecolor('white')

    ax.set_title("Benchmark Results: A Priori SVRP Algorithms\n"
                 "(Illustrative values based on chapter discussion; N = number of customers)",
                 fontsize=9.5, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_benchmark_table.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_benchmark_table.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: VRPSC – single vehicle model illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_vrpsc_single_vehicle():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    np.random.seed(42)
    n = 8
    pos = {0: (0, 0)}
    for i in range(1, n+1):
        angle = 2*np.pi*i/n + np.random.uniform(-0.2, 0.2)
        r = np.random.uniform(2, 4)
        pos[i] = (r*np.cos(angle), r*np.sin(angle))
    probs = {1: 0.8, 2: 0.6, 3: 0.9, 4: 0.5, 5: 0.7, 6: 0.4, 7: 0.85, 8: 0.55}

    for ax_idx, ax in enumerate(axes):
        ax.plot(*pos[0], 's', color='black', markersize=14, zorder=5)
        ax.text(pos[0][0]+0.2, pos[0][1]+0.2, 'Depot', fontsize=8)
        for i in range(1, n+1):
            p = probs[i]
            ax.plot(*pos[i], 'o', color=plt.cm.RdYlGn(p), markersize=10+6*p, zorder=5)
            ax.text(pos[i][0]+0.15, pos[i][1]+0.15, f'C{i}\n({p})', fontsize=7)
        ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # Left: planned route (all)
    route_full = [0,3,7,1,5,2,8,6,4,0]
    xs = [pos[r][0] for r in route_full]; ys = [pos[r][1] for r in route_full]
    axes[0].plot(xs, ys, '--b', lw=1.5, zorder=1)
    axes[0].set_title("VRPSC: A Priori Route\n(all potential customers included)", fontsize=9)

    # Right: one realization (high-prob present)
    present = [i for i in range(1, n+1) if probs[i] >= 0.6]
    route_real = [0] + present + [0]
    xs2 = [pos[r][0] for r in route_real]; ys2 = [pos[r][1] for r in route_real]
    axes[1].plot(xs2, ys2, '-r', lw=2.5, zorder=2)
    for i in range(1, n+1):
        if probs[i] < 0.6:
            axes[1].plot(*pos[i], 'rx', markersize=13, markeredgewidth=2.5, zorder=6)
    axes[1].set_title("One Realization: Only Present Customers Served\n(absent = red cross)", fontsize=9)

    fig.suptitle("Vehicle Routing Problem with Stochastic Customers (VRPSC)",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_vrpsc_single_vehicle.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_vrpsc_single_vehicle.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Crop from book PDF – Table 8.1 (p229)
# ─────────────────────────────────────────────────────────────────────────────
def crop_book_figure():
    pdf_path = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
                "vehicle_routing_problems_methods_applications_2e_2014/"
                "Vehicle Routing_ Problems, Methods, and Applications, Second Edition 2014.pdf")
    try:
        doc = fitz.open(pdf_path)
        # page 229 of book = 0-indexed page 228
        page = doc[228]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
        clip = fitz.Rect(50, 350, 520, 520)  # crop rect (points)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out = os.path.join(OUTDIR, "fig_book_table81.png")
        pix.save(out)
        doc.close()
        print(f"fig_book_table81.png  OK  ({out})")
    except Exception as e:
        print(f"WARNING: Could not crop book figure: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: Summary comparison of SVRP variants
# ─────────────────────────────────────────────────────────────────────────────
def fig_svrp_summary():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    categories = ['Stochastic\nDemands', 'Stochastic\nCustomers', 'Stochastic\nTravel Times']
    aspects = ['Uncertain\nparameter', 'Typical\nrecourse', 'Solution\napproach', 'Key\nchallenge']
    data = [
        ['Customer\ndemand dᵢ', 'Customer\npresence yᵢ', 'Arc travel\ntime tᵢⱼ'],
        ['Restocking\nor penalty', 'Skip absent\ncustomers', 'Wait / accept\nlate penalty'],
        ['Chance\nconstraints', 'VRPSD\n+ stoch prog', 'Robust\nscheduling'],
        ['Capacity\nviolation', 'Expected\ncost', 'Time window\nfeasibility'],
    ]

    colors_row = ['#d4e6f1', '#d5f5e3', '#fdebd0', '#f9ebea']
    x_positions = [0.28, 0.55, 0.82]
    y_positions = [0.82, 0.62, 0.42, 0.22]

    # Header
    for j, (cat, xp) in enumerate(zip(categories, x_positions)):
        ax.add_patch(mpatches.FancyBboxPatch((xp-0.13, 0.90), 0.26, 0.12,
            boxstyle="round,pad=0.01", facecolor='#2c7bb6', ec='black', lw=1,
            transform=ax.transAxes))
        ax.text(xp, 0.96, cat, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', transform=ax.transAxes)

    for i, (aspect, yp, rc) in enumerate(zip(aspects, y_positions, colors_row)):
        ax.add_patch(mpatches.FancyBboxPatch((0.01, yp-0.08), 0.14, 0.16,
            boxstyle="round,pad=0.01", facecolor='#566573', ec='black', lw=1,
            transform=ax.transAxes))
        ax.text(0.08, yp, aspect, ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold', transform=ax.transAxes)
        for j, (val, xp) in enumerate(zip(data[i], x_positions)):
            ax.add_patch(mpatches.FancyBboxPatch((xp-0.13, yp-0.08), 0.26, 0.16,
                boxstyle="round,pad=0.01", facecolor=rc, ec='#aaa', lw=0.8,
                transform=ax.transAxes))
            ax.text(xp, yp, val, ha='center', va='center', fontsize=8.5,
                    transform=ax.transAxes)

    ax.set_title("Summary Comparison: Three Main SVRP Variants",
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_svrp_summary.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_svrp_summary.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14: Markov decision process for reoptimization
# ─────────────────────────────────────────────────────────────────────────────
def fig_mdp_reoptimization():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    states = [
        (0.10, 0.50, "State s₀\n(at depot,\nfull load)"),
        (0.30, 0.75, "State s₁\n(en route C1,\nload L₁)"),
        (0.55, 0.75, "State s₂\n(at C1, demand\nrevealed)"),
        (0.80, 0.75, "State s₃\n(served C1,\nnew load)"),
        (0.55, 0.25, "State s₄\n(route fail:\nreturn depot)"),
        (0.90, 0.50, "..."),
    ]
    colors = ['#2c7bb6','#74add1','#abd9e9','#74add1','#fdae61','gray']

    for (x, y, lbl, col) in [(s[0],s[1],s[2],colors[i]) for i,s in enumerate(states)]:
        ax.add_patch(mpatches.FancyBboxPatch((x-0.09, y-0.13), 0.18, 0.26,
            boxstyle="round,pad=0.02", facecolor=col, ec='black', lw=1.2,
            transform=ax.transAxes))
        ax.text(x, y, lbl, ha='center', va='center', fontsize=7.5, transform=ax.transAxes)

    transitions = [
        (0.10, 0.50, 0.30, 0.75, "drive", 0.1),
        (0.30, 0.75, 0.55, 0.75, "arrive", 0.0),
        (0.55, 0.75, 0.80, 0.75, "serve\n(capacity ok)", 0.1),
        (0.55, 0.75, 0.55, 0.25, "overflow\n→ restock", -0.2),
        (0.80, 0.75, 0.90, 0.50, "continue", -0.1),
    ]
    for (x1,y1,x2,y2,lbl,rad) in transitions:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5,
                                   connectionstyle=f'arc3,rad={rad}'))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.07, lbl, ha='center', fontsize=7, color='darkblue',
                transform=ax.transAxes, style='italic')

    ax.set_title("Markov Decision Process Structure for the Reoptimization Model",
                 fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_mdp_reoptimization.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("fig_mdp_reoptimization.pdf  OK")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig_vrp_taxonomy()
    fig_apriori_recourse()
    fig_network_flow()
    fig_lshaped_convergence()
    fig_stochastic_demands_recourse()
    fig_stochastic_customers()
    fig_stochastic_travel_times()
    fig_chance_constraint()
    fig_reoptimization_timeline()
    fig_benchmark_table()
    crop_book_figure()
    fig_vrpsc_single_vehicle()
    fig_svrp_summary()
    fig_mdp_reoptimization()
    print("\nAll figures generated successfully.")
