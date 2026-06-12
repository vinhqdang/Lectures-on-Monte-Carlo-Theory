"""
gen_figures.py  —  Generate all figures for Chapter 11: Dynamic VRP slides.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ═════════════════════════════════════════════════════════════════
# Figure 1 — Chapter structure / overview diagram
# ═════════════════════════════════════════════════════════════════
def fig_chapter_structure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    # Central box
    cx, cy = 6, 3
    main = FancyBboxPatch((cx-1.8, cy-0.4), 3.6, 0.8,
                           boxstyle="round,pad=0.1", fc='#2c5f8a', ec='white', lw=2)
    ax.add_patch(main)
    ax.text(cx, cy, "Dynamic VRP", ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')

    # Satellite boxes
    satellites = [
        (1.8, 5.0, "Dynamic\nRequests",    '#e07b39'),
        (1.8, 1.0, "Dynamic Travel\nTimes", '#4caf50'),
        (10.2, 5.0, "Dynamic Vehicle\nAvailability", '#9c27b0'),
        (10.2, 1.0, "Performance\nMeasurement", '#f44336'),
        (6.0, 5.5, "Degree of\nDynamism",  '#607d8b'),
    ]
    for sx, sy, label, color in satellites:
        w, h = 2.4, 0.75
        box = FancyBboxPatch((sx-w/2, sy-h/2), w, h,
                              boxstyle="round,pad=0.08", fc=color, ec='white', lw=1.5, alpha=0.88)
        ax.add_patch(box)
        ax.text(sx, sy, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white')
        ax.annotate("", xy=(cx, cy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color='#555555',
                                   lw=1.2, connectionstyle="arc3,rad=0.0"))

    ax.set_title("Chapter 11 — Dynamic Vehicle Routing Problems: Topic Map",
                 fontsize=13, fontweight='bold', pad=10, color='#2c3e50')
    save(fig, "ch11_overview.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 2 — Static vs Dynamic problem timeline
# ═════════════════════════════════════════════════════════════════
def fig_static_vs_dynamic():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor='white')
    fig.suptitle("Static vs. Dynamic Vehicle Routing", fontsize=13,
                 fontweight='bold', color='#2c3e50')

    colors = {'known': '#2196F3', 'new': '#f44336', 'depot': '#333333'}

    for ax, (title, events) in zip(axes, [
        ("Static VRP — all customer locations known before departure",
         [(0.5, 'Depot', 'depot'), (2, 'C1', 'known'), (4, 'C2', 'known'),
          (6, 'C3', 'known'), (8, 'C4', 'known'), (10, 'Depot', 'depot')]),
        ("Dynamic VRP — new requests arrive while vehicles are en route",
         [(0.5, 'Depot', 'depot'), (2, 'C1', 'known'), (4, 'C2', 'known'),
          (5.5, 'C3\n(new!)', 'new'), (7, 'C4', 'known'),
          (8.8, 'C5\n(new!)', 'new'), (10.5, 'Depot', 'depot')]),
    ]):
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.6, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=10.5, loc='left', pad=4, color='#333')
        # timeline
        ax.axhline(0, color='#aaa', lw=1.5, zorder=1)
        for x, label, ctype in events:
            ax.scatter(x, 0, s=140, color=colors[ctype], zorder=3,
                       edgecolors='white', linewidths=1.2)
            ax.text(x, 0.25, label, ha='center', va='bottom', fontsize=8.5,
                    color=colors[ctype], fontweight='bold')
        # legend
        handles = [mpatches.Patch(color=colors['known'], label='Known before planning'),
                   mpatches.Patch(color=colors['new'],   label='Arrives in real time'),
                   mpatches.Patch(color=colors['depot'], label='Depot')]
        ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "ch11_static_vs_dynamic.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 3 — Degree of Dynamism illustration
# ═════════════════════════════════════════════════════════════════
def fig_degree_of_dynamism():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor='white')
    fig.suptitle("Degree of Dynamism (dod) — Three Scenarios",
                 fontsize=12, fontweight='bold', color='#2c3e50')

    scenarios = [
        ("Low dod\n(mostly static)", 0.1,  '#4caf50'),
        ("Medium dod\n(mixed)", 0.5, '#ff9800'),
        ("High dod\n(fully dynamic)", 0.9,  '#f44336'),
    ]
    np.random.seed(42)
    for ax, (label, dod, color) in zip(axes, scenarios):
        n_total = 20
        n_dynamic = int(dod * n_total)
        n_static  = n_total - n_dynamic
        xs = np.random.uniform(0.1, 0.9, n_total)
        ys = np.random.uniform(0.1, 0.9, n_total)
        # depot
        ax.scatter(0.5, 0.5, marker='s', s=200, color='#333', zorder=5)
        ax.text(0.5, 0.52, 'Depot', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold')
        # static customers
        if n_static > 0:
            ax.scatter(xs[:n_static], ys[:n_static], s=80,
                       color='#2196F3', label='Static', zorder=3, alpha=0.85)
        # dynamic customers
        if n_dynamic > 0:
            ax.scatter(xs[n_static:], ys[n_static:], s=80, marker='^',
                       color=color, label='Dynamic', zorder=3, alpha=0.85)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}\ndod = {dod:.0%} dynamic", fontsize=10,
                     color='#333', pad=6)
        ax.legend(fontsize=7.5, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "ch11_degree_of_dynamism.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 4 — Information flow in a GPS-based dispatching system
# ═════════════════════════════════════════════════════════════════
def fig_information_flow():
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='white')
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("Information Flow in a GPS/Radio-Based Dispatching System",
                 fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

    nodes = {
        'Customer': (1.5, 4.5, '#e07b39'),
        'Call\nCenter':  (1.5, 1.0, '#9c27b0'),
        'Dispatch\nServer': (5.5, 2.75, '#2c5f8a'),
        'Route\nOptimiser': (5.5, 4.5, '#4caf50'),
        'Vehicle\n(GPS)': (9.5, 2.75, '#f44336'),
    }
    for name, (x, y, c) in nodes.items():
        box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8,
                              boxstyle="round,pad=0.1", fc=c, ec='white', lw=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

    edges = [
        ('Customer', 'Dispatch\nServer', 'New request\n(location, time)'),
        ('Call\nCenter', 'Dispatch\nServer', 'Aggregated request'),
        ('Dispatch\nServer', 'Route\nOptimiser', 'Solve sub-problem'),
        ('Route\nOptimiser', 'Dispatch\nServer', 'Updated plan'),
        ('Dispatch\nServer', 'Vehicle\n(GPS)', 'Dispatch instruction'),
        ('Vehicle\n(GPS)', 'Dispatch\nServer', 'Position / status'),
    ]
    for src, dst, label in edges:
        sx, sy, _ = nodes[src]
        dx, dy, _ = nodes[dst]
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color='#555',
                                   lw=1.5, connectionstyle="arc3,rad=0.12"))
        mx, my = (sx+dx)/2, (sy+dy)/2
        ax.text(mx, my+0.18, label, ha='center', va='bottom', fontsize=7.5,
                color='#444', style='italic',
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

    save(fig, "ch11_information_flow.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 5 — Insertion heuristic illustration (dynamic request)
# ═════════════════════════════════════════════════════════════════
def fig_insertion_heuristic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    fig.suptitle("Cheapest Insertion of a New Dynamic Request",
                 fontsize=12, fontweight='bold', color='#2c3e50')

    depot = np.array([0, 0])
    customers_before = {
        'Depot':  depot,
        'A': np.array([2, 3]),
        'B': np.array([4, 1]),
        'C': np.array([6, 4]),
    }
    route_before = ['Depot', 'A', 'B', 'C', 'Depot']
    new_request = np.array([3, 2.5])

    def plot_route(ax, customers, route, new=None, insert_pos=None, title=''):
        colors_map = {'Depot': '#333', 'A': '#2196F3', 'B': '#2196F3',
                      'C': '#2196F3', 'NEW': '#f44336'}
        pts = {k: customers[k] for k in customers}
        if new is not None:
            pts['NEW'] = new

        # draw route
        route_pts = [pts[n] for n in route]
        xs = [p[0] for p in route_pts]
        ys = [p[1] for p in route_pts]
        ax.plot(xs, ys, '-o', color='#555', lw=1.8, zorder=2, markersize=0)

        for name, pos in pts.items():
            c = '#f44336' if name == 'NEW' else ('#555' if name == 'Depot' else '#2196F3')
            m = 's' if name == 'Depot' else ('*' if name == 'NEW' else 'o')
            ax.scatter(*pos, s=200 if name != 'Depot' else 250, color=c,
                       zorder=4, edgecolors='white', linewidths=1.2, marker=m)
            off = [0.15, 0.15]
            ax.text(pos[0]+off[0], pos[1]+off[1], name, fontsize=10,
                    fontweight='bold', color=c)

        ax.set_xlim(-0.5, 8); ax.set_ylim(-1, 5.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=10, pad=5)

    # Before
    plot_route(axes[0], customers_before, route_before,
               new=new_request,
               title='Before: existing route + new request (red star)')

    # After insertion between B and C
    customers_after = dict(customers_before)
    route_after = ['Depot', 'A', 'B', 'NEW', 'C', 'Depot']
    customers_final = {**customers_before, 'NEW': new_request}
    plot_route(axes[1], customers_final, route_after,
               title='After: cheapest insertion between B and C')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "ch11_insertion_heuristic.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 6 — Anticipatory vs. Reactive routing comparison
# ═════════════════════════════════════════════════════════════════
def fig_reactive_vs_anticipatory():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')
    fig.suptitle("Reactive vs. Anticipatory Routing Strategies",
                 fontsize=12, fontweight='bold', color='#2c3e50')

    np.random.seed(7)
    depot = np.array([5, 5])
    known = np.random.uniform(1, 9, (8, 2))
    future = np.random.uniform(2, 8, (4, 2))  # anticipated future requests

    for ax, (title, show_future, color_future) in zip(axes, [
        ("Reactive: only known requests\nare served in current plan", False, None),
        ("Anticipatory: vehicles positioned\nto cover expected future requests", True, '#ff9800'),
    ]):
        ax.scatter(*depot, marker='s', s=300, color='#333', zorder=5)
        ax.text(depot[0]+0.2, depot[1]+0.2, 'Depot', fontsize=9, fontweight='bold')
        ax.scatter(known[:, 0], known[:, 1], s=100, color='#2196F3',
                   label='Known customers', zorder=4, edgecolors='white')

        # simple reactive tour (nearest-neighbour approximation for illustration)
        order = list(range(len(known)))
        if not show_future:
            pts = [depot] + [known[i] for i in order] + [depot]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, '-', color='#2196F3', lw=1.8, alpha=0.7, label='Route')
        else:
            # Show anticipated positions
            ax.scatter(future[:, 0], future[:, 1], s=100, marker='^',
                       color=color_future, label='Anticipated future', zorder=4,
                       edgecolors='white', alpha=0.7)
            pts = [depot] + [known[i] for i in order] + [depot]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, '-', color='#2196F3', lw=1.8, alpha=0.7, label='Route')
            # Dashed lines to anticipated
            for fp in future:
                ax.plot([depot[0], fp[0]], [depot[1], fp[1]], '--',
                        color=color_future, lw=1, alpha=0.5)

        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=10, pad=5)
        ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "ch11_reactive_vs_anticipatory.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 7 — Time-dependent travel times (congestion model)
# ═════════════════════════════════════════════════════════════════
def fig_time_dependent_travel():
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')

    hours = np.linspace(0, 24, 300)
    # Synthetic congestion profile: morning + evening peaks
    speed_factor = (
        0.4 * np.exp(-((hours - 8)**2) / 2)   # morning rush
        + 0.5 * np.exp(-((hours - 17)**2) / 2.5)  # evening rush
    )
    base_speed = 60  # km/h
    speed = base_speed * (1 - speed_factor)
    travel_time = 30 / speed * 60  # minutes to travel 30 km

    ax.plot(hours, travel_time, color='#e07b39', lw=2.5, label='Travel time (30 km arc)')
    ax.axhline(30, color='#2196F3', lw=1.5, linestyle='--',
               label='Free-flow baseline (30 min)')
    ax.fill_between(hours, 30, travel_time, where=(travel_time > 30),
                    color='#f44336', alpha=0.15, label='Congestion delay')

    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Travel time (minutes)", fontsize=11)
    ax.set_title("Time-Dependent Travel Times — Congestion Profile",
                 fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], rotation=30)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(fig, "ch11_time_dependent_travel.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 8 — Dynamic vehicle availability (breakdowns)
# ═════════════════════════════════════════════════════════════════
def fig_vehicle_availability():
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    vehicles = ['Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4']
    np.random.seed(99)
    yticks = range(len(vehicles))

    for i, v in enumerate(vehicles):
        # Normal operation segments
        starts = [0]
        ends = [24]
        color = '#4caf50'
        ax.barh(i, 24, left=0, height=0.5, color='#4caf50', alpha=0.4, label='Available' if i == 0 else '')

        if i == 1:
            # breakdown 10–14
            ax.barh(i, 4, left=10, height=0.5, color='#f44336', alpha=0.7,
                    label='Breakdown' if i == 1 else '')
            ax.text(12, i+0.32, 'Breakdown\n(10–14h)', ha='center', va='bottom',
                    fontsize=7.5, color='white', fontweight='bold')
        if i == 3:
            # not available until 8
            ax.barh(i, 8, left=0, height=0.5, color='#607d8b', alpha=0.7,
                    label='Unavailable' if i == 3 else '')
            ax.text(4, i+0.32, 'Unavailable\n(0–8h)', ha='center', va='bottom',
                    fontsize=7.5, color='white', fontweight='bold')

    ax.set_yticks(list(yticks))
    ax.set_yticklabels(vehicles, fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=30, fontsize=8)
    ax.set_title("Dynamic Vehicle Availability — Breakdowns and Availability Windows",
                 fontsize=12, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(fig, "ch11_vehicle_availability.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 9 — Sampling-based anticipatory algorithm flowchart
# ═════════════════════════════════════════════════════════════════
def fig_sampling_algorithm():
    fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
    ax.set_xlim(0, 9); ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Sampling-Based Anticipatory Algorithm for Dynamic VRP",
                 fontsize=11, fontweight='bold', color='#2c3e50', pad=8)

    boxes = [
        (4.5, 7.3, "Observe current system state\n(confirmed + in-progress requests)",
         '#2c5f8a', 1.2),
        (4.5, 5.9, "Generate S sample scenarios\nof future request arrivals",
         '#4caf50', 1.0),
        (4.5, 4.5, "For each sample: solve VRP\n(offline optimiser)",
         '#e07b39', 1.0),
        (4.5, 3.1, "Score each action by\naverage objective across samples",
         '#9c27b0', 1.0),
        (4.5, 1.7, "Execute best-scoring\nimmediate action",
         '#f44336', 1.0),
        (4.5, 0.5, "Wait for next event\n(new request or vehicle idle)",
         '#607d8b', 0.8),
    ]

    prev_y = None
    for (x, y, text, color, h) in boxes:
        box = FancyBboxPatch((x-2.2, y-h/2), 4.4, h,
                              boxstyle="round,pad=0.1", fc=color, ec='white', lw=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', multialignment='center')
        if prev_y is not None:
            ax.annotate("", xy=(x, y + h/2), xytext=(x, prev_y - boxes[boxes.index((x,y,text,color,h))-1][4]/2),
                        arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.5))
        prev_y = y

    # Loop arrow
    ax.annotate("", xy=(6.8, 7.0), xytext=(6.8, 0.5),
                arrowprops=dict(arrowstyle="-|>", color='#777', lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(7.6, 3.8, "Repeat\nloop", ha='center', fontsize=8.5, color='#777', style='italic')

    save(fig, "ch11_sampling_algorithm.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 10 — Multiple plan / rolling horizon approach
# ═════════════════════════════════════════════════════════════════
def fig_rolling_horizon():
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Rolling Horizon / Periodic Re-optimisation in Dynamic VRP",
                 fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

    # Timeline
    ax.axhline(1.5, xmin=0.04, xmax=0.96, color='#aaa', lw=2)

    periods = [(1.0, 3.5, 'Period 1\n(plan built)', '#2196F3'),
               (4.0, 6.5, 'Period 2\n(re-optimise)', '#4caf50'),
               (7.0, 9.5, 'Period 3\n(re-optimise)', '#e07b39'),
               (10.0, 11.5, 'Period 4', '#9c27b0')]

    for x0, x1, label, c in periods:
        ax.barh(1.5, x1-x0, left=x0, height=0.6, color=c, alpha=0.7)
        ax.text((x0+x1)/2, 1.5, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')
        # vertical divider
        ax.axvline(x1, ymin=0.25, ymax=0.8, color='#888', lw=1, linestyle='--')

    # New requests appearing
    new_reqs = [(3.0, 'New req R1', '#f44336'), (5.5, 'New req R2', '#f44336'),
                (8.0, 'New req R3', '#f44336')]
    for xr, label, c in new_reqs:
        ax.scatter(xr, 1.5, s=150, color=c, zorder=5, edgecolors='white', lw=1.5)
        ax.text(xr, 0.75, label, ha='center', fontsize=8, color=c, fontweight='bold')
        ax.annotate("", xy=(xr, 1.2), xytext=(xr, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))

    # Executed portion
    ax.barh(3.0, 3.0, left=1.0, height=0.4, color='#607d8b', alpha=0.8)
    ax.text(2.5, 3.0, 'Committed / executed portion', ha='center', va='center',
            fontsize=8, color='white')

    ax.text(0.5, 1.5, 'Time →', ha='center', va='center', fontsize=9, color='#555')

    save(fig, "ch11_rolling_horizon.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 11 — Benchmark performance comparison (bar chart)
# ═════════════════════════════════════════════════════════════════
def fig_benchmark_comparison():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='white')

    methods = ['Reactive\nInsertion', 'Periodic\nRe-opt', 'Multiple\nPlans',
               'Sampling\n(S=10)', 'Sampling\n(S=100)', 'Anticipatory\nDP']
    # Illustrative values (% of offline optimal cost — lower is better)
    perf_low_dod  = [108, 103, 101, 100, 99, 98]
    perf_high_dod = [135, 125, 118, 112, 108, 105]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, perf_low_dod,  w, label='Low dod (10%)',  color='#4caf50', alpha=0.85)
    bars2 = ax.bar(x + w/2, perf_high_dod, w, label='High dod (90%)', color='#f44336', alpha=0.85)

    ax.axhline(100, color='#333', lw=1.5, linestyle='--', label='Offline optimal (100%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel("Total distance (% of offline optimal)", fontsize=10)
    ax.set_title("Illustrative Performance Comparison across Dynamic VRP Methods",
                 fontsize=11, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=9)
    ax.set_ylim(90, 145)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h}%',
                ha='center', va='bottom', fontsize=7.5)

    save(fig, "ch11_benchmark_comparison.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 12 — Dynamic VRPTW: time windows and travel times
# ═════════════════════════════════════════════════════════════════
def fig_vrptw_dynamic():
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='white')
    ax.set_xlim(0, 24); ax.set_ylim(-0.5, 6)
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_title("Dynamic VRPTW: Time Windows for Known and Dynamic Requests",
                 fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_facecolor('#f8f9fa')

    customers = [
        ('C1 (known)',   1, 5,  4.5, '#2196F3'),
        ('C2 (known)',   3, 9,  3.5, '#2196F3'),
        ('C3 (known)',   6, 12, 2.5, '#2196F3'),
        ('C4 (dynamic)', 8, 14, 1.5, '#f44336'),   # arrives at t=7
        ('C5 (dynamic)', 11, 18, 0.5, '#f44336'),  # arrives at t=10
    ]

    for label, tw_open, tw_close, y, color in customers:
        ax.barh(y, tw_close - tw_open, left=tw_open, height=0.5,
                color=color, alpha=0.7, edgecolor='white')
        ax.text(tw_open - 0.2, y, label, ha='right', va='center',
                fontsize=9, color=color, fontweight='bold')
        ax.text((tw_open + tw_close)/2, y, f"[{tw_open}h–{tw_close}h]",
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Mark request arrivals
    ax.axvline(7,  color='#f44336', lw=1.5, linestyle=':', alpha=0.7)
    ax.text(7.1, 5.5, 'C4 arrives\nat t=7h', fontsize=8, color='#f44336', style='italic')
    ax.axvline(10, color='#f44336', lw=1.5, linestyle=':', alpha=0.7)
    ax.text(10.1, 5.0, 'C5 arrives\nat t=10h', fontsize=8, color='#f44336', style='italic')

    handles = [mpatches.Patch(color='#2196F3', label='Known requests'),
               mpatches.Patch(color='#f44336', label='Dynamic requests')]
    ax.legend(handles=handles, fontsize=9, loc='upper right')
    ax.set_yticks([])
    ax.set_xticks(range(0, 25, 2))
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(fig, "ch11_vrptw_dynamic.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 13 — Markov Decision Process structure for dynamic VRP
# ═════════════════════════════════════════════════════════════════
def fig_mdp_structure():
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='white')
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("Markov Decision Process (MDP) Framework for Dynamic VRP",
                 fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

    states  = [(1.5, 2.75, 'State $S_t$\n(vehicle pos,\npending requests)'),
               (5.5, 2.75, 'State $S_{t+1}$'),
               (9.5, 2.75, 'State $S_{t+2}$')]
    actions = [(3.5, 4.2, 'Action $A_t$\n(dispatch decision)'),
               (7.5, 4.2, 'Action $A_{t+1}$')]
    rewards = [(3.5, 1.3, 'Reward $R_t$\n(−distance)'),
               (7.5, 1.3, 'Reward $R_{t+1}$')]

    for (x, y, label), color in zip(states, ['#2c5f8a', '#2c5f8a', '#2c5f8a']):
        circle = Circle((x, y), 0.65, fc=color, ec='white', lw=1.5, alpha=0.9, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', multialignment='center')

    for (x, y, label), color in zip(actions, ['#4caf50', '#4caf50']):
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                              boxstyle="round,pad=0.05", fc=color, ec='white', lw=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', multialignment='center')

    for (x, y, label), color in zip(rewards, ['#e07b39', '#e07b39']):
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                              boxstyle="round,pad=0.05", fc=color, ec='white', lw=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', multialignment='center')

    # Arrows: state -> action -> state+1
    arrows = [
        ((2.15, 2.75), (2.7, 4.0)),   # S0 -> A0
        ((4.3,  4.0),  (4.85, 2.75)), # A0 -> S1
        ((6.15, 2.75), (6.7, 4.0)),   # S1 -> A1
        ((8.3,  4.0),  (8.85, 2.75)), # A1 -> S2
        ((3.5, 1.6),   (3.5, 2.1)),   # R0 up
        ((7.5, 1.6),   (7.5, 2.1)),   # R1 up
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color='#555', lw=1.3))

    save(fig, "ch11_mdp_structure.pdf")


# ═════════════════════════════════════════════════════════════════
# Figure 14 — Waiting strategies comparison
# ═════════════════════════════════════════════════════════════════
def fig_waiting_strategies():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor='white')
    fig.suptitle("Waiting Strategies for Idle Vehicles in Dynamic VRP",
                 fontsize=12, fontweight='bold', color='#2c3e50')

    np.random.seed(42)
    depot = np.array([5, 5])
    served = np.random.uniform(1, 9, (5, 2))

    strategies = [
        ("No Wait\n(vehicle returns\nto depot)", depot, '#f44336'),
        ("Wait at Last\nCustomer Location", served[-1], '#e07b39'),
        ("Optimal Wait\n(expected future\ncenter)", np.array([4.5, 5.5]), '#4caf50'),
    ]

    for ax, (title, wait_pos, color) in zip(axes, strategies):
        ax.scatter(*depot, marker='s', s=250, color='#333', zorder=5)
        ax.text(depot[0]+0.2, depot[1]+0.2, 'Depot', fontsize=8.5, fontweight='bold')
        ax.scatter(served[:, 0], served[:, 1], s=80, color='#2196F3',
                   zorder=4, edgecolors='white')
        # Vehicle waiting position
        ax.scatter(*wait_pos, marker='D', s=200, color=color, zorder=6,
                   edgecolors='white', lw=1.5)
        ax.text(wait_pos[0]+0.2, wait_pos[1]+0.2, 'Wait\nhere', fontsize=8,
                color=color, fontweight='bold')
        # Route to wait position
        ax.plot([served[-1][0], wait_pos[0]], [served[-1][1], wait_pos[1]],
                '--', color=color, lw=1.5, alpha=0.7)

        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "ch11_waiting_strategies.pdf")


# ═════════════════════════════════════════════════════════════════
# Run all figure generators
# ═════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures for Chapter 11: Dynamic VRP...")
    fig_chapter_structure()
    fig_static_vs_dynamic()
    fig_degree_of_dynamism()
    fig_information_flow()
    fig_insertion_heuristic()
    fig_reactive_vs_anticipatory()
    fig_time_dependent_travel()
    fig_vehicle_availability()
    fig_sampling_algorithm()
    fig_rolling_horizon()
    fig_benchmark_comparison()
    fig_vrptw_dynamic()
    fig_mdp_structure()
    fig_waiting_strategies()
    print(f"\nAll figures written to: {OUT}")
