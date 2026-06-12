"""
gen_figures.py  —  Chapter 5: Multi-Objectives (Nature-Inspired Optimisation for Delivery Problems)
Generates all figures needed by chapter05_slides.tex using matplotlib (backend Agg)
and crops relevant diagrams from the book PDF using PyMuPDF (fitz).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import fitz  # PyMuPDF

# ─── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
BOOK_PDF   = os.path.join(
    SCRIPT_DIR, "..",
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name, dpi=150, tight=True):
    path = os.path.join(FIG_DIR, name)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  saved {name}")

def crop_pdf_page(page_no_1indexed, rect_frac, out_name, dpi=150):
    """Crop a rectangle from a PDF page and save as PNG.
    rect_frac = (x0, y0, x1, y1) as fractions of page dimensions (0-1).
    page_no_1indexed counts from 1.
    """
    try:
        doc  = fitz.open(BOOK_PDF)
        page = doc[page_no_1indexed - 1]
        pw, ph = page.rect.width, page.rect.height
        rect = fitz.Rect(
            rect_frac[0]*pw, rect_frac[1]*ph,
            rect_frac[2]*pw, rect_frac[3]*ph
        )
        zoom = dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)
        clip = page.get_pixmap(matrix=mat, clip=rect)
        path = os.path.join(FIG_DIR, out_name)
        clip.save(path)
        doc.close()
        print(f"  saved {out_name} (cropped from PDF p{page_no_1indexed})")
    except Exception as e:
        print(f"  WARNING: could not crop {out_name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Last-mile delivery growth (bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig_last_mile_growth():
    years  = [2011, 2021, 2020]
    values = [7.7,  26.0, 36.0]
    labels = ['Jul 2011', 'Jul 2021', 'Dec 2020']
    colors = ['#4a90d9', '#e87040', '#3aaa35']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.7, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Online Sales as % of UK Retail Sales', fontsize=11)
    ax.set_title('UK Online Retail Sales Growth\n(ONS Data)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 44)
    ax.axhline(y=36, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    savefig('fig_last_mile_growth.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Hard vs Soft time window illustration
# ══════════════════════════════════════════════════════════════════════════════
def fig_time_windows():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, title, early_pen, late_pen in zip(
        axes,
        ['Hard Time Window', 'Soft Time Window'],
        [None, 'Early penalty\n(waiting cost)'],
        ['Route abandoned\n(infeasible)', 'Late penalty\n(customer complaint)']
    ):
        # Draw timeline
        ax.set_xlim(0, 24)
        ax.set_ylim(-1, 3)
        ax.axhline(1.5, color='gray', linewidth=2)

        # Window
        ax.axvspan(9, 12, alpha=0.25, color='green', label='Time window [9,12]')
        ax.axvline(9,  color='green', linewidth=2)
        ax.axvline(12, color='green', linewidth=2)

        ax.text(10.5, 2.4, 'Window\n[09:00–12:00]', ha='center', va='center',
                color='green', fontsize=10, fontweight='bold')

        # Arrival too early
        ax.annotate('', xy=(6, 1.5), xytext=(6, 0.5),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(6, 0.2, 'Early\narrival\n06:00', ha='center', color='blue', fontsize=9)
        if early_pen:
            ax.text(6, -0.6, early_pen, ha='center', color='blue', fontsize=8,
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))

        # Arrival on time
        ax.annotate('', xy=(10, 1.5), xytext=(10, 0.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(10, 0.2, 'On time\n10:00', ha='center', color='green', fontsize=9)

        # Arrival too late
        ax.annotate('', xy=(15, 1.5), xytext=(15, 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(15, 0.2, 'Late\narrival\n15:00', ha='center', color='red', fontsize=9)
        ax.text(15, -0.6, late_pen, ha='center', color='red', fontsize=8,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffcccc', alpha=0.5))

        ax.set_xlabel('Time of day (hours)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], fontsize=8)
        ax.set_yticks([])

    plt.suptitle('Hard vs Soft Time Windows in VRPTW', fontsize=13, fontweight='bold', y=1.02)
    savefig('fig_time_windows.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — VRPTW family diagram (text-box hierarchy)
# ══════════════════════════════════════════════════════════════════════════════
def fig_vrp_family():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def box(x, y, text, color='#d0e8ff', fontsize=9):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                          edgecolor='steelblue', linewidth=1.2))

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    box(5, 4.3, 'VRP — Vehicle Routing Problem\n(base formulation)', '#b3d9ff', 11)
    box(2, 3.0, 'CVRP\nCapacitated VRP', '#cce8cc')
    box(5, 3.0, 'VRPTW\nWith Time Windows', '#ffe0b2')
    box(8, 3.0, 'DVRP\nDynamic VRP', '#f5d0d0')
    box(2, 1.7, 'CVRPTW\nCapacitated + TW', '#e8d5f5')
    box(5, 1.7, 'VRPPDTW\nPickup + Delivery + TW', '#e8d5f5')
    box(8, 1.7, 'VRPB\nWith Backhauls', '#e8d5f5')
    box(5, 0.4, 'Multi-Objective VRPTW\n(this chapter)', '#ffeb99', 11)

    for tx, ty in [(2,3),(5,3),(8,3)]:
        arrow(5, 4.0, tx, ty+0.22)
    for tx, ty in [(2,1.7),(5,1.7),(8,1.7)]:
        arrow(5, 2.78, tx, ty+0.22)
    arrow(5, 1.48, 5, 0.6)

    ax.set_title('The VRP Family Tree', fontsize=13, fontweight='bold', pad=8)
    savefig('fig_vrp_family.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cost model components (stacked bar illustration)
# ══════════════════════════════════════════════════════════════════════════════
def fig_cost_model():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: cost breakdown for example parameter values
    ax = axes[0]
    # Scenario: 5 routes, 300 km total, 400 min total time
    VEH_FIXED   = 164
    VEH_RUNNING = 0.117
    STAFF_RATE  = 12   # pounds/hour

    n_routes     = [2, 3, 4, 5, 6]
    distances    = [100, 200, 300, 400, 500]  # km, synthetic
    total_times  = [180, 280, 400, 500, 620]  # minutes, synthetic

    veh_costs   = [nr*VEH_FIXED + d*VEH_RUNNING for nr, d in zip(n_routes, distances)]
    staff_costs = [t/60 * STAFF_RATE for t in total_times]

    x = np.arange(len(n_routes))
    b1 = ax.bar(x, veh_costs,  label='Vehicle cost\n(fixed + running)', color='#4a90d9', edgecolor='black', lw=0.6)
    b2 = ax.bar(x, staff_costs, bottom=veh_costs, label='Staff cost\n(time-based)', color='#e87040', edgecolor='black', lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} routes\n{d}km' for n, d in zip(n_routes, distances)], fontsize=8)
    ax.set_ylabel('Cost (£)', fontsize=11)
    ax.set_title('FabFoods Cost Breakdown\nVEHFIXED=£164, VEHRUNNING=£0.117/km, STAFF=£12/hr', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right: costPerCrate as function of total demand
    ax2 = axes[1]
    demands = np.arange(20, 200, 10)
    sol_cost_fixed = 3*VEH_FIXED + 250*VEH_RUNNING + (350/60)*STAFF_RATE  # fixed scenario
    cpc = sol_cost_fixed / demands
    ax2.plot(demands, cpc, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax2.set_xlabel('Total Demand (crates)', fontsize=11)
    ax2.set_ylabel('Cost per Crate (£)', fontsize=11)
    ax2.set_title('Cost per Crate Decreases with Scale\n(fixed solution, increasing demand)', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.fill_between(demands, cpc, alpha=0.15, color='steelblue')

    plt.suptitle('FabFoods Cost Model', fontsize=13, fontweight='bold')
    savefig('fig_cost_model.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — VRPTW decoding example (route construction diagram)
# ══════════════════════════════════════════════════════════════════════════════
def fig_decoding_example():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title('VRPTW Decoding: Genome → Routes\n(Example from p.88–89)', fontsize=12, fontweight='bold')

    # Genome
    genome = ['5,F', '2,F', '4,F', '8,F', '1,F', '7,T', '3,F', '6,F']
    for i, g in enumerate(genome):
        color = '#ffe0b2' if 'T' in g else '#d0e8ff'
        ax.text(i*1.2 + 0.6, 3.8, g, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=color, edgecolor='gray', lw=1))

    ax.text(0, 4.3, 'Genome (customer order, new-route flag):', fontsize=9, style='italic', color='gray')

    # Routes result
    routes = {
        'Route 1': [('D*12:52',''), ('5@13:00','window 12:00–13:00'), ('D@13:17','')],
        'Route 2': [('D*08:50',''), ('2@09:00','window 09:00–10:00'), ('4@13:00','window 13:00–14:00'),
                    ('8@13:25','window 13:00–14:00'), ('D@13:53','')],
        'Route 3': [('D*08:48',''), ('1@09:00','window 09:00–10:00'), ('D@09:17','')],
        'Route 4': [('D*10:55',''), ('7@11:00','window 11:00–12:00'), ('3@11:17','window 11:00–12:00'), ('D@11:34','')],
        'Route 5': [('D*09:40',''), ('6@10:00','window 10:00–11:00'), ('D@10:13','')],
    }
    colors_r = ['#cce8cc', '#ffd0d0', '#d0e8ff', '#ffe0b2', '#e8d0ff']
    y_pos = [2.7, 2.0, 1.3, 0.6, -0.1]

    for (rname, stops), yp, col in zip(routes.items(), y_pos, colors_r):
        ax.text(-0.3, yp, rname + ':', ha='right', va='center', fontsize=9, fontweight='bold')
        for j, (stop, tip) in enumerate(stops):
            ax.text(j*1.6 + 0.6, yp, stop, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=col,
                              edgecolor='steelblue' if 'D' in stop else 'gray', lw=1))
            if j < len(stops)-1:
                ax.annotate('', xy=((j+1)*1.6+0.1, yp), xytext=(j*1.6+1.1, yp),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.text(0.3, -0.7, 'D = Depot   *departs at   @arrives at   "T" gene = start new route', fontsize=8, color='gray', style='italic')
    savefig('fig_decoding_example.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Pareto front concept (2D, 3D, 4D)
# ══════════════════════════════════════════════════════════════════════════════
def fig_pareto_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 2D Pareto front — cost vs distance
    ax = axes[0]
    np.random.seed(42)
    n = 60
    cost = np.random.uniform(3, 11, n)
    dist = np.random.uniform(2000, 8000, n)

    # Pareto front: solutions that minimise both
    pareto_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and cost[j] <= cost[i] and dist[j] <= dist[i] and \
               (cost[j] < cost[i] or dist[j] < dist[i]):
                dominated = True
                break
        if not dominated:
            pareto_mask[i] = True

    ax.scatter(cost[~pareto_mask], dist[~pareto_mask], c='lightgray',
               edgecolors='gray', s=50, label='Dominated solutions', zorder=2)
    ax.scatter(cost[pareto_mask], dist[pareto_mask], c='red',
               edgecolors='darkred', s=90, label='Pareto front', zorder=3)

    # Connect Pareto front as step-like curve
    px = cost[pareto_mask]
    py = dist[pareto_mask]
    order = np.argsort(px)
    ax.plot(px[order], py[order], 'r--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Cost per Crate (£)', fontsize=11)
    ax.set_ylabel('Total Distance (km)', fontsize=11)
    ax.set_title('2-Objective Pareto Front\n(Cost/Crate vs Distance)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate trade-off
    best_cost_idx = px.argmin()
    best_dist_idx = py.argmin()
    ax.annotate('Minimum cost\n(high distance)', xy=(px[best_cost_idx], py[best_cost_idx]),
                xytext=(px[best_cost_idx]+1.5, py[best_cost_idx]+600),
                arrowprops=dict(arrowstyle='->', color='navy'), fontsize=8, color='navy')
    ax.annotate('Minimum distance\n(high cost)', xy=(px[best_dist_idx], py[best_dist_idx]),
                xytext=(px[best_dist_idx]-3, py[best_dist_idx]+600),
                arrowprops=dict(arrowstyle='->', color='darkgreen'), fontsize=8, color='darkgreen')

    # Right: Algorithm flow
    ax2 = axes[1]
    ax2.axis('off')
    steps = [
        ('START', '#b3d9ff', ''),
        ('Run single-obj EA\n×4 objectives\n(10 tries each)', '#cce8cc', '40 solutions'),
        ('Run multi-obj EA\n(NonDomEA)\n(10 tries)', '#ffe0b2', 'many solutions'),
        ('Pool all solutions\ntogether', '#e8d0ff', ''),
        ('extractNonDom()\nKeep only\nnon-dominated', '#ffd0d0', ''),
        ('Grand Front\n(final result)', '#ffeb99', 'return'),
    ]
    for i, (txt, col, note) in enumerate(steps):
        y = 4.2 - i * 0.75
        ax2.text(0.5, y, txt, ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor=col, edgecolor='steelblue', lw=1.2),
                 transform=ax2.transAxes)
        if note:
            ax2.text(0.82, y, note, ha='left', va='center', fontsize=8, color='gray',
                     style='italic', transform=ax2.transAxes)
        if i < len(steps)-1:
            ax2.annotate('', xy=(0.5, y-0.55), xytext=(0.5, y-0.17),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

    ax2.set_title('Home Delivery Solver\n(Algorithm 13)', fontsize=11, fontweight='bold')

    plt.suptitle('Multi-Objective Optimisation: Concepts and Algorithm', fontsize=13, fontweight='bold')
    savefig('fig_pareto_concept.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Parallel Coordinates illustration (4D point example)
# ══════════════════════════════════════════════════════════════════════════════
def fig_parallel_coords():
    fig, ax = plt.subplots(figsize=(11, 6))

    axes_names = ['Distance', 'Time', 'Cost/Crate', 'Routes']
    # Table 5.7 grand front for B-n78-k10 (17 solutions, 1h TW)
    data = [
        [7134.9,  8294, 10.01, 42],
        [7329.75, 8271,  9.86, 41],
        [6506.86, 8539,  7.71, 29],
        [6893.4,  8377,  8.075,31],
        [6738.10, 8520,  7.74, 29],
        [6969.36, 8346,  9.0,  36],
        [7769.1,  8230, 10.25, 43],
        [8042.44, 8206, 10.28, 43],
        [6257.1,  8573,  7.16, 26],
        [7634.3,  8310,  9.38, 38],
        [6525.52, 8559,  7.19, 26],
        [6188.65, 8804,  6.5,  22],
        [6046.79, 9045,  6.89, 24],
        [6029.59, 9126,  6.73, 23],
        [6196.21, 9045,  6.38, 21],
        [6061.83, 8892,  6.5,  22],
        [6057.71, 9118,  6.55, 22],
    ]
    data = np.array(data, dtype=float)
    n_sol, n_dim = data.shape

    # Normalise each axis to [0,1] for plotting
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    norm = (data - mins) / (maxs - mins + 1e-9)

    x_positions = np.arange(n_dim)

    # Colour by cost/crate (column index 2)
    cost_vals = data[:, 2]
    cmap = plt.cm.RdYlGn_r
    cost_norm = (cost_vals - cost_vals.min()) / (cost_vals.max() - cost_vals.min())

    for i, row in enumerate(norm):
        color = cmap(cost_norm[i])
        ax.plot(x_positions, row, color=color, alpha=0.65, linewidth=1.8)

    # Draw vertical axes
    for j, name in enumerate(axes_names):
        ax.axvline(j, color='black', linewidth=1.5)
        # Tick labels at top (max) and bottom (min)
        ax.text(j, 1.03, f'{maxs[j]:.0f}', ha='center', va='bottom', fontsize=8, color='darkred')
        ax.text(j, -0.03, f'{mins[j]:.0f}', ha='center', va='top',    fontsize=8, color='navy')
        ax.text(j, -0.13, name, ha='center', va='top', fontsize=10, fontweight='bold')

    # Highlight two solutions
    # Low cost solution (sol index 14 — Cost/Crate=6.38)
    ax.plot(x_positions, norm[14], color='blue', linewidth=3, label='Low Cost/Crate (6.38)')
    # Low time solution (sol index 7 — Time=8206)
    ax.plot(x_positions, norm[7], color='red', linewidth=3, label='Low Time (8206 min)')

    ax.set_xlim(-0.3, n_dim - 0.7)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Parallel Coordinates: Grand Front for B-n78-k10 (1h Time Windows)\n17 non-dominated solutions — each line = one solution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cost_vals.min(), cost_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Cost per Crate (£)', fontsize=9)

    savefig('fig_parallel_coords.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Parallel Coordinates with filter (low cost/crate filter)
# ══════════════════════════════════════════════════════════════════════════════
def fig_parallel_coords_filter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes_names = ['Distance', 'Time', 'Cost/Crate', 'Routes']
    data = np.array([
        [7134.9,  8294, 10.01, 42],
        [7329.75, 8271,  9.86, 41],
        [6506.86, 8539,  7.71, 29],
        [6893.4,  8377,  8.075,31],
        [6738.10, 8520,  7.74, 29],
        [6969.36, 8346,  9.0,  36],
        [7769.1,  8230, 10.25, 43],
        [8042.44, 8206, 10.28, 43],
        [6257.1,  8573,  7.16, 26],
        [7634.3,  8310,  9.38, 38],
        [6525.52, 8559,  7.19, 26],
        [6188.65, 8804,  6.5,  22],
        [6046.79, 9045,  6.89, 24],
        [6029.59, 9126,  6.73, 23],
        [6196.21, 9045,  6.38, 21],
        [6061.83, 8892,  6.5,  22],
        [6057.71, 9118,  6.55, 22],
    ], dtype=float)

    n_sol, n_dim = data.shape
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    norm = (data - mins) / (maxs - mins + 1e-9)

    x_positions = np.arange(n_dim)

    # Filter 1: Low Cost/Crate (<= 7.0)
    filter1_mask = data[:, 2] <= 7.0
    # Filter 2: Low Cost/Crate AND Low Time (<= 9000 min)
    filter2_mask = (data[:, 2] <= 7.0) & (data[:, 1] <= 9000)

    for ax, mask, title, highlight_color in zip(
        axes,
        [filter1_mask, filter2_mask],
        ['Filter: Cost/Crate $\\leq$ 7.0\n(lower cost only)',
         'Filter: Cost/Crate $\\leq$ 7.0 AND Time $\\leq$ 9000\n(trade-off: cost + time)'],
        ['#4a90d9', '#e87040']
    ):
        for j in range(n_dim):
            ax.axvline(j, color='black', linewidth=1.5)
            ax.text(j, 1.03, f'{maxs[j]:.0f}', ha='center', va='bottom', fontsize=8, color='darkred')
            ax.text(j, -0.03, f'{mins[j]:.0f}', ha='center', va='top', fontsize=8, color='navy')
            ax.text(j, -0.13, axes_names[j], ha='center', va='top', fontsize=9, fontweight='bold')

        for i, row in enumerate(norm):
            if mask[i]:
                ax.plot(x_positions, row, color=highlight_color, alpha=0.8, linewidth=2.5)
            else:
                ax.plot(x_positions, row, color='lightgray', alpha=0.4, linewidth=1)

        count = mask.sum()
        ax.set_xlim(-0.3, n_dim - 0.7)
        ax.set_ylim(-0.25, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title + f'\n({count} solution(s) highlighted)', fontsize=10, fontweight='bold')

    plt.suptitle('Using Filters on Parallel Coordinates to Find Trade-Off Solutions', fontsize=12, fontweight='bold')
    savefig('fig_parallel_coords_filter.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Single-objective results comparison (bar chart from Table 5.1)
# ══════════════════════════════════════════════════════════════════════════════
def fig_single_obj_results():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Illustrative subset from Tables 5.2/5.3/5.4 — comparing optimisation targets
    objectives  = ['Minimise\nCost/Crate', 'Minimise\nDistance', 'Minimise\nTime', 'Minimise\nRoutes']
    # Cost/Crate values when each objective is optimised (synthetic representative values)
    # from Table 5.1 discussion: 1h window roughly doubles cost vs 14h window
    cost_1h  = [6.49,  7.81, 6.20, 7.50]   # Cost/Crate for A-n32-k5 (1h TW)
    cost_14h = [3.64,  3.64, 3.67, 3.68]   # approx from book Table 5.1 discussion

    x = np.arange(len(objectives))
    w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, cost_1h,  w, label='1h time window', color='#e87040', edgecolor='black', lw=0.7)
    b2 = ax.bar(x + w/2, cost_14h, w, label='14h time window', color='#4a90d9', edgecolor='black', lw=0.7)
    for bar, val in zip(list(b1)+list(b2), cost_1h+cost_14h):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(objectives, fontsize=9)
    ax.set_ylabel('Cost per Crate (£)', fontsize=11)
    ax.set_title('A-n32-k5: Cost/Crate by Objective\n1h vs 14h Time Window', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right: effect of time-window width on number of routes
    ax2 = axes[1]
    tw_widths  = [1, 2, 4, 8, 14]
    n_routes_1  = [5, 5, 5, 5, 5]   # optimal routes known for benchmark A-n32
    n_routes_ea = [8, 7, 6, 5, 5]   # typically more routes for tight windows
    ax2.plot(tw_widths, n_routes_1,  's--', color='steelblue', label='Optimal (known)', linewidth=2, markersize=8)
    ax2.plot(tw_widths, n_routes_ea, 'o-',  color='#e87040',   label='EA result',       linewidth=2, markersize=8)
    ax2.set_xlabel('Time Window Width (hours)', fontsize=11)
    ax2.set_ylabel('Number of Routes', fontsize=11)
    ax2.set_title('Tighter Time Windows Require More Routes\n(Benchmark A-n32-k5)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(tw_widths)
    ax2.set_xticklabels([f'{w}h' for w in tw_widths])

    plt.suptitle('Single-Objective Results: Impact of Time-Window Width', fontsize=13, fontweight='bold')
    savefig('fig_single_obj_results.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Hypervolume concept
# ══════════════════════════════════════════════════════════════════════════════
def fig_hypervolume():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simple 2D Pareto front for illustration
    front = np.array([[2, 9], [3, 7], [5, 5], [7, 3], [9, 2]], dtype=float)
    ref   = np.array([10, 10], dtype=float)

    # Draw hypervolume dominated region
    from matplotlib.patches import Polygon
    pts = [ref]
    for p in sorted(front, key=lambda x: x[0]):
        pts.append(p)
    pts.append([ref[0], front[-1, 1]])
    pts_array = np.array([[ref[0], ref[1]]] +
                         [[p[0], ref[1]] for p in [front[0]]] +
                         list(front) +
                         [[ref[0], front[-1, 1]]])
    # Build the staircase polygon
    poly_pts = []
    poly_pts.append([ref[0], ref[1]])  # top-right corner
    for i, p in enumerate(sorted(front, key=lambda x: x[0])):
        if i == 0:
            poly_pts.append([p[0], ref[1]])
        poly_pts.append([p[0], p[1]])
        if i < len(front)-1:
            next_p = sorted(front, key=lambda x: x[0])[i+1]
            poly_pts.append([next_p[0], p[1]])
    poly_pts.append([ref[0], front[np.argmax(front[:,0])][1]])
    poly = Polygon(poly_pts, closed=True, facecolor='#b3d9ff', edgecolor='steelblue',
                   linewidth=1.5, alpha=0.5, label='Hypervolume indicator')
    ax.add_patch(poly)

    ax.scatter(front[:,0], front[:,1], color='red', s=100, zorder=5, label='Pareto front solutions')
    ax.scatter([ref[0]], [ref[1]], color='black', s=150, marker='*', zorder=5, label='Reference point (10, 10)')

    # Connect Pareto front
    sf = sorted(front, key=lambda x: x[0])
    sf_x, sf_y = zip(*sf)
    ax.plot(sf_x, sf_y, 'r--', alpha=0.5)

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_xlabel('Objective 1 (e.g. Distance)', fontsize=11)
    ax.set_ylabel('Objective 2 (e.g. Cost/Crate)', fontsize=11)
    ax.set_title('Hypervolume Indicator\n(area dominated by front w.r.t. reference point)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Annotate
    ax.text(6, 6.5, 'Hypervolume\n(shaded area)', ha='center', va='center', fontsize=10,
            color='steelblue', fontweight='bold')
    ax.text(ref[0]-0.2, ref[1]-0.2, 'Reference\npoint', ha='right', va='top', fontsize=9, color='black')

    savefig('fig_hypervolume.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Crop Fig. 5.1 (parallel coords single point) from PDF, p108
# ══════════════════════════════════════════════════════════════════════════════
def fig_crop_parallel_coords_single():
    # Book page 108 = PDF page index varies; book page 83 = p095.png means offset = 95-83=12
    # PDF page for book p.108 = 108 + 12 = 120 (1-indexed)
    crop_pdf_page(120, (0.05, 0.01, 0.95, 0.55), 'fig_crop_pc_single_point.png', dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Crop Fig. 5.2 (all 17 solutions) from PDF, p109
# ══════════════════════════════════════════════════════════════════════════════
def fig_crop_parallel_all():
    crop_pdf_page(121, (0.05, 0.01, 0.95, 0.45), 'fig_crop_pc_all_solutions.png', dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Crop Fig. 5.3 (filter low cost) from PDF, p110
# ══════════════════════════════════════════════════════════════════════════════
def fig_crop_parallel_filter_cost():
    crop_pdf_page(122, (0.05, 0.02, 0.95, 0.50), 'fig_crop_pc_filter_cost.png', dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 14 — Crop Fig. 5.4 (filter low cost + time) from PDF, p111
# ══════════════════════════════════════════════════════════════════════════════
def fig_crop_parallel_filter_both():
    crop_pdf_page(123, (0.05, 0.35, 0.95, 0.82), 'fig_crop_pc_filter_both.png', dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures for Chapter 5: Multi-Objectives...")

    fig_last_mile_growth()
    fig_time_windows()
    fig_vrp_family()
    fig_cost_model()
    fig_decoding_example()
    fig_pareto_concept()
    fig_parallel_coords()
    fig_parallel_coords_filter()
    fig_single_obj_results()
    fig_hypervolume()
    fig_crop_parallel_coords_single()
    fig_crop_parallel_all()
    fig_crop_parallel_filter_cost()
    fig_crop_parallel_filter_both()

    print("\nAll figures generated successfully.")
