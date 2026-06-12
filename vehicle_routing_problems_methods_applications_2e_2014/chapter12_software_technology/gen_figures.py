"""
gen_figures.py  —  Generate all figures for Chapter 12 slides
(Software Tools and Emerging Technologies for VRP)
Uses matplotlib backend='Agg' (no display required).
Also crops key diagrams from the book PDF via PyMuPDF (fitz).
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── output directory ─────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

PDF_PATH = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
            "vehicle_routing_problems_methods_applications_2e_2014/"
            "Vehicle Routing_ Problems, Methods, and Applications,"
            " Second Edition 2014.pdf")


def savefig(name, fig=None, dpi=150):
    path = os.path.join(OUT, name)
    (fig or plt).savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — VRP Software Architecture (from book p.356 / Figure 12.1)
# ─────────────────────────────────────────────────────────────────────────────
def fig_software_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    def box(x, y, w, h, label, color='#4472C4', fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white', linewidth=1.5,
                              zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                color='white', fontweight='bold', zorder=4, wrap=True,
                multialignment='center')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
                    zorder=5)

    # Input blocks
    box(1.2, 6.0, 2.0, 0.7, 'Digital Map', '#2E75B6')
    box(3.5, 6.0, 2.0, 0.7, 'Problem data', '#2E75B6')
    box(5.8, 6.0, 1.6, 0.7, 'Resource data', '#2E75B6')
    box(7.5, 6.0, 1.6, 0.7, 'Parameters', '#2E75B6')

    # Middle layer
    box(2.2, 4.5, 2.5, 0.8, 'GIS Module', '#70AD47')
    box(5.0, 4.5, 2.5, 0.8, 'Problem data\nModule', '#70AD47')
    box(7.8, 4.5, 2.2, 0.8, 'Optimization\nModule', '#C55A11')

    # Central engine
    box(5.0, 2.8, 5.5, 1.0, 'Routing / Scheduling Engine', '#C55A11', fontsize=10)

    # Output
    box(5.0, 1.2, 3.0, 0.8, 'Output\n(Routes, KPIs, Maps)', '#7030A0')

    # Arrows — inputs to modules
    arrow(1.2, 5.65, 2.2, 4.9)
    arrow(3.5, 5.65, 5.0, 4.9)
    arrow(5.8, 5.65, 5.0, 4.9)
    arrow(7.5, 5.65, 7.8, 4.9)

    # Modules to engine
    arrow(2.2, 4.1, 3.5, 3.3)
    arrow(5.0, 4.1, 5.0, 3.3)
    arrow(7.8, 4.1, 6.5, 3.3)

    # Engine to output
    arrow(5.0, 2.3, 5.0, 1.6)

    ax.set_title("Figure 12.1 — Architecture of VRP Routing Software",
                 fontsize=11, fontweight='bold', pad=10)
    savefig("fig_software_architecture.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — VRP Model Properties (radar / feature chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_properties():
    categories = [
        'Capacity\nConstraints', 'Time\nWindows', 'Multi-depot',
        'Heterogeneous\nFleet', 'Pickup &\nDelivery',
        'Truck &\nTrailer', 'Stochastic\nDemand', 'Split\nDeliveries'
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Hypothetical coverage scores (0–5) for commercial vs open-source
    commercial = [5, 5, 4, 4, 4, 3, 3, 3]
    open_src    = [5, 4, 3, 2, 3, 1, 2, 2]
    commercial += commercial[:1]
    open_src    += open_src[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    ax.plot(angles, commercial, 'o-', color='#4472C4', lw=2, label='Commercial tools')
    ax.fill(angles, commercial, alpha=0.15, color='#4472C4')
    ax.plot(angles, open_src, 's--', color='#ED7D31', lw=2, label='Open-source tools')
    ax.fill(angles, open_src, alpha=0.15, color='#ED7D31')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=7)
    ax.set_title("Supported VRP Model Properties\n(schematic comparison)", size=11,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    savefig("fig_model_properties.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Algorithm landscape in commercial VRP software
# ─────────────────────────────────────────────────────────────────────────────
def fig_algorithm_landscape():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    categories = {
        'Construction\nHeuristics': ['Clarke-Wright\nSavings', 'Nearest\nNeighbour',
                                     'Sweep\nAlgorithm', 'Christofides'],
        'Local Search /\nImprovement': ['2-opt', '3-opt', 'Or-opt', 'Lin-Kernighan'],
        'Metaheuristics': ['Tabu Search', 'Simulated\nAnnealing', 'Genetic\nAlgorithm',
                           'ALNS'],
        'Exact Methods': ['Branch &\nBound', 'Branch &\nCut', 'Column\nGeneration',
                          'Dynamic\nProgramming'],
    }

    col_colors = ['#2E75B6', '#70AD47', '#C55A11', '#7030A0']
    x_starts = [0.02, 0.27, 0.52, 0.77]
    col_w = 0.23

    for idx, (cat, methods) in enumerate(categories.items()):
        x0 = x_starts[idx]
        # header
        rect = FancyBboxPatch((x0, 0.80), col_w, 0.17,
                              boxstyle="round,pad=0.01",
                              facecolor=col_colors[idx], edgecolor='white',
                              transform=ax.transAxes, zorder=3, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0 + col_w/2, 0.885, cat, ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='white',
                fontweight='bold', zorder=4, multialignment='center')
        # items
        for jdx, m in enumerate(methods):
            y = 0.62 - jdx * 0.19
            rect2 = FancyBboxPatch((x0 + 0.01, y), col_w - 0.02, 0.16,
                                   boxstyle="round,pad=0.01",
                                   facecolor=col_colors[idx] + '33',
                                   edgecolor=col_colors[idx], linewidth=1,
                                   transform=ax.transAxes, zorder=3, clip_on=False)
            ax.add_patch(rect2)
            ax.text(x0 + col_w/2, y + 0.08, m, ha='center', va='center',
                    transform=ax.transAxes, fontsize=8,
                    color='#222222', multialignment='center')

    ax.set_title("Algorithm Landscape in Commercial VRP Software",
                 fontsize=12, fontweight='bold', y=1.02)
    savefig("fig_algorithm_landscape.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Time-window VRP illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_time_window():
    np.random.seed(42)
    n = 8  # customers
    depot = np.array([5.0, 5.0])
    pts = np.random.uniform(1, 9, (n, 2))

    # Two routes (simple split)
    r1 = [0, 1, 2, 3, 0]   # indices into pts (0 = depot for route)
    r2 = [0, 4, 5, 6, 7, 0]

    all_pts = np.vstack([depot, pts])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Left: route map
    ax = axes[0]
    ax.set_facecolor('#f0f4f8')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    colors = ['#4472C4', '#ED7D31']
    routes = [r1, r2]
    labels = ['Route 1', 'Route 2']
    for ridx, route in enumerate(routes):
        coords = all_pts[route]
        ax.plot(coords[:, 0], coords[:, 1], '-o',
                color=colors[ridx], lw=2, ms=8, label=labels[ridx])
        for j, ci in enumerate(route[:-1]):
            ax.annotate('', xy=all_pts[route[j+1]], xytext=all_pts[route[j]],
                        arrowprops=dict(arrowstyle='->', color=colors[ridx], lw=1.5))

    # depot
    ax.plot(*depot, 's', color='black', ms=12, zorder=5)
    ax.text(depot[0] + 0.2, depot[1] + 0.2, 'Depot', fontsize=9, fontweight='bold')

    for i, p in enumerate(pts):
        ax.text(p[0] + 0.15, p[1] + 0.15, f'C{i+1}', fontsize=8)

    ax.legend(fontsize=9)
    ax.set_title('VRP Routes with Time Windows', fontsize=10, fontweight='bold')
    ax.set_xlabel('x-coordinate'); ax.set_ylabel('y-coordinate')

    # Right: time-window bar chart
    ax2 = axes[1]
    ax2.set_facecolor('#f0f4f8')
    tw_open  = [7, 8, 9, 10, 7,  8,  9, 11]
    tw_close = [10, 12, 13, 14, 11, 13, 14, 15]
    arr_time = [7.5, 9, 10, 11.5, 8, 10.5, 11, 12]

    y_pos = np.arange(n)
    ax2.barh(y_pos, [tw_close[i] - tw_open[i] for i in range(n)],
             left=tw_open, color='#AED6F1', edgecolor='#2E75B6', height=0.6,
             label='Allowed window')
    ax2.plot(arr_time, y_pos, 'D', color='#C0392B', ms=7, label='Arrival time', zorder=5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'C{i+1}' for i in range(n)])
    ax2.set_xlabel('Time (hours from midnight)')
    ax2.set_title('Customer Time Windows', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.axvline(7, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvline(16, color='gray', lw=1, ls='--', alpha=0.5)

    plt.tight_layout()
    savefig("fig_time_window.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — VRP Technology Survey: respondents table (visual)
# ─────────────────────────────────────────────────────────────────────────────
def fig_vendor_table():
    companies = [
        ('GIRO Inc', 'Montreal, Canada', 'GeoRoute, GIRO/ACCES'),
        ('GTS Systems & Consulting', 'Herzogenrath, Germany', 'TransIT'),
        ('MJC2 Limited', 'Berkshire, UK', 'DISC, REACT'),
        ('Optrak Dist. Software', 'Hertford, UK', 'Optrak'),
        ('ORTEC', 'Zoetermeer, Netherlands\n& Atlanta, USA', 'ORTEC'),
        ('Oprit srl', 'Inola, Italy', 'EasyRoute, OptiRoute'),
        ('Procomp Solutions Oy', 'Oulu, Finland', 'R2'),
        ('PTV AG', 'Karlsruhe, Germany', 'SmarTour'),
        ('Spider Solutions AS', 'Oslo, Norway', 'Spider 5'),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')

    cols = ['Company', 'Location', 'Product(s)']
    data = [[c[0], c[1], c[2]] for c in companies]

    tbl = ax.table(cellText=data, colLabels=cols,
                   cellLoc='left', loc='center',
                   colWidths=[0.32, 0.38, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header
    for j in range(3):
        tbl[0, j].set_facecolor('#2E75B6')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(companies)+1):
        bg = '#EBF5FB' if i % 2 == 0 else 'white'
        for j in range(3):
            tbl[i, j].set_facecolor(bg)

    ax.set_title('Table 12.1 — Responding VRP Technology Providers (Survey)',
                 fontsize=11, fontweight='bold', pad=20)
    savefig("fig_vendor_table.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Emerging technologies for VRP
# ─────────────────────────────────────────────────────────────────────────────
def fig_emerging_technologies():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    # Central node
    cx, cy = 5, 3.5
    ax.text(cx, cy, 'VRP\nSoftware', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='circle,pad=0.6', facecolor='#C55A11',
                      edgecolor='white', lw=2))

    techs = [
        ('GPS &\nTelematics', 1.5, 6.0, '#4472C4'),
        ('Real-time\nTraffic Data', 4.0, 6.2, '#4472C4'),
        ('Mobile\nDevices', 7.0, 6.0, '#4472C4'),
        ('Cloud\nComputing', 8.8, 4.5, '#70AD47'),
        ('Machine\nLearning', 8.5, 2.0, '#70AD47'),
        ('Internet of\nThings (IoT)', 6.5, 0.6, '#7030A0'),
        ('Parallel /\nGPU Computing', 3.5, 0.5, '#7030A0'),
        ('SaaS / Web\nRouting APIs', 1.2, 2.0, '#ED7D31'),
    ]

    for label, tx, ty, col in techs:
        ax.annotate('', xy=(cx, cy), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5,
                                   connectionstyle='arc3,rad=0.1'))
        ax.text(tx, ty, label, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=col,
                          edgecolor='white', lw=1.5))

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.5)
    ax.set_title('Emerging Technologies Influencing VRP Software', fontsize=12,
                 fontweight='bold', pad=12)
    savefig("fig_emerging_technologies.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Clarke-Wright savings illustration (5 customers)
# ─────────────────────────────────────────────────────────────────────────────
def fig_clarke_wright():
    depot = np.array([0.0, 0.0])
    customers = {
        1: np.array([2.0, 3.0]),
        2: np.array([4.0, 1.0]),
        3: np.array([5.0, 4.0]),
        4: np.array([1.0, 5.0]),
        5: np.array([6.0, 2.0]),
    }

    def dist(a, b):
        return np.linalg.norm(a - b)

    # Savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = {}
    for i in customers:
        for j in customers:
            if j > i:
                s = dist(depot, customers[i]) + dist(depot, customers[j]) - dist(customers[i], customers[j])
                savings[(i, j)] = round(s, 2)

    sorted_sav = sorted(savings.items(), key=lambda x: -x[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Left — "star" routes (before merging)
    ax = axes[0]
    ax.set_facecolor('#f0f4f8')
    ax.set_xlim(-1, 8); ax.set_ylim(-1, 7)
    ax.plot(*depot, 's', color='black', ms=14, zorder=5)
    ax.text(depot[0] + 0.1, depot[1] + 0.2, 'Depot', fontsize=9, fontweight='bold')
    cols = ['#4472C4', '#ED7D31', '#70AD47', '#C55A11', '#7030A0']
    for idx, (ci, pt) in enumerate(customers.items()):
        ax.plot([depot[0], pt[0]], [depot[1], pt[1]], '-',
                color=cols[idx], lw=2, alpha=0.7)
        ax.plot(*pt, 'o', color=cols[idx], ms=10, zorder=5)
        ax.text(pt[0] + 0.15, pt[1] + 0.15, f'C{ci}', fontsize=9, fontweight='bold')
    ax.set_title('Step 1: Individual (star) routes\nBefore savings are applied', fontsize=10)

    # Right — merged routes after top savings
    ax2 = axes[1]
    ax2.set_facecolor('#f0f4f8')
    ax2.set_xlim(-1, 8); ax2.set_ylim(-1, 7)
    ax2.plot(*depot, 's', color='black', ms=14, zorder=5)
    ax2.text(depot[0] + 0.1, depot[1] + 0.2, 'Depot', fontsize=9, fontweight='bold')

    # Show two merged routes
    route1 = [depot, customers[3], customers[1], customers[4], depot]
    route2 = [depot, customers[2], customers[5], depot]
    for route, col, lbl in [(route1, '#4472C4', 'Route 1'), (route2, '#ED7D31', 'Route 2')]:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax2.plot(xs, ys, '-o', color=col, lw=2.5, ms=8, label=lbl, zorder=3)

    for ci, pt in customers.items():
        ax2.text(pt[0] + 0.15, pt[1] + 0.15, f'C{ci}', fontsize=9, fontweight='bold')

    ax2.legend(fontsize=9)
    ax2.set_title('Step 2: Merged routes\nAfter applying top savings', fontsize=10)

    # Savings table inset
    ax2.text(7.0, 6.5, 'Top savings:', fontsize=8, fontweight='bold', ha='center')
    for k, ((i, j), s) in enumerate(sorted_sav[:5]):
        ax2.text(7.0, 6.0 - k*0.5,
                 f's({i},{j}) = {s:.2f}', fontsize=7.5, ha='center',
                 color='#333333')

    plt.tight_layout()
    savefig("fig_clarke_wright.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Performance vs. Problem Size
# ─────────────────────────────────────────────────────────────────────────────
def fig_performance_vs_size():
    sizes = np.array([50, 100, 200, 500, 1000, 2000, 5000])

    # Schematic time curves (seconds)
    exact  = np.array([0.1, 2, 60, np.nan, np.nan, np.nan, np.nan])
    meta   = np.array([0.5, 1, 3,   12,    50,    200,    1200])
    heur   = np.array([0.05, 0.1, 0.3, 1, 4, 15, 80])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor('#f0f4f8')
    fig.patch.set_facecolor('#f8f9fa')

    ax.semilogy(sizes, exact, 'D-', color='#C55A11', lw=2, ms=7,
                label='Exact (B&C/Column Gen.)', markerfacecolor='white')
    ax.semilogy(sizes, meta, 's-', color='#4472C4', lw=2, ms=7,
                label='Metaheuristic (Tabu, ALNS)')
    ax.semilogy(sizes, heur, 'o-', color='#70AD47', lw=2, ms=7,
                label='Construction heuristic')

    ax.set_xlabel('Number of customers', fontsize=11)
    ax.set_ylabel('Computation time (seconds, log scale)', fontsize=11)
    ax.set_title('Schematic Performance vs. Problem Size\nfor Different Algorithm Classes',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.set_xlim(0, 5500)

    savefig("fig_performance_vs_size.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Real-time replanning loop
# ─────────────────────────────────────────────────────────────────────────────
def fig_realtime_loop():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    steps = [
        ('Current\nRoute Plan', 1.2, 3.0, '#2E75B6'),
        ('New Event\n(order/traffic)', 3.2, 3.0, '#C55A11'),
        ('Re-optimise\n(ALNS / TS)', 5.2, 3.0, '#4472C4'),
        ('Updated\nPlan', 7.2, 3.0, '#70AD47'),
        ('Driver\nNotification\n(GPS/mobile)', 9.0, 3.0, '#7030A0'),
    ]

    for label, x, y, col in steps:
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=col,
                          edgecolor='white', lw=2))

    # Arrows between steps
    arrow_xs = [(2.05, 2.7), (4.05, 4.7), (6.05, 6.7), (7.95, 8.45)]
    for x1, x2 in arrow_xs:
        ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

    # Feedback arrow
    ax.annotate('', xy=(1.2, 2.4), xytext=(9.0, 2.4),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5,
                                connectionstyle='arc3,rad=0'))
    ax.text(5.1, 2.1, 'Feedback loop (continuous monitoring)',
            ha='center', va='center', fontsize=8.5, color='#555555', style='italic')

    ax.set_xlim(0, 10.5)
    ax.set_ylim(1.5, 4.5)
    ax.set_title('Real-Time VRP Replanning Loop', fontsize=12, fontweight='bold', pad=12)
    savefig("fig_realtime_loop.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Cloud SaaS vs On-premise architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_saas_vs_onpremise():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f8f9fa')

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor('#f8f9fa')

    def draw_stack(ax, layers, title, base_col):
        ax.set_xlim(0, 10); ax.set_ylim(0, len(layers) + 1)
        for i, (label, col) in enumerate(layers):
            y = i * 1.1 + 0.2
            rect = FancyBboxPatch((0.5, y), 9, 0.9,
                                  boxstyle="round,pad=0.05",
                                  facecolor=col, edgecolor='white', lw=1.5)
            ax.add_patch(rect)
            ax.text(5, y + 0.45, label, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    onprem_layers = [
        ('Hardware (servers, storage)', '#555555'),
        ('OS & Middleware', '#777777'),
        ('VRP Application Software', '#2E75B6'),
        ('Database & GIS', '#2E75B6'),
        ('User Interface (desktop)', '#4472C4'),
    ]
    saas_layers = [
        ('Cloud Infrastructure (AWS/Azure/GCP)', '#555555'),
        ('SaaS VRP Platform', '#C55A11'),
        ('REST API / Web Service', '#ED7D31'),
        ('Browser / Mobile App', '#F4A460'),
        ('Customer Data (uploaded)', '#70AD47'),
    ]

    draw_stack(axes[0], onprem_layers, 'On-Premise VRP Software', '#2E75B6')
    draw_stack(axes[1], saas_layers, 'Cloud / SaaS VRP Platform', '#C55A11')

    plt.tight_layout()
    savefig("fig_saas_vs_onpremise.pdf", fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Crop Figure 12.1 from book PDF (software module diagram)
# ─────────────────────────────────────────────────────────────────────────────
def fig_crop_book_fig12_1():
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(PDF_PATH)
        # Page 356 (0-indexed = 355) contains Figure 12.1
        page = doc[355]
        # Crop lower half where the figure typically sits
        rect = fitz.Rect(70, 400, 520, 650)
        clip = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        out = os.path.join(OUT, "fig_book_fig12_1.png")
        clip.save(out)
        print(f"  saved {out}")
        doc.close()
    except Exception as e:
        print(f"  [warn] Could not crop PDF figure: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 12 ...")
    fig_software_architecture()
    fig_model_properties()
    fig_algorithm_landscape()
    fig_time_window()
    fig_vendor_table()
    fig_emerging_technologies()
    fig_clarke_wright()
    fig_performance_vs_size()
    fig_realtime_loop()
    fig_saas_vs_onpremise()
    fig_crop_book_fig12_1()
    print("Done.")
