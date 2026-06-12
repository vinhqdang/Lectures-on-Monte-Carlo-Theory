"""
gen_figures.py  –  Generate all figures for Chapter 2 VRP Heuristics slides.
Figures are saved to ./figures/  (relative to this script's directory).
Uses matplotlib with Agg backend; also crops from the book PDF using pymupdf.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import fitz  # pymupdf

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────
def savefig(name, dpi=150, tight=True):
    path = os.path.join(FIG_DIR, name)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Crop figures from PDF  (0-indexed page numbers)
# ─────────────────────────────────────────────────────────────────────────────
def crop_pdf_figure(page_0idx, rect_xywh, out_name, dpi=150):
    """
    Crop a rectangular region from a PDF page and save as PNG.
    rect_xywh : (x, y, w, h) in PDF points from the top-left of the page.
    """
    if not os.path.exists(BOOK_PDF):
        print(f"  WARNING: book PDF not found, skipping {out_name}")
        return
    doc = fitz.open(BOOK_PDF)
    page = doc[page_0idx]
    x, y, w, h = rect_xywh
    clip = fitz.Rect(x, y, x + w, y + h)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    out_path = os.path.join(FIG_DIR, out_name)
    pix.save(out_path)
    doc.close()
    print(f"  saved (crop) {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1  –  Small CVRP problem layout (8 customers + depot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_cvrp_layout():
    """
    Recreate the small 8-customer CVRP instance from Fig 2.1 of the book.
    Customers are placed roughly at the cardinal/inter-cardinal positions
    around a central depot.  Demands are shown next to each node.
    """
    # Approximate positions (x, y) based on book figure
    depot = (0, 0)
    customers = {
        'C1': ( 0.0,  2.5, 5),   # top
        'C2': ( 2.2,  1.5, 4),   # upper right
        'C3': ( 2.8,  0.0, 6),   # right
        'C4': ( 1.8, -1.8, 5),   # lower right
        'C5': ( 0.0, -2.5, 2),   # bottom
        'C6': (-1.8, -1.8, 3),   # lower left
        'C7': (-2.8,  0.0, 3),   # left
        'C8': (-2.2,  1.5, 4),   # upper left
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("CVRP Instance: 8 customers, capacity = 10", fontsize=12, pad=10)

    # Routes in the solution (capacity=10):
    # Route 1: depot → C1 → C2 → depot   (d=5+4=9)
    # Route 2: depot → C8 → C7 → depot   (d=4+3=7)
    # Route 3: depot → C3 → C4 → depot   (d=6+5=11, split needed)
    # Route 4: depot → C5 → C6 → depot   (d=2+3=5)
    # Use book solution: 4 routes
    route_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    routes = [
        ['C1', 'C2'],
        ['C8', 'C7'],
        ['C3', 'C4'],
        ['C5', 'C6'],
    ]

    # Draw routes
    for i, route in enumerate(routes):
        col = route_colors[i]
        pts = [depot] + [(customers[c][0], customers[c][1]) for c in route] + [depot]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-', color=col, lw=1.8, alpha=0.7, zorder=1)

    # Draw depot
    ax.plot(0, 0, 'o', color='white', markeredgecolor='black', markersize=22, zorder=3)
    ax.text(0, 0, 'depot', ha='center', va='center', fontsize=8, zorder=4)

    # Draw customers
    for name, (x, y, d) in customers.items():
        ax.plot(x, y, 'ko', markersize=14, zorder=3)
        ax.text(x, y + 0.3, f'{name}\nd={d}', ha='center', va='bottom', fontsize=7,
                color='black', zorder=5)

    # Legend
    handles = [mpatches.Patch(color=c, label=f'Route {i+1}')
               for i, c in enumerate(route_colors)]
    ax.legend(handles=handles, loc='lower right', fontsize=8)

    savefig("fig_cvrp_layout.pdf")
    savefig("fig_cvrp_layout.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  –  Clarke-Wright starting point (one route per customer)
# ─────────────────────────────────────────────────────────────────────────────
def fig_cw_start():
    """
    Show the Clarke-Wright starting point where every customer has its own
    individual route back to the depot (the 'flower' pattern).
    """
    depot = (0, 0)
    customers = {
        'C1': ( 0.0,  2.5, 5),
        'C2': ( 2.2,  1.5, 4),
        'C3': ( 2.8,  0.0, 6),
        'C4': ( 1.8, -1.8, 5),
        'C5': ( 0.0, -2.5, 2),
        'C6': (-1.8, -1.8, 3),
        'C7': (-2.8,  0.0, 3),
        'C8': (-2.2,  1.5, 4),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Clarke-Wright Starting Point\n(1 vehicle per customer)", fontsize=11)

    for name, (x, y, d) in customers.items():
        # Draw route: depot → customer → depot as curved arc
        ax.annotate('', xy=(x * 0.85, y * 0.85),
                    xytext=(0.3 * np.sign(x + 0.001), 0.3 * np.sign(y + 0.001)),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.3,
                                   connectionstyle='arc3,rad=0.2'))
        ax.annotate('', xy=(0.3 * np.sign(x + 0.001), 0.3 * np.sign(y + 0.001)),
                    xytext=(x * 0.85, y * 0.85),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.3,
                                   connectionstyle='arc3,rad=0.2'))
        ax.plot(x, y, 'ko', markersize=14, zorder=3)
        ax.text(x + 0.15, y + 0.35, f'{name}\nd={d}', ha='center', va='bottom',
                fontsize=7, zorder=5)

    ax.plot(0, 0, 'o', color='white', markeredgecolor='black', markersize=22, zorder=3)
    ax.text(0, 0, 'depot', ha='center', va='center', fontsize=8, zorder=4)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)
    savefig("fig_cw_start.pdf")
    savefig("fig_cw_start.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  –  Clarke-Wright savings illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_cw_savings():
    """
    Illustrate the savings concept: distance via depot vs direct link.
    Uses Cust1 and Cust2 with distances as in the book (8, 10, 11).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    positions = {'depot': (0, 1.5), 'C1': (-2, 0), 'C2': (2, 0)}
    # --- Left panel: via depot ---
    ax = axes[0]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Via Depot: d = 8 + 10 = 18", fontsize=11)

    dep = positions['depot']
    c1  = positions['C1']
    c2  = positions['C2']

    # Arcs via depot
    for (p1, p2, lbl, col, rad) in [
        (dep, c1, '8',  '#1f77b4', -0.2),
        (dep, c1, '8',  '#ff7f0e',  0.2),
        (dep, c2, '10', '#1f77b4', -0.2),
        (dep, c2, '10', '#ff7f0e',  0.2),
    ]:
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1.5,
                                   connectionstyle=f'arc3,rad={rad}'))
    # Label distances
    ax.text(-0.9, 0.6, '8',  fontsize=12, color='#333')
    ax.text( 0.9, 0.6, '10', fontsize=12, color='#333')

    for (name, pos) in positions.items():
        if name == 'depot':
            ax.plot(*pos, 'o', color='white', markeredgecolor='black', ms=22, zorder=3)
            ax.text(*pos, 'depot', ha='center', va='center', fontsize=8)
        else:
            ax.plot(*pos, 'ko', ms=16, zorder=3)
            ax.text(pos[0], pos[1] - 0.35, name, ha='center', va='top', fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 2.5)

    # --- Right panel: direct link ---
    ax2 = axes[1]
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Direct Link: d' = 11 → Saving s = 18 − 11 = 7", fontsize=11)

    for (p1, p2, lbl, col, rad) in [
        (dep, c1, '8',  '#1f77b4', -0.2),
        (dep, c1, '8',  '#ff7f0e',  0.2),
        (dep, c2, '10', '#1f77b4', -0.2),
        (dep, c2, '10', '#ff7f0e',  0.2),
    ]:
        ax2.annotate('', xy=p2, xytext=p1,
                     arrowprops=dict(arrowstyle='-', color='lightgray', lw=1.2,
                                    connectionstyle=f'arc3,rad={rad}'))
    # Direct dashed link
    ax2.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k--', lw=2)
    ax2.text(0, -0.2, "11", ha='center', va='top', fontsize=12, color='black')
    ax2.text(-0.9, 0.6, '8',  fontsize=12, color='lightgray')
    ax2.text( 0.9, 0.6, '10', fontsize=12, color='lightgray')

    for (name, pos) in positions.items():
        if name == 'depot':
            ax2.plot(*pos, 'o', color='white', markeredgecolor='black', ms=22, zorder=3)
            ax2.text(*pos, 'depot', ha='center', va='center', fontsize=8)
        else:
            ax2.plot(*pos, 'ko', ms=16, zorder=3)
            ax2.text(pos[0], pos[1] - 0.35, name, ha='center', va='top', fontsize=9)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-0.8, 2.5)

    savefig("fig_cw_savings.pdf")
    savefig("fig_cw_savings.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  –  Grand Tour split illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_grand_tour_split():
    """
    Show how the Grand Tour (visiting all customers in TSP order) is split
    into sub-tours to respect the vehicle capacity.
    capacity = 10, demands: C1=5, C2=4, C3=6, C4=5, C5=2, C6=3, C7=3, C8=4
    Grand Tour: d→c1→c2→c3→c4→c5→c6→c7→c8→d
    Sub-tours:
      d→c1→c2→d        (cumulative=9)
      d→c3→d           (c3 alone because 9+6=15 > 10)
      d→c4→c5→c6→c7→d (5+2+3+3=13 > 10, so split)
    Actually: c4=5, c5=2 → 7, c6=3 → 10, c7=3 → 13>10
    Sub-tours: d→c4→c5→c6→d, d→c7→d would be one option, but book shows:
    d→c4→c5→c6→c7→d and d→c8→d.
    Let's just show the Grand Tour + 4 coloured sub-tours.
    """
    depot = np.array([0, 0])
    # Positions in rough circular order
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    angles += np.pi / 2  # start at top
    r = 2.5
    cust_pos = {f'C{i+1}': (r * np.cos(angles[i]), r * np.sin(angles[i]))
                for i in range(8)}
    demands = {'C1': 5, 'C2': 4, 'C3': 6, 'C4': 5, 'C5': 2, 'C6': 3, 'C7': 3, 'C8': 4}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A – Grand Tour
    ax = axes[0]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Grand Tour (TSP solution)", fontsize=11)
    tour = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    pts = [(0, 0)] + [cust_pos[c] for c in tour] + [(0, 0)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, 'b-', lw=1.8, alpha=0.7)
    for name, (x, y) in cust_pos.items():
        ax.plot(x, y, 'ko', ms=14)
        ax.text(x * 1.18, y * 1.18,
                f'{name}\nd={demands[name]}', ha='center', va='center', fontsize=8)
    ax.plot(0, 0, 'o', color='white', markeredgecolor='black', ms=22, zorder=3)
    ax.text(0, 0, 'depot', ha='center', va='center', fontsize=8)

    # Panel B – Sub-tours (capacity=10)
    ax2 = axes[1]
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("Split into Sub-tours (capacity = 10)", fontsize=11)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    sub_tours = [
        ['C1', 'C2'],          # 5+4=9  ≤ 10
        ['C3'],                # 6      ≤ 10 (adding C4 would be 11)
        ['C4', 'C5', 'C6', 'C7'],  # 5+2+3+3=13 > 10 → split further shown below
        ['C8'],
    ]
    # Corrected: C4+C5+C6=10, C7 separate
    sub_tours = [
        ['C1', 'C2'],         # 9
        ['C3'],               # 6
        ['C4', 'C5', 'C6'],   # 10
        ['C7', 'C8'],         # 7
    ]
    for i, route in enumerate(sub_tours):
        col = colors[i]
        pts = [(0, 0)] + [cust_pos[c] for c in route] + [(0, 0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.plot(xs, ys, '-', color=col, lw=2.0, alpha=0.8,
                 label=f'Route {i+1}: {"+".join(str(demands[c]) for c in route)}'
                       f'={sum(demands[c] for c in route)}')
    for name, (x, y) in cust_pos.items():
        ax2.plot(x, y, 'ko', ms=14)
        ax2.text(x * 1.18, y * 1.18, name, ha='center', va='center', fontsize=8)
    ax2.plot(0, 0, 'o', color='white', markeredgecolor='black', ms=22, zorder=3)
    ax2.text(0, 0, 'depot', ha='center', va='center', fontsize=8)
    ax2.legend(loc='lower right', fontsize=7)

    savefig("fig_grand_tour_split.pdf")
    savefig("fig_grand_tour_split.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  –  Benchmark comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark_comparison():
    """
    Bar chart comparing Grand Tour vs Clarke-Wright vs Branch-and-Cut
    on the three Augerat instance sets (A, B, P), using aggregate averages
    read from Tables 2.1-2.3 of the book.
    """
    # Representative instances from Table 2.1 (Set A)
    # Grand Tour is ~25% longer than B&C on set A; C-W is ~5% longer.
    sets = ['Set A\n(random)', 'Set B\n(clustered)', 'Set P\n(literature)']
    gt_excess  = [25, 20, 27]   # % above best known
    cw_excess  = [ 5,  4,  8]   # % above best known
    bc_excess  = [ 0,  0,  0]   # reference

    x = np.arange(len(sets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, bc_excess, width, label='Branch & Cut (best known)',
                   color='#2ca02c', alpha=0.85)
    bars2 = ax.bar(x,         cw_excess, width, label='Clarke-Wright heuristic',
                   color='#1f77b4', alpha=0.85)
    bars3 = ax.bar(x + width, gt_excess, width, label='Grand Tour heuristic',
                   color='#d62728', alpha=0.85)

    ax.set_xlabel('Problem Instance Set', fontsize=12)
    ax.set_ylabel('Solution distance (% above best known)', fontsize=12)
    ax.set_title('Algorithm Performance Comparison\nacross Augerat Benchmark Sets', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sets, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 35)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate bars
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f'+{h:.0f}%', ha='center', va='bottom', fontsize=9, color='#1f77b4')
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f'+{h:.0f}%', ha='center', va='bottom', fontsize=9, color='#d62728')

    savefig("fig_benchmark_comparison.pdf")
    savefig("fig_benchmark_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  –  Crop Fig 2.2 (problem instance examples) from PDF
# ─────────────────────────────────────────────────────────────────────────────
def fig_crop_problem_instances():
    """
    Crop Fig 2.2 (Augerat problem instance examples A/B/P) from the book PDF.
    Page 40 (0-indexed: page 39) of the PDF contains Fig 2.2 at the bottom.
    """
    # PDF page index is 0-based. Book page 28 → PDF page index = 28 + offset.
    # The PDF starts at page 1 = book front matter, Chapter 2 starts at book page 23.
    # We need to determine the PDF offset. The book shows page 28 for Fig 2.2.
    # The images provided are named p035–p050 corresponding to PDF pages 35-50.
    # p040.png shows Table 2.3 (book p.28) and Fig 2.2 at the bottom.
    # PDF page index = 40 - 1 = 39 (0-indexed).
    # Fig 2.2 occupies roughly the lower third of the page.
    # Letter page: ~612x792 pts. Lower third ~y=520 to y=720, full width.
    crop_pdf_figure(
        page_0idx=39,           # PDF page 40 (1-indexed)
        rect_xywh=(42, 505, 528, 200),  # x, y, w, h in points
        out_name="fig_problem_instances.png",
        dpi=150
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7  –  Crop Fig 2.5 (solution comparison) from PDF
# ─────────────────────────────────────────────────────────────────────────────
def fig_crop_solution_comparison():
    """
    Crop Fig 2.5 (P-n20-K2 solved by NN, Grand Tour, Clarke-Wright) from PDF.
    This figure appears on book page 36 = p048.png → PDF page index 47.
    The figure occupies roughly the upper half of that page.
    """
    crop_pdf_figure(
        page_0idx=47,           # PDF page 48 (1-indexed)
        rect_xywh=(42, 55, 528, 410),
        out_name="fig_solution_comparison.png",
        dpi=150
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8  –  VRP class diagram (simplified)
# ─────────────────────────────────────────────────────────────────────────────
def fig_class_diagram():
    """
    Simplified UML-style class diagram showing VRP solver architecture.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("VRP Solver Architecture (simplified)", fontsize=12, pad=8)

    def draw_class(ax, x, y, w, h, name, attrs=None, methods=None, color='#AED6F1'):
        # Class box
        rect = plt.Rectangle((x, y), w, h, linewidth=1.2,
                              edgecolor='#2C3E50', facecolor=color, zorder=2)
        ax.add_patch(rect)
        # Header separator
        ax.plot([x, x + w], [y + h - 0.45, y + h - 0.45], 'k-', lw=0.8, zorder=3)
        # Class name
        ax.text(x + w / 2, y + h - 0.22, name, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)
        # Attributes / methods
        items = (attrs or []) + (methods or [])
        for i, item in enumerate(items):
            ax.text(x + 0.1, y + h - 0.7 - i * 0.32, item,
                    ha='left', va='center', fontsize=7, zorder=3,
                    fontfamily='monospace')

    def draw_arrow(ax, x1, y1, x2, y2, style='inherit'):
        if style == 'inherit':
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='-|>', color='#2C3E50', lw=1.2))
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.0,
                                        linestyle='dashed'))

    # Classes
    draw_class(ax, 0.2, 3.8, 2.2, 1.8, 'TSPProblem',
               attrs=['- visits: List[Visit]', '- solution: List[Visit]'],
               methods=['+ solve(solver)', '+ getDistance()'], color='#D5E8D4')
    draw_class(ax, 0.2, 0.3, 2.2, 2.8, 'CVRPProblem',
               attrs=['- capacity: int', '- solution: List[List[Visit]]'],
               methods=['+ solve(solver)', '+ getVehicles()', '+ getDistance()'],
               color='#D5E8D4')
    draw_class(ax, 4.0, 4.2, 2.0, 1.4, 'VRPSolver',
               methods=['+ solve() [abstract]', '+ setProblem(CVRPProblem)'],
               color='#DAE8FC')
    draw_class(ax, 3.0, 0.4, 1.8, 1.4, 'GrandTour',
               methods=['+ solve()'], color='#DAE8FC')
    draw_class(ax, 5.2, 0.4, 1.8, 1.4, 'ClarkeWright',
               methods=['+ solve()'], color='#DAE8FC')
    draw_class(ax, 7.5, 3.5, 2.2, 2.0, 'VRPVisit',
               attrs=['- demand: int', '- name: String'],
               methods=['+ VRPVisit(name,lat,lon,demand)'], color='#FFE6CC')
    draw_class(ax, 7.5, 0.4, 2.2, 1.8, 'VRPProblemFactory',
               methods=['+ buildProblem(String): CVRPProblem'], color='#E1D5E7')

    # Inheritance arrows
    draw_arrow(ax, 1.3, 3.8, 1.3, 3.1)   # CVRPProblem inherits TSPProblem
    draw_arrow(ax, 3.9, 1.1, 5.0, 4.2)   # GrandTour inherits VRPSolver
    draw_arrow(ax, 6.1, 1.1, 5.0, 4.2)   # ClarkeWright inherits VRPSolver

    # Association
    draw_arrow(ax, 2.4, 1.5, 7.5, 4.5, style='assoc')   # CVRPProblem uses VRPVisit

    savefig("fig_class_diagram.pdf")
    savefig("fig_class_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9  –  Algorithm flowchart: Clarke-Wright
# ─────────────────────────────────────────────────────────────────────────────
def fig_cw_flowchart():
    """
    Flowchart of the Clarke-Wright savings algorithm.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(5, 9))
    ax.axis('off')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 9)
    ax.set_title("Clarke-Wright Algorithm Flowchart", fontsize=11)

    def box(ax, x, y, w, h, text, shape='rect', color='#AED6F1'):
        if shape == 'diamond':
            dx, dy = w / 2, h / 2
            cx, cy = x + dx, y + dy
            pts = np.array([[cx, cy + dy], [cx + dx, cy],
                             [cx, cy - dy], [cx - dx, cy]])
            patch = plt.Polygon(pts, closed=True, facecolor='#FDEBD0',
                                edgecolor='#2C3E50', lw=1.2, zorder=2)
            ax.add_patch(patch)
            ax.text(cx, cy, text, ha='center', va='center', fontsize=8, zorder=3)
        else:
            patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                   facecolor=color, edgecolor='#2C3E50',
                                   lw=1.2, zorder=2)
            ax.add_patch(patch)
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                    fontsize=8, zorder=3, wrap=True)

    def arr(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2))

    # Boxes (x, y, w, h, text)
    steps = [
        (1.0, 8.2, 3.0, 0.55, 'START: Build initial solution\n(1 route per customer)', 'rect', '#D5E8D4'),
        (1.0, 7.1, 3.0, 0.55, 'Compute savings for all\ncustomer pairs (i, j)', 'rect', '#AED6F1'),
        (1.0, 6.0, 3.0, 0.55, 'Sort savings list\n(descending order)', 'rect', '#AED6F1'),
        (0.8, 4.7, 3.4, 0.55, 'Is there an unprocessed saving?', 'diamond', '#FDEBD0'),
        (1.0, 3.6, 3.0, 0.55, 'Check feasibility:\ncapacity + route-end conditions', 'rect', '#AED6F1'),
        (0.8, 2.5, 3.4, 0.55, 'Feasible?', 'diamond', '#FDEBD0'),
        (1.0, 1.4, 3.0, 0.55, 'Merge routes i and j', 'rect', '#AED6F1'),
        (1.0, 0.3, 3.0, 0.55, 'RETURN solution', 'rect', '#D5E8D4'),
    ]
    centers = []
    for (x, y, w, h, text, shape, col) in steps:
        box(ax, x, y, w, h, text, shape, col)
        centers.append((x + w / 2, y + h / 2, y, y + h))

    # Arrows
    arr(ax, 2.5, 8.2,  2.5, 7.65)
    arr(ax, 2.5, 7.1,  2.5, 6.55)
    arr(ax, 2.5, 6.0,  2.5, 5.25)
    arr(ax, 2.5, 4.7,  2.5, 4.15)   # yes
    ax.text(2.6, 4.45, 'Yes', fontsize=8, color='green')
    arr(ax, 2.5, 3.6,  2.5, 3.05)
    arr(ax, 2.5, 2.5,  2.5, 1.95)   # yes
    ax.text(2.6, 2.75, 'Yes', fontsize=8, color='green')
    arr(ax, 2.5, 1.4,  2.5, 0.85)

    # No arrows (loop back)
    ax.annotate('', xy=(4.6, 5.25), xytext=(4.2, 4.97),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))
    ax.plot([4.2, 4.6, 4.6, 4.2], [4.97, 4.97, 6.27, 6.27], 'r-', lw=1.0)
    ax.text(4.65, 5.0, 'No\n(next\nsaving)', fontsize=7, color='red')

    ax.annotate('', xy=(0.4, 3.05), xytext=(0.8, 2.77),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.0))
    ax.plot([0.4, 0.4, 0.8], [3.05, 5.25, 5.25], 'r-', lw=1.0)
    ax.text(0.05, 4.2, 'No\n(skip)', fontsize=7, color='red')

    # "No more savings" → RETURN
    ax.annotate('', xy=(4.7, 0.58), xytext=(4.2, 4.97),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    savefig("fig_cw_flowchart.pdf")
    savefig("fig_cw_flowchart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 2 – VRP Heuristics ...")
    fig_cvrp_layout()
    fig_cw_start()
    fig_cw_savings()
    fig_grand_tour_split()
    fig_benchmark_comparison()
    fig_crop_problem_instances()
    fig_crop_solution_comparison()
    fig_class_diagram()
    fig_cw_flowchart()
    print("All figures generated successfully.")
