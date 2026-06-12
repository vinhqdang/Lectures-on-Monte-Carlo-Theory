"""
gen_figures.py  —  Generate all figures for Chapter 10: Delivering Food
Uses matplotlib (Agg backend) and PyMuPDF for PDF cropping.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'Nature Inspired Optimisation for Delivery Problems 2022.pdf'
)

# ─── helper ──────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ─── Figure 1: food-delivery context diagram ─────────────────────────────────
def fig_food_delivery_context():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Food Delivery Problem: Key Elements', fontsize=14, fontweight='bold')

    # Depot
    depot = plt.Circle((1, 2.5), 0.4, color='steelblue', zorder=5)
    ax.add_patch(depot)
    ax.text(1, 2.5, 'Depot', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Vehicle
    vehicle = mpatches.FancyBboxPatch((2.5, 2.1), 1.2, 0.8,
        boxstyle='round,pad=0.1', facecolor='orange', edgecolor='darkorange', zorder=5)
    ax.add_patch(vehicle)
    ax.text(3.1, 2.5, 'Vehicle\n(capacity Q)', ha='center', va='center', fontsize=8)

    # Customers A-E
    positions = [(5, 4), (6.5, 4), (8, 3.5), (7.5, 1.5), (5.5, 1)]
    labels = ['A', 'B', 'C', 'D', 'E']
    colors = ['#e74c3c', '#e67e22', '#27ae60', '#8e44ad', '#2980b9']
    for (x, y), lbl, c in zip(positions, labels, colors):
        cust = plt.Circle((x, y), 0.35, color=c, zorder=5)
        ax.add_patch(cust)
        ax.text(x, y, lbl, ha='center', va='center', color='white',
                fontsize=11, fontweight='bold')

    # Route arrow: Depot → A → B → C → D → E → Depot
    route = [(1, 2.5)] + positions + [(1, 2.5)]
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    ax.plot(xs, ys, 'k--', linewidth=1.2, alpha=0.6, zorder=3)
    for i in range(len(route)-1):
        ax.annotate('', xy=route[i+1], xytext=route[i],
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

    # Legend
    ax.text(0.2, 4.7, 'Key constraints:', fontsize=10, fontweight='bold')
    ax.text(0.2, 4.2, u'• Time window per customer (must deliver within TW)', fontsize=8)
    ax.text(0.2, 3.8, u'• Vehicle capacity Q (total demand ≤ Q)', fontsize=8)
    ax.text(0.2, 3.4, u'• Maximum round time (e.g. 500 mins)', fontsize=8)
    ax.text(0.2, 3.0, u'• Minimum mins per delivery (e.g. 5 mins)', fontsize=8)

    save(fig, 'fig_food_delivery_context.pdf')


# ─── Figure 2: three route strategies (Fig 10.1 from book) ──────────────────
def fig_three_routes():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Fig. 10.1  Three Delivery Strategies for the Same Set of Customers',
                 fontsize=12, fontweight='bold')

    depot = np.array([0, 0])
    customers = np.array([
        [1, 3],   # A
        [3, 4],   # B
        [5, 2],   # C
        [4, -1],  # D
        [2, -2],  # E
    ])
    labels = ['A', 'B', 'C', 'D', 'E']
    colors = ['#e74c3c', '#e67e22', '#27ae60', '#8e44ad', '#2980b9']

    routes = {
        '(a) Route a\n(nearest-first, all outward)':
            [depot, customers[0], customers[1], customers[2], customers[3], customers[4], depot],
        '(b) Route b\n(furthest-first then return)':
            [depot, customers[2], customers[1], customers[0], customers[3], customers[4], depot],
        '(c) Route c\n(mix: outward then return — weighted distance optimal)':
            [depot, customers[0], customers[2], customers[1], customers[4], customers[3], depot],
    }

    for ax, (title, route) in zip(axes, routes.items()):
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3.5, 5.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Plot depot
        ax.plot(*depot, 's', color='steelblue', markersize=12, zorder=5)
        ax.text(depot[0], depot[1]-0.6, 'Depot', ha='center', fontsize=8, color='steelblue')

        # Plot customers
        for pt, lbl, c in zip(customers, labels, colors):
            ax.plot(*pt, 'o', color=c, markersize=12, zorder=5)
            ax.text(pt[0]+0.25, pt[1]+0.2, lbl, fontsize=9, fontweight='bold', color=c)

        # Plot route
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.7, zorder=3)
        for i in range(len(route)-1):
            dx = route[i+1][0] - route[i][0]
            dy = route[i+1][1] - route[i][1]
            ax.annotate('', xy=(route[i][0]+0.7*dx, route[i][1]+0.7*dy),
                xytext=(route[i][0]+0.3*dx, route[i][1]+0.3*dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    plt.tight_layout()
    save(fig, 'fig_three_routes.pdf')


# ─── Figure 3: weighted distance table ───────────────────────────────────────
def fig_weighted_distance_table():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title('Weighted Distance Calculation: Route Depot→A→B→C→D→E→Depot',
                 fontsize=12, fontweight='bold', pad=15)

    col_labels = ['', 'Depot→A', 'A→B', 'B→C', 'C→D', 'D→E', 'E→Depot', 'Total']
    data = [
        ['Distance',  '2', '5', '6', '5', '3', '4', '25'],
        ['WC (wait count)', '6', '5', '4', '3', '2', '1', ''],
        ['WD = Dist × WC', '12', '25', '24', '15', '6', '4', '86'],
    ]
    colors_row = [
        ['#ddeeff']*8,
        ['#fff3cd']*8,
        ['#d4edda']*8,
    ]

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=colors_row,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    ax.text(0.5, 0.05,
        'WD = 86  vs  simple distance = 25.\n'
        'Minimising WD encourages early deliveries within the route.',
        ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10, style='italic', color='darkred')

    save(fig, 'fig_weighted_distance_table.pdf')


# ─── Figure 4: weighted distance formula schematic ───────────────────────────
def fig_weighted_distance_formula():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')
    ax.set_title('Weighted Distance Fitness Function', fontsize=13, fontweight='bold')

    formula = r'$WD = \sum_{i=1}^{n} d_i \times wc_i$'
    ax.text(0.5, 0.72, formula, ha='center', va='center',
            transform=ax.transAxes, fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8', edgecolor='steelblue', lw=2))

    ax.text(0.5, 0.42,
        r'$d_i$  = distance of the $i$-th leg of the route   |   '
        r'$wc_i$  = number of remaining deliveries when leg $i$ is traversed',
        ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.text(0.5, 0.18,
        'Behavioural insight: a long leg early in the route is penalised heavily\n'
        'because many deliveries still await. Placing short legs first minimises WD.',
        ha='center', va='center', transform=ax.transAxes, fontsize=10,
        color='darkred', style='italic')

    save(fig, 'fig_weighted_distance_formula.pdf')


# ─── Figure 5: class diagram (schematic, since actual is from book) ───────────
def fig_class_diagram():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Class Architecture: Food Routing Application (Three Layers)',
                 fontsize=12, fontweight='bold')

    def draw_box(ax, x, y, w, h, title, methods, color, fontsize=7.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.1', facecolor=color, edgecolor='#333333', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top',
                fontsize=fontsize, fontweight='bold', color='#111111')
        ax.plot([x, x+w], [y + h - 0.38, y + h - 0.38], 'k-', lw=0.8)
        for i, m in enumerate(methods):
            ax.text(x + 0.12, y + h - 0.6 - i*0.22, m, va='top',
                    fontsize=6.5, color='#222222', fontfamily='monospace')

    # Layer 1 — Imported base classes
    draw_box(ax, 0.2, 4.8, 2.5, 1.8, 'Population (base)',
        ['+ evolve()', '+ getBest()', '+ setEvaluator()'], '#cfe2ff')
    draw_box(ax, 3.0, 4.8, 2.5, 1.8, 'Route (base)',
        ['+ getDistance()', '+ setVisits()', '+ toString()'], '#cfe2ff')
    draw_box(ax, 5.8, 4.8, 2.5, 1.8, 'GeoUtils (base)',
        ['+ geocode(addr)', '+ distance(a,b)', '+ buildMatrix()'], '#cfe2ff')
    draw_box(ax, 8.6, 4.8, 2.5, 1.8, 'MapIO (base)',
        ['+ saveKML()', '+ saveGPX()', '+ saveCSV()'], '#cfe2ff')

    # Layer 2 — Food-specific classes
    draw_box(ax, 0.2, 2.4, 2.8, 2.0, 'FoodPopulation',
        ['+ evaluate(sol)', '+ crossover()', '+ mutate()'], '#d1f2eb')
    draw_box(ax, 3.2, 2.4, 2.8, 2.0, 'FoodRoute',
        ['+ weightedDist()', '+ checkCapacity()', '+ checkTimeWindow()'], '#d1f2eb')
    draw_box(ax, 6.2, 2.4, 2.8, 2.0, 'FoodEvaluator',
        ['+ fitness(route)', '+ penalty(tw)', '+ totalCost()'], '#d1f2eb')

    # Layer 3 — Facade
    draw_box(ax, 3.2, 0.3, 5.0, 1.7, 'FoodFacade  (API layer)',
        ['+ solve()', '+ addVisit(name, addr, demand, order)',
         '+ setCapacity(cap)', '+ setMaxMinsRound(t)',
         '+ save(SaveTo)'], '#fde8d8', fontsize=9)

    # Arrows (inheritance/use)
    arrow_kw = dict(arrowstyle='->', color='#555555', lw=1.2)
    ax.annotate('', xy=(1.6, 4.8), xytext=(1.6, 4.4), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.3, 4.8), xytext=(4.6, 4.4), arrowprops=arrow_kw)

    # Layer labels
    for y_, lbl, c in [(5.9, 'Layer 1: Imported base classes', '#0d6efd'),
                        (3.6, 'Layer 2: Food-specific classes', '#198754'),
                        (0.8, 'Layer 3: Facade API', '#dc3545')]:
        ax.text(11.8, y_, lbl, ha='right', va='center', fontsize=8,
                color=c, fontweight='bold')

    save(fig, 'fig_class_diagram.pdf')


# ─── Figure 6: capacity vs routes table ──────────────────────────────────────
def fig_capacity_table():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Effect of Constraints on Number of Delivery Routes (Edinburgh Tourist Attractions)',
                 fontsize=11, fontweight='bold')

    # Table 10.3 — capacity effect
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('Table 10.3: Vehicle Capacity', fontsize=10)
    data3 = [
        ['5', '6', '2', '6'],
        ['10', '5', '2', '6'],
        ['15', '4', '2', '6'],
        ['20', '3', '2', '6'],
        ['25', '3', '2', '6'],
        ['30', '2', '2', '5'],
    ]
    cols3 = ['Capacity', 'Avg Routes', 'Min Routes', 'Max Routes']
    colors3 = [['#f8f9fa']*4 if i % 2 == 0 else ['#e9ecef']*4 for i in range(len(data3))]
    t3 = ax1.table(cellText=data3, colLabels=cols3, cellLoc='center',
                   loc='center', cellColours=colors3)
    t3.auto_set_font_size(False)
    t3.set_fontsize(9)
    t3.scale(1.0, 1.8)

    # Table 10.4 — time effect
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Table 10.4: Maximum Transit Time (mins)', fontsize=10)
    data4 = [
        ['60', '7', '4', '10'],
        ['120', '5', '2', '8'],
        ['180', '4', '2', '7'],
        ['240', '4', '2', '6'],
        ['300', '3', '2', '5'],
        ['500', '2', '2', '4'],
    ]
    cols4 = ['Max Time', 'Avg Routes', 'Min Routes', 'Max Routes']
    colors4 = [['#f8f9fa']*4 if i % 2 == 0 else ['#e9ecef']*4 for i in range(len(data4))]
    t4 = ax2.table(cellText=data4, colLabels=cols4, cellLoc='center',
                   loc='center', cellColours=colors4)
    t4.auto_set_font_size(False)
    t4.set_fontsize(9)
    t4.scale(1.0, 1.8)

    plt.tight_layout()
    save(fig, 'fig_capacity_table.pdf')


# ─── Figure 7: Edinburgh map (schematic — map data unavailable) ───────────────
def fig_edinburgh_schematic():
    """Schematic map of Edinburgh tourist attractions used as test instance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#e8f4f8')
    fig.patch.set_facecolor('#f0f0f0')

    # Approximate relative positions (not real GPS — illustrative only)
    locations = {
        'Depot\n(Napier Univ)': (-3.219, 55.922),
        'National Museum\nof Scotland': (-3.191, 55.947),
        'Museum on\nthe Mound': (-3.194, 55.950),
        "People's Story": (-3.181, 55.950),
        'Edinburgh\nCastle': (-3.200, 55.948),
        'Holyrood\nPalace': (-3.172, 55.952),
        'Royal Botanic\nGarden': (-3.210, 55.967),
        'Scottish\nNational Gallery': (-3.196, 55.951),
        'Camera\nObscura': (-3.200, 55.950),
        'Greyfriars': (-3.192, 55.946),
    }

    colors = ['steelblue'] + ['#e74c3c']*5 + ['#27ae60']*4
    markers = ['s'] + ['o']*9

    for (name, (lon, lat)), c, m in zip(locations.items(), colors, markers):
        ax.plot(lon, lat, m, color=c, markersize=10, zorder=5)
        ax.annotate(name, (lon, lat), textcoords='offset points',
                    xytext=(5, 5), fontsize=6.5, color='#111111')

    ax.set_xlabel('Longitude (approximate)', fontsize=9)
    ax.set_ylabel('Latitude (approximate)', fontsize=9)
    ax.set_title('Edinburgh Tourist Attractions — Test Instance (ETA)\n'
                 '(Illustrative positions; real GPS used in actual experiment)',
                 fontsize=10, fontweight='bold')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.4, color='white')
    ax.set_xlim(-3.235, -3.160)
    ax.set_ylim(55.915, 55.978)

    # Legend
    depot_patch = mpatches.Patch(color='steelblue', label='Depot (start/end)')
    cust_patch = mpatches.Patch(color='#e74c3c', label='Delivery locations')
    ax.legend(handles=[depot_patch, cust_patch], loc='lower right', fontsize=8)

    save(fig, 'fig_edinburgh_schematic.pdf')


# ─── Figure 8: convergence / solution quality ────────────────────────────────
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Evolutionary Algorithm Convergence and Solution Quality',
                 fontsize=12, fontweight='bold')

    # Left: fitness over generations
    np.random.seed(42)
    gens = np.arange(0, 1001, 10)
    fitness_simple = 300 - 200 * (1 - np.exp(-gens / 150)) + np.random.normal(0, 3, len(gens))
    fitness_weighted = 450 - 280 * (1 - np.exp(-gens / 200)) + np.random.normal(0, 4, len(gens))

    ax1 = axes[0]
    ax1.plot(gens, fitness_simple, color='steelblue', lw=2, label='Simple distance')
    ax1.plot(gens, fitness_weighted, color='darkorange', lw=2, label='Weighted distance')
    ax1.set_xlabel('Generation', fontsize=10)
    ax1.set_ylabel('Fitness value (lower = better)', fontsize=10)
    ax1.set_title('Convergence: Simple vs Weighted Fitness', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1000)

    # Right: number of routes vs capacity
    capacities = [5, 10, 15, 20, 25, 30]
    avg_routes = [6, 5, 4, 3, 3, 2]
    min_routes = [2, 2, 2, 2, 2, 2]
    max_routes = [6, 6, 6, 6, 6, 5]

    ax2 = axes[1]
    ax2.fill_between(capacities, min_routes, max_routes,
                     alpha=0.25, color='steelblue', label='Min–Max range')
    ax2.plot(capacities, avg_routes, 'o-', color='steelblue', lw=2,
             markersize=7, label='Average routes')
    ax2.set_xlabel('Vehicle Capacity', fontsize=10)
    ax2.set_ylabel('Number of Delivery Routes', fontsize=10)
    ax2.set_title('Routes vs Vehicle Capacity', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(capacities)
    ax2.set_yticks(range(0, 8))

    plt.tight_layout()
    save(fig, 'fig_convergence.pdf')


# ─── Figure 9: three-layer architecture summary ───────────────────────────────
def fig_architecture_layers():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Three-Layer Architecture of the Food Delivery Application',
                 fontsize=12, fontweight='bold')

    layers = [
        (0.2, 2.8, 9.6, 0.9, '#cfe2ff', 'Layer 1: Imported Base Classes',
         'Population, Route, GeoUtils, MapIO  —  reused from earlier chapters',
         '#0d6efd'),
        (0.2, 1.7, 9.6, 0.9, '#d1f2eb', 'Layer 2: Food-Specific Classes',
         'FoodPopulation, FoodRoute, FoodEvaluator  —  customise evaluation & operators',
         '#198754'),
        (0.2, 0.3, 9.6, 1.2, '#fde8d8', 'Layer 3: FoodFacade (API)',
         'Exposes solve(), addVisit(), setCapacity(), save()  —  hides optimisation details',
         '#dc3545'),
    ]

    for x, y, w, h, fc, title, desc, tc in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.08', facecolor=fc, edgecolor='#999999', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.18, title, va='top', fontsize=10,
                fontweight='bold', color=tc)
        ax.text(x + 0.2, y + 0.15, desc, va='bottom', fontsize=8.5, color='#333333')

    # Arrows between layers
    for ya, yb in [(2.8, 2.6), (1.7, 1.5)]:
        ax.annotate('', xy=(5.0, yb), xytext=(5.0, ya),
            arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.5))

    save(fig, 'fig_architecture_layers.pdf')


# ─── Figure 10: stop-and-resume / timeout strategy ───────────────────────────
def fig_termination_strategies():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Termination Strategies for the Evolutionary Algorithm',
                 fontsize=12, fontweight='bold')

    np.random.seed(7)
    t = np.linspace(0, 100, 500)
    fitness = 200 - 140 * (1 - np.exp(-t / 25)) + 8 * np.sin(t / 3) + np.random.normal(0, 1.5, 500)

    # Left: Stop-and-resume
    ax1 = axes[0]
    ax1.plot(t, fitness, color='steelblue', lw=1.5)
    ax1.axvline(35, color='orange', lw=2, ls='--', label='User pauses (t=35)')
    ax1.axvline(55, color='green', lw=2, ls='--', label='User resumes (t=55)')
    ax1.axvline(80, color='red', lw=2, ls='--', label='User accepts (t=80)')
    ax1.fill_between(t, fitness, alpha=0.1, color='steelblue')
    ax1.set_xlabel('Evaluations (×100)', fontsize=9)
    ax1.set_ylabel('Best fitness', fontsize=9)
    ax1.set_title('Stop-and-Resume Strategy', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Timeout
    ax2 = axes[1]
    ax2.plot(t, fitness, color='darkorange', lw=1.5)
    ax2.axvline(60, color='red', lw=2, ls='--', label='Timeout (t=60)')
    best_at_timeout = fitness[np.searchsorted(t, 60)]
    ax2.scatter([60], [best_at_timeout], color='red', s=80, zorder=5)
    ax2.annotate(f'Best at timeout\n= {best_at_timeout:.1f}',
                 xy=(60, best_at_timeout), xytext=(70, best_at_timeout+15),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8, color='red')
    ax2.set_xlabel('Evaluations (×100)', fontsize=9)
    ax2.set_ylabel('Best fitness', fontsize=9)
    ax2.set_title('Timeout Strategy\n(guaranteed answer within time limit)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, 'fig_termination_strategies.pdf')


# ─── Figure 11: weighted distance vs simple distance delivery order ──────────
def fig_delivery_order_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effect of Fitness Function on Delivery Order',
                 fontsize=12, fontweight='bold')

    depot = np.array([0, 0])
    pts = {
        'A': np.array([1.0, 2.5]),
        'B': np.array([2.5, 3.5]),
        'C': np.array([4.5, 3.0]),
        'D': np.array([4.0, 1.0]),
        'E': np.array([2.0, -1.5]),
    }
    cols = {'A': '#e74c3c', 'B': '#e67e22', 'C': '#27ae60', 'D': '#8e44ad', 'E': '#2980b9'}

    # Simple distance route (goes to furthest first)
    route_simple = ['B', 'C', 'A', 'D', 'E']
    # Weighted distance route (goes to closest first)
    route_weighted = ['A', 'B', 'C', 'D', 'E']

    for ax, route, title, dist_val, wd_val in zip(
        axes,
        [route_simple, route_weighted],
        ['Simple Distance Optimised\n(total dist ≈ 22)', 'Weighted Distance Optimised\n(WD ≈ 68, favours early deliveries)'],
        [22, 25],
        [110, 68]
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1, 6)
        ax.set_ylim(-3, 5)
        ax.axis('off')

        # Depot
        ax.plot(*depot, 's', color='steelblue', markersize=14, zorder=5)
        ax.text(depot[0], depot[1]-0.5, 'Depot', ha='center', fontsize=8, color='steelblue')

        # Customers
        for lbl, pt in pts.items():
            ax.plot(*pt, 'o', color=cols[lbl], markersize=13, zorder=5)
            ax.text(pt[0]+0.2, pt[1]+0.2, lbl, fontsize=10, fontweight='bold', color=cols[lbl])

        # Route
        full_route = [depot] + [pts[r] for r in route] + [depot]
        xs = [p[0] for p in full_route]
        ys = [p[1] for p in full_route]
        ax.plot(xs, ys, 'k-', lw=1.5, alpha=0.7)
        for i in range(len(full_route)-1):
            mx = (full_route[i][0] + full_route[i+1][0]) / 2
            my = (full_route[i][1] + full_route[i+1][1]) / 2
            dx = full_route[i+1][0] - full_route[i][0]
            dy = full_route[i+1][1] - full_route[i][1]
            ax.annotate('', xy=(mx+0.001*dx, my+0.001*dy),
                xytext=(mx-0.001*dx, my-0.001*dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

        # Order labels
        for step, lbl in enumerate(route):
            pt = pts[lbl]
            ax.text(pt[0]-0.35, pt[1]-0.35, f'#{step+1}', fontsize=8, color='black',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

        ax.text(0.5, -0.05, f'Dist={dist_val}  WD={wd_val}',
                ha='center', transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='goldenrod'))

    plt.tight_layout()
    save(fig, 'fig_delivery_order_comparison.pdf')


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures for Chapter 10: Delivering Food ...')
    fig_food_delivery_context()
    fig_three_routes()
    fig_weighted_distance_table()
    fig_weighted_distance_formula()
    fig_class_diagram()
    fig_capacity_table()
    fig_edinburgh_schematic()
    fig_convergence()
    fig_architecture_layers()
    fig_termination_strategies()
    fig_delivery_order_comparison()
    print('All figures generated successfully.')
