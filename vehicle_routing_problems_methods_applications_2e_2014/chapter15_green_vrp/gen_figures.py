"""
gen_figures.py  —  Generate all figures for Chapter 15: Green Vehicle Routing
Requires: matplotlib, numpy, pymupdf (fitz)
Run: conda run -n py313 python3 gen_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── Output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def save(name, dpi=150):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Fuel consumption vs speed curve (convex U-shape)
# Shows that fuel use per km is minimised at an intermediate speed (~80 km/h)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fuel_vs_speed():
    v = np.linspace(20, 130, 300)   # speed in km/h
    # Simplified MEET-style model: F(v) = a/v + b + c*v^2
    a, b, c = 800, 0.10, 0.00018
    F = a/v + b + c*v**2            # fuel consumption in ml/km (shape only)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(v, F, 'b-', linewidth=2.5, label='Fuel consumption (ml/km)')

    v_opt = v[np.argmin(F)]
    ax.axvline(v_opt, color='red', linestyle='--', linewidth=1.5,
               label=f'Optimal speed ≈ {v_opt:.0f} km/h')
    ax.annotate(f'Min fuel\n≈ {v_opt:.0f} km/h',
                xy=(v_opt, min(F)), xytext=(v_opt+15, min(F)+0.08),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    ax.set_xlabel('Vehicle speed  v  (km/h)', fontsize=11)
    ax.set_ylabel('Fuel consumption  F(v)  (ml/km)', fontsize=11)
    ax.set_title('Fuel Consumption vs. Speed\n(convex relationship — eco-driving target)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save("fuel_vs_speed.pdf")
    save("fuel_vs_speed.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Fuel consumption vs load at fixed speed (linear relationship)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fuel_vs_load():
    load = np.linspace(0, 20, 200)   # tonnes
    v = 80                           # km/h fixed
    # F = alpha * (a + b*w) * d  — linear in weight w
    alpha, a_coeff, b_coeff = 0.27, 0.85, 0.043
    F = alpha * (a_coeff + b_coeff * load)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(load, F, 'g-', linewidth=2.5)
    ax.fill_between(load, F, alpha=0.15, color='green')

    ax.set_xlabel('Vehicle load  w  (tonnes)', fontsize=11)
    ax.set_ylabel('Fuel consumption  F  (litres/km)', fontsize=11)
    ax.set_title('Fuel Consumption vs. Payload\n(linear in load — heavier = more fuel)',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.annotate('Each extra tonne of cargo\nincreases fuel linearly',
                xy=(10, alpha*(a_coeff+b_coeff*10)),
                xytext=(12, alpha*(a_coeff+b_coeff*10)-0.05),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                fontsize=9, color='darkgreen')
    save("fuel_vs_load.pdf")
    save("fuel_vs_load.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Speed vs fuel consumption surface (speed × load joint effect)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fuel_surface():
    v = np.linspace(30, 120, 60)
    w = np.linspace(0, 20, 60)
    V, W = np.meshgrid(v, w)
    # Comprehensive model: F = (a + b*W)*V^2 + (c + d*W)/V  (approximate CMEM)
    a, b, c, d = 0.00010, 0.000008, 12.0, 1.0
    F = (a + b*W)*V**2 + (c + d*W)/V

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(V, W, F, cmap='viridis', alpha=0.85, linewidth=0)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Fuel (ml/km)')
    ax.set_xlabel('Speed v (km/h)', fontsize=9)
    ax.set_ylabel('Load w (tonnes)', fontsize=9)
    ax.set_zlabel('Fuel F (ml/km)', fontsize=9)
    ax.set_title('Joint Effect of Speed & Load on Fuel Consumption', fontsize=11)
    save("fuel_surface.pdf")
    save("fuel_surface.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Green VRP concept diagram — depot + customers + routes
# ─────────────────────────────────────────────────────────────────────────────
def fig_gvrp_concept():
    np.random.seed(42)
    n_customers = 8
    depot = np.array([0.5, 0.5])
    customers = np.random.rand(n_customers, 2)

    # Two simple routes (hardcoded for clarity)
    route1 = [0, 1, 3, 5, 0]   # indices into combined [depot] + customers
    route2 = [0, 2, 4, 6, 7, 0]
    coords = np.vstack([depot, customers])

    fig, ax = plt.subplots(figsize=(7, 5))

    colors_route = ['#1f77b4', '#d62728']
    route_labels = ['Route 1 (low emission)', 'Route 2 (low emission)']
    for ri, (route, col, label) in enumerate(zip([route1, route2], colors_route, route_labels)):
        xs = coords[route, 0]
        ys = coords[route, 1]
        ax.plot(xs, ys, '-o', color=col, linewidth=2, markersize=7,
                label=label, zorder=3)
        # arrows
        for k in range(len(route)-1):
            ax.annotate('', xy=coords[route[k+1]], xytext=coords[route[k]],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # depot
    ax.scatter(*depot, s=200, color='black', marker='s', zorder=5, label='Depot')
    ax.text(depot[0]+0.02, depot[1]+0.02, 'Depot', fontsize=10, weight='bold')

    # customers
    for i in range(n_customers):
        ax.text(customers[i,0]+0.02, customers[i,1]+0.02, f'C{i+1}', fontsize=9)

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Green VRP: Optimising Routes to Minimise Fuel & Emissions\n'
                 '(each vehicle leaves and returns to the depot)',
                 fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.axis('off')
    save("gvrp_concept.pdf")
    save("gvrp_concept.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Pollution Routing Problem — speed segments on arcs
# ─────────────────────────────────────────────────────────────────────────────
def fig_prp_speed_segments():
    fig, ax = plt.subplots(figsize=(8, 3))

    # Draw a route: 5 nodes
    nodes = np.array([[0.05, 0.5], [0.25, 0.5], [0.50, 0.5],
                      [0.72, 0.5], [0.92, 0.5]])
    labels = ['Depot', 'C1', 'C2', 'C3', 'Depot']
    speeds = ['v=60 km/h', 'v=90 km/h', 'v=70 km/h', 'v=80 km/h']
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    for i in range(len(nodes)-1):
        x0, y0 = nodes[i]
        x1, y1 = nodes[i+1]
        mid = ((x0+x1)/2, (y0+y1)/2 + 0.12)
        ax.annotate('', xy=nodes[i+1], xytext=nodes[i],
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
        ax.text(mid[0], mid[1], speeds[i], ha='center', fontsize=9,
                color=colors[i], weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=colors[i], alpha=0.8))

    for i, (node, label) in enumerate(zip(nodes, labels)):
        marker = 's' if label == 'Depot' else 'o'
        ax.scatter(*node, s=150, zorder=5,
                   color='black' if label == 'Depot' else '#9C27B0',
                   marker=marker)
        ax.text(node[0], node[1]-0.15, label, ha='center', fontsize=9, weight='bold')

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.1, 0.85)
    ax.set_title('Pollution Routing Problem (PRP): Each Arc Has an Optimised Speed\n'
                 '(speed is a decision variable, balancing time vs. emissions)',
                 fontsize=11)
    ax.axis('off')
    save("prp_speed_segments.pdf")
    save("prp_speed_segments.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Pareto front — cost vs. CO2 emissions
# ─────────────────────────────────────────────────────────────────────────────
def fig_pareto_front():
    np.random.seed(7)
    # Simulated Pareto-efficient solutions
    cost = np.linspace(1000, 2500, 20)
    # As cost decreases, emissions increase (trade-off)
    emissions = 500 + 8000 / (cost - 800) + np.random.randn(20)*5
    emissions = np.sort(emissions)[::-1]

    # Dominated solutions (cloud)
    dom_cost = np.random.uniform(1200, 2600, 60)
    dom_emis = np.random.uniform(min(emissions)+50, max(emissions)+200, 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dom_cost, dom_emis, color='lightgray', s=40, label='Dominated solutions',
               zorder=2)
    ax.plot(cost, emissions, 'b-o', linewidth=2.5, markersize=8,
            label='Pareto front (efficient solutions)', zorder=3)

    ax.fill_between(cost, emissions, max(dom_emis)+50, alpha=0.08, color='blue')

    ax.set_xlabel('Total Cost (€)', fontsize=12)
    ax.set_ylabel('CO$_2$ Emissions (kg)', fontsize=12)
    ax.set_title('Multicriteria Analysis: Cost vs. Emissions Pareto Front\n'
                 '(no single solution dominates both objectives)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate two extreme points
    ax.annotate('Cheapest\n(most emissions)',
                xy=(cost[0], emissions[0]), xytext=(cost[0]-300, emissions[0]+30),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')
    ax.annotate('Cleanest\n(most expensive)',
                xy=(cost[-1], emissions[-1]), xytext=(cost[-1]+50, emissions[-1]+30),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')
    save("pareto_front.pdf")
    save("pareto_front.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Electric VRP (E-VRP) — range anxiety & recharging stations
# ─────────────────────────────────────────────────────────────────────────────
def fig_evrp_stations():
    np.random.seed(11)
    n_cust = 7
    depot = np.array([0.5, 0.5])
    customers = np.random.rand(n_cust, 2) * 0.8 + 0.1
    # Two recharging stations
    stations = np.array([[0.25, 0.75], [0.72, 0.28]])

    # One route visiting a station mid-way
    route = [0, 1, 2, 8, 3, 4, 0]  # indices: 0=depot, 1..7=customers, 8..9=stations
    all_pts = np.vstack([depot, customers, stations])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Route
    xs = all_pts[route, 0]
    ys = all_pts[route, 1]
    ax.plot(xs, ys, '-', color='#1976D2', linewidth=2, zorder=2)
    for k in range(len(route)-1):
        ax.annotate('', xy=all_pts[route[k+1]], xytext=all_pts[route[k]],
                    arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))

    # Battery level schematic below the route
    battery = [100, 65, 30, 95, 60, 20, 80, 100]  # approx levels
    cmap_bat = plt.cm.RdYlGn

    # Depot
    ax.scatter(*depot, s=250, color='black', marker='s', zorder=5)
    ax.text(depot[0]+0.02, depot[1]+0.03, 'Depot', fontsize=10, weight='bold')

    # Customers
    for i in range(n_cust):
        ax.scatter(*customers[i], s=120, color='#1976D2', zorder=5)
        ax.text(customers[i,0]+0.02, customers[i,1]+0.02, f'C{i+1}', fontsize=8)

    # Stations
    for i, st in enumerate(stations):
        ax.scatter(*st, s=200, color='#E91E63', marker='^', zorder=5,
                   label='Recharging Station' if i == 0 else '')
        ax.text(st[0]+0.02, st[1]+0.03, f'R{i+1}', fontsize=9,
                color='#E91E63', weight='bold')

    # Battery "range circle" around depot
    circle = plt.Circle(depot, 0.38, color='orange', fill=False,
                        linestyle='--', linewidth=1.5, label='Battery range limit')
    ax.add_patch(circle)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Electric VRP (E-VRP): Recharging Stations Overcome Range Anxiety\n'
                 '(vehicle detours to station when battery runs low)',
                 fontsize=10)
    ax.legend(loc='lower left', fontsize=9)
    ax.axis('off')
    save("evrp_stations.pdf")
    save("evrp_stations.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Speed optimisation on a fixed route — optimal speed per arc
# ─────────────────────────────────────────────────────────────────────────────
def fig_speed_optimisation():
    arcs = ['0→C1', 'C1→C2', 'C2→C3', 'C3→0']
    dist = [15, 22, 18, 20]          # km per arc
    loads = [12, 9, 6, 0]           # tonnes (decreasing as deliveries made)
    # Optimal speed per arc from model (varies with load and time window)
    v_opt = [72, 85, 80, 95]
    v_max = 100
    v_min = 50

    x = np.arange(len(arcs))
    width = 0.45

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: speed per arc
    ax = axes[0]
    bars = ax.bar(x, v_opt, width, color='#42A5F5', edgecolor='navy', linewidth=1.2)
    ax.axhline(v_max, color='red', linestyle='--', linewidth=1.2, label=f'Speed limit ({v_max} km/h)')
    ax.axhline(v_min, color='orange', linestyle=':', linewidth=1.2, label=f'Min speed ({v_min} km/h)')
    ax.set_xticks(x); ax.set_xticklabels(arcs, fontsize=10)
    ax.set_ylabel('Optimal Speed (km/h)', fontsize=11)
    ax.set_title('Optimal Speed per Arc\n(lower when heavily loaded)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 115)
    for bar, v in zip(bars, v_opt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v}',
                ha='center', fontsize=10, weight='bold')

    # Right: load per arc
    ax2 = axes[1]
    bars2 = ax2.bar(x, loads, width, color='#66BB6A', edgecolor='darkgreen', linewidth=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(arcs, fontsize=10)
    ax2.set_ylabel('Vehicle Load (tonnes)', fontsize=11)
    ax2.set_title('Payload per Arc\n(decreases after each delivery)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for bar, ld in zip(bars2, loads):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{ld}t',
                 ha='center', fontsize=10, weight='bold')

    plt.suptitle('Speed Optimisation on a Fixed Route', fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    save("speed_optimisation.pdf")
    save("speed_optimisation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Emission comparison — traditional VRP vs green VRP
# ─────────────────────────────────────────────────────────────────────────────
def fig_emission_comparison():
    categories = ['Traditional\nVRP\n(min distance)', 'Green VRP\n(min fuel)',
                  'PRP\n(optimise speed)', 'E-VRP\n(electric)']
    emissions = [100, 82, 71, 35]   # relative CO2 index
    colors = ['#EF5350', '#FF9800', '#FFC107', '#66BB6A']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(categories, emissions, color=colors, edgecolor='black',
                  linewidth=1.2, width=0.55)
    for bar, em in zip(bars, emissions):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                f'{em}', ha='center', fontsize=13, weight='bold')

    ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='Baseline (traditional VRP)')
    ax.set_ylabel('Relative CO$_2$ Emissions (index, baseline=100)', fontsize=11)
    ax.set_title('Emission Reductions Across Green Routing Approaches\n'
                 '(green methods achieve 18%–65% reductions)',
                 fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Percentage reduction annotations
    for i, (bar, em) in enumerate(zip(bars, emissions)):
        if i > 0:
            pct = 100 - em
            ax.text(bar.get_x()+bar.get_width()/2, em/2,
                    f'−{pct}%', ha='center', va='center',
                    fontsize=10, color='white', weight='bold')
    save("emission_comparison.pdf")
    save("emission_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Time-window effect on speed choice
# ─────────────────────────────────────────────────────────────────────────────
def fig_time_window_speed():
    time = np.linspace(0, 4, 300)    # hours
    dist = 100                        # km

    # Tight window: must arrive at t=2h → speed=50 km/h
    # Loose window: can arrive at t=2..4h → choose optimal speed ~80 km/h
    v_tight = dist / 2.0             # must go 50 km/h exactly
    v_optimal = 80.0                 # fuel-optimal

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Position vs time for each strategy
    t_tight = dist / v_tight          # = 2 h
    t_opt   = dist / v_optimal        # = 1.25 h (arrives early, waits)

    ax.plot([0, t_tight], [0, dist], 'r-', linewidth=2.5,
            label=f'Tight window: v={v_tight:.0f} km/h (forced speed)')
    ax.plot([0, t_opt, 2.0], [0, dist, dist], 'g-', linewidth=2.5,
            label=f'Loose window: v={v_optimal:.0f} km/h → wait at customer')

    # Window
    ax.axvspan(2.0, 3.5, alpha=0.12, color='blue', label='Time window [2h, 3.5h]')
    ax.axhline(dist, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('Travel time (hours)', fontsize=11)
    ax.set_ylabel('Distance covered (km)', fontsize=11)
    ax.set_title('Time Windows and Speed Choice\n'
                 '(slack in time window allows eco-driving at optimal speed)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save("time_window_speed.pdf")
    save("time_window_speed.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Crop from book PDF — fuel vs speed at varying loads (Figure 15.1)
# ─────────────────────────────────────────────────────────────────────────────
def fig_crop_book_fuel_speed():
    """Crop Figure 15.1 from the book PDF (page 441, showing fuel vs speed curves)."""
    try:
        import fitz  # PyMuPDF
        PDF_PATH = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
                    "vehicle_routing_problems_methods_applications_2e_2014/"
                    "Vehicle Routing_ Problems, Methods, and Applications, "
                    "Second Edition 2014.pdf")
        doc = fitz.open(PDF_PATH)
        # Page 445 (0-indexed: 444) contains Figure 15.1 (fuel vs speed)
        page = doc[444]   # p445 in the book
        # Crop to the figure area (approximate bounding box in PDF points)
        rect = fitz.Rect(70, 300, 480, 520)
        mat = fitz.Matrix(2.5, 2.5)
        clip = page.get_pixmap(matrix=mat, clip=rect)
        out = os.path.join(FIG_DIR, "book_fig15_1.png")
        clip.save(out)
        doc.close()
        print(f"  saved: {out}")
    except Exception as e:
        print(f"  [warn] Could not crop book figure: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Modal comparison — road vs rail vs air emissions per tonne-km
# ─────────────────────────────────────────────────────────────────────────────
def fig_modal_comparison():
    modes   = ['Heavy truck\n(diesel)', 'Light van\n(diesel)', 'Rail\n(freight)',
               'Air cargo', 'Electric\ntruck']
    co2_gtkm = [62, 160, 22, 570, 25]   # g CO2 per tonne-km (approximate)
    colors   = ['#EF5350', '#FF7043', '#42A5F5', '#AB47BC', '#66BB6A']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(modes, co2_gtkm, color=colors, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, co2_gtkm):
        ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
                f'{val} g', va='center', fontsize=10, weight='bold')

    ax.set_xlabel('CO$_2$ Emissions (g per tonne-km)', fontsize=11)
    ax.set_title('Emission Intensity by Transport Mode\n'
                 '(rail and electric vehicles far cleaner than air freight)',
                 fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 650)
    save("modal_comparison.pdf")
    save("modal_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 15: Green Vehicle Routing...")
    fig_fuel_vs_speed()
    fig_fuel_vs_load()
    fig_fuel_surface()
    fig_gvrp_concept()
    fig_prp_speed_segments()
    fig_pareto_front()
    fig_evrp_stations()
    fig_speed_optimisation()
    fig_emission_comparison()
    fig_time_window_speed()
    fig_crop_book_fuel_speed()
    fig_modal_comparison()
    print("Done. All figures saved to:", FIG_DIR)
