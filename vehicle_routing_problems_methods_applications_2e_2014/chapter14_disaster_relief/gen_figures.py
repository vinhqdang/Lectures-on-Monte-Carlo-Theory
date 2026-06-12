"""
gen_figures.py  –  Generate all figures for Chapter 14 Beamer slides.
Vehicle Routing Applications in Disaster Relief
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os
import fitz  # pymupdf

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

BOOK_PDF = (
    "/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
    "vehicle_routing_problems_methods_applications_2e_2014/"
    "Vehicle Routing_ Problems, Methods, and Applications, Second Edition 2014.pdf"
)

# ──────────────────────────────────────────────────────────────
# Helper: crop a figure from the PDF by (page_index, rect)
# page_index is 0-based; rect = (x0, y0, x1, y1) in pt units
# ──────────────────────────────────────────────────────────────
def crop_from_pdf(page_index, rect, out_name, dpi=180):
    """Crop a region from the PDF and save as PNG."""
    doc = fitz.open(BOOK_PDF)
    page = doc[page_index]
    clip = fitz.Rect(*rect)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    out_path = os.path.join(FIGURES_DIR, out_name)
    pix.save(out_path)
    doc.close()
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════
# Figure 1: Disaster management phases (cycle diagram)
# ══════════════════════════════════════════════════════════════
def fig_disaster_phases():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Phases of Disaster Management", fontsize=14, fontweight='bold', pad=10)

    phases = ["Preparedness", "Response", "Recovery", "Mitigation"]
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    angles = [90, 0, 270, 180]  # degrees, where each phase circle sits

    r = 0.9  # radius of circle layout
    for i, (phase, color, angle_deg) in enumerate(zip(phases, colors, angles)):
        angle_rad = np.radians(angle_deg)
        x = r * np.cos(angle_rad)
        y = r * np.sin(angle_rad)
        circle = plt.Circle((x, y), 0.35, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, phase, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=4, wrap=True,
                multialignment='center')

    # Draw curved arrows between phases
    arrow_props = dict(arrowstyle='->', color='gray', lw=2,
                       connectionstyle='arc3,rad=0.3')
    phase_coords = [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for a in angles]
    order = [0, 1, 2, 3, 0]
    for j in range(4):
        ax.annotate("", xy=phase_coords[order[j+1]], xytext=phase_coords[order[j]],
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8,
                                   connectionstyle='arc3,rad=0.3'))

    # Central label
    ax.text(0, 0, "Disaster\nManagement\nCycle", ha='center', va='center',
            fontsize=9, color='#333333', style='italic')

    ax.text(0, -1.3,
            "Preparedness (pre-event) → Response (immediate) → Recovery (restoration) → Mitigation (risk reduction)",
            ha='center', va='center', fontsize=7.5, color='#444444', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_disaster_phases.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_disaster_phases.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 2: Network flow model (depot + demand locations)
# Mirrors Figure 14.2 in the book
# ══════════════════════════════════════════════════════════════
def fig_network_flow():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(-0.3, 3.5)
    ax.axis('off')
    ax.set_title("Network Flow Model: Depot, Routes, and Demand Locations",
                 fontsize=13, fontweight='bold')

    # Depot (square)
    depot = (0.5, 1.5)
    ax.add_patch(mpatches.FancyBboxPatch((depot[0]-0.18, depot[1]-0.18), 0.36, 0.36,
                 boxstyle="round,pad=0.05", facecolor='#1565C0', edgecolor='black', lw=1.5, zorder=3))
    ax.text(depot[0], depot[1], "D\n(Depot)", ha='center', va='center', color='white',
            fontsize=9, fontweight='bold', zorder=4)

    # Demand locations (circles)
    demand_nodes = {
        "L1": (1.8, 2.8), "L2": (2.5, 3.2), "L3": (3.5, 2.9),
        "L4": (4.0, 1.5), "L5": (3.5, 0.3), "L6": (2.2, 0.5),
        "L7": (1.5, 0.8),
    }
    colors_d = ['#E53935', '#8E24AA', '#00ACC1', '#43A047', '#FB8C00', '#6D4C41', '#546E7A']
    for (lbl, (x, y)), col in zip(demand_nodes.items(), colors_d):
        circle = plt.Circle((x, y), 0.22, color=col, zorder=3, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white', zorder=4)

    # Routes (two vehicles)
    route1 = [depot, demand_nodes["L1"], demand_nodes["L2"], demand_nodes["L3"], depot]
    route2 = [depot, demand_nodes["L7"], demand_nodes["L6"], demand_nodes["L5"], demand_nodes["L4"], depot]
    for route, col, lbl in [(route1, '#1565C0', 'Vehicle 1'), (route2, '#C62828', 'Vehicle 2')]:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, '-o', color=col, lw=2, ms=6, zorder=2, label=lbl)
        for k in range(len(route)-1):
            dx = route[k+1][0] - route[k][0]
            dy = route[k+1][1] - route[k][1]
            ax.annotate("", xy=(route[k+1][0]-dx*0.15, route[k+1][1]-dy*0.15),
                        xytext=(route[k][0]+dx*0.15, route[k][1]+dy*0.15),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    ax.legend(loc='upper left', fontsize=9)
    ax.text(0.5, -0.2,
            "Each vehicle departs from the depot, serves a subset of demand locations, and returns.",
            ha='center', va='center', fontsize=8, color='#333333', transform=ax.transAxes,
            style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_network_flow.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_network_flow.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 3: Response time vs. coverage trade-off
# ══════════════════════════════════════════════════════════════
def fig_response_coverage():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    vehicles = np.arange(1, 11)
    coverage = 100 * (1 - np.exp(-0.35 * vehicles))
    resp_time = 120 / vehicles + 5

    ax2 = ax.twinx()
    line1, = ax.plot(vehicles, coverage, 'b-o', lw=2, ms=6, label='Coverage (%)')
    line2, = ax2.plot(vehicles, resp_time, 'r-s', lw=2, ms=6, label='Avg. Response Time (min)')

    ax.set_xlabel("Number of Vehicles Deployed", fontsize=11)
    ax.set_ylabel("Population Coverage (%)", color='b', fontsize=11)
    ax2.set_ylabel("Avg. Response Time (minutes)", color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title("Trade-off: Coverage vs. Response Time", fontsize=13, fontweight='bold')
    ax.set_xticks(vehicles)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_response_coverage.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_response_coverage.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 4: Priority routing illustration
# (local visiting priority – Figure 14.3 analogue)
# ══════════════════════════════════════════════════════════════
def fig_priority_routing():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    def draw_scenario(ax, title, route_order, priority_order):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 4.5)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold')

        depot = (0.3, 2.0)
        nodes = {
            "A (High)": (2.0, 3.5),
            "B (Low)":  (2.0, 0.5),
            "C (High)": (4.0, 3.5),
            "D (Low)":  (4.0, 0.5),
        }
        prio_colors = {'High': '#D32F2F', 'Low': '#1976D2'}

        ax.add_patch(mpatches.FancyBboxPatch((depot[0]-0.2, depot[1]-0.2), 0.4, 0.4,
                     boxstyle="round,pad=0.05", facecolor='#37474F', edgecolor='black', lw=1.5))
        ax.text(depot[0], depot[1], "Depot", ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        node_positions = {}
        for (lbl, (x, y)) in nodes.items():
            prio = 'High' if 'High' in lbl else 'Low'
            c = plt.Circle((x, y), 0.3, color=prio_colors[prio], zorder=3, alpha=0.85)
            ax.add_patch(c)
            ax.text(x, y, lbl.split()[0], ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white', zorder=4)
            ax.text(x, y-0.5, lbl.split()[1], ha='center', va='center', fontsize=7,
                    color=prio_colors[prio])
            node_positions[lbl] = (x, y)

        # Draw route
        route = [depot] + [node_positions[k] for k in route_order] + [depot]
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, 'g-', lw=2, zorder=2, alpha=0.7)
        for k in range(len(route)-1):
            dx = route[k+1][0] - route[k][0]
            dy = route[k+1][1] - route[k][1]
            norm = np.sqrt(dx**2 + dy**2) + 1e-9
            ax.annotate("", xy=(route[k+1][0] - dx/norm*0.32, route[k+1][1] - dy/norm*0.32),
                        xytext=(route[k][0] + dx/norm*0.32, route[k][1] + dy/norm*0.32),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            mid_x = (route[k][0] + route[k+1][0]) / 2
            mid_y = (route[k][1] + route[k+1][1]) / 2
            ax.text(mid_x, mid_y + 0.12, str(k+1), ha='center', va='bottom',
                    fontsize=8, color='green', fontweight='bold')

        high_p = mpatches.Patch(color='#D32F2F', label='High Priority')
        low_p  = mpatches.Patch(color='#1976D2', label='Low Priority')
        ax.legend(handles=[high_p, low_p], loc='lower right', fontsize=7)

    # Scenario 1: ignore priority (nearest-neighbour)
    draw_scenario(axes[0], "Ignoring Priority\n(Nearest-Neighbour Route)",
                  ["A (High)", "C (High)", "D (Low)", "B (Low)"],
                  ["A (High)", "C (High)"])

    # Scenario 2: respect priority (serve high-priority first)
    draw_scenario(axes[1], "Respecting Priority\n(High-Priority Locations First)",
                  ["A (High)", "C (High)", "B (Low)", "D (Low)"],
                  ["A (High)", "C (High)"])

    plt.suptitle("Effect of Priority on Route Order", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_priority_routing.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_priority_routing.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 5: Demand Satisfaction / Service Equity illustration
# ══════════════════════════════════════════════════════════════
def fig_service_equity():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    locations = ['Loc 1', 'Loc 2', 'Loc 3', 'Loc 4', 'Loc 5']
    demand    = [10, 8, 6, 12, 4]

    # Solution A: maximise total satisfaction (ignores equity)
    supplied_a = [10, 8, 6, 6, 0]  # loc 5 gets nothing
    # Solution B: equitable distribution
    total_supply = sum(supplied_a)
    supplied_b = [int(d / sum(demand) * total_supply) for d in demand]
    # adjust to match total
    supplied_b[-1] = total_supply - sum(supplied_b[:-1])

    x = np.arange(len(locations))
    width = 0.35

    for ax, supplied, title, color in [
        (axes[0], supplied_a, "Solution A:\nMaximise Total Delivery", '#1565C0'),
        (axes[1], supplied_b, "Solution B:\nEquitable Distribution", '#2E7D32'),
    ]:
        bars_d = ax.bar(x - width/2, demand,   width, label='Demand',   color='#BDBDBD', edgecolor='black')
        bars_s = ax.bar(x + width/2, supplied, width, label='Supplied',  color=color,    edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, fontsize=9)
        ax.set_ylabel("Units", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 15)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars_s, supplied):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(val), ha='center', va='bottom', fontsize=8)

    plt.suptitle("Equity vs. Efficiency in Relief Distribution", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_service_equity.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_service_equity.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 6: Commercial VRP vs Disaster Relief VRP comparison
# ══════════════════════════════════════════════════════════════
def fig_commercial_vs_disaster():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')
    ax.set_title("Commercial VRP vs. Disaster Relief VRP: Key Differences",
                 fontsize=13, fontweight='bold', pad=15)

    headers = ["Dimension", "Commercial VRP", "Disaster Relief VRP"]
    rows = [
        ["Primary Objective",    "Minimise cost / distance",       "Minimise suffering / response time"],
        ["Fleet Availability",   "Known, controlled",              "Uncertain, may be damaged"],
        ["Demand",               "Known or forecast",              "Unknown / rapidly evolving"],
        ["Road Network",         "Reliable",                       "May be blocked or destroyed"],
        ["Time Horizon",         "Planned in advance",             "Real-time, dynamic re-routing"],
        ["Objectives",           "Single (cost)",                  "Multiple (equity, speed, coverage)"],
        ["Vehicle Type",         "Homogeneous fleets",             "Diverse (trucks, helicopters, boats)"],
        ["Priority",             "Not typically considered",       "Life-saving priority is critical"],
        ["Return to Depot",      "Required each trip",             "May not return; one-way routes"],
        ["Uncertainty",          "Low",                            "Extremely high"],
    ]

    col_widths = [0.26, 0.33, 0.38]
    col_x = [0.01, 0.28, 0.62]
    row_h = 0.085
    y_start = 0.92

    # Header row
    for j, (hdr, cx, cw) in enumerate(zip(headers, col_x, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch((cx, y_start - row_h), cw - 0.01, row_h,
                     boxstyle="round,pad=0.01", facecolor='#1565C0', edgecolor='white', lw=0.5,
                     transform=ax.transAxes))
        ax.text(cx + (cw-0.01)/2, y_start - row_h/2, hdr, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', transform=ax.transAxes)

    row_colors = ['#E3F2FD', '#FFFFFF']
    for i, row in enumerate(rows):
        y = y_start - (i + 2) * row_h
        bg = row_colors[i % 2]
        for j, (cell, cx, cw) in enumerate(zip(row, col_x, col_widths)):
            ax.add_patch(mpatches.FancyBboxPatch((cx, y), cw - 0.01, row_h,
                         boxstyle="round,pad=0.005", facecolor=bg, edgecolor='#BDBDBD', lw=0.4,
                         transform=ax.transAxes))
            text_color = '#1A237E' if j == 0 else ('#1B5E20' if j == 2 else '#1A237E')
            ax.text(cx + 0.01, y + row_h/2, cell, ha='left', va='center',
                    fontsize=8.2, color=text_color, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_commercial_vs_disaster.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_commercial_vs_disaster.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 7: TOPlike routing – priorities and arcs (analogue to Fig 14.5)
# ══════════════════════════════════════════════════════════════
def fig_top_routing():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Team Orienteering Problem:\nPriority-Based Route Selection",
                 fontsize=12, fontweight='bold')

    depot = (0.4, 2.0)
    end_depot = (5.0, 2.0)

    nodes = [
        # (x, y, priority, label)
        (1.5, 3.5, 5, "P=5"),
        (1.5, 0.5, 2, "P=2"),
        (2.5, 3.8, 4, "P=4"),
        (2.5, 2.0, 3, "P=3"),
        (2.5, 0.2, 1, "P=1"),
        (3.5, 3.5, 5, "P=5"),
        (3.5, 1.0, 2, "P=2"),
        (4.0, 2.5, 4, "P=4"),
    ]

    # Color by priority
    cmap = plt.cm.RdYlGn
    prios = [n[2] for n in nodes]
    max_p, min_p = max(prios), min(prios)

    for (x, y, p, lbl) in nodes:
        norm_p = (p - min_p) / (max_p - min_p + 1e-9)
        col = cmap(norm_p)
        c = plt.Circle((x, y), 0.28, color=col, zorder=3, edgecolor='black', lw=1)
        ax.add_patch(c)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)

    # Depot squares
    for dp, lbl_dp in [(depot, "Start\nDepot"), (end_depot, "End\nDepot")]:
        ax.add_patch(mpatches.FancyBboxPatch((dp[0]-0.22, dp[1]-0.22), 0.44, 0.44,
                     boxstyle="round,pad=0.04", facecolor='#37474F', edgecolor='black', lw=1.5, zorder=3))
        ax.text(dp[0], dp[1], lbl_dp, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='white', zorder=4)

    # Vehicle 1: high priority route
    route1 = [depot, (1.5, 3.5), (2.5, 3.8), (3.5, 3.5), (4.0, 2.5), end_depot]
    # Vehicle 2: medium priority
    route2 = [depot, (2.5, 2.0), (3.5, 1.0), end_depot]

    for route, col, lbl in [(route1, '#1565C0', 'Vehicle 1 (high-prio)'),
                            (route2, '#C62828', 'Vehicle 2 (med-prio)')]:
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        ax.plot(xs, ys, '-', color=col, lw=2, zorder=2, alpha=0.7, label=lbl)
        for k in range(len(route)-1):
            dx = route[k+1][0] - route[k][0]
            dy = route[k+1][1] - route[k][1]
            norm = np.sqrt(dx**2 + dy**2) + 1e-9
            ax.annotate("", xy=(route[k+1][0]-dx/norm*0.3, route[k+1][1]-dy/norm*0.3),
                        xytext=(route[k][0]+dx/norm*0.3, route[k][1]+dy/norm*0.3),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    ax.legend(loc='lower right', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_p, vmax=max_p))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Location Priority", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_top_routing.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_top_routing.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 8: Evacuation routing – contraflow illustration
# ══════════════════════════════════════════════════════════════
def fig_evacuation_contraflow():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def draw_network(ax, title, contraflow=False):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold')

        shelter = (5.0, 1.5)
        evac_zones = [(0.4, 0.4), (0.4, 1.5), (0.4, 2.8)]
        road_nodes = [(1.8, 0.4), (1.8, 1.5), (1.8, 2.8),
                      (3.2, 0.4), (3.2, 1.5), (3.2, 2.8)]

        # Draw shelter
        ax.add_patch(mpatches.FancyBboxPatch((shelter[0]-0.3, shelter[1]-0.25), 0.6, 0.5,
                     boxstyle="round,pad=0.05", facecolor='#2E7D32', edgecolor='black', lw=1.5))
        ax.text(shelter[0], shelter[1], "Shelter", ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')

        # Evacuation zones
        for i, (x, y) in enumerate(evac_zones):
            c = plt.Circle((x, y), 0.28, color='#D32F2F', zorder=3, alpha=0.85)
            ax.add_patch(c)
            ax.text(x, y, f"Zone\n{i+1}", ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=4)

        # Road nodes
        for (x, y) in road_nodes:
            c = plt.Circle((x, y), 0.12, color='#607D8B', zorder=3)
            ax.add_patch(c)

        # Edges
        edges = [
            (evac_zones[0], road_nodes[0]), (evac_zones[1], road_nodes[1]),
            (evac_zones[2], road_nodes[2]),
            (road_nodes[0], road_nodes[1]), (road_nodes[1], road_nodes[2]),
            (road_nodes[0], road_nodes[3]), (road_nodes[1], road_nodes[4]),
            (road_nodes[2], road_nodes[5]),
            (road_nodes[3], road_nodes[4]), (road_nodes[4], road_nodes[5]),
            (road_nodes[3], shelter), (road_nodes[4], shelter),
        ]

        evac_col = '#D32F2F' if contraflow else '#607D8B'
        relief_col = '#1565C0' if contraflow else '#607D8B'

        for (a, b) in edges:
            lw = 2 if contraflow else 1.5
            # In contraflow: outbound lanes become inbound (evacuation)
            if contraflow:
                color = evac_col
                ax.annotate("", xy=b, xytext=a,
                            arrowprops=dict(arrowstyle='->', color=color, lw=lw))
            else:
                ax.plot([a[0], b[0]], [a[1], b[1]], color='#90A4AE', lw=1.5, zorder=1)

        evac_patch = mpatches.Patch(color='#D32F2F', label='Evacuation flow')
        ax.legend(handles=[evac_patch], loc='upper left', fontsize=7)

    draw_network(axes[0], "Normal Road Network\n(Bidirectional)", contraflow=False)
    draw_network(axes[1], "Contraflow Network\n(All Lanes: Evacuation Direction)", contraflow=True)

    plt.suptitle("Evacuation Routing: Standard vs. Contraflow", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_evacuation_contraflow.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_evacuation_contraflow.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 9: Supply chain in humanitarian logistics
# ══════════════════════════════════════════════════════════════
def fig_supply_chain():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    ax.set_title("Humanitarian Logistics Supply Chain", fontsize=13, fontweight='bold')

    stages = [
        ("Donors /\nSuppliers", '#1565C0'),
        ("Pre-positioned\nWarehouses", '#6A1B9A'),
        ("Local\nDistribution\nPoints", '#E65100'),
        ("Beneficiaries\n(Affected\nPeople)", '#1B5E20'),
    ]

    y_center = 0.5
    x_positions = [0.1, 0.32, 0.58, 0.82]
    box_w, box_h = 0.16, 0.35

    for (lbl, col), xc in zip(stages, x_positions):
        ax.add_patch(mpatches.FancyBboxPatch((xc - box_w/2, y_center - box_h/2),
                     box_w, box_h, boxstyle="round,pad=0.02",
                     facecolor=col, edgecolor='black', lw=1.5, transform=ax.transAxes))
        ax.text(xc, y_center, lbl, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='white', transform=ax.transAxes, multialignment='center')

    # Arrows
    for i in range(len(x_positions) - 1):
        x_start = x_positions[i] + box_w/2
        x_end   = x_positions[i+1] - box_w/2
        ax.annotate("", xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5),
                    xycoords='axes fraction', textcoords='axes fraction')

    # Labels on arrows
    arrow_labels = ["Procurement\n& Shipping", "Pre-disaster\nStocking", "Last-Mile\nDelivery"]
    mid_xs = [(x_positions[i] + x_positions[i+1])/2 for i in range(len(x_positions)-1)]
    for lbl, xm in zip(arrow_labels, mid_xs):
        ax.text(xm, y_center + 0.25, lbl, ha='center', va='center', fontsize=8.5,
                color='#333333', transform=ax.transAxes, style='italic', multialignment='center')

    ax.text(0.5, 0.06,
            "Last-mile delivery (depot → beneficiaries) is the primary VRP application in disaster relief.",
            ha='center', va='center', fontsize=8.5, color='#333333',
            transform=ax.transAxes, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_supply_chain.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_supply_chain.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 10: Crop book figures (14.1, 14.2, etc.) from PDF
# Book page 415 (0-indexed: 414 for p415) contains Figure 14.2
# ══════════════════════════════════════════════════════════════
def crop_book_figures():
    # Page numbers are 1-indexed in the book; PyMuPDF uses 0-indexed
    # Figure 14.2 is on book page 415 (PDF page index = 415-1 = 414)
    # Let's check and crop the main figures
    figures_to_crop = [
        # (pdf_page_0indexed, (x0,y0,x1,y1 in pts), output_name)
        # Figure 14.2 (Network flow model) – p416 of book (0-indexed 415)
        (415, (50, 370, 500, 640), "fig_book_14_2_network.png"),
        # Figure 14.3 (priority-weighted routes) – p424 of book (0-indexed 423)
        (422, (50, 350, 500, 620), "fig_book_14_3_priority.png"),
        # Figure 14.4 (local training) – p423 (0-indexed 422)
        (423, (50, 350, 500, 650), "fig_book_14_4_local_training.png"),
        # Figure 14.5 (demand satisfaction) – p425 (0-indexed 424)
        (424, (50, 350, 500, 640), "fig_book_14_5_demand_sat.png"),
        # Figure 14.6 (De Angelis VRP) – p426 (0-indexed 425)
        (425, (50, 380, 520, 640), "fig_book_14_6_vrp.png"),
        # Figure 14.7 (full vs short run) – p427 (0-indexed 426)
        (426, (50, 300, 520, 660), "fig_book_14_7_fullshort.png"),
        # Figure 14.8 (arrival validation) – p427 (0-indexed 426)
        (426, (50, 100, 520, 320), "fig_book_14_8_arrival.png"),
        # Figure 14.9 (phantom copies) – p428 (0-indexed 427)
        (427, (50, 380, 520, 660), "fig_book_14_9_phantom.png"),
        # Figure 14.10 (equity supply) – p429 (0-indexed 428)
        (428, (50, 350, 520, 660), "fig_book_14_10_equity.png"),
        # Figure 14.11 (equity split) – p430 (0-indexed 429)
        (429, (50, 380, 520, 680), "fig_book_14_11_equity_split.png"),
        # Figure 14.12 (free carrier) – p431 (0-indexed 430)
        (430, (50, 380, 520, 660), "fig_book_14_12_free_carrier.png"),
        # Figure 14.13 (max-tour-min-max) – p432 (0-indexed 431)
        (431, (50, 380, 520, 680), "fig_book_14_13_mintour.png"),
    ]

    doc = fitz.open(BOOK_PDF)
    n_pages = len(doc)
    doc.close()
    print(f"  PDF has {n_pages} pages total.")

    for (pidx, rect, out_name) in figures_to_crop:
        if pidx >= n_pages:
            print(f"  Skipping {out_name}: page {pidx} out of range ({n_pages} pages).")
            continue
        try:
            crop_from_pdf(pidx, rect, out_name, dpi=200)
        except Exception as e:
            print(f"  Warning: could not crop {out_name}: {e}")


# ══════════════════════════════════════════════════════════════
# Figure 11: Min-max tour length comparison
# ══════════════════════════════════════════════════════════════
def fig_minmax_tour():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    np.random.seed(42)
    n_loc = 10
    locs_x = np.random.uniform(0.5, 4.5, n_loc)
    locs_y = np.random.uniform(0.5, 4.5, n_loc)
    depot = np.array([0.3, 2.5])

    def tour_len(order, locs_x, locs_y, depot):
        total = 0
        x0, y0 = depot
        for i in order:
            total += np.sqrt((locs_x[i]-x0)**2 + (locs_y[i]-y0)**2)
            x0, y0 = locs_x[i], locs_y[i]
        total += np.sqrt((x0-depot[0])**2 + (y0-depot[1])**2)
        return total

    # Two routes: unbalanced vs balanced
    route_a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    route_b1 = [0, 2, 4, 6, 8]
    route_b2 = [1, 3, 5, 7, 9]

    for ax, title, routes, colors in [
        (axes[0], "One Vehicle\n(Long Tour)", [route_a], ['#D32F2F']),
        (axes[1], "Two Vehicles\n(Balanced – Shorter Max Tour)", [route_b1, route_b2], ['#1565C0', '#2E7D32'])
    ]:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        # Depot
        ax.plot(*depot, 's', color='black', ms=12, zorder=5)
        ax.text(depot[0], depot[1]+0.2, "Depot", ha='center', va='bottom', fontsize=8)
        # Locations
        ax.scatter(locs_x, locs_y, color='gray', s=60, zorder=4)
        for i in range(n_loc):
            ax.text(locs_x[i]+0.12, locs_y[i]+0.12, str(i+1), fontsize=7, color='#333333')
        # Routes
        for route, col in zip(routes, colors):
            path = [depot] + [(locs_x[i], locs_y[i]) for i in route] + [depot.tolist()]
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            tlen = tour_len(route, locs_x, locs_y, depot)
            ax.plot(xs, ys, '-', color=col, lw=2, alpha=0.8, label=f"Len={tlen:.1f}")
        ax.legend(fontsize=8)
        ax.axis('off')

    plt.suptitle("Min-Max Tour Length: Balancing Vehicle Workloads", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_minmax_tour.pdf"), bbox_inches='tight')
    plt.close()
    print("  Saved fig_minmax_tour.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures for Chapter 14...")

    print("\n[1] Disaster management phases cycle...")
    fig_disaster_phases()

    print("\n[2] Network flow model...")
    fig_network_flow()

    print("\n[3] Response time vs. coverage...")
    fig_response_coverage()

    print("\n[4] Priority routing...")
    fig_priority_routing()

    print("\n[5] Service equity...")
    fig_service_equity()

    print("\n[6] Commercial vs. Disaster VRP comparison table...")
    fig_commercial_vs_disaster()

    print("\n[7] Team Orienteering Problem routing...")
    fig_top_routing()

    print("\n[8] Evacuation contraflow...")
    fig_evacuation_contraflow()

    print("\n[9] Humanitarian supply chain...")
    fig_supply_chain()

    print("\n[10] Min-max tour length...")
    fig_minmax_tour()

    print("\n[11] Cropping figures from book PDF...")
    crop_book_figures()

    print("\nAll figures generated.")
