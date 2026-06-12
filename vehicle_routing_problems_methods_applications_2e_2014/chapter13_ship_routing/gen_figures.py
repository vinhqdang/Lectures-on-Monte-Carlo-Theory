"""
gen_figures.py  –  Generate all figures for Chapter 13 Ship Routing slides.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os, sys

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name, dpi=150):
    # Always save as PNG regardless of extension passed
    name_png = os.path.splitext(name)[0] + '.png'
    path = os.path.join(OUTDIR, name_png)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {path}")

# ─────────────────────────────────────────────
# Fig 1: Shipping segment overview (pie chart)
# ─────────────────────────────────────────────
def fig_shipping_segments():
    fig, ax = plt.subplots(figsize=(7, 4))
    segments = ['Industrial\n(Captive fleet)', 'Liner\n(Fixed schedules)', 'Tramp\n(Charter / spot)']
    sizes = [28, 35, 37]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=segments,
                                      colors=colors, autopct='%1.0f%%',
                                      startangle=120, textprops={'fontsize': 11})
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title('Maritime Shipping Segments\n(approximate market share)', fontsize=13, fontweight='bold')
    savefig('shipping_segments.pdf')

# ─────────────────────────────────────────────
# Fig 2: Tramp problem – pickup-delivery network
# ─────────────────────────────────────────────
def fig_tramp_network():
    fig, ax = plt.subplots(figsize=(7, 5))
    # Nodes: origin 0, dest 7 (artificial), cargo pickups 1-4, deliveries 5-8
    nodes = {
        0:  (0.5, 2.5,  'Origin\n(depot)', '#2ca02c'),
        7:  (9.5, 2.5,  'Dest.\n(artificial)', '#d62728'),
        1:  (2.0, 4.0,  'P1', '#4C72B0'),
        2:  (2.5, 1.0,  'P2', '#4C72B0'),
        3:  (3.5, 3.5,  'P3', '#4C72B0'),
        4:  (4.0, 1.5,  'P4', '#4C72B0'),
        5:  (6.0, 4.0,  'D1', '#DD8452'),
        6:  (7.5, 1.0,  'D2', '#DD8452'),
        7:  (9.5, 2.5,  'Dest.', '#d62728'),
        8:  (6.5, 3.0,  'D3', '#DD8452'),
        9:  (7.0, 1.8,  'D4', '#DD8452'),
    }
    # Re-assign properly
    nodes = {
        'O':  (0.5, 2.5,  'Origin', '#2ca02c'),
        'D':  (9.5, 2.5,  'Dest.',  '#d62728'),
        'P1': (2.0, 4.2,  'P1',     '#4C72B0'),
        'P2': (2.5, 0.8,  'P2',     '#4C72B0'),
        'P3': (3.8, 3.8,  'P3',     '#4C72B0'),
        'P4': (4.2, 1.5,  'P4',     '#4C72B0'),
        'D1': (6.2, 4.2,  'D1',     '#DD8452'),
        'D2': (7.0, 0.8,  'D2',     '#DD8452'),
        'D3': (6.8, 3.0,  'D3',     '#DD8452'),
        'D4': (7.5, 1.8,  'D4',     '#DD8452'),
    }
    for key, (x, y, lbl, col) in nodes.items():
        ax.scatter(x, y, s=350, c=col, zorder=5)
        ax.text(x, y+0.28, lbl, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mandatory routes P->D
    pairs = [('P1','D1'), ('P2','D2'), ('P3','D3'), ('P4','D4')]
    for (p, d) in pairs:
        x0, y0 = nodes[p][0], nodes[p][1]
        x1, y1 = nodes[d][0], nodes[d][1]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.5, linestyle='dashed'))

    # Ship route: O -> P1 -> P3 -> D3 -> D1 -> D2 -> P2 -> P4 -> D4 -> D
    route = ['O','P1','P3','D3','D1','D2','P2','P4','D4','D']
    rx = [nodes[k][0] for k in route]
    ry = [nodes[k][1] for k in route]
    ax.plot(rx, ry, 'k-', lw=2, alpha=0.5, zorder=3)
    ax.plot(rx, ry, 'k>', markersize=6, alpha=0.5, zorder=4)

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 5.0)
    ax.axis('off')
    ax.set_title('Tramp Pickup-and-Delivery Network\n(blue = pickup nodes, orange = delivery nodes)', fontsize=11)

    # Legend
    blue_patch  = mpatches.Patch(color='#4C72B0', label='Pickup ports')
    orange_patch= mpatches.Patch(color='#DD8452', label='Delivery ports')
    ax.legend(handles=[blue_patch, orange_patch], loc='lower right', fontsize=9)
    savefig('tramp_network.pdf')

# ─────────────────────────────────────────────
# Fig 3: Split loads – two ships sharing one cargo
# ─────────────────────────────────────────────
def fig_split_loads():
    fig, ax = plt.subplots(figsize=(8, 4))
    # Timeline for two ships
    for ship_y, ship_id, col in [(3.0, 'Ship 1', '#4C72B0'), (1.5, 'Ship 2', '#DD8452')]:
        ax.text(-0.3, ship_y, ship_id, ha='right', va='center', fontsize=11, fontweight='bold')
        ax.hlines(ship_y, 0, 10, color='gray', lw=1, linestyle='--')

    # Ship 1: picks up 60% of cargo at P, delivers at D
    ax.annotate('', xy=(7, 3.0), xytext=(2, 3.0),
                arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=2.5))
    ax.text(4.5, 3.2, '60% of cargo', ha='center', fontsize=9, color='#4C72B0')

    # Ship 2: picks up remaining 40%
    ax.annotate('', xy=(7, 1.5), xytext=(3.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#DD8452', lw=2.5))
    ax.text(5.25, 1.7, '40% of cargo', ha='center', fontsize=9, color='#DD8452')

    # Pickup and delivery markers
    for x, lbl in [(2, 'Pickup\n(P)'), (3.5, 'Pickup\n(P)'), (7, 'Delivery\n(D)')]:
        ax.axvline(x, color='gray', lw=1, alpha=0.5)
        ax.text(x, 0.6, lbl, ha='center', va='bottom', fontsize=9)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0.2, 4.0)
    ax.axis('off')
    ax.set_title('Split-Load Delivery: One Cargo Transported by Two Ships', fontsize=11, fontweight='bold')
    savefig('split_loads.pdf')

# ─────────────────────────────────────────────
# Fig 4: Variable speed – fuel consumption curve
# ─────────────────────────────────────────────
def fig_variable_speed():
    fig, ax = plt.subplots(figsize=(7, 4))
    v = np.linspace(8, 24, 200)   # knots
    # Fuel consumption roughly cubic: F(v) = alpha * v^3
    alpha = 0.002
    F = alpha * v**3
    ax.plot(v, F, 'b-', lw=2.5, label=r'$F(v) \propto v^3$')

    # Mark design speed and slow-steaming
    for vs, label, col in [(14, 'Slow steaming\n(14 kn)', '#2ca02c'),
                           (18, 'Service speed\n(18 kn)', '#d62728')]:
        ax.axvline(vs, color=col, lw=1.5, linestyle='--')
        ax.text(vs, alpha*vs**3 + 0.5, label, ha='center', fontsize=9, color=col)

    ax.set_xlabel('Ship speed (knots)', fontsize=11)
    ax.set_ylabel('Fuel consumption rate (tonnes/day)', fontsize=11)
    ax.set_title('Fuel Consumption vs. Speed (cubic relationship)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(8, 24)
    ax.set_ylim(0, None)
    savefig('variable_speed.pdf')

# ─────────────────────────────────────────────
# Fig 5: MIR inventory levels at two ports
# ─────────────────────────────────────────────
def fig_inventory_levels():
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    T = np.linspace(0, 40, 400)

    # Port 4 – production port (inventory rises linearly, ship visits cause drops)
    inv4 = np.zeros_like(T)
    rate4 = 0.8   # production rate
    visits4 = [(10, -8), (22, -8), (34, -8)]   # (time, change)
    for t_i in range(len(T)):
        if T[t_i] == 0:
            inv4[t_i] = 10
        else:
            inv4[t_i] = inv4[t_i-1] + rate4*(T[t_i]-T[t_i-1])
            for tv, dv in visits4:
                if T[t_i-1] < tv <= T[t_i]:
                    inv4[t_i] += dv
    inv4 = np.clip(inv4, 1, 32)

    axes[0].plot(T, inv4, 'b-', lw=2)
    axes[0].axhline(32, color='red', lw=1.5, linestyle='--', label='Max level = 32')
    axes[0].axhline(1,  color='orange', lw=1.5, linestyle='--', label='Min level = 1')
    axes[0].set_ylabel('Inventory level', fontsize=10)
    axes[0].set_title('(a) Inventory level at port 4 (production port)', fontsize=10)
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_ylim(0, 36)
    for tv, _ in visits4:
        axes[0].axvline(tv, color='green', lw=1, linestyle=':', alpha=0.7)
    axes[0].text(1, 34, 'Ship visits', fontsize=8, color='green')

    # Port 3 – consumption port (inventory falls, ship visits cause rises)
    inv3 = np.zeros_like(T)
    rate3 = -0.6
    visits3_feasible   = [(8, 10), (20, 10), (32, 10)]
    visits3_infeasible = [(8, 10), (25, 10)]
    # feasible
    inv3f = np.zeros_like(T)
    for t_i in range(len(T)):
        if T[t_i] == 0:
            inv3f[t_i] = 20
        else:
            inv3f[t_i] = inv3f[t_i-1] + rate3*(T[t_i]-T[t_i-1])
            for tv, dv in visits3_feasible:
                if T[t_i-1] < tv <= T[t_i]:
                    inv3f[t_i] += dv
    inv3f = np.clip(inv3f, 1, 28)
    # infeasible
    inv3i = np.zeros_like(T)
    for t_i in range(len(T)):
        if T[t_i] == 0:
            inv3i[t_i] = 20
        else:
            inv3i[t_i] = inv3i[t_i-1] + rate3*(T[t_i]-T[t_i-1])
            for tv, dv in visits3_infeasible:
                if T[t_i-1] < tv <= T[t_i]:
                    inv3i[t_i] += dv

    axes[1].plot(T, inv3f, 'g-',  lw=2, label='Feasible (3 visits)')
    axes[1].plot(T, inv3i, 'r--', lw=2, label='Infeasible (2 visits)')
    axes[1].axhline(28, color='red',    lw=1.5, linestyle='--', label='Max = 28')
    axes[1].axhline(1,  color='orange', lw=1.5, linestyle='--', label='Min = 1')
    axes[1].set_xlabel('Time (days)', fontsize=10)
    axes[1].set_ylabel('Inventory level', fontsize=10)
    axes[1].set_title('(b) Inventory level at port 3 (consumption port)', fontsize=10)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_ylim(-2, 32)

    plt.tight_layout()
    savefig('inventory_levels.pdf')

# ─────────────────────────────────────────────
# Fig 6: MIR example routes and schedules
# ─────────────────────────────────────────────
def fig_mir_routes():
    fig, ax = plt.subplots(figsize=(9, 5))
    # Ports on y-axis, time on x-axis
    ports  = {1: 1, 2: 2, 3: 3, 4: 4}
    port_labels = {1: 'Port 1', 2: 'Port 2', 3: 'Port 3', 4: 'Port 4'}

    ax.set_yticks(list(ports.values()))
    ax.set_yticklabels([port_labels[k] for k in sorted(port_labels)], fontsize=10)
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_title('MIR Example: Routes and Schedules for Two Ships\n(numbers = quantity loaded/unloaded)', fontsize=11)

    # Ship 1 schedule
    s1 = [(0, 1, 'start'), (4, 2, '14'), (8, 4, '8'), (12, 3, '3'), (17, 2, '2'), (22, 4, '12'), (27, 3, '3'), (34, 2, '2')]
    xs1 = [e[0] for e in s1]
    ys1 = [ports[e[1]] for e in s1]
    ax.plot(xs1, ys1, 'b-o', lw=2, markersize=8, label='Ship 1', zorder=4)
    for (t, p, q) in s1:
        if q not in ('start',):
            ax.text(t, ports[p]+0.15, q, ha='center', fontsize=8, color='#4C72B0', fontweight='bold')

    # Ship 2 schedule
    s2 = [(0, 2, 'start'), (5, 1, ''), (10, 4, ''), (15, 3, ''), (22, 2, ''), (30, 4, ''), (38, 2, '')]
    xs2 = [e[0] for e in s2]
    ys2 = [ports[e[1]] for e in s2]
    ax.plot(xs2, ys2, 'r-s', lw=2, markersize=8, label='Ship 2', zorder=4)

    ax.set_xlim(-1, 42)
    ax.set_ylim(0.5, 4.8)
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    savefig('mir_routes.pdf')

# ─────────────────────────────────────────────
# Fig 7: Dynamic / stochastic – decision timeline
# ─────────────────────────────────────────────
def fig_dynamic_timeline():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 2)

    # Timeline arrow
    ax.annotate('', xy=(9.8, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(5, -0.5, 'Planning horizon', ha='center', fontsize=11)

    events = [
        (1.0, 'Initial\nplan', '#2ca02c'),
        (3.0, 'Weather\nupdate', '#d62728'),
        (5.0, 'New cargo\narrival', '#4C72B0'),
        (6.5, 'Port delay\n(stochastic)', '#FF7F0E'),
        (8.5, 'Re-optimize\n& execute', '#9467bd'),
    ]
    for x, lbl, col in events:
        ax.plot(x, 0.5, 'o', markersize=14, color=col, zorder=5)
        ax.text(x, 1.1, lbl, ha='center', va='bottom', fontsize=9, color=col, fontweight='bold')
        ax.vlines(x, 0.5, 1.0, color=col, lw=1.5)

    ax.set_title('Dynamic/Stochastic Ship Routing: Rolling Horizon Decision Sequence',
                 fontsize=11, fontweight='bold', y=0.95)
    savefig('dynamic_timeline.pdf')

# ─────────────────────────────────────────────
# Fig 8: Comparison chart – industrial vs liner vs tramp
# ─────────────────────────────────────────────
def fig_segment_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ['Fleet\ntype', 'Cargo\ntype', 'Schedule\nflexibility', 'Revenue\nmodel', 'Routing\nflexibility']
    x = np.arange(len(categories))
    width = 0.25

    # Score 1–5 across three segments
    industrial = [4, 3, 2, 2, 2]
    liner      = [3, 2, 1, 3, 1]
    tramp      = [2, 4, 5, 5, 5]

    ax.bar(x - width, industrial, width, label='Industrial',  color='#4C72B0', alpha=0.85)
    ax.bar(x,         liner,      width, label='Liner',       color='#DD8452', alpha=0.85)
    ax.bar(x + width, tramp,      width, label='Tramp',       color='#55A868', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Relative characteristic (1=low, 5=high)', fontsize=10)
    ax.set_title('Comparison of Maritime Shipping Segments', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 6)
    ax.grid(True, axis='y', alpha=0.3)
    savefig('segment_comparison.pdf')

# ─────────────────────────────────────────────
# Fig 9: Branch-and-price schematic
# ─────────────────────────────────────────────
def fig_branch_price():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title('Branch-and-Price Algorithm for Ship Routing', fontsize=12, fontweight='bold')

    def node_box(x, y, text, color='#4C72B0', width=2.2, height=0.7):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                              boxstyle='round,pad=0.08', facecolor=color,
                              edgecolor='black', alpha=0.85, zorder=5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold', zorder=6)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1+0.35), xytext=(x0, y0-0.35),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    node_box(5, 5.3, 'Root LP relaxation', color='#2ca02c')
    node_box(5, 4.1, 'Column generation\n(pricing sub-problem)', color='#4C72B0', height=0.85)
    node_box(5, 2.8, 'Branch on\nfractional x_ijk', color='#DD8452')
    node_box(2.5, 1.6, 'Left branch\n(x_ijk = 0)', color='#9467bd')
    node_box(7.5, 1.6, 'Right branch\n(x_ijk = 1)', color='#9467bd')
    node_box(5,   0.5, 'Integer solution\n(optimal route plan)', color='#d62728')

    arrow(5, 5.3, 5, 4.1)
    arrow(5, 4.1, 5, 2.8)
    ax.annotate('', xy=(2.5, 1.95), xytext=(4.0, 2.45),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7.5, 1.95), xytext=(6.0, 2.45),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(5, 0.85), xytext=(5, 1.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    savefig('branch_price.pdf')

# ─────────────────────────────────────────────
# Fig 10: Tabu search for tramp routing – convergence
# ─────────────────────────────────────────────
def fig_tabu_convergence():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(200)
    # Simulate TS: starts high, drops, oscillates, eventually improves
    best = 1000.0
    bests = []
    current = 1000.0
    for i in iters:
        delta = np.random.normal(-1.5, 8)
        current = max(600, current + delta)
        if current < best:
            best = current
        bests.append(best)
    current_vals = [max(600, 1000 + np.random.normal(-1.5*i, 15)) for i in range(200)]

    ax.plot(iters, current_vals, 'b-', alpha=0.4, lw=1, label='Current solution value')
    ax.plot(iters, bests, 'r-', lw=2.5, label='Best solution found')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Objective (profit)', fontsize=11)
    ax.set_title('Tabu Search Convergence – Tramp Ship Routing', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    savefig('tabu_convergence.pdf')

# ─────────────────────────────────────────────
# Fig 11: Supply chain integration (MIR concept)
# ─────────────────────────────────────────────
def fig_supply_chain():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title('Maritime Inventory Routing: Integrated Supply Chain', fontsize=11, fontweight='bold')

    components = [
        (1.0, 2.0, 'Production\nFacilities\n(supply ports)', '#4C72B0'),
        (4.0, 3.2, 'Fleet\nManagement\n(ships)', '#DD8452'),
        (4.0, 0.8, 'Inventory\nControl\n(port stocks)', '#55A868'),
        (7.5, 2.0, 'Demand\nSites\n(consumption)', '#9467bd'),
        (4.0, 2.0, 'Integrated\nMIR\nOptimizer', '#d62728'),
    ]
    for x, y, lbl, col in components:
        box = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1.0,
                              boxstyle='round,pad=0.1', facecolor=col,
                              edgecolor='white', alpha=0.9, zorder=5)
        ax.add_patch(box)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=6)

    arrows = [(1.9, 2.0, 3.1, 2.0), (4.0, 1.3, 4.0, 1.7),
              (4.0, 2.3, 4.0, 2.7), (4.9, 2.0, 6.6, 2.0)]
    for x0, y0, x1, y1 in arrows:
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    savefig('supply_chain.pdf')

# ─────────────────────────────────────────────
# Fig 12: Crop from book PDF – tramp example figure (Fig 13.1)
# ─────────────────────────────────────────────
def fig_crop_from_pdf():
    try:
        import fitz  # pymupdf
        pdf_path = ("/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/"
                    "vehicle_routing_problems_methods_applications_2e_2014/"
                    "Vehicle Routing_ Problems, Methods, and Applications, Second Edition 2014.pdf")
        doc = fitz.open(pdf_path)
        # Page 387 (0-indexed: 386+1 for cover etc. – actual index may vary, try index 386)
        # Fig 13.1 is on book page 387/388
        for page_idx, out_name, clip_frac in [
            (386, 'fig13_1_route_example.pdf', (0.05, 0.40, 0.95, 0.85)),
            (387, 'fig13_2_example_route.pdf', (0.05, 0.02, 0.95, 0.50)),
        ]:
            try:
                page = doc[page_idx]
                rect = page.rect
                clip = fitz.Rect(
                    rect.x0 + clip_frac[0]*rect.width,
                    rect.y0 + clip_frac[1]*rect.height,
                    rect.x0 + clip_frac[2]*rect.width,
                    rect.y0 + clip_frac[3]*rect.height,
                )
                mat = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=mat, clip=clip)
                out_path = os.path.join(OUTDIR, os.path.splitext(out_name)[0] + '.png')
                pix.save(out_path)
                print(f"  saved crop: {out_path}")
            except Exception as e:
                print(f"  WARNING: could not crop page {page_idx}: {e}")
        doc.close()
    except ImportError:
        print("  WARNING: pymupdf not available, skipping PDF crops")
    except Exception as e:
        print(f"  WARNING: PDF crop failed: {e}")

# ─────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures for Chapter 13...")
    fig_shipping_segments()
    fig_tramp_network()
    fig_split_loads()
    fig_variable_speed()
    fig_inventory_levels()
    fig_mir_routes()
    fig_dynamic_timeline()
    fig_segment_comparison()
    fig_branch_price()
    fig_tabu_convergence()
    fig_supply_chain()
    fig_crop_from_pdf()
    print("All figures done.")
