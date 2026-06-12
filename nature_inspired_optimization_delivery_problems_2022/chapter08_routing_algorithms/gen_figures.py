"""
gen_figures.py  --  Generate all figures for Chapter 8: Routing Algorithms
Book: Nature Inspired Optimisation for Delivery Problems (2022)
Run: conda run -n py313 python3 gen_figures.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import networkx as nx

# output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name, dpi=150):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved: {name}")

# -------------------------------------------------------------------------
# Figure 1: Example weighted graph (nodes a-f, as in book Fig 8.1)
# -------------------------------------------------------------------------
def fig_example_graph():
    pos = {
        'a': (2.0, 3.0),
        'b': (3.5, 3.0),
        'c': (3.5, 1.5),
        'd': (0.8, 1.8),
        'e': (0.5, 0.5),
        'f': (2.0, 0.2),
    }
    edges = [
        ('a','b',10), ('a','d',25),
        ('b','c',12), ('b','d',16),
        ('c','d',16), ('c','f',15),
        ('d','e',4),  ('e','f',5),
    ]
    G = nx.Graph()
    for u,v,w in edges:
        G.add_edge(u, v, weight=w)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_aspect('equal')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white',
                           edgecolors='#2b5fa0', linewidths=2, node_size=600)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='#555555')
    edge_labels = {(u,v):w for u,v,w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=10, bbox=dict(fc='white', ec='none', pad=1))
    ax.set_title("Example weighted graph (nodes a-f)", fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    savefig("fig01_example_graph.pdf")


# -------------------------------------------------------------------------
# Figure 2: Dijkstra step-by-step (6 steps)
# -------------------------------------------------------------------------
def fig_dijkstra_steps():
    pos = {
        'a': (2.0, 3.0),
        'b': (3.5, 3.0),
        'c': (3.5, 1.5),
        'd': (0.8, 1.8),
        'e': (0.5, 0.5),
        'f': (2.0, 0.2),
    }
    edges = [
        ('a','b',10), ('a','d',25),
        ('b','c',12), ('b','d',16),
        ('c','d',16), ('c','f',15),
        ('d','e',4),  ('e','f',5),
    ]
    INF = float('inf')

    steps = [
        dict(current='a',
             dists={'a':0,'b':INF,'c':INF,'d':INF,'e':INF,'f':INF},
             prev={'a':None,'b':None,'c':None,'d':None,'e':None,'f':None},
             unvisited='[bcdef]', title='Step 1: current=a'),
        dict(current='b',
             dists={'a':0,'b':10,'c':INF,'d':25,'e':INF,'f':INF},
             prev={'a':None,'b':'a','c':None,'d':'a','e':None,'f':None},
             unvisited='[cdef]', title='Step 2: current=b'),
        dict(current='c',
             dists={'a':0,'b':10,'c':22,'d':25,'e':INF,'f':INF},
             prev={'a':None,'b':'a','c':'b','d':'a','e':None,'f':None},
             unvisited='[def]', title='Step 3: current=c'),
        dict(current='d',
             dists={'a':0,'b':10,'c':22,'d':25,'e':29,'f':37},
             prev={'a':None,'b':'a','c':'b','d':'a','e':'d','f':'c'},
             unvisited='[ef]', title='Step 4: current=d'),
        dict(current='e',
             dists={'a':0,'b':10,'c':22,'d':25,'e':29,'f':34},
             prev={'a':None,'b':'a','c':'b','d':'a','e':'d','f':'e'},
             unvisited='[f]', title='Step 5: current=e'),
        dict(current='f',
             dists={'a':0,'b':10,'c':22,'d':25,'e':29,'f':34},
             prev={'a':None,'b':'a','c':'b','d':'a','e':'d','f':'e'},
             unvisited='[]', title='Step 6: current=f'),
    ]

    G = nx.Graph()
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)
    edge_labels = {(u,v):w for u,v,w in edges}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, step in enumerate(steps):
        ax = axes[idx]
        cur = step['current']
        dists = step['dists']
        prev = step['prev']

        node_colors = []
        for n in sorted(G.nodes()):
            if n == cur:
                node_colors.append('#f0a500')
            else:
                node_colors.append('#d0e8ff')

        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(G.nodes()),
                               node_color=node_colors,
                               edgecolors='#2b5fa0', linewidths=1.5, node_size=700)
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5,
                               edge_color='#777777', alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                      font_size=8,
                                      bbox=dict(fc='white', ec='none', pad=0))

        custom_labels = {}
        for n in G.nodes():
            d = dists[n]
            p = prev[n] if prev[n] else 'null'
            ds = str(d) if d != INF else 'inf'
            custom_labels[n] = f"{n}\np={p}\nd={ds}"
        nx.draw_networkx_labels(G, pos, labels=custom_labels, ax=ax,
                                font_size=7.5)

        ax.set_title(f"{step['title']}\nUnvisited={step['unvisited']}",
                     fontsize=9, pad=4)
        ax.axis('off')

    plt.suptitle("Dijkstra's Algorithm -- Step-by-Step Trace (Start=a, End=f)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("fig02_dijkstra_steps.pdf")


# -------------------------------------------------------------------------
# Figure 3: Dijkstra final route A->C
# -------------------------------------------------------------------------
def fig_dijkstra_route():
    pos = {
        'a': (2.0, 3.0),
        'b': (3.5, 3.0),
        'c': (3.5, 1.5),
        'd': (0.8, 1.8),
        'e': (0.5, 0.5),
        'f': (2.0, 0.2),
    }
    edges = [
        ('a','b',10), ('a','d',25),
        ('b','c',12), ('b','d',16),
        ('c','d',16), ('c','f',15),
        ('d','e',4),  ('e','f',5),
    ]
    route_edges = [('a','b'), ('b','c')]
    dists = {'a':0,'b':10,'c':22,'d':25,'e':29,'f':34}
    prev  = {'a':None,'b':'a','c':'b','d':'a','e':'d','f':'e'}

    G = nx.Graph()
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)
    edge_labels = {(u,v):w for u,v,w in edges}

    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_aspect('equal')

    edge_colors = ['#e03030' if (u,v) in route_edges or (v,u) in route_edges
                   else '#aaaaaa' for u,v in G.edges()]
    edge_widths = [3.0 if (u,v) in route_edges or (v,u) in route_edges
                   else 1.5 for u,v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=['#f0a500' if n in ('a','b','c') else '#d0e8ff'
                                       for n in sorted(G.nodes())],
                           nodelist=sorted(G.nodes()),
                           edgecolors='#2b5fa0', linewidths=1.5, node_size=700)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=9, bbox=dict(fc='white', ec='none', pad=1))

    custom_labels = {}
    for n in G.nodes():
        d = dists[n]
        p = prev[n] if prev[n] else 'null'
        custom_labels[n] = f"{n}\np={p}\nd={d}"
    nx.draw_networkx_labels(G, pos, labels=custom_labels, ax=ax, font_size=8)

    ax.set_title("Route a->c found by Dijkstra\n(red edges: a->b->c, total dist=22)", fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    savefig("fig03_dijkstra_route.pdf")


# -------------------------------------------------------------------------
# Figure 4: A* step-by-step (4 steps, same graph)
# -------------------------------------------------------------------------
def fig_astar_steps():
    pos = {
        'a': (2.0, 3.0),
        'b': (3.5, 3.0),
        'c': (3.5, 1.5),
        'd': (0.8, 1.8),
        'e': (0.5, 0.5),
        'f': (2.0, 0.2),
    }
    edges = [
        ('a','b',10), ('a','d',25),
        ('b','c',12), ('b','d',16),
        ('c','d',16), ('c','f',15),
        ('d','e',4),  ('e','f',5),
    ]

    def de(n):
        cx, cy = pos['c']
        nx_, ny = pos[n]
        return round(math.hypot(cx-nx_, cy-ny)*10, 1)

    INF = float('inf')
    steps = [
        dict(current='a', openlist=['a'],
             dists={'a':0,'b':INF,'c':INF,'d':INF,'e':INF,'f':INF},
             prev={'a':None},
             title='Step 1: a added to open list\nopen={a:0}'),
        dict(current='b', openlist=['b','d'],
             dists={'a':0,'b':10,'c':INF,'d':25,'e':INF,'f':INF},
             prev={'a':None,'b':'a','d':'a'},
             title='Step 2: b selected (h=ds+de)\nopenlist ordered by h'),
        dict(current='d', openlist=['d','c'],
             dists={'a':0,'b':10,'c':22,'d':25,'e':INF,'f':INF},
             prev={'a':None,'b':'a','d':'a','c':'b'},
             title='Step 3: d is current node\nc and e added to open list'),
        dict(current='c', openlist=['c'],
             dists={'a':0,'b':10,'c':22,'d':25,'e':INF,'f':INF},
             prev={'a':None,'b':'a','d':'a','c':'b'},
             title='Step 4: finish c reached!\nTrace prev labels: c<-b<-a'),
    ]

    G = nx.Graph()
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)
    edge_labels = {(u,v):w for u,v,w in edges}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, step in enumerate(steps):
        ax = axes[idx]
        cur = step['current']
        open_nodes = step['openlist']
        dists = step['dists']
        prev  = step['prev']

        node_colors = []
        for n in sorted(G.nodes()):
            if n == cur:
                node_colors.append('#f0a500')
            elif n in open_nodes:
                node_colors.append('#90ee90')
            else:
                node_colors.append('#d0e8ff')

        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(G.nodes()),
                               node_color=node_colors,
                               edgecolors='#2b5fa0', linewidths=1.5, node_size=700)
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5,
                               edge_color='#777777', alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                      font_size=8,
                                      bbox=dict(fc='white', ec='none', pad=0))

        custom_labels = {}
        for n in G.nodes():
            d = dists.get(n, INF)
            p = prev.get(n, None)
            ps = p if p else 'null'
            ds = str(d) if d != INF else 'inf'
            custom_labels[n] = f"{n}\np={ps}\nd={ds}"
        nx.draw_networkx_labels(G, pos, labels=custom_labels, ax=ax, font_size=7.5)

        ax.set_title(step['title'], fontsize=9, pad=4)
        ax.axis('off')

    legend_handles = [
        mpatches.Patch(color='#f0a500', label='Current node'),
        mpatches.Patch(color='#90ee90', label='In open list'),
        mpatches.Patch(color='#d0e8ff', label='Not yet considered'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("A* Algorithm -- Step-by-Step Trace (Start=a, Finish=c, h=ds+de)",
                 fontsize=13)
    plt.tight_layout()
    savefig("fig04_astar_steps.pdf")


# -------------------------------------------------------------------------
# Figure 5: A* heuristic formula diagram
# -------------------------------------------------------------------------
def fig_heuristic_formula():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    def node(cx, cy, label, color='#d0e8ff'):
        c = plt.Circle((cx, cy), 0.4, color=color, zorder=3, linewidth=2,
                        fill=True, ec='#2b5fa0')
        ax.add_patch(c)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=11,
                fontweight='bold', zorder=4)

    node(1.5, 2, 'S', '#a8d8a8')
    node(5,   2, 'C', '#f0a500')
    node(8.5, 2, 'F', '#ffaaaa')

    ax.annotate('', xy=(4.55, 2), xytext=(1.95, 2),
                arrowprops=dict(arrowstyle='->', color='#2b5fa0', lw=2))
    ax.text(3.25, 2.35, r'$d_s$ = known distance from start', ha='center',
            fontsize=10, color='#2b5fa0')

    ax.annotate('', xy=(8.05, 2), xytext=(5.45, 2),
                arrowprops=dict(arrowstyle='->', color='#c03030',
                                linestyle='dashed', lw=2))
    ax.text(6.75, 2.35, r'$d_e$ = estimated distance to finish', ha='center',
            fontsize=10, color='#c03030')

    ax.text(5, 0.8,
            r'$h = d_s + d_e$   (A* selects node with smallest $h$)',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', fc='#fffde7', ec='#f0a500', lw=1.5))

    ax.set_title("A* Heuristic: combining known distance and estimated distance to finish",
                 fontsize=11)
    plt.tight_layout()
    savefig("fig05_astar_heuristic.pdf")


# -------------------------------------------------------------------------
# Figure 6: Performance comparison bar chart (Edinburgh)
# -------------------------------------------------------------------------
def fig_performance_comparison():
    algorithms = ['DijkstraFlood', 'Dijkstra', 'A*']
    mean_times = [7300, 4600, 270]
    std_times  = [600,  1500, 300]

    colors = ['#4e79a7', '#59a14f', '#e15759']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(algorithms, mean_times, yerr=std_times, capsize=6,
                  color=colors, edgecolor='white', linewidth=1.2,
                  error_kw=dict(ecolor='#444444', elinewidth=1.5))

    ax.set_ylabel('Mean execution time (ms)', fontsize=11)
    ax.set_title('Algorithm Performance -- Edinburgh Graph (63267 nodes)\n'
                 '28 test routes, average of 10 runs', fontsize=11)
    ax.set_ylim(0, 10000)
    ax.axhline(0, color='black', linewidth=0.8)

    for bar, val in zip(bars, mean_times):
        ax.text(bar.get_x() + bar.get_width()/2, val + 200,
                f'{val:,} ms', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig("fig06_performance_comparison.pdf")


# -------------------------------------------------------------------------
# Figure 7: Bidirectional search concept
# -------------------------------------------------------------------------
def fig_bidirectional():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, title, fwd_r, rev_r in [
        (axes[0], "Unidirectional Dijkstra\n(searches entire reachable graph)", 2.2, 0),
        (axes[1], "Bidirectional Dijkstra\n(two small frontiers meet in the middle)", 1.3, 1.3),
    ]:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
        ax.axis('off')

        start_circle = plt.Circle((-2, 0), fwd_r, color='#aec6e8', alpha=0.5, zorder=2)
        ax.add_patch(start_circle)
        ax.plot(-2, 0, 'o', ms=12, color='#2b5fa0', zorder=5)
        ax.text(-2, 0, 'S', ha='center', va='center', color='white',
                fontsize=11, fontweight='bold', zorder=6)

        finish_circle = plt.Circle((2, 0), rev_r, color='#f4a7a7', alpha=0.5, zorder=2)
        ax.add_patch(finish_circle)
        ax.plot(2, 0, 'o', ms=12, color='#c03030', zorder=5)
        ax.text(2, 0, 'F', ha='center', va='center', color='white',
                fontsize=11, fontweight='bold', zorder=6)

        ax.set_title(title, fontsize=10)

    plt.suptitle("Bidirectional Search: running two algorithms simultaneously from S and F",
                 fontsize=12)
    plt.tight_layout()
    savefig("fig07_bidirectional.pdf")


# -------------------------------------------------------------------------
# Figure 8: Hierarchical road network structure (3 tiers)
# -------------------------------------------------------------------------
def fig_hierarchy_structure():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    colors_band = ['#cce5ff', '#d4edda', '#fff3cd']
    labels_band = ['Graph 0 -- Motorway / Trunk (level 0, ~5% of roads)',
                   'Graph 1 -- Primary / Secondary / Tertiary (level 1)',
                   'Graph 2 -- Unclassified / Residential (level 2, majority)']
    for i, (col, lab) in enumerate(zip(colors_band, labels_band)):
        y_bot = i * 1.55 + 0.1
        rect = plt.Rectangle((0.2, y_bot), 9.6, 1.4, color=col, alpha=0.7, zorder=1)
        ax.add_patch(rect)
        ax.text(5, y_bot + 0.7, lab, ha='center', va='center', fontsize=10,
                fontweight='bold')

    path_x = [0.5, 1.2, 1.8, 2.4, 3.2, 4.0, 5.0, 6.0, 6.8, 7.6, 8.4, 9.2, 9.6]
    path_y = [0.5, 0.8, 1.7, 2.4, 3.0, 3.8, 4.0, 3.8, 3.0, 2.4, 1.7, 0.8, 0.5]
    ax.plot(path_x, path_y, 'o-', color='#d62728', lw=2.5, ms=5, zorder=3)
    ax.annotate('Start', xy=(0.5, 0.5), xytext=(0.5, -0.3), fontsize=9,
                ha='center', arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('End', xy=(9.6, 0.5), xytext=(9.6, -0.3), fontsize=9,
                ha='center', arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_title("Hierarchical Road Network: a journey traverses local roads -> motorway -> local roads",
                 fontsize=11, pad=18)
    plt.tight_layout()
    savefig("fig08_hierarchy_structure.pdf")


# -------------------------------------------------------------------------
# Figure 9: Hierarchy algorithm route assembly
# -------------------------------------------------------------------------
def fig_hierarchy_route_assembly():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4); ax.axis('off')

    segments = [
        dict(x1=0.5, x2=2.5, y=2.0, color='#59a14f', lw=3,
             label='routeSy\n(level 1 start leg)', yt=2.6),
        dict(x1=2.5, x2=6.5, y=2.0, color='#4e79a7', lw=4,
             label='routeMiddle\n(level 0 motorway)', yt=2.6),
        dict(x1=6.5, x2=8.5, y=2.0, color='#f28e2b', lw=3,
             label='routeEy\n(level 1 end leg)', yt=2.6),
    ]
    for seg in segments:
        ax.annotate('', xy=(seg['x2'], seg['y']), xytext=(seg['x1'], seg['y']),
                    arrowprops=dict(arrowstyle='->', color=seg['color'],
                                   lw=seg['lw']))
        mid = (seg['x1']+seg['x2'])/2
        ax.text(mid, seg['yt'], seg['label'], ha='center', fontsize=9,
                color=seg['color'], fontweight='bold')

    for xp, label in [(2.5,'UpLink\n(routeSx.end)'), (6.5,'UpLink\n(routeEx.end)')]:
        ax.plot(xp, 2.0, 's', ms=12, color='#e15759', zorder=5)
        ax.text(xp, 1.3, label, ha='center', fontsize=8.5, color='#e15759')

    ax.plot(0.5, 2.0, 'o', ms=12, color='#2b5fa0', zorder=5)
    ax.text(0.5, 1.3, 'Start\n(level 2)', ha='center', fontsize=9)
    ax.plot(8.5, 2.0, 'o', ms=12, color='#c03030', zorder=5)
    ax.text(8.5, 1.3, 'End\n(level 2)', ha='center', fontsize=9)

    ax.set_title("Hierarchy Algorithm: 3-segment route assembly\n"
                 "result = routeSy + routeMiddle + routeEy  (Algorithm 21)",
                 fontsize=11)
    plt.tight_layout()
    savefig("fig09_hierarchy_route_assembly.pdf")


# -------------------------------------------------------------------------
# Figure 10: Algorithm complexity comparison table
# -------------------------------------------------------------------------
def fig_complexity_table():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')

    columns = ['Algorithm', 'Finds\nOptimum?', 'Time\nComplexity',
               'Space\nComplexity', 'Key Characteristic']
    data = [
        ['DijkstraFlood', 'Yes', 'O((V+E) log V)', 'O(V)', 'Full shortest-path tree'],
        ['Dijkstra (mod.)', 'Yes', 'O((V+E) log V)', 'O(V)', 'Halts at finish node'],
        ['A*', 'Yes*', 'O(E)', 'O(V)', 'Heuristic reduces nodes visited'],
        ['Hierarchical', 'Approx.', 'O(log V) eff.', 'O(V) per level', 'Multi-graph, parallelisable'],
        ['BiDirectional', 'Yes', 'O((V+E)/2 log V)', 'O(V)', 'Two simultaneous searches'],
    ]

    col_widths = [0.18, 0.12, 0.18, 0.18, 0.34]

    table = ax.table(cellText=data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(len(columns)):
        table[0, j].set_facecolor('#4e79a7')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(data)+1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f0f4f8')

    ax.set_title("Routing Algorithm Comparison", fontsize=13, pad=20)
    plt.tight_layout()
    savefig("fig10_complexity_table.pdf")


# -------------------------------------------------------------------------
# Figure 11: Sirmione results (Table 8.1)
# -------------------------------------------------------------------------
def fig_sirmione_results():
    problem_ids = ['1','1R','2','2R','3','3R','4','4R','5','5R','6','6R']
    dijkstra_flood_times = [2,1,1,1,1,1,1,1,1,0,1,1]
    dijkstra_times       = [2,2,1,1,0,0,1,0,1,0,0,0]
    astar_times          = [5,1,1,0,1,0,1,0,0,0,0,0]

    x = np.arange(len(problem_ids))
    w = 0.28
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w, dijkstra_flood_times, w, label='DijkstraFlood', color='#4e79a7')
    ax.bar(x,     dijkstra_times,       w, label='Dijkstra',      color='#59a14f')
    ax.bar(x + w, astar_times,          w, label='A*',            color='#e15759')

    ax.set_xticks(x); ax.set_xticklabels(problem_ids)
    ax.set_xlabel('Problem ID', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Sirmione Graph (255 nodes) -- Algorithm Execution Times\n'
                 'All algorithms find identical distances', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig("fig11_sirmione_results.pdf")


# -------------------------------------------------------------------------
# Figure 12: Edinburgh execution time vs distance scatter
# -------------------------------------------------------------------------
def fig_edinburgh_scatter():
    distances = [4.57,6.64,2.07,3.13,1.29,4.89,1.69,1.94,3.47,5.73,
                 6.16,4.54,4.86,3.88]
    flood_t   = [6823,6869,7279,7113,7536,6713,6309,7726,7215,6813,
                 6640,6564,6792,7522]
    dijk_t    = [4763,6203,1259,2424, 554,5927, 925, 997,2698,6164,
                 6199,6098,6391,4846]
    astar_t   = [  94, 542,  14,  27,  10, 370,  15,   7, 122, 436,
                  500, 249, 461,  85]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, flood_t, color='#4e79a7', s=60,
               label='DijkstraFlood', zorder=3)
    ax.scatter(distances, dijk_t,  color='#59a14f', s=60,
               label='Dijkstra',      zorder=3)
    ax.scatter(distances, astar_t, color='#e15759', s=60,
               label='A*',            zorder=3)
    ax.set_xlabel('Route distance (km)', fontsize=11)
    ax.set_ylabel('Execution time (ms)', fontsize=11)
    ax.set_title('Edinburgh: Execution time vs Route distance\n'
                 'A* is 20-100x faster; DijkstraFlood is distance-independent',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig("fig12_edinburgh_scatter.pdf")


# -------------------------------------------------------------------------
# Figure 13: OD matrix concept
# -------------------------------------------------------------------------
def fig_od_matrix():
    n = 5
    labels = [f'N{i}' for i in range(n)]
    np.random.seed(42)
    mat = np.random.randint(1, 30, size=(n,n)).astype(float)
    np.fill_diagonal(mat, 0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Destination', fontsize=11)
    ax.set_ylabel('Origin', fontsize=11)
    for i in range(n):
        for j in range(n):
            val = int(mat[i,j])
            ax.text(j, i, str(val) if val > 0 else '-',
                    ha='center', va='center', fontsize=10,
                    color='white' if mat[i,j] > 20 else 'black')
    ax.set_title('Origin-Destination (OD) Matrix\n'
                 'DijkstraFlood computes all entries in one pass per origin',
                 fontsize=10)
    plt.colorbar(im, ax=ax, label='Distance (km)')
    plt.tight_layout()
    savefig("fig13_od_matrix.pdf")


# -------------------------------------------------------------------------
# Figure 09b: Graph sizes by hierarchy level (fig09_graph_sizes.pdf)
# -------------------------------------------------------------------------
def fig_graph_sizes():
    """Bar chart: number of nodes in each level of the hierarchical graph."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    levels = ['Full Graph\n(all roads)', 'Level 0\n(Motorway/Trunk)', 'Level 1\n(Primary-Tertiary)', 'Level 2\n(Unclassified/\nResidential)']
    # Approximate values based on the book (Scotland dataset, Table 8.5 context)
    node_counts = [542133, 27107, 108400, 406626]
    pcts = [100, 5, 20, 75]
    colors = ['#888888', '#4e79a7', '#f28e2b', '#59a14f']

    bars = ax.bar(levels, node_counts, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.55)
    ax.set_ylabel('Approximate node count', fontsize=11)
    ax.set_title('Hierarchical Graph: Node Counts per Level (Scotland road network)\n'
                 'Level 0 contains only ~5% of total nodes -- crucial for fast routing',
                 fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val, pct in zip(bars, node_counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 8000,
                f'{val:,}\n({pct}%)', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    plt.tight_layout()
    savefig("fig09_graph_sizes.pdf")


# -------------------------------------------------------------------------
# Figure 14: Crop from PDF (optional, requires PyMuPDF)
# -------------------------------------------------------------------------
def fig_crop_from_pdf():
    try:
        import fitz
    except ImportError:
        print("  [skip] PyMuPDF not installed -- skipping PDF crops")
        return

    pdf_path = os.path.join(
        SCRIPT_DIR, "..",
        "Nature Inspired Optimisation for Delivery Problems 2022.pdf"
    )
    if not os.path.exists(pdf_path):
        print(f"  [skip] PDF not found -- skipping PDF crops")
        return

    doc = fitz.open(pdf_path)
    crops = [
        ("fig_book_dijkstra_steps.png", 162, (0.05, 0.05, 0.95, 0.72)),
        ("fig_book_astar_steps.png",    166, (0.05, 0.05, 0.95, 0.90)),
        ("fig_book_hierarchy_journey.png", 182, (0.05, 0.05, 0.95, 0.55)),
    ]
    for name, page_idx, rect_frac in crops:
        if page_idx >= len(doc):
            print(f"  [skip] page {page_idx} out of range for {name}")
            continue
        page = doc[page_idx]
        pw, ph = page.rect.width, page.rect.height
        x0f, y0f, x1f, y1f = rect_frac
        clip = fitz.Rect(x0f*pw, y0f*ph, x1f*pw, y1f*ph)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out_path = os.path.join(FIG_DIR, name)
        pix.save(out_path)
        print(f"  saved: {name}")
    doc.close()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
    print("Generating figures for Chapter 8: Routing Algorithms")
    fig_example_graph()
    fig_dijkstra_steps()
    fig_dijkstra_route()
    fig_astar_steps()
    fig_heuristic_formula()
    fig_performance_comparison()
    fig_bidirectional()
    fig_hierarchy_structure()
    fig_hierarchy_route_assembly()
    fig_complexity_table()
    fig_sirmione_results()
    fig_edinburgh_scatter()
    fig_od_matrix()
    fig_graph_sizes()
    fig_crop_from_pdf()
    print("\nAll figures generated successfully.")
