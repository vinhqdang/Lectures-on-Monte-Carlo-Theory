"""
gen_figures.py  –  Generate all figures for Chapter 24 (Multidisciplinary Optimization)
Algorithms for Optimization, 2nd ed., Kochenderfer & Wheeler (2026)

Run with:
  conda run -n py313 python3 gen_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  saved: {path}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: MDO general problem structure diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_mdo_structure():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    def box(cx, cy, w, h, label, color):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.15', facecolor=color,
                              edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    box(2.0, 4.0, 2.8, 0.8, 'Optimizer', '#3B528B')
    box(7.5, 4.0, 2.8, 0.8, 'Multidisciplinary\nAnalysis', '#21908C')
    box(2.0, 1.5, 2.8, 0.8, 'minimize  $f(\\mathcal{A})$', '#3B528B')
    box(7.5, 1.5, 2.8, 0.8, '$\\mathcal{A} = [\\mathbf{x}, \\mathbf{y}^{(1)}, \\ldots, \\mathbf{y}^{(m)}]$', '#21908C')

    arrow(3.4, 4.0, 6.1, 4.0)
    arrow(6.1, 3.7, 3.4, 3.7)
    ax.text(4.75, 4.15, '$\\mathbf{x}$', ha='center', fontsize=11)
    ax.text(4.75, 3.55, '$\\mathbf{y}$', ha='center', fontsize=11)

    ax.text(5.0, 2.8, 'Design variables $\\mathbf{x} \\in \\mathbb{R}^n$',
            ha='center', fontsize=9, color='#333')
    ax.text(5.0, 2.4, 'Response variables $\\mathbf{y}^{(i)} \\in \\mathbb{R}^{n_i}$',
            ha='center', fontsize=9, color='#333')

    ax.set_title('MDO General Problem Structure', fontsize=12, fontweight='bold', pad=8)
    savefig('fig_mdo_structure.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Gauss-Seidel convergence example (Example 24.2)
# Three disciplinary analyses with x=1, different orderings
# ─────────────────────────────────────────────────────────────────────────────
def gauss_seidel_ordering(order, n_iter=20):
    """Run Gauss-Seidel with given ordering of [F1, F2, F3]."""
    x = 1.0
    y = np.ones(3)   # y1, y2, y3
    history = [y.copy()]
    for _ in range(n_iter):
        for idx in order:
            if idx == 0:   # F1: y1 = y2 - x
                y[0] = y[1] - x
            elif idx == 1: # F2: y2 = sin(y1 + y3)
                y[1] = np.sin(y[0] + y[2])
            else:           # F3: y3 = cos(x + y2 + y1)
                y[2] = np.cos(x + y[1] + y[0])
        history.append(y.copy())
    return np.array(history)


def fig_gauss_seidel_convergence():
    hist_conv = gauss_seidel_ordering([0, 1, 2])   # F1, F2, F3 – converges
    hist_div  = gauss_seidel_ordering([0, 2, 1])   # F1, F3, F2 – diverges

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    iters = np.arange(len(hist_conv))
    colors = ['#5B2C8D', '#1ABC9C', '#F39C12']
    labels = ['$y^{(1)}$', '$y^{(2)}$', '$y^{(3)}$']

    for ax, hist, title in zip(axes,
                               [hist_conv, hist_div],
                               ['Ordering $F_1, F_2, F_3$ (converges)',
                                'Ordering $F_1, F_3, F_2$ (diverges)']):
        for j in range(3):
            ax.plot(iters, hist[:, j], marker='o', markersize=4,
                    color=colors[j], label=labels[j], linewidth=1.5)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Response value', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='#aaa', linewidth=0.8, linestyle='--')

    fig.suptitle('Gauss-Seidel MDA — Effect of Disciplinary Ordering', fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig('fig_gauss_seidel.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Dependency graph for ride-sharing example
# ─────────────────────────────────────────────────────────────────────────────
def fig_dependency_graph():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.axis('off')

    nodes = {
        'Vehicle\nAnalysis':  (1.5, 6.0),
        'Sensor\nAnalysis':   (3.5, 5.0),
        'Autonomy\nAnalysis': (5.0, 4.0),
        'Routing\nAnalysis':  (5.0, 2.8),
        'Demand\nAnalysis':   (5.0, 1.6),
        'Profit\nAnalysis':   (6.3, 0.5),
    }
    colors = {
        'Vehicle\nAnalysis':  '#E74C3C',
        'Sensor\nAnalysis':   '#3498DB',
        'Autonomy\nAnalysis': '#2ECC71',
        'Routing\nAnalysis':  '#2ECC71',
        'Demand\nAnalysis':   '#F39C12',
        'Profit\nAnalysis':   '#E74C3C',
    }
    edges = [
        ('Vehicle\nAnalysis', 'Sensor\nAnalysis'),
        ('Sensor\nAnalysis',  'Vehicle\nAnalysis'),
        ('Sensor\nAnalysis',  'Autonomy\nAnalysis'),
        ('Autonomy\nAnalysis','Routing\nAnalysis'),
        ('Routing\nAnalysis', 'Demand\nAnalysis'),
        ('Demand\nAnalysis',  'Routing\nAnalysis'),
        ('Demand\nAnalysis',  'Profit\nAnalysis'),
        ('Autonomy\nAnalysis','Profit\nAnalysis'),
    ]

    for (src, dst) in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5,
                                   connectionstyle='arc3,rad=0.15'))

    for name, (cx, cy) in nodes.items():
        rect = FancyBboxPatch((cx - 0.9, cy - 0.4), 1.8, 0.8,
                              boxstyle='round,pad=0.1',
                              facecolor=colors[name], edgecolor='#222', linewidth=1.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(cx, cy, name, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')

    ax.set_title('Ride-Sharing Problem: Dependency Graph', fontsize=11, fontweight='bold')
    savefig('fig_dependency_graph.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: MDF architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_mdf_architecture():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis('off')

    def box(cx, cy, w, h, txt, fc, ec='#333', fs=9):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2, lbl='', color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if lbl:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 0.15
            ax.text(mx, my, lbl, ha='center', fontsize=8, color='#555')

    # System optimizer
    box(2.5, 4.8, 3.2, 0.8, 'System Optimizer\n(minimize $f(\\mathcal{A})$)', '#3B528B')
    # MDA block
    box(7.0, 4.8, 3.2, 0.8, 'Multidisciplinary\nAnalysis (MDA)', '#21908C')
    # Sub-discipline blocks
    box(7.0, 3.2, 2.0, 0.7, 'Discipline 1', '#5D8AA8')
    box(7.0, 2.2, 2.0, 0.7, 'Discipline 2', '#5D8AA8')
    ax.text(7.0, 1.55, '$\\vdots$', ha='center', fontsize=13)
    box(7.0, 1.0, 2.0, 0.7, 'Discipline $m$', '#5D8AA8')

    arrow(4.1, 4.8, 5.4, 4.8, '$\\mathbf{x}$')
    arrow(5.4, 4.5, 4.1, 4.5, '$\\mathbf{y}$')

    # MDA to disciplines
    for cy in [3.2, 2.2, 1.0]:
        ax.plot([7.0, 7.0], [4.4, cy+0.35], color='#21908C', lw=1.2)

    ax.set_title('Multidisciplinary Design Feasible (MDF) Architecture', fontsize=11, fontweight='bold')
    savefig('fig_mdf_architecture.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Sequential optimization architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_sequential_architecture():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.axis('off')

    def box(cx, cy, w, h, txt, fc, fs=8.5):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor='#333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2, lbl=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
        if lbl:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.12, lbl, fontsize=7.5, color='#555')

    # Top-level optimizer
    box(3.5, 7.0, 3.5, 0.75, 'minimize $f(\\mathcal{A})$ s.t. $\\mathcal{A}\\in\\mathcal{X}$\n(over $\\mathbf{x}_g$)', '#3B528B')
    arrow(3.5, 6.62, 3.5, 6.12, 'update $\\mathbf{x}_g$')

    subdiscipline_colors = ['#8E44AD', '#2471A3', '#1A8754', '#A04000']
    labels = ['minimize $f_1(\\mathcal{A})$ s.t. $[\\mathbf{x}^{(1)},\\mathbf{y}^{(1)}]\\in\\mathcal{X}_1$',
              'minimize $f_2(\\mathcal{A})$ s.t. $[\\mathbf{x}^{(2)},\\mathbf{y}^{(2)}]\\in\\mathcal{X}_2$',
              '$\\vdots$',
              'minimize $f_m(\\mathcal{A})$ s.t. $[\\mathbf{x}^{(m)},\\mathbf{y}^{(m)}]\\in\\mathcal{X}_m$']
    da_labels = ['Disciplinary\nAnalysis 1', 'Disciplinary\nAnalysis 2', None, 'Disciplinary\nAnalysis $m$']
    ys = [5.5, 4.2, 3.3, 2.2]

    prev_y = 6.12
    for i, (lbl, da, cy, col) in enumerate(zip(labels, da_labels, ys, subdiscipline_colors)):
        if lbl == '$\\vdots$':
            ax.text(3.5, cy, '$\\vdots$', ha='center', va='center', fontsize=14)
            continue
        # Blue box for sub-optimizer
        rect = FancyBboxPatch((0.3, cy-0.42), 5.8, 0.84,
                              boxstyle='round,pad=0.1',
                              facecolor=col, edgecolor='#222', linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(2.85, cy, lbl, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color='white')
        # DA box
        box(7.8, cy, 2.2, 0.72, da, '#457B9D', fs=8)
        # arrows
        arrow(6.1, cy+0.15, 7.0, cy+0.15, 'update $\\mathbf{x}_\\ell^{(%d)}$' % (i+1) if i < 2 else '')
        arrow(7.0, cy-0.15, 6.1, cy-0.15, 'update $\\mathbf{y}^{(%d)}$' % (i+1) if i < 2 else '')
        if prev_y < cy - 0.42 + 0.84:
            arrow(3.5, prev_y, 3.5, cy+0.42)
        prev_y = cy - 0.42

    # Back arrow
    ax.annotate('', xy=(0.3, 6.62), xytext=(0.3, 2.2-0.42),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                                connectionstyle='angle,angleA=0,angleB=90'))
    ax.set_title('Sequential Optimization Architecture', fontsize=11, fontweight='bold')
    savefig('fig_sequential_architecture.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: IDF architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_idf_architecture():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis('off')

    def box(cx, cy, w, h, txt, fc, fs=9):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor='#333', linewidth=1.4)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', wrap=True)

    def arrow(x1, y1, x2, y2, lbl=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
        if lbl:
            ax.text((x1+x2)/2+0.05, (y1+y2)/2+0.12, lbl, fontsize=7.5, color='#555')

    # System optimizer at top
    box(4.5, 6.0, 7.5, 0.85,
        'System Optimizer: minimize $f(\\mathbf{x}, \\mathbf{c})$ s.t. $[\\mathbf{x},\\mathbf{c}]\\in\\mathcal{X}$, $\\mathbf{c}=[\\mathbf{y}^{(1)},\\ldots,\\mathbf{y}^{(m)}]$',
        '#3B528B', fs=8)

    da_colors = ['#8E44AD', '#2471A3', '#1A8754', '#A04000', '#21908C', '#C0392B']
    da_names  = ['Vehicle\nAnalysis', 'Sensor\nAnalysis', 'Autonomy\nAnalysis',
                 'Routing\nAnalysis', 'Demand\nAnalysis', 'Profit\nAnalysis']

    xs = [0.8, 2.5, 4.2, 5.9, 7.6, 9.3]
    ys_top = 5.57
    ys_box = 2.5
    ys_bot = 5.15

    for i, (x, name, col) in enumerate(zip(xs, da_names, da_colors)):
        # Arrow down from optimizer
        arrow(x, ys_top, x, ys_box + 0.42, 'copy $\\mathcal{A}$' if i == 0 else '')
        # DA box
        box(x, ys_box, 1.4, 0.72, name, col, fs=7.5)
        # Arrow up to optimizer
        arrow(x, ys_box - 0.42, x, ys_bot, 'update $\\mathbf{y}^{(%d)}$' % (i+1) if i == 0 else '')

    ax.text(4.5, 1.4, 'All disciplinary analyses receive same $\\mathcal{A}$; run in parallel', ha='center', fontsize=9, style='italic')
    ax.text(4.5, 1.0, 'Coupling constraint: $\\mathbf{c}^{(i)} = \\mathbf{y}^{(i)}$ enforced by optimizer', ha='center', fontsize=9)

    ax.set_title('Individual Discipline Feasible (IDF) Architecture', fontsize=11, fontweight='bold')
    savefig('fig_idf_architecture.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Collaborative Optimization (CO) architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_co_architecture():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

    def box(cx, cy, w, h, txt, fc, fs=8.5):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor='#333', linewidth=1.4)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2, lbl=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
        if lbl:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.12, lbl, fontsize=7.5, color='#555')

    # System optimizer
    box(5.0, 6.5, 8.5, 0.82,
        'System: min $f(\\mathcal{A}_g)$ s.t. $J^{(i)}(\\mathcal{A}_g)\\leq\\delta$  for all $i=1,\\ldots,m$',
        '#3B528B', fs=8.5)

    colors = ['#8E44AD', '#2471A3', '#1A8754']
    xs = [1.8, 5.0, 8.2]
    subtitles = ['min $J^{(1)}$\ns.t. $[\\mathbf{x}^{(1)},\\mathbf{y}^{(1)}]\\in\\mathcal{X}_1$',
                 'min $J^{(2)}$\ns.t. $[\\mathbf{x}^{(2)},\\mathbf{y}^{(2)}]\\in\\mathcal{X}_2$',
                 'min $J^{(m)}$\ns.t. $[\\mathbf{x}^{(m)},\\mathbf{y}^{(m)}]\\in\\mathcal{X}_m$']
    da_names = ['Analysis 1', 'Analysis 2', 'Analysis $m$']

    for i, (x, sub, da, col) in enumerate(zip(xs, subtitles, da_names, colors)):
        arrow(x, 6.09, x, 5.0 + 0.38)
        box(x, 5.0, 2.3, 0.72, sub, col, fs=7.5)
        arrow(x, 4.62, x, 3.7+0.38, 'update $\\mathbf{x}_\\ell$' if i == 0 else '')
        box(x, 3.7, 1.8, 0.65, da, '#457B9D', fs=8)
        arrow(x, 3.37, x, 6.09)
        ax.text(x, 6.0+0.18, '$\\mathcal{A}_g$', ha='center', fontsize=8, color='#555')

    ax.text(5.0, 2.8, '$J^{(i)} = \\sum_v (x_v^g - \\mathcal{A}_v^{(i)})^2 + \\sum_v (y_v^g - \\mathcal{A}_v^{(i)})^2$',
            ha='center', fontsize=9)
    ax.text(5.0, 2.3, 'Coupling: each subproblem drives local values to match system-level targets',
            ha='center', fontsize=8.5, style='italic')

    ax.set_title('Collaborative Optimization (CO) Architecture', fontsize=11, fontweight='bold')
    savefig('fig_co_architecture.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: SAND architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_sand_architecture():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis('off')

    def box(cx, cy, w, h, txt, fc, fs=9):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor='#333', linewidth=1.4)
        ax.add_patch(rect)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2, lbl=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
        if lbl:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.12, lbl, fontsize=7.5, color='#555')

    # Large single optimizer
    box(5.0, 4.5, 8.5, 0.85,
        'SAND Optimizer: min $f(\\mathbf{x},\\mathbf{y})$ s.t. $R_i(\\mathbf{x},\\mathbf{y})=\\mathbf{0}$, $[\\mathbf{x},\\mathbf{y}]\\in\\mathcal{X}$',
        '#3B528B', fs=8.5)

    colors = ['#8E44AD', '#2471A3', '#1A8754', '#A04000']
    xs = [1.5, 4.0, 6.5, 9.0]
    lbls = ['$R_1 = 0$', '$R_2 = 0$', '$R_3 = 0$', '$R_m = 0$']

    for i, (x, lbl, col) in enumerate(zip(xs, lbls, colors)):
        arrow(x, 4.07, x, 3.2+0.35)
        box(x, 3.2, 1.6, 0.65, lbl, col, fs=8.5)
        ax.text(x, 2.7, f'Residual\nDiscipline {i+1}', ha='center', va='top', fontsize=7.5)

    ax.text(5.0, 1.9,
            'All design $\\mathbf{x}$ and response $\\mathbf{y}$ variables passed simultaneously to optimizer',
            ha='center', fontsize=9)
    ax.text(5.0, 1.45,
            'Residuals $R_i(\\mathbf{x},\\mathbf{y}) = \\mathbf{0}$ enforced as equality constraints',
            ha='center', fontsize=9)
    ax.text(5.0, 1.0,
            'Enables gradient-based methods; disciplinary analyses run once per iteration',
            ha='center', fontsize=9, style='italic', color='#555')

    ax.set_title('Simultaneous Analysis and Design (SAND) Architecture', fontsize=11, fontweight='bold')
    savefig('fig_sand_architecture.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Architecture comparison table (visual)
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture_comparison():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')

    cols = ['Architecture', 'Optimizer\nLevel', 'MDA\nRequired', 'Parallel\nDisciplines',
            'Variable\nCount', 'Notes']
    rows = [
        ['MDF', 'Single', 'Yes (inner loop)', 'No', 'Low ($n$)', 'Simple; MDA cost dominates'],
        ['Sequential', 'Hierarchical', 'Per discipline', 'No', 'Low–Med', 'Good for loosely coupled'],
        ['IDF', 'Single', 'No', 'Yes', 'High ($n + \\sum n_i$)', 'Coupling as constraint'],
        ['CO', 'Bi-level', 'No', 'Yes', 'High', 'Each discipline independent'],
        ['SAND', 'Single (large)', 'No (residuals)', 'Yes', 'Very high', 'Needs analytic gradients'],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#3B528B')
        table[0, j].set_text_props(color='white', fontweight='bold')

    row_colors = ['#EAF2FB', '#FDFEFE', '#EAF2FB', '#FDFEFE', '#EAF2FB']
    for i, rc in enumerate(row_colors, 1):
        for j in range(len(cols)):
            table[i, j].set_facecolor(rc)

    ax.set_title('MDO Architecture Comparison', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    savefig('fig_architecture_comparison.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Spring-pendulum numerical example
# ─────────────────────────────────────────────────────────────────────────────
def fig_spring_pendulum():
    from scipy.optimize import minimize_scalar

    m, ell, g = 1.0, 1.0, 9.81
    theta_max = np.deg2rad(10.0)

    def mda(k):
        M = m * g * ell   # loads analysis: M = mgl*cos(theta), simplified to mgl
        theta = M / k     # displacement analysis
        converged = True
        return theta, M, converged

    ks = np.linspace(1, 200, 500)
    thetas = np.array([mda(k)[0] for k in ks])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: theta vs k
    ax = axes[0]
    ax.plot(ks, np.rad2deg(thetas), color='#3B528B', lw=2)
    ax.axhline(np.rad2deg(theta_max), color='#E74C3C', lw=1.5, linestyle='--', label='$\\theta_{\\max}=10^\\circ$')
    k_opt = m * g * ell / theta_max
    ax.axvline(k_opt, color='#21908C', lw=1.5, linestyle=':', label=f'$k^*={k_opt:.1f}$ N/m')
    ax.set_xlabel('Spring stiffness $k$ (N/m)', fontsize=10)
    ax.set_ylabel('Displacement $\\theta$ (degrees)', fontsize=10)
    ax.set_title('Displacement vs. Spring Stiffness', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: pendulum schematic
    ax2 = axes[1]
    ax2.set_xlim(-0.2, 1.5); ax2.set_ylim(-1.3, 0.3); ax2.set_aspect('equal')
    ax2.axis('off')
    theta_draw = np.deg2rad(20)
    lx = np.sin(theta_draw); ly = -np.cos(theta_draw)
    ax2.plot([0, lx], [0, ly], 'k-', lw=2)
    circle = plt.Circle((lx, ly), 0.07, color='#3B528B')
    ax2.add_patch(circle)
    ax2.plot([0], [0], 'ko', markersize=8)
    ax2.annotate('', xy=(lx, ly-0.3), xytext=(lx, ly),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    ax2.text(lx+0.08, ly-0.15, '$mg$', fontsize=9, color='#E74C3C')
    ax2.annotate('', xy=(-0.15, 0.15), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#1A8754', lw=1.5))
    ax2.text(-0.3, 0.18, '$k\\theta$', fontsize=9, color='#1A8754')
    ax2.text(0.25, -0.05, '$\\ell$', fontsize=10)
    theta_arc = np.linspace(np.pi/2+0.1, np.pi/2+theta_draw, 30)
    ax2.plot(0.35*np.cos(theta_arc), 0.35*np.sin(theta_arc), 'gray', lw=1)
    ax2.text(0.25, -0.32, '$\\theta$', fontsize=9, color='gray')
    ax2.set_title('Spring-Pendulum Schematic', fontsize=10)
    ax2.text(0.65, 0.22, f'$m={m}$ kg, $\\ell={ell}$ m\n$g={g}$ m/s$^2$\n$\\theta_{{\\max}}={np.rad2deg(theta_max):.0f}^\\circ$\n$k^*\\approx {k_opt:.1f}$ N/m',
             transform=ax2.transData, fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='#EAF2FB', alpha=0.8))

    fig.suptitle('Spring-Pendulum MDF Example (Exercise 24.4)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig('fig_spring_pendulum.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: CO subproblem coupling objective
# ─────────────────────────────────────────────────────────────────────────────
def fig_co_coupling():
    fig, ax = plt.subplots(figsize=(7, 4))
    theta_g_vals = np.linspace(-1.5, 3.5, 300)

    # J_loads for spring pendulum: J = (theta_g - theta)^2 + (F_loads(theta_g) - M)^2
    # Illustrative
    theta_true = 0.5
    M_true = 3.0
    def F_loads(tg): return 4.0 * tg + 1.0  # dummy linear model
    def J_loads(tg): return (tg - theta_true)**2 + (F_loads(tg) - M_true)**2

    J = np.array([J_loads(t) for t in theta_g_vals])
    ax.plot(theta_g_vals, J, color='#3B528B', lw=2, label='$J_{\\mathrm{loads}}(\\theta_g)$')
    idx = np.argmin(J)
    ax.axvline(theta_g_vals[idx], color='#E74C3C', lw=1.5, linestyle='--',
               label=f'Optimum $\\theta_g^*={theta_g_vals[idx]:.2f}$')
    ax.set_xlabel('$\\theta_g$ (coupling variable)', fontsize=10)
    ax.set_ylabel('$J_{\\mathrm{loads}}$', fontsize=10)
    ax.set_title('CO Subproblem: Coupling Objective vs. Global Variable', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('fig_co_coupling.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures for Chapter 24 ...')
    fig_mdo_structure()
    fig_gauss_seidel_convergence()
    fig_dependency_graph()
    fig_mdf_architecture()
    fig_sequential_architecture()
    fig_idf_architecture()
    fig_co_architecture()
    fig_sand_architecture()
    fig_architecture_comparison()
    fig_spring_pendulum()
    fig_co_coupling()
    print('Done. All figures saved to', FIGDIR)
