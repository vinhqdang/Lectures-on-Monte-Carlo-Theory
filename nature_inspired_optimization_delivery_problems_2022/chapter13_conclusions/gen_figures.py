"""
gen_figures.py  —  Chapter 13: Conclusions and Future Developments
Nature Inspired Optimisation for Delivery Problems (2022)

Generates all figures needed by chapter13_slides.tex.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Figure 1: Software Strategy Comparison
# ─────────────────────────────────────────────────────────────────
def fig_software_strategies():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    strategies = [
        ("Custom Solver\nper Variant", 1.5, 2.5, '#4472C4',
         "Ideal quality\nHigh cost\nNot scalable"),
        ("Complex Universal\nSolver", 5.0, 2.5, '#ED7D31',
         "Technically feasible\nHard to use\nBloated code"),
        ("Adapt Problem\nto Existing Solver", 8.5, 2.5, '#A9D18E',
         "Low cost\nUnsatisfactory\nStifles practice"),
    ]

    for label, x, y, color, notes in strategies:
        box = mpatches.FancyBboxPatch(
            (x - 1.2, y - 1.0), 2.4, 2.0,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor='#333333', facecolor=color, alpha=0.85, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.55, label, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=4, wrap=True)
        ax.text(x, y - 0.35, notes, ha='center', va='center',
                fontsize=7.5, color='#222222', zorder=4, linespacing=1.5)

    # Arrow pointing down to "None is ideal"
    ax.text(5.0, 0.35, 'None of these approaches is ideal — AGI / Domain-Specific AI is the future',
            ha='center', va='center', fontsize=9, style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2CC', edgecolor='#CCCC00'))

    ax.set_title('Three Strategies for Building Delivery Optimisation Solvers',
                 fontsize=11, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'software_strategies.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'software_strategies.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved software_strategies')


# ─────────────────────────────────────────────────────────────────
# Figure 2: AI Evolution Spectrum
# ─────────────────────────────────────────────────────────────────
def fig_ai_evolution():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Arrow baseline
    ax.annotate('', xy=(9.5, 1.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(5.0, 0.15, 'Increasing intelligence and autonomy →',
            ha='center', fontsize=9, color='#555555', style='italic')

    milestones = [
        (1.5, 'Rule-based\nSolvers\n(current)', '#4472C4'),
        (4.0, 'Domain-Specific\nAI\n(near future)', '#ED7D31'),
        (7.0, 'AGI\n(Artificial General\nIntelligence)', '#A9D18E'),
    ]

    for x, label, color in milestones:
        ax.plot(x, 1.5, 'o', markersize=18, color=color,
                markeredgecolor='#333333', markeredgewidth=1.5, zorder=4)
        ax.text(x, 2.55, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                          edgecolor='#333333', alpha=0.7))

    ax.set_title('The Road from Today\'s Solvers to Artificial General Intelligence (AGI)',
                 fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ai_evolution.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'ai_evolution.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved ai_evolution')


# ─────────────────────────────────────────────────────────────────
# Figure 3: Domain-Specific AI Architecture
# ─────────────────────────────────────────────────────────────────
def fig_domain_specific_ai():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    boxes = [
        # (x_center, y_center, width, height, label, color)
        (5.0, 4.8, 8.0, 0.8, 'User describes the problem (natural language / examples)', '#D9E1F2'),
        (5.0, 3.7, 8.0, 0.8, 'Problem Taxonomy — recognises & categorises characteristics\n(TSP-like? Capacitated? Time-windowed?)', '#FCE4D6'),
        (5.0, 2.5, 8.0, 0.8, 'Algorithm Library — assembles appropriate solvers\n(from building blocks)', '#E2EFDA'),
        (5.0, 1.3, 8.0, 0.8, 'Illumination / Archive — diverse archive of solutions', '#FFF2CC'),
        (5.0, 0.2, 8.0, 0.7, 'Machine Learning — learns user preferences, highlights best solutions', '#FCE4D6'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.08', linewidth=1.2,
            edgecolor='#444444', facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8.5, zorder=4, linespacing=1.4)

    # Arrows between boxes
    for y_from, y_to in [(4.4, 4.1), (4.1-0.8+0.4, 3.7-0.4+0.1),
                         (3.3, 2.9), (2.1, 1.7), (0.9, 0.55)]:
        ax.annotate('', xy=(5.0, y_to), xytext=(5.0, y_from),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    ax.set_title('Architecture of a Domain-Specific AI Solver for Delivery Problems',
                 fontsize=11, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'domain_specific_ai.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'domain_specific_ai.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved domain_specific_ai')


# ─────────────────────────────────────────────────────────────────
# Figure 4: Transportation Modes Comparison
# ─────────────────────────────────────────────────────────────────
def fig_transport_modes():
    modes = ['Traditional\nVehicle', 'Electric\nVehicle', 'Bicycle\nCourier',
             'Walking\nCourier', 'Drone', 'Public\nTransport\n(Tram/Bus)']
    speed     = [8, 7, 5, 2, 6, 4]
    payload   = [10, 9, 3, 1, 2, 7]
    carbon    = [10, 4, 1, 0, 2, 2]   # lower is better; reversed later
    autonomy  = [1, 3, 1, 1, 8, 2]

    x = np.arange(len(modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_s = ax.bar(x - 1.5*width, speed,    width, label='Speed (relative)', color='#4472C4', alpha=0.85)
    bars_p = ax.bar(x - 0.5*width, payload,  width, label='Payload capacity', color='#ED7D31', alpha=0.85)
    bars_c = ax.bar(x + 0.5*width, carbon,   width, label='Carbon emissions (lower=better)', color='#FF0000', alpha=0.70)
    bars_a = ax.bar(x + 1.5*width, autonomy, width, label='Autonomy level',  color='#A9D18E', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9)
    ax.set_ylabel('Score (relative, 1–10 scale)', fontsize=9)
    ax.set_title('Comparative Profile of Delivery Transport Modes\n'
                 '(all scores are relative/illustrative)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'transport_modes.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'transport_modes.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved transport_modes')


# ─────────────────────────────────────────────────────────────────
# Figure 5: Drone Fleet Swarm Intelligence Diagram
# ─────────────────────────────────────────────────────────────────
def fig_drone_swarm():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 7))

    # City grid background
    for xi in range(0, 11, 2):
        ax.axvline(xi, color='#cccccc', lw=0.8, zorder=1)
    for yi in range(0, 11, 2):
        ax.axhline(yi, color='#cccccc', lw=0.8, zorder=1)

    # Depots
    depots = [(1, 1), (9, 1), (5, 9)]
    for dx, dy in depots:
        ax.plot(dx, dy, 's', markersize=16, color='#4472C4',
                markeredgecolor='#333333', markeredgewidth=1.5, zorder=5)
        ax.text(dx, dy - 0.6, 'Depot', ha='center', fontsize=7.5,
                color='#4472C4', fontweight='bold')

    # Drones
    org_colors = ['#ED7D31', '#A9D18E', '#FF0000']
    org_labels = ['Org A', 'Org B', 'Org C']
    for oi, (col, lbl) in enumerate(zip(org_colors, org_labels)):
        n_drones = 4
        for j in range(n_drones):
            dx = np.random.uniform(1, 9)
            dy = np.random.uniform(2, 9)
            ax.plot(dx, dy, '^', markersize=12, color=col,
                    markeredgecolor='#333333', markeredgewidth=1, zorder=4,
                    label=lbl if j == 0 else '')
            # Collision avoidance bubble
            circle = plt.Circle((dx, dy), 0.5, color=col,
                                 alpha=0.12, zorder=3)
            ax.add_patch(circle)

    # Central coordinator label
    ax.text(5, 5, 'Swarm\nCoordinator', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF2CC',
                      edgecolor='#CCCC00', alpha=0.9), zorder=6)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_xlabel('City East–West (km)', fontsize=9)
    ax.set_ylabel('City North–South (km)', fontsize=9)
    ax.set_title('Swarm Intelligence Coordinates Multi-Organisation Drone Fleets\n'
                 '(collision avoidance bubbles shown)',
                 fontsize=10, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'drone_swarm.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'drone_swarm.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved drone_swarm')


# ─────────────────────────────────────────────────────────────────
# Figure 6: Public Transport Integration (Tram + Micro-depot)
# ─────────────────────────────────────────────────────────────────
def fig_public_transport():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Tram line
    ax.plot([1, 11], [3.5, 3.5], '-', lw=5, color='#4472C4', zorder=2)
    ax.plot([1, 11], [3.3, 3.3], '-', lw=2, color='#cccccc', zorder=3)
    ax.text(6, 4.0, 'Tram / Bus Line', ha='center', fontsize=9,
            color='#4472C4', fontweight='bold')

    # Tram stop / micro-depot nodes
    stops = [2.0, 5.0, 8.0, 11.0]
    stop_labels = ['Depot\n(Distribution\nCentre)', 'Micro-Depot\nA', 'Micro-Depot\nB', 'Micro-Depot\nC']
    stop_colors = ['#4472C4', '#ED7D31', '#ED7D31', '#ED7D31']
    for sx, lbl, col in zip(stops, stop_labels, stop_colors):
        ax.plot(sx, 3.5, 'o', markersize=18, color=col,
                markeredgecolor='#333333', markeredgewidth=1.5, zorder=4)
        ax.text(sx, 2.75, lbl, ha='center', va='top', fontsize=8,
                linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#F2F2F2',
                          edgecolor='#aaaaaa', alpha=0.8))

    # Last-mile delivery arrows from micro-depots
    for sx in [5.0, 8.0, 11.0]:
        for dy in [1.5, 0.8]:
            ax.annotate('', xy=(sx + np.random.uniform(-0.8, 0.8), dy),
                        xytext=(sx, 2.55),
                        arrowprops=dict(arrowstyle='->', color='#A9D18E', lw=1.5))
    ax.text(6, 0.5, 'Last-mile delivery by bicycle / walking courier',
            ha='center', fontsize=8.5, color='#2E7D32', style='italic')

    ax.set_title('Public Transport as a Freight Carrier: Tram + Micro-Depot Network',
                 fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'public_transport.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'public_transport.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved public_transport')


# ─────────────────────────────────────────────────────────────────
# Figure 7: Consumer Trends Driving Future Logistics
# ─────────────────────────────────────────────────────────────────
def fig_consumer_trends():
    labels = ['Home Delivery\nDemand', 'Climate Change\nAwareness',
              'Sustainability\nExpectation', 'Online Shopping\nGrowth',
              'Cooperation\nBetween Orgs']
    values_2020 = [55, 45, 40, 60, 20]
    values_2022 = [80, 70, 65, 85, 35]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, values_2020, width,
                   label='Pre-COVID (2019)', color='#9DC3E6', alpha=0.9)
    bars2 = ax.bar(x + width/2, values_2022, width,
                   label='Post-COVID (2022)', color='#2E75B6', alpha=0.9)

    ax.set_ylabel('Relative Index (illustrative)', fontsize=9)
    ax.set_title('Key Consumer Trends Driving Future Delivery Optimisation',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f'{h}', ha='center', va='bottom', fontsize=8, color='#2E75B6')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'consumer_trends.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'consumer_trends.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved consumer_trends')


# ─────────────────────────────────────────────────────────────────
# Figure 8: Cooperation vs Competition in Deliveries
# ─────────────────────────────────────────────────────────────────
def fig_cooperation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: No cooperation — separate routes
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('No Cooperation\n(redundant routes, high emissions)', fontsize=10, fontweight='bold')

    np.random.seed(7)
    depots_xy = [(2, 8), (8, 8)]
    depot_colors = ['#4472C4', '#ED7D31']
    customers = [(np.random.uniform(1, 9), np.random.uniform(1, 7)) for _ in range(10)]

    for (dx, dy), col in zip(depots_xy, depot_colors):
        ax.plot(dx, dy, 's', markersize=14, color=col, zorder=5,
                markeredgecolor='#333333')
        # Each org serves all customers
        for cx, cy in customers[:5] if col == '#4472C4' else customers[5:]:
            ax.plot([dx, cx], [dy, cy], '-', color=col, alpha=0.6, lw=1.2)
            ax.plot(cx, cy, 'o', markersize=7, color=col, alpha=0.7,
                    markeredgecolor='#333333', markeredgewidth=0.5)

    ax.set_xlabel('City grid', fontsize=8)
    ax.set_ylabel('City grid', fontsize=8)

    # Right: Cooperation — shared micro-depot
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Cooperation via Shared Micro-Depot\n(fewer vehicles, lower emissions)', fontsize=10, fontweight='bold')

    # Shared micro-depot at centre
    ax2.plot(5, 5, 'D', markersize=16, color='#A9D18E', zorder=5,
             markeredgecolor='#333333', markeredgewidth=1.5)
    ax2.text(5, 4.3, 'Shared\nMicro-Depot', ha='center', fontsize=7.5,
             color='#2E7D32', fontweight='bold')

    for (dx, dy), col in zip(depots_xy, depot_colors):
        ax2.plot(dx, dy, 's', markersize=14, color=col, zorder=5,
                 markeredgecolor='#333333')
        ax2.plot([dx, 5], [dy, 5], '--', color=col, alpha=0.8, lw=1.5)

    all_customers = customers
    for i, (cx, cy) in enumerate(all_customers):
        ax2.plot([5, cx], [5, cy], '-', color='#A9D18E', alpha=0.7, lw=1.2)
        ax2.plot(cx, cy, 'o', markersize=7, color='#555555', alpha=0.8,
                 markeredgecolor='#333333', markeredgewidth=0.5)

    ax2.set_xlabel('City grid', fontsize=8)

    plt.suptitle('Organisational Cooperation Reduces Total Distance and Emissions',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'cooperation.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'cooperation.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved cooperation')


# ─────────────────────────────────────────────────────────────────
# Figure 9: Summary Roadmap — Book Themes
# ─────────────────────────────────────────────────────────────────
def fig_book_summary():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    themes = [
        (1.5, 3.5, '#4472C4', 'Routing\nProblems\n(VRP, TSP,\nCVRP)', ''),
        (3.8, 3.5, '#ED7D31', 'Nature-Inspired\nAlgorithms\n(EA, ACO,\nPSO, SA)', ''),
        (6.1, 3.5, '#A9D18E', 'Multi-Modal\n& Green\nDelivery', ''),
        (8.4, 3.5, '#FFC000', 'Illumination\n& MAP-Elites\nArchive', ''),
        (10.7, 3.5, '#FF0000', 'Future:\nAGI &\nAutonomy', ''),
    ]

    for x, y, col, lbl, _ in themes:
        patch = mpatches.FancyBboxPatch(
            (x - 1.1, y - 0.9), 2.2, 1.8,
            boxstyle='round,pad=0.1', linewidth=1.5,
            edgecolor='#333333', facecolor=col, alpha=0.82, zorder=3)
        ax.add_patch(patch)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=8.5, fontweight='bold', zorder=4, linespacing=1.4,
                color='white' if col in ['#4472C4', '#FF0000', '#ED7D31'] else '#222222')

    # Connecting arrows
    for x1, x2 in [(2.6, 2.7), (4.9, 5.0), (7.2, 7.3), (9.5, 9.6)]:
        ax.annotate('', xy=(x2, 3.5), xytext=(x1, 3.5),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    ax.text(6.0, 1.2,
            'Across all chapters, the central challenge is the same:\n'
            'deliver more goods, with fewer emissions, at lower cost — '
            'using intelligent algorithms.',
            ha='center', va='center', fontsize=9.5, style='italic',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF3FB',
                      edgecolor='#4472C4', alpha=0.9))

    ax.set_title('Book Summary: Themes Covered Across 13 Chapters',
                 fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'book_summary.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(OUT, 'book_summary.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved book_summary')


if __name__ == '__main__':
    fig_software_strategies()
    fig_ai_evolution()
    fig_domain_specific_ai()
    fig_transport_modes()
    fig_drone_swarm()
    fig_public_transport()
    fig_consumer_trends()
    fig_cooperation()
    fig_book_summary()
    print('\nAll figures generated successfully.')
