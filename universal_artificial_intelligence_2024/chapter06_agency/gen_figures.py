"""
Generate all figures used in chapter06_agency_slides.tex
(Chapter 6: Agency, "An Introduction to Universal Artificial Intelligence", 2024)

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Figure 6.1 : The cybernetic model (agent <-> environment interaction loop)
# ---------------------------------------------------------------------------
def fig_cybernetic_model():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Agent box
    agent_box = FancyBboxPatch((0.6, 1.0), 3.0, 2.2, boxstyle="round,pad=0.05,rounding_size=0.15",
                                linewidth=1.8, edgecolor='#2b3a67', facecolor='#dbe4f0')
    ax.add_patch(agent_box)
    ax.text(2.1, 2.1, "Agent", ha='center', va='center', fontsize=15, fontweight='bold', color='#2b3a67')

    # Environment box
    env_box = FancyBboxPatch((6.4, 1.0), 3.0, 2.2, boxstyle="round,pad=0.05,rounding_size=0.15",
                              linewidth=1.8, edgecolor='#6b3a2b', facecolor='#f0ded3')
    ax.add_patch(env_box)
    ax.text(7.9, 2.1, "Environment", ha='center', va='center', fontsize=15, fontweight='bold', color='#6b3a2b')

    # Top arrow: action a_t  (agent -> environment)
    ax.annotate("", xy=(6.35, 3.35), xytext=(3.65, 3.0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color='black',
                                connectionstyle="arc3,rad=0.35"))
    ax.text(5.0, 3.85, r"$a_t$", ha='center', va='center', fontsize=15)

    # Bottom arrows: percepts e_{t-1} (environment -> agent, already delivered) and e_t (new one)
    ax.annotate("", xy=(3.65, 1.35), xytext=(6.35, 1.15),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color='black',
                                connectionstyle="arc3,rad=0.35"))
    ax.text(5.0, 0.55, r"$e_t$", ha='center', va='center', fontsize=15)

    ax.text(4.35, 1.95, r"$e_{t-1}$", ha='center', va='center', fontsize=13, color='#555555')
    ax.plot([5.0, 5.0], [1.05, 2.55], linestyle=(0, (3, 3)), color='#777777', lw=1.2)

    # Thought bubbles
    ax.text(2.1, 4.55, r"$a_t \sim \pi(\cdot\mid a_1e_1\ldots a_{t-1}e_{t-1})$",
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2b3a67", lw=1.0))
    ax.text(7.9, 4.55, r"$e_t \sim \mu(\cdot\mid a_1e_1\ldots a_{t-1}e_{t-1}a_t)$",
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#6b3a2b", lw=1.0))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "cybernetic_model.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6.2 : Discount function gamma_t and normalizer Gamma_{t'}
# ---------------------------------------------------------------------------
def fig_discount_function():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    t = np.arange(0, 15)
    gamma = 0.82 ** t  # illustrative geometric-like decaying discount

    tprime = 4
    ax.bar(t, gamma, width=0.85, color='#cfcfcf', edgecolor='black', linewidth=0.8, zorder=2)
    for i in t:
        if i >= tprime:
            ax.bar(i, gamma[i], width=0.85, color='#9e9e9e', edgecolor='black', linewidth=0.8, zorder=3)

    tt = np.linspace(0, 14, 400)
    ax.plot(tt, 0.82 ** tt, color='black', lw=1.8, zorder=4)

    ax.axvline(tprime, color='black', lw=1.0, linestyle='--')
    ax.set_xticks([tprime, 14])
    ax.set_xticklabels([r"$t'$", r"$t$"], fontsize=13)
    ax.set_yticks([])
    ax.set_ylabel(r"$\gamma_t$", fontsize=14, rotation=0, labelpad=18)

    ax.text(9.5, 0.55, r"$\Gamma_{t'} = \sum_{t=t'}^{\infty}\gamma_t$", fontsize=13, ha='center')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_position(('data', 0))

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "discount_function.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6.3 : The delay environment nu_delay
# ---------------------------------------------------------------------------
def fig_delay_environment():
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(-0.6, 3.6)
    ax.axis('off')
    ax.set_aspect('equal')

    n_bottom = 5  # S plus 4 circles
    bottom_x = [0, 2.2, 4.4, 6.6, 8.8]
    top_x = [2.2, 4.4, 6.6, 8.8]

    # Bottom row nodes
    ax.add_patch(Circle((bottom_x[0], 0), 0.42, facecolor='white', edgecolor='black', lw=1.6, zorder=3))
    ax.text(bottom_x[0], 0, "S", ha='center', va='center', fontsize=13, zorder=4)
    for x in bottom_x[1:]:
        ax.add_patch(Circle((x, 0), 0.42, facecolor='white', edgecolor='black', lw=1.6, zorder=3))

    # Top row nodes
    for x in top_x:
        ax.add_patch(Circle((x, 2.4), 0.42, facecolor='white', edgecolor='black', lw=1.6, zorder=3))

    up_labels = [r"$1/2$", r"$2/3$", r"$3/4$", r"$4/5$"]
    right_labels_bottom = ["0", "0", "0", "0"]
    right_labels_top = ["0", "0", "0"]

    # Up arrows (bottom -> top), labelled with rewards
    for i, x in enumerate(bottom_x[:-1]):
        ax.annotate("", xy=(x, 2.0), xytext=(x, 0.42),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
        ax.text(x - 0.42, 1.2, up_labels[i], fontsize=12, ha='right', va='center')

    # Right arrows along the bottom row, labelled reward 0
    for i in range(len(bottom_x) - 1):
        ax.annotate("", xy=(bottom_x[i + 1] - 0.42, 0), xytext=(bottom_x[i] + 0.42, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
        ax.text((bottom_x[i] + bottom_x[i + 1]) / 2, -0.35, right_labels_bottom[i], fontsize=11, ha='center')

    # Right arrows along the top row, labelled reward 0, with trailing "..."
    for i in range(len(top_x) - 1):
        ax.annotate("", xy=(top_x[i + 1] - 0.42, 2.4), xytext=(top_x[i] + 0.42, 2.4),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
        ax.text((top_x[i] + top_x[i + 1]) / 2, 2.75, right_labels_top[i], fontsize=11, ha='center')

    # trailing dots / arrows to infinity
    ax.annotate("", xy=(9.4, 2.4), xytext=(top_x[-1] + 0.42, 2.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
    ax.text(9.55, 2.4, r"$\cdots$", fontsize=13, ha='left', va='center')
    ax.annotate("", xy=(9.4, 0), xytext=(bottom_x[-1] + 0.42, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
    ax.text(9.55, 0, r"$\cdots$", fontsize=13, ha='left', va='center')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "delay_environment.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6.5 : Optimal gridworld policies for various penalties epsilon
# ---------------------------------------------------------------------------
ARROWS = {
    '>': (0.32, 0), '<': (-0.32, 0), '^': (0, 0.32), 'v': (0, -0.32)
}

def draw_grid(ax, arrows, title):
    # arrows: 3x3 nested list of one of '>','<','^','v','W' (wall)
    n = 3
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')

    # cell colors: top-right = +1 (light gray), middle-right = -1 (gray), wall = black
    cell_colors = {
        (0, 2): '#bdbdbd',  # +1 cell, row0(top) col2 -> plotted row index 2 (top)
        (1, 2): '#8a8a8a',  # -1 cell
    }

    for r in range(n):       # r=0 top row, r=1 middle, r=2 bottom (as printed)
        for c in range(n):
            y = n - 1 - r  # flip so row0 is drawn at top
            val = arrows[r][c]
            color = 'white'
            if val == 'W':
                color = 'black'
            elif (r, c) == (0, 2):
                color = '#bdbdbd'
            elif (r, c) == (1, 2):
                color = '#8a8a8a'
            rect = Rectangle((c, y), 1, 1, facecolor=color, edgecolor='black', lw=1.3)
            ax.add_patch(rect)
            cx, cy = c + 0.5, y + 0.5
            if val == 'W':
                pass
            elif val == '+1':
                ax.text(cx, cy, '+1', ha='center', va='center', fontsize=12, fontweight='bold')
            elif val == '-1':
                ax.text(cx, cy, r'$-1$', ha='center', va='center', fontsize=12, fontweight='bold')
            elif val in ARROWS:
                dx, dy = ARROWS[val]
                ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx * 0.15, cy - dy * 0.15),
                            arrowprops=dict(arrowstyle="-|>", lw=2.0, color='black'))
    ax.set_title(title, fontsize=12, pad=6)


def fig_gridworld_policies():
    # rows top->bottom, cols left->right, as printed in Figure 6.5
    grid_a = [['>', '>', '>', '+1'],
              ['^', 'W', '^', '-1'],
              ['^', '<', '<', '<']]
    grid_b = [['>', '>', '>', '+1'],
              ['^', 'W', '^', '-1'],
              ['^', '>', '^', '<']]
    grid_c = [['>', '>', '>', '+1'],
              ['^', 'W', '>', '-1'],
              ['>', '>', '^', '^']]
    grid_d = [['v', '<', '<', '+1'],
              ['>', 'W', 'v', '-1'],
              ['^', '<', '<', '<']]

    # convert 3x4 lists (3 rows, col0..2 are moves, col3 is terminal label) into 3x3 arrow grid + terminal markers
    def to_arrows(g):
        out = [[None] * 3 for _ in range(3)]
        for r in range(3):
            out[r][0] = g[r][0]
            out[r][1] = g[r][1]
            out[r][2] = g[r][2] if g[r][2] not in ('+1', '-1') else g[r][2]
        return out

    grids = [grid_a, grid_b, grid_c, grid_d]
    titles = [r"(a) $\varepsilon=-0.01$", r"(b) $\varepsilon=-0.1$",
              r"(c) $\varepsilon=-2$", r"(d) $\varepsilon=0.01$"]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, g, title in zip(axes, grids, titles):
        # Build a proper 3x3 grid where column index 2 holds +1/-1 special cells
        # Actual layout: 3 rows x 3 cols, rightmost column top=+1, middle=-1
        arr = [[g[0][0], g[0][1], '+1'],
               [g[1][0], 'W', '-1'],
               [g[2][0], g[2][1], g[2][2]]]
        draw_grid(ax, arr, title)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gridworld_policies.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extra figure: geometric vs finite-life vs hyperbolic-like discount comparison
# (supports the Python worked example in Section 6.4)
# ---------------------------------------------------------------------------
def fig_discount_comparison():
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    k = np.arange(0, 21)
    geo = 0.85 ** k
    finite = (k <= 8).astype(float)
    power = 1.0 / (k + 1) ** 1.5
    power = power / power[0]

    ax.plot(k, geo, marker='o', ms=4, lw=1.8, label=r"geometric, $\gamma=0.85$")
    ax.step(k, finite, where='mid', lw=1.8, label=r"finite life, $m=8$")
    ax.plot(k, power, marker='s', ms=4, lw=1.8, label=r"power, $\delta=0.5$")

    ax.set_xlabel(r"future time step $k$")
    ax.set_ylabel(r"discount weight $\gamma_k$")
    ax.legend(fontsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "discount_comparison.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_cybernetic_model()
    fig_discount_function()
    fig_delay_environment()
    fig_gridworld_policies()
    fig_discount_comparison()
    print("All figures written to", OUTDIR)
