"""
Generate all figures used in chapter08_optimality_of_universal_agents_slides.tex
(Chapter 8: Optimality of Universal Agents, "An Introduction to Universal
Artificial Intelligence", Hutter, Quarel & Catt, CRC Press, 2024)

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Figure 8.1 : Heaven-Hell environment (Example 8.1.3)
# ---------------------------------------------------------------------------
def fig_heaven_hell():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.6, 2.6)
    ax.axis('off')
    ax.set_aspect('equal')

    # Circles: hell, start, heaven
    hell = Circle((1.3, 0.5), 0.85, facecolor='#f3d3d3', edgecolor='#7a1f1f', lw=2.0, zorder=3)
    start = Circle((5.0, 0.5), 0.85, facecolor='#dbe4f0', edgecolor='#2b3a67', lw=2.0, zorder=3)
    heaven = Circle((8.7, 0.5), 0.85, facecolor='#d7f0d3', edgecolor='#1f6b2b', lw=2.0, zorder=3)
    for c in (hell, start, heaven):
        ax.add_patch(c)
    ax.text(1.3, 0.5, "hell", ha='center', va='center', fontsize=13, fontweight='bold', color='#7a1f1f')
    ax.text(5.0, 0.5, "start", ha='center', va='center', fontsize=13, fontweight='bold', color='#2b3a67')
    ax.text(8.7, 0.5, "heaven", ha='center', va='center', fontsize=13, fontweight='bold', color='#1f6b2b')

    # self-loops on hell and heaven
    theta = np.linspace(0.35*np.pi, 1.65*np.pi, 100)
    loop_x = 1.3 - 1.55*np.cos(theta) - 0.55
    loop_y = 0.5 + 1.35*np.sin(theta)
    ax.plot(loop_x, loop_y, color='#7a1f1f', lw=1.8)
    ax.annotate("", xy=(loop_x[3], loop_y[3]), xytext=(loop_x[8], loop_y[8]),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#7a1f1f'))
    ax.text(-0.55, 0.5, r"$r=0$", ha='center', va='center', fontsize=11, color='#7a1f1f')

    loop_x2 = 8.7 + 1.55*np.cos(theta) + 0.55
    loop_y2 = 0.5 + 1.35*np.sin(theta)
    ax.plot(loop_x2, loop_y2, color='#1f6b2b', lw=1.8)
    ax.annotate("", xy=(loop_x2[8], loop_y2[8]), xytext=(loop_x2[3], loop_y2[3]),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color='#1f6b2b'))
    ax.text(10.55, 0.5, r"$r=1$", ha='center', va='center', fontsize=11, color='#1f6b2b')

    # start -> hell (a=left)
    ax.annotate("", xy=(2.2, 0.75), xytext=(4.1, 0.75),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color='black'))
    ax.text(3.15, 1.35, r"$a=\mathrm{left}$", ha='center', va='center', fontsize=11)
    ax.text(3.15, 0.35, r"$r=0$", ha='center', va='center', fontsize=10, color='#555555')

    # start -> heaven (a=right)
    ax.annotate("", xy=(7.8, 0.75), xytext=(5.9, 0.75),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color='black'))
    ax.text(6.85, 1.35, r"$a=\mathrm{right}$", ha='center', va='center', fontsize=11)
    ax.text(6.85, 0.35, r"$r=1$", ha='center', va='center', fontsize=10, color='#555555')

    ax.text(5.0, -1.4, r"$\nu_1$: left $\to$ hell forever, right $\to$ heaven forever"
                       r" ($\nu_2$ swaps the two)",
            ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "heaven_hell.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Hierarchy of asymptotic-optimality notions (implications)
# ---------------------------------------------------------------------------
def fig_optimality_hierarchy():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 8.6)
    ax.axis('off')

    boxes = {
        'strong': (5.0, 7.3, "Strong asymptotic\noptimality (a.s.)", '#2b3a67', '#dbe4f0'),
        'mean':   (2.3, 5.0, "Asymptotic optimality\nin mean", '#6b3a2b', '#f0ded3'),
        'prob':   (7.7, 5.0, "Asymptotic optimality\nin probability", '#6b3a2b', '#f0ded3'),
        'weak':   (5.0, 2.7, "Weak asymptotic optimality\n(Cesàro average)", '#1f6b2b', '#d7f0d3'),
        'regret': (5.0, 0.4, "Sublinear regret\n(recoverable env., Thm 8.1.13)", '#7a1f1f', '#f3d3d3'),
    }

    patches = {}
    for key, (x, y, label, edge, face) in boxes.items():
        w, h = (3.6, 1.35) if key != 'regret' else (5.4, 1.0)
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                              linewidth=1.8, edgecolor=edge, facecolor=face, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10.5, fontweight='bold', color=edge, zorder=4)
        patches[key] = (x, y, h)

    def arrow(a, b, style='-|>'):
        xa, ya, ha_ = patches[a]
        xb, yb, hb_ = patches[b]
        ax.annotate("", xy=(xb, yb + hb_/2 + 0.05) if yb < ya else (xb, yb - hb_/2 - 0.05),
                    xytext=(xa, ya - ha_/2 - 0.05) if yb < ya else (xa, ya + ha_/2 + 0.05),
                    arrowprops=dict(arrowstyle=style, lw=1.9, color='black'))

    arrow('strong', 'mean')
    arrow('strong', 'prob')
    arrow('mean', 'weak')
    arrow('prob', 'weak')
    arrow('weak', 'regret', style='<|-|>')

    ax.text(5.0, 1.55, "implies (recoverable env.\n+ Assumption 8.1.11)", ha='center', va='center',
            fontsize=8, style='italic', color='#444444')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "optimality_hierarchy.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Sublinear vs. linear regret (numeric illustration)
# ---------------------------------------------------------------------------
def fig_regret_curves():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    m = np.arange(1, 201)
    linear_regret = 0.5 * m                      # an agent that never learns
    sublinear_regret = 3.0 * np.sqrt(m)           # a "sublinear regret" agent
    finite_regret = 8.0 * (1 - np.exp(-m/15.0))   # Example 8.1.10 : finite regret

    ax.plot(m, linear_regret, color='#7a1f1f', lw=2.2, label=r'linear: $\mathrm{Regret}_m = 0.5\,m$')
    ax.plot(m, sublinear_regret, color='#2b3a67', lw=2.2, label=r'sublinear: $\mathrm{Regret}_m = 3\sqrt{m}$')
    ax.plot(m, finite_regret, color='#1f6b2b', lw=2.2, label=r'finite (Ex.\ 8.1.10): converges')

    ax.set_xlabel(r'horizon $m$ (number of time steps)')
    ax.set_ylabel(r'$\mathrm{Regret}_m(\pi,\mu)$')
    ax.set_title('Growth of regret under three policies')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "regret_curves.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Pareto frontier over two environments
# ---------------------------------------------------------------------------
def fig_pareto_frontier():
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    rng = np.random.default_rng(7)
    n = 60
    V_nu = rng.uniform(0.05, 0.95, n)
    V_rho = rng.uniform(0.05, 0.95, n)

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dom = (V_nu >= V_nu[i]) & (V_rho >= V_rho[i]) & ((V_nu > V_nu[i]) | (V_rho > V_rho[i]))
        is_dominated[i] = dom.any()

    ax.scatter(V_nu[is_dominated], V_rho[is_dominated], color='#b0b0b0', s=35,
               label='dominated policies', zorder=2)
    ax.scatter(V_nu[~is_dominated], V_rho[~is_dominated], color='#1f6b2b', s=55,
               edgecolor='black', linewidth=0.6, label='Pareto-optimal policies', zorder=3)

    order = np.argsort(V_nu[~is_dominated])
    frontier_x = V_nu[~is_dominated][order]
    frontier_y = V_rho[~is_dominated][order]
    ax.step(frontier_x, frontier_y, where='post', color='#1f6b2b', lw=1.6, linestyle='--', zorder=1)

    ax.set_xlabel(r'value $V_\nu^\pi$ in environment $\nu$')
    ax.set_ylabel(r'value $V_\rho^\pi$ in environment $\rho$')
    ax.set_title('Pareto frontier over $\\mathcal{M}=\\{\\nu,\\rho\\}$')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "pareto_frontier.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Dogmatic-prior weight redistribution (toy 3-environment example)
# ---------------------------------------------------------------------------
def fig_dogmatic_prior():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    labels = [r'$\nu_1$', r'$\nu_2$', r'$\nu_3$']
    w = np.array([0.5, 0.3, 0.2])
    eps = 0.3
    # w'_{rho_{pi,nu}} = (1-eps) w_nu + eps*w_rho ; here rho_{pi,nu1} absorbs mass from nu1
    w_prime = np.array([(1-eps)*w[0], eps*w[1], eps*w[2]])
    rho_mass = eps*w[0] + (1-eps)*w[1] + (1-eps)*w[2]  # illustrative aggregate for rho states

    axes[0].bar(labels, w, color='#6b3a2b', edgecolor='black')
    axes[0].set_title(r'original prior $w_\nu$')
    axes[0].set_ylim(0, 0.6)
    axes[0].set_ylabel('prior weight')

    labels2 = [r"$\rho_{\pi,\nu_1}$", r'$\nu_2$-mass kept', r'$\nu_3$-mass kept', r'$\rho$-mass (traps)']
    vals2 = [(1-eps)*w[0], eps*w[1], eps*w[2], eps*(w[1]+w[2])]
    axes[1].bar(range(len(labels2)), vals2, color=['#7a1f1f', '#6b3a2b', '#6b3a2b', '#7a1f1f'], edgecolor='black')
    axes[1].set_xticks(range(len(labels2)))
    axes[1].set_xticklabels(labels2, rotation=20, ha='right', fontsize=8)
    axes[1].set_title(r"reweighted prior $w'$ ($\varepsilon={:.1f}$)".format(eps))
    axes[1].set_ylim(0, 0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "dogmatic_prior.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure : Adversarial "moving target" environment for Theorem 8.3.1 / 8.3.2
# ---------------------------------------------------------------------------
def fig_adversarial_switch():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    ax.annotate("", xy=(11.4, 1.6), xytext=(0.6, 1.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black'))
    ax.text(11.6, 1.6, r'$t$', fontsize=12)

    ax.plot([4.0, 4.0], [0.9, 2.3], color='#7a1f1f', lw=1.8, linestyle='--')
    ax.text(4.0, 2.55, r'time $T$', ha='center', fontsize=11, color='#7a1f1f')

    ax.text(2.0, 1.15, r'$\mu_1=\mu_2$', ha='center', fontsize=10)
    ax.text(2.0, 0.55, "(identical so far)", ha='center', fontsize=9, style='italic', color='#555555')

    box1 = FancyBboxPatch((5.6, 2.3), 5.4, 1.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                           linewidth=1.6, edgecolor='#2b3a67', facecolor='#dbe4f0')
    ax.add_patch(box1)
    ax.text(8.3, 2.8, r'$\mu_1$: optimal action forever is $up$', ha='center', va='center', fontsize=10)

    box2 = FancyBboxPatch((5.6, 0.1), 5.4, 1.0, boxstyle="round,pad=0.05,rounding_size=0.1",
                           linewidth=1.6, edgecolor='#7a1f1f', facecolor='#f3d3d3')
    ax.add_patch(box2)
    ax.text(8.3, 0.6, r'$\mu_2$: optimal action forever is $down$', ha='center', va='center', fontsize=10)

    ax.annotate("", xy=(5.5, 2.8), xytext=(4.1, 1.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color='#2b3a67'))
    ax.annotate("", xy=(5.5, 0.6), xytext=(4.1, 1.45),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color='#7a1f1f'))

    ax.text(6.0, 3.55, "A deterministic $\\pi$ that eventually commits to $up$ forever\n"
                       "fails in $\\mu_2$ -- and vice versa.",
            ha='left', fontsize=9, style='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "adversarial_switch.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig_heaven_hell()
    fig_optimality_hierarchy()
    fig_regret_curves()
    fig_pareto_frontier()
    fig_dogmatic_prior()
    fig_adversarial_switch()
    print("All figures generated in:", OUTDIR)
