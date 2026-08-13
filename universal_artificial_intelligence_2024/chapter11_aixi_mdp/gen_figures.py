"""
Generate all native (non-cropped) figures used in
chapter11_aixi_mdp_slides.tex
(Chapter 11: AIXI-MDP, "An Introduction to Universal Artificial
Intelligence", Hutter, Quarel & Catt, CRC Press, 2024)

Figure 11.2 (the five experimental-result panels, which are real recorded
simulation curves from [PH06b] that cannot be regenerated from the text
alone) is cropped directly from the book PDF by crop_figs.py instead.

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


# ---------------------------------------------------------------------------
# Figure 11.1: Agent / Opponent / Environment interaction loop
# ---------------------------------------------------------------------------
def fig_11_1_interaction_loop():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.0, 4.6)
    ax.axis('off')
    ax.set_aspect('equal')

    # Outer "Environment" box
    env_box = Rectangle((5.4, 0.6), 5.7, 3.6, facecolor='none',
                         edgecolor='black', lw=1.6, zorder=1)
    ax.add_patch(env_box)
    ax.text(5.55, 4.35, "Environment", fontsize=12, ha='left', va='top')

    # Agent circle (outside environment box)
    agent = Circle((1.6, 2.4), 0.85, facecolor='#dbe4f0', edgecolor='#2b3a67',
                    lw=1.8, zorder=3)
    ax.add_patch(agent)
    ax.text(1.6, 2.4, "Agent", fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2b3a67')

    # Opponent circle (inside environment box, upper area)
    opp = Circle((8.1, 3.5), 0.75, facecolor='#f3e3c3', edgecolor='#7a5a1f',
                 lw=1.8, zorder=3)
    ax.add_patch(opp)
    ax.text(8.1, 3.5, "Opponent", fontsize=11, ha='center', va='center',
            fontweight='bold', color='#7a5a1f')

    # Reward-matrix diamond (inside environment box, lower area)
    diamond = Polygon([(8.1, 1.15), (8.75, 1.75), (8.1, 2.35), (7.45, 1.75)],
                       closed=True, facecolor='#e6e6e6', edgecolor='black', lw=1.6,
                       zorder=3)
    ax.add_patch(diamond)
    ax.text(8.1, 1.75, "Reward\nMatrix", fontsize=9.5, ha='center', va='center')

    # Arrow: Agent -> Environment box (top edge), labeled "Agent Action"
    ax.annotate("", xy=(5.4, 3.15), xytext=(2.45, 2.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black',
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(3.7, 3.35, "Agent Action", fontsize=10, ha='center')

    # Arrow: Environment box -> Agent (bottom edge), labeled "Opponent Action & Reward"
    ax.annotate("", xy=(2.45, 2.05), xytext=(5.4, 1.55),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color='black',
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(3.9, 1.15, "Opponent\nAction & Reward", fontsize=10, ha='center')

    # Arrow: Opponent -> Reward Matrix, labeled "Opponent Action"
    ax.annotate("", xy=(8.1, 2.35+0.35), xytext=(7.55, 3.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color='black',
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text(6.55, 2.55, "Opponent\nAction", fontsize=9.5, ha='center')

    # Arrow: Reward Matrix -> Opponent, labeled "Agent Action & Reward"
    ax.annotate("", xy=(8.55, 3.15), xytext=(8.75, 2.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color='black',
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text(10.35, 2.55, "Agent Action\n& Reward", fontsize=9.5, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig11_1_interaction_loop.pdf"),
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Laplace estimator convergence (illustrative, for the KT/Laplace
# comparison discussion in Section 11.2) -- shows xi_MDP(o=1|.) converging to
# the true theta as the count of a fixed action-observation pair grows.
# ---------------------------------------------------------------------------
def fig_laplace_convergence():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rng = np.random.default_rng(0)
    theta_true = 0.7
    n_max = 200
    outcomes = rng.random(n_max) < theta_true
    n1 = np.cumsum(outcomes)
    n = np.arange(1, n_max + 1)
    laplace = (n1 + 1) / (n + 2)
    kt = (n1 + 0.5) / (n + 1)

    ax.plot(n, laplace, label=r"Laplace estimator $\frac{n_1+1}{n+2}$", lw=1.8)
    ax.plot(n, kt, label=r"KT estimator $\frac{n_1+0.5}{n+1}$", lw=1.8, ls='--')
    ax.axhline(theta_true, color='gray', ls=':', lw=1.5,
               label=r"true $\theta=0.7$")
    ax.set_xlabel("number of observed outcomes $n$")
    ax.set_ylabel(r"estimated $\mathbb{P}(o_{t}=1)$")
    ax.set_title("Laplace vs. KT estimator converging to the true rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_laplace_convergence.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_11_1_interaction_loop()
    fig_laplace_convergence()
    print("Figures written to", OUTDIR)
