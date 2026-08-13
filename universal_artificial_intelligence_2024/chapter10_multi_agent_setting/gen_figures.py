"""
gen_figures.py
Generate all figures for Chapter 10: "Multi-Agent Setting" slides.

Run with:
    conda run -n py313 python3 gen_figures.py

Produces (in ./figures/):
    multi_agent_diagram.pdf     -- Figure 10.1 recreation: agents interacting with environment sigma
    payoff_pd.pdf               -- Prisoner's Dilemma payoff matrix (heatmap + numbers)
    payoff_sh.pdf               -- Stag Hunt payoff matrix
    payoff_chicken.pdf          -- Chicken payoff matrix
    payoff_bos.pdf              -- Battle of the Sexes payoff matrix
    payoff_mp.pdf               -- Matching Pennies payoff matrix
    mixed_nash_mp.pdf           -- Matching pennies best-response / mixed Nash illustration
    reflective_oracle_schematic.pdf  -- oracle query/response loop diagram
    grain_of_truth_schematic.pdf     -- recursive "agent models opponent models agent..." diagram
    dogmatic_prior_convergence.pdf   -- Example 10.7.3: two dogmatic Bayesians fail to reach Nash eq.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

COL = {
    "blue": "#1f5fa8",
    "red": "#c0392b",
    "green": "#1e8449",
    "orange": "#d68910",
    "purple": "#6c3483",
    "gray": "#5d6d7e",
    "lightblue": "#d6e4f0",
    "lightred": "#f5d6d3",
}


# ─────────────────────────────────────────────────────────────────────────
# Figure: Multi-agent environment diagram (recreation of Figure 10.1)
# ─────────────────────────────────────────────────────────────────────────
def multi_agent_diagram():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    n_agents = 3
    ys = [4.6, 3.0, 0.9]
    labels = [r"agent $\pi_1$", r"agent $\pi_2$", r"agent $\pi_n$"]

    # environment box
    env_box = FancyBboxPatch((6.6, 0.4), 2.6, 4.6, boxstyle="round,pad=0.05",
                              linewidth=1.8, edgecolor="black", facecolor=COL["lightblue"])
    ax.add_patch(env_box)
    ax.text(7.9, 2.7, "multi-agent\nenvironment $\\sigma$", ha="center", va="center", fontsize=13, fontweight="bold")

    for i, (y, lab) in enumerate(zip(ys, labels)):
        box = FancyBboxPatch((0.4, y - 0.45), 2.6, 0.9, boxstyle="round,pad=0.05",
                              linewidth=1.6, edgecolor="black", facecolor=COL["lightred"])
        ax.add_patch(box)
        ax.text(1.7, y, lab, ha="center", va="center", fontsize=12)

        # action arrow agent -> env
        ax.annotate("", xy=(6.55, y + 0.18), xytext=(3.05, y + 0.18),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=COL["blue"]))
        ax.text(4.7, y + 0.42, rf"$a_t^{{{i+1 if i < 2 else 'n'}}}$", ha="center", fontsize=11, color=COL["blue"])

        # percept arrow env -> agent
        ax.annotate("", xy=(3.05, y - 0.22), xytext=(6.55, y - 0.22),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=COL["red"]))
        ax.text(4.7, y - 0.48, rf"$e_t^{{{i+1 if i < 2 else 'n'}}}$", ha="center", fontsize=11, color=COL["red"])

    ax.text(1.7, 1.85, r"$\vdots$", ha="center", fontsize=18)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "multi_agent_diagram.pdf"), bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Generic 2x2 payoff-matrix plotter
# ─────────────────────────────────────────────────────────────────────────
def payoff_matrix(filename, title, row_labels, col_labels, payoffs, row_name="Player 1", col_name="Player 2",
                   nash_cells=None, cmap_hi="#dff0d8", cmap_lo="#f2f2f2"):
    """
    payoffs: dict[(r,c)] = (u1, u2)  r,c in {0,1}
    nash_cells: list of (r,c) to highlight as Nash equilibria
    """
    nash_cells = nash_cells or []
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.invert_yaxis()

    for r in range(2):
        for c in range(2):
            u1, u2 = payoffs[(r, c)]
            face = "#fdf2c9" if (r, c) in nash_cells else "#f7f7f7"
            rect = mpatches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor="black", linewidth=1.4)
            ax.add_patch(rect)
            txt = f"({u1:g}, {u2:g})"
            ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold" if (r, c) in nash_cells else "normal")

    for c in range(2):
        ax.text(c + 0.5, -0.18, col_labels[c], ha="center", va="center", fontsize=12, style="italic")
    for r in range(2):
        ax.text(-0.25, r + 0.5, row_labels[r], ha="center", va="center", fontsize=12, style="italic", rotation=90)

    ax.text(1.0, -0.55, col_name, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(-0.65, 1.0, row_name, ha="center", va="center", fontsize=12, fontweight="bold", rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close(fig)


def all_payoff_matrices():
    # Prisoner's Dilemma: rows/cols = defect, cooperate ; Nash = (defect,defect)
    payoff_matrix(
        "payoff_pd.pdf", "Prisoner's Dilemma",
        row_labels=["defect", "cooperate"], col_labels=["defect", "cooperate"],
        payoffs={(0, 0): (1, 1), (0, 1): (4, 0), (1, 0): (0, 4), (1, 1): (2, 2)},
        nash_cells=[(0, 0)],
    )
    # Stag Hunt: rows/cols = alone, together ; Nash = (alone,alone) and (together,together)
    payoff_matrix(
        "payoff_sh.pdf", "Stag Hunt",
        row_labels=["alone", "together"], col_labels=["alone", "together"],
        payoffs={(0, 0): (2, 2), (0, 1): (3, 0), (1, 0): (0, 3), (1, 1): (4, 4)},
        nash_cells=[(0, 0), (1, 1)],
    )
    # Chicken: rows/cols = no swerve, swerve ; Nash = (no swerve,swerve) and (swerve,no swerve)
    payoff_matrix(
        "payoff_chicken.pdf", "Chicken",
        row_labels=["no swerve", "swerve"], col_labels=["no swerve", "swerve"],
        payoffs={(0, 0): (0, 0), (0, 1): (4, 1), (1, 0): (1, 4), (1, 1): (2, 2)},
        nash_cells=[(0, 1), (1, 0)],
    )
    # Battle of the Sexes: rows(wife)=Musical,Western ; cols(husband)=Musical,Western
    payoff_matrix(
        "payoff_bos.pdf", "Battle of the Sexes",
        row_labels=["Musical", "Western"], col_labels=["Musical", "Western"],
        payoffs={(0, 0): (2, 4), (0, 1): (0, 0), (1, 0): (0, 0), (1, 1): (4, 2)},
        row_name="Wife", col_name="Husband",
        nash_cells=[(0, 0), (1, 1)],
    )
    # Matching Pennies: rows/cols = Heads, Tails ; no pure Nash
    payoff_matrix(
        "payoff_mp.pdf", "Matching Pennies",
        row_labels=["Heads", "Tails"], col_labels=["Heads", "Tails"],
        payoffs={(0, 0): (1, 0), (0, 1): (0, 1), (1, 0): (0, 1), (1, 1): (1, 0)},
        nash_cells=[],
    )


# ─────────────────────────────────────────────────────────────────────────
# Figure: Mixed-strategy best response / Nash for matching pennies
# ─────────────────────────────────────────────────────────────────────────
def mixed_nash_mp():
    theta = np.linspace(0, 1, 400)  # player 1's prob of Heads
    # Player 2's expected utility for playing Heads vs Tails, given player 1 mixes theta
    # u2(Heads | theta) = P(P1=Tails)*1 + P(P1=Heads)*0 = 1-theta
    # u2(Tails | theta) = theta
    u2_heads = 1 - theta
    u2_tails = theta

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(theta, u2_heads, color=COL["blue"], lw=2.2, label=r"$U_2(\mathrm{Heads}\mid\theta)=1-\theta$")
    ax.plot(theta, u2_tails, color=COL["red"], lw=2.2, label=r"$U_2(\mathrm{Tails}\mid\theta)=\theta$")
    ax.axvline(0.5, color="gray", ls="--", lw=1.3)
    ax.plot([0.5], [0.5], marker="o", ms=9, color=COL["green"], zorder=5)
    ax.annotate("mixed Nash\n" + r"$\theta=\theta_2=\frac{1}{2}$", xy=(0.5, 0.5), xytext=(0.62, 0.72),
                fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel(r"$\theta$ = Player 1's probability of choosing Heads")
    ax.set_ylabel("Player 2's expected utility")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mixed_nash_mp.pdf"), bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure: Reflective oracle schematic
# ─────────────────────────────────────────────────────────────────────────
def reflective_oracle_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    tm_box = FancyBboxPatch((0.5, 1.7), 3.0, 1.6, boxstyle="round,pad=0.06",
                             linewidth=1.8, edgecolor="black", facecolor=COL["lightblue"])
    ax.add_patch(tm_box)
    ax.text(2.0, 2.5, r"Turing machine $T^O$" "\n(may query $O$" "\nabout itself/others)",
            ha="center", va="center", fontsize=11)

    or_box = FancyBboxPatch((6.5, 1.7), 3.0, 1.6, boxstyle="round,pad=0.06",
                             linewidth=1.8, edgecolor="black", facecolor=COL["lightred"])
    ax.add_patch(or_box)
    ax.text(8.0, 2.5, r"Reflective Oracle $O$" "\n" r"$O(T,x,z)\in\{0,1\}$",
            ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(6.4, 2.9), xytext=(3.6, 2.9),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=COL["blue"]))
    ax.text(5.0, 3.15, r"query $\langle T,x,z\rangle$", ha="center", fontsize=10.5, color=COL["blue"])

    ax.annotate("", xy=(3.6, 2.1), xytext=(6.4, 2.1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=COL["red"]))
    ax.text(5.0, 1.85, r"answer $0$ or $1$", ha="center", fontsize=10.5, color=COL["red"])

    # self-loop on oracle to indicate it must be consistent about itself
    ax.annotate("", xy=(9.6, 1.55), xytext=(9.6, 3.45),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=COL["gray"],
                                 connectionstyle="arc3,rad=1.3"))
    ax.text(9.95, 2.5, r"$T$ may itself" "\ncall $O$", ha="left", va="center", fontsize=9.5, color=COL["gray"])

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "reflective_oracle_schematic.pdf"), bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure: Grain of truth recursive modelling schematic
# ─────────────────────────────────────────────────────────────────────────
def grain_of_truth_schematic():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.25, 1.35)
    ax.axis("off")
    ax.set_aspect("equal")

    n = 3
    radii = [1.05, 0.68, 0.32]
    names = ["Agent 1 models", "Agent 2's model of", "Agent 1's model of ..."]
    colors = [COL["blue"], COL["red"], COL["green"]]
    for i, (r, name, c) in enumerate(zip(radii, names, colors)):
        circ = plt.Circle((0, 0), r, fill=False, edgecolor=c, linewidth=2.2)
        ax.add_patch(circ)
        ax.text(0, r + 0.12, name, ha="center", va="bottom", fontsize=10.5, color=c)

    ax.text(0, 0, r"$\ddots$", ha="center", va="center", fontsize=20)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "grain_of_truth_schematic.pdf"), bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure: Dogmatic prior example (Example 10.7.3) -- value trajectories
# ─────────────────────────────────────────────────────────────────────────
def dogmatic_prior_convergence():
    # pi_1 plays (HHT)^inf with average reward 2/3; pi_2 = pi_H always plays H with
    # average reward 1/3, in the "matching pennies with rewards depending on history"
    # example. We simply illustrate the two constant asymptotic average values that
    # a dogmatic Bayesian will lock onto and never leave, vs. time.
    T = 60
    t = np.arange(1, T + 1)

    # simulate empirical average reward converging to 2/3 and 1/3 respectively (illustrative)
    rng = np.random.default_rng(0)
    seq1 = np.tile([1, 1, 0], T // 3 + 1)[:T]  # HHT pattern reward proxy -> average 2/3
    seq2 = np.zeros(T)  # policy pi_2 always H -> average reward 1/3 in this construction
    avg1 = np.cumsum(seq1) / t
    avg2 = np.full(T, 1 / 3)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(t, avg1, color=COL["blue"], lw=2.0, label=r"$V_\xi^{\pi_1}\to c_1\approx 2/3$")
    ax.plot(t, avg2, color=COL["red"], lw=2.0, label=r"$V_\xi^{\pi_2}\to c_2\approx 1/3$")
    ax.axhline(2 / 3, color=COL["blue"], ls=":", lw=1.0)
    ax.axhline(1 / 3, color=COL["red"], ls=":", lw=1.0)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("running average reward")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "dogmatic_prior_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    multi_agent_diagram()
    all_payoff_matrices()
    mixed_nash_mp()
    reflective_oracle_schematic()
    grain_of_truth_schematic()
    dogmatic_prior_convergence()
    print("All figures generated in", FIGDIR)
