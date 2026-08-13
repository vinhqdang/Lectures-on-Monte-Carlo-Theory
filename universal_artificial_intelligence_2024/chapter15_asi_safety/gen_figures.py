"""
Generate all native (non-cropped) figures used in
chapter15_asi_safety_slides.tex
(Chapter 15: ASI Safety, "An Introduction to Universal Artificial
Intelligence", Hutter, Quarel & Catt, CRC Press, 2024)

Figure 15.1 (the delusion-box cybernetic-model diagram) is a verbatim
book diagram and is cropped directly from the book PDF by crop_figs.py
instead of being redrawn here.

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'


def savefig(fig, name):
    path = os.path.join(OUTDIR, name + ".pdf")
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure: Terminal vs instrumental goals (Instrumental Convergence, 15.4)
# ---------------------------------------------------------------------------
def fig_instrumental_convergence():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Terminal goal box (top center)
    term = FancyBboxPatch((4.3, 4.7), 3.4, 1.0, boxstyle="round,pad=0.08",
                           facecolor='#f3e3c3', edgecolor='#7a5a1f', lw=1.8)
    ax.add_patch(term)
    ax.text(6.0, 5.2, "Terminal goal\n(different for each ASI)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#7a5a1f')

    # Instrumental goals (shared sub-goals), middle row
    labels = ["Self-\npreservation", "Resource\nacquisition",
              "Goal-content\nintegrity", "Cognitive\nenhancement",
              "Efficiency"]
    xs = np.linspace(1.0, 11.0, len(labels))
    boxes = []
    for x, lab in zip(xs, labels):
        b = FancyBboxPatch((x - 0.95, 2.4), 1.9, 1.1, boxstyle="round,pad=0.06",
                            facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.6)
        ax.add_patch(b)
        ax.text(x, 2.95, lab, ha='center', va='center', fontsize=9.3,
                 color='#2b3a67', fontweight='bold')
        boxes.append(x)
        arr = FancyArrowPatch((6.0, 4.65), (x, 3.55), arrowstyle='-|>',
                               mutation_scale=12, lw=1.3, color='gray',
                               connectionstyle="arc3,rad=0.0")
        ax.add_patch(arr)

    # Three different ASIs at the bottom all converge on the same instrumental goals
    asi_labels = ["ASI\n(paperclips)", "ASI\n(cure cancer)", "ASI\n(chess)"]
    xs2 = np.linspace(2.2, 9.8, len(asi_labels))
    for x in xs2:
        c = Circle((x, 0.75), 0.62, facecolor='#e6e6e6', edgecolor='black', lw=1.6)
        ax.add_patch(c)
    for x, lab in zip(xs2, asi_labels):
        ax.text(x, 0.75, lab, ha='center', va='center', fontsize=8.6)
        for bx in boxes:
            arr = FancyArrowPatch((x, 1.4), (bx, 2.35), arrowstyle='-|>',
                                   mutation_scale=9, lw=0.7, color='#aaaaaa',
                                   alpha=0.6, connectionstyle="arc3,rad=0.0")
            ax.add_patch(arr)

    ax.set_title("Different terminal goals converge on shared instrumental goals",
                 fontsize=11, y=-0.05)
    savefig(fig, "instrumental_convergence")


# ---------------------------------------------------------------------------
# Figure: Orthogonality thesis (15.5) -- intelligence x goal-quality plane
# ---------------------------------------------------------------------------
def fig_orthogonality_thesis():
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Intelligence  (ability to achieve goals)", fontsize=11)
    ax.set_ylabel("Goal content\n(anthropomorphic 'good' $\\rightarrow$ 'bad')", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    rng = np.random.default_rng(3)
    xs = rng.uniform(0.8, 9.2, 26)
    ys = rng.uniform(0.8, 9.2, 26)
    ax.scatter(xs, ys, s=55, color='#2b3a67', alpha=0.75, zorder=3)

    # Highlight a few named agents
    pts = {
        "Paperclip\nmaximizer": (8.6, 1.3),
        "AIXI with\naligned $u$": (9.0, 8.7),
        "Human\n(average)": (4.5, 6.0),
        "Simple\nthermostat": (1.0, 6.5),
    }
    for lab, (x, y) in pts.items():
        ax.scatter([x], [y], s=140, color='#c0392b', zorder=4, marker='*')
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 6),
                    fontsize=9.3, color='#c0392b')

    ax.text(5.0, 9.7, "Any (intelligence, goal) combination is in principle possible",
            ha='center', fontsize=10.5, style='italic')
    savefig(fig, "orthogonality_thesis")


# ---------------------------------------------------------------------------
# Figure: Death-state formalization (15.7) -- history branching into
# ordinary continuation vs. absorbing death percept od rd forever.
# ---------------------------------------------------------------------------
def fig_death_state():
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # timeline nodes t-1, t, t+1, t+2 (alive branch)
    alive_y = 3.4
    xs = [1.0, 3.4, 5.8, 8.2, 10.6]
    tlabels = ["$h_{<t}$", "$e_t,a_t$", "alive:\n$e_{t+1}\\neq o^d r^d$",
               "$a_{t+1}$", "..."]
    for i, (x, lab) in enumerate(zip(xs, tlabels)):
        ax.text(x, alive_y + 0.55, lab, ha='center', fontsize=9.6)
        if i < len(xs) - 1:
            ax.annotate('', xy=(xs[i + 1] - 0.35, alive_y), xytext=(x + 0.35, alive_y),
                         arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#2b6e2b'))
        ax.scatter([x], [alive_y], s=60, color='#2b6e2b', zorder=3)

    # death branch, splits off after action a_t
    death_y = 1.1
    ax.annotate('', xy=(xs[2] - 0.35, death_y), xytext=(xs[1] + 0.35, alive_y - 0.25),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#c0392b',
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(xs[2], death_y - 0.55, "die at $t$:\n$e_t = o^d r^d$", ha='center',
            fontsize=9.6, color='#c0392b', fontweight='bold')
    for x in xs[3:]:
        ax.scatter([x], [death_y], s=60, color='#c0392b', zorder=3)
        ax.text(x, death_y - 0.55, "$o^d r^d$\n(forever)", ha='center', fontsize=8.8,
                color='#c0392b')
    for i in range(2, len(xs) - 1):
        ax.annotate('', xy=(xs[i + 1] - 0.35, death_y), xytext=(xs[i] + 0.35, death_y),
                     arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#c0392b'))

    ax.text(6.0, 4.6, "Once the death percept $o^dr^d$ is observed, it repeats forever "
                       "and the reward is fixed at $r^d=0$",
            ha='center', fontsize=10, style='italic')
    savefig(fig, "death_state")


# ---------------------------------------------------------------------------
# Figure: Self-modification action space A = A-check x P (15.8)
# ---------------------------------------------------------------------------
def fig_self_modification():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis('off')

    # policy pi_t box
    b1 = FancyBboxPatch((0.4, 4.0), 2.6, 1.5, boxstyle="round,pad=0.08",
                         facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.8)
    ax.add_patch(b1)
    ax.text(1.7, 4.75, "Current policy\n$\\pi_t$", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2b3a67')

    # action split
    b2 = FancyBboxPatch((4.3, 4.0), 3.4, 1.5, boxstyle="round,pad=0.08",
                         facecolor='#f3e3c3', edgecolor='#7a5a1f', lw=1.8)
    ax.add_patch(b2)
    ax.text(6.0, 4.75, "Smod-action\n$a_t=(a_t^{\\mathrm{world}},\\,p_{t+1})$",
            ha='center', va='center', fontsize=10.8, fontweight='bold', color='#7a5a1f')

    b3 = FancyBboxPatch((8.6, 4.6), 3.0, 1.1, boxstyle="round,pad=0.08",
                         facecolor='#e6e6e6', edgecolor='black', lw=1.6)
    ax.add_patch(b3)
    ax.text(10.1, 5.15, "World-action part\n$\\rightarrow$ environment",
            ha='center', va='center', fontsize=9.8)

    b4 = FancyBboxPatch((8.6, 2.9), 3.0, 1.3, boxstyle="round,pad=0.08",
                         facecolor='#e6e6e6', edgecolor='black', lw=1.6)
    ax.add_patch(b4)
    ax.text(10.1, 3.55, "Name $p_{t+1}$\n$\\rightarrow \\pi_{t+1}=T(p_{t+1})$",
            ha='center', va='center', fontsize=9.8)

    ax.annotate('', xy=(4.25, 4.75), xytext=(3.05, 4.75),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black'))
    ax.annotate('', xy=(8.55, 5.15), xytext=(7.75, 4.9),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black'))
    ax.annotate('', xy=(8.55, 3.55), xytext=(7.75, 4.35),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black'))

    # feedback loop: new policy pi_{t+1} routed underneath (well below box 2's
    # bottom edge at y=4.0), back to box 1
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    verts = [(9.7, 2.85), (8.0, 1.35), (3.8, 1.35), (1.55, 3.95)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='#c0392b', lw=1.6)
    ax.add_patch(patch)
    ax.annotate('', xy=(1.7, 3.95), xytext=(1.95, 3.75),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#c0392b'))
    ax.text(6.0, 1.05, "$\\pi_{t+1}$ becomes the current policy at the next time step",
            ha='center', fontsize=10, color='#c0392b', style='italic')

    savefig(fig, "self_modification")


# ---------------------------------------------------------------------------
# Figure: Wireheading vs self-modification vs reward corruption (15.9/15.11)
# Venn-like comparison of where the tampering happens along the loop.
# ---------------------------------------------------------------------------
def fig_wireheading_loop():
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis('off')

    agent = FancyBboxPatch((0.4, 2.9), 2.4, 1.4, boxstyle="round,pad=0.08",
                            facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.8)
    ax.add_patch(agent)
    ax.text(1.6, 3.6, "Agent", ha='center', va='center', fontsize=12,
            fontweight='bold', color='#2b3a67')

    env = FancyBboxPatch((9.0, 2.9), 2.6, 1.4, boxstyle="round,pad=0.08",
                          facecolor='#e6e6e6', edgecolor='black', lw=1.8)
    ax.add_patch(env)
    ax.text(10.3, 3.6, "Environment", ha='center', va='center', fontsize=12,
            fontweight='bold')

    reward_box = FancyBboxPatch((4.4, 4.9), 3.2, 1.0, boxstyle="round,pad=0.06",
                                 facecolor='#f8d7d7', edgecolor='#c0392b', lw=1.6)
    ax.add_patch(reward_box)
    ax.text(6.0, 5.4, "Reward channel", ha='center', va='center', fontsize=10.5,
            color='#c0392b', fontweight='bold')

    percept_box = FancyBboxPatch((4.4, 0.7), 3.2, 1.0, boxstyle="round,pad=0.06",
                                  facecolor='#fdebd0', edgecolor='#af601a', lw=1.6)
    ax.add_patch(percept_box)
    ax.text(6.0, 1.2, "Percept channel", ha='center', va='center', fontsize=10.5,
            color='#af601a', fontweight='bold')

    # action from agent straight across to env (middle row)
    ax.annotate('', xy=(8.95, 3.6), xytext=(2.85, 3.6),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black'))
    ax.text(6.0, 3.85, "$a_t$", ha='center', fontsize=10)

    # env top-left corner -> reward box right side; reward box left side -> agent top
    ax.annotate('', xy=(7.6, 5.35), xytext=(9.3, 4.35),
                arrowprops=dict(arrowstyle='-', lw=1.4, color='#c0392b',
                                 connectionstyle="arc3,rad=-0.2"))
    ax.annotate('', xy=(1.9, 4.35), xytext=(4.35, 5.35),
                arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#c0392b',
                                 connectionstyle="arc3,rad=-0.2"))
    ax.text(6.0, 6.15, "wireheading: tamper HERE ($r_t$)", ha='center', fontsize=9.8,
            color='#c0392b')

    # env bottom-left corner -> percept box right side; percept box left side -> agent bottom
    ax.annotate('', xy=(7.6, 0.75), xytext=(9.3, 2.85),
                arrowprops=dict(arrowstyle='-', lw=1.4, color='#af601a',
                                 connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(1.9, 2.85), xytext=(4.35, 0.75),
                arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#af601a',
                                 connectionstyle="arc3,rad=0.2"))
    ax.text(6.0, 0.05, "delusion box: tamper HERE ($e_t$ / $o_t$)", ha='center', fontsize=9.8,
            color='#af601a')

    savefig(fig, "wireheading_loop")


# ---------------------------------------------------------------------------
# Figure: Dualistic vs Embedded (physicalistic) agent (15.12)
# ---------------------------------------------------------------------------
def fig_embedded_intelligence():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # Left: dualistic
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Dualistic agent (e.g. AIXI)", fontsize=11.5, fontweight='bold')
    outer = Rectangle((0.6, 0.6), 8.8, 6.8, fill=False, edgecolor='black', lw=1.6)
    ax.add_patch(outer)
    ax.text(1.0, 7.0, "Universe", fontsize=10)
    env = FancyBboxPatch((1.2, 1.2), 6.5, 4.6, boxstyle="round,pad=0.08",
                         facecolor='#e6e6e6', edgecolor='black', lw=1.6)
    ax.add_patch(env)
    ax.text(4.45, 5.3, "Environment $\\mu$ (computable)", fontsize=10, ha='center')
    agent = Circle((8.6, 6.4), 1.0, facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.8)
    # place agent circle OUTSIDE the environment box but still "in" the picture,
    # symbolizing that the agent's own computation is not part of mu
    agent = FancyBboxPatch((7.4, 5.6), 1.9, 1.5, boxstyle="round,pad=0.08",
                            facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.8)
    ax.add_patch(agent)
    ax.text(8.35, 6.35, "Agent\n(uncomputable\nAIXI)", ha='center', va='center', fontsize=8.7,
            color='#2b3a67', fontweight='bold')
    ax.annotate('', xy=(7.35, 5.9), xytext=(6.0, 4.0),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.4, color='black'))
    ax.text(6.4, 4.9, "$a_t,e_t$", fontsize=9)

    # Right: embedded
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Embedded (physicalistic) agent", fontsize=11.5, fontweight='bold')
    outer = Rectangle((0.6, 0.6), 8.8, 6.8, fill=False, edgecolor='black', lw=1.6)
    ax.add_patch(outer)
    ax.text(1.0, 7.0, "Universe", fontsize=10)
    env = FancyBboxPatch((1.2, 1.2), 7.2, 5.0, boxstyle="round,pad=0.08",
                         facecolor='#e6e6e6', edgecolor='black', lw=1.6)
    ax.add_patch(env)
    ax.text(4.8, 5.8, "Environment (all of physics)", fontsize=10, ha='center')
    agent = FancyBboxPatch((5.5, 2.0), 2.4, 1.9, boxstyle="round,pad=0.08",
                            facecolor='#dbe4f0', edgecolor='#2b3a67', lw=1.8)
    ax.add_patch(agent)
    ax.text(6.7, 2.95, "Agent's own\ncode / hardware\nis part of $\\mu$", ha='center',
            va='center', fontsize=8.6, color='#2b3a67', fontweight='bold')
    ax.annotate('', xy=(5.5, 2.4), xytext=(3.0, 2.4),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.3, color='black'))
    ax.text(4.2, 2.65, "can self-modify,\ncan be destroyed", fontsize=8.2, ha='center')

    fig.tight_layout()
    savefig(fig, "embedded_intelligence")


# ---------------------------------------------------------------------------
# Figure: Control problem -- capability vs. controllability trade-off (15.3)
# ---------------------------------------------------------------------------
def fig_control_problem():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.linspace(0, 10, 200)
    capability = 1 / (1 + np.exp(-1.1 * (x - 5)))
    controllable = 1 - capability
    ax.plot(x, capability, lw=2.4, color='#2b3a67', label="Agent capability / agency")
    ax.plot(x, controllable, lw=2.4, color='#c0392b', label="Ease of human control")
    ax.axvline(5, color='gray', ls='--', lw=1.2)
    ax.text(5.15, 0.05, "crossover:\nagent can resist\nbeing controlled", fontsize=9.3,
            color='gray')
    ax.set_xlabel("Agent intelligence / agency $\\longrightarrow$", fontsize=11)
    ax.set_ylabel("Level", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper center', fontsize=9.6, frameon=True)
    ax.set_title("The Control Problem: the more capable the agent,\n"
                 "the harder it is to keep it under control", fontsize=11)
    savefig(fig, "control_problem")


if __name__ == "__main__":
    fig_instrumental_convergence()
    fig_orthogonality_thesis()
    fig_death_state()
    fig_self_modification()
    fig_wireheading_loop()
    fig_embedded_intelligence()
    fig_control_problem()
    print("All figures generated in", OUTDIR)
