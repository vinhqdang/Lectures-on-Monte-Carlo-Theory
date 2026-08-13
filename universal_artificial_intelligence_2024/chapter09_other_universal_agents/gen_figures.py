"""
gen_figures.py
Generate all figures for Chapter 9: "Other Universal Agents" slides.

Run with:
    conda run -n py313 python3 gen_figures.py

Produces (in ./figures/):
    toy_environments.png       -- Figure 9.1 cropped from the book PDF (state diagrams)
    optimism_bound.pdf         -- illustration of Theorem 9.1.2 (finite error bound)
    thompson_sampling.pdf      -- Thompson sampling posterior convergence
    ksa_infogain.pdf           -- expected information gain per action-prefix (Example 9.3.11)
    bayesexp_timeline.pdf      -- BayesExp explore/exploit decision timeline
    selfaixi_schematic.pdf     -- dual Bayesian-mixture schematic for Self-AIXI
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(HERE), "An Introduction to Universal Artificial Intelligence 2024.pdf"
)

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
}


# ─────────────────────────────────────────────────────────────────────────
# Figure 0: crop Figure 9.1 (toy environments) directly from the book PDF
# ─────────────────────────────────────────────────────────────────────────
def crop_toy_environments():
    try:
        import fitz  # pymupdf
        import PIL.Image
    except ImportError:
        print("pymupdf/PIL not available -- skipping crop_toy_environments()")
        return
    doc = fitz.open(BOOK_PDF)
    page = doc[297]  # 0-indexed page containing printed page 277 / Figure 9.1
    clip = fitz.Rect(38, 448, 436, 552)
    pix = page.get_pixmap(matrix=fitz.Matrix(6, 6), clip=clip)
    img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
    img.save(os.path.join(FIGDIR, "toy_environments.png"))
    print("Saved toy_environments.png", img.size)


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Theorem 9.1.2 finite error bound illustration
# ─────────────────────────────────────────────────────────────────────────
def fig_optimism_bound():
    gamma, eps = 0.8, 0.2
    ell_plus_1 = int(np.ceil(np.log(eps) / np.log(gamma)))  # = 8
    M = 3
    bound = (M - 1) * ell_plus_1  # = 16

    T = 60
    t = np.arange(1, T + 1)
    Mt = np.full(T, M, dtype=float)
    gap = np.zeros(T)

    # Two "inconsistency events": at t=6 and t=32, an environment is refuted.
    events = [6, 32]
    for i, te in enumerate(events):
        Mt[te - 1:] -= 1
        window = slice(te - 1, min(te - 1 + ell_plus_1, T))
        decay = eps * np.exp(-0.6 * np.arange(0, min(ell_plus_1, T - te + 1)))
        gap[window] = np.maximum(gap[window], decay)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.3]})

    ax = axes[0]
    ax.step(t, Mt, where="post", color=COL["blue"], linewidth=2.4)
    ax.set_ylabel(r"$|\mathcal{M}_t|$")
    ax.set_yticks([1, 2, 3])
    ax.set_title(r"Consistent environment class $\mathcal{M}_t$ shrinks as history rules out hypotheses")
    ax.set_ylim(0.5, 3.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, gap, color=COL["red"], linewidth=2.0)
    ax.axhline(eps, color=COL["gray"], linestyle="--", linewidth=1.4,
                label=rf"$\varepsilon={eps}$")
    for te in events:
        ax.axvspan(te, min(te + ell_plus_1 - 1, T), color=COL["orange"], alpha=0.15)
    ax.set_ylabel(r"$\max_\pi V^\pi_\mu(h_{<t}) - V^{\pi^\circ}_\mu(h_{<t})$")
    ax.set_xlabel(r"time step $t$")
    ax.set_title(rf"Suboptimality gap: exceeds $\varepsilon$ on $\leq |\mathcal{{M}}|\cdot(\ell{{+}}1)={bound}$ steps (shaded)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "optimism_bound.pdf"))
    plt.close(fig)
    print("Saved optimism_bound.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Thompson sampling posterior convergence (3-armed Bernoulli bandit)
# ─────────────────────────────────────────────────────────────────────────
def fig_thompson_sampling():
    rng = np.random.default_rng(7)
    true_p = np.array([0.30, 0.50, 0.80])
    K = len(true_p)
    T = 300
    alpha = np.ones(K)
    beta_ = np.ones(K)
    chosen = np.zeros(T, dtype=int)
    running_frac = np.zeros((T, K))

    for tt in range(T):
        samples = rng.beta(alpha, beta_)
        a = int(np.argmax(samples))
        chosen[tt] = a
        r = rng.binomial(1, true_p[a])
        alpha[a] += r
        beta_[a] += (1 - r)
        counts = np.bincount(chosen[: tt + 1], minlength=K)
        running_frac[tt] = counts / (tt + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    colors = [COL["red"], COL["orange"], COL["green"]]
    labels = [rf"$\nu_{i+1}$ ($p={true_p[i]:.1f}$)" for i in range(K)]
    for i in range(K):
        ax.plot(np.arange(1, T + 1), running_frac[:, i], color=colors[i],
                 linewidth=2.0, label=labels[i])
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"fraction of steps sampled so far")
    ax.set_title("Thompson sampling: which environment gets\nsampled from the posterior")
    ax.legend(loc="center right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    xs = np.linspace(0, 1, 400)
    for i in range(K):
        pdf = stats.beta(alpha[i], beta_[i]).pdf(xs)
        ax.plot(xs, pdf, color=colors[i], linewidth=2.0, label=labels[i])
        ax.axvline(true_p[i], color=colors[i], linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"believed success probability")
    ax.set_ylabel("posterior density")
    ax.set_title(rf"Posterior over each $\nu_i$ after $t={T}$ steps")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "thompson_sampling.pdf"))
    plt.close(fig)
    print("Saved thompson_sampling.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Expected information gain per action-prefix (Example 9.3.11)
# ─────────────────────────────────────────────────────────────────────────
def step(state, action, env):
    """One step of the toy automaton (Figure 9.1) for environment env in {1,2,3}."""
    obs_q1_a0 = {1: 0, 2: 0, 3: 1}[env]
    obs_q2_a0 = {1: 0, 2: 1, 3: 0}[env]
    if state == "q0":
        return ("q1", 0) if action == 0 else ("q2", 0)
    if state == "q1":
        return ("q1", obs_q1_a0 if action == 0 else 0)
    if state == "q2":
        return ("q2", obs_q2_a0 if action == 0 else 0) if action == 0 else ("q0", 0)
    raise ValueError(state)


def run_actions(actions, env):
    state = "q0"
    obs = []
    for a in actions:
        state, o = step(state, a, env)
        obs.append(o)
    return tuple(obs)


def expected_info_gain(actions, envs=(1, 2, 3), prior=None):
    """Sum over t of KL(posterior_t || posterior_{t-1}), expected over the true env
    under a uniform prior over which env is true (matches Definition 9.3.1/9.3.2,
    undiscounted, m = len(actions))."""
    if prior is None:
        prior = {e: 1.0 / len(envs) for e in envs}
    obs_by_env = {e: run_actions(actions, e) for e in envs}
    total = 0.0
    for true_env in envs:
        w = {e: prior[e] for e in envs}  # w(.|h_<1)
        gain = 0.0
        for k in range(len(actions)):
            ok = obs_by_env[true_env][k]
            w_new = {}
            for e in envs:
                w_new[e] = w[e] if obs_by_env[e][:k + 1] == obs_by_env[true_env][:k + 1] else 0.0
            z = sum(w_new.values())
            if z > 0:
                w_new = {e: v / z for e, v in w_new.items()}
            for e in envs:
                if w_new[e] > 0 and w[e] > 0:
                    gain += w_new[e] * np.log(w_new[e] / w[e])
            w = w_new
        total += prior[true_env] * gain
    return total


def fig_ksa_infogain():
    candidates = ["10100", "00000", "11111", "01010", "10101", "11000", "10010", "01100"]
    values = []
    for c in candidates:
        actions = [int(ch) for ch in c]
        values.append(expected_info_gain(actions))

    order = np.argsort(values)
    candidates = [candidates[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = [COL["green"] if c == "10100" else COL["blue"] for c in candidates]
    bars = ax.barh(candidates, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=10)
    ax.set_xlabel(r"expected total information gain $\;\mathbf{E}_\xi^{\pi}[\mathrm{IG}_{1:5}]$")
    ax.set_title(r"KSA agent's ranking of length-5 action prefixes $\dot{a}_{1:5}$" "\n"
                 r"($\dot{a}_{1:5}=\mathtt{10100}$ wins -- matches Example 9.3.11)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ksa_infogain.pdf"))
    plt.close(fig)
    print("Saved ksa_infogain.pdf; values:", dict(zip(candidates, values)))


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: BayesExp explore/exploit timeline
# ─────────────────────────────────────────────────────────────────────────
def fig_bayesexp_timeline():
    rng = np.random.default_rng(3)
    T = 80
    t = np.arange(1, T + 1)
    beta_exp = 0.4
    eps_t = t ** (-beta_exp)

    # Synthetic expected information gain of pi*_IG: decays with noisy bumps
    base = 0.9 * np.exp(-t / 25.0)
    bumps = 0.15 * np.exp(-((t - 45) ** 2) / 40.0)
    VIG = np.clip(base + bumps + rng.normal(0, 0.02, T), 0, None)

    explore = VIG > eps_t

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(t, eps_t, color=COL["gray"], linestyle="--", linewidth=1.8,
             label=r"exploration threshold $\varepsilon_t = t^{-0.4}$")
    ax.plot(t, VIG, color=COL["purple"], linewidth=2.0,
             label=r"max expected info. gain $V_{\mathrm{IG}}^{*,\,t+H_t(\varepsilon_t)}(h_{<t})$")
    ax.fill_between(t, 0, ax.get_ylim()[1] if False else 1.05, where=explore,
                     color=COL["orange"], alpha=0.18, label="explore (follow $\\pi^*_{\\mathrm{IG}}$)")
    ax.fill_between(t, 0, 1.05, where=~explore,
                     color=COL["blue"], alpha=0.10, label=r"exploit (follow $\pi^*_\xi$)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, T)
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel("value")
    ax.set_title("BayesExp: switching between exploration and exploitation")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "bayesexp_timeline.pdf"))
    plt.close(fig)
    print("Saved bayesexp_timeline.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5: Self-AIXI dual-mixture schematic
# ─────────────────────────────────────────────────────────────────────────
def fig_selfaixi_schematic():
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=11.5):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                            linewidth=1.6, edgecolor=color, facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return b

    def arrow(p, q, color="black", style="-|>", lw=1.8, rad=0.0):
        a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16,
                             linewidth=lw, color=color,
                             connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)

    hist = box(0.3, 2.6, 2.0, 0.9, r"history" "\n" r"$h_{<t}$", "#5d6d7e")
    envmix = box(3.4, 4.0, 2.6, 1.1, r"environment mixture" "\n" r"$\xi(\cdot|h_{<t}a_t)=\sum_\nu w_\nu\,\nu$",
                 "#1f5fa8", fontsize=10.5)
    polmix = box(3.4, 0.9, 2.6, 1.1, r"policy mixture" "\n" r"$\zeta(a_t|h_{<t})=\sum_\pi \omega_\pi\,\pi(a_t|h_{<t})$",
                 "#c0392b", fontsize=10.5)
    qbox = box(6.9, 2.5, 2.2, 1.1, r"$Q_\xi^{\zeta}(h_{<t}, a_t)$", "#1e8449", fontsize=12.5)
    act = box(9.3, 2.6, 0.55, 0.9, r"$a_t$" "\n" r"$=$" "\n" r"argmax", "#d68910", fontsize=9)

    arrow((2.3, 3.3), (3.4, 4.3), color="#1f5fa8", rad=0.15)
    arrow((2.3, 3.1), (3.4, 1.7), color="#c0392b", rad=-0.15)
    arrow((6.0, 4.4), (7.0, 3.2), color="#1f5fa8", rad=-0.15)
    arrow((6.0, 1.6), (7.0, 2.8), color="#c0392b", rad=0.15)
    arrow((9.1, 3.0), (9.3, 3.05), color="#d68910")
    arrow((7.9, 3.05), (9.3, 3.05), color="#d68910")

    # feedback loop: action appended to history, updates omega_pi via zeta
    arrow((9.55, 2.6), (9.55, 0.4), color="black", lw=1.4, rad=0.0)
    arrow((9.55, 0.4), (1.3, 0.4), color="black", lw=1.4, rad=0.0)
    arrow((1.3, 0.4), (1.3, 2.6), color="black", lw=1.4, rad=0.0)
    ax.text(5.3, 0.15, r"$a_t$ appended to history $\Rightarrow$ updates weights $\omega_\pi \propto \omega_\pi\,\pi(a_t|h_{<t})$",
            ha="center", va="center", fontsize=9.5, style="italic", color="#333333")

    ax.set_title("Self-AIXI: two Bayesian mixtures (over environments and over policies)\n"
                 "jointly determine the one-step-optimal action", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "selfaixi_schematic.pdf"))
    plt.close(fig)
    print("Saved selfaixi_schematic.pdf")


if __name__ == "__main__":
    crop_toy_environments()
    fig_optimism_bound()
    fig_thompson_sampling()
    fig_ksa_infogain()
    fig_bayesexp_timeline()
    fig_selfaixi_schematic()
    print("All figures generated in", FIGDIR)
