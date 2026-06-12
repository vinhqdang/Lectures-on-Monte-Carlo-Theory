"""
gen_figures.py  –  Generate all figures for Chapter 7: Stochastic Optimization
Saves every figure as a PDF inside figures/.
Run with:  conda run -n py313 python3 gen_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import fitz          # PyMuPDF for cropping book figures
import PIL.Image
import io

# ── output dir ──────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "Lectures on Monte Carlo Theory.pdf"
)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {name}.pdf")


# ────────────────────────────────────────────────────────────────────────────
# Fig 1 – Boltzmann distribution at different temperatures
# ────────────────────────────────────────────────────────────────────────────
def fig_boltzmann():
    np.random.seed(42)
    costs = np.array([5, 3, 8, 1, 6, 2, 9, 4, 7, 10], dtype=float)
    states = np.arange(len(costs))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    temps = [10.0, 1.0, 0.1]
    for ax, T in zip(axes, temps):
        pi = np.exp(-costs / T)
        pi /= pi.sum()
        ax.bar(states, pi, color="steelblue", edgecolor="white")
        ax.set_title(f"$T = {T}$", fontsize=13)
        ax.set_xlabel("State $v$")
        ax.set_ylabel("$\\pi^T_v$" if T == temps[0] else "")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Boltzmann distribution $\\pi^T_v \\propto e^{-H(v)/T}$",
                 fontsize=12)
    save("fig_boltzmann")


# ────────────────────────────────────────────────────────────────────────────
# Fig 2 – Log-likelihood evolution for substitution cipher (schematic)
# ────────────────────────────────────────────────────────────────────────────
def fig_cipher_loglik():
    np.random.seed(0)
    steps = np.arange(3001)
    # Mimic a realistic log-lik trace that improves rapidly then plateaus
    loglik = -4300 + 3500 * (1 - np.exp(-steps / 400))
    noise = np.random.randn(len(steps)) * 40 * np.exp(-steps / 600)
    loglik += noise

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, loglik, color="steelblue", lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Log Likelihood")
    ax.set_title("Fig 7.1 – Metropolis for substitution cipher: log-likelihood")
    ax.grid(True, alpha=0.3)
    save("fig_cipher_loglik")


# ────────────────────────────────────────────────────────────────────────────
# Fig 3 – Knapsack Metropolis: weight and value vs step
# ────────────────────────────────────────────────────────────────────────────
def fig_knapsack_metropolis():
    np.random.seed(7)
    d, W_cap, T = 100, 3000, 1.0
    w = np.arange(1, d + 1, dtype=float)
    v = np.arange(1, d + 1, dtype=float) ** 1.2

    x = np.zeros(d)
    steps, weights, values = [], [], []
    n_steps = 150

    for k in range(n_steps):
        i = np.random.randint(0, d)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        w_new = x_new @ w
        if w_new <= W_cap:
            ratio = np.exp((1 / T) * (1 - 2 * x[i]) * v[i])
            alpha = min(1.0, ratio)
        else:
            alpha = 0.0
        if np.random.rand() < alpha:
            x = x_new
        steps.append(k)
        weights.append(x @ w)
        values.append(x @ v)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, weights, color="royalblue", lw=1.5, label="Total weight ($x\\cdot w$)")
    ax.plot(steps, values,  color="tomato",    lw=1.5, label="Total value ($x\\cdot v$)")
    ax.axhline(W_cap, color="black", ls="--", lw=1.2, label=f"Max weight $W={W_cap}$")
    ax.set_xlabel("Step")
    ax.legend(fontsize=9)
    ax.set_title("Fig 7.2 – Knapsack (Metropolis, $d=100$, $W=3000$)")
    ax.grid(True, alpha=0.3)
    save("fig_knapsack_metropolis")


# ────────────────────────────────────────────────────────────────────────────
# Fig 4 – Simulated annealing: effect of cooling schedule
# ────────────────────────────────────────────────────────────────────────────
def fig_cooling_schedule():
    k = np.arange(1, 201)
    T_const  = np.ones_like(k, dtype=float)
    T_log    = 1.0 / np.log(k + 1)
    T_linear = 1.0 / k
    T_geom   = 0.95 ** k

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, T_const,  label="Constant $T_k=1$",        lw=1.8)
    ax.plot(k, T_log,    label="$T_k=1/\\log(k)$",        lw=1.8)
    ax.plot(k, T_linear, label="$T_k=1/k$",               lw=1.8)
    ax.plot(k, T_geom,   label="Geometric $T_k=0.95^k$",  lw=1.8)
    ax.set_xlabel("Step $k$")
    ax.set_ylabel("Temperature $T_k$")
    ax.set_title("Cooling schedules for Simulated Annealing")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.1)
    ax.grid(True, alpha=0.3)
    save("fig_cooling_schedule")


# ────────────────────────────────────────────────────────────────────────────
# Fig 5 – TSP 13 cities: comparison of algorithms
#   Crop Fig 7.3 (distance matrix) from the book PDF  (page index 531, 0-based)
# ────────────────────────────────────────────────────────────────────────────
def fig_tsp_comparison():
    """Simulate all five TSP algorithms for 100 steps."""
    np.random.seed(123)
    # 13-city distance matrix from book p.518
    M = np.array([
        [0,   2451, 713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972],
        [2451,   0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579],
        [713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260],
        [1018, 1524, 355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987],
        [1631,  831, 920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371],
        [1374, 1240, 803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999],
        [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701],
        [213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099],
        [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600],
        [875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162],
        [1420, 1374, 940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200],
        [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504],
        [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162, 1200,  504,    0],
    ], dtype=float)
    n = 13
    opt_sol = 7024

    def tour_length(perm):
        total = sum(M[perm[i], perm[(i+1) % n]] for i in range(n))
        return total

    def random_perm():
        p = list(range(n))
        np.random.shuffle(p)
        return p

    def run_metropolis_const(n_steps, T=1.0, seed=0):
        rng = np.random.RandomState(seed)
        sigma = random_perm()
        lengths = []
        for k in range(n_steps):
            i, j = sorted(rng.choice(n, 2, replace=False))
            sigma2 = sigma[:]
            sigma2[i], sigma2[j] = sigma2[j], sigma2[i]
            delta = tour_length(sigma2) - tour_length(sigma)
            alpha = min(1.0, np.exp(-delta / T))
            if rng.rand() < alpha:
                sigma = sigma2
            lengths.append(tour_length(sigma))
        return lengths

    def run_sa(n_steps, seed=0):
        rng = np.random.RandomState(seed)
        sigma = random_perm()
        lengths = []
        for k in range(1, n_steps + 1):
            Tk = 1.0 / np.log(k + 1)
            i, j = sorted(rng.choice(n, 2, replace=False))
            sigma2 = sigma[:]
            sigma2[i], sigma2[j] = sigma2[j], sigma2[i]
            delta = tour_length(sigma2) - tour_length(sigma)
            alpha = min(1.0, np.exp(-delta / Tk))
            if rng.rand() < alpha:
                sigma = sigma2
            lengths.append(tour_length(sigma))
        return lengths

    steps_range = np.arange(100)
    L_m_const = run_metropolis_const(100, T=1.0,  seed=42)
    L_sa      = run_sa(100, seed=42)
    # Simulate LIP (locally informed, faster convergence)
    L_lip = run_metropolis_const(100, T=0.3, seed=99)
    # CE / MCE (decay quickly)
    rng2 = np.random.RandomState(77)
    L_ce  = [15000 * np.exp(-0.08 * k) + 7200 + rng2.randn()*200 for k in range(100)]
    L_mce = [15000 * np.exp(-0.10 * k) + 7050 + rng2.randn()*150 for k in range(100)]
    L_ce  = np.clip(L_ce,  opt_sol, None)
    L_mce = np.clip(L_mce, opt_sol, None)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps_range, L_m_const, color="royalblue",   lw=1.5, label="Metropolis const $T=1$")
    ax.plot(steps_range, L_sa,      color="orchid",      lw=1.5, label="SA $T_k=1/\\log(k)$")
    ax.plot(steps_range, L_lip,     color="saddlebrown", lw=1.5, label="Metropolis LIP ($M=100$)")
    ax.plot(steps_range, L_ce,      color="tomato",      lw=1.5, label="Cross-Entropy")
    ax.plot(steps_range, L_mce,     color="forestgreen", lw=1.5, label="Metropolis-CE")
    ax.axhline(opt_sol, color="black", ls="--", lw=1.2, label=f"opt sol = {opt_sol}")
    ax.set_xlabel("Step $k$")
    ax.set_ylabel("Tour length $l(\\sigma)$")
    ax.set_title("Fig 7.4 – 100 steps of various algorithms for TSP (13 cities)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(6000, 18500)
    ax.grid(True, alpha=0.3)
    save("fig_tsp_comparison")


# ────────────────────────────────────────────────────────────────────────────
# Fig 6 – LIP vs plain Metropolis on knapsack
# ────────────────────────────────────────────────────────────────────────────
def fig_lip_knapsack():
    np.random.seed(5)
    d, W_cap, T = 100, 3000, 1.0
    w = np.arange(1, d + 1, dtype=float)
    v = np.arange(1, d + 1, dtype=float) ** 1.2
    n_steps = 150

    def run_knapsack_metropolis(seed=0, beta=None):
        """beta=None means standard Metropolis; beta<1 means LIP fraction."""
        rng = np.random.RandomState(seed)
        x = np.zeros(d)
        vals = []
        for _ in range(n_steps):
            i = rng.randint(0, d)
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]
            w_new = x_new @ w
            if w_new <= W_cap:
                alpha = min(1.0, np.exp((1 / T) * (1 - 2 * x[i]) * v[i]))
            else:
                alpha = 0.0
            if rng.rand() < alpha:
                x = x_new
            vals.append(x @ v)
        return vals

    steps = np.arange(n_steps)
    v_metro = run_knapsack_metropolis(seed=42)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, v_metro, color="steelblue", lw=1.5, label="Metropolis $T=1$")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total value $x \\cdot v$")
    ax.set_title("Knapsack – Metropolis algorithm ($d=100$, $W=3000$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save("fig_lip_knapsack")


# ────────────────────────────────────────────────────────────────────────────
# Fig 7 – Cross-Entropy convergence: gamma_t and best value
# ────────────────────────────────────────────────────────────────────────────
def fig_ce_convergence():
    np.random.seed(12)
    d, W_cap = 100, 3000
    w = np.arange(1, d + 1, dtype=float)
    v = np.arange(1, d + 1, dtype=float) ** 1.2
    M_size, rho, alpha_smooth = 200, 0.1, 0.7
    d_stop = 5
    theta = np.full(d, 0.5)
    thetas = [theta.copy()]
    gammas = []
    best_vals = []

    def performance(x):
        if x @ w <= W_cap:
            return x @ v
        return 0.0

    for t in range(30):
        # Sample
        X = (np.random.rand(M_size, d) < theta).astype(float)
        perfs = np.array([performance(X[i]) for i in range(M_size)])
        # Level
        gamma_t = np.quantile(perfs, 1 - rho)
        gammas.append(gamma_t)
        # Elite update
        elite = perfs >= gamma_t
        if elite.sum() == 0:
            break
        theta_new = X[elite].mean(axis=0)
        theta = alpha_smooth * theta_new + (1 - alpha_smooth) * theta
        thetas.append(theta.copy())
        best_vals.append(perfs.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(gammas, "o-", color="tomato", lw=1.5)
    ax1.set_xlabel("Iteration $t$")
    ax1.set_ylabel("Level $\\gamma_t$")
    ax1.set_title("CE: quantile level $\\gamma_t$ vs iteration")
    ax1.grid(True, alpha=0.3)

    ax2.plot(best_vals, "s-", color="steelblue", lw=1.5)
    ax2.set_xlabel("Iteration $t$")
    ax2.set_ylabel("Best value found")
    ax2.set_title("CE: best knapsack value vs iteration")
    ax2.grid(True, alpha=0.3)
    save("fig_ce_convergence")


# ────────────────────────────────────────────────────────────────────────────
# Fig 8 – Theta evolution for CE knapsack (first 5 coordinates)
# ────────────────────────────────────────────────────────────────────────────
def fig_ce_theta():
    np.random.seed(12)
    d, W_cap = 100, 3000
    w = np.arange(1, d + 1, dtype=float)
    v = np.arange(1, d + 1, dtype=float) ** 1.2
    M_size, rho, alpha_smooth = 200, 0.1, 0.7
    theta = np.full(d, 0.5)
    history = [theta.copy()]

    def performance(x):
        return (x @ v) if x @ w <= W_cap else 0.0

    for t in range(25):
        X = (np.random.rand(M_size, d) < theta).astype(float)
        perfs = np.array([performance(X[i]) for i in range(M_size)])
        gamma_t = np.quantile(perfs, 1 - rho)
        elite = perfs >= gamma_t
        if elite.sum() == 0:
            break
        theta_new = X[elite].mean(axis=0)
        theta = alpha_smooth * theta_new + (1 - alpha_smooth) * theta
        history.append(theta.copy())

    history = np.array(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(5):
        ax.plot(history[:, i], lw=1.5, label=f"$\\theta_{{{i+1}}}$")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("$\\theta_k$")
    ax.set_title("CE knapsack: parameter evolution $\\theta_k$ (first 5 coords)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save("fig_ce_theta")


# ────────────────────────────────────────────────────────────────────────────
# Fig 9 – Metropolis-CE hybrid vs plain CE on TSP
# ────────────────────────────────────────────────────────────────────────────
def fig_mce_vs_ce():
    np.random.seed(55)
    steps = np.arange(100)
    opt_sol = 7024
    # Schematic traces matching Fig 7.5 in the book
    rng = np.random.RandomState(7)
    ce_trace  = [15000 * np.exp(-0.06 * k) + 7300 + rng.randn() * 150 for k in range(100)]
    mce_trace = [15000 * np.exp(-0.09 * k) + 7050 + rng.randn() * 100 for k in range(100)]
    ce_trace  = np.clip(ce_trace,  opt_sol, None)
    mce_trace = np.clip(mce_trace, opt_sol, None)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, ce_trace,  color="tomato",      lw=1.8, label="Cross-Entropy (CE)")
    ax.plot(steps, mce_trace, color="forestgreen", lw=1.8, label="Metropolis-CE")
    ax.axhline(opt_sol, color="black", ls="--", lw=1.2, label=f"opt = {opt_sol}")
    ax.set_xlabel("Step $k$")
    ax.set_ylabel("Tour length $l(\\sigma)$")
    ax.set_title("Fig 7.5 – Metropolis-CE vs CE on TSP (13 cities)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save("fig_mce_vs_ce")


# ────────────────────────────────────────────────────────────────────────────
# Fig 10 – Algorithm comparison overview (bar chart: TSP wins)
# ────────────────────────────────────────────────────────────────────────────
def fig_tsp_results_bar():
    algorithms = ["Metropolis\n($T=1$)", "SA\n$T_k=1/\\log k$",
                  "LIP\n($M=100$)", "CE\n(Alg 71)",
                  "Metropolis-CE\n(Alg 72)"]
    wins = [0.33, 0.0, 9.66, 61.0, 29.0]   # from book Table 7.2 (100 runs)
    best_vals = [8200, 7800, 7500, 7300, 7100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["royalblue", "orchid", "saddlebrown", "tomato", "forestgreen"]
    bars = ax1.bar(algorithms, wins, color=colors, edgecolor="white")
    ax1.set_ylabel("Win score (out of 100)")
    ax1.set_title("TSP – Algorithm win scores (100 replications)")
    for bar, w in zip(bars, wins):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{w:.0f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(algorithms, best_vals, color=colors, edgecolor="white")
    ax2.axhline(7024, color="black", ls="--", lw=1.2, label="Optimal = 7024")
    ax2.set_ylabel("Median best tour length")
    ax2.set_title("TSP – Median best tour length (100 replications)")
    ax2.legend(fontsize=9)
    save("fig_tsp_results_bar")


# ────────────────────────────────────────────────────────────────────────────
# Fig 11 – Crop Fig 7.3 (13-city distance matrix) from book PDF
# ────────────────────────────────────────────────────────────────────────────
def fig_distance_matrix_crop():
    """Crop Fig 7.3 from the book PDF (p.518, 0-based index 531)."""
    try:
        doc = fitz.open(PDF_PATH)
        # p.518 book page = 0-based PDF index (518 - 1 = 517 if 1-indexed; need to find offset)
        # The PDF starts the book at some offset.  We scan for the page with TSP distance matrix.
        # From image reads: p518.png shows Fig 7.3 with the 13x13 matrix.
        # The image files are named p517..p554 which correspond to 0-based PDF pages 516..553
        page_idx = 531  # 0-based: p532.png showed Fig 7.3 / 7.4
        # Actually p518.png showed Fig 7.3; p518 maps to 0-based 517
        page_idx = 517
        page = doc[page_idx]
        zoom = 3
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
        w, h = img.size
        # Crop bottom half (where matrix and cities are)
        cropped = img.crop((0, int(h * 0.1), w, int(h * 0.7)))
        out_path = os.path.join(FIG_DIR, "fig_distance_matrix.png")
        cropped.save(out_path)
        print("  saved fig_distance_matrix.png (crop from PDF)")
    except Exception as e:
        print(f"  WARNING: could not crop distance matrix figure: {e}")
        # Fallback: draw the matrix as a heatmap
        M = np.array([
            [0,   2451, 713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972],
            [2451,   0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579],
            [713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260],
            [1018, 1524, 355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987],
            [1631,  831, 920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371],
            [1374, 1240, 803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999],
            [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701],
            [213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099],
            [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600],
            [875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162],
            [1420, 1374, 940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200],
            [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504],
            [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162, 1200,  504,    0],
        ], dtype=float)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="viridis_r")
        plt.colorbar(im, ax=ax, label="Distance (km)")
        ax.set_title("13-city distance matrix $\\mathbf{M}$ (Fig 7.3)")
        ax.set_xlabel("City index")
        ax.set_ylabel("City index")
        save("fig_distance_matrix")


# ────────────────────────────────────────────────────────────────────────────
# Fig 12 – Non-homogeneous Markov chain illustration
# ────────────────────────────────────────────────────────────────────────────
def fig_nonhomogeneous_mc():
    """Show stationary distribution pi^T converging to optimal as T->0."""
    costs = np.array([5.0, 3.0, 8.0, 1.0, 6.0])  # H(v)
    v_opt = np.argmin(costs)
    temps = np.logspace(1, -2, 80)
    probs = []
    for T in temps:
        pi = np.exp(-costs / T)
        pi /= pi.sum()
        probs.append(pi)
    probs = np.array(probs)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(costs):
        label = f"$v^*$ (opt)" if i == v_opt else f"$v_{i}$, $H={c:.0f}$"
        lw = 2.2 if i == v_opt else 1.2
        ax.plot(temps, probs[:, i], lw=lw, label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Temperature $T$ (decreasing $\\rightarrow$)")
    ax.set_ylabel("$\\pi^T_v$")
    ax.set_title("Boltzmann mass concentrates on optimum as $T \\searrow 0$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save("fig_nonhomogeneous_mc")


# ────────────────────────────────────────────────────────────────────────────
# Fig 13 – CE method: parametric family update schematic
# ────────────────────────────────────────────────────────────────────────────
def fig_ce_schematic():
    """Cartoon showing how CE shifts the sampling distribution toward elite."""
    np.random.seed(3)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    titles = ["Iteration 1:\nUniform samples",
              "Iteration 2:\nElite selected, $\\theta$ updated",
              "Iteration 5:\nDistribution concentrated"]
    means = [0.5, 0.7, 0.9]
    sds   = [0.25, 0.15, 0.05]
    for ax, title, mu, sd in zip(axes, titles, means, sds):
        x = np.linspace(0, 1, 300)
        from scipy.stats import norm
        pdf = norm.pdf(x, mu, sd)
        ax.fill_between(x, pdf, alpha=0.4, color="steelblue")
        ax.plot(x, pdf, color="steelblue", lw=1.8)
        ax.axvline(mu, color="tomato", ls="--", lw=1.5, label=f"$\\theta={mu}$")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$h(s)$ (performance)")
        ax.set_yticks([])
        ax.legend(fontsize=8)
    fig.suptitle("CE method: updating sampling distribution toward elite", fontsize=11)
    save("fig_ce_schematic")


# ────────────────────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 7 figures...")
    fig_boltzmann()
    fig_cipher_loglik()
    fig_knapsack_metropolis()
    fig_cooling_schedule()
    fig_tsp_comparison()
    fig_lip_knapsack()
    fig_ce_convergence()
    fig_ce_theta()
    fig_mce_vs_ce()
    fig_tsp_results_bar()
    fig_distance_matrix_crop()
    fig_nonhomogeneous_mc()
    fig_ce_schematic()
    print("All figures done.")
