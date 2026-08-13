"""
Figure generator for Chapter 7 -- Universal Artificial Intelligence
(An Introduction to Universal Artificial Intelligence, Hutter/Quarel/Catt 2024)

Run with:
    conda run -n py313 python3 gen_figures.py

All plotted figures are saved as PDF into ./figures/
One diagram (Figure 7.1, the taxonomy-of-environments figure) is a book
diagram that is not naturally re-plottable, so it is cropped directly from
the book PDF with pymupdf and saved as a PNG into ./figures/.
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

BOOK_PDF = os.path.join(
    os.path.dirname(HERE),
    "An Introduction to Universal Artificial Intelligence 2024.pdf",
)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10.5,
    'figure.dpi': 150,
})

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------
# Figure 1: Bayesian mixture posterior update (Example 7.2.3, coin flip)
# ----------------------------------------------------------------------
def fig_mixture_weights():
    """
    Reproduce Example 7.2.3 exactly: M = {nu_HH, nu_HT, nu_TT}, uniform
    prior w_nu = 1/3. Agent takes a_1 = H, environment returns (o_1,r_1) =
    (H,1). We plot the prior weights and the exact posterior weights
    w_nu(h_1) computed in the book: 2/3, 1/3, 0.
    """
    envs = [r'$\nu_{HH}$', r'$\nu_{HT}$', r'$\nu_{TT}$']
    prior = np.array([1/3, 1/3, 1/3])
    posterior = np.array([2/3, 1/3, 0.0])

    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b1 = ax.bar(x - width/2, prior, width, label=r'prior $w_\nu$',
                color='#8aa6c2', edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + width/2, posterior, width,
                label=r'posterior $w_\nu(h_1)$ after $(a_1,o_1)=(H,H)$',
                color='#c26a4d', edgecolor='black', linewidth=0.6)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=13)
    ax.set_ylabel('mixture weight')
    ax.set_ylim(0, 0.85)
    ax.set_title('Bayes mixture: belief update after observing one Head')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'mixture_weights.pdf'))
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: Merging-of-opinions / on-policy value convergence
#   Simulate a small class M of 3 stochastic Bernoulli environments,
#   a Bayes mixture xi over them with a uniform prior, and show that as
#   the agent observes the true environment mu acting under some fixed
#   policy pi (here: always predict "guess most likely symbol so far"),
#   the total-variation-style gap between xi and mu predictions --
#   and the self-optimizing value gap delta_t = V*_mu - V^{pi*_xi}_mu --
#   shrink towards 0. This mirrors Theorem 7.2.6 / Theorem 7.3.1.
# ----------------------------------------------------------------------
def fig_convergence():
    T = 200
    thetas = {'nu_1': 0.2, 'nu_2': 0.5, 'nu_3': 0.8}
    true_env = 'nu_3'
    theta_true = thetas[true_env]

    n_trials = 60
    gap_curves = np.zeros((n_trials, T))
    delta_curves = np.zeros((n_trials, T))

    for trial in range(n_trials):
        w = {k: 1/3 for k in thetas}
        x_seq = RNG.binomial(1, theta_true, size=T)
        for t in range(T):
            # xi's current predictive probability of the next symbol = 1
            xi_pred = sum(w[k] * thetas[k] for k in thetas)
            mu_pred = theta_true
            gap_curves[trial, t] = abs(xi_pred - mu_pred)
            # self-optimizing value gap proxy: |E_mu[reward under
            # xi-greedy policy] - E_mu[reward under mu-optimal policy]|,
            # where reward = 1{action == outcome}, greedy action = round(pred)
            a_xi = 1 if xi_pred >= 0.5 else 0
            a_mu = 1 if mu_pred >= 0.5 else 0
            r_xi = theta_true if a_xi == 1 else (1 - theta_true)
            r_mu = theta_true if a_mu == 1 else (1 - theta_true)
            delta_curves[trial, t] = max(r_mu - r_xi, 0.0)
            # Bayes update using the realised symbol x_t
            x_t = x_seq[t]
            like = {k: (thetas[k] if x_t == 1 else 1 - thetas[k]) for k in thetas}
            norm = sum(w[k] * like[k] for k in thetas)
            w = {k: w[k] * like[k] / norm for k in thetas}

    gap_mean = gap_curves.mean(axis=0)
    gap_lo = np.percentile(gap_curves, 10, axis=0)
    gap_hi = np.percentile(gap_curves, 90, axis=0)
    delta_mean = delta_curves.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    t_axis = np.arange(1, T + 1)
    ax = axes[0]
    ax.plot(t_axis, gap_mean, color='#2f6f4f', lw=2,
             label=r'mean $|\xi(o_t{=}1|h_{<t}) - \mu(o_t{=}1|h_{<t})|$')
    ax.fill_between(t_axis, gap_lo, gap_hi, color='#2f6f4f', alpha=0.2,
                     label='10--90th percentile (60 runs)')
    ax.set_xlabel('time step $t$')
    ax.set_ylabel('prediction gap')
    ax.set_title('Merging of opinions (Thm 7.2.6)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(linestyle=':', alpha=0.5)

    ax = axes[1]
    ax.plot(t_axis, delta_mean, color='#b5442d', lw=2,
             label=r'mean self-optimizing gap $\delta_t$')
    ax.set_xlabel('time step $t$')
    ax.set_ylabel(r'$V_\mu^*(h_{<t}) - V_\mu^{\pi_\xi^*}(h_{<t})$ (proxy)')
    ax.set_title('Self-optimizing value gap (Thm 7.3.6)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(linestyle=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'convergence.pdf'))
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Universal prior 2^{-K(nu)} -- illustrative Occam's-razor bar
# chart over a handful of toy "environments" of increasing description
# length, showing how the universal weight decays exponentially in K(nu).
# ----------------------------------------------------------------------
def fig_universal_prior():
    K = np.array([1, 2, 3, 4, 5, 6, 8, 10, 14, 20])
    w = 2.0 ** (-K.astype(float))

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar([str(k) for k in K], w, color='#5b7fa6', edgecolor='black', linewidth=0.6)
    ax.set_yscale('log')
    ax.set_xlabel(r'Kolmogorov complexity $K(\nu)$ (bits)')
    ax.set_ylabel(r'universal weight $w^U_\nu = 2^{-K(\nu)}$ (log scale)')
    ax.set_title("Occam's razor: simpler environments\nget exponentially more weight", fontsize=12)
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'universal_prior.pdf'))
    plt.close(fig)


# ----------------------------------------------------------------------
# Crop Figure 7.1 (taxonomy of environments) directly from the book PDF.
# ----------------------------------------------------------------------
def crop_taxonomy_figure():
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as e:
        print(f"Skipping PDF crop (missing dependency: {e})")
        return

    if not os.path.exists(BOOK_PDF):
        print(f"Book PDF not found at {BOOK_PDF}; skipping crop.")
        return

    doc = fitz.open(BOOK_PDF)
    page_index = 264  # book-printed page 244, containing Figure 7.1
    page = doc[page_index]
    mat = fitz.Matrix(4, 4)  # 4x zoom for crisp raster crop
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes('png')))

    # Page rect is 504 x 720 pt -> at 4x zoom, 2016 x 2880 px.
    # The diagram (boxes + arrows + labels) occupies roughly
    # y in [48, 513] pt, x in [35, 475] pt (caption starts at y=514.4 pt).
    left, top, right, bottom = 30 * 4, 90 * 4, 480 * 4, 510 * 4
    cropped = img.crop((left, top, right, bottom))
    cropped.save(os.path.join(FIGDIR, 'taxonomy.png'))
    print(f"Saved cropped taxonomy figure: {cropped.size}")


if __name__ == '__main__':
    fig_mixture_weights()
    fig_convergence()
    fig_universal_prior()
    crop_taxonomy_figure()
    print("All figures generated in", FIGDIR)
