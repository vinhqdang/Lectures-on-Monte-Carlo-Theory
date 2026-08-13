"""
Generate all figures for Chapter 2.4 (Bayesian Probability Theory) slides.

Run with:
    conda run -n py313 python3 gen_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betainc
from scipy.stats import beta as beta_dist
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ---------------------------------------------------------------------
# Figure 2.10 (book): Beta(alpha,beta) density for several (alpha,beta)
# ---------------------------------------------------------------------
def fig_beta_family():
    theta = np.linspace(0.001, 0.999, 500)
    params = [
        (1, 1, '-', 'black', r'$\alpha=1,\beta=1$'),
        (0.5, 0.5, '--', 'gray', r'$\alpha=0.5,\beta=0.5$'),
        (2, 2, ':', 'darkblue', r'$\alpha=2,\beta=2$'),
        (5, 2, '-.', 'darkred', r'$\alpha=5,\beta=2$'),
        (1, 5, '--', 'darkgreen', r'$\alpha=1,\beta=5$'),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for a, b, ls, c, lbl in params:
        y = beta_dist.pdf(theta, a, b)
        ax.plot(theta, y, ls, color=c, linewidth=2, label=lbl)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'Beta$(\theta;\alpha,\beta)$')
    ax.set_ylim(0, 2.6)
    ax.set_xlim(0, 1)
    ax.legend(loc='upper center', fontsize=10)
    ax.set_title('Beta density for various shape parameters')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'beta_family.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2.11 (book): posterior evolution for x_{1:3}=001 under uniform
# prior, updated one bit at a time, plus long-run x_{1:100}=(01)^50
# ---------------------------------------------------------------------
def fig_posterior_evolution():
    theta = np.linspace(0.001, 0.999, 500)

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax2 = ax1.twinx()

    # Sequence 0,0,1 observed one bit at a time. Posterior after b ones
    # and a zeros (uniform prior) is Beta(theta; b+1, a+1).
    seqs = [
        (0, 0, 'Prior' + r'$\equiv 1$', '-', 'black', ax1),
        (0, 1, r'$x_1=0$', '--', 'gray', ax1),
        (0, 2, r'$x_{1:2}=00$', ':', 'darkblue', ax1),
        (1, 2, r'$x_{1:3}=001$', '-.', 'darkred', ax1),
    ]
    for b, a, lbl, ls, c, ax in seqs:
        y = beta_dist.pdf(theta, b + 1, a + 1)
        ax.plot(theta, y, ls, color=c, linewidth=2, label=lbl)

    # Long run: 50 zeros and 50 ones alternating -> Beta(51,51), plotted
    # on secondary axis since its peak is much taller.
    y_long = beta_dist.pdf(theta, 51, 51)
    ax2.plot(theta, y_long, '-', color='seagreen', linewidth=2.5,
              label=r'$x_{1:100}=(01)^{50}$')

    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('posterior density (small $n$)')
    ax2.set_ylabel('posterior density (large $n$)')
    ax1.set_ylim(0, 3.2)
    ax2.set_ylim(0, 8.5)
    ax1.set_xlim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax1.set_title('Posterior after observing bits one at a time (uniform prior)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'posterior_evolution.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------
# Extra figure: Jeffreys prior Beta(theta;1/2,1/2) vs uniform prior
# ---------------------------------------------------------------------
def fig_jeffreys_vs_uniform():
    theta = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(theta, beta_dist.pdf(theta, 1, 1), '-', color='black',
            linewidth=2, label=r'Uniform prior: Beta$(\theta;1,1)$')
    ax.plot(theta, beta_dist.pdf(theta, 0.5, 0.5), '--', color='crimson',
            linewidth=2, label=r'Jeffreys prior: Beta$(\theta;\frac{1}{2},\frac{1}{2})$')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('density')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)
    ax.legend(fontsize=10)
    ax.set_title('Jeffreys prior places more mass near 0 and 1')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'jeffreys_prior.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------
# Extra figure: MLE vs Laplace-rule estimate vs true theta, showing
# convergence and the "no premature certainty" effect for small n.
# ---------------------------------------------------------------------
def fig_mle_vs_laplace():
    rng = np.random.default_rng(0)
    theta_true = 0.7
    n_max = 200
    xs = rng.binomial(1, theta_true, size=n_max)
    ns = np.arange(1, n_max + 1)
    cums = np.cumsum(xs)
    mle = cums / ns
    laplace = (cums + 1) / (ns + 2)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ns, mle, color='darkred', linewidth=1.6, label='MLE $\\bar x_n = k/n$')
    ax.plot(ns, laplace, color='darkblue', linewidth=1.6,
            label="Laplace rule $(k{+}1)/(n{+}2)$")
    ax.axhline(theta_true, color='black', linestyle=':', linewidth=1.5,
               label=r'true $\theta=0.7$')
    ax.set_xlabel('number of coin flips $n$')
    ax.set_ylabel('estimate of $\\theta$')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.set_title('MLE vs. Laplace-rule estimate as data accumulates')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'mle_vs_laplace.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    fig_beta_family()
    fig_posterior_evolution()
    fig_jeffreys_vs_uniform()
    fig_mle_vs_laplace()
    print('All figures written to', FIGDIR)
