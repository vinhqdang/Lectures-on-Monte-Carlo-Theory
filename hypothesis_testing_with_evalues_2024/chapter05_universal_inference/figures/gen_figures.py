"""
Figure generation for Chapter 5: The universal inference e-value.
(Ramdas & Wang, "Hypothesis Testing with E-Values", arXiv:2410.23614)

Produces (all as PDF, in this directory):
  fig_split_diagram.pdf   -- schematic of the data-splitting construction
  fig_running_example.pdf -- the n=40 running numerical example (histogram +
                              fitted null / alternative densities + E value)
  fig_power_vs_n.pdf      -- power of the split / subsampled LR e-value test
                              vs sample size, for the Gaussian-mixture
                              model-selection problem of Section 5.7

Running `python3 gen_figures.py` also prints the numbers that are quoted
verbatim in the slides (running-example MLEs and log E, and a small power
table), so the slide content can be regenerated/checked against real output.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RNG_SEED = 2024

# ----------------------------------------------------------------------
# Figure 1: schematic of the split-likelihood-ratio construction
# ----------------------------------------------------------------------

def fig_split_diagram():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Full data box
    ax.add_patch(plt.Rectangle((0.5, 4.7), 9, 0.9, fill=True,
                                facecolor='#dddddd', edgecolor='black', lw=1.3))
    ax.text(5, 5.15, r'Full data $X^n = (X_1,\ldots,X_n)$, iid $\sim p \in \mathcal{P}$ (if $H_0$ true)',
            ha='center', va='center', fontsize=11)

    # Arrow down, splitting
    ax.annotate('', xy=(2.6, 3.9), xytext=(4.5, 4.65),
                arrowprops=dict(arrowstyle='->', lw=1.4))
    ax.annotate('', xy=(7.4, 3.9), xytext=(5.5, 4.65),
                arrowprops=dict(arrowstyle='->', lw=1.4))
    ax.text(5, 4.15, 'random / fixed split', ha='center', fontsize=9, style='italic')

    # D0 box
    ax.add_patch(plt.Rectangle((0.5, 2.9), 4.1, 1.0, fill=True,
                                facecolor='#fde9c8', edgecolor='#b8860b', lw=1.3))
    ax.text(2.55, 3.4, r'$D_0 = \{X_i : i \in I\}$', ha='center', va='center', fontsize=12)

    # D1 box
    ax.add_patch(plt.Rectangle((5.4, 2.9), 4.1, 1.0, fill=True,
                                facecolor='#cfe3f7', edgecolor='#2b5d8c', lw=1.3))
    ax.text(7.45, 3.4, r'$D_1 = \{X_i : i \notin I\}$', ha='center', va='center', fontsize=12)

    # D1 -> estimate alternative
    ax.annotate('', xy=(7.45, 2.05), xytext=(7.45, 2.85),
                arrowprops=dict(arrowstyle='->', lw=1.4, color='#2b5d8c'))
    ax.add_patch(plt.Rectangle((5.7, 1.15), 3.5, 0.9, fill=True,
                                facecolor='white', edgecolor='#2b5d8c', lw=1.3))
    ax.text(7.45, 1.6, r'pick $\hat q_1 \in \mathcal{Q}$ using $D_1$' + '\n(e.g. MLE / Bayes on $D_1$)',
            ha='center', va='center', fontsize=9.3, color='#2b5d8c')

    # D0 -> estimate null MLE
    ax.annotate('', xy=(2.55, 2.05), xytext=(2.55, 2.85),
                arrowprops=dict(arrowstyle='->', lw=1.4, color='#b8860b'))
    ax.add_patch(plt.Rectangle((0.8, 1.15), 3.5, 0.9, fill=True,
                                facecolor='white', edgecolor='#b8860b', lw=1.3))
    ax.text(2.55, 1.6, r'$\hat p_0 = \arg\max_{p\in\mathcal{P}} \prod_{i\in D_0} p(X_i)$' + '\n(null MLE on $D_0$)',
            ha='center', va='center', fontsize=9.3, color='#b8860b')

    # combine
    ax.annotate('', xy=(5, 0.65), xytext=(2.55, 1.1), arrowprops=dict(arrowstyle='->', lw=1.4, color='#b8860b'))
    ax.annotate('', xy=(5, 0.65), xytext=(7.45, 1.1), arrowprops=dict(arrowstyle='->', lw=1.4, color='#2b5d8c'))

    ax.add_patch(plt.Rectangle((2.4, -0.15), 5.2, 0.85, fill=True,
                                facecolor='#e3f2e1', edgecolor='#2e7d32', lw=1.5))
    ax.text(5, 0.28, r'$E = \prod_{i \in D_0} \frac{\hat q_1(X_i)}{\hat p_0(X_i)}$  evaluated on $D_0$',
            ha='center', va='center', fontsize=12, color='#1b5e20')

    plt.tight_layout()
    fig.savefig('fig_split_diagram.pdf')
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: the n=40 running numerical example
# ----------------------------------------------------------------------

def mixture_negloglik(params, x):
    mu1, mu2, lam = params
    lam = min(max(lam, 1e-6), 1 - 1e-6)
    dens = (1 - lam) * norm.pdf(x, mu1, 1) + lam * norm.pdf(x, mu2, 1)
    dens = np.clip(dens, 1e-300, None)
    return -np.sum(np.log(dens))


def fit_mixture(x):
    best = None
    # a handful of restarts to avoid bad local optima
    for mu1_0, mu2_0 in [(-1, 1), (0, 2), (-2, 0), (0.5, -0.5)]:
        res = minimize(mixture_negloglik, x0=[mu1_0, mu2_0, 0.5], args=(x,),
                        method='Nelder-Mead',
                        options=dict(xatol=1e-6, fatol=1e-6, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res
    mu1, mu2, lam = best.x
    lam = min(max(lam, 0), 1)
    return mu1, mu2, lam


def mixture_density(x, mu1, mu2, lam):
    return (1 - lam) * norm.pdf(x, mu1, 1) + lam * norm.pdf(x, mu2, 1)


def fig_running_example():
    rng = np.random.default_rng(RNG_SEED)
    n = 40
    # True data-generating process used for the illustration: a genuine
    # two-component mixture (so H0 is false and we expect a large E).
    true_mu1, true_mu2, true_lam = -1.2, 1.2, 0.5
    comp = rng.random(n) < true_lam
    x = np.where(comp, rng.normal(true_mu2, 1, n), rng.normal(true_mu1, 1, n))

    idx = rng.permutation(n)
    D0, D1 = x[idx[:20]], x[idx[20:]]

    # Step 2: null MLE on D0 (single Gaussian, variance 1 known) -> sample mean
    mu0_hat = D0.mean()

    # Step 1: alternative fit on D1 (2-component mixture)
    mu1_hat, mu2_hat, lam_hat = fit_mixture(D1)

    # Step 3: split LR e-variable, evaluated on D0
    q1_D0 = mixture_density(D0, mu1_hat, mu2_hat, lam_hat)
    p0_D0 = norm.pdf(D0, mu0_hat, 1)
    logE = np.sum(np.log(q1_D0) - np.log(p0_D0))
    E = np.exp(logE)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    grid = np.linspace(x.min() - 1, x.max() + 1, 400)
    ax.hist(D0, bins=10, density=True, alpha=0.35, color='#b8860b', label=r'$D_0$ (20 pts)')
    ax.hist(D1, bins=10, density=True, alpha=0.35, color='#2b5d8c', label=r'$D_1$ (20 pts)')
    ax.plot(grid, norm.pdf(grid, mu0_hat, 1), color='#b8860b', lw=2.4,
            label=r'$\hat p_0 = N(\hat\mu_0, 1)$, fit on $D_0$')
    ax.plot(grid, mixture_density(grid, mu1_hat, mu2_hat, lam_hat), color='#2b5d8c', lw=2.4,
            label=r'$\hat q_1$: 2-comp. mixture, fit on $D_1$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('density')
    ax.set_title(rf'Running example: $\hat\mu_0={mu0_hat:.2f}$,  '
                 rf'$(\hat\mu_1,\hat\mu_2,\hat\lambda)=({mu1_hat:.2f},{mu2_hat:.2f},{lam_hat:.2f})$,  '
                 rf'$\log E={logE:.2f}$', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='upper left')
    plt.tight_layout()
    fig.savefig('fig_running_example.pdf')
    plt.close(fig)

    print("=== Running example (Figure fig_running_example.pdf) ===")
    print(f"D0 = {np.round(D0,2)}")
    print(f"D1 = {np.round(D1,2)}")
    print(f"null MLE on D0: mu0_hat = {mu0_hat:.3f}")
    print(f"alt fit on D1: mu1_hat={mu1_hat:.3f}, mu2_hat={mu2_hat:.3f}, lambda_hat={lam_hat:.3f}")
    print(f"log E = {logE:.3f},  E = {E:.3e}")
    print()
    return dict(mu0_hat=mu0_hat, mu1_hat=mu1_hat, mu2_hat=mu2_hat, lam_hat=lam_hat, logE=logE, E=E)


# ----------------------------------------------------------------------
# Figure 3: power vs sample size for the split / subsampled LR e-value
#           test, on the Gaussian-mixture model-selection problem of 5.7
# ----------------------------------------------------------------------

def null_mle_mean(x):
    return x.mean()


def split_lr_logE(D0, D1, lam_fixed=0.75):
    """One split-LR e-variable: alt fit (mu1,mu2) on D1 with mixing prob
    lam_fixed known (as in the book's Section 5.7 numerical study, eq 5.4),
    null MLE (single Gaussian mean) on D0, ratio evaluated on D0."""
    def negloglik(params, x):
        mu1, mu2 = params
        dens = (1 - lam_fixed) * norm.pdf(x, mu1, 1) + lam_fixed * norm.pdf(x, mu2, 1)
        return -np.sum(np.log(np.clip(dens, 1e-300, None)))
    res = minimize(negloglik, x0=[D1.mean() - 0.5, D1.mean() + 0.5], args=(D1,),
                    method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6))
    mu1_hat, mu2_hat = res.x
    mu0_hat = null_mle_mean(D0)
    q1 = (1 - lam_fixed) * norm.pdf(D0, mu1_hat, 1) + lam_fixed * norm.pdf(D0, mu2_hat, 1)
    p0 = norm.pdf(D0, mu0_hat, 1)
    return np.sum(np.log(np.clip(q1, 1e-300, None)) - np.log(np.clip(p0, 1e-300, None)))


def power_curve(n_list, mu, lam_fixed=0.75, alpha=0.05, reps=200, B=3, seed=RNG_SEED):
    """Estimate power of (a) the plain split-LRT (B=1) and (b) the
    subsampled LRT averaged over B splits, at H1: mu1=-mu, mu2=+mu
    (mu != 0, so H0: mu1=mu2 is false and we are measuring power)."""
    rng = np.random.default_rng(seed)
    thresh = 1 / alpha
    power_split = []
    power_sub = []
    for n in n_list:
        rej_split = 0
        rej_sub = 0
        for _ in range(reps):
            comp = rng.random(n) < lam_fixed
            x = np.where(comp, rng.normal(mu, 1, n), rng.normal(-mu, 1, n))
            logEs = []
            for b in range(B):
                idx = rng.permutation(n)
                half = n // 2
                D0, D1 = x[idx[:half]], x[idx[half:]]
                logEs.append(split_lr_logE(D0, D1, lam_fixed))
            # plain split LRT uses only the first split
            if logEs[0] >= np.log(thresh):
                rej_split += 1
            # subsampled LRT: average of E's (not of logE's), then Markov test
            Ebar = np.mean(np.exp(logEs))
            if Ebar >= thresh:
                rej_sub += 1
        power_split.append(rej_split / reps)
        power_sub.append(rej_sub / reps)
    return np.array(power_split), np.array(power_sub)


def fig_power_vs_n():
    n_list = [20, 40, 80, 160, 320, 640]
    mu = 0.8  # fixed separation between the two mixture components
    power_split, power_sub = power_curve(n_list, mu, reps=150, B=3)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(n_list, power_split, 'o-', color='#e08214', label='split LRT (single split)')
    ax.plot(n_list, power_sub, 's-', color='#2b5d8c', label='subsampled LRT ($B=3$)')
    ax.axhline(0.05, ls='--', color='gray', lw=1, label=r'nominal level $\alpha=0.05$')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_list)
    ax.set_xticklabels(n_list)
    ax.set_xlabel('sample size $n$')
    ax.set_ylabel('estimated power')
    ax.set_title(r'Power of the split/subsampled LR e-value test' + '\n'
                 r'$H_0: \mu_1=\mu_2$ vs. mixture with true gap $\mu_1-\mu_2 = -2\times 0.8$',
                 fontsize=10.5)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig('fig_power_vs_n.pdf')
    plt.close(fig)

    print("=== Power vs n (Figure fig_power_vs_n.pdf) ===")
    for n, ps, pb in zip(n_list, power_split, power_sub):
        print(f"n={n:4d}: power(split)={ps:.3f}   power(subsampled,B=3)={pb:.3f}")
    print()


if __name__ == '__main__':
    fig_split_diagram()
    fig_running_example()
    fig_power_vs_n()
    print("All figures written to the figures/ directory.")
