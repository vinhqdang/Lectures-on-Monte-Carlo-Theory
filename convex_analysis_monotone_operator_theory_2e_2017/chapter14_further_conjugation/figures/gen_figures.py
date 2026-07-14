"""
gen_figures.py -- Generate all figures for Chapter 14: Further Conjugation Results
Bauschke & Combettes, "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd ed. (2017).

Run with:  python3 gen_figures.py

Figures produced (all saved as vector PDF in this directory):
  fig_moreau_decomposition.pdf  -- Example 14.5 / Remark 14.4: f = rho*|.|,
                                    its conjugate (indicator of a ball),
                                    Prox_f (soft thresholder), and the Moreau
                                    envelope ^1 f (generalized Huber function).
  fig_proximal_average.pdf      -- Section 14.2: proximal average of
                                    f(x) = x^2/2 and g(x) = |x|, sandwiched
                                    between (f+g)/2 and (f*[+]g*)*, together
                                    with the "dual picture" pav(f*,g*), which
                                    Corollary 14.8(ii) says equals (pav(f,g))*.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import minimize_scalar

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIGURES_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Moreau's decomposition for f = rho*|.|   (Remark 14.4 / Example 14.5)
# ─────────────────────────────────────────────────────────────────────────────
def fig_moreau_decomposition():
    rho = 1.5
    x = np.linspace(-4, 4, 800)

    f = rho * np.abs(x)

    # Prox_f: soft thresholder at level rho, Eq. (14.8)
    prox = np.where(np.abs(x) > rho, (1 - rho / np.abs(x)) * x, 0.0)

    # Moreau envelope ^1 f: generalized Huber function, Eq. (14.9)
    huber = np.where(np.abs(x) > rho, rho * np.abs(x) - rho ** 2 / 2, x ** 2 / 2)

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.2))

    # (a) f itself
    ax = axs[0, 0]
    ax.plot(x, f, color='#2c6aad', lw=2.2)
    ax.set_title(r'(a) $f = \rho\,|\cdot|,\ \ \rho = 1.5$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.grid(alpha=0.3)

    # (b) f* = iota_{B(0;rho)}: 0 on [-rho, rho], +infty outside
    ax = axs[0, 1]
    u = np.linspace(-rho, rho, 200)
    ax.plot(u, np.zeros_like(u), color='#c0392b', lw=2.5,
            label=r'$f^*(u)=0$ on $[-\rho,\rho]$')
    for sign in (-1, 1):
        ax.annotate('', xy=(sign * rho, 3.6), xytext=(sign * rho, 0.15),
                    arrowprops=dict(arrowstyle='-|>', color='#c0392b',
                                     lw=1.6, ls='dashed'))
        ax.text(sign * rho, 3.8, r'$+\infty$', color='#c0392b',
                ha='center', fontsize=10)
    ax.axvline(-rho, color='gray', lw=0.8, ls=':')
    ax.axvline(rho, color='gray', lw=0.8, ls=':')
    ax.set_xlim(-4, 4); ax.set_ylim(-0.5, 4.3)
    ax.set_title(r'(b) $f^* = \iota_{B(0;\rho)}$')
    ax.set_xlabel('$u$'); ax.set_ylabel('$f^*(u)$')
    ax.legend(loc='upper center', fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Prox_f = soft thresholder, Eq. (14.8)
    ax = axs[1, 0]
    ax.plot(x, prox, color='#3a7d3a', lw=2.2, label=r'$\mathrm{Prox}_f\,x$')
    ax.plot(x, x, color='gray', lw=1, ls='--', label='identity')
    ax.set_title(r'(c) $\mathrm{Prox}_f$ (soft thresholder)')
    ax.set_xlabel('$x$'); ax.set_ylabel(r'$\mathrm{Prox}_f\,x$')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Moreau envelope ^1 f = generalized Huber function, Eq. (14.9)
    ax = axs[1, 1]
    ax.plot(x, f, color='#2c6aad', lw=1.4, ls='--', label=r'$f(x)=\rho|x|$')
    ax.plot(x, huber, color='#c4900a', lw=2.2, label=r'${}^{1}f(x)$ (Huber)')
    ax.set_title(r'(d) Moreau envelope ${}^{1}f$')
    ax.set_xlabel('$x$'); ax.set_ylabel('value')
    ax.legend(loc='upper center', fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(r"Moreau's decomposition (Remark 14.4) for $f=\rho|\cdot|$"
                 r" -- Example 14.5", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig('fig_moreau_decomposition.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Proximal average of f(x) = x^2/2 and g(x) = |x|   (Section 14.2)
# ─────────────────────────────────────────────────────────────────────────────
def f_primal(y):
    return 0.5 * y ** 2


def g_primal(z):
    return np.abs(z)


def f_conj(v):
    return 0.5 * v ** 2  # q is self-conjugate


def g_conj_value(w, big=1e6):
    # g*(w) = 0 if |w| <= 1, else +infty (indicator of [-1,1])
    return 0.0 if abs(w) <= 1.0 else big


def pav_primal(x):
    """pav(f,g)(x) = (1/2) inf_y [ y^2/2 + |2x-y| + (y-x)^2 ]  (Def. 14.6)."""
    def obj(y):
        z = 2 * x - y
        return f_primal(y) + g_primal(z) + (y - z) ** 2 / 4.0
    res = minimize_scalar(obj, bounds=(x - 20, x + 20), method='bounded',
                           options={'xatol': 1e-10})
    return 0.5 * res.fun


def pav_dual(u):
    """pav(f*,g*)(u) = (1/2) inf_v [ v^2/2 + g*(2u-v) + (v-w)^2/4 ] with
    w = 2u - v; since g* is an indicator, restrict v to 2u-v in [-1,1]."""
    lo, hi = 2 * u - 1, 2 * u + 1

    def obj(v):
        w = 2 * u - v
        return f_conj(v) + 0.0 + (v - w) ** 2 / 4.0  # g*(w)=0 enforced by bounds
    res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                           options={'xatol': 1e-10})
    return 0.5 * res.fun


def legendre_transform(func, u, xs):
    vals = u * xs - np.array([func(xx) for xx in xs])
    return np.max(vals)


def fig_proximal_average():
    xs = np.linspace(-4, 4, 161)
    f_vals = f_primal(xs)
    g_vals = g_primal(xs)
    upper = 0.5 * (f_vals + g_vals)                      # (1/2)f + (1/2)g
    pav_vals = np.array([pav_primal(xx) for xx in xs])    # pav(f,g)(x)

    # Lower bound (1/2 f* + 1/2 g*)^* computed via numerical Legendre transform
    grid = np.linspace(-6, 6, 4001)
    half_conj_sum = np.array([0.5 * f_conj(v) + 0.5 * g_conj_value(v) for v in grid])
    lower = np.array([np.max(x_ * grid - half_conj_sum) for x_ in xs])

    us = np.linspace(-2.5, 2.5, 161)
    f_conj_vals = f_conj(us)
    g_conj_vals = np.array([g_conj_value(u) for u in us])
    pav_dual_vals = np.array([pav_dual(u) for u in us])

    # Numerical conjugate of pav(f,g), to confirm Corollary 14.8(ii):
    # (pav(f,g))^* = pav(f*,g*)
    pav_conj_vals = np.array([legendre_transform(pav_primal, u, xs) for u in us])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axs[0]
    ax.plot(xs, f_vals, color='#2c6aad', lw=1.4, ls='--', label=r'$f(x)=x^2/2$')
    ax.plot(xs, g_vals, color='#c4900a', lw=1.4, ls='--', label=r'$g(x)=|x|$')
    ax.plot(xs, upper, color='gray', lw=1.6, ls=':', label=r'$\frac{1}{2} f+\frac{1}{2} g$ (upper)')
    ax.fill_between(xs, lower, upper, color='#dce9f5', alpha=0.5)
    ax.plot(xs, pav_vals, color='#c0392b', lw=2.4, label=r'$\mathrm{pav}(f,g)(x)$')
    ax.plot(xs, lower, color='#3a7d3a', lw=1.6, ls='-.',
            label=r'$(\frac{1}{2} f^*+\frac{1}{2} g^*)^*$ (lower)')
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel('$x$'); ax.set_ylabel('value')
    ax.set_title('(a) Primal picture: Prop. 14.9 sandwich')
    ax.legend(loc='upper center', fontsize=7.5, ncol=1)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(us, f_conj_vals, color='#2c6aad', lw=1.4, ls='--', label=r'$f^*(u)=u^2/2$')
    plot_g_conj = np.where(np.array(g_conj_vals) > 5, np.nan, g_conj_vals)
    ax.plot(us, plot_g_conj, color='#c4900a', lw=1.4, ls='--',
            label=r'$g^*(u)=\iota_{[-1,1]}(u)$')
    ax.axvline(-1, color='#c4900a', lw=0.8, ls=':')
    ax.axvline(1, color='#c4900a', lw=0.8, ls=':')
    ax.plot(us, pav_dual_vals, color='#3a7d3a', lw=2.4,
            label=r'$\mathrm{pav}(f^*,g^*)(u)$')
    ax.plot(us, pav_conj_vals, color='#c0392b', lw=1.6, ls=(0, (4, 2)),
            label=r'$(\mathrm{pav}(f,g))^*(u)$ [numeric]')
    ax.set_ylim(-0.2, 3)
    ax.set_xlabel('$u$'); ax.set_ylabel('value')
    ax.set_title('(b) Dual picture: self-duality, Cor.\\,14.8(ii)')
    ax.legend(loc='upper center', fontsize=7.5, ncol=1)
    ax.grid(alpha=0.3)

    fig.suptitle(r'Proximal average of $f(x)=x^2/2$ and $g(x)=|x|$', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig('fig_proximal_average.pdf')

    max_gap = np.max(np.abs(pav_dual_vals - pav_conj_vals))
    print(f"  max |pav(f*,g*) - (pav(f,g))*| over sampled grid = {max_gap:.6f}")


def fig_coercivity_example():
    """Figure showing coercive vs. non-coercive functions."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.linspace(-3, 3, 300)

    # Non-coercive: f(x) = x^2 on [-3, 3], unbounded level sets outside
    ax = axs[0]
    f_noncoercive = 0.1 * x**2
    ax.plot(x, f_noncoercive, color='#c0392b', lw=2.2)
    ax.axhline(y=0.2, color='gray', lw=1, ls='--', alpha=0.5)
    ax.fill_between(x, 0, f_noncoercive, alpha=0.2, color='#c0392b')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title('(a) Non-coercive: $f(x) = 0.1 x^2$')
    ax.set_ylim(-0.2, 1.5)
    ax.grid(alpha=0.3)
    ax.text(0, -0.1, 'Level set unbounded', ha='center', fontsize=9)

    # Coercive: f(x) = x^2
    ax = axs[1]
    f_coercive = x**2
    ax.plot(x, f_coercive, color='#2c6aad', lw=2.2)
    ax.axhline(y=2, color='gray', lw=1, ls='--', alpha=0.5)
    ax.fill_between(x[x**2 <= 2], 0, f_coercive[x**2 <= 2], alpha=0.2, color='#2c6aad')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title('(b) Coercive: $f(x) = x^2$')
    ax.set_ylim(-0.2, 9)
    ax.grid(alpha=0.3)
    ax.text(0, -0.5, 'Level set is bounded', ha='center', fontsize=9)

    fig.suptitle('Coercivity (Prop. 14.16): boundedness of level sets', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig('fig_coercivity.pdf')


def fig_positively_homogeneous():
    """Figure showing positively homogeneous functions."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.linspace(-3, 3, 300)

    # Homogeneous: f(x) = |x| (norm)
    ax = axs[0]
    f_hom = np.abs(x)
    ax.plot(x, f_hom, color='#2c6aad', lw=2.2, label=r'$f(x)=|x|$')
    for lam, style in [(0.5, ':'), (1.0, '-'), (2.0, '--')]:
        y = lam * np.abs(x)
        ax.plot(x, y, color='#2c6aad', lw=1.2, ls=style, alpha=0.6,
                label=f'$\\lambda={lam}$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title(r'(a) Positively homogeneous')
    ax.set_ylim(-0.5, 6)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # Not homogeneous: f(x) = x^2 + 1
    ax = axs[1]
    f_nonhom = x**2 + 1
    ax.plot(x, f_nonhom, color='#c0392b', lw=2.2, label=r'$f(x)=x^2+1$')
    for lam, style in [(0.5, ':'), (1.0, '-'), (2.0, '--')]:
        y = (lam * x)**2 + 1
        ax.plot(x, y, color='#c0392b', lw=1.2, ls=style, alpha=0.6,
                label=f'$\\lambda={lam}$')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title(r'(b) NOT homogeneous')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    fig.suptitle('Positively homogeneous functions (Prop. 14.11, 14.12)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig('fig_positively_homogeneous.pdf')


def fig_strongly_convex():
    """Figure showing strongly convex quadratic and its dynamics."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.linspace(-2.5, 2.5, 300)

    # Strongly convex function f(x) = (1/2) x^2
    ax = axs[0]
    gamma = 1.0
    f = (gamma / 2) * x**2
    ax.plot(x, f, color='#2c6aad', lw=2.2, label=r'$f(x) = \frac{\gamma}{2}x^2$')
    ax.fill_between(x, 0, f, alpha=0.15, color='#2c6aad')
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_title(r'(a) Strongly convex: $\gamma = 1.0$')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.2, 4)

    # Proximal operator Prox_f
    ax = axs[1]
    prox_f = 1.0 / (1.0 + gamma) * x  # Prox_{f}(x) = x / (1 + gamma)
    ax.plot(x, x, color='gray', lw=1.2, ls='--', label='Identity')
    ax.plot(x, prox_f, color='#3a7d3a', lw=2.2, label=r'$\mathrm{Prox}_f(x)$')
    ax.set_xlabel('$x$'); ax.set_ylabel(r'$\mathrm{Prox}_f(x)$')
    ax.set_title(r'(b) Proximal operator')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Strongly convex functions and their proximal operators', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig('fig_strongly_convex.pdf')


if __name__ == '__main__':
    print("Generating Chapter 14 figures...")
    fig_moreau_decomposition()
    fig_proximal_average()
    fig_coercivity_example()
    fig_positively_homogeneous()
    fig_strongly_convex()
    print("Done.")
