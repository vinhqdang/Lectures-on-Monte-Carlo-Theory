"""
Generate figures for Chapter 7: Sequential anytime-valid inference using e-processes.

Produces three PDF figures:
  (a) ville_null.pdf        - many e-process (wealth) paths under the null, illustrating
                               Ville's inequality: P(sup_t M_t >= 1/alpha) <= alpha.
  (b) eprocess_alt.pdf      - an e-process path under the alternative, growing exponentially,
                               with an early stopping time marked.
  (c) optional_stopping.pdf - a p-value process that becomes invalid under optional stopping,
                               contrasted with an e-process that remains valid.

Plain python3 + matplotlib only (Agg backend, no display needed).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(11)

# ----------------------------------------------------------------------
# Figure (a): Ville's inequality under the null
# ----------------------------------------------------------------------
# We simulate an e-process built by betting on iid Bernoulli(0.5) data
# (the true null), using the likelihood-ratio bet towards q=0.6:
#   S_t(x) = (2q)^x (2(1-q))^(1-x),  bet factor 1.2 for heads, 0.8 for tails.
# Under the null this is a nonnegative martingale started at 1 (an e-process).

alpha = 0.10
threshold = 1.0 / alpha  # = 10
n_steps = 400
n_paths = 300
q = 0.6

paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = 1.0
breach_count = 0
breached = np.zeros(n_paths, dtype=bool)
for i in range(n_paths):
    x = rng.random(n_steps) < 0.5  # true null: fair coin
    factors = np.where(x, 2 * q, 2 * (1 - q))
    wealth = np.cumprod(factors)
    paths[i, 1:] = wealth
    if np.any(wealth >= threshold):
        breached[i] = True
        breach_count += 1

frac_breach = breach_count / n_paths

fig, ax = plt.subplots(figsize=(9, 5.2))
t_axis = np.arange(n_steps + 1)
for i in range(n_paths):
    color = 'crimson' if breached[i] else 'steelblue'
    lw = 1.2 if breached[i] else 0.5
    alpha_line = 0.9 if breached[i] else 0.15
    ax.plot(t_axis, paths[i], color=color, linewidth=lw, alpha=alpha_line)

ax.axhline(threshold, color='black', linestyle='--', linewidth=1.6,
           label=r'threshold $1/\alpha = %d$' % threshold)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
ax.set_yscale('log')
ax.set_xlabel('time $t$ (number of flips)')
ax.set_ylabel(r'wealth / e-process value $M_t$ (log scale)')
ax.set_title(
    r"Ville's inequality: $\mathbb{P}(\exists t: M_t \geq 1/\alpha) \leq \alpha$" +
    "\n(under $H_0$: %d/%d = %.1f%% of %d null paths ever breach $1/\\alpha$, "
    r"$\alpha=%.2f$)" % (breach_count, n_paths, 100 * frac_breach, n_paths, alpha)
)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('ville_null.pdf')
plt.close(fig)

# ----------------------------------------------------------------------
# Figure (b): e-process path under the alternative (exponential growth)
# ----------------------------------------------------------------------
n_steps_alt = 120
p_true = 0.6  # true bias, matches the bet -> exponential growth
x_alt = rng.random(n_steps_alt) < p_true
factors_alt = np.where(x_alt, 2 * q, 2 * (1 - q))
wealth_alt = np.concatenate(([1.0], np.cumprod(factors_alt)))

alpha2 = 0.05
thresh2 = 1.0 / alpha2
stop_time = None
for t, w in enumerate(wealth_alt):
    if w >= thresh2:
        stop_time = t
        break

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(range(len(wealth_alt)), wealth_alt, color='darkgreen', linewidth=1.8,
        label=r'e-process $M_t$ under $\mathbb{Q}$ ($p=0.6$)')
ax.axhline(thresh2, color='black', linestyle='--', linewidth=1.6,
           label=r'rejection threshold $1/\alpha = %d$' % int(thresh2))
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
if stop_time is not None:
    ax.axvline(stop_time, color='crimson', linestyle='-.', linewidth=1.3)
    ax.scatter([stop_time], [wealth_alt[stop_time]], color='crimson', zorder=5,
               label=r'stop \& reject at $t=%d$' % stop_time)
ax.set_yscale('log')
ax.set_xlabel('time $t$ (number of flips)')
ax.set_ylabel(r'wealth / e-process value $M_t$ (log scale)')
ax.set_title('Anytime-valid sequential test: e-process growth under the alternative')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('eprocess_alt.pdf')
plt.close(fig)

# ----------------------------------------------------------------------
# Figure (c): optional stopping breaks p-values, but not e-processes
# ----------------------------------------------------------------------
# Simulate ONE null sequence (p=0.5 truly). Track:
#  - a p-value process P_n (two-sided normal approx p-value for the running mean)
#  - the corresponding e-process (likelihood ratio towards q=0.6, as above)
# Show that min_n P_n can dip below 0.05 purely by chance (falsely "significant"
# under optional stopping) while the e-process stays under 1/alpha with the
# guaranteed probability.

from scipy import stats

n_steps_c = 500
rng2 = np.random.default_rng(7)
x_c = rng2.random(n_steps_c) < 0.5  # true null
n_idx = np.arange(1, n_steps_c + 1)
cum_heads = np.cumsum(x_c)
phat = cum_heads / n_idx
z = (phat - 0.5) / np.sqrt(0.25 / n_idx)
pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
pvals[0] = 1.0  # n=1 undefined-ish, clip

factors_c = np.where(x_c, 2 * q, 2 * (1 - q))
wealth_c = np.cumprod(factors_c)

alpha3 = 0.05
first_peek_below = np.argmax(pvals <= alpha3) if np.any(pvals <= alpha3) else None

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.plot(n_idx, pvals, color='purple', linewidth=1.2)
ax.axhline(alpha3, color='black', linestyle='--', linewidth=1.4, label=r'$\alpha=0.05$')
if first_peek_below is not None and pvals[first_peek_below] <= alpha3:
    ax.scatter([n_idx[first_peek_below]], [pvals[first_peek_below]], color='crimson',
               zorder=5, label='first dip below 0.05\n(false "significance")')
ax.set_ylim(0, 1)
ax.set_xlabel('time $t$ (number of flips)')
ax.set_ylabel(r'running p-value $P_t$')
ax.set_title('P-value process under $H_0$:\n"stop when $P_t \\leq 0.05$" is invalid')
ax.legend(loc='upper right', fontsize=9)

ax = axes[1]
ax.plot(n_idx, wealth_c, color='darkgreen', linewidth=1.4)
ax.axhline(1.0 / alpha3, color='black', linestyle='--', linewidth=1.4,
           label=r'threshold $1/\alpha=%d$' % int(1 / alpha3))
ax.set_yscale('log')
ax.set_xlabel('time $t$ (number of flips)')
ax.set_ylabel(r'e-process value $M_t$ (log scale)')
ax.set_title('Same null data, e-process:\nstopping any time keeps $\\mathbb{P}(M_t\\geq 1/\\alpha)\\leq\\alpha$')
ax.legend(loc='upper left', fontsize=9)

fig.tight_layout()
fig.savefig('optional_stopping.pdf')
plt.close(fig)

print("Figures written: ville_null.pdf, eprocess_alt.pdf, optional_stopping.pdf")
print("Ville breach fraction under null: %.4f (alpha=%.2f)" % (frac_breach, alpha))
