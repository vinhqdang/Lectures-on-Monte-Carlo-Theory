#!/usr/bin/env python3
"""
gen_figures.py
Generates figures for Chapter 8: Two Applications
(The Krasnosel'skii-Mann Iterative Method, Dong-Cho-He-Pardalos-Rassias, 2022).

Figures produced (saved as vector PDF in this directory):
  fig_coordinate_updates.pdf -- toy 3-coordinate example contrasting the
                                 cyclic coordinate update order (Algorithm 10)
                                 with the asynchronous parallel coordinate
                                 update order (Algorithm 9 / ARock), including
                                 a stale ("delayed") read used by one agent.
  fig_convergence.pdf        -- convergence of the cyclic coordinate update
                                 (a Gauss-Seidel-type sweep) applied to the
                                 running example: a 3x3 SPD linear system
                                 Ax = b, showing the iterate x_n -> x*.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

np.random.seed(0)

# ---------------------------------------------------------------------------
# Colours (shared with the rest of the chapter set)
# ---------------------------------------------------------------------------
COORD_COLORS = {1: '#d6604d', 2: '#4393c3', 3: '#1a9850'}  # coord 1,2,3
GRAY = '#666666'

# ===========================================================================
# Figure (a): Cyclic vs. Asynchronous coordinate updates on a 3-coordinate
# toy example
# ===========================================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.0, 4.5),
                                      gridspec_kw={'height_ratios': [1, 1.35]})

# --- Top panel: cyclic order -------------------------------------------------
# A single agent visits coordinates 1,2,3,1,2,3,... in fixed order (one
# "epoch" = one full sweep through all m=3 coordinates).
cyclic_schedule = [1, 2, 3, 1, 2, 3, 1, 2, 3]
for t, coord in enumerate(cyclic_schedule, start=1):
    ax_top.scatter([t], [coord], s=420, color=COORD_COLORS[coord],
                    edgecolor='black', linewidth=1.0, zorder=3)
    ax_top.text(t, coord, str(coord), ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=4)

# Epoch separators
for epoch_end in [3.5, 6.5]:
    ax_top.axvline(epoch_end, color=GRAY, linestyle=':', linewidth=1.0)
ax_top.text(2, 3.65, 'epoch $n=1$', ha='center', fontsize=9, color=GRAY)
ax_top.text(5, 3.65, 'epoch $n=2$', ha='center', fontsize=9, color=GRAY)
ax_top.text(8, 3.65, 'epoch $n=3$', ha='center', fontsize=9, color=GRAY)

ax_top.set_xlim(0.3, 9.7)
ax_top.set_ylim(0.4, 3.9)
ax_top.set_yticks([1, 2, 3])
ax_top.set_yticklabels(['$x_1$', '$x_2$', '$x_3$'])
ax_top.set_xticks(range(1, 10))
ax_top.set_xlabel('update step $j$ (single agent, one coordinate at a time)',
                   fontsize=9)
ax_top.set_title('Cyclic coordinate update (Algorithm 10): '
                  'fixed order, no staleness', fontsize=11, fontweight='bold')
ax_top.tick_params(labelsize=9)
for spine in ['top', 'right']:
    ax_top.spines[spine].set_visible(False)

# --- Bottom panel: asynchronous parallel order ------------------------------
# Three "agents" (processors) each update coordinates at their own pace,
# in a random (disordered) sequence, without waiting for one another.
rng = np.random.default_rng(3)
agents = {'Agent A': [], 'Agent B': [], 'Agent C': []}
agent_names = list(agents.keys())
# random event times for each agent on a shared global clock
all_events = []
for a_idx, name in enumerate(agent_names):
    n_events = 4
    times = np.sort(rng.uniform(0.6, 9.4, n_events) + a_idx * 0.15)
    coords = rng.choice([1, 2, 3], size=n_events)
    for t, c in zip(times, coords):
        all_events.append((t, a_idx, c))
all_events.sort(key=lambda e: e[0])

y_positions = {0: 3, 1: 2, 2: 1}  # rows for Agent A, B, C
for y in [3, 2, 1]:
    ax_bot.axhline(y, color='#dddddd', linewidth=8, zorder=0)

for t, a_idx, c in all_events:
    y = y_positions[a_idx]
    ax_bot.scatter([t], [y], s=420, color=COORD_COLORS[c],
                    edgecolor='black', linewidth=1.0, zorder=3)
    ax_bot.text(t, y, str(c), ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', zorder=4)

# Illustrate a stale / delayed read: Agent C's 2nd update (say at index 1)
# reads a value of coordinate 2 that is delayed relative to Agent B's most
# recent write of coordinate 2.
c_events = [e for e in all_events if e[1] == 2]
b_events = [e for e in all_events if e[1] == 1]
if len(c_events) >= 2 and len(b_events) >= 1:
    t_c = c_events[1][0]
    # find the most recent Agent-B write strictly before t_c
    prior_b = [e for e in b_events if e[0] < t_c]
    if prior_b:
        t_b_latest = prior_b[-1][0]
        # but the "stale" read actually used is an *older* write (one before)
        older_b = [e for e in b_events if e[0] < t_b_latest]
        t_stale = older_b[-1][0] if older_b else t_b_latest
        arrow = FancyArrowPatch((t_stale, y_positions[1] + 0.28),
                                 (t_c, y_positions[2] + 0.28),
                                 connectionstyle='arc3,rad=-0.35',
                                 arrowstyle='-|>', mutation_scale=14,
                                 color='#7a1f1f', linewidth=1.6,
                                 linestyle='--', zorder=5)
        ax_bot.add_patch(arrow)
        ax_bot.text((t_stale + t_c) / 2, 3.55,
                    'stale read: uses an\nolder write (delay $\\tau \\geq 1$)',
                    ha='center', fontsize=8.3, color='#7a1f1f')

ax_bot.set_xlim(0.3, 9.7)
ax_bot.set_ylim(0.4, 3.9)
ax_bot.set_yticks([1, 2, 3])
ax_bot.set_yticklabels(agent_names[::-1])
ax_bot.set_xlabel('global (wall-clock) time -- agents run concurrently, '
                   'no waiting', fontsize=9)
ax_bot.set_title('Asynchronous parallel coordinate update (Algorithm 9 / '
                  'ARock): random order, possibly stale reads',
                  fontsize=11, fontweight='bold')
ax_bot.tick_params(labelsize=9)
for spine in ['top', 'right']:
    ax_bot.spines[spine].set_visible(False)

# shared legend
handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=13,
                      markerfacecolor=COORD_COLORS[c], markeredgecolor='black',
                      label=f'coordinate $x_{{{c}}}$ updated') for c in [1, 2, 3]]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig('fig_coordinate_updates.pdf', bbox_inches='tight')
plt.close(fig)


# ===========================================================================
# Figure (b): convergence of the cyclic coordinate update on the 3x3 SPD
# linear-system running example  A x = b
# ===========================================================================
A = np.array([[4.0, 1.0, 1.0],
              [1.0, 4.0, 1.0],
              [1.0, 1.0, 4.0]])
b = np.array([6.0, 9.0, 12.0])
x_star = np.linalg.solve(A, b)  # = (0.5, 1.5, 2.5)

def S(x):
    """S = Id - T = the residual operator A x - b (T = Id - grad of
    f(x) = 1/2 x^T A x - b^T x is the underlying averaged/gradient operator)."""
    return A @ x - b

def cyclic_coordinate_update(x0, n_epochs, lam, order=None):
    """Algorithm 10: one full cyclic sweep = one epoch."""
    m = len(x0)
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for n in range(1, n_epochs + 1):
        seq = order if order is not None else list(range(m))
        y = x.copy()
        for i in seq:
            Sy = S(y)
            Si = np.zeros(m)
            Si[i] = Sy[i]
            y = y - lam * Si
        x = y.copy()
        history.append(x.copy())
    return np.array(history)

lam = 1.0 / 4.0  # = 1/A_ii, since A_ii = 4 for every i (exact coordinate step)
x0 = np.array([0.0, 0.0, 0.0])
hist = cyclic_coordinate_update(x0, n_epochs=8, lam=lam)
errors = np.linalg.norm(hist - x_star, axis=1)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))

epochs = np.arange(len(hist))
labels = ['$x_1$', '$x_2$', '$x_3$']
colors = [COORD_COLORS[1], COORD_COLORS[2], COORD_COLORS[3]]
for k in range(3):
    axL.plot(epochs, hist[:, k], marker='o', markersize=5, color=colors[k],
              label=labels[k] + f' (coord {k+1})', linewidth=1.8)
    axL.axhline(x_star[k], color=colors[k], linestyle='--', linewidth=1.0,
                alpha=0.6)
axL.set_xlabel('epoch $n$ (one full cyclic sweep of all 3 coordinates)')
axL.set_ylabel('iterate value')
axL.set_title('Coordinate values $x_n \\to x^*$')
axL.legend(fontsize=8, loc='center right')
axL.grid(alpha=0.25)

axR.semilogy(epochs, errors, marker='s', color='#542788', linewidth=1.8,
             markersize=5)
axR.set_xlabel('epoch $n$')
axR.set_ylabel('$\\|x_n - x^*\\|$  (log scale)')
axR.set_title('Distance to the fixed point $x^*=(0.5,1.5,2.5)$')
axR.grid(alpha=0.25, which='both')

fig.suptitle('Cyclic coordinate updates solving $Ax=b$ '
              '(running example: 3x3 SPD system, $b=(6,9,12)$)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('fig_convergence.pdf', bbox_inches='tight')
plt.close(fig)

print('Saved fig_coordinate_updates.pdf and fig_convergence.pdf')
print('x_star =', x_star)
print('errors by epoch =', errors)
