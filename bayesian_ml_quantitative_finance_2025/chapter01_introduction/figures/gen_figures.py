"""
Generate figures for Chapter 1 (Introduction) slides on:
Bayesian Machine Learning in Quantitative Finance (Mongwe, Mbuvha & Marwala, 2025)

Figures produced (all saved as vector PDF):
  1. fig_bayes_coin.pdf     -- Bayes' theorem illustrated with a coin-flip toy example
                               (prior belief, likelihood of observed data, posterior belief)
  2. fig_book_roadmap.pdf   -- Visual roadmap of the book's four themes and 14 chapters
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import beta as beta_dist

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Figure 1: Bayes' theorem via a coin-flipping toy example
# ---------------------------------------------------------------------------
# Prior belief: coin is "probably close to fair" -> Beta(4,4), centered at 0.5
# Data: 10 flips, 8 heads, 2 tails
# Posterior (conjugate Beta-Binomial update): Beta(4+8, 4+2) = Beta(12,6)

a_prior, b_prior = 4.0, 4.0
n_flips, n_heads = 10, 8
n_tails = n_flips - n_heads
a_post, b_post = a_prior + n_heads, b_prior + n_tails

theta = np.linspace(0.001, 0.999, 500)
prior_pdf = beta_dist.pdf(theta, a_prior, b_prior)
post_pdf = beta_dist.pdf(theta, a_post, b_post)

# Likelihood (as a function of theta, for fixed data): Binomial kernel theta^h (1-theta)^t
likelihood = theta**n_heads * (1 - theta)**n_tails
likelihood_scaled = likelihood / likelihood.max() * max(prior_pdf.max(), post_pdf.max()) * 0.9

post_mean = a_post / (a_post + b_post)
ci_low, ci_high = beta_dist.ppf(0.025, a_post, b_post), beta_dist.ppf(0.975, a_post, b_post)
prior_mean = a_prior / (a_prior + b_prior)

fig, ax = plt.subplots(figsize=(7.6, 5.0))

ax.plot(theta, prior_pdf, color="#4C72B0", lw=2.5, label=r"Prior: Beta(4,4), belief before data")
ax.fill_between(theta, prior_pdf, color="#4C72B0", alpha=0.15)

ax.plot(theta, likelihood_scaled, color="#DD8452", lw=2.0, ls="--",
        label=r"Likelihood: $\propto \theta^{8}(1-\theta)^{2}$ (rescaled)")

ax.plot(theta, post_pdf, color="#55A868", lw=2.5,
        label=r"Posterior: Beta(12,6), belief after data")
ax.fill_between(theta, post_pdf, color="#55A868", alpha=0.20)

ax.axvline(prior_mean, color="#4C72B0", ls=":", lw=1.2)
ax.axvline(post_mean, color="#55A868", ls=":", lw=1.2)
ax.axvspan(ci_low, ci_high, color="#55A868", alpha=0.08)

ax.annotate(f"posterior mean = {post_mean:.2f}",
            xy=(post_mean, beta_dist.pdf(post_mean, a_post, b_post)),
            xytext=(post_mean + 0.08, beta_dist.pdf(post_mean, a_post, b_post) - 0.6),
            fontsize=9.5, color="#2F5C36", ha="left",
            arrowprops=dict(arrowstyle="->", color="#2F5C36", lw=1))

ax.set_xlabel(r"$\theta$ = probability the coin lands heads")
ax.set_ylabel("density")
ax.set_title("Coin-flip example: data (8 heads / 10 flips) pulls belief\n"
              r"from a fair-coin prior towards $\theta \approx 0.8$", pad=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, max(prior_pdf.max(), post_pdf.max()) * 1.35)
ax.legend(loc="upper left", fontsize=9, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("fig_bayes_coin.pdf")
plt.close(fig)

print("Coin-flip example summary statistics:")
print(f"  Prior: Beta({a_prior:.0f},{b_prior:.0f}), prior mean = {prior_mean:.3f}")
print(f"  Data: {n_heads} heads out of {n_flips} flips")
print(f"  Posterior: Beta({a_post:.0f},{b_post:.0f})")
print(f"  Posterior mean = {post_mean:.4f}")
print(f"  95% credible interval = [{ci_low:.4f}, {ci_high:.4f}]")

# ---------------------------------------------------------------------------
# Figure 2: Roadmap of the book's themes and chapters
# ---------------------------------------------------------------------------
# Four themes (as per Fig. 1.2 in the book) and the chapters under each theme
themes = [
    {
        "name": "Background to\nBayesian Inference",
        "color": "#4C72B0",
        "chapters": ["Ch 2: Variational Inference & MCMC\nfor training Bayesian models"],
    },
    {
        "name": "Derivative\nModeling",
        "color": "#DD8452",
        "chapters": [
            "Ch 3: SABR Model &\nHamiltonian Monte Carlo",
            "Ch 4: Equity Volatility Surfaces\nvia Sparse Gaussian Processes",
            "Ch 5: SA Equity Option Prices\nvia Normalizing Flows",
        ],
    },
    {
        "name": "Financial\nManagement",
        "color": "#55A868",
        "chapters": [
            "Ch 6: Corporate Credit Ratings\nvia Sparse/Distributed GPs",
            "Ch 7: Recovery on Charged-Off\nLoan Accounts",
            "Ch 8: Audit Outcome Model\nSelection via Normalizing Flows",
            "Ch 9: Unauthorized Expenditure\nvia Langevin & Hamiltonian MC",
        ],
    },
    {
        "name": "Insurance &\nInvestments",
        "color": "#C44E52",
        "chapters": [
            "Ch 10: Motor Insurance Claims\nvia Bayesian Neural Networks",
            "Ch 11: Nelson-Siegel Model via\nShadow/Adaptive HMC",
            "Ch 12: Yield Curve Model Selection\nvia Nested Sampling",
            "Ch 13: Bayesian Investment Analyst\non the JSE",
        ],
    },
]

max_chapters = max(len(t["chapters"]) for t in themes)
ch_h = 0.72
ch_gap = 0.14
theme_h = 0.9
theme_y = 6.3
theme_top = theme_y + theme_h
stack_top = theme_y - 0.25
stack_bottom = stack_top - max_chapters * (ch_h + ch_gap)
concl_h = 0.7
concl_y = stack_bottom - 0.45 - concl_h

# Core box (Bayes' theorem) and Chapter 1 box, stacked above the theme row
core_h = 0.8
core_gap = 0.35
core_y = theme_top + core_gap
core_top = core_y + core_h

ch1_h = 0.8
ch1_gap = 0.35
ch1_y = core_top + ch1_gap
ch1_top = ch1_y + ch1_h

fig_height = 9.4
fig, ax = plt.subplots(figsize=(13, fig_height))
ax.set_xlim(0, 13)
ax.set_ylim(concl_y - 0.25, ch1_top + 0.3)
ax.axis("off")

# Chapter 1 box (entry point)
ch1_box = FancyBboxPatch((5.0, ch1_y), 3.0, ch1_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="black", facecolor="#EAEAEA")
ax.add_patch(ch1_box)
ax.text(6.5, ch1_y + ch1_h / 2, "Chapter 1: Introduction\n(this chapter)", ha="center",
        va="center", fontsize=10.5, fontweight="bold")

# Bayes' theorem core box
core_box = FancyBboxPatch((5.3, core_y), 2.4, core_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                           linewidth=1.5, edgecolor="black", facecolor="#FFF3CD")
ax.add_patch(core_box)
ax.text(6.5, core_y + core_h / 2, "Bayesian Inference\nFramework (Bayes' Theorem)", ha="center",
        va="center", fontsize=9.5)

ax.add_patch(FancyArrowPatch((6.5, ch1_y), (6.5, core_top), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="black"))

n_themes = len(themes)
x_centers = np.linspace(1.6, 11.4, n_themes)
theme_w = 2.6

for xc, theme in zip(x_centers, themes):
    ax.add_patch(FancyArrowPatch((6.5, core_y), (xc, theme_y + theme_h), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="gray",
                                 connectionstyle="arc3,rad=0.0"))
    box = FancyBboxPatch((xc - theme_w / 2, theme_y), theme_w, theme_h,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="black", facecolor=theme.get("color"),
                          alpha=0.85)
    ax.add_patch(box)
    ax.text(xc, theme_y + theme_h / 2, theme["name"], ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold")

    for i, ch_text in enumerate(theme["chapters"]):
        y_top = stack_top - i * (ch_h + ch_gap)
        y_bottom = y_top - ch_h
        ch_box = FancyBboxPatch((xc - theme_w / 2, y_bottom), theme_w, ch_h,
                                 boxstyle="round,pad=0.04,rounding_size=0.06",
                                 linewidth=1.0, edgecolor=theme.get("color"),
                                 facecolor=theme.get("color"), alpha=0.12)
        ax.add_patch(ch_box)
        ax.text(xc, (y_top + y_bottom) / 2, ch_text, ha="center", va="center", fontsize=7.6)
        if i == 0:
            ax.add_patch(FancyArrowPatch((xc, theme_y), (xc, y_top), arrowstyle="-",
                                         mutation_scale=8, linewidth=0.9, color="gray"))

# Chapter 14 -- conclusions, bringing every theme together
concl_box = FancyBboxPatch((5.0, concl_y), 3.0, concl_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                            linewidth=1.5, edgecolor="black", facecolor="#EAEAEA")
ax.add_patch(concl_box)
ax.text(6.5, concl_y + concl_h / 2, "Chapter 14: Conclusions", ha="center", va="center",
        fontsize=10.5, fontweight="bold")
connector_y = concl_y + concl_h + 0.22
for xc in x_centers:
    ax.add_patch(FancyArrowPatch((xc, stack_bottom), (xc, connector_y), arrowstyle="-",
                                 mutation_scale=8, linewidth=0.9, color="gray"))
    ax.add_patch(FancyArrowPatch((xc, connector_y), (6.5, connector_y), arrowstyle="-",
                                 mutation_scale=8, linewidth=0.9, color="gray"))
ax.add_patch(FancyArrowPatch((6.5, connector_y), (6.5, concl_y + concl_h), arrowstyle="-|>",
                             mutation_scale=12, linewidth=1.0, color="gray"))

ax.set_title("Roadmap of the Book: Four Themes, 14 Chapters", fontsize=14, fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig("fig_book_roadmap.pdf")
plt.close(fig)

print("\nFigures written: fig_bayes_coin.pdf, fig_book_roadmap.pdf")
