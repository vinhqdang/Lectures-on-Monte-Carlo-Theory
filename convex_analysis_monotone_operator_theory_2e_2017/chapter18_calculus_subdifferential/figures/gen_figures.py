#!/usr/bin/env python3
"""
Generate figures for Chapter 18: Calculus of Subdifferentials
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')

def create_figure_18_1():
    """
    Figure 18.1: The functions a_x and q_x visualizing affine minorant and
    quadratic majorant of a convex function f at x
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create domain
    x_vals = np.linspace(-2, 4, 200)

    # Convex function: f(x) = (x - 1)^2 + 0.5
    f_vals = (x_vals - 1)**2 + 0.5

    # Point of interest
    x0 = 1.0
    f_x0 = (x0 - 1)**2 + 0.5

    # Gradient at x0
    grad_f_x0 = 2*(x0 - 1)  # = 0 at x0=1

    # Create affine minorant a_x: a_x(y) = f(x) + <y - x | grad f(x)>
    # For visualization, let's use a perturbed x0 to show non-zero gradient
    x0_perturb = 0.3
    grad_f_x0_perturb = 2*(x0_perturb - 1)  # = -1.4
    f_x0_perturb = (x0_perturb - 1)**2 + 0.5
    a_x_vals = f_x0_perturb + grad_f_x0_perturb * (x_vals - x0_perturb)

    # Quadratic majorant q_x: q_x(y) = f(x) + <y-x | grad f(x)> + (beta/2)||y-x||^2
    # Using beta = 4 (Lipschitz constant of gradient)
    beta = 4.0
    q_x_vals = f_x0_perturb + grad_f_x0_perturb * (x_vals - x0_perturb) + \
               (beta/2) * (x_vals - x0_perturb)**2

    # Plot
    ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label=r'$f(y)$')
    ax.plot(x_vals, a_x_vals, 'g--', linewidth=2, label=r'$a_x(y)$ (affine minorant)')
    ax.plot(x_vals, q_x_vals, 'r-.', linewidth=2, label=r'$q_x(y)$ (quadratic majorant)')

    # Mark the point x0_perturb
    ax.plot(x0_perturb, f_x0_perturb, 'ko', markersize=8)
    ax.axhline(y=f_x0_perturb, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=x0_perturb, color='gray', linestyle=':', alpha=0.5)

    # Annotations
    ax.text(x0_perturb + 0.15, f_x0_perturb + 0.3, r'$(x, f(x))$', fontsize=11)

    ax.set_xlabel(r'$y$', fontsize=12)
    ax.set_ylabel(r'$\mathbb{R}$', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 4)
    ax.set_ylim(-1, 8)

    plt.tight_layout()
    plt.savefig('figures/fig_18_1_affine_quadratic.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_18_1_affine_quadratic.pdf")


def create_figure_ekeland_lebourn():
    """
    Illustration of the Ekeland-Lebourn theorem concept
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Domain and function
    x_vals = np.linspace(0, 5, 200)
    f_vals = 0.1*(x_vals - 2.5)**2 + 0.3

    # Highlight the set where f is Fréchet differentiable (dense subset)
    ax.fill_between(x_vals, -0.5, 3, alpha=0.15, color='green',
                     label='Dense subset where $f$ is Fréchet differentiable')

    ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label='Convex function $f$')

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title('Ekeland-Lebourn Theorem: Dense Differentiability', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.savefig('figures/fig_ekeland_lebourn.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_ekeland_lebourn.pdf")


def create_figure_subdifferential_max():
    """
    Illustration of subdifferential of maximum of convex functions
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x_vals = np.linspace(-2, 3, 200)

    # Two convex functions
    f1_vals = (x_vals + 1)**2
    f2_vals = 0.5*(x_vals - 1)**2 + 0.8
    f_max_vals = np.maximum(f1_vals, f2_vals)

    ax.plot(x_vals, f1_vals, 'b--', linewidth=2, label=r'$f_1(x)$', alpha=0.7)
    ax.plot(x_vals, f2_vals, 'r--', linewidth=2, label=r'$f_2(x)$', alpha=0.7)
    ax.plot(x_vals, f_max_vals, 'g-', linewidth=2.5, label=r'$f(x) = \max\{f_1(x), f_2(x)\}$')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title(r'Subdifferential of Maximum: $\partial f(x) = \text{conv}\bigcup_{i \in I(x)} \partial f_i(x)$',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_subdifferential_max.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_subdifferential_max.pdf")


def create_figure_distance_function():
    """
    Illustration of distance function to a convex set
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create a convex set C (disk)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 2.5 + 1.5 * np.cos(theta)
    circle_y = 2 + 1.5 * np.sin(theta)

    ax.fill(circle_x, circle_y, color='lightblue', alpha=0.5, label='Convex set $C$')
    ax.plot(circle_x, circle_y, 'b-', linewidth=2)

    # Point outside the set
    x_outside = 5.5
    y_outside = 2.0
    ax.plot(x_outside, y_outside, 'ro', markersize=10, label='Point $x$ (outside $C$)')

    # Projection onto C
    proj_x = 4.0
    proj_y = 2.0
    ax.plot(proj_x, proj_y, 'go', markersize=10, label=r'Projection $P_C(x)$')

    # Distance line
    ax.plot([x_outside, proj_x], [y_outside, proj_y], 'r-', linewidth=2,
            label=r'$\nabla d_C(x)$ direction')
    ax.arrow(proj_x, proj_y, 0.6, 0, head_width=0.15, head_length=0.15, fc='red', ec='red')

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('$H$', fontsize=12)
    ax.set_title(r'Distance Function: $d_C(x) = \inf_{z \in C} \|x - z\|$', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_distance_function.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_distance_function.pdf")


def create_figure_moreau_envelope():
    """
    Illustration of Moreau envelope
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x_vals = np.linspace(-3, 3, 200)

    # Original non-smooth function
    f_vals = np.abs(x_vals)

    # Moreau envelope with parameter 1/beta
    beta = 2.0  # Lipschitz constant
    gamma = 1.0 / beta
    moreau_vals = np.zeros_like(x_vals)

    # Moreau envelope: m_gamma(f)(x) = inf_y [f(y) + (1/(2*gamma))||x-y||^2]
    for i, x in enumerate(x_vals):
        # Minimize over y
        y_vals = np.linspace(-3, 3, 500)
        obj_vals = np.abs(y_vals) + (1/(2*gamma)) * (x - y_vals)**2
        moreau_vals[i] = np.min(obj_vals)

    ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label=r'$f(x) = |x|$ (non-smooth)')
    ax.plot(x_vals, moreau_vals, 'r-', linewidth=2.5, label=r'Moreau envelope $m_\gamma(f)$')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(r'Moreau Envelope: $m_\gamma(f)(x) = \inf_y \left[ f(y) + \frac{1}{2\gamma}\|x-y\|^2 \right]$',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 2.5)

    plt.tight_layout()
    plt.savefig('figures/fig_moreau_envelope.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_moreau_envelope.pdf")


def create_figure_lipschitz_gradient():
    """
    Illustration of Lipschitz continuous gradients
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_vals = np.linspace(-2, 3, 200)

    # Function with Lipschitz continuous gradient
    f_vals = 0.5 * x_vals**2 + 0.2 * x_vals
    grad_f_vals = x_vals + 0.2

    ax1.plot(x_vals, f_vals, 'b-', linewidth=2.5)
    ax1.set_xlabel('$x$', fontsize=11)
    ax1.set_ylabel('$f(x)$', fontsize=11)
    ax1.set_title(r'Function $f$ with $\beta$-Lipschitz continuous gradient', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Gradient function
    ax2.plot(x_vals, grad_f_vals, 'r-', linewidth=2.5, label=r'$\nabla f(x)$')
    ax2.fill_between(x_vals, grad_f_vals - 0.3, grad_f_vals + 0.3, alpha=0.2,
                     label=r'Lipschitz band: $|\nabla f(x) - \nabla f(y)| \leq \beta\|x-y\|$')
    ax2.set_xlabel('$x$', fontsize=11)
    ax2.set_ylabel(r'$\nabla f(x)$', fontsize=11)
    ax2.set_title(r'Gradient is Lipschitz continuous', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_lipschitz_gradient.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_lipschitz_gradient.pdf")


def create_figure_strict_convexity():
    """
    Illustration of strict convexity and Gâteaux differentiability
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x_vals = np.linspace(-2, 3, 200)

    # Strictly convex function
    f_vals = x_vals**2 + 0.5

    # Gâteaux differential at a point
    x0 = 0.5
    f_x0 = x0**2 + 0.5
    grad_f_x0 = 2*x0

    # Affine function (tangent)
    tangent_vals = f_x0 + grad_f_x0 * (x_vals - x0)

    ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label=r'Strictly convex $f$')
    ax.plot(x_vals, tangent_vals, 'g--', linewidth=2, label=r'Gâteaux differential at $x$')
    ax.plot(x0, f_x0, 'ko', markersize=8)

    # Shade region where f > tangent
    ax.fill_between(x_vals, tangent_vals, f_vals, where=(f_vals >= tangent_vals),
                    alpha=0.2, color='blue', label='Strict support: $f(y) > \langle y-x | u \\rangle + f(x)$')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title(r'Strict Convexity: Unique support hyperplane implies singleton subdifferential',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_strict_convexity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/fig_strict_convexity.pdf")


if __name__ == '__main__':
    print("Generating figures for Chapter 18...")
    create_figure_18_1()
    create_figure_ekeland_lebourn()
    create_figure_subdifferential_max()
    create_figure_distance_function()
    create_figure_moreau_envelope()
    create_figure_lipschitz_gradient()
    create_figure_strict_convexity()
    print("All figures generated successfully!")
