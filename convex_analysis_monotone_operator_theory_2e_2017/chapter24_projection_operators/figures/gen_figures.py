#!/usr/bin/env python3
"""
Generate figures for Chapter 24: Proximity Operators
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up plot style for Beamer
rcParams['font.size'] = 10
rcParams['lines.linewidth'] = 2
rcParams['figure.figsize'] = (8, 5)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def figure_soft_thresholding():
    """Figure showing soft thresholding (proximal operator of L1 norm)"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Soft thresholding function
    gamma = 1.0
    x = np.linspace(-4, 4, 1000)

    # Soft thresholding: sign(x) * max(|x| - gamma, 0)
    prox = np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    ax = axes[0]
    ax.plot(x, x, 'k--', label='Identity', linewidth=1.5, alpha=0.5)
    ax.plot(x, prox, 'b-', label=f'Prox_γ||·||¹(x), γ={gamma}', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=gamma, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=-gamma, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{Prox}_{\gamma \|\cdot\|_1}(x)$', fontsize=12)
    ax.set_title('Soft Thresholding (L1 Norm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([-3, 3])

    # Right plot: Hard thresholding-like behavior
    gamma = 1.0
    x = np.linspace(-3, 3, 1000)

    # For illustration: comparison with hard thresholding concept
    hard_thresh = np.where(np.abs(x) > gamma, x, 0)
    soft_thresh = np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    ax = axes[1]
    ax.plot(x, soft_thresh, 'b-', label='Soft Thresholding', linewidth=2)
    ax.plot(x, hard_thresh, 'r--', label='Hard Thresholding', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)
    ax.set_title('Soft vs Hard Thresholding', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([-3, 3])

    plt.tight_layout()
    plt.savefig('fig_soft_thresholding.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_soft_thresholding.pdf")
    plt.close()

def figure_proximity_operator():
    """Figure illustrating proximity operator concept"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show the convex function and its proximal operation
    x = np.linspace(-3, 3, 1000)

    # Example: f(x) = |x|
    f_x = np.abs(x)

    # Point of evaluation
    x_point = 2.5
    gamma = 0.5

    # Proximal operator (for L1 norm with gamma=0.5)
    prox_value = np.sign(x_point) * max(abs(x_point) - gamma, 0)

    ax.plot(x, f_x, 'b-', linewidth=2.5, label=r'$f(x) = |x|$')

    # Mark the point
    ax.plot(x_point, f_x[np.argmin(np.abs(x - x_point))], 'ro', markersize=10,
            label=f'Point: x = {x_point}', zorder=5)

    # Mark the proximal point
    prox_f_val = np.abs(prox_value)
    ax.plot(prox_value, prox_f_val, 'g^', markersize=10,
            label=f'Prox_γf(x) = {prox_value:.2f}', zorder=5)

    # Draw distance (illustrative)
    ax.plot([x_point, prox_value], [f_x[np.argmin(np.abs(x - x_point))], prox_f_val],
            'r--', linewidth=1.5, alpha=0.7, label=f'Distance: γ = {gamma}')

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$f(x)$', fontsize=13)
    ax.set_title(r'Proximity Operator: $\mathrm{Prox}_{\gamma f}(x)$', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper center')
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 3])

    plt.tight_layout()
    plt.savefig('fig_proximity_operator.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_proximity_operator.pdf")
    plt.close()

def figure_projection_operator():
    """Figure showing projection onto a convex set"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw a circle (convex set C)
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1.0
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'b-', linewidth=2.5, label='Convex Set C')
    ax.fill(circle_x, circle_y, color='blue', alpha=0.1)

    # Point outside the set
    point_x, point_y = 2.5, 1.8
    ax.plot(point_x, point_y, 'ro', markersize=12, label='Point x', zorder=5)

    # Projection (on the ray from origin through the point)
    dist = np.sqrt(point_x**2 + point_y**2)
    proj_x = radius * point_x / dist
    proj_y = radius * point_y / dist
    ax.plot(proj_x, proj_y, 'g^', markersize=12, label=r'$P_C(x)$ (projection)', zorder=5)

    # Draw line from point to projection
    ax.plot([point_x, proj_x], [point_y, proj_y], 'r--', linewidth=2, alpha=0.7,
            label='Distance minimized')

    # Draw line from origin
    ax.plot([0, point_x], [0, point_y], 'k:', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title('Projection Operator onto Convex Set C', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim([-2, 3])
    ax.set_ylim([-2, 2.5])
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_projection_operator.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_projection_operator.pdf")
    plt.close()

def figure_huber_function():
    """Figure showing Huber function and its proximal operator"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rho = 1.0
    x = np.linspace(-3, 3, 1000)

    # Huber function
    f_huber = np.where(
        np.abs(x) <= rho,
        x**2 / 2,
        rho * np.abs(x) - rho**2 / 2
    )

    ax = axes[0]
    ax.plot(x, f_huber, 'b-', linewidth=2.5, label=f'Huber function (ρ={rho})')
    ax.axvline(x=rho, color='r', linestyle=':', alpha=0.6, label=f'Transition: x=±ρ')
    ax.axvline(x=-rho, color='r', linestyle=':', alpha=0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$f(x)$', fontsize=12)
    ax.set_title('Huber Function', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Proximal operator of Huber function with gamma=1
    gamma = 1.0
    prox_x = np.where(
        np.abs(x) > (gamma + 1) * rho,
        (1 - gamma * rho / np.abs(x)) * x,
        x / (gamma + 1)
    )

    ax = axes[1]
    ax.plot(x, x, 'k--', linewidth=1.5, alpha=0.5, label='Identity')
    ax.plot(x, prox_x, 'g-', linewidth=2.5, label=f'Prox_γ Huber (γ={gamma})')
    ax.axvline(x=(gamma+1)*rho, color='r', linestyle=':', alpha=0.6)
    ax.axvline(x=-(gamma+1)*rho, color='r', linestyle=':', alpha=0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'Output', fontsize=12)
    ax.set_title('Proximal Operator of Huber', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_huber_function.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_huber_function.pdf")
    plt.close()

def figure_convergence():
    """Figure showing proximal algorithm convergence"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulate proximal gradient descent
    np.random.seed(42)

    # Simple quadratic function: f(x) = (x-3)^2
    x_opt = 3.0
    gamma = 0.1  # step size

    x_init = 0.0
    iterates = [x_init]
    x_curr = x_init

    for _ in range(20):
        # Gradient step
        grad = 2 * (x_curr - x_opt)
        x_new = x_curr - gamma * grad
        iterates.append(x_new)
        x_curr = x_new

    # Plot function
    x = np.linspace(-1, 5, 1000)
    f_x = (x - x_opt)**2

    ax.plot(x, f_x, 'b-', linewidth=2.5, label=r'$f(x) = (x-3)^2$')

    # Plot iterates
    f_iterates = [(xi - x_opt)**2 for xi in iterates]
    ax.plot(iterates, f_iterates, 'ro-', markersize=6, linewidth=1.5,
            label='Proximal gradient iterates', alpha=0.7)

    # Mark optimum
    ax.plot(x_opt, 0, 'g*', markersize=15, label=f'Optimum: x*={x_opt}', zorder=5)

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$f(x)$', fontsize=13)
    ax.set_title('Proximal Gradient Algorithm Convergence', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([-1, 5])
    ax.set_ylim([0, 10])

    plt.tight_layout()
    plt.savefig('fig_convergence.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_convergence.pdf")
    plt.close()

def figure_proximal_threshold():
    """Figure showing proximal thresholding (Berhu function)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    rho = 1.0
    x = np.linspace(-2.5, 2.5, 1000)

    # Berhu-like function (absolute value with parameter)
    f_x = np.abs(x)

    # Compute proximal operator with gamma=0.5
    gamma = 0.5

    # For soft-threshold on absolute value
    prox_x = np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    ax.plot(x, f_x, 'b-', linewidth=2.5, label=r'$f(x) = |x|$')
    ax.plot(x, prox_x, 'g-', linewidth=2.5, label=f'Prox_γ|·|(x), γ={gamma}')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=gamma, color='r', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.axvline(x=-gamma, color='r', linestyle=':', alpha=0.6, linewidth=1.5)

    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel('Function value', fontsize=13)
    ax.set_title('Proximal Thresholding', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper center')
    ax.set_ylim([-0.5, 2.5])

    plt.tight_layout()
    plt.savefig('fig_proximal_threshold.pdf', bbox_inches='tight', dpi=150)
    print("Saved: fig_proximal_threshold.pdf")
    plt.close()

if __name__ == '__main__':
    print("Generating figures for Chapter 24: Proximity Operators...")
    figure_proximity_operator()
    figure_projection_operator()
    figure_soft_thresholding()
    figure_huber_function()
    figure_convergence()
    figure_proximal_threshold()
    print("\nAll figures generated successfully!")
