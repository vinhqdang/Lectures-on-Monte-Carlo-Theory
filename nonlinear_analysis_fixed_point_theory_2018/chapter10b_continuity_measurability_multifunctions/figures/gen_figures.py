#!/usr/bin/env python3
"""
Generate figures for Chapter 10b: Continuity & Measurability of Multifunctions
Pathak's "Introduction to Nonlinear Analysis and Fixed Point Theory"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from scipy.integrate import odeint, trapezoid
import os

# Create figures directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def save_figure(filename):
    """Helper to save figure as PDF"""
    filepath = f"{filename}.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")

# Figure 1: Multifunction Visualization
def fig_multifunction_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Single-valued function
    ax = axes[0]
    x_vals = np.linspace(0, 2*np.pi, 100)
    y_vals = np.sin(x_vals)

    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='$f(x) = \sin(x)$')
    ax.scatter([np.pi/2, np.pi, 3*np.pi/2], [1, 0, -1], color='red', s=100, zorder=5)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('Single-Valued Function', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 2*np.pi)

    # Right: Multifunction (set-valued)
    ax = axes[1]
    # Draw sets for different x values
    x_points = [np.pi/2, np.pi, 3*np.pi/2]
    y_centers = [1.5, 0.5, -0.5]
    radii = [0.3, 0.4, 0.35]
    colors = ['red', 'green', 'blue']

    for i, (x_pt, y_center, r, color) in enumerate(zip(x_points, y_centers, radii, colors)):
        circle = Circle((x_pt, y_center), r, fill=True, alpha=0.3,
                       edgecolor=color, facecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.plot(x_pt, y_center, 'o', color=color, markersize=8)

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('Multifunction $F: X \\to 2^Y$ (Set-Valued)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-2, 2.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure('01_multifunction_concept')

# Figure 2: Upper Semicontinuity
def fig_semicontinuity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Upper semicontinuous
    ax = axes[0]
    x = np.linspace(0, 3, 200)

    # Create upper semicontinuous multifunction visualization
    for x_val in [0.5, 1.5, 2.5]:
        if x_val < 1:
            width = 0.3 * (1 - x_val/2)
        elif x_val < 2:
            width = 0.2
        else:
            width = 0.3 * (3 - x_val)

        y_center = 1.5 if x_val < 1.5 else 1
        rect = Rectangle((x_val - width/2, y_center - 0.2), width, 0.4,
                         fill=True, alpha=0.4, edgecolor='darkred',
                         facecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(0.5, 2)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$F(x)$', fontsize=12)
    ax.set_title('Upper Semicontinuous (U.S.C.)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(1.5, 0.7, 'Values "jump down" only', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Lower semicontinuous
    ax = axes[1]
    for x_val in [0.5, 1.5, 2.5]:
        if x_val < 1:
            width = 0.2 + 0.2 * x_val
        elif x_val < 2:
            width = 0.4
        else:
            width = 0.4 - 0.1 * (x_val - 2)

        y_center = 1
        rect = Rectangle((x_val - width/2, y_center - 0.2), width, 0.4,
                         fill=True, alpha=0.4, edgecolor='darkgreen',
                         facecolor='green', linewidth=2)
        ax.add_patch(rect)

    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$F(x)$', fontsize=12)
    ax.set_title('Lower Semicontinuous (L.S.C.)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(1.5, 0.65, 'Values "jump up" only', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    save_figure('02_semicontinuity')

# Figure 3: Measurable Selection
def fig_measurable_selection():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Draw multifunction as band
    t = np.linspace(0, 1, 100)
    upper = 1 + 0.5 * np.sin(2 * np.pi * t)
    lower = 0.2 + 0.3 * np.sin(2 * np.pi * t + np.pi/4)

    ax.fill_between(t, lower, upper, alpha=0.3, color='blue', label='$F(t)$ (multifunction)')
    ax.plot(t, upper, 'b--', linewidth=1.5, alpha=0.7)
    ax.plot(t, lower, 'b--', linewidth=1.5, alpha=0.7)

    # Draw a measurable selection
    selection = lower + 0.5 * (upper - lower) * np.sin(np.pi * t)
    ax.plot(t, selection, 'r-', linewidth=2.5, label='$f(t) \\in F(t)$ (measurable selection)')

    # Add arrows
    for t_val in [0.2, 0.5, 0.8]:
        idx = int(t_val * len(t))
        ax.annotate('', xy=(t_val, selection[idx]), xytext=(t_val, upper[idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='darkred'))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 2)
    ax.set_xlabel('$s$ (parameter)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Measurable Selection Theorem', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, -0.05, 'Selection $f(s)$ exists with $f(s) \\in F(s)$ a.e.',
           ha='center', fontsize=11, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    save_figure('03_measurable_selection')

# Figure 4: Integral Inclusion Geometry
def fig_integral_inclusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Solution set visualization
    ax = axes[0]
    t = np.linspace(0, 1, 100)

    # Plot multiple solution trajectories
    for i, x0 in enumerate([0.3, 0.5, 0.7]):
        # Approximate solutions
        integrand = 0.1 * np.sin(t) * x0
        solution = x0 + np.cumsum(integrand) * (t[1] - t[0])
        solution = x0 + (np.exp(0.05) - 1) * x0 * t  # approximation
        ax.plot(t, solution, linewidth=2, alpha=0.7, label=f'$x_0={x0}$')

    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$x(t)$', fontsize=12)
    ax.set_title('Solution Trajectories', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Operator diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw operator diagram
    boxes = [
        (2, 8, 'Domain\n$C[0,T]$', 'lightblue'),
        (8, 8, 'Codomain\n$2^{C[0,T]}$', 'lightcoral'),
        (2, 2, 'Fixed point\n$x \\in Ax$', 'lightgreen'),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x-1, y-0.8), 2, 1.6,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color,
                              linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrow1 = FancyArrowPatch((3.2, 7.5), (6.8, 7.5),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2.5, color='darkblue')
    ax.add_patch(arrow1)
    ax.text(5, 7.9, 'Operator $A$', ha='center', fontsize=11, fontweight='bold')

    arrow2 = FancyArrowPatch((2, 1.2), (2, 0.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='darkgreen', linestyle='--')
    ax.add_patch(arrow2)
    ax.text(3, 1.0, 'Fixed Point\nTheorem', ha='left', fontsize=10)

    ax.set_title('Fixed Point Problem Formulation', fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_figure('04_integral_inclusion')

# Figure 5: Numerical Solution
def fig_numerical_solution():
    # Solve x(t) = 0.5 + int_0^t 0.1*sin(s)*x(s) ds
    def f(x, t):
        return 0.1 * np.sin(t) * x

    t = np.linspace(0, 2, 200)
    x_solutions = []
    x0_values = [0.3, 0.5, 0.7, 0.9]

    for x0 in x0_values:
        sol = odeint(f, x0, t)
        x_solutions.append(sol.flatten())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot solutions
    ax = axes[0, 0]
    for x0, sol in zip(x0_values, x_solutions):
        ax.plot(t, sol, linewidth=2, label=f'$x_0 = {x0}$', alpha=0.7)
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('$x(t)$', fontsize=11)
    ax.set_title('Solutions for Different Initial Conditions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Phase portrait
    ax = axes[0, 1]
    for i, x0 in enumerate(x0_values):
        sol = x_solutions[i]
        t_plot = t[::5]
        sol_plot = sol[::5]
        ax.scatter(t_plot[:-1], sol_plot[:-1], s=30, alpha=0.5, label=f'$x_0={x0}$')
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('$x(t)$', fontsize=11)
    ax.set_title('Phase Portrait', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Verification: check inclusion
    ax = axes[1, 0]
    x0 = 0.5
    sol = odeint(f, x0, t).flatten()

    # Compute integral term
    dt = t[1] - t[0]
    integral_term = np.zeros_like(t)
    for i in range(1, len(t)):
        # Trapezoidal rule
        integral_term[i] = trapezoid(0.1 * np.sin(t[:i]) * sol[:i], t[:i])

    rhs = 0.5 + integral_term
    error = np.abs(sol - rhs)

    ax.semilogy(t, error + 1e-10, 'r-', linewidth=2)
    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('Inclusion Error', fontsize=11)
    ax.set_title('Verification: Residual Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Stability analysis
    ax = axes[1, 1]
    growth_rates = []
    for x0 in x0_values:
        sol = odeint(f, x0, t).flatten()
        growth = np.log(sol[-1] / x0) / t[-1] if sol[-1] > 0 else 0
        growth_rates.append(growth)

    ax.bar(range(len(x0_values)), growth_rates, color='steelblue', alpha=0.7,
           edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(x0_values)))
    ax.set_xticklabels([f'$x_0={x0}$' for x0 in x0_values])
    ax.set_ylabel('Growth Rate', fontsize=11)
    ax.set_title('Growth Rate Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    plt.tight_layout()
    save_figure('05_numerical_solution')

# Figure 6: Convergence of Approximations
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Stepwise approximation
    ax = axes[0]
    t_exact = np.linspace(0, 1, 200)
    x_exact = np.exp(0.05 * t_exact)  # approximate solution

    # Create step approximation
    n_steps_vals = [2, 4, 8, 16]
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_steps_vals)))

    for n_steps, color in zip(n_steps_vals, colors):
        t_step = np.linspace(0, 1, n_steps + 1)
        x_step = np.exp(0.05 * t_step)
        ax.step(t_step, x_step, where='post', linewidth=2, alpha=0.7,
               color=color, label=f'$n={n_steps}$ steps')

    ax.plot(t_exact, x_exact, 'k--', linewidth=2.5, label='Exact solution')
    ax.set_xlabel('Time $t$', fontsize=12)
    ax.set_ylabel('Approximation', fontsize=12)
    ax.set_title('Stepwise Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Error analysis
    ax = axes[1]
    errors = []
    n_values = np.array([2, 4, 8, 16, 32, 64])

    for n in n_values:
        t_step = np.linspace(0, 1, n + 1)
        x_step = np.exp(0.05 * t_step)
        t_interp = np.interp(t_exact, t_step, x_step)
        error = np.max(np.abs(x_exact - t_interp))
        errors.append(error)

    errors = np.array(errors)
    ax.loglog(n_values, errors, 'bo-', markersize=8, linewidth=2.5,
             label='Convergence rate')

    # Add reference line for O(1/n)
    ref_line = errors[0] * (n_values[0] / n_values)
    ax.loglog(n_values, ref_line, 'r--', linewidth=2, label='$O(1/n)$ reference')

    ax.set_xlabel('Number of steps $n$', fontsize=12)
    ax.set_ylabel('Maximum error', fontsize=12)
    ax.set_title('Error Decay: Convergence Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure('06_convergence')

# Figure 7: Hammerstein-type problem illustration
def fig_hammerstein():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Kernel function
    ax = axes[0]
    t_vals = np.linspace(0, 1, 50)
    s_vals = np.linspace(0, 1, 50)
    T_mesh, S_mesh = np.meshgrid(t_vals, s_vals)

    # Simple kernel: k(t,s) = exp(-(t-s)^2)
    K_mesh = np.exp(-((T_mesh - S_mesh)**2) / 0.1)

    contour = ax.contourf(T_mesh, S_mesh, K_mesh, levels=15, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('$k(t,s)$', fontsize=11)
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$s$', fontsize=12)
    ax.set_title('Hammerstein Kernel $k(t,s)$', fontsize=13, fontweight='bold')

    # Right: Nonlinearity
    ax = axes[1]
    x_vals = np.linspace(-1, 1, 200)

    # Monotone nonlinearity
    g1 = x_vals**3
    ax.plot(x_vals, g1, 'b-', linewidth=2.5, label='$g(x) = x^3$ (monotone)')

    # Another example
    g2 = np.tanh(x_vals)
    ax.plot(x_vals, g2, 'r-', linewidth=2.5, label='$g(x) = \\tanh(x)$ (bounded)')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$g(x)$', fontsize=12)
    ax.set_title('Nonlinearity Examples', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure('07_hammerstein')

# Figure 8: Banach Fixed Point Contraction
def fig_contraction():
    fig, ax = plt.subplots(figsize=(11, 7))

    # Draw unit square domain
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # Draw y=x line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='$y = x$')

    # Draw contraction map examples
    x_vals = np.linspace(0, 1, 100)

    # Example 1: x -> 0.5*x + 0.2
    y1 = 0.5 * x_vals + 0.2
    ax.plot(x_vals, y1, 'b-', linewidth=2.5, label='$T(x) = 0.5x + 0.2$ (contraction, $\\rho=0.5$)')

    # Example 2: x -> 0.8*x + 0.1
    y2 = 0.8 * x_vals + 0.1
    ax.plot(x_vals, y2, 'r-', linewidth=2.5, label='$T(x) = 0.8x + 0.1$ (contraction, $\\rho=0.8$)')

    # Example 3: NOT a contraction
    y3 = 1.2 * x_vals - 0.05
    y3 = np.clip(y3, 0, 1)
    ax.plot(x_vals, y3, 'g--', linewidth=2, alpha=0.7, label='$T(x) = 1.2x - 0.05$ (NOT a contraction)')

    # Show fixed points
    fixed_pt1 = (0.2 / (1 - 0.5),)  # 0.4
    fixed_pt2 = (0.1 / (1 - 0.8),)  # 0.5

    ax.scatter([fixed_pt1[0], fixed_pt2[0]], [fixed_pt1[0], fixed_pt2[0]],
              color=['blue', 'red'], s=150, zorder=5, edgecolors='black', linewidth=2)
    ax.text(fixed_pt1[0], fixed_pt1[0] - 0.08, f'$x^* \\approx 0.40$', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(fixed_pt2[0], fixed_pt2[0] - 0.08, f'$x^* \\approx 0.50$', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Show iteration
    x_iter = [0.1]
    for _ in range(5):
        x_iter.append(0.5 * x_iter[-1] + 0.2)

    for i in range(len(x_iter) - 1):
        ax.plot([x_iter[i], x_iter[i]], [x_iter[i], 0.5*x_iter[i] + 0.2],
               'b-', alpha=0.4, linewidth=1)
        ax.plot([x_iter[i], x_iter[i+1]], [0.5*x_iter[i] + 0.2, 0.5*x_iter[i] + 0.2],
               'b-', alpha=0.4, linewidth=1)
        ax.scatter(x_iter[i], x_iter[i], color='blue', s=40, alpha=0.6)

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$T(x)$', fontsize=12)
    ax.set_title('Banach Contraction Principle', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure('08_contraction')

def main():
    """Generate all figures"""
    print("Generating figures for Chapter 10b...")

    fig_multifunction_concept()
    fig_semicontinuity()
    fig_measurable_selection()
    fig_integral_inclusion()
    fig_numerical_solution()
    fig_convergence()
    fig_hammerstein()
    fig_contraction()

    print("\nAll figures generated successfully!")
    print("PDF files saved in: ./figures/")

if __name__ == '__main__':
    main()
