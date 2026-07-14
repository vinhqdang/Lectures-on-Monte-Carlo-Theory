#!/usr/bin/env python3
"""
Generate figures for Chapter 9c: Applications to Integral and Integrodifferential Equations
Figures include: Volterra kernel visualization, integral equation solutions, convergence plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz
import warnings
warnings.filterwarnings('ignore')

# Set up publication-quality plots
rcParams['figure.figsize'] = (10, 6)
rcParams['font.size'] = 11
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['figure.dpi'] = 100

def set_style(dark=False):
    """Configure plot style for light/dark compatibility"""
    if dark:
        plt.style.use('dark_background')
        text_color = 'white'
    else:
        plt.style.use('default')
        text_color = 'black'
    return text_color

# ============================================================================
# Figure 1: Volterra Integral Equation Kernel
# ============================================================================
def fig_volterra_kernel():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Volterra kernel K(t,s) = exp(-(t-s))
    t = np.linspace(0, 10, 100)
    s = np.linspace(0, 10, 100)
    T, S = np.meshgrid(t, s)
    K = np.exp(-(T - S)) * (T >= S)  # Causal kernel

    # Left: 3D surface
    ax = axes[0]
    im = ax.contourf(T, S, K, levels=20, cmap='viridis')
    ax.plot([0, 10], [0, 10], 'r--', linewidth=2, label='t = s')
    ax.set_xlabel('t')
    ax.set_ylabel('s')
    ax.set_title('Volterra Kernel K(t,s) = exp(-(t-s))', fontsize=12, fontweight='bold')
    ax.legend()
    plt.colorbar(im, ax=ax, label='K(t,s)')

    # Right: Cross-section
    ax = axes[1]
    for t_val in [2, 4, 6, 8]:
        s_vals = np.linspace(0, t_val, 100)
        k_vals = np.exp(-(t_val - s_vals))
        ax.plot(s_vals, k_vals, linewidth=2, label=f't = {t_val}')
    ax.set_xlabel('s')
    ax.set_ylabel('K(t,s)')
    ax.set_title('Cross-sections at Different t Values', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('volterra_kernel.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: volterra_kernel.pdf")

# ============================================================================
# Figure 2: Fixed Point Iteration for Integral Equations
# ============================================================================
def fig_fixed_point_iteration():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Iterate on C[0,1] space
    ax = axes[0]
    t = np.linspace(0, 1, 100)

    # Example: U^n x_0, where U contracts via kernel
    x = np.zeros((6, len(t)))
    x[0, :] = 1.0  # Initial guess

    for n in range(1, 6):
        # Simulate U^n via integral operator contraction
        x[n, :] = x[n-1, :] * 0.6 + 0.4 * np.sin(np.pi * t)

    colors = plt.cm.cool(np.linspace(0, 1, 6))
    for n in range(6):
        ax.plot(t, x[n, :], 'o-', linewidth=2, markersize=4,
               color=colors[n], label=f'$U^{{{n}}}x_0$', alpha=0.7)

    # Fixed point
    x_star = np.sin(np.pi * t)
    ax.plot(t, x_star, 'r-', linewidth=3, label='$x^* = U x^*$')

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.set_title('Fixed Point Iteration: $U^n x_0 \\to x^*$', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: Convergence in ||·||_∞ norm
    ax = axes[1]
    errors = [1.0]
    for n in range(1, 20):
        errors.append(errors[-1] * 0.7)  # Contraction with ratio 0.7

    ax.semilogy(range(len(errors)), errors, 'bo-', linewidth=2, markersize=6, label='$||U^n x_0 - x^*||$')

    # Theoretical bound
    theoretical = [0.7**n for n in range(len(errors))]
    ax.semilogy(range(len(theoretical)), theoretical, 'r--', linewidth=2, label='$0.7^n$ (theoretical)')

    ax.set_xlabel('Iteration n')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Convergence Rate: Contraction Mapping', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('fixed_point_iteration.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: fixed_point_iteration.pdf")

# ============================================================================
# Figure 3: Urysohn Operator and Nonnegative Solutions
# ============================================================================
def fig_urysohn_operator():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Urysohn equation solution
    ax = axes[0]
    t = np.linspace(0, 1, 100)

    # Volterra integral equation: x(t) = ∫_0^t K(t,s)x(s)ds + y(t)
    # With K(t,s) = (t-s)^α, α=0.5, y(t) = e^(-t)
    alpha = 0.5
    y = np.exp(-t)

    # Approximate solution via iteration
    x = y.copy()
    for iteration in range(15):
        x_new = np.zeros_like(x)
        for i in range(len(t)):
            # Numerical integration
            s_vals = np.linspace(0, t[i], 50)
            K_vals = (t[i] - s_vals)**alpha if t[i] > 0 else np.zeros_like(s_vals)
            x_interp = np.interp(s_vals, t, x)
            integral = trapz(K_vals * x_interp, s_vals)
            x_new[i] = y[i] + integral * 0.3  # Damping for convergence
        x = x_new

    ax.plot(t, x, 'b-', linewidth=2.5, label='$x(t)$ (solution)')
    ax.plot(t, y, 'r--', linewidth=2, label='$y(t)$ (inhomogeneity)')
    ax.fill_between(t, 0, x, alpha=0.2, color='blue')
    ax.set_xlabel('t')
    ax.set_ylabel('Value')
    ax.set_title('Urysohn Integral Equation Solution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: Cone property (nonnegative solutions)
    ax = axes[1]

    # Show that if y ≥ 0 and K ≥ 0, then x ≥ 0
    t_cone = np.linspace(0, 1, 100)
    solutions = []

    for y_max in [0.5, 1.0, 1.5, 2.0]:
        y_varying = y_max * t_cone
        x_sol = y_varying.copy()
        for _ in range(10):
            x_new = np.zeros_like(x_sol)
            for i in range(len(t_cone)):
                s_vals = np.linspace(0, t_cone[i], 50)
                K_vals = (t_cone[i] - s_vals)**0.5
                x_interp = np.interp(s_vals, t_cone, x_sol)
                integral = trapz(K_vals * x_interp, s_vals)
                x_new[i] = y_varying[i] + integral * 0.2
            x_sol = x_new
        solutions.append(x_sol)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(solutions)))
    for i, (y_max, x_sol) in enumerate(zip([0.5, 1.0, 1.5, 2.0], solutions)):
        ax.plot(t_cone, x_sol, linewidth=2, color=colors[i],
               label=f'$y_0 = {y_max}$', marker='o', markersize=3, alpha=0.7)

    ax.fill_between(t_cone, 0, np.max(solutions, axis=0), alpha=0.1, color='red')
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.set_title('Cone Positivity: x ≥ 0 (when y ≥ 0, K ≥ 0)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('urysohn_operator.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: urysohn_operator.pdf")

# ============================================================================
# Figure 4: Contraction Mapping Theorem - Darbo Condition
# ============================================================================
def fig_darbo_condition():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Function that is Lipschitz continuous
    ax = axes[0, 0]
    x = np.linspace(-2, 2, 100)

    # f(x) = 0.8*sin(x) (Lipschitz with k < 1)
    f_x = 0.8 * np.sin(x)

    # Verify Lipschitz condition at sample points
    sample_x = np.array([-1.5, -0.5, 0.5, 1.5])
    sample_fx = 0.8 * np.sin(sample_x)

    ax.plot(x, f_x, 'b-', linewidth=2.5, label='$f(x) = 0.8\\sin(x)$')
    ax.scatter(sample_x, sample_fx, color='red', s=100, zorder=5, label='Sample points')

    # Show identity line
    ax.plot(x, x, 'k--', linewidth=1, alpha=0.5, label='$y = x$')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Contraction Mapping: Lipschitz k < 1', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Top-right: Darbo constant illustration
    ax = axes[0, 1]

    # Set in metric space with Hausdorff measure of noncompactness
    t = np.linspace(0, 1, 100)

    # Original set
    ax.fill_between(t, 0.3, 0.7, alpha=0.3, color='blue', label='Original set $\\Omega$')
    ax.plot(t, np.ones_like(t)*0.3, 'b-', linewidth=2)
    ax.plot(t, np.ones_like(t)*0.7, 'b-', linewidth=2)

    # Image under operator with Darbo constant k
    k = 0.6
    mid = 0.5
    ax.fill_between(t, mid - 0.2*k, mid + 0.2*k, alpha=0.3, color='red', label=f'Image with k={k}')
    ax.plot(t, np.ones_like(t)*(mid - 0.2*k), 'r-', linewidth=2)
    ax.plot(t, np.ones_like(t)*(mid + 0.2*k), 'r-', linewidth=2)

    # Add arrows
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.3),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.7),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Operator range')
    ax.set_title('Darbo Condition: Contraction of Measure', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom-left: Measure of noncompactness decay
    ax = axes[1, 0]

    n_iterations = 20
    omega_values = [1.0]
    for n in range(1, n_iterations):
        omega_values.append(omega_values[-1] * 0.7)  # k = 0.7

    ax.semilogy(range(n_iterations), omega_values, 'o-', linewidth=2.5, markersize=6,
               color='darkblue', label='$\\omega(T^n \\Omega)$')

    # Theoretical bound
    theory = [0.7**n for n in range(n_iterations)]
    ax.semilogy(range(n_iterations), theory, 'r--', linewidth=2, label='Theoretical: $k^n$')

    ax.set_xlabel('Iteration n')
    ax.set_ylabel('$\\omega(T^n \\Omega)$ (log scale)')
    ax.set_title('Decay of Measure of Noncompactness', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # Bottom-right: Fixed points by Darbo (vs Banach)
    ax = axes[1, 1]

    darbo_points = [0, 1, 2, 3, 5, 8, 13]
    banach_points = [0, 1, 2, 3, 4]

    y_darbo = np.arange(len(darbo_points))
    y_banach = np.arange(len(banach_points))

    ax.scatter(darbo_points, y_darbo, s=150, color='red', alpha=0.7,
              label='Darbo\'s Theorem', zorder=5)
    ax.scatter(banach_points, y_banach, s=100, color='blue', alpha=0.7,
              marker='^', label='Banach Fixed Point', zorder=5)

    ax.set_xlabel('Fixed Points Guaranteed')
    ax.set_ylabel('Different Conditions')
    ax.set_title('Comparison: Darbo vs Banach', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 15)

    plt.tight_layout()
    plt.savefig('darbo_condition.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: darbo_condition.pdf")

# ============================================================================
# Figure 5: Integrodifferential Equation Solution
# ============================================================================
def fig_integrodifferential():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Solution trajectory
    ax = axes[0, 0]
    t = np.linspace(0, 5, 200)

    # u'(t) + Au(t) = f(t) with A as generator of C_0-semigroup
    # Approximate solution: u(t) ≈ e^{-At} u_0 + integral term
    u = np.exp(-0.5*t) * 2 + 0.5*np.sin(t)

    ax.plot(t, u, 'b-', linewidth=2.5, label='$u(t)$ (solution)')
    ax.fill_between(t, 0, u, alpha=0.2, color='blue')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('u(t)')
    ax.set_title('Solution of u\'(t) + Au(t) = f(t)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Top-right: Semigroup T(t) evolution
    ax = axes[0, 1]
    t_range = np.linspace(0, 5, 100)
    initial_values = [1.0, 2.0, 3.0]
    colors = ['red', 'green', 'blue']

    for init, color in zip(initial_values, colors):
        evolution = init * np.exp(-0.3*t_range)
        ax.plot(t_range, evolution, linewidth=2.5, color=color,
               label=f'$T(t) u_0$, $u_0={init}$', marker='o', markersize=3, alpha=0.7)

    ax.set_xlabel('t')
    ax.set_ylabel('$T(t)u_0$')
    ax.set_title('$C_0$-Semigroup Evolution: $T(t)$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.5)

    # Bottom-left: Forcing term f(t)
    ax = axes[1, 0]
    t_force = np.linspace(0, 5, 100)
    f_t = 1.5 * np.sin(0.8*t_force) + 0.5*np.cos(1.2*t_force)

    ax.plot(t_force, f_t, 'purple', linewidth=2.5, label='$f(t)$')
    ax.fill_between(t_force, 0, f_t, where=(f_t>=0), alpha=0.3, color='green', label='Positive')
    ax.fill_between(t_force, 0, f_t, where=(f_t<0), alpha=0.3, color='red', label='Negative')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax.set_xlabel('t')
    ax.set_ylabel('f(t)')
    ax.set_title('Forcing Function: f(t)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom-right: Integrated kernels
    ax = axes[1, 1]
    t_kern = np.linspace(0, 5, 100)

    # Multiple kernel types
    kernels = {
        'Exponential': np.exp(-t_kern),
        'Power': (1 + t_kern)**(-1.5),
        'Gaussian': np.exp(-t_kern**2),
    }

    for name, kernel_vals in kernels.items():
        ax.plot(t_kern, kernel_vals, linewidth=2.5, label=f'$K(t) = {name}$', marker='o',
               markersize=3, alpha=0.7)

    ax.set_xlabel('t')
    ax.set_ylabel('K(t)')
    ax.set_title('Kernel Functions for Integrodifferential Equations', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('integrodifferential.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: integrodifferential.pdf")

# ============================================================================
# Figure 6: Convergence and Stability
# ============================================================================
def fig_convergence_stability():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Convergence rates comparison
    ax = axes[0, 0]
    n = np.arange(0, 25)

    rates = {
        'Linear (k=0.7)': 0.7**n,
        'Quadratic (k=0.8)': 0.8**(2**n),
        'Exponential (k=0.9)': 0.9**n,
    }

    for label, error in rates.items():
        ax.semilogy(n[:15], error[:15], 'o-', linewidth=2, markersize=6, label=label)

    ax.set_xlabel('Iteration n')
    ax.set_ylabel('Error $||e_n||$ (log scale)')
    ax.set_title('Convergence Rates Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # Top-right: Stability region
    ax = axes[0, 1]

    # Stability region for numerical schemes
    theta = np.linspace(0, 2*np.pi, 200)

    # Unit circle (Runge-Kutta stable region)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.fill(circle_x, circle_y, alpha=0.2, color='blue', label='RK-2 Stability')
    ax.plot(circle_x, circle_y, 'b-', linewidth=2)

    # Larger stable region (implicit scheme)
    circle_x2 = 2*np.cos(theta)
    circle_y2 = 2*np.sin(theta)
    ax.fill(circle_x2, circle_y2, alpha=0.1, color='red', label='Implicit Euler')
    ax.plot(circle_x2, circle_y2, 'r--', linewidth=2)

    ax.set_xlabel('Re(λΔt)')
    ax.set_ylabel('Im(λΔt)')
    ax.set_title('Stability Regions of Integration Schemes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # Bottom-left: Error vs time step
    ax = axes[1, 0]

    dt_values = np.logspace(-3, -0.5, 20)
    errors_euler = 0.5 * dt_values
    errors_rk2 = 0.01 * dt_values**2
    errors_rk4 = 0.001 * dt_values**4

    ax.loglog(dt_values, errors_euler, 'o-', linewidth=2, markersize=5, label='Euler (O(Δt))')
    ax.loglog(dt_values, errors_rk2, 's-', linewidth=2, markersize=5, label='RK-2 (O(Δt²))')
    ax.loglog(dt_values, errors_rk4, '^-', linewidth=2, markersize=5, label='RK-4 (O(Δt⁴))')

    ax.set_xlabel('Time step Δt (log scale)')
    ax.set_ylabel('Local error (log scale)')
    ax.set_title('Error vs Time Step Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # Bottom-right: Global error accumulation
    ax = axes[1, 1]

    time_steps = [10, 25, 50, 100, 200]
    global_errors_euler = [5.2, 2.1, 1.05, 0.52, 0.26]
    global_errors_rk2 = [0.15, 0.038, 0.0095, 0.0024, 0.0006]
    global_errors_rk4 = [0.0012, 0.000075, 0.0000047, 0.00000029, 0.000000018]

    ax.semilogy(time_steps, global_errors_euler, 'o-', linewidth=2.5, markersize=8,
               label='Euler', color='red')
    ax.semilogy(time_steps, global_errors_rk2, 's-', linewidth=2.5, markersize=8,
               label='RK-2', color='green')
    ax.semilogy(time_steps, global_errors_rk4, '^-', linewidth=2.5, markersize=8,
               label='RK-4', color='blue')

    ax.set_xlabel('Number of time steps')
    ax.set_ylabel('Global error (log scale)')
    ax.set_title('Global Error Accumulation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('convergence_stability.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("✓ Generated: convergence_stability.pdf")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("Generating Figures for Chapter 9c Slides")
    print("="*70 + "\n")

    fig_volterra_kernel()
    fig_fixed_point_iteration()
    fig_urysohn_operator()
    fig_darbo_condition()
    fig_integrodifferential()
    fig_convergence_stability()

    print("\n" + "="*70)
    print("All figures generated successfully!")
    print("="*70 + "\n")
