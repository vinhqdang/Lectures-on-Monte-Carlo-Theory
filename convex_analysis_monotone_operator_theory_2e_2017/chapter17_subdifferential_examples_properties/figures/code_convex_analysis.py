import numpy as np
import matplotlib.pyplot as plt

class ConvexFunction:
    """Analysis of convex functions"""

    def __init__(self, f, grad_f, name="f"):
        """
        f: function
        grad_f: gradient function
        """
        self.f = f
        self.grad_f = grad_f
        self.name = name

    def is_monotone_gradient(self, x_vals):
        """Check if gradient is monotone"""
        grads = [self.grad_f(x) for x in x_vals]
        return all(grads[i] <= grads[i+1]
                  for i in range(len(grads)-1))

    def tangent_line(self, x0, x):
        """Compute tangent line at x0"""
        f_x0 = self.f(x0)
        grad = self.grad_f(x0)
        return f_x0 + grad * (x - x0)

    def verify_convexity(self, x_vals):
        """Verify convexity using first-order condition"""
        for x in x_vals:
            tangent = self.tangent_line(x, x_vals)
            f_vals = np.array([self.f(xi) for xi in x_vals])
            if not np.all(f_vals >= tangent - 1e-10):
                return False
        return True

# Example 1: f(x) = x^2
f1 = lambda x: x**2
grad_f1 = lambda x: 2*x

func1 = ConvexFunction(f1, grad_f1, "f(x) = x²")
x_test = np.linspace(-2, 2, 100)

print(f"Function: {func1.name}")
print(f"Gradient is monotone: {func1.is_monotone_gradient(x_test)}")
print(f"Function is convex: {func1.verify_convexity(x_test)}")

# Example 2: f(x) = e^x
f2 = lambda x: np.exp(x)
grad_f2 = lambda x: np.exp(x)

func2 = ConvexFunction(f2, grad_f2, "f(x) = exp(x)")
print(f"\nFunction: {func2.name}")
print(f"Gradient is monotone: {func2.is_monotone_gradient(x_test)}")
print(f"Function is convex: {func2.verify_convexity(x_test)}")
