# Chapter 30: Best Approximation Algorithms

Comprehensive LaTeX Beamer slides for Chapter 30 of "Convex Analysis and Monotone Operator Theory in Hilbert Spaces" (2e, 2017).

## Contents

### Main Presentation
- **chapter30_slides.pdf** (36 pages)
  - Professional Beamer slides with Madrid theme
  - 16:9 aspect ratio, 10pt font

### Structure

#### Section 1: Introduction and Halper's Algorithm
- Best approximation problem definition
- Theorem 30.1: Halper's algorithm with convergence conditions
- Corollaries 30.2-30.5
  - Finite family of operators
  - Averaged nonexpansive operators
  - Linear mean ergodic theorem
- Python implementation example

#### Section 2: Dykstra's Algorithm
- Motivation: projection onto intersection of convex sets
- Theorem 30.7: Dykstra's cyclic projection algorithm
- Key insight: residual correction terms
- Comparison with alternating projections
- Algorithm visualization and implementation
- Linear convergence properties

#### Section 3: Haugazeau's Algorithm
- Extension to firmly quasinonexpansive operators
- Theorem 30.8: general framework
- Corollaries 30.9-30.15:
  - Finite family of convex functions
  - Firmly nonexpansive operators
  - Proximal point algorithm for maximally monotone operators
  - Forward-backward splitting for composite optimization
  - Example: composite convex function minimization

#### Section 4: Applications and Numerical Examples
- Projections onto common convex sets (ball, hyperplane, box, cone)
- Convergence rate comparisons
- Numerical examples with Python code
- Forward-backward splitting illustration

#### Section 5: Key Theoretical Results
- Demiclosedness and quasinonexpansiveness definitions
- Convergence analysis lemmata
- Connection to fixed point methods
- Summary of theorems and corollaries

#### Section 6: Exercises and Further Reading
- Selected exercises from the chapter
- Key takeaways
- References

## Generated Figures

All figures are high-quality PDFs in the `figures/` subdirectory:

1. **fig_alternating_projections.pdf**
   - Visualizes method of alternating projections
   - Shows convergence to intersection of two convex sets

2. **fig_halpers_convergence.pdf**
   - Theoretical convergence rates for different λ values
   - Iterate sequence in 2D

3. **fig_dykstra_convergence.pdf**
   - Comparison: Dykstra vs. cyclic projection convergence
   - Projection onto intersection of two circles

4. **fig_haugazeau_algorithm.pdf**
   - Multiple sets intersection visualization
   - Convergence behavior for different numbers of sets

5. **fig_forward_backward.pdf**
   - Problem setup: smooth + nonsmooth objectives
   - Convergence of iterates to minimizer

6. **fig_proximal_point.pdf**
   - Comparison of convergence rates:
     - Proximal point algorithm
     - Gradient descent
     - Accelerated gradient

7. **fig_projections.pdf**
   - Examples of projection onto different convex sets:
     - Ball (Euclidean)
     - Hyperplane
     - Box constraints
     - Cone

## Python Code

**figures/gen_figures.py**: Standalone Python script generating all figures
- Uses matplotlib for high-quality visualizations
- Includes numerical examples of algorithms
- Can be run independently: `python3 figures/gen_figures.py`

## Features

- **Self-contained**: Includes all major theorems, definitions, and key results
- **Illustrated**: 7 professional figures with clear mathematical content
- **Practical**: Python implementations and numerical examples
- **Comprehensive**: 36 slides covering 42 pages of dense mathematical material
- **Professional**: Beamer Madrid theme with seahorse color scheme
- **Accessible**: Explanations of convergence conditions and intuition

## Compilation

To regenerate the presentation:

```bash
cd chapter30_best_approximation_algorithms

# Generate figures
python3 figures/gen_figures.py

# Compile LaTeX (twice for cross-references)
pdflatex -interaction=nonstopmode chapter30_slides.tex
pdflatex -interaction=nonstopmode chapter30_slides.tex
```

## Key Topics Covered

- Best approximation and projection onto convex sets
- Fixed point algorithms and convergence theory
- Alternating projections and cyclic methods
- Residual correction techniques
- Applications to optimization problems
- Monotone operator theory connections
- Forward-backward and proximal point algorithms

## Mathematical Prerequisites

- Hilbert space theory (from Chapter 2)
- Convex analysis basics (from Chapter 4)
- Monotone operator theory (from Chapter 29)
- Projections and properties (from Chapter 4)

## Notes

- All equations follow the notation of Bauschke & Combettes (2017)
- Theorem and corollary numbering matches the original text
- Examples and exercises reference the original chapter
- Convergence properties are illustrated both theoretically and numerically

---

Generated: July 14, 2026
