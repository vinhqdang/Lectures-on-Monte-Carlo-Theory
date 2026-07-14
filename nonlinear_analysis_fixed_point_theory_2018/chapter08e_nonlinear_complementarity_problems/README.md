# Chapter 8e: Nonlinear Complementarity Problems

## Overview
This directory contains comprehensive LaTeX Beamer slides on Nonlinear Complementarity Problems, based on section 9.8 of Pathak's *Introduction to Nonlinear Analysis and Fixed Point Theory*.

## Contents

### Main Files
- **chapter8e_slides.tex** - Main LaTeX Beamer source file (35 pages)
- **chapter8e_slides.pdf** - Compiled presentation (385 KB, 35 pages)

### Generated Figures
All figures are in PDF format for high-quality printing and embedding:
- **01_cone_dual_cone.pdf** - Visualization of cones and dual cones
- **02_complementarity_condition.pdf** - Illustration of complementarity conditions
- **03_convergence.pdf** - Convergence analysis of iterative methods
- **04_problem_types.pdf** - Taxonomy of complementarity problem types
- **05_monotonicity_properties.pdf** - Strongly monotone and Lipschitz mappings
- **06_applications.pdf** - Applications framework

### Scripts
- **figures/gen_figures.py** - Python script to regenerate all figures using matplotlib

## Content Structure

### 1. Introduction and Motivation (Slides 1-3)
- Historical development of complementarity problems
- Wide range of applications in optimization, engineering, and economics

### 2. Basic Concepts and Definitions (Slides 4-6)
- Cones and dual cones
- Complementarity conditions
- Geometric interpretation

### 3. Problem Formulations (Slides 7-9)
- Explicit Complementarity Problem (E.C.P)
- Implicit Complementarity Problem (I.C.P)
- Simultaneous Complementarity Problems (S.E.C.P, S.I.C.P)

### 4. Key Definitions and Properties (Slides 10-11)
- Pairwise n-Lipschitz mappings
- Pairwise n-strongly monotone mappings
- Mathematical framework

### 5. Main Existence and Uniqueness Results (Slides 12-14)
- **Theorem 9.39 (Pathak et al.)** - Main result
- Proof strategy and outline
- Contractivity analysis

### 6. Numerical Methods and Examples (Slides 15-19)
- Fixed point iteration algorithm
- 2D complementarity example with Python code
- Convergence behavior analysis

### 7. Applications (Slides 20-23)
- Variational inequalities
- Economic equilibrium problems
- Mechanics and contact problems
- Optimization with complementarity constraints

### 8. Special Cases and Remarks (Slides 24-26)
- Strongly nonlinear quasi-complementarity
- Problem variants and reductions

### 9. Summary and Exercises (Slides 27-35)
- Key takeaways
- 9 exercises with varying difficulty
- Further reading and references

## Key Topics Covered

### Mathematical Framework
- Hilbert space theory
- Fixed point theorems
- Monotone operators
- Projection operators
- Contractivity conditions

### Problem Classes
- **E.C.P**: $x_0 \in K$, $f(x_0) \in K^*$, $\langle x_0, f(x_0) \rangle = 0$
- **I.C.P**: Generalization with auxiliary mapping $g$
- **S.E.C.P**: Simultaneous explicit complementarity
- **S.I.C.P**: Simultaneous implicit complementarity

### Main Results
- Existence conditions via strong monotonicity
- Uniqueness via injectivity of reference mapping
- Linear convergence of fixed point iteration

### Applications
1. **Variational Inequalities** - Connection to optimization problems
2. **Economic Equilibrium** - Market clearing and price theory
3. **Mechanics** - Contact problems, friction, adhesion
4. **Optimization** - Bilevel programming, inverse problems

## Technical Details

### Compilation
```bash
# Generate figures
python3 figures/gen_figures.py

# Compile LaTeX (twice for correct numbering)
pdflatex -interaction=nonstopmode chapter8e_slides.tex
pdflatex -interaction=nonstopmode chapter8e_slides.tex
```

### Requirements
- LaTeX with Beamer class
- matplotlib (for figure generation)
- numpy, scipy (for numerical examples)

### File Statistics
- LaTeX source: ~1000 lines
- Figures: 6 PDF files (172 KB total)
- Compiled slides: 35 pages (385 KB)

## Source Material
- Pathak, H. K. (2018). *An Introduction to Nonlinear Analysis and Fixed Point Theory*. Springer Nature Singapore.
- Section 9.8: Application to Simultaneous Complementarity Problems (Pages 758-764)

## Mathematical Notation
- $\mathbb{R}^n_+$ - Positive orthant (non-negative reals)
- $K$ - Closed convex cone
- $K^*$ - Dual cone
- $\langle \cdot, \cdot \rangle$ - Inner product
- $P_K$ - Projection operator onto $K$
- $f^n$ - n-fold composition of $f$

## Theorem 9.39 (Main Result)
Let $\mathcal{H}$ be a Hilbert space and $K$ a closed convex cone. If $f_1, f_2, g : D \to \mathcal{H}$ satisfy:
1. Pairwise n-strongly monotone with constant $a > 0$
2. Pairwise n-Lipschitz with constant $\beta > 0$
3. There exists $r > 0$ such that $r\beta^2 < 2a < \frac{1}{r} + r\beta^2$
4. $K \subseteq g(D)$

Then S.I.C.P$(f_1, f_2, g, K)$ has a solution, and if $g$ is injective, the solution is unique.

## Contact and References
For more information about nonlinear analysis and fixed point theory, see:
- Pathak, H. K. et al. (1996). Selected papers on complementarity problems
- Related topics: Variational inequalities, Fixed point theory, Operator theory
