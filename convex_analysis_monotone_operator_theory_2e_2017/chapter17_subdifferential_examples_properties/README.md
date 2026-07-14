# Chapter 17: Differentiability of Convex Functions - Beamer Slides

## Contents

This directory contains comprehensive LaTeX Beamer slides for Chapter 17 of "Convex Analysis and Monotone Operator Theory in Hilbert Spaces" (2nd Edition, 2017).

### Files

- **chapter17_slides.tex**: Main LaTeX Beamer presentation (25 slides)
- **chapter17_slides.pdf**: Compiled PDF (ready to present)
- **figures/gen_figures.py**: Python script to generate all figures
- **figures/*.pdf**: Generated figures for each section

### Topics Covered

1. **Section 17.1: Directional Derivatives**
   - Definition and basic properties (Propositions 17.2-17.4)

2. **Section 17.2: Characterizations of Convexity**
   - First-order conditions via gradient monotonicity (Proposition 17.7)
   - Examples: quadratic forms and extensions

3. **Section 17.3: Strict Convexity**
   - Strict convexity characterizations (Proposition 17.10)
   - Examples with explicit computation

4. **Section 17.4: Directional Derivatives and Subgradients**
   - Descent directions (Propositions 17.21-17.22)
   - Steepest descent direction
   - Chebyshev center application (Proposition 17.25)

5. **Section 17.5: Directional Derivatives and Convexity**
   - Sufficient conditions for convexity (Propositions 17.27-17.29)

6. **Section 17.6: Gâteaux and Fréchet Differentiability**
   - Relationship to subdifferentials (Propositions 17.31-17.41)
   - Chain rules and examples
   - Finite-dimensional results (Corollary 17.44)

7. **Section 17.7: Differentiability and Continuity**
   - Continuity from differentiability (Propositions 17.48-17.51)
   - Counterexamples

8. **Python Numerical Examples**
   - Computing directional derivatives
   - Verifying convexity properties

### Figures

- **directional_derivative.pdf**: Visual concept of directional derivatives
- **convexity_characterizations.pdf**: Convex vs gradient properties
- **subdifferential.pdf**: Smooth vs non-smooth subdifferentials
- **descent_directions.pdf**: Steepest descent visualization
- **gateau_frechet.pdf**: Comparison of differentiability concepts

### How to Compile

```bash
cd chapter17_subdifferential_examples_properties
pdflatex -interaction=nonstopmode chapter17_slides.tex
pdflatex -interaction=nonstopmode chapter17_slides.tex  # Run twice for all refs
```

### Regenerate Figures

```bash
cd figures
python3 gen_figures.py
```

### LaTeX Features

- **Theme**: Madrid with seahorse color scheme
- **Aspect Ratio**: 16:9 widescreen
- **Font Size**: 10pt
- **Frame Numbers**: Enabled in footer
- **Code Listings**: Python with syntax highlighting

### Mathematical Notation

- Uses standard LaTeX math mode with amsmath, amssymb, amsthm
- Hilbert space notation: $\mathcal{H}$
- Subdifferential: $\partial f(x)$
- Gradient: $\nabla f(x)$
- Expected value: $\mathbb{E}$

## Reference

**Book**: Convex Analysis and Monotone Operator Theory in Hilbert Spaces (2nd Edition)
**Authors**: H.H. Bauschke and P.L. Combettes
**Publisher**: Springer International Publishing AG, 2017
**Pages**: 533-565 (Chapter 17)

