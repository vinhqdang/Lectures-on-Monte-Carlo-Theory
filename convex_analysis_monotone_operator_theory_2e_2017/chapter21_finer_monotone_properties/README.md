# Chapter 21: Finer Monotone Properties - Lecture Slides

## Overview
This directory contains Beamer slides for Chapter 21 of "Convex Analysis and Monotone Operator Theory in Hilbert Spaces" (2nd ed., 2017) by Bauschke & Combettes.

**Chapter Scope**: Pages 370-381 (BC book), covering pages 671-700 of the original reference.

## Main Files

### PDF Slides
- **chapter21_slides.pdf** - Main presentation (31 pages)

### LaTeX Source
- **chapter21_slides.tex** - Complete Beamer presentation source

### Python Figures Generator
- **figures/gen_figures.py** - Python script that generates all PDF figures

### Figures (in figures/ directory)
1. **fig_monotone_property.pdf** - Core monotone operator property visualization
2. **fig_minty.pdf** - Minty's Theorem: Surjectivity of (Id + A)
3. **fig_domain_range.pdf** - Domain vs. Range relationships
4. **fig_local_bdd.pdf** - Rockafellar-Vesely Theorem: Local Boundedness
5. **fig_example_operator.pdf** - Subdifferential as monotone operator
6. **fig_summary.pdf** - Chapter overview diagram

## Slide Content Structure

### Sections Covered

1. **Introduction: Maximally Monotone Operators**
   - Core definitions and concepts
   - Graph, domain, and range terminology

2. **Section 21.1: Minty's Theorem**
   - Theorem 21.1: Maximal monotonicity characterization
   - Minty's Theorem: ran(Id+A) = H
   - Applications (Theorem 21.2, Examples 21.3-21.5)

3. **Section 21.2: The Debrunner-Flor Theorem**
   - Theorem 21.8: Separation property for monotone sets
   - Theorem 21.9: Maximal monotone extension

4. **Section 21.3: Domain and Range**
   - Definition 21.10: Local boundedness
   - Propositions 21.11-21.12: Domain relations
   - Corollary 21.14: Convexity of domain and range
   - Corollary 21.16: Bunt-Kritikos-Motzkin result

5. **Section 21.4: Local Boundedness and Surjectivity**
   - Theorem 21.18: Rockafellar-Vesely characterization
   - Corollaries 21.19-21.26: Surjectivity conditions

6. **Section 21.5: Kenderov's Theorem and Fréchet Differentiability**
   - Theorem 21.27: Generic single-valuedness and continuity
   - Corollary 21.28: Generic Fréchet differentiability of convex functions

7. **Numerical Examples**
   - Python implementation example
   - Subdifferential as monotone operator

## Key Theorems Presented

| Theorem | Result |
|---------|--------|
| 21.1 | Maximal monotonicity ⟺ 0 ∈ ran(A + Id) |
| 21.2 | Subdifferentials ∂f are maximally monotone |
| 21.8 | Debrunner-Flor separation property |
| 21.9 | Existence of maximal monotone extensions |
| 21.18 | Local boundedness ⟺ interior of domain |
| 21.27 | Generic G_δ single-valuedness (Kenderov) |

## How to Regenerate Figures

```bash
cd chapter21_finer_monotone_properties
python3 figures/gen_figures.py
pdflatex -interaction=nonstopmode chapter21_slides.tex
pdflatex -interaction=nonstopmode chapter21_slides.tex
```

## LaTeX Compilation

The slides have been compiled twice to ensure proper:
- Table of contents
- References and cross-references
- Navigation markers for Beamer

## Features

- **Clean Design**: Uses Madrid theme with seahorse color scheme
- **Mathematical Notation**: Comprehensive use of \mathbb for sets and spaces
- **Vector Graphics**: All figures are PDF-based (scalable, high-quality)
- **Code Examples**: Python implementation of key concepts
- **Self-Contained**: No references to the book—all key content is included

## Page Count: 31 Pages

Including:
- 1 Title slide
- 1 Outline slide
- 28 Content slides
- 1 Closing slide

## Technical Details

- **LaTeX Engine**: pdflatex
- **Beamer Theme**: Madrid with seahorse colors
- **Python Version**: 3.11+
- **Dependencies**: matplotlib, numpy (for figure generation)
- **File Size**: ~413 KB (PDF)

## Notes

- All figures are embedded as PDF to maintain vector quality
- The slides are designed for 16:9 aspect ratio (widescreen)
- Font size is 10pt for readability in presentations
