# Chapter 1: Background - Beamer Presentation

This directory contains a comprehensive LaTeX Beamer presentation covering Chapter 1 ("Background") from "Convex Analysis and Monotone Operator Theory in Hilbert Spaces, 2e (2017)" by H. H. Bauschke and P. L. Combettes.

## Files

### Main Presentation
- **chapter1_slides.tex** - Complete LaTeX Beamer source (20 KB, 39 slides)
- **chapter1_slides.pdf** - Compiled PDF presentation (446 KB)

### Figure Generation
- **figures/gen_figures.py** - Python script to generate all diagrams
- **figures/operators_example.py** - Example code demonstrating operators

### Generated Figures
All figures are stored in `figures/` directory:
- `lower_semicontinuous.pdf` - Lower semicontinuous function illustration
- `order_relations.pdf` - Order relation properties and hierarchies
- `extended_real_line.pdf` - The extended real line [-∞, +∞]
- `function_concepts.pdf` - Graph, epigraph, and level sets
- `net_vs_sequence.pdf` - Nets vs. sequences comparison
- `closure_interior.pdf` - Closure and interior of sets
- `compactness.pdf` - Compact vs. non-compact sets
- `operators_diagram.pdf` - Single-valued and set-valued operators

## Content Overview

The presentation covers 11 main sections:

1. **1.1 Sets and Basic Notation** - Foundational concepts and line segments
2. **1.2 Operators** - Single-valued and set-valued operators, graphs, inverses
3. **1.3 Order** - Binary relations, ordered sets, Zorn's lemma
4. **1.4 Nets** - Generalized sequences and net convergence
5. **1.5 Extended Real Line** - The space [-∞, +∞] and arithmetic rules
6. **1.6 Functions** - Functions to extended reals, domains, epigraphs, level sets
7. **1.7 Topological Spaces** - Topology, closure, interior, compactness
8. **1.8 Two-Point Compactification** - Compactness of [-∞, +∞]
9. **1.10 Lower Semicontinuity** - Definition and characterizations
10. **1.11 Sequential Topological Notions** - Sequential vs. topological properties
11. **Summary and Examples** - Key results and computational examples

## Key Theorems Covered

- **Fact 1.1 (Zorn's Lemma)** - Existence of maximal elements
- **Fact 1.11** - Compactness characterization
- **Fact 1.15** - Net convergence in extended real line
- **Lemma 1.23** - Lower semicontinuity characterizations
- **Lemma 1.6** - Epigraph properties

## Features

- **39 slides** organized into clear sections
- **8 custom figures** illustrating key concepts
- **Python code examples** demonstrating operator concepts
- **Computational examples** with function analysis
- **Tables and summaries** of key definitions and theorems
- **Self-contained**: Every concept explained without referring to the book
- **Madrid theme** with seahorse color scheme for professional appearance

## Compilation

To regenerate the figures:
```bash
python3 figures/gen_figures.py
```

To compile the presentation:
```bash
pdflatex -interaction=nonstopmode chapter1_slides.tex
```

Run pdflatex twice to ensure all cross-references are correct.

## Requirements

- TeXLive or MiKTeX (for pdflatex)
- Python 3 with matplotlib and numpy (for figure generation)
- All figures are provided as PDFs, so regeneration is optional

## Notes

The presentation follows the conventions:
- Beamer document class with 16:9 aspect ratio
- All figures embedded as PDFs in the figures/ subdirectory
- Mathematical notation using standard LaTeX/AMS packages
- Code listings with syntax highlighting
- Frame numbering in footer for easy reference

This presentation is ideal for:
- Graduate courses on convex analysis
- Functional analysis instruction
- Self-study and review of background material
- Teaching monotone operator theory

## Author Reference

Based on: H. H. Bauschke, P. L. Combettes, "Convex Analysis and Monotone Operator Theory in Hilbert Spaces," CMS Books in Mathematics, Springer, 2017.
