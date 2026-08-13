/**
 * BOOK → BEAMER SLIDES WORKFLOW
 * Book: An Introduction to Universal Artificial Intelligence (Hutter, Quarel, Catt; 2024)
 * Generated from book_slides_workflow_TEMPLATE.js
 */

export const meta = {
  name: 'universal-ai-slides',
  description: 'Generate self-contained LaTeX Beamer slides for "An Introduction to Universal Artificial Intelligence"',
  phases: [
    { title: 'Render Pages',    detail: 'Convert book PDF pages to PNG images' },
    { title: 'Generate Slides', detail: 'Parallel Beamer slide generation per chapter' },
  ],
}

const BOOK_PDF = '/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/universal_artificial_intelligence_2024/An Introduction to Universal Artificial Intelligence 2024.pdf'
const REPO     = '/Users/vinhdq1/work/Lectures-on-Monte-Carlo-Theory/universal_artificial_intelligence_2024'

const CHAPTERS = [
  { num: 1, dir: 'chapter01_introduction', title: 'Introduction', start: 24, end: 37,
    toc: `Overview of the book's six parts:
I. Background
II. Algorithmic Prediction
III. A Family of Universal Agents
IV. Approximating Universal Agents
V. Alternative Approaches
VI. Safety and Discussion` },

  { num: 2, dir: 'chapter02a_binary_strings', title: 'Background: Binary Strings', start: 38, end: 44,
    toc: `2.1 Binary Strings (chapter opening overview of Ch.2 sections also on p.38-40)` },
  { num: 2, dir: 'chapter02b_measure_probability_theory', title: 'Background: Measure Theory and Probability Theory', start: 45, end: 73,
    toc: `2.2 Measure Theory and Probability Theory` },
  { num: 2, dir: 'chapter02c_statistical_inference', title: 'Background: Statistical Inference and Estimation', start: 74, end: 79,
    toc: `2.3 Statistical Inference and Estimation` },
  { num: 2, dir: 'chapter02d_bayesian_probability', title: 'Background: Bayesian Probability Theory', start: 80, end: 88,
    toc: `2.4 Bayesian Probability Theory` },
  { num: 2, dir: 'chapter02e_information_theory_coding', title: 'Background: Information Theory and Coding', start: 89, end: 101,
    toc: `2.5 Information Theory and Coding
  2.5.1 Shannon Entropy
  2.5.2 Shannon-Fano Code
  2.5.3 Kullback-Leibler Divergence
  2.5.4 The Kraft Inequality` },
  { num: 2, dir: 'chapter02f_computability_theory', title: 'Background: Computability Theory', start: 102, end: 111,
    toc: `2.6 Computability Theory` },
  { num: 2, dir: 'chapter02g_kolmogorov_complexity', title: 'Background: Kolmogorov Complexity', start: 112, end: 128,
    toc: `2.7 Kolmogorov Complexity` },
  { num: 2, dir: 'chapter02h_misc_history_references', title: 'Background: Miscellaneous, History and References', start: 129, end: 143,
    toc: `2.8 Miscellaneous
2.9 History and References` },

  { num: 3, dir: 'chapter03_bayesian_sequence_prediction', title: 'Bayesian Sequence Prediction', start: 146, end: 177,
    toc: `3.1 Bayes Mixture ξ
3.2 Generalized Solomonoff Bound
3.3 Predictive Convergence
3.4 Model Misspecification
3.5 Bounds on Prediction Loss
3.6 Pareto-Optimality of ξ
3.7 Choices of Class M and Prior wν
3.8 Solomonoff Distribution MU
3.9 Martingales
3.10 Exercises
3.11 History and References` },

  { num: 4, dir: 'chapter04_context_tree_weighting_algorithm', title: 'The Context Tree Weighting Algorithm', start: 178, end: 219,
    toc: `4.1 Krichevsky-Trofimov (KT) Estimator
4.2 Context
4.3 Variable Length Context
4.4 Mixing Distributions
4.5 Context Tree Weighting
4.6 Exercises
4.7 History and References` },

  { num: 5, dir: 'chapter05_variations_on_ctw', title: 'Variations on CTW', start: 220, end: 233,
    toc: `5.1 Adaptive CTW
5.2 Context Tree Switching
5.3 Partition Tree Weighting
5.4 Forget-Me-Not Process
5.5 Context Tree Maximization
5.6 Exercises
5.7 History and References` },

  { num: 6, dir: 'chapter06_agency', title: 'Agency', start: 236, end: 255,
    toc: `6.1 Policy and Environment
6.2 Assigning Rewards
6.3 (PO)MDP vs. History RL
6.4 Time Discounting
6.5 Time Consistency
6.6 Value Functions
6.7 Q-Value
6.8 Exercises
6.9 History and References` },

  { num: 7, dir: 'chapter07_universal_artificial_intelligence', title: 'Universal Artificial Intelligence', start: 256, end: 271,
    toc: `7.1 Acting Optimally in Known Environments
7.2 Bayesian Mixture of Environments
7.3 Acting Optimally in Unknown Environments
7.4 Universal Optimal Agent AIXI
7.5 Exercises
7.6 History and References` },

  { num: 8, dir: 'chapter08_optimality_of_universal_agents', title: 'Optimality of Universal Agents', start: 272, end: 287,
    toc: `8.1 Definitions of Optimality
8.2 Bad Priors
8.3 Problems with Optimality Criteria
8.4 Exercises
8.5 History and References` },

  { num: 9, dir: 'chapter09_other_universal_agents', title: 'Other Universal Agents', start: 288, end: 305,
    toc: `9.1 Optimistic Agents
9.2 (Thompson) Sampling Agents
9.3 Knowledge-Seeking Agents
9.4 Exploring Agents (BayesExp and Inq)
9.5 Planning-Avoiding Agents (Self-AIXI)
9.6 Exercises
9.7 History and References` },

  { num: 10, dir: 'chapter10_multi_agent_setting', title: 'Multi-Agent Setting', start: 306, end: 325,
    toc: `10.1 From Preferences to Utilities
10.2 Game Theory
10.3 Multi-Agent Extensive-Form Games
10.4 Strategic Games vs Reinforcement Learning
10.5 Reflective Oracles
10.6 The Grain of Truth
10.7 Reflective AIXI
10.8 Exercises
10.9 History and References` },

  { num: 11, dir: 'chapter11_aixi_mdp', title: 'AIXI-MDP', start: 328, end: 335,
    toc: `11.1 AIXI-MDP Setup
11.2 Definition of AIXI-MDP
11.3 Experimental Results
11.4 Exercises
11.5 History and References` },

  { num: 12, dir: 'chapter12_monte_carlo_aixi_ctw', title: 'Monte Carlo AIXI with Context Tree Weighting', start: 336, end: 371,
    toc: `12.1 Learning and Searching
12.2 Searching via Monte Carlo Tree Search
12.3 Learning via Context Tree Weighting
12.4 All Together
12.5 Experiments
12.6 AIXIjs Implementation
12.7 Discussion
12.8 Exercises
12.9 History and References` },

  { num: 13, dir: 'chapter13_computational_aspects', title: 'Computational Aspects', start: 372, end: 381,
    toc: `13.1 Computability of AIXI
13.2 Time- and Space-Bounded AIXI
13.3 Exercises
13.4 History and References` },

  { num: 14, dir: 'chapter14_feature_reinforcement_learning', title: 'Feature Reinforcement Learning', start: 384, end: 403,
    toc: `14.1 Feature Reinforcement Learning Setup
14.2 History Aggregation beyond MDPs
14.3 Feature MDP
14.4 Context Tree Maximization Reinforcement Learning
14.5 Exercises
14.6 History and References` },

  { num: 15, dir: 'chapter15_asi_safety', title: 'ASI Safety', start: 406, end: 433,
    toc: `15.1 The Technological Singularity
15.2 Safety Subtopics
15.3 The Control Problem
15.4 Instrumental Convergence
15.5 Orthogonality Thesis
15.6 Value - Reward - Utility
15.7 Death and Suicide of Agents
15.8 Self-Modification
15.9 Wireheading
15.10 Delusion Boxes, Survival, and Exploration
15.11 Corrupted Reward Channel
15.12 Embedded Intelligence
15.13 Exercises
15.14 History and References` },

  { num: 16, dir: 'chapter16_philosophy_of_ai', title: 'Philosophy of AI', start: 434, end: 461,
    toc: `16.1 Philosophy of Universal Induction
16.2 Consciousness, Free Will, and Other Qualia
16.3 Moral Considerations
16.4 Teleporting and Copying AGI
16.5 Arguments against AGI
16.6 Arguments for AGI
16.7 Intelligence
16.8 Deep Learning
16.9 Conclusion` },
]

const CONVENTIONS = `
LATEX / BEAMER CONVENTIONS:
- \\documentclass[aspectratio=169,10pt]{beamer}
- \\usetheme{Madrid}  \\usecolortheme{seahorse}
- File name: chapter{N}_slides.tex  (NOT slides.tex) — use the numeric part of the chapter dir, e.g. chapter02b_slides.tex
- \\graphicspath{{figures/}}  — all figures live in figures/ subdir
- Text-heavy frames: \\begin{frame}[allowframebreaks]{Title}
- Code frames:       \\begin{frame}[fragile]{Title}
  (NEVER combine allowframebreaks with fragile on the same frame)
- listings style: basicstyle=\\ttfamily\\scriptsize, numbers=left,
  frame=single, backgroundcolor=\\color{gray!7}
- Python comments: commentstyle=\\color{green!50!black}\\itshape
- Use \\mathbb{P}, \\mathbb{E}, \\mathbb{R}, \\mathbb{N}, \\mathbb{B} for prob/expectation/reals/naturals/bit-strings
- Compile: pdflatex -interaction=nonstopmode (run TWICE)
- \\setbeamertemplate{footline}[frame number]
- NO "see Figure X.Y in the book" — every figure must be embedded

READER-FRIENDLINESS (mandatory — readers are NOT mathematicians):
- First time a Greek letter appears on a slide, spell out its pronunciation in parentheses:
  ξ (xi), ν (nu), μ (mu), θ (theta), λ (lambda), α (alpha), γ (gamma), etc.
- After every displayed formula, add a plain-English line explaining every symbol
  (what it represents conceptually, not just its type).
- Explain subscripts/superscripts and any new notation (e.g. AIXI, CTW, KT estimator,
  Bayes mixture, Kolmogorov complexity) in plain language the first time it appears.
- Write explanatory text in complete sentences, not fragment bullets.

FIGURES:
- Write gen_figures.py; save every plot as a PDF in figures/
  (import matplotlib; matplotlib.use('Agg'))
- Run:  conda run -n py313 python3 gen_figures.py
- For book diagrams hard to reproduce: crop from the PDF with pymupdf:
    import fitz, PIL.Image, io
    doc  = fitz.open(BOOK_PDF)
    pix  = doc[PAGE_INDEX].get_pixmap(matrix=fitz.Matrix(3,3))
    img  = PIL.Image.open(io.BytesIO(pix.tobytes('png')))
    img.crop((left,top,right,bottom)).save('figures/name.png')
- Include: \\includegraphics[width=...]{figname}  (no extension for PDF)

PYTHON env: use "conda run -n py313" for all Python/pip commands.

GIT: agents must NOT run git — the main process handles commits.
`

phase('Render Pages')

const rangeLines = CHAPTERS.map((ch, i) =>
  `  ${i}: (${ch.start - 1}, ${ch.end - 1}),  # ${ch.dir} pages ${ch.start}-${ch.end}`
).join('\n')

await agent(`Render PDF pages for all chapter folders to /tmp/uai_book_pages/.
Run with: conda run -n py313 python3 -c "<script>"

Script to run:
import fitz, os
doc = fitz.open('${BOOK_PDF}')
ranges = {
${rangeLines}
}
for idx, (lo, hi) in ranges.items():
    os.makedirs(f'/tmp/uai_book_pages/ch{idx}', exist_ok=True)
    for pg in range(lo, hi+1):
        pix = doc[pg].get_pixmap(matrix=fitz.Matrix(1.6,1.6))
        pix.save(f'/tmp/uai_book_pages/ch{idx}/p{pg+1:03d}.png')
    print(f'{idx}: rendered {hi-lo+1} pages')
print('Done')

Report "OK" when finished.`, { label: 'render-pages', phase: 'Render Pages' })

phase('Generate Slides')

function buildChapterPrompt(ch, idx) {
  const texBase = ch.dir.replace(/^chapter0*/, 'chapter')
  return `Create self-contained LaTeX Beamer slides for: "${ch.title}" (book Chapter ${ch.num}).

REPOSITORY    : ${REPO}
CHAPTER DIR   : ${REPO}/${ch.dir}/   (create if it doesn't exist)
PAGE IMAGES   : /tmp/uai_book_pages/ch${idx}/  (p0XX.png, 1-indexed, matching book PDF page numbers)
BOOK PDF      : ${BOOK_PDF}
TEX FILE NAME : ${ch.dir}_slides.tex

SECTION(S) COVERED IN THIS DECK:
${ch.toc}

══════════════════════════════════════════
STEP 1 — Read every page image
══════════════════════════════════════════
Use the Read tool on EACH image in /tmp/uai_book_pages/ch${idx}/.
Process in batches of 8-10. Read ALL pages before writing any slides.
Note: definitions, theorems, key formulas (copy exactly, including all sub/superscripts),
figures present, pseudocode/algorithms, worked examples, section boundaries.
This book is dense and notation-heavy (measure theory, Kolmogorov complexity, AIXI) —
be extremely careful to transcribe formulas exactly as printed.

══════════════════════════════════════════
STEP 2 — Create figures/gen_figures.py
══════════════════════════════════════════
Write ${REPO}/${ch.dir}/gen_figures.py.
mkdir -p ${REPO}/${ch.dir}/figures
Every figure/diagram needed by the slides must be generated here, or cropped from the
book PDF with pymupdf if it's a diagram (not a plottable function).
Run with: conda run -n py313 python3 ${REPO}/${ch.dir}/gen_figures.py

══════════════════════════════════════════
STEP 3 — Write ${ch.dir}_slides.tex
══════════════════════════════════════════
${CONVENTIONS}

CONTENT RULES:
• Cover EVERY (sub)section listed above in full — do not skip content for brevity.
• Every important formula/theorem/definition: put in \\begin{block}{Name}; explain
  every symbol below it in plain English (see READER-FRIENDLINESS above).
• Include a title slide (chapter/section title + book citation) and an outline slide.
• At least one \\begin{frame}[fragile]{Python: ...} worked example per major section
  where the concept is programmable (e.g. KT estimator, CTW, entropy/coding, AIXI-MDP
  simulation) — use small numeric examples with actual numbers, not just pseudocode.
• Include all figures/diagrams with \\includegraphics.
• Self-contained: reader does not need the book to follow the slides.

══════════════════════════════════════════
STEP 4 — Compile
══════════════════════════════════════════
cd ${REPO}/${ch.dir}
pdflatex -interaction=nonstopmode ${ch.dir}_slides.tex
pdflatex -interaction=nonstopmode ${ch.dir}_slides.tex
Fix all LaTeX errors until output compiles cleanly (check the .log for "!" errors).

DO NOT run any git commands.`
}

await parallel(CHAPTERS.map((ch, idx) => () =>
  agent(buildChapterPrompt(ch, idx), {
    label: `${ch.dir}`,
    phase: 'Generate Slides',
  })
))

log('All chapter agents finished. Commit and push with the main agent.')
return { status: 'done', chapters: CHAPTERS.map(c => c.dir) }
