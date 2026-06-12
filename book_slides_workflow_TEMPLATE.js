/**
 * BOOK → BEAMER SLIDES WORKFLOW TEMPLATE
 * ========================================
 * Copy this file to your project, fill in the three sections marked
 * ★ CONFIGURE, then run it in Claude Code:
 *
 *   Workflow({ scriptPath: "book_slides_workflow_TEMPLATE.js" })
 *
 * Or paste the filled-in script directly into a Workflow() call.
 *
 * What it produces per chapter:
 *   chapterN_dir/
 *     chapter{N}_slides.tex   — LaTeX Beamer source
 *     chapter{N}_slides.pdf   — compiled PDF
 *     figures/                — Python-generated + cropped book figures
 *     gen_figures.py          — reproduces every figure from scratch
 */

export const meta = {
  name: 'book-slides',
  description: 'Generate self-contained LaTeX Beamer slides from a book PDF',
  phases: [
    { title: 'Render Pages',    detail: 'Convert book PDF pages to PNG images' },
    { title: 'Generate Slides', detail: 'Parallel Beamer slide generation per chapter' },
  ],
}

// ══════════════════════════════════════════════════════════════════
// ★ CONFIGURE — edit these three sections only
// ══════════════════════════════════════════════════════════════════

/** Absolute path to the book PDF */
const BOOK_PDF = '/path/to/your/book.pdf'

/** Absolute path to the git repository root */
const REPO = '/path/to/your/repo'

/**
 * One entry per chapter.  Fields:
 *   num   — chapter number (integer)
 *   dir   — subdirectory name inside REPO (will be created)
 *   title — full chapter title
 *   start — first page of the chapter, 1-indexed (match PDF page numbers)
 *   end   — last page of the chapter, 1-indexed
 *   toc   — full table of contents for this chapter (plain text, indented)
 *
 * How to get page numbers: open the PDF, navigate to each chapter start,
 * and read the page number shown in your PDF viewer.
 */
const CHAPTERS = [
  {
    num: 1,
    dir: 'chapter1_intro',
    title: 'Introduction',
    start: 1,   // ← first PDF page of this chapter
    end:   30,  // ← last  PDF page of this chapter
    toc: `
1.1 Section title
  1.1.1 Subsection
1.2 Another section
  1.2.1 Subsection
    `,
  },
  // ── add more chapters here ──────────────────────────────────────
  // {
  //   num: 2, dir: 'chapter2_foo', title: 'Foo', start: 31, end: 80,
  //   toc: `2.1 ...`,
  // },
]

// ══════════════════════════════════════════════════════════════════
// LaTeX + slide conventions — edit only if you want different styling
// ══════════════════════════════════════════════════════════════════
const CONVENTIONS = `
LATEX / BEAMER CONVENTIONS:
- \\documentclass[aspectratio=169,10pt]{beamer}
- \\usetheme{Madrid}  \\usecolortheme{seahorse}
- File name: chapter{N}_slides.tex  (NOT slides.tex)
- \\graphicspath{{figures/}}  — all figures live in figures/ subdir
- Text-heavy frames: \\begin{frame}[allowframebreaks]{Title}
- Code frames:       \\begin{frame}[fragile]{Title}
  (NEVER combine allowframebreaks with fragile on the same frame)
- listings style: basicstyle=\\ttfamily\\scriptsize, numbers=left,
  frame=single, backgroundcolor=\\color{gray!7}
- Python comments: commentstyle=\\color{green!50!black}\\itshape
- Use \\mathbb{P}, \\mathbb{E}, \\mathbb{R} for prob/expectation/reals
- Compile: pdflatex -interaction=nonstopmode (run TWICE)
- \\setbeamertemplate{footline}[frame number]
- NO "see Figure X.Y in the book" — every figure must be embedded

FIGURES:
- Write gen_figures.py; save every plot as a PDF in figures/
  (import matplotlib; matplotlib.use('Agg'))
- Run:  conda run -n py313 python3 gen_figures.py
- For book diagrams hard to reproduce: crop from PDF with pymupdf:
    import fitz, PIL.Image, io
    doc  = fitz.open(BOOK_PDF)
    pix  = doc[PAGE_INDEX].get_pixmap(matrix=fitz.Matrix(3,3))
    img  = PIL.Image.open(io.BytesIO(pix.tobytes('png')))
    img.crop((left,top,right,bottom)).save('figures/name.png')
- Include: \\includegraphics[width=...]{figname}  (no extension for PDF)

PYTHON env: use "conda run -n py313" for all Python/pip commands.

GIT: agents must NOT run git — the main process handles commits.
`

// ══════════════════════════════════════════════════════════════════
// Workflow logic — no need to edit below this line
// ══════════════════════════════════════════════════════════════════

// ── Phase 1: render all pages to /tmp/book_slides_pages/chN/ ───────
phase('Render Pages')

const rangeLines = CHAPTERS.map(ch =>
  `  ${ch.num}: (${ch.start - 1}, ${ch.end - 1}),  # pages ${ch.start}-${ch.end}`
).join('\n')

await agent(`Render PDF pages for all chapters to /tmp/book_slides_pages/.
Use: conda run -n py313 python3 -c "..."

Script to run:
import fitz, os
doc = fitz.open('${BOOK_PDF}')
ranges = {
${rangeLines}
}
for ch, (lo, hi) in ranges.items():
    os.makedirs(f'/tmp/book_slides_pages/ch{ch}', exist_ok=True)
    for pg in range(lo, hi+1):
        pix = doc[pg].get_pixmap(matrix=fitz.Matrix(1.6,1.6))
        pix.save(f'/tmp/book_slides_pages/ch{ch}/p{pg+1:03d}.png')
    print(f'Chapter {ch}: rendered {hi-lo+1} pages')
print('Done')

Report "OK" when finished.`, { label: 'render-pages', phase: 'Render Pages' })

// ── Phase 2: parallel chapter agents ───────────────────────────────
phase('Generate Slides')

function buildChapterPrompt(ch) {
  return `Create self-contained LaTeX Beamer slides for Chapter ${ch.num}: "${ch.title}".

REPOSITORY    : ${REPO}
CHAPTER DIR   : ${REPO}/${ch.dir}/   (create if it doesn't exist)
PAGE IMAGES   : /tmp/book_slides_pages/ch${ch.num}/  (p0XX.png, 1-indexed)
BOOK PDF      : ${BOOK_PDF}

CHAPTER TABLE OF CONTENTS:
${ch.toc}

══════════════════════════════════════════
STEP 1 — Read every page image
══════════════════════════════════════════
Use the Read tool on EACH image in /tmp/book_slides_pages/ch${ch.num}/.
Process in batches of 8-10. Read ALL pages before writing any slides.
Note: definitions, theorems, key formulas (copy exactly), figures present,
Python listings, section boundaries.

══════════════════════════════════════════
STEP 2 — Create figures/gen_figures.py
══════════════════════════════════════════
Write ${REPO}/${ch.dir}/gen_figures.py.
mkdir -p ${REPO}/${ch.dir}/figures
Every figure needed by the slides must be generated here.
Run with: conda run -n py313 python3 ${REPO}/${ch.dir}/gen_figures.py
For book diagrams: crop from the PDF with pymupdf (see conventions).

══════════════════════════════════════════
STEP 3 — Write chapter${ch.num}_slides.tex
══════════════════════════════════════════
${CONVENTIONS}

CONTENT RULES:
• Cover EVERY section and subsection listed in the TOC.
• Every important formula: put in \\begin{block}{Name}; explain each symbol below.
• At least one \\begin{frame}[fragile]{Python: ...} per major section.
• Running numerical examples with actual numbers.
• Include all figures with \\includegraphics.
• Self-contained: reader does not need the book to follow the slides.

══════════════════════════════════════════
STEP 4 — Compile
══════════════════════════════════════════
cd ${REPO}/${ch.dir}
pdflatex -interaction=nonstopmode chapter${ch.num}_slides.tex
pdflatex -interaction=nonstopmode chapter${ch.num}_slides.tex
Fix all LaTeX errors until output is clean.

DO NOT run any git commands.`
}

await parallel(CHAPTERS.map(ch => () =>
  agent(buildChapterPrompt(ch), {
    label: `ch${ch.num}-slides`,
    phase: 'Generate Slides',
  })
))

log('All chapter agents finished. Commit and push with the main agent.')
return { status: 'done', chapters: CHAPTERS.map(c => c.dir) }
