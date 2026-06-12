# How to Generate Beamer Slides from a Book PDF

This repo contains a reusable workflow for turning any textbook PDF into
self-contained LaTeX Beamer slides, one folder per chapter.

---

## Quick start (new book)

### 1 — Copy the template

```bash
cp book_slides_workflow_TEMPLATE.js /path/to/new-book-repo/workflow.js
```

### 2 — Fill in three fields

Open `workflow.js` and edit the three `★ CONFIGURE` sections:

```js
const BOOK_PDF = '/absolute/path/to/book.pdf'
const REPO     = '/absolute/path/to/repo'

const CHAPTERS = [
  { num: 1, dir: 'chapter1_intro', title: 'Introduction',
    start: 17, end: 36,
    toc: `1.1 About the Book\n1.2 Examples\n  1.2.1 ...` },
  { num: 2, dir: 'chapter2_foo',  title: 'Foo', start: 37, end: 113, toc: `...` },
  // one entry per chapter
]
```

**How to get page numbers:** open the PDF in any viewer and read the page
numbers at the start and end of each chapter.  Use the viewer's page number
(1-indexed), not the typeset number in the book.

**How to get the TOC:** paste the table of contents from the PDF (or type
it by hand). The more detail the better — agents use it to know what to
cover.

### 3 — Run in Claude Code

Paste this into the Claude Code prompt:

```
Run the workflow in workflow.js to generate slides for all chapters.
After it finishes, commit and push everything.
```

Claude Code will call:

```js
Workflow({ scriptPath: "workflow.js" })
```

It spins up **one agent per chapter in parallel**, so all chapters are
processed simultaneously.  Wall-clock time ≈ time for the longest single
chapter, not the sum.

---

## What gets produced

```
chapter1_intro/
  chapter1_slides.tex    ← LaTeX Beamer source
  chapter1_slides.pdf    ← compiled PDF (pdflatex)
  figures/
    fig_something.pdf    ← Python-generated figures
    fig_crop.png         ← figures cropped from the book PDF
  gen_figures.py         ← reproduces every figure from scratch
chapter2_foo/
  ...
```

---

## Re-running a single chapter

If one chapter failed or you want to improve it, tell Claude Code:

```
Re-run chapter 3 slides only, using the same conventions as the other chapters.
The chapter covers pages 114–201 of the PDF. TOC: [paste TOC here].
```

Or edit `workflow.js` to keep only that one chapter in the `CHAPTERS` array
and re-run the workflow.

---

## Conventions baked in

| Item | Value |
|------|-------|
| Beamer theme | Madrid / seahorse |
| Aspect ratio | 16:9 |
| Font size | 10pt |
| Long frames | `[allowframebreaks]` |
| Code frames | `[fragile]` (never mixed with `allowframebreaks`) |
| Figure format | PDF (vector) preferred; PNG for cropped images |
| Python env | `conda run -n py313` |
| File naming | `chapter{N}_slides.tex` |
| Figures path | `figures/` inside each chapter folder |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| LaTeX won't compile | Check `slides.log`; common issues: `#` in code needs `[fragile]`, missing `\end{frame}` |
| Figure not found | Run `gen_figures.py` manually; check `\graphicspath{{figures/}}` is present |
| Agent timed out on large chapter | Shrink the chapter range and run two agents |
| `pymupdf` not installed | `conda run -n py313 pip install pymupdf` |
| Wrong page range | Open PDF in viewer; use the viewer page counter, not the typeset page number |
