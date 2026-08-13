"""
Crop Figure 11.2 (five experimental-result panels) from the book PDF using
PyMuPDF. Figure 11.1 (agent/opponent/environment loop) is instead redrawn
natively in gen_figures.py since it is simple enough to reproduce exactly.

Run with:
    conda run -n py313 python3 crop_figs.py
"""
import fitz
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_PDF = os.path.join(os.path.dirname(HERE),
                         "An Introduction to Universal Artificial Intelligence 2024.pdf")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

doc = fitz.open(BOOK_PDF)
PAGE_INDEX = 332  # printed page 312, contains Figure 11.2 (a)-(e)
page = doc[PAGE_INDEX]

ZOOM = 5
mat = fitz.Matrix(ZOOM, ZOOM)

crops = {
    # name: (x0, y0, x1, y1) in PDF points (page is 504 x 720)
    "fig11_2a_pd":      (62, 244, 264, 368),
    "fig11_2b_staghunt": (263, 244, 466, 368),
    "fig11_2c_chicken":  (62, 372, 264, 491),
    "fig11_2d_bos":      (263, 372, 466, 491),
    "fig11_2e_mpennies":(150, 492, 375, 611),
    # full combined figure (all 5 panels) for an overview slide
    "fig11_2_all":      (62, 244, 466, 640),
}

for name, (x0, y0, x1, y1) in crops.items():
    clip = fitz.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    outpath = os.path.join(OUTDIR, f"{name}.png")
    pix.save(outpath)
    print("saved", outpath, pix.width, pix.height)
