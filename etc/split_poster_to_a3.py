#!/usr/bin/env python3
"""
split_poster_to_a3.py

Split a high‑resolution poster image into four quadrants and export them
as a single 4‑page PDF, where each page is exactly A3 size (portrait).
Print each page on A3 paper and assemble to recreate the full poster.

Usage::

    python split_poster_to_a3.py eye_poster.png eye_target_A3.pdf

Dependencies:
    pip install pillow reportlab

**Fix 2025‑05‑06**: Avoid ReportLab’s filename‑based caching, which was
causing all four pages to show the same quadrant. We now feed each
quadrant to `ImageReader` directly (in‑memory), guaranteeing unique
content on every page.
"""

import sys
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import A3
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def split_into_quadrants(img: Image.Image):
    """Return the four image quadrants in TL, TR, BL, BR order."""
    w, h = img.size
    mid_x, mid_y = w // 2, h // 2
    return [
        img.crop((0, 0, mid_x, mid_y)),        # top‑left
        img.crop((mid_x, 0, w, mid_y)),        # top‑right
        img.crop((0, mid_y, mid_x, h)),        # bottom‑left
        img.crop((mid_x, mid_y, w, h)),        # bottom‑right
    ]


def save_quadrants_to_a3_pdf(quadrants, out_pdf: Path):
    """Save the quadrants as a 4‑page A3 PDF."""
    page_w, page_h = A3  # points (1 pt = 1/72 in)
    c = canvas.Canvas(str(out_pdf), pagesize=A3)

    for quadrant in quadrants:
        # Ensure RGB mode (ImageReader needs this for JPEG compression)
        if quadrant.mode != "RGB":
            quadrant = quadrant.convert("RGB")

        img_reader = ImageReader(quadrant)  # <— direct, no temp files!

        img_w, img_h = quadrant.size
        # Scale to fit the full A3 page while preserving aspect ratio
        scale = min(page_w / img_w, page_h / img_h)
        draw_w, draw_h = img_w * scale, img_h * scale
        # Center on page
        x = (page_w - draw_w) / 2
        y = (page_h - draw_h) / 2
        c.drawImage(img_reader, x, y, draw_w, draw_h)
        c.showPage()

    c.save()


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python split_poster_to_a3.py <input_image> <output_pdf>\n")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.exists():
        sys.stderr.write(f"Input image not found: {src}\n")
        sys.exit(1)

    img = Image.open(src)
    quadrants = split_into_quadrants(img)
    save_quadrants_to_a3_pdf(quadrants, dst)
    print(f"Created {dst} with 4 A3 pages. Ready to print!")


if __name__ == "__main__":
    main()
