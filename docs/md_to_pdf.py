"""Basit Markdown -> PDF donusturucu (teslim dokumanlari icin).

Kullanim: python docs/md_to_pdf.py docs/mathematical_model.md docs/technical_report.md
Basliklar, madde isaretleri, kod bloklari ve basit tablolari destekler.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"

# Fontlarda bulunmayan matematik sembollerini ASCII karsiliklarina indirger.
SANITIZE = {
    "∀": "for all ",
    "∈": " in ",
    "⁰": "^0",
    "⁺": "^+",
    "−": "-",
    "Σ": "SUM ",
    "≤": "<=",
    "≥": ">=",
    "⇔": "<=>",
    "⇒": "=>",
    "·": "*",
    "×": "x",
    "π": "pi",
    "ρ": "rho",
    "α": "alpha",
    "Γ": "Gamma",
    "ε": "eps",
    "τ": "tau",
    "⊆": " subset of ",
    "≠": "!=",
    "≈": "~",
    "→": "->",
    "±": "+/-",
    "≡": "==",
    "•": "-",
}


def sanitize(text: str) -> str:
    for src, dst in SANITIZE.items():
        text = text.replace(src, dst)
    return text


class DocPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.add_font("Body", "", FONT)
        self.add_font("Body", "B", FONT_BOLD)
        self.add_font("Mono", "", FONT_MONO)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 16, 18)

    def footer(self):
        self.set_y(-12)
        self.set_font("Body", "", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Sayfa {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0)


def render_inline(pdf: DocPDF, text: str, size: float, bold: bool = False):
    """Render a paragraph with **bold** and `code` inline markers."""

    pdf.set_font("Body", "B" if bold else "", size)
    parts = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Body", "B", size)
            pdf.write(5.2, part[2:-2])
            pdf.set_font("Body", "B" if bold else "", size)
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font("Mono", "", size - 0.5)
            pdf.write(5.2, part[1:-1])
            pdf.set_font("Body", "B" if bold else "", size)
        elif part:
            pdf.write(5.2, part)
    pdf.ln(6.4)


def render_table(pdf: DocPDF, rows: list[list[str]]):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    width = (pdf.w - pdf.l_margin - pdf.r_margin) / ncol
    pdf.set_font("Body", "", 8.2)
    for i, row in enumerate(rows):
        style = "B" if i == 0 else ""
        pdf.set_font("Body", style, 8.2)
        cells = [re.sub(r"\*\*(.+?)\*\*|`([^`]+)`", lambda m: m.group(1) or m.group(2), c) for c in row]
        cells += [""] * (ncol - len(cells))
        line_height = 4.6
        max_lines = max(len(pdf.multi_cell(width, line_height, c, dry_run=True, output="LINES")) for c in cells)
        row_height = line_height * max_lines
        if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
            pdf.add_page()
        x0, y0 = pdf.get_x(), pdf.get_y()
        for j, cell in enumerate(cells):
            pdf.set_xy(x0 + j * width, y0)
            pdf.multi_cell(width, line_height, cell, border=1)
        pdf.set_xy(x0, y0 + row_height)
    pdf.ln(2)


def convert(md_path: Path) -> Path:
    lines = [sanitize(line) for line in md_path.read_text(encoding="utf-8").splitlines()]
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            render_table(pdf, table_buf)
            table_buf = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                pdf.set_font("Mono", "", 8)
                pdf.set_fill_color(245, 245, 245)
                for code_line in code_buf:
                    pdf.multi_cell(0, 4.4, code_line if code_line else " ", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.set_fill_color(255, 255, 255)
                pdf.ln(2)
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            table_buf.append(cells)
            continue
        flush_table()
        if not stripped:
            pdf.ln(1.6)
        elif stripped.startswith("# "):
            pdf.set_font("Body", "B", 15)
            pdf.multi_cell(0, 7.5, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.ln(1.5)
            pdf.set_font("Body", "B", 12)
            pdf.multi_cell(0, 6.5, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.8)
        elif stripped.startswith("### "):
            pdf.ln(1)
            pdf.set_font("Body", "B", 10.5)
            pdf.multi_cell(0, 6, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)
        elif re.match(r"^\d+\.\s", stripped):
            num, rest = stripped.split(" ", 1)
            pdf.set_x(pdf.l_margin + 4)
            render_inline(pdf, f"{num} {rest}", 9.2)
        elif stripped.startswith("- "):
            pdf.set_x(pdf.l_margin + 4)
            render_inline(pdf, "• " + stripped[2:], 9.2)
        else:
            render_inline(pdf, stripped, 9.2)
    flush_table()
    out = md_path.with_suffix(".pdf")
    pdf.output(str(out))
    print(f"[OK] {out} ({pdf.page_no()} sayfa)")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        convert(Path(arg))
