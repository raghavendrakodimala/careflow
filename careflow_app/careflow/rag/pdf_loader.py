"""Read a corpus PDF back into markdown, so the PDF copy goes through the same chunker as the markdown copy.

The toolkit renders the PDFs lossily on purpose ("the PDF is what production looks like"): heading
markers are gone, every heading level looks the same, blank lines are gone and long lines wrap. What is
left, and what this loader uses:

- fonts: Helvetica-Bold 15pt on the first page is the title block; Helvetica-Bold 12pt is a section
  heading; Helvetica 10pt is body text;
- line positions: pypdf reports each rendered line with its offset inside its paragraph, and the offset
  starts again at every paragraph, so a line that is not below the previous one starts a new paragraph;
- line starts: table rows ("|"), list items ("- ") and lettered or numbered clauses keep their own line;
  any other line inside a paragraph is a wrapped continuation and is joined back with a space;
- clause numbering: a heading numbered "4.2 ..." or "7A. ..." under one numbered "4. ..." / "7. ..." is a subsection, which
  gives back the one level of structure the renderer flattens.

The output is markdown text; chunker.chunk_document() does the rest. tests/test_pdf_chunker.py proves
that no clause, table row or figure of the markdown copy is lost on the PDF path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TITLE_SIZE = 15.0
HEADING_SIZE = 12.0
_OWN_LINE = re.compile(r"^(\||- |\(?[a-z]{1,3}\)\s|\(?\d{1,2}\)\s|\d+(\.\d+)+\s)")
_NUMBERED = re.compile(r"^(\d+)(?:\.(\d+)|([A-Z])\.)?")  # "4.", "4.2", "7A."
# First metadata label inside a title block ("Document ref: ...", "Document ID: ... | Version: ...").
_META = re.compile(r"\s(?=(?:Document (?:ID|ref(?:erence)?)|Owner|Version|Effective(?: date)?|Applies to)\s*:)")


@dataclass
class _Line:
    page: int
    y: float
    bold: bool
    size: float
    text: str


def _lines(path: Path) -> list[_Line]:
    from pypdf import PdfReader  # imported here: only the PDF path needs it

    out: list[_Line] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages):

        def visit(text, cm, tm, font, font_size, page_no=page_no):
            if not text.strip():
                return
            name = (font or {}).get("/BaseFont", "") or ""
            for part in text.split("\n"):  # pypdf can hand over two rendered lines at once
                if part.strip():
                    out.append(_Line(page_no, round(tm[5], 1), "Bold" in name, round(font_size * tm[0], 1), part.strip()))

        page.extract_text(visitor_text=visit)
    return out


def _paragraphs(lines: list[_Line]) -> list[tuple[str, list[_Line]]]:
    """Group lines into (kind, lines) paragraphs; kind is "title", "heading" or "body"."""
    paras: list[tuple[str, list[_Line]]] = []
    prev: _Line | None = None
    for ln in lines:
        kind = "title" if ln.bold and ln.size >= TITLE_SIZE else "heading" if ln.bold and ln.size >= HEADING_SIZE else "body"
        same_block = (
            prev is not None
            and paras
            and paras[-1][0] == kind
            and ln.page == prev.page
            and ln.y < prev.y  # still going down inside the same paragraph
        )
        if same_block:
            paras[-1][1].append(ln)
        elif (
            # A paragraph the renderer split over a page break: body text continues on the next page with
            # a line that cannot start a paragraph of its own.
            prev is not None and paras and kind == "body" == paras[-1][0] and ln.page == prev.page + 1
            and not _OWN_LINE.match(ln.text) and not ln.text[:1].isupper()
            and not paras[-1][1][-1].text.endswith((".", ":", "|"))
        ):
            paras[-1][1].append(ln)
        else:
            paras.append((kind, [ln]))
        prev = ln
    return paras


def _join(lines: list[_Line]) -> str:
    """Join wrapped lines; table rows, list items and clauses keep their own line."""
    out: list[str] = []
    for ln in lines:
        if out and not _OWN_LINE.match(ln.text):
            out[-1] = f"{out[-1]} {ln.text}"
        else:
            out.append(ln.text)
    return "\n".join(out)


def pdf_to_markdown(path: Path | str) -> str:
    md: list[str] = []
    top: str | None = None  # number of the current top-level section ("4" for "4. Fees")
    for kind, lines in _paragraphs(_lines(Path(path))):
        text = " ".join(ln.text for ln in lines) if kind != "body" else _join(lines)
        if kind == "title":
            # The title block is the document name, sometimes a second title line (the renderer leaves a
            # stray "##" before it) and, in some handbooks, the document's metadata in the same bold font.
            # The title lines become level-1 headings (outside every section path, as in the markdown
            # copy); the metadata is body text, so the document-information chunk keeps it.
            title, meta = _META.split(text, maxsplit=1) if _META.search(text) else (text, "")
            for part in title.split(" ## "):
                if part.strip():
                    md.append(f"# {part.strip()}")
            if meta.strip():
                md.append(meta.strip())
        elif kind == "heading":
            m = _NUMBERED.match(text)
            sub = m and (m.group(2) or m.group(3))
            if sub and m.group(1) == top:
                md.append(f"#### {text}")  # "4.2 Fees" under "4. Fees", "7A. Cardiology" under "7. ..."
            else:
                top = m.group(1) if m and not sub else None
                md.append(f"### {text}")
        else:
            md.append(text)
        md.append("")
    return "\n".join(md)
