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
- clause numbering: a heading numbered "4.2 ..." or "7A. ..." under one numbered "4. ..." / "7. ..." is a
  subsection, which gives back the one level of structure the renderer flattens.

The output is markdown text; chunker.chunk_document() does the rest. tests/test_pdf_chunker.py proves
that no clause, table row or figure of the markdown copy is lost on the PDF path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

TITLE_SIZE = 15.0  # smallest bold font size, in points, of the title block
HEADING_SIZE = 12.0  # smallest bold font size, in points, of a section heading
# A line that starts its own line in the source: a table row "|", a list item "- ", a lettered "(a) " / "iv) "
# or numbered "(1) " / "2) " clause, or a numbered sub-clause "4.2 " / "4.2.1 ".
_OWN_LINE = re.compile(r"^(\||- |\(?[a-z]{1,3}\)\s|\(?\d{1,2}\)\s|\d+(\.\d+)+\s)")
# A heading's number: group 1 the section ("4."), group 2 a numeric subsection ("4.2"), group 3 a lettered one ("7A.").
_NUMBERED = re.compile(r"^(\d+)(?:\.(\d+)|([A-Z])\.)?")
# First metadata label inside a title block ("Document ref: ...", "Document ID: ... | Version: ...").
_METADATA_LABEL = re.compile(r"\s(?=(?:Document (?:ID|ref(?:erence)?)|Owner|Version|Effective(?: date)?|Applies to)\s*:)")


@dataclass
class _Line:
    """One rendered line of text and the layout facts the loader reads structure from."""

    page: int  # page index, from 0
    vertical: float  # vertical position in PDF points; it decreases down the page
    bold: bool
    size: float  # effective font size in points
    text: str


def _read_lines(path: Path) -> list[_Line]:
    """Every non-blank rendered line of the PDF, in reading order."""
    from pypdf import PdfReader  # imported here: only the PDF path needs it

    rendered: list[_Line] = []
    for page_index, page in enumerate(PdfReader(str(path)).pages):

        # pypdf's text visitor. In the text matrix, item 5 is the line's vertical position and item 0 the
        # horizontal scale, which multiplies the nominal font size into the size actually drawn.
        def visit(text, transform_matrix, text_matrix, font, font_size, page_index=page_index):
            if not text.strip():
                return
            font_name = (font or {}).get("/BaseFont", "") or ""
            vertical, size = round(text_matrix[5], 1), round(font_size * text_matrix[0], 1)
            for part in text.split("\n"):  # pypdf can hand over two rendered lines at once
                if part.strip():
                    rendered.append(_Line(page_index, vertical, "Bold" in font_name, size, part.strip()))

        page.extract_text(visitor_text=visit)
    return rendered


class _LineKind(StrEnum):
    """What a line of the PDF is, read from its font. Used inside this module only."""

    TITLE = "title"
    HEADING = "heading"
    BODY = "body"


def _classify_line(line: _Line) -> _LineKind:
    """What a line is, read from its font."""
    if line.bold and line.size >= TITLE_SIZE:
        return _LineKind.TITLE
    if line.bold and line.size >= HEADING_SIZE:
        return _LineKind.HEADING
    return _LineKind.BODY


def _group_paragraphs(lines: list[_Line]) -> list[tuple[_LineKind, list[_Line]]]:
    """Group lines into (kind, lines) paragraphs."""
    paragraphs: list[tuple[_LineKind, list[_Line]]] = []
    previous: _Line | None = None
    for line in lines:
        kind = _classify_line(line)
        same_block = (
            previous is not None
            and paragraphs
            and paragraphs[-1][0] == kind
            and line.page == previous.page
            and line.vertical < previous.vertical  # still going down inside the same paragraph
        )
        if same_block or (
            # A paragraph the renderer split over a page break: body text continues on the next page with
            # a line that cannot start a paragraph of its own.
            previous is not None and paragraphs and kind == _LineKind.BODY == paragraphs[-1][0]
            and line.page == previous.page + 1
            and not _OWN_LINE.match(line.text) and not line.text[:1].isupper()
            and not paragraphs[-1][1][-1].text.endswith((".", ":", "|"))
        ):
            paragraphs[-1][1].append(line)
        else:
            paragraphs.append((kind, [line]))
        previous = line
    return paragraphs


def _join_lines(lines: list[_Line]) -> str:
    """Join wrapped lines; table rows, list items and clauses keep their own line."""
    joined: list[str] = []
    for line in lines:
        if joined and not _OWN_LINE.match(line.text):
            joined[-1] = f"{joined[-1]} {line.text}"
        else:
            joined.append(line.text)
    return "\n".join(joined)


def convert_pdf_to_markdown(path: Path | str) -> str:
    """The PDF as markdown the chunker understands: "#" title lines, "###" sections, "####" subsections and
    body paragraphs, separated by blank lines."""
    markdown: list[str] = []
    top: str | None = None  # number of the current top-level section ("4" for "4. Fees")
    for kind, lines in _group_paragraphs(_read_lines(Path(path))):
        text = " ".join(line.text for line in lines) if kind != _LineKind.BODY else _join_lines(lines)
        if kind == _LineKind.TITLE:
            # The title block is the document name, sometimes a second title line (the renderer leaves a
            # stray "##" before it) and, in some handbooks, the document's metadata in the same bold font.
            # The title lines become level-1 headings (outside every section path, as in the markdown
            # copy); the metadata is body text, so the document-information chunk keeps it.
            title, metadata = _METADATA_LABEL.split(text, maxsplit=1) if _METADATA_LABEL.search(text) else (text, "")
            for part in title.split(" ## "):
                if part.strip():
                    markdown.append(f"# {part.strip()}")
            if metadata.strip():
                markdown.append(metadata.strip())
        elif kind == _LineKind.HEADING:
            match = _NUMBERED.match(text)
            subsection = match and (match.group(2) or match.group(3))
            if subsection and match.group(1) == top:
                markdown.append(f"#### {text}")  # "4.2 Fees" under "4. Fees", "7A. Cardiology" under "7. ..."
            else:
                top = match.group(1) if match and not subsection else None
                markdown.append(f"### {text}")
        else:
            markdown.append(text)
        markdown.append("")
    return "\n".join(markdown)
