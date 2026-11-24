"""Structure-aware chunking of the policy corpus (markdown copy, or the PDF copy read back by pdf_loader).

Every heading starts a new section, and a chunk never crosses a section boundary. A long section is
split between paragraphs (numbered clauses are paragraphs), never inside one, so a table or a clause
stays whole. Each chunk's text starts with a context header naming the document and its section path:

    [Cardiology Service Line Handbook | MHP-SLH-CARD | 4. Consultation length]

That header is what separates the eight near-duplicate specialty handbooks: their clauses differ mostly
in numbers, so the document name has to be inside the embedded text. The payload also carries the
specialty, so a search can be filtered to one handbook.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import DATA_DIR, PolicySource

MAX_CHARACTERS = 1800  # about 400 to 500 tokens; a single longer paragraph (e.g. a table) is kept whole

# Corpus slug of each specialty handbook -> the specialty name stored in the payload (the search scope filter).
SPECIALTY_BY_SLUG = {
    "specialty-cardiology": "Cardiology",
    "specialty-orthopaedics": "Orthopaedics",
    "specialty-dermatology": "Dermatology",
    "specialty-endocrinology": "Endocrinology",
    "specialty-gastroenterology": "Gastroenterology",
    "specialty-pulmonology": "Pulmonology",
    "specialty-neurology": "Neurology",
    "specialty-ent": "ENT",
}
PLAN_CODES = ("MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE")  # insurance plans the corpus names
# An ATX heading: group 1 is the "#" run (its length is the level), group 2 the text without closing "#"s.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# "**Document ID:** MHP-SLH-CARD", "Document reference/ref: ...", "## Operations Policy OP-CS-014: ..."
_DOCUMENT_ID = re.compile(r"(?:Document (?:ID|ref(?:erence)?)\W*|(?:Policy|Procedure) )([A-Z]{2,4}(?:-[A-Z0-9]{1,5}){1,4})\b")
# A procedure code: a specialty prefix and a short procedure mnemonic, e.g. GAST-COLO.
_PROCEDURE_CODE = re.compile(r"\b(?:CARD|ORTH|DERM|ENDO|GAST|PULM|NEUR|ENT)-[A-Z]{3,5}\b")
_NAMESPACE = uuid.UUID("6f1c2b0e-3c1e-4c55-9d0a-6b7c1f0e5a11")  # fixed, so chunk IDs are stable across builds


@dataclass(frozen=True)
class Chunk:
    """One indexed passage. Every field is stored as the Qdrant point's payload (see to_payload())."""

    chunk_id: str  # UUID (Qdrant point ID), derived from slug + ordinal
    doc_slug: str
    doc_title: str  # exactly the corpus/index.json title; eval cases cite documents by it
    doc_group: str  # index.json "section": operations, insurance, safety, compliance, clinical_ops
    doc_id: str | None  # policy number, e.g. OP-CS-014
    heading: str  # section path inside the document
    text: str  # context header + section body: this is what is embedded and returned
    ordinal: int  # position of the chunk inside its document, from 0
    specialty: str | None = None  # set for the specialty handbooks only; None for clinic-wide documents
    procedure_codes: list[str] = field(default_factory=list)  # codes named in the heading or body
    plan_codes: list[str] = field(default_factory=list)
    source_format: str = PolicySource.MARKDOWN.value  # which corpus copy it was cut from (a PolicySource value)

    def to_payload(self) -> dict:
        return asdict(self)


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """(heading path, body) for every heading that has body text. Title headings (#) are left out of the
    path: every document opens with two of them and they repeat the title."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[list[str], str]] = []
    body: list[str] = []

    def flush():  # close the section collected so far under the current heading path
        text = "\n".join(body).strip()  # "---" rules are already dropped; a leading "- " is a list item
        if text:
            sections.append(([title for depth, title in path if depth > 1], text))
        body.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [entry for entry in path if entry[0] < level] + [(level, match.group(2).strip())]
        elif line.strip() != "---":
            body.append(line)
    flush()
    return sections


def _split_pieces(body: str, limit: int) -> list[str]:
    """Split a section between paragraphs so each piece stays under `limit` where possible."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]
    pieces, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    slug: str, title: str, group: str, markdown: str, max_characters: int = MAX_CHARACTERS,
    source_format: str = PolicySource.MARKDOWN.value,
) -> list[Chunk]:
    """Cut one document into chunks. `title` and `group` come from corpus/index.json; `max_characters` bounds a
    chunk's body (the context header is extra)."""
    # The two copies get different IDs, so an index built from one never overwrites points of the other.
    prefix = "" if source_format == PolicySource.MARKDOWN else f"{source_format}:"
    document_id_match = _DOCUMENT_ID.search(markdown)
    document_id = document_id_match.group(1) if document_id_match else None
    specialty = SPECIALTY_BY_SLUG.get(slug)
    chunks: list[Chunk] = []
    for path, body in _split_sections(markdown):
        heading = " > ".join(path) or "Document information"
        for piece in _split_pieces(body, max_characters):
            header = " | ".join(part for part in (title, document_id, heading) if part)
            ordinal = len(chunks)
            tagged_text = f"{heading}\n{piece}"  # where procedure and plan codes are looked for
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_NAMESPACE, f"{prefix}{slug}#{ordinal}")),
                    doc_slug=slug,
                    doc_title=title,
                    doc_group=group,
                    doc_id=document_id,
                    heading=heading,
                    text=f"[{header}]\n{piece}",
                    ordinal=ordinal,
                    specialty=specialty,
                    procedure_codes=sorted(set(_PROCEDURE_CODE.findall(tagged_text))),
                    plan_codes=[plan_code for plan_code in PLAN_CODES if plan_code in tagged_text],
                    source_format=source_format,
                )
            )
    return chunks


SOURCES = (PolicySource.MARKDOWN.value, PolicySource.PDF.value)  # the two copies of the corpus that can be indexed


def load_corpus_chunks(
    data_dir: Path | str | None = None, max_characters: int = MAX_CHARACTERS, source: str = PolicySource.MARKDOWN.value
) -> list[Chunk]:
    """Chunk every document listed in corpus/index.json, from its markdown or its PDF copy."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    root = Path(data_dir) if data_dir else DATA_DIR
    index = json.loads((root / "corpus" / "index.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for document in index:
        if source == PolicySource.PDF:
            from .pdf_loader import convert_pdf_to_markdown  # pypdf is needed only for this path

            text = convert_pdf_to_markdown(root / document["pdf"])
        else:
            text = (root / document["markdown"]).read_text(encoding="utf-8")
        chunks.extend(chunk_document(document["slug"], document["title"], document["section"], text, max_characters, source))
    return chunks
