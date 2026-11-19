"""Structure-aware chunking of the policy corpus (markdown copy).

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

from ..config import DATA_DIR

MAX_CHARS = 1800  # about 400 to 500 tokens; a single longer paragraph (e.g. a table) is kept whole

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
PLAN_CODES = ("MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# "**Document ID:** MHP-SLH-CARD", "Document reference/ref: ...", "## Operations Policy OP-CS-014: ..."
_DOC_ID = re.compile(r"(?:Document (?:ID|ref(?:erence)?)\W*|(?:Policy|Procedure) )([A-Z]{2,4}(?:-[A-Z0-9]{1,5}){1,4})\b")
_PROC_CODE = re.compile(r"\b(?:CARD|ORTH|DERM|ENDO|GAST|PULM|NEUR|ENT)-[A-Z]{3,5}\b")
_NAMESPACE = uuid.UUID("6f1c2b0e-3c1e-4c55-9d0a-6b7c1f0e5a11")  # fixed, so chunk IDs are stable across builds


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # UUID (Qdrant point ID), derived from slug + ordinal
    doc_slug: str
    doc_title: str  # exactly the corpus/index.json title; eval cases cite documents by it
    doc_group: str  # index.json "section": operations, insurance, safety, compliance, clinical_ops
    doc_id: str | None  # policy number, e.g. OP-CS-014
    heading: str  # section path inside the document
    text: str  # context header + section body: this is what is embedded and returned
    ordinal: int
    specialty: str | None = None
    procedure_codes: list[str] = field(default_factory=list)
    plan_codes: list[str] = field(default_factory=list)
    source_format: str = "markdown"

    def payload(self) -> dict:
        return asdict(self)


def _sections(markdown: str) -> list[tuple[list[str], str]]:
    """(heading path, body) for every heading that has body text. Title headings (#) are left out of the
    path: every document opens with two of them and they repeat the title."""
    path: list[tuple[int, str]] = []
    out: list[tuple[list[str], str]] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()  # "---" rules are already dropped; a leading "- " is a list item
        if text:
            out.append(([h for lvl, h in path if lvl > 1], text))
        body.clear()

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = [p for p in path if p[0] < level] + [(level, m.group(2).strip())]
        elif line.strip() != "---":
            body.append(line)
    flush()
    return out


def _pieces(body: str, limit: int) -> list[str]:
    """Split a section between paragraphs so each piece stays under `limit` where possible."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > limit:
            pieces.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        pieces.append(current)
    return pieces


def chunk_document(slug: str, title: str, group: str, markdown: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    doc_id_match = _DOC_ID.search(markdown)
    doc_id = doc_id_match.group(1) if doc_id_match else None
    specialty = SPECIALTY_BY_SLUG.get(slug)
    chunks: list[Chunk] = []
    for path, body in _sections(markdown):
        heading = " > ".join(path) or "Document information"
        for piece in _pieces(body, max_chars):
            header = " | ".join(x for x in (title, doc_id, heading) if x)
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_NAMESPACE, f"{slug}#{ordinal}")),
                    doc_slug=slug,
                    doc_title=title,
                    doc_group=group,
                    doc_id=doc_id,
                    heading=heading,
                    text=f"[{header}]\n{piece}",
                    ordinal=ordinal,
                    specialty=specialty,
                    procedure_codes=sorted(set(_PROC_CODE.findall(f"{heading}\n{piece}"))),
                    plan_codes=[p for p in PLAN_CODES if p in f"{heading}\n{piece}"],
                )
            )
    return chunks


def load_corpus_chunks(data_dir: Path | str | None = None, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Chunk every document listed in corpus/index.json (markdown copies)."""
    root = Path(data_dir) if data_dir else DATA_DIR
    index = json.loads((root / "corpus" / "index.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for doc in index:
        text = (root / doc["markdown"]).read_text(encoding="utf-8")
        chunks.extend(chunk_document(doc["slug"], doc["title"], doc["section"], text, max_chars))
    return chunks
