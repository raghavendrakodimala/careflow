"""M4: the PDF copy of the corpus goes through the same chunker as the markdown copy and loses nothing.

The toolkit renders the PDFs lossily on purpose (headings flattened, lines wrapped, blank lines gone):
"If your M4 chunker only performs well on the markdown copy, it is not ready for the PDF." The markdown
copy is the reference: every clause, table row and figure in it must reach a PDF chunk.
"""

import json
import re

import pytest

from careflow.config import DATA_DIR
from careflow.rag.chunker import MAX_CHARS, SPECIALTY_BY_SLUG, load_corpus_chunks
from careflow.rag.pdf_loader import pdf_to_markdown

INDEX = json.loads((DATA_DIR / "corpus" / "index.json").read_text(encoding="utf-8"))
SLUGS = [d["slug"] for d in INDEX]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"\*\*?|`", "", text).split())  # markdown emphasis is not in the PDF


@pytest.fixture(scope="module")
def md():
    return load_corpus_chunks()


@pytest.fixture(scope="module")
def pdf():
    return load_corpus_chunks(source="pdf")


def _of(chunks, slug):
    return [c for c in chunks if c.doc_slug == slug]


@pytest.mark.parametrize("slug", SLUGS)
def test_every_markdown_line_reaches_a_pdf_chunk(pdf, slug):
    doc = next(d for d in INDEX if d["slug"] == slug)
    body = _norm("\n".join(c.text for c in _of(pdf, slug)))
    for line in (DATA_DIR / doc["markdown"]).read_text(encoding="utf-8").splitlines():
        line = _norm(line)
        if line and not line.startswith("#") and line != "---":
            assert line in body, line[:80]


@pytest.mark.parametrize("slug", SLUGS)
def test_pdf_sections_match_the_markdown_sections(md, pdf, slug):
    m, p = _of(md, slug), _of(pdf, slug)
    assert len(p) == len(m)
    # The renderer flattens heading levels, so compare section names, not full paths. The PDF's untitled
    # opening section (the document's metadata) is called "Document information".
    leaf = lambda chunks: {c.heading.split(" > ")[-1] for c in chunks}  # noqa: E731
    assert leaf(p) - {"Document information"} <= leaf(m)
    assert {c.doc_id for c in p} == {c.doc_id for c in m}
    assert {c.doc_title for c in p} == {c.doc_title for c in m}
    assert set().union(*(c.procedure_codes for c in p)) == set().union(*(c.procedure_codes for c in m))
    assert set().union(*(c.plan_codes for c in p)) == set().union(*(c.plan_codes for c in m))
    assert all(c.specialty == SPECIALTY_BY_SLUG.get(slug) for c in p)


def test_pdf_chunks_keep_tables_whole_and_carry_the_context_header(pdf):
    card = [c for c in pdf if c.doc_slug == "specialty-cardiology" and "| CARD-CONS |" in c.text]
    assert card and all("| CARD-CATH |" in c.text for c in card)
    for c in pdf:
        assert c.text.split("\n", 1)[0].startswith(f"[{c.doc_title}")
        rows = [ln for ln in c.text.splitlines() if ln.startswith("|")]
        if len(rows) > 1:
            assert re.match(r"\|[-| :]+\|", rows[1]), (c.doc_slug, c.heading)  # header + separator together
        if len(c.text) > MAX_CHARS + 200:
            assert "\n\n" not in c.text.split("\n", 1)[1]  # only one over-long paragraph may exceed the target


def test_pdf_and_markdown_chunks_have_distinct_stable_ids(md, pdf):
    assert {c.source_format for c in pdf} == {"pdf"} and {c.source_format for c in md} == {"markdown"}
    assert not {c.chunk_id for c in md} & {c.chunk_id for c in pdf}
    assert [c.chunk_id for c in load_corpus_chunks(source="pdf")] == [c.chunk_id for c in pdf]


def test_pdf_reader_rebuilds_headings_and_joins_wrapped_lines():
    text = pdf_to_markdown(DATA_DIR / "corpus" / "pdf" / "copay-schedule.pdf")
    assert text.startswith("# Co-payment and Cost-Share Schedule")
    assert "### 2. Plan Terms" in text
    # 4.2(b) wraps over two lines in the PDF; it comes back as one line.
    assert ("- (b) the plan coinsurance rate applied to the remainder of the list price after the unmet "
            "deductible is subtracted,") in text
    neuro = pdf_to_markdown(DATA_DIR / "corpus" / "pdf" / "specialty-neurology.pdf")
    assert "\nDocument ref: MHP-SLH-NEUR-01" in neuro  # metadata split from the bold title block


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        load_corpus_chunks(source="docx")
