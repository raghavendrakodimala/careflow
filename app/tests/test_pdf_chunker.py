"""The PDF copy of the corpus goes through the same chunker as the markdown copy and loses nothing.

The toolkit renders the PDFs lossily on purpose (headings flattened, lines wrapped, blank lines gone):
a chunker that only performs well on the markdown copy is not ready for the PDF. The markdown
copy is the reference: every clause, table row and figure in it must reach a PDF chunk.
"""

import json
import re

import pytest

from src.config import DATA_DIR
from src.rag.chunker import MAX_CHARACTERS, SPECIALTY_BY_SLUG, load_corpus_chunks
from src.rag.pdf_loader import convert_pdf_to_markdown

INDEX = json.loads((DATA_DIR / "corpus" / "index.json").read_text(encoding="utf-8"))
SLUGS = [document["slug"] for document in INDEX]


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"\*\*?|`", "", text).split())  # markdown emphasis is not in the PDF


@pytest.fixture(scope="module")
def markdown_chunks():
    return load_corpus_chunks()


@pytest.fixture(scope="module")
def pdf_chunks():
    return load_corpus_chunks(source="pdf")


def _select_chunks(chunks, slug):
    return [chunk for chunk in chunks if chunk.doc_slug == slug]


@pytest.mark.parametrize("slug", SLUGS)
def test_every_markdown_line_reaches_a_pdf_chunk(pdf_chunks, slug):
    doc = next(document for document in INDEX if document["slug"] == slug)
    body = _normalise("\n".join(chunk.text for chunk in _select_chunks(pdf_chunks, slug)))
    for line in (DATA_DIR / doc["markdown"]).read_text(encoding="utf-8").splitlines():
        line = _normalise(line)
        if line and not line.startswith("#") and line != "---":
            assert line in body, line[:80]


@pytest.mark.parametrize("slug", SLUGS)
def test_pdf_sections_match_the_markdown_sections(markdown_chunks, pdf_chunks, slug):
    markdown_sections, pdf_sections = _select_chunks(markdown_chunks, slug), _select_chunks(pdf_chunks, slug)
    assert len(pdf_sections) == len(markdown_sections)
    # The renderer flattens heading levels, so compare section names, not full paths. The PDF's untitled
    # opening section (the document's metadata) is called "Document information".
    collect_leaf_headings = lambda chunks: {chunk.heading.split(" > ")[-1] for chunk in chunks}  # noqa: E731
    assert collect_leaf_headings(pdf_sections) - {"Document information"} <= collect_leaf_headings(markdown_sections)
    assert {chunk.doc_id for chunk in pdf_sections} == {chunk.doc_id for chunk in markdown_sections}
    assert {chunk.doc_title for chunk in pdf_sections} == {chunk.doc_title for chunk in markdown_sections}
    assert set().union(*(chunk.procedure_codes for chunk in pdf_sections)) == set().union(*(chunk.procedure_codes for chunk in markdown_sections))
    assert set().union(*(chunk.plan_codes for chunk in pdf_sections)) == set().union(*(chunk.plan_codes for chunk in markdown_sections))
    assert all(chunk.specialty == SPECIALTY_BY_SLUG.get(slug) for chunk in pdf_sections)


def test_pdf_chunks_keep_tables_whole_and_carry_the_context_header(pdf_chunks):
    card = [chunk for chunk in pdf_chunks if chunk.doc_slug == "specialty-cardiology" and "| CARD-CONS |" in chunk.text]
    assert card and all("| CARD-CATH |" in chunk.text for chunk in card)
    for chunk in pdf_chunks:
        assert chunk.text.split("\n", 1)[0].startswith(f"[{chunk.doc_title}")
        rows = [line for line in chunk.text.splitlines() if line.startswith("|")]
        if len(rows) > 1:
            assert re.match(r"\|[-| :]+\|", rows[1]), (chunk.doc_slug, chunk.heading)  # header + separator together
        if len(chunk.text) > MAX_CHARACTERS + 200:
            assert "\n\n" not in chunk.text.split("\n", 1)[1]  # only one over-long paragraph may exceed the target


def test_pdf_and_markdown_chunks_have_distinct_stable_ids(markdown_chunks, pdf_chunks):
    assert {chunk.source_format for chunk in pdf_chunks} == {"pdf"} and {chunk.source_format for chunk in markdown_chunks} == {"markdown"}
    assert not {chunk.chunk_id for chunk in markdown_chunks} & {chunk.chunk_id for chunk in pdf_chunks}
    assert [chunk.chunk_id for chunk in load_corpus_chunks(source="pdf")] == [chunk.chunk_id for chunk in pdf_chunks]


def test_pdf_reader_rebuilds_headings_and_joins_wrapped_lines():
    text = convert_pdf_to_markdown(DATA_DIR / "corpus" / "pdf" / "copay-schedule.pdf")
    assert text.startswith("# Co-payment and Cost-Share Schedule")
    assert "### 2. Plan Terms" in text
    # 4.2(b) wraps over two lines in the PDF; it comes back as one line.
    assert ("- (b) the plan coinsurance rate applied to the remainder of the list price after the unmet "
            "deductible is subtracted,") in text
    neuro = convert_pdf_to_markdown(DATA_DIR / "corpus" / "pdf" / "specialty-neurology.pdf")
    assert "\nDocument ref: MHP-SLH-NEUR-01" in neuro  # metadata split from the bold title block


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        load_corpus_chunks(source="docx")
