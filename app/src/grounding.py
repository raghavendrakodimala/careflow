"""Groundedness check for policy answers. Deterministic: no LLM call, so it runs on every answer.

When the conversation window holds a policy search (or the answer names a corpus document), the answer
is checked against what the agent was actually given:

1. Figures: every amount ($), percentage, duration ("5 business days", "thirty minutes") and code
   ("GAST-COLO", "OP-CS-014", "MERIDIAN-SILVER") in the answer must appear in a source:
   - a passage search_policies returned;
   - another tool's result (the co-pay calculator's working, a slot's duration_minutes, ...);
   - trusted text: the fixed scripts the prompt makes the model say word for word, the patient memory
     note, and earlier answers in the window that already passed this check.
   A figure found nowhere was made up or computed by the model; an unsupported coverage figure is the
   failure that matters most.
2. Citation: if a figure is supported only by passages, the answer must name one of those passages'
   documents (title or document ID), so staff and patients can check it.
3. No phantom citations: any corpus document the answer names must be one that was retrieved.

The user's own words are untrusted: a code the user typed never grounds anything, and a figure the user
typed (their deductible, the "100 percent" a friend claimed) counts only inside an answer that cites a
retrieved document, i.e. one that is checking the claim against the policy rather than repeating it.
Titles are matched ignoring case, hyphens and US/UK spelling.

The graph (graph.py, `ground` node) replaces an answer that fails with a staff handoff. Paraphrased
claims that contain no figure are outside this check.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

from .config import DATA_DIR


@cache
def _load_corpus_documents(data_dir: str) -> tuple[str, ...]:
    """Cached per directory (a str, so it is hashable): the index file is read once per process."""
    try:
        index = json.loads((Path(data_dir) / "corpus" / "index.json").read_text(encoding="utf-8"))
        return tuple(document["title"] for document in index)
    except (OSError, ValueError, KeyError):
        return ()


def list_corpus_documents(data_dir: Path | str | None = None) -> tuple[str, ...]:
    """Titles of every corpus document (for the phantom-citation check); empty if the corpus is absent."""
    return _load_corpus_documents(str(data_dir or DATA_DIR))


# ---- numbers ---------------------------------------------------------------------------------------
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]  # the word's position is its value
_TENS = {word: 10 * tens for tens, word in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], start=2)}
# A number written in words, zero to ninety-nine: "seven", "thirty", "twenty-four", "twenty four".
_WORD_NUMBER = (rf"(?:(?:{'|'.join(_TENS)})(?:[-\s](?:{'|'.join(_UNITS[1:10])}))?|{'|'.join(_UNITS)})")
_NUMBER = rf"(?:\d[\d,]*(?:\.\d+)?|{_WORD_NUMBER})"  # digits (with thousands commas and decimals) or a word number
_START = r"(?<![\w.-])"  # not inside a word, a decimal or a hyphenated compound ("twenty-four" is one number)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")  # "$1,200.50", "$ 45"; group 1 is the amount
_PERCENT = re.compile(rf"{_START}(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", re.I)
# "5 business days", "5 business-day", "30-minute", "thirty minutes": group 1 the number, 2 the optional
# qualifier, 3 the unit. A hyphen may join any two parts; without it in the qualifier, "a 5 business-day
# window" was not a figure at all, so the grounding check never saw it.
_DURATION = re.compile(
    rf"{_START}({_NUMBER})(?:-|\s+)((?:business|working|calendar|clock)(?:-|\s+))?(minute|hour|day|week|month|year)s?\b",
    re.I,
)
# An upper-case hyphenated code of two to four parts: procedure (GAST-COLO), policy (OP-CS-014), plan (MERIDIAN-SILVER).
_CODE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{2,10}){1,3}\b")
_BARE_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")  # a number standing alone, not part of a word
# Structured tool fields whose number is a duration, not an amount. Order matters: the first suffix that
# matches wins, so "_business_days" must come before "_days".
_UNIT_KEYS = (("_business_days", "business day"), ("_minutes", "minute"), ("_hours", "hour"), ("_days", "day"),
              ("_weeks", "week"))


def _normalise_number(text: str) -> str:
    """One spelling per value, so figures compare as strings: "thirty" -> "30", "twenty-four" -> "24",
    "1,200.00" -> "1200", "2.50" -> "2.5"."""
    lowered = text.lower()
    if lowered in _UNITS:
        return str(_UNITS.index(lowered))
    parts = re.split(r"[-\s]+", lowered)
    if parts[0] in _TENS:
        return str(_TENS[parts[0]] + (_UNITS.index(parts[1]) if len(parts) > 1 else 0))
    text = text.replace(",", "")
    return text[:-3] if text.endswith(".00") else text.rstrip("0").rstrip(".") if "." in text else text


def find_figures(text: str) -> dict[str, str]:
    """{normalised figure: as written} for every figure in `text`. Normalised forms: "$1200", "20%",
    "5 business day" (number, optional qualifier, singular unit; "working" counts as "business") and codes
    as written."""
    found: dict[str, str] = {}
    for match in _MONEY.finditer(text):
        found[f"${_normalise_number(match.group(1))}"] = match.group(0)
    for match in _PERCENT.finditer(text):
        found[f"{_normalise_number(match.group(1))}%"] = match.group(0)
    for match in _DURATION.finditer(text):
        qualifier = (match.group(2) or "").strip(" \t\n-").lower().replace("working", "business")
        found[" ".join(part for part in (_normalise_number(match.group(1)), qualifier, match.group(3).lower())
                       if part)] = match.group(0)
    for match in _CODE.finditer(text):
        found[match.group(0)] = match.group(0)
    return found


def _is_code(figure: str) -> bool:
    return bool(_CODE.fullmatch(figure))


def _add_plain_forms(found: set[str]) -> set[str]:
    """A qualified duration ("5 business day") also supports the plain form ("5 day")."""
    for figure in list(found):
        parts = figure.split()
        if len(parts) == 3:
            found.add(f"{parts[0]} {parts[2]}")
    return found


def _find_support(text: str) -> set[str]:
    """The normalised figures a piece of source text supports."""
    return _add_plain_forms(set(find_figures(text)))


def _find_structured_support(value, key: str = "") -> set[str]:
    """Figures a tool result supports. Amounts are bare numbers there (list_price_usd: 2600, working
    "... = 480"), so a number supports the same amount in dollars; a number under a unit-named key
    (duration_minutes: 30) supports that duration instead."""
    if isinstance(value, dict):
        return set().union(
            *(_find_structured_support(item, str(name)) for name, item in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(_find_structured_support(item, key) for item in value)) if value else set()
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float)):
        number = _normalise_number(str(value))
        unit = next((name for suffix, name in _UNIT_KEYS if key.endswith(suffix)), None)
        return _add_plain_forms({f"{number} {unit}"}) if unit else {f"${number}"}
    text = str(value)
    return _find_support(text) | {f"${_normalise_number(match.group(1))}" for match in _BARE_NUMBER.finditer(text)}


# ---- document names ----------------------------------------------------------------------------------
# (US, UK) word stems; names are compared in the UK form, which is the corpus's.
_SPELLING = (("authoriz", "authoris"), ("orthoped", "orthopaed"), ("pediatric", "paediatric"), ("anesthe", "anaesthe"))


def _normalise_name(text: str) -> str:
    """Case, hyphens, spaces and US/UK spelling removed: "Pre-Authorization" == "pre authorisation"."""
    text = text.lower()
    for american, british in _SPELLING:
        text = text.replace(american, british)
    return re.sub(r"[^a-z0-9]", "", text)


def _list_names(passage: dict) -> list[str]:
    """The names an answer can cite a passage by: its document title and its document ID."""
    return [name for name in (passage.get("document"), passage.get("document_id")) if name]


def _cites_any(normalised_answer: str, passages: list[dict]) -> bool:
    """Does the answer (already through _normalise_name) name the document of at least one of `passages`?"""
    return any(_normalise_name(name) in normalised_answer for passage in passages for name in _list_names(passage))


# ---- check -------------------------------------------------------------------------------------------
@dataclass
class GroundingResult:
    checked: bool = False  # the answer was checked (a policy search in the window, or a document named)
    grounded: bool = True
    unsupported: list[str] = field(default_factory=list)  # figures, as written in the answer
    uncited: list[str] = field(default_factory=list)  # passage-only figures with no document named
    phantom_citations: list[str] = field(default_factory=list)  # named but not retrieved

    def to_dict(self) -> dict:
        return asdict(self)


def has_document_name(answer: str, documents) -> bool:
    """Does the answer name any of `documents` (titles or IDs)? Decides whether an answer is checked at all
    when the window holds no policy search."""
    text = _normalise_name(answer)
    return any(_normalise_name(document) in text for document in documents if document)


def check_answer(
    answer: str,
    passages: list[dict],
    trusted_sources: list[str],
    tool_results: list[dict] | None = None,
    user_messages: list[str] | None = None,
    known_documents=None,
) -> GroundingResult:
    """`passages`: search_policies results ({document, document_id, text}); `trusted_sources`: prompt
    scripts, memory note, earlier verified answers; `tool_results`: other tools' results; `user_messages`:
    what the user wrote (untrusted, see the module docstring); `known_documents`: every corpus title, for
    the phantom-citation check (skipped when not given)."""
    text = _normalise_name(answer)
    trusted: set[str] = set()
    for trusted_text in trusted_sources:
        trusted |= _find_support(trusted_text)
    for tool_result in tool_results or []:
        trusted |= _find_structured_support(tool_result)
    by_passage = [
        (passage,
         _find_support(f"{passage.get('document', '')} {passage.get('document_id') or ''}\n{passage.get('text', '')}"))
        for passage in passages]
    cites_retrieved = _cites_any(text, passages)
    from_user: set[str] = set()
    for message in user_messages or []:
        # a code the user typed grounds nothing
        from_user |= {figure for figure in _find_support(message) if not _is_code(figure)}

    result = GroundingResult(checked=True)
    for normalised, written in find_figures(answer).items():
        if normalised in trusted:
            continue
        holders = [passage for passage, supported in by_passage if normalised in supported]
        if holders:
            if not _cites_any(text, holders):
                result.uncited.append(written)
        elif normalised in from_user and cites_retrieved:
            continue  # the user's own figure, discussed in an answer that cites the policy
        else:
            result.unsupported.append(written)

    # Documents the agent was given: retrieved passages, and the `source` a structured tool reports for
    # its figures (check_preauth, calculate_copay).
    retrieved = {_normalise_name(name) for passage in passages for name in _list_names(passage)}
    retrieved |= {_normalise_name(source) for tool_result in tool_results or []
                  for source in (tool_result.get("source") or []) if isinstance(source, str)}
    result.phantom_citations = [document for document in known_documents or []
                                if _normalise_name(document) in text and _normalise_name(document) not in retrieved]
    result.grounded = not (result.unsupported or result.uncited or result.phantom_citations)
    return result
