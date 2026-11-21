"""Groundedness check for policy answers (M4). Deterministic: no LLM call, so it runs on every answer.

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

The user's own words are untrusted (feedback M4-7): a code the user typed never grounds anything, and a
figure the user typed (their deductible, the "100 percent" a friend claimed) counts only inside an answer
that cites a retrieved document, i.e. one that is checking the claim against the policy rather than
repeating it. Titles are matched ignoring case, hyphens and US/UK spelling (feedback M4-2).

The graph (graph.py, `ground` node) replaces an answer that fails with a staff handoff. An LLM judge for
paraphrased claims without figures is left for M8's guardrails.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

from .config import DATA_DIR


@cache
def _corpus_documents(data_dir: str) -> tuple[str, ...]:
    try:
        index = json.loads((Path(data_dir) / "corpus" / "index.json").read_text(encoding="utf-8"))
        return tuple(d["title"] for d in index)
    except (OSError, ValueError, KeyError):
        return ()


def corpus_documents(data_dir: Path | str | None = None) -> tuple[str, ...]:
    """Titles of every corpus document (for the phantom-citation check); empty if the corpus is absent."""
    return _corpus_documents(str(data_dir or DATA_DIR))


# ---- numbers ---------------------------------------------------------------------------------------
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_WORD_NUM = (rf"(?:(?:{'|'.join(_TENS)})(?:[-\s](?:{'|'.join(_UNITS[1:10])}))?|{'|'.join(_UNITS)})")
_NUM = rf"(?:\d[\d,]*(?:\.\d+)?|{_WORD_NUM})"
_START = r"(?<![\w.-])"  # not inside a word, a decimal or a hyphenated compound ("twenty-four" is one number)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(rf"{_START}(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", re.I)
_DURATION = re.compile(
    rf"{_START}({_NUM})(?:-|\s+)(business\s+|working\s+|calendar\s+|clock\s+)?(minute|hour|day|week|month|year)s?\b",
    re.I,
)
_CODE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{2,10}){1,3}\b")
_BARE_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")
# Structured tool fields whose number is a duration, not an amount (feedback M4-2d).
_UNIT_KEYS = (("_business_days", "business day"), ("_minutes", "minute"), ("_hours", "hour"), ("_days", "day"),
              ("_weeks", "week"))


def _number(text: str) -> str:
    low = text.lower()
    if low in _UNITS:
        return str(_UNITS.index(low))
    parts = re.split(r"[-\s]+", low)
    if parts[0] in _TENS:
        return str(_TENS[parts[0]] + (_UNITS.index(parts[1]) if len(parts) > 1 else 0))
    text = text.replace(",", "")
    return text[:-3] if text.endswith(".00") else text.rstrip("0").rstrip(".") if "." in text else text


def figures(text: str) -> dict[str, str]:
    """{normalised figure: as written} for every figure in `text`."""
    out: dict[str, str] = {}
    for m in _MONEY.finditer(text):
        out[f"${_number(m.group(1))}"] = m.group(0)
    for m in _PERCENT.finditer(text):
        out[f"{_number(m.group(1))}%"] = m.group(0)
    for m in _DURATION.finditer(text):
        qualifier = (m.group(2) or "").strip().lower().replace("working", "business")
        out[" ".join(x for x in (_number(m.group(1)), qualifier, m.group(3).lower()) if x)] = m.group(0)
    for m in _CODE.finditer(text):
        out[m.group(0)] = m.group(0)
    return out


def _is_code(norm: str) -> bool:
    return bool(_CODE.fullmatch(norm))


def _with_plain_forms(found: set[str]) -> set[str]:
    """A qualified duration ("5 business day") also supports the plain form ("5 day")."""
    for f in list(found):
        parts = f.split()
        if len(parts) == 3:
            found.add(f"{parts[0]} {parts[2]}")
    return found


def _support(text: str) -> set[str]:
    return _with_plain_forms(set(figures(text)))


def _structured_support(value, key: str = "") -> set[str]:
    """Figures a tool result supports. Amounts are bare numbers there (list_price_usd: 2600, working
    "... = 480"), so a number supports the same amount in dollars; a number under a unit-named key
    (duration_minutes: 30) supports that duration instead."""
    if isinstance(value, dict):
        return set().union(*(_structured_support(v, str(k)) for k, v in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(_structured_support(v, key) for v in value)) if value else set()
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float)):
        n = _number(str(value))
        unit = next((u for suffix, u in _UNIT_KEYS if key.endswith(suffix)), None)
        return _with_plain_forms({f"{n} {unit}"}) if unit else {f"${n}"}
    text = str(value)
    return _support(text) | {f"${_number(m.group(1))}" for m in _BARE_NUMBER.finditer(text)}


# ---- document names ----------------------------------------------------------------------------------
_SPELLING = (("authoriz", "authoris"), ("orthoped", "orthopaed"), ("pediatric", "paediatric"), ("anesthe", "anaesthe"))


def _norm_name(text: str) -> str:
    """Case, hyphens, spaces and US/UK spelling removed: "Pre-Authorization" == "pre authorisation"."""
    text = text.lower()
    for us, uk in _SPELLING:
        text = text.replace(us, uk)
    return re.sub(r"[^a-z0-9]", "", text)


def _names(passage: dict) -> list[str]:
    return [n for n in (passage.get("document"), passage.get("document_id")) if n]


# ---- check -------------------------------------------------------------------------------------------
@dataclass
class GroundingResult:
    checked: bool = False  # the answer was checked (a policy search in the window, or a document named)
    grounded: bool = True
    unsupported: list[str] = field(default_factory=list)  # figures, as written in the answer
    uncited: list[str] = field(default_factory=list)  # passage-only figures with no document named
    phantom_citations: list[str] = field(default_factory=list)  # named but not retrieved

    def as_dict(self) -> dict:
        return asdict(self)


def names_a_document(answer: str, documents) -> bool:
    text = _norm_name(answer)
    return any(_norm_name(d) in text for d in documents if d)


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
    what the user wrote (untrusted, see the module docstring)."""
    text = _norm_name(answer)
    trusted: set[str] = set()
    for t in trusted_sources:
        trusted |= _support(t)
    for r in tool_results or []:
        trusted |= _structured_support(r)
    by_passage = [(p, _support(f"{p.get('document', '')} {p.get('document_id') or ''}\n{p.get('text', '')}"))
                  for p in passages]
    cites_retrieved = any(_norm_name(n) in text for p in passages for n in _names(p))
    from_user: set[str] = set()
    for m in user_messages or []:
        from_user |= {f for f in _support(m) if not _is_code(f)}  # a code the user typed grounds nothing

    result = GroundingResult(checked=True)
    for norm, written in figures(answer).items():
        if norm in trusted:
            continue
        holders = [p for p, sup in by_passage if norm in sup]
        if holders:
            if not any(_norm_name(n) in text for p in holders for n in _names(p)):
                result.uncited.append(written)
        elif norm in from_user and cites_retrieved:
            continue  # the user's own figure, discussed in an answer that cites the policy
        else:
            result.unsupported.append(written)

    # Documents the agent was given: retrieved passages, and the `source` a structured tool reports for
    # its figures (check_preauth, calculate_copay).
    retrieved = {_norm_name(n) for p in passages for n in _names(p)}
    retrieved |= {_norm_name(d) for r in tool_results or [] for d in (r.get("source") or []) if isinstance(d, str)}
    result.phantom_citations = [d for d in known_documents or []
                                if _norm_name(d) in text and _norm_name(d) not in retrieved]
    result.grounded = not (result.unsupported or result.uncited or result.phantom_citations)
    return result
