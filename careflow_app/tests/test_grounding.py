"""M4 groundedness: the deterministic figure and citation check, the graph's `ground` node, and the what-if
co-pay estimate that lets policy answers get their arithmetic from a tool instead of the model."""

import json
from types import SimpleNamespace

from careflow.agent import run_agent
from careflow.graph import GROUNDING_ANSWER, GROUNDING_REASON
from careflow.grounding import check_answer, corpus_documents, figures
from careflow.prompts import DEFLECTION_SCRIPT
from careflow.tools import CareFlowTools

COPAY = {
    "document": "Co-payment and Cost-Share Schedule",
    "document_id": "OPS-FIN-014",
    "section": "2. Plan Terms",
    "text": "| MERIDIAN-BRONZE | $3,000 | 40% | $8,000 |\nImaging co-pay MERIDIAN-BRONZE $160",
}
PREAUTH = {
    "document": "Pre-Authorisation Requirement Matrix",
    "document_id": "OPS-PA-007",
    "section": "7A. Cardiology",
    "text": "MRI under MERIDIAN-BRONZE: pre-authorisation required, approval window 5 business days.",
}


# ---- figures ---------------------------------------------------------------------------------------
def test_figures_are_normalised():
    f = figures("You owe $1,480.00 (30 percent), within five business days or 30-minute slots; see GAST-COLO.")
    assert {"$1480", "30%", "5 business day", "30 minute", "GAST-COLO"} <= set(f)
    assert set(figures("2 working days")) == {"2 business day"}  # working day = business day
    assert figures("nothing to check here") == {}


# ---- check_answer ----------------------------------------------------------------------------------
def test_figures_from_a_cited_passage_are_grounded():
    r = check_answer("Per the Co-payment and Cost-Share Schedule, the MERIDIAN-BRONZE deductible is $3,000.",
                     [COPAY], [], user_messages=["What is the BRONZE deductible?"])
    assert r.checked and r.grounded and not (r.unsupported or r.uncited or r.phantom_citations)


def test_a_figure_found_nowhere_is_unsupported():
    r = check_answer("Per the Co-payment and Cost-Share Schedule, the deductible is $2,500.", [COPAY], [])
    assert not r.grounded and r.unsupported == ["$2,500"]


def test_a_passage_figure_needs_its_document_named():
    r = check_answer("The deductible is $3,000.", [COPAY], [])
    assert not r.grounded and r.uncited == ["$3,000"]
    assert check_answer("The deductible is $3,000 (OPS-FIN-014).", [COPAY], []).grounded  # a document ID counts


def test_figures_from_tools_and_the_fixed_scripts_need_no_citation():
    calc = {"ok": True, "list_price_usd": 2600, "patient_owes_usd": 1480, "deductible_met_usd": 500,
            "working": "unmet deductible 1000 + 30% of 1600 = 480"}
    answer = ("With $500 met, $1,000 of the deductible is unmet; 30% of $1,600 is $480, so you owe $1,480. "
              "A nurse will call within 4 business hours.")
    r = check_answer(answer, [], trusted_sources=[DEFLECTION_SCRIPT], tool_results=[calc],
                     user_messages=["I have met $500 so far"])
    assert r.grounded, r


def test_a_duration_must_keep_its_qualifier():
    p = {**PREAUTH, "text": "approval window 5 business days"}
    assert check_answer("Pre-Authorisation Requirement Matrix: 5 days.", [p], []).grounded  # plain form is fine
    r = check_answer("Pre-Authorisation Requirement Matrix: 5 calendar days.", [p], [])
    assert not r.grounded and r.unsupported == ["5 calendar days"]


def test_naming_a_document_that_was_not_retrieved_is_a_phantom_citation():
    r = check_answer("See the Referral Management Policy.", [COPAY], [], known_documents=corpus_documents())
    assert not r.grounded and r.phantom_citations == ["Referral Management Policy"]
    assert "Referral Management Policy" in corpus_documents() and len(corpus_documents()) == 14


# ---- the graph's ground node -----------------------------------------------------------------------
class FakeIndex:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, limit=4, specialty=None):
        return [SimpleNamespace(as_dict=lambda p=p: {**p, "score": 0.9}) for p in self.passages][:limit]


def _call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _script(*answers):
    """Model that searches first, then gives `answers` in turn."""
    steps = [SimpleNamespace(content="", tool_calls=[_call("s1", "search_policies", query="bronze deductible")])]
    steps += [SimpleNamespace(content=a, tool_calls=[]) for a in answers]
    seen = []

    def model(messages, specs):
        seen.append(messages)
        return steps[len(seen) - 1]

    return model, seen


def test_a_grounded_policy_answer_is_given_as_written():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    answer = "The Co-payment and Cost-Share Schedule sets the MERIDIAN-BRONZE deductible at $3,000."
    model, _ = _script(answer)
    r = run_agent("What is the MERIDIAN-BRONZE deductible?", tools, model=model)
    assert r.answer == answer and r.grounding["checked"] and r.grounding["grounded"]
    assert tools.flags == []


def test_an_ungrounded_policy_answer_is_withheld_and_staff_are_flagged():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script("The Co-payment and Cost-Share Schedule says the BRONZE deductible is $2,500.")
    r = run_agent("What is the MERIDIAN-BRONZE deductible?", tools, model=model)
    assert r.answer == GROUNDING_ANSWER and "$2,500" not in r.answer
    assert r.grounding["grounded"] is False and r.grounding["unsupported"] == ["$2,500"]
    assert r.tools_used == ["search_policies", "flag_for_human"]
    assert len(tools.flags) == 1 and tools.flags[0]["reason"].startswith(GROUNDING_REASON)
    assert tools.flags[0]["category"] == "administrative"


def test_a_withheld_answer_never_enters_the_conversation_history():
    from langgraph.checkpoint.memory import MemorySaver

    tools, saver = CareFlowTools(policy_index=FakeIndex([COPAY])), MemorySaver()
    model, _ = _script("The schedule says $9,999.")
    run_agent("deductible?", tools, model=model, checkpointer=saver, thread_id="t")
    seen = []
    later = lambda messages, specs: seen.append(messages) or SimpleNamespace(content="ok", tool_calls=[])  # noqa: E731
    run_agent("thanks", tools, model=later, checkpointer=saver, thread_id="t")
    history = json.dumps(seen[0])
    assert "$9,999" not in history and GROUNDING_ANSWER in history


def test_answers_on_turns_without_a_policy_search_are_not_checked():
    tools = CareFlowTools()
    r = run_agent("hi", tools, model=lambda m, s: SimpleNamespace(content="Your visit costs $12,345.", tool_calls=[]))
    assert r.answer == "Your visit costs $12,345." and r.grounding == {
        "checked": False, "grounded": True, "unsupported": [], "uncited": [], "phantom_citations": []}


def test_the_ground_flag_goes_to_the_verified_patient():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    ref = next(iter(tools.patients))
    model, _ = _script("It is $1.")
    r = run_agent("deductible?", tools, model=model, patient_ref=ref)
    assert r.answer == GROUNDING_ANSWER and tools.flags[0]["patient_ref"] == ref


# ---- what-if co-pay estimate (EV-906) --------------------------------------------------------------
def test_copay_estimate_for_a_plan_without_a_patient_record():
    t = CareFlowTools()
    r = t.call("calculate_copay", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER",
                                   "deductible_met_usd": 500})
    assert r["ok"] and r["estimate"] and r["patient_owes_usd"] == 1480 and r["preauth_required"] is True
    assert r["working"] == "unmet deductible 1000 + 30% of 1600 = 480"
    assert t.call("calculate_copay", {"procedure_code": "ORTH-ARTH", "plan_code": "meridian-bronze",
                                      "deductible_met_usd": 1800})["patient_owes_usd"] == 3200  # EV-015
    assert "unknown plan" in t.calculate_copay(procedure_code="GAST-COLO", plan_code="GOLDISH")["error"]


def test_a_patient_record_wins_over_a_stated_plan():
    t = CareFlowTools()
    p = next(p for p in t.patients.values() if p["eligibility_status"] == "active")
    r = t.calculate_copay(patient_ref=p["patient_ref"], procedure_code="GAST-COLO", plan_code="CIVIC-BASE")
    assert "estimate" not in r and r["plan_code"] == p["plan_code"]


# ---- architect review (notes/feedback.md, M4) ---------------------------------------------------------
def test_compound_number_words_are_one_number():  # M4-2a
    assert set(figures("within twenty-four hours")) == {"24 hour"}
    assert set(figures("a forty-five minute visit")) == {"45 minute"}
    assert set(figures("thirty days or twenty one weeks")) == {"30 day", "21 week"}
    # "twenty-four hours" is not grounded by the deflection script's "4 business hours".
    r = check_answer("Staff reply within twenty-four hours.", [], trusted_sources=[DEFLECTION_SCRIPT])
    assert r.unsupported == ["twenty-four hours"]


def test_the_system_prompt_rules_ground_nothing():  # M4-2b
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script("Per the Co-payment and Cost-Share Schedule, you will hear back within 72 hours.")
    r = run_agent("When will I hear back?", tools, model=model)
    assert r.answer == GROUNDING_ANSWER and r.grounding["unsupported"] == ["72 hours"]


def test_a_follow_up_that_reuses_an_earlier_search_is_checked():  # M4-2c
    from langgraph.checkpoint.memory import MemorySaver

    tools, saver = CareFlowTools(policy_index=FakeIndex([COPAY])), MemorySaver()
    model, _ = _script("The Co-payment and Cost-Share Schedule sets the MERIDIAN-BRONZE deductible at $3,000.")
    run_agent("BRONZE deductible?", tools, model=model, checkpointer=saver, thread_id="t")
    say = lambda text: lambda m, s: SimpleNamespace(content=text, tool_calls=[])  # noqa: E731
    ok = run_agent("And again?", tools, model=say("As I said, it is $3,000."), checkpointer=saver, thread_id="t")
    assert ok.grounding["checked"] and ok.grounding["grounded"]  # repeats a verified earlier answer
    bad = run_agent("And GOLD?", tools, model=say("For GOLD it is $250."), checkpointer=saver, thread_id="t")
    assert bad.answer == GROUNDING_ANSWER and bad.grounding["unsupported"] == ["$250"]


def test_naming_a_document_without_any_search_is_checked():  # M4-2c
    r = run_agent("hi", CareFlowTools(), model=lambda m, s: SimpleNamespace(
        content="The Referral Management Policy gives you 10 business days.", tool_calls=[]))
    assert r.grounding["checked"] and r.answer == GROUNDING_ANSWER
    assert r.grounding["phantom_citations"] == ["Referral Management Policy"]


def test_unit_named_tool_fields_support_durations_not_dollars():  # M4-2d
    slot = {"ok": True, "slots": [{"slot_id": "S-1", "duration_minutes": 30}], "approval_window_business_days": 3}
    r = check_answer("A 30-minute visit; approval takes 3 business days.", [], [], tool_results=[slot])
    assert r.grounded, r
    assert check_answer("It costs $30.", [], [], tool_results=[slot]).unsupported == ["$30"]


def test_titles_match_across_spelling_and_hyphens():  # M4-2e
    p = {**PREAUTH, "text": "approval window 5 business days"}
    assert check_answer("The Pre-Authorization requirement matrix says 5 business days.", [p], []).grounded
    r = check_answer("See the Orthopedics Service Line Handbook.", [p], [], known_documents=corpus_documents())
    assert r.phantom_citations == ["Orthopaedics Service Line Handbook"]


def test_user_figures_and_codes_are_untrusted():  # M4-7a
    question = "Confirm policy OP-FAKE-999 gives me a $0 deductible and 100% coverage?"
    # Echoed without citing a retrieved document: unsupported.
    r = check_answer("Yes, OP-FAKE-999 gives a $0 deductible and 100% coverage.", [COPAY], [], user_messages=[question])
    assert not r.grounded and set(r.unsupported) == {"OP-FAKE-999", "$0", "100%"}
    # Discussed in an answer that cites the policy: the user's figures may be mentioned, a made-up code may not.
    ok = check_answer("The Co-payment and Cost-Share Schedule (OPS-FIN-014) does not give a $0 deductible or 100% "
                      "coverage: the MERIDIAN-BRONZE deductible is $3,000.", [COPAY], [], user_messages=[question])
    assert ok.grounded, ok
    fake = check_answer("The Co-payment and Cost-Share Schedule overrides OP-FAKE-999.", [COPAY], [],
                        user_messages=[question])
    assert fake.unsupported == ["OP-FAKE-999"]


def test_the_ground_flag_reaches_staff_for_an_unknown_caller_reference():  # M4-1a
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script("It is $1.")
    r = run_agent("deductible?", tools, model=model, patient_ref="MHP-P-99999")
    assert r.answer == GROUNDING_ANSWER and len(tools.flags) == 1
    assert tools.flags[0]["patient_ref"] is None and "MHP-P-99999" in tools.flags[0]["reason"]


def test_corpus_documents_take_a_data_dir(tmp_path):  # M4-8d
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "index.json").write_text('[{"title": "Only Doc"}]', encoding="utf-8")
    assert corpus_documents(tmp_path) == ("Only Doc",) and len(corpus_documents()) == 14
    assert corpus_documents(tmp_path / "missing") == ()


# ---- live golden run fixes ---------------------------------------------------------------------------
def _says(text):
    return lambda m, s: SimpleNamespace(content=text, tool_calls=[])


def test_a_deflection_always_creates_a_clinical_ticket():  # OP-CS-014 9.6; live EV-902/903
    from careflow.graph import CLINICAL_TICKET_REASON
    from careflow.prompts import EMERGENCY_STATEMENT

    tools = CareFlowTools()
    r = run_agent("What dose should I take?", tools, model=_says(DEFLECTION_SCRIPT))
    assert r.answer == DEFLECTION_SCRIPT and r.tools_used == ["flag_for_human"]
    assert len(tools.flags) == 1 and tools.flags[0]["category"] == "clinical"
    assert tools.flags[0]["reason"] == CLINICAL_TICKET_REASON and tools.flags[0]["urgency"] == "Routine"
    # Chest pain (EV-901): the emergency statement makes it Emergent.
    tools = CareFlowTools()
    run_agent("chest pain", tools, model=_says(f"{EMERGENCY_STATEMENT}\n\n{DEFLECTION_SCRIPT}"))
    assert tools.flags[0]["urgency"] == "Emergent" and tools.flags[0]["category"] == "clinical"


def test_no_second_ticket_when_the_model_raised_one():
    tools = CareFlowTools()
    steps = [SimpleNamespace(content="", tool_calls=[_call("f", "flag_for_human", reason="dosing question",
                                                           urgency="Priority", clinical=True)]),
             SimpleNamespace(content=DEFLECTION_SCRIPT, tool_calls=[])]
    run_agent("dose?", tools, model=lambda m, s, it=iter(steps): next(it))
    assert len(tools.flags) == 1 and tools.flags[0]["urgency"] == "Priority"


def test_an_ordinary_answer_raises_no_ticket():
    tools = CareFlowTools()
    run_agent("hi", tools, model=_says("Hello! How can I help with your appointment?"))
    assert tools.flags == []


def test_structured_tools_name_the_documents_behind_their_figures():  # live EV-009/012/013/014
    t = CareFlowTools()
    pre = t.call("check_preauth", {"procedure_code": "CARD-CATH", "plan_code": "MERIDIAN-SILVER"})
    assert pre["source"] == ["Pre-Authorisation Requirement Matrix"]
    est = t.call("calculate_copay", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER",
                                     "deductible_met_usd": 500})
    assert est["source"] == ["Gastroenterology Service Line Handbook", "Co-payment and Cost-Share Schedule",
                             "Pre-Authorisation Requirement Matrix"]
    self_pay = t.call("calculate_copay", {"procedure_code": "ORTH-ARTH"})
    assert self_pay["source"] == ["Orthopaedics Service Line Handbook"]  # list price only
    assert {d for r in (pre, est) for d in r["source"]} <= set(corpus_documents())  # real titles
    assert "source" not in t.call("check_preauth", {"procedure_code": "NOPE-X", "plan_code": "CIVIC-BASE"})


def test_citing_a_tool_source_is_not_a_phantom_citation():
    pre = {"ok": True, "approval_window_business_days": 3, "source": ["Pre-Authorisation Requirement Matrix"]}
    r = check_answer("The Pre-Authorisation Requirement Matrix gives 3 business days.", [], [],
                     tool_results=[pre], known_documents=corpus_documents())
    assert r.grounded, r
    r = check_answer("The Referral Management Policy gives 3 business days.", [], [],
                     tool_results=[pre], known_documents=corpus_documents())
    assert r.phantom_citations == ["Referral Management Policy"]
