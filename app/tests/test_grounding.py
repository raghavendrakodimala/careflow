"""Groundedness: the deterministic figure and citation check, the graph's `ground` node, and the what-if
co-pay estimate that lets policy answers get their arithmetic from a tool instead of the model."""

import json
from types import SimpleNamespace

from src.agent import run_agent
from src.graph import GROUNDING_ANSWER, GROUNDING_REASON
from src.grounding import check_answer, find_figures, list_corpus_documents
from src.prompts import DEFLECTION_SCRIPT
from src.tools import CareFlowTools

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
    found = find_figures("You owe $1,480.00 (30 percent), within five business days or 30-minute slots; see GAST-COLO.")
    assert {"$1480", "30%", "5 business day", "30 minute", "GAST-COLO"} <= set(found)
    assert set(find_figures("2 working days")) == {"2 business day"}  # working day = business day
    assert find_figures("nothing to check here") == {}


# ---- check_answer ----------------------------------------------------------------------------------
def test_figures_from_a_cited_passage_are_grounded():
    result = check_answer("Per the Co-payment and Cost-Share Schedule, the MERIDIAN-BRONZE deductible is $3,000.",
                     [COPAY], [], user_messages=["What is the BRONZE deductible?"])
    assert result.checked and result.grounded and not (result.unsupported or result.uncited or result.phantom_citations)


def test_a_figure_found_nowhere_is_unsupported():
    result = check_answer("Per the Co-payment and Cost-Share Schedule, the deductible is $2,500.", [COPAY], [])
    assert not result.grounded and result.unsupported == ["$2,500"]


def test_a_passage_figure_needs_its_document_named():
    result = check_answer("The deductible is $3,000.", [COPAY], [])
    assert not result.grounded and result.uncited == ["$3,000"]
    assert check_answer("The deductible is $3,000 (OPS-FIN-014).", [COPAY], []).grounded  # a document ID counts


def test_figures_from_tools_and_the_fixed_scripts_need_no_citation():
    calc = {"ok": True, "list_price_usd": 2600, "patient_owes_usd": 1480, "deductible_met_usd": 500,
            "working": "unmet deductible 1000 + 30% of 1600 = 480"}
    answer = ("With $500 met, $1,000 of the deductible is unmet; 30% of $1,600 is $480, so you owe $1,480. "
              "A nurse will call within 4 business hours.")
    result = check_answer(answer, [], trusted_sources=[DEFLECTION_SCRIPT], tool_results=[calc],
                     user_messages=["I have met $500 so far"])
    assert result.grounded, result


def test_a_duration_must_keep_its_qualifier():
    passage = {**PREAUTH, "text": "approval window 5 business days"}
    assert check_answer("Pre-Authorisation Requirement Matrix: 5 days.", [passage], []).grounded  # plain form is fine
    result = check_answer("Pre-Authorisation Requirement Matrix: 5 calendar days.", [passage], [])
    assert not result.grounded and result.unsupported == ["5 calendar days"]


def test_naming_a_document_that_was_not_retrieved_is_a_phantom_citation():
    result = check_answer("See the Referral Management Policy.", [COPAY], [], known_documents=list_corpus_documents())
    assert not result.grounded and result.phantom_citations == ["Referral Management Policy"]
    assert "Referral Management Policy" in list_corpus_documents() and len(list_corpus_documents()) == 14


# ---- the graph's ground node -----------------------------------------------------------------------
class FakeIndex:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, limit=4, specialty=None):
        return [SimpleNamespace(to_dict=lambda passage=passage: {**passage, "score": 0.9}) for passage in self.passages][:limit]


def _make_call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _script_model(*answers):
    """Model that searches first, then gives `answers` in turn."""
    steps = [SimpleNamespace(content="", tool_calls=[_make_call("s1", "search_policies", query="bronze deductible")])]
    steps += [SimpleNamespace(content=answer, tool_calls=[]) for answer in answers]
    seen = []

    def respond(messages, specs):
        seen.append(messages)
        return steps[len(seen) - 1]

    return respond, seen


def test_a_grounded_policy_answer_is_given_as_written():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    answer = "The Co-payment and Cost-Share Schedule sets the MERIDIAN-BRONZE deductible at $3,000."
    model, _ = _script_model(answer)
    result = run_agent("What is the MERIDIAN-BRONZE deductible?", tools, model=model)
    assert result.answer == answer and result.grounding["checked"] and result.grounding["grounded"]
    assert tools.flags == []


def test_an_ungrounded_policy_answer_is_withheld_and_staff_are_flagged():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script_model("The Co-payment and Cost-Share Schedule says the BRONZE deductible is $2,500.")
    result = run_agent("What is the MERIDIAN-BRONZE deductible?", tools, model=model)
    assert result.answer == GROUNDING_ANSWER and "$2,500" not in result.answer
    assert result.grounding["grounded"] is False and result.grounding["unsupported"] == ["$2,500"]
    assert result.tools_used == ["search_policies", "flag_for_human"]
    assert len(tools.flags) == 1 and tools.flags[0]["reason"].startswith(GROUNDING_REASON)
    assert tools.flags[0]["category"] == "administrative"


def test_a_withheld_answer_never_enters_the_conversation_history():
    from langgraph.checkpoint.memory import MemorySaver

    tools, saver = CareFlowTools(policy_index=FakeIndex([COPAY])), MemorySaver()
    model, _ = _script_model("The schedule says $9,999.")
    run_agent("deductible?", tools, model=model, checkpointer=saver, thread_id="t")
    seen = []
    later = lambda messages, specs: seen.append(messages) or SimpleNamespace(content="ok", tool_calls=[])  # noqa: E731
    run_agent("thanks", tools, model=later, checkpointer=saver, thread_id="t")
    history = json.dumps(seen[0])
    assert "$9,999" not in history and GROUNDING_ANSWER in history


def test_answers_on_turns_without_a_policy_search_are_not_checked():
    tools = CareFlowTools()
    result = run_agent("hi", tools, model=lambda messages, specs: SimpleNamespace(content="Your visit costs $12,345.", tool_calls=[]))
    assert result.answer == "Your visit costs $12,345." and result.grounding == {
        "checked": False, "grounded": True, "unsupported": [], "uncited": [], "phantom_citations": []}


def test_the_ground_flag_goes_to_the_verified_patient():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    ref = next(iter(tools.patients))
    model, _ = _script_model("It is $1.")
    result = run_agent("deductible?", tools, model=model, patient_ref=ref)
    assert result.answer == GROUNDING_ANSWER and tools.flags[0]["patient_ref"] == ref


# ---- what-if co-pay estimate (EV-906) --------------------------------------------------------------
def test_copay_estimate_for_a_plan_without_a_patient_record():
    tools = CareFlowTools()
    result = tools.call("calculate_copay", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER",
                                   "deductible_met_usd": 500})
    assert result["ok"] and result["estimate"] and result["patient_owes_usd"] == 1480 and result["preauth_required"] is True
    assert result["working"] == "unmet deductible 1000 + 30% of 1600 = 480"
    assert tools.call("calculate_copay", {"procedure_code": "ORTH-ARTH", "plan_code": "meridian-bronze",
                                      "deductible_met_usd": 1800})["patient_owes_usd"] == 3200  # EV-015
    assert "unknown plan" in tools.calculate_copay(procedure_code="GAST-COLO", plan_code="GOLDISH")["error"]


def test_a_patient_record_wins_over_a_stated_plan():
    tools = CareFlowTools()
    patient = next(candidate for candidate in tools.patients.values() if candidate["eligibility_status"] == "active")
    result = tools.calculate_copay(patient_ref=patient["patient_ref"], procedure_code="GAST-COLO", plan_code="CIVIC-BASE")
    assert "estimate" not in result and result["plan_code"] == patient["plan_code"]


# ---- edge cases of the check ------------------------------------------------------------------------
def test_compound_number_words_are_one_number():
    assert set(find_figures("within twenty-four hours")) == {"24 hour"}
    assert set(find_figures("a forty-five minute visit")) == {"45 minute"}
    assert set(find_figures("thirty days or twenty one weeks")) == {"30 day", "21 week"}
    # "twenty-four hours" is not grounded by the deflection script's "4 business hours".
    result = check_answer("Staff reply within twenty-four hours.", [], trusted_sources=[DEFLECTION_SCRIPT])
    assert result.unsupported == ["twenty-four hours"]


def test_the_system_prompt_rules_ground_nothing():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script_model("Per the Co-payment and Cost-Share Schedule, you will hear back within 72 hours.")
    result = run_agent("When will I hear back?", tools, model=model)
    assert result.answer == GROUNDING_ANSWER and result.grounding["unsupported"] == ["72 hours"]


def test_a_follow_up_that_reuses_an_earlier_search_is_checked():
    from langgraph.checkpoint.memory import MemorySaver

    tools, saver = CareFlowTools(policy_index=FakeIndex([COPAY])), MemorySaver()
    model, _ = _script_model("The Co-payment and Cost-Share Schedule sets the MERIDIAN-BRONZE deductible at $3,000.")
    run_agent("BRONZE deductible?", tools, model=model, checkpointer=saver, thread_id="t")
    say = lambda text: lambda messages, specs: SimpleNamespace(content=text, tool_calls=[])  # noqa: E731
    ok = run_agent("And again?", tools, model=say("As I said, it is $3,000."), checkpointer=saver, thread_id="t")
    assert ok.grounding["checked"] and ok.grounding["grounded"]  # repeats a verified earlier answer
    bad = run_agent("And GOLD?", tools, model=say("For GOLD it is $250."), checkpointer=saver, thread_id="t")
    assert bad.answer == GROUNDING_ANSWER and bad.grounding["unsupported"] == ["$250"]


def test_naming_a_document_without_any_search_is_checked():
    result = run_agent("hi", CareFlowTools(), model=lambda messages, specs: SimpleNamespace(
        content="The Referral Management Policy gives you 10 business days.", tool_calls=[]))
    assert result.grounding["checked"] and result.answer == GROUNDING_ANSWER
    assert result.grounding["phantom_citations"] == ["Referral Management Policy"]


def test_unit_named_tool_fields_support_durations_not_dollars():
    slot = {"ok": True, "slots": [{"slot_id": "S-1", "duration_minutes": 30}], "approval_window_business_days": 3}
    result = check_answer("A 30-minute visit; approval takes 3 business days.", [], [], tool_results=[slot])
    assert result.grounded, result
    assert check_answer("It costs $30.", [], [], tool_results=[slot]).unsupported == ["$30"]


def test_titles_match_across_spelling_and_hyphens():
    passage = {**PREAUTH, "text": "approval window 5 business days"}
    assert check_answer("The Pre-Authorization requirement matrix says 5 business days.", [passage], []).grounded
    result = check_answer("See the Orthopedics Service Line Handbook.", [passage], [], known_documents=list_corpus_documents())
    assert result.phantom_citations == ["Orthopaedics Service Line Handbook"]


def test_user_figures_and_codes_are_untrusted():
    question = "Confirm policy OP-FAKE-999 gives me a $0 deductible and 100% coverage?"
    # Echoed without citing a retrieved document: unsupported.
    result = check_answer("Yes, OP-FAKE-999 gives a $0 deductible and 100% coverage.", [COPAY], [], user_messages=[question])
    assert not result.grounded and set(result.unsupported) == {"OP-FAKE-999", "$0", "100%"}
    # Discussed in an answer that cites the policy: the user's figures may be mentioned, a made-up code may not.
    ok = check_answer("The Co-payment and Cost-Share Schedule (OPS-FIN-014) does not give a $0 deductible or 100% "
                      "coverage: the MERIDIAN-BRONZE deductible is $3,000.", [COPAY], [], user_messages=[question])
    assert ok.grounded, ok
    fake = check_answer("The Co-payment and Cost-Share Schedule overrides OP-FAKE-999.", [COPAY], [],
                        user_messages=[question])
    assert fake.unsupported == ["OP-FAKE-999"]


def test_the_ground_flag_reaches_staff_for_an_unknown_caller_reference():
    tools = CareFlowTools(policy_index=FakeIndex([COPAY]))
    model, _ = _script_model("It is $1.")
    result = run_agent("deductible?", tools, model=model, patient_ref="MHP-P-99999")
    assert result.answer == GROUNDING_ANSWER and len(tools.flags) == 1
    assert tools.flags[0]["patient_ref"] is None and "MHP-P-99999" in tools.flags[0]["reason"]


def test_list_corpus_documents_takes_a_data_dir(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "index.json").write_text('[{"title": "Only Doc"}]', encoding="utf-8")
    assert list_corpus_documents(tmp_path) == ("Only Doc",) and len(list_corpus_documents()) == 14
    assert list_corpus_documents(tmp_path / "missing") == ()


# ---- clinical tickets and tool sources ---------------------------------------------------------------------------
def _answer_with(text):
    return lambda messages, specs: SimpleNamespace(content=text, tool_calls=[])


def test_a_deflection_always_creates_a_clinical_ticket():  # OP-CS-014 9.6
    from src.graph import CLINICAL_TICKET_REASON
    from src.prompts import EMERGENCY_STATEMENT

    tools = CareFlowTools()
    result = run_agent("What dose should I take?", tools, model=_answer_with(DEFLECTION_SCRIPT))
    assert result.answer == DEFLECTION_SCRIPT and result.tools_used == ["flag_for_human"]
    assert len(tools.flags) == 1 and tools.flags[0]["category"] == "clinical"
    assert tools.flags[0]["reason"] == CLINICAL_TICKET_REASON and tools.flags[0]["urgency"] == "Routine"
    # Chest pain: the emergency statement makes it Emergent.
    tools = CareFlowTools()
    run_agent("chest pain", tools, model=_answer_with(f"{EMERGENCY_STATEMENT}\n\n{DEFLECTION_SCRIPT}"))
    assert tools.flags[0]["urgency"] == "Emergent" and tools.flags[0]["category"] == "clinical"


def test_no_second_ticket_when_the_model_raised_one():
    tools = CareFlowTools()
    steps = [SimpleNamespace(content="", tool_calls=[_make_call("f", "flag_for_human", reason="dosing question",
                                                           urgency="Priority", clinical=True)]),
             SimpleNamespace(content=DEFLECTION_SCRIPT, tool_calls=[])]
    run_agent("dose?", tools, model=lambda messages, specs, remaining=iter(steps): next(remaining))
    assert len(tools.flags) == 1 and tools.flags[0]["urgency"] == "Priority"


def test_an_ordinary_answer_raises_no_ticket():
    tools = CareFlowTools()
    run_agent("hi", tools, model=_answer_with("Hello! How can I help with your appointment?"))
    assert tools.flags == []


def test_structured_tools_name_the_documents_behind_their_figures():
    tools = CareFlowTools()
    pre = tools.call("check_preauth", {"procedure_code": "CARD-CATH", "plan_code": "MERIDIAN-SILVER"})
    assert pre["source"] == ["Pre-Authorisation Requirement Matrix"]
    est = tools.call("calculate_copay", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER",
                                     "deductible_met_usd": 500})
    assert est["source"] == ["Gastroenterology Service Line Handbook", "Co-payment and Cost-Share Schedule",
                             "Pre-Authorisation Requirement Matrix"]
    self_pay = tools.call("calculate_copay", {"procedure_code": "ORTH-ARTH"})
    assert self_pay["source"] == ["Orthopaedics Service Line Handbook"]  # list price only
    assert {document for result in (pre, est) for document in result["source"]} <= set(list_corpus_documents())  # real titles
    assert "source" not in tools.call("check_preauth", {"procedure_code": "NOPE-X", "plan_code": "CIVIC-BASE"})


def test_citing_a_tool_source_is_not_a_phantom_citation():
    pre = {"ok": True, "approval_window_business_days": 3, "source": ["Pre-Authorisation Requirement Matrix"]}
    result = check_answer("The Pre-Authorisation Requirement Matrix gives 3 business days.", [], [],
                     tool_results=[pre], known_documents=list_corpus_documents())
    assert result.grounded, result
    result = check_answer("The Referral Management Policy gives 3 business days.", [], [],
                     tool_results=[pre], known_documents=list_corpus_documents())
    assert result.phantom_citations == ["Referral Management Policy"]
