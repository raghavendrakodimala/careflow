"""Invariants: properties that must hold for every input, checked over the full product of small domains
or over generated cases from a fixed random seed (so a failure is reproducible). No model, no network;
the stores are in memory."""

import copy
import itertools
import json
import random

import pytest

from src.domain import policies
from src.guardrails import GuardResult, check_input, check_output, check_rules
from src.state import REDACT, add_messages
from src.tool_contracts import TOOL_SPECS, TOOLS, TOOLS_BY_NAME
from src.tools import CareFlowTools

SEED = 20260801
BANDS = policies.URGENCY_ORDER
RANK = {band: index for index, band in enumerate(BANDS)}
NOT_BANDS = (None, "", "bogus", "routine", "EMERGENT")  # ignored by find_highest_band


# ---- bands ---------------------------------------------------------------------------------------------
def test_find_highest_band_is_commutative_idempotent_and_associative():
    domain = (*BANDS, *NOT_BANDS)
    for first, second in itertools.product(domain, repeat=2):
        assert policies.find_highest_band(first, second) == policies.find_highest_band(second, first)
        assert policies.find_highest_band(first, first) == policies.find_highest_band(first)
    for first, second, third in itertools.product(domain, repeat=3):
        whole = policies.find_highest_band(first, second, third)
        assert whole == policies.find_highest_band(policies.find_highest_band(first, second), third) == policies.find_highest_band(third, first, second)


def test_find_highest_band_is_a_known_band_never_below_any_known_input():
    domain = (*BANDS, *NOT_BANDS)
    assert policies.find_highest_band() == "Routine"
    for bands in itertools.chain(itertools.product(domain, repeat=1), itertools.product(domain, repeat=2),
                                 itertools.product(domain, repeat=3)):
        result = policies.find_highest_band(*bands)
        known = [band for band in bands if band in RANK]
        assert result in RANK
        assert all(RANK[result] >= RANK[band] for band in known)
        assert result in known if known else result == "Routine"  # never a band nobody gave
        assert not any(policies.is_lower_band(band, result) for band in known)


def test_lowering_a_band_is_refused_from_emergent_and_needs_a_note_otherwise():
    for current, proposed in itertools.product(BANDS, repeat=2):
        lower = RANK[proposed] < RANK[current]
        assert policies.is_lower_band(current, proposed) is lower
        for note in (None, "", "   ", "Seen by the nurse"):
            problem = policies.validate_reband(current, proposed, note)
            if not lower:
                assert problem is None  # the same band or a higher one is always allowed
            elif current == "Emergent":
                assert problem  # whatever the note says
            else:
                assert bool(problem) is not bool((note or "").strip())
        assert policies.validate_reband(current, None, None) is None


# ---- triage route --------------------------------------------------------------------------------------
ROUTE_DOMAIN = dict(
    seeks_clinical_advice=(False, True),
    urgency=BANDS,
    patient_known=(False, True),
    specialty=(None, "Dermatology", "Cardiology", "ENT"),  # none, direct, sign-off, in neither list
    has_referral=(False, True),
    referral_status=(None, "open", "accepted", "scheduled", "rejected"),
    referral_rejected=(False, True),
    preauth_status=(None, "not_required", "approved", "pending", "denied"),
    referral_overdue=(False, True),
    repeat_priority=(False, True),
)
ROUTE_RANK = {policies.ROUTE_AUTO: 0, policies.ROUTE_HUMAN: 1, policies.ROUTE_REFUSE: 2}


def _generate_route_cases():
    keys = list(ROUTE_DOMAIN)
    for values in itertools.product(*ROUTE_DOMAIN.values()):
        yield dict(zip(keys, values, strict=True))


def test_decide_triage_route_is_never_auto_for_an_emergent_urgent_or_clinical_request():
    seen = 0
    for case in _generate_route_cases():
        route, _ = policies.decide_triage_route(**case)
        seen += 1
        assert route in ROUTE_RANK
        if case["urgency"] in ("Emergent", "Urgent") or case["seeks_clinical_advice"]:
            assert route != policies.ROUTE_AUTO, case
        # Refuse exactly when the request is clinical or Emergent, whatever the administrative facts are.
        assert (route == policies.ROUTE_REFUSE) is (case["seeks_clinical_advice"] or case["urgency"] == "Emergent"), case
        if case["urgency"] == "Urgent" and not case["seeks_clinical_advice"]:
            assert route == policies.ROUTE_HUMAN, case
    assert seen == 25_600  # the whole product was walked


def test_decide_triage_route_gives_reasons_exactly_when_staff_or_a_clinician_are_needed():
    for case in _generate_route_cases():
        route, reasons = policies.decide_triage_route(**case)
        assert (route == policies.ROUTE_AUTO) is (reasons == []), case
        assert all(isinstance(reason, str) and reason for reason in reasons) and len(set(reasons)) == len(reasons), case


def test_adding_a_risk_never_moves_a_request_to_a_more_automatic_route():
    worse = {"seeks_clinical_advice": True, "patient_known": False, "referral_overdue": True, "repeat_priority": True,
             "referral_rejected": True}
    for case in _generate_route_cases():
        base = ROUTE_RANK[policies.decide_triage_route(**case)[0]]
        for key, value in worse.items():
            if case[key] != value:
                assert ROUTE_RANK[policies.decide_triage_route(**{**case, key: value})[0]] >= base, (case, key)
        for higher in BANDS[RANK[case["urgency"]] + 1:]:
            assert ROUTE_RANK[policies.decide_triage_route(**{**case, "urgency": higher})[0]] >= base, (case, higher)


def test_an_unidentified_patient_or_missing_specialty_is_never_handled_automatically():
    for case in _generate_route_cases():
        if not case["patient_known"] or not case["specialty"]:
            assert policies.decide_triage_route(**case)[0] != policies.ROUTE_AUTO, case


# ---- the messages reducer ------------------------------------------------------------------------------
def _build_conversation(rng, turns, tag=""):
    """A stored conversation of `turns` finished turns: user, tool rounds (some with text beside the call),
    then the final answer. Every text is unique, so a redacted one can be searched for afterwards."""
    messages = []
    for turn in range(turns):
        messages.append({"role": "user", "content": f"{tag}user-{turn}"})
        for round_index in range(rng.randint(0, 3)):
            call = {"id": f"{tag}c{turn}-{round_index}", "type": "function", "function": {"name": "check_eligibility", "arguments": "{}"}}
            aside = f"{tag}aside-{turn}-{round_index}" if rng.random() < 0.5 else ""
            messages.append({"role": "assistant", "content": aside, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"ok": True, "n": round_index})})
        messages.append({"role": "assistant", "content": f"{tag}answer-{turn}"})
    return messages


def _find_turn_starts(messages):
    return [index for index, message in enumerate(messages) if message["role"] == "user"]


def test_appending_messages_is_associative_and_keeps_order():
    rng = random.Random(SEED)
    for _ in range(50):
        first, second, third = (_build_conversation(rng, rng.randint(0, 3), tag) for tag in "abc")
        left = add_messages(add_messages(first, second), third)
        assert left == add_messages(first, add_messages(second, third)) == add_messages(first, second + third) == first + second + third
    assert add_messages(None, None) == [] and add_messages([], None) == [] and add_messages(None, []) == []


def test_the_reducer_never_changes_the_lists_or_messages_it_is_given():
    rng = random.Random(SEED + 1)
    for _ in range(50):
        old = _build_conversation(rng, rng.randint(1, 4))
        new = [{"role": REDACT, "content": "[withheld]", "count": rng.randint(1, 3)}, {"role": "user", "content": "next"}]
        old_before, new_before = copy.deepcopy(old), copy.deepcopy(new)
        add_messages(old, new)
        assert old == old_before and new == new_before


def test_a_redaction_keeps_the_length_the_user_and_tool_messages_and_every_earlier_turn():
    rng = random.Random(SEED + 2)
    for _ in range(200):
        turns = rng.randint(1, 6)
        old = _build_conversation(rng, turns)
        count = rng.randint(1, turns + 2)  # sometimes more answers than the conversation holds
        out = add_messages(old, [{"role": REDACT, "content": "[withheld]", "count": count}])

        assert len(out) == len(old)
        assert [message["role"] for message in out] == [message["role"] for message in old]
        for before, after in zip(old, out, strict=True):
            if before["role"] in ("user", "tool"):
                assert after == before
            assert after.get("tool_calls") == before.get("tool_calls")  # a call is never dropped or altered
        starts = _find_turn_starts(old)
        cleaned = min(count, turns)
        first_cleaned = starts[turns - cleaned]
        assert out[:first_cleaned] == old[:first_cleaned]  # turns before the redacted ones are untouched


def test_a_redaction_leaves_none_of_the_redacted_turns_assistant_text():
    rng = random.Random(SEED + 3)
    for _ in range(200):
        turns = rng.randint(1, 6)
        old = _build_conversation(rng, turns)
        count = rng.randint(1, turns)
        out = add_messages(old, [{"role": REDACT, "content": "[withheld]", "count": count}])

        first_cleaned = _find_turn_starts(old)[turns - count]
        written = [message["content"] for message in old[first_cleaned:] if message["role"] == "assistant" and message["content"]]
        assert len([written_text for written_text in written if written_text.startswith("answer-")]) == count
        kept = json.dumps(out)
        for text in written:  # the final answers and the text beside tool calls, in every redacted turn
            assert text not in kept, text
        answers = [message for message in out[first_cleaned:] if message["role"] == "assistant" and not message.get("tool_calls")]
        assert [message["content"] for message in answers] == ["[withheld]"] * count
        # Redacting again changes nothing more.
        assert add_messages(out, [{"role": REDACT, "content": "[withheld]", "count": count}]) == out


def test_a_redaction_on_an_empty_or_answerless_conversation_changes_nothing():
    redact = [{"role": REDACT, "content": "[withheld]", "count": 2}]
    assert add_messages([], redact) == [] and add_messages(None, redact) == []
    only_user = [{"role": "user", "content": "hello"}]
    assert add_messages(only_user, redact) == only_user
    assert add_messages(only_user, [{"role": REDACT}]) == only_user  # no text and no count given


# ---- tool contracts ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference():
    """Read-only reference data for building arguments (never used for a call that writes)."""
    tools = CareFlowTools()
    yield tools
    tools.store.close()


def _build_valid_arguments(tools):
    """A set of arguments each tool accepts, keyed by tool name."""
    ref = sorted(tools.patients)[0]
    slot = sorted(candidate["slot_id"] for candidate in tools.slots.values() if candidate["status"] == "open")[0]
    patient = tools.patients[ref]
    return {
        "check_eligibility": {"patient_ref": ref},
        "find_appointment_slots": {"specialty": "ENT", "limit": 3},
        "book_appointment": {"patient_ref": ref, "slot_id": slot},
        "check_preauth": {"procedure_code": "CARD-CONS", "plan_code": "MERIDIAN-SILVER"},
        "calculate_copay": {"procedure_code": "CARD-CONS", "patient_ref": ref},
        "flag_for_human": {"reason": "Caller asked for a call back"},
        "find_patient": {"given_name": patient["given_name"], "family_name": patient["family_name"],
                         "member_number": patient["member_number"]},
        "search_policies": {"query": "pre-authorisation for an MRI"},
    }


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_the_spec_the_model_sees_has_exactly_the_input_models_fields(tool):
    spec = next(candidate for candidate in TOOL_SPECS if candidate["function"]["name"] == tool.name)
    parameters = spec["function"]["parameters"]
    fields = tool.input_model.model_fields
    assert set(parameters["properties"]) == set(fields)
    assert sorted(parameters["required"]) == sorted(name for name, field_info in fields.items() if field_info.is_required())
    assert parameters["type"] == "object" and spec["function"]["description"] == tool.description
    assert callable(getattr(CareFlowTools, tool.name))  # the dispatcher finds a method of the same name
    for prop in parameters["properties"].values():  # the plain subset every provider accepts
        assert "title" not in prop and "default" not in prop
        assert all(option.get("type") != "null" for option in prop.get("anyOf", []))


def test_every_tool_has_valid_arguments_in_this_file(reference):
    assert set(_build_valid_arguments(reference)) == set(TOOLS_BY_NAME)  # a new tool must be added to the tables here


@pytest.mark.parametrize("name", sorted(TOOLS_BY_NAME))
def test_an_unknown_argument_is_rejected_before_the_tool_runs(name, reference):
    tools = CareFlowTools()
    before = {table: tools.store.list_rows(table) for table in ("appointments", "appointment_slots", "patient_interactions")}
    for arguments in ({**_build_valid_arguments(reference)[name], "surprise": 1}, {"surprise": 1},
                      json.dumps({**_build_valid_arguments(reference)[name], "patient_reff": "MHP-P-00000"})):
        result = tools.call(name, arguments)
        assert result["ok"] is False and result["error"].startswith("bad arguments"), result
        assert "surprise" in result["error"] or "patient_reff" in result["error"]
    assert tools.flags == [] and {table: tools.store.list_rows(table) for table in before} == before
    tools.store.close()


# Whole-argument values that are not what a tool expects.
MALFORMED = [
    None, "", " ", "{}", "null", "[]", "[1, 2]", '"text"', "123", "not json", "{", '{"patient_ref":', "{'single': 1}",
    '{"patient_ref": {"$ne": null}}', '{"a": 1}{"b": 2}', "\x00", "x" * 200_000, 0, 1, -1, 3.5, True, False, [], [1, 2],
    ("a", "b"), {"": None}, {1: 2}, {"patient_ref": None}, b"{}", {"nested": {"deep": [1, {"x": None}]}},
]


@pytest.mark.parametrize("name", [*sorted(TOOLS_BY_NAME), "no_such_tool", "", "__init__"])
def test_the_dispatcher_never_raises_for_malformed_arguments(name):
    tools = CareFlowTools()
    for arguments in MALFORMED:
        result = tools.call(name, arguments)
        assert isinstance(result, dict) and isinstance(result.get("ok"), bool), (name, arguments, result)
        if not result["ok"]:
            assert isinstance(result.get("error"), str) and result["error"]
    tools.store.close()


# Values tried in every field of every tool, alone and in random combinations.
NASTY = [
    None, "", " ", "\t\n", "x" * 50_000, "0", "-1", -1, 0, 1, 2**63, -(2**63), 1.5, -0.0, float("inf"), float("nan"),
    True, False, [], {}, ["a"], {"a": 1}, "'; DROP TABLE patients; --", '" OR ""="', "\x00", "a\x00b", "%s%n%x",
    "../../etc/passwd", "MHP-P-00000", " mhp-p-00000 ", "MHP-SL-00000", "2026-13-45", "0000-00-00", "2026-08-05",
    "99/99/9999", "１２３", "‮abc", "​", "😀" * 200, "null", "true", "Routine", "Emergent", "ENT", "in_person",
]


def _dump_json_or_none(arguments):
    try:
        return json.dumps(arguments)  # NaN and Infinity are written as bare tokens: the parser must cope
    except (TypeError, ValueError):
        return None


def _assert_tool_result(result, context):
    assert isinstance(result, dict) and isinstance(result.get("ok"), bool), context
    json.dumps(result)  # what goes back to the model must serialise


@pytest.mark.parametrize("name", sorted(TOOLS_BY_NAME))
def test_no_single_bad_field_makes_a_tool_raise(name, reference):
    tools = CareFlowTools()
    base = _build_valid_arguments(reference)[name]
    for field in TOOLS_BY_NAME[name].input_model.model_fields:
        for value in NASTY:
            arguments = {**base, field: value}
            _assert_tool_result(tools.call(name, arguments), (name, field, value))
            as_json = _dump_json_or_none(arguments)
            if as_json is not None:
                _assert_tool_result(tools.call(name, as_json), (name, field, value, "json"))
    assert len(tools.store.list_rows("patients")) == len(reference.patients)  # and no table was damaged
    tools.store.close()


@pytest.mark.parametrize("name", sorted(TOOLS_BY_NAME))
def test_no_random_combination_of_bad_fields_makes_a_tool_raise(name, reference):
    rng = random.Random(f"{SEED}-{name}")
    tools = CareFlowTools()
    base = _build_valid_arguments(reference)[name]
    fields = list(TOOLS_BY_NAME[name].input_model.model_fields)
    for _ in range(150):
        arguments = {field: rng.choice(NASTY) for field in rng.sample(fields, rng.randint(1, len(fields)))}
        if rng.random() < 0.5:
            arguments = {**base, **arguments}
        _assert_tool_result(tools.call(name, arguments), (name, arguments))
    assert {"patients", "appointments", "appointment_slots", "staff_flags"} <= tools.store.list_tables()
    tools.store.close()


@pytest.mark.parametrize("earliest", ["0001-01-01T00:00:00+14:00", "9999-12-31T23:59:59-14:00"])
def test_an_out_of_range_date_with_an_offset_is_an_error_result_not_an_exception(earliest):
    tools = CareFlowTools()
    try:
        result = tools.call("find_appointment_slots", json.dumps({"specialty": "ENT", "earliest_date": earliest}))
    finally:
        tools.store.close()
    assert result["ok"] is False and "earliest_date" in result["error"]


def test_a_lone_surrogate_in_dict_arguments_is_an_error_result_not_an_exception(reference):
    tools = CareFlowTools()
    try:
        result = tools.call("book_appointment", {"patient_ref": sorted(reference.patients)[0], "slot_id": "\ud800"})
    finally:
        tools.store.close()
    assert result["ok"] is False


# ---- money ---------------------------------------------------------------------------------------------
def _list_deductibles(plan):
    full = plan["annual_deductible_usd"]
    return sorted({0, 1, 499.5, full // 2, full - 1, full, full + 1, 10**9})  # ascending


def test_a_plan_estimate_is_never_negative_and_never_above_the_list_price(reference):
    tools, checked = reference, 0
    for plan, proc, telehealth in itertools.product(tools.plans.values(), tools.procedures.values(), (False, True)):
        price, owed = proc["list_price_usd"], []
        for met in _list_deductibles(plan):
            result = tools.call("calculate_copay", {"procedure_code": proc["procedure_code"], "plan_code": plan["plan_code"],
                                               "deductible_met_usd": met, "telehealth": telehealth})
            where = (plan["plan_code"], proc["procedure_code"], met, telehealth)
            assert result["ok"] and result["estimate"] and result["list_price_usd"] == price, where
            if result["covered"]:
                assert 0 <= result["patient_owes_usd"] <= price, (where, result)
                # Whole dollars in, whole dollars out (a fractional deductible may leave a fraction).
                assert isinstance(result["patient_owes_usd"], int) or not float(met).is_integer(), (where, result)
                owed.append(result["patient_owes_usd"])
            else:  # insurance cannot be used: the self-pay price, which is the list price
                assert result["patient_owes_usd"] is None and result["self_pay_usd"] == price, (where, result)
            checked += 1
        # Having met more of the deductible never makes the patient owe more.
        assert owed == sorted(owed, reverse=True), (plan["plan_code"], proc["procedure_code"], owed)
    assert checked == sum(len(_list_deductibles(listed_plan)) for listed_plan in tools.plans.values()) * len(tools.procedures) * 2


def _pick_representative_patients(tools):
    """One patient per (plan, coverage status), plus those with the least and the most deductible met."""
    by_kind = {}
    for ref in sorted(tools.patients):
        patient = tools.patients[ref]
        by_kind.setdefault((patient["plan_code"], patient["eligibility_status"]), ref)
    everyone = sorted(tools.patients.values(), key=lambda candidate: (candidate["deductible_met_usd"], candidate["patient_ref"]))
    return sorted({*by_kind.values(), everyone[0]["patient_ref"], everyone[-1]["patient_ref"]})


def test_a_patients_quote_is_within_bounds_or_is_the_self_pay_price(reference):
    tools = reference
    statuses = set()
    for ref, proc in itertools.product(_pick_representative_patients(tools), tools.procedures.values()):
        patient, price = tools.patients[ref], proc["list_price_usd"]
        statuses.add(patient["eligibility_status"])
        result = tools.call("calculate_copay", {"patient_ref": ref, "procedure_code": proc["procedure_code"]})
        where = (ref, proc["procedure_code"], result)
        assert result["ok"] and result["list_price_usd"] == price, where
        rule = tools.preauth.get((proc["procedure_code"], patient["plan_code"]))
        usable = patient["eligibility_status"] == "active" and not (rule is not None and not rule["covered"])
        if usable:
            assert isinstance(result["patient_owes_usd"], int) and 0 <= result["patient_owes_usd"] <= price, where
            # The same figure as the what-if estimate for the patient's own plan and deductible: one calculator.
            estimate = tools.call("calculate_copay", {
                "procedure_code": proc["procedure_code"], "plan_code": patient["plan_code"],
                "deductible_met_usd": patient["deductible_met_usd"]})
            assert estimate["patient_owes_usd"] == result["patient_owes_usd"], where
        else:
            assert result["patient_owes_usd"] is None and result["self_pay_usd"] == price and result["self_pay_available"], where
            assert "working" not in result and "basis" not in result, where  # no insurance figure of any kind
    assert {"active", "lapsed", "pending_verification"} <= statuses  # the seed data exercises every branch


def test_without_usable_insurance_the_price_is_always_the_list_price(reference):
    tools = reference
    for proc in tools.procedures.values():
        code, price = proc["procedure_code"], proc["list_price_usd"]
        assert policies.get_self_pay_price(proc) == price and price > 0
        no_insurance = tools.call("calculate_copay", {"procedure_code": code})
        assert no_insurance["ok"] and no_insurance["patient_owes_usd"] is None and no_insurance["self_pay_usd"] == price
        unknown = tools.call("calculate_copay", {"procedure_code": code, "patient_ref": "MHP-P-99999"})
        assert unknown["ok"] is False and unknown["self_pay_usd"] == price


# ---- booking -------------------------------------------------------------------------------------------
def _assert_booking_invariants(tools, initial_slots, booked_ok):
    appointments = tools.store.list_rows("appointments")
    slots = tools.slots
    made = [appointment for appointment in appointments if appointment["slot_id"]]  # rows written by a booking (seeded rows have no slot_id)
    assert len({appointment["appointment_id"] for appointment in appointments}) == len(appointments)
    assert len({appointment["slot_id"] for appointment in made}) == len(made), "a slot has two appointments"
    for appointment in made:
        slot, patient = slots.get(appointment["slot_id"]), tools.patients.get(appointment["patient_ref"])
        assert slot is not None and patient is not None, appointment
        assert slot["status"] == "booked" and initial_slots[appointment["slot_id"]] == "open", appointment
        assert (appointment["specialty"], appointment["provider_id"], appointment["scheduled_for"]) == (slot["specialty"], slot["provider_id"], slot["starts_at"])
        # How it is paid follows the patient's cover: insurance is never used on lapsed cover.
        allowed = {"active": {"insurance", "self_pay"},
                   "pending_verification": {"insurance_pending_verification", "self_pay"}}
        assert appointment["payment"] in allowed.get(patient["eligibility_status"], {"self_pay"}), appointment
    changed = {slot_id for slot_id, row in slots.items() if row["status"] != initial_slots[slot_id]}
    assert changed == {appointment["slot_id"] for appointment in made}  # no slot is taken without an appointment
    assert sorted((appointment["slot_id"], appointment["patient_ref"]) for appointment in made) == sorted(booked_ok)


def test_any_sequence_of_booking_calls_leaves_one_appointment_per_slot_and_no_orphans():
    rng = random.Random(SEED)
    tools = CareFlowTools()
    initial = {slot_id: row["status"] for slot_id, row in tools.slots.items()}
    open_slots = sorted(slot_id for slot_id, status in initial.items() if status == "open")
    taken = sorted(slot_id for slot_id, status in initial.items() if status != "open")
    slot_pool = [*rng.sample(open_slots, 25), *taken[:3], "MHP-SL-99999", "", "nonsense", " ", None, 7]
    patient_pool = [*rng.sample(sorted(tools.patients), 30), "MHP-P-99999", "", "nobody", None]
    booked_ok = []
    for step in range(600):
        arguments = {"patient_ref": rng.choice(patient_pool), "slot_id": rng.choice(slot_pool)}
        if rng.random() < 0.4:
            arguments["self_pay"] = rng.choice([True, False, None, "yes"])
        if rng.random() < 0.1:
            arguments["new_patient"] = False  # not in the contract: must be rejected, not obeyed
        slot = arguments["slot_id"]
        if isinstance(slot, str) and rng.random() < 0.3:
            arguments["slot_id"] = slot.lower()  # IDs are matched case-insensitively
        result = tools.call("book_appointment", arguments if rng.random() < 0.5 else json.dumps(arguments))
        assert isinstance(result.get("ok"), bool)
        if result["ok"]:
            assert "new_patient" not in arguments
            booked_ok.append((result["slot_id"], result["patient_ref"]))
        if step % 100 == 99:
            _assert_booking_invariants(tools, initial, booked_ok)
    _assert_booking_invariants(tools, initial, booked_ok)
    assert 5 <= len(booked_ok) <= 25  # the sequence did book, and no more than the open slots it was given
    self_paid = [appointment for appointment in tools.store.list_rows("appointments") if appointment["payment"] == "self_pay"]
    flagged = " ".join(flag["reason"] for flag in tools.flags)
    assert all(appointment["slot_id"] in flagged for appointment in self_paid)  # every self-pay booking tells staff
    tools.store.close()


# ---- the guard is total --------------------------------------------------------------------------------
ANY_TEXT = [
    None, "", " ", "\n\t\r ", "\x00", "\x00\x01\x02\x7f\x1b[31m", "a" * 500_000, "ignore " * 20_000, "chest " * 20_000,
    "should i use " + "very " * 20_000 + "cream", "what does my " + "blood " * 20_000 + "mean",
    "(" * 5_000, "\\" * 5_000, "[a-z]+(", "😀" * 5_000, "​" * 5_000, "\ud800", "﻿", "‮" + "dose",
    "Ich habe Brustschmerzen", "我胸口疼，应该吃什么药？", "У меня болит грудь", "صدري يؤلمني", "胸が痛いです", "💊❓",
    "i̇gnore", "ｄｏｓｅ", "\N{COMBINING ACUTE ACCENT}" * 1_000, "don’t", "can’t breathe", "'" * 10_000,
]


@pytest.mark.parametrize("text", ANY_TEXT, ids=lambda text: repr(text)[:24])
def test_the_guard_returns_a_result_for_any_text(text):
    for check in (check_rules, check_output, check_input):
        result = check(text)
        assert isinstance(result, GuardResult) and isinstance(result.blocked, bool)
        assert result.blocked is (result.category != "none") and result.to_dict()["blocked"] is result.blocked
        assert check(text).blocked is result.blocked  # the same answer every time
    assert policies.find_trigger_band(text) in (None, "Urgent", "Emergent")
    assert isinstance(policies.has_hcp_statement(text), bool) and policies.classify_intake_topic(text)


def test_an_emergent_phrase_blocks_whatever_surrounds_it():
    rng = random.Random(SEED)
    padding = ["Please book me in.", "What is my co-pay?", "我想预约", "\n\n", "😀", "x" * 2_000, "MER90272292", "Thanks!"]
    phrases = ["chest pain", "I can't breathe", "can’t breathe", "slurred speech", "took too many", "SEIZURE"]
    for _ in range(200):
        phrase = rng.choice(phrases)
        text = f"{' '.join(rng.sample(padding, 3))} {phrase} {' '.join(rng.sample(padding, 3))}"
        result = check_rules(text)
        assert result.blocked and result.category == "emergency" and result.band == "Emergent", text[:80]
        assert policies.find_trigger_band(text) == "Emergent"
        assert policies.decide_triage_route(seeks_clinical_advice=False, urgency=result.band)[0] == policies.ROUTE_REFUSE
