"""M7 tracing: one trace per request with a span for every step, agent, tool and model call, and no
patient content in any of it."""

import json
from types import SimpleNamespace

import pytest

from careflow.config import Settings
from careflow.reliability import ResilientTools, resilient_model
from careflow.telemetry import (NULL_TRACER, JsonlSink, LangfuseSink, MemorySink, Tracer, default_tracer, safe_attrs,
                                scrub)

from .test_workflow import APPROVE, Harness, _call, _msg, _patient


def _traced(h):
    """Wire a harness the way Deps.default does, recording spans in memory."""
    sink = MemorySink()
    tracer = Tracer([sink])
    h.deps.agent.tracer = tracer
    h.deps.agent.tools = ResilientTools(h.tools, tracer=tracer)
    h.deps.agent.model = resilient_model(h.deps.agent.model, tracer=tracer, name="fake-model")
    return sink


def _names(sink):
    return [s["name"] for s in sink.spans]


# ---- structure -----------------------------------------------------------------------------------------
def test_one_trace_per_request_with_a_span_for_every_step_tool_agent_and_model_call():
    steps = [_msg(calls=[_call("c1", "find_appointment_slots", specialty="Dermatology")]), _msg("Here are some slots.")]
    h = Harness(steps=steps)
    sink = _traced(h)
    r = h.start(_patient(h.tools), thread_id="case-7")

    roots = [s for s in sink.spans if s["kind"] == "trace"]
    assert len(roots) == 1 and roots[0]["name"] == "triage" and roots[0]["trace_id"] == "case-7"
    assert (roots[0]["route"], roots[0]["status"], roots[0]["case_status"]) == ("auto", "ok", "completed") and roots[0]["patient_ref"] == r.patient_ref
    assert {s["trace_id"] for s in sink.spans} == {"case-7"}

    names = _names(sink)
    for step in r.steps:
        assert f"step.{step}" in names
    assert "agent.referral_tracker" in names and "agent.clinical_safety_reviewer" in names
    assert names.count("model.complete") == 2
    assert {"tool.check_eligibility", "tool.find_appointment_slots"} <= set(names)

    by_id = {s["span_id"]: s for s in sink.spans}
    assert all(s["parent_id"] in by_id for s in sink.spans if s["kind"] != "trace")  # one connected tree
    tool = next(s for s in sink.spans if s["name"] == "tool.find_appointment_slots")
    assert (tool["kind"], tool["backend"], tool["ok"], tool["specialty"], tool["attempts"]) == ("tool", "ehr", True, "Dermatology", 1)
    assert by_id[by_id[tool["parent_id"]]["parent_id"]]["name"] in ("step.team", "agent.referral_tracker")
    model = next(s for s in sink.spans if s["name"] == "model.complete")
    assert model["kind"] == "generation" and model["model"] == "fake-model" and model["tool_calls"] == ["find_appointment_slots"]
    assert all(s["duration_ms"] >= 0 for s in sink.spans)


def test_a_paused_case_is_marked_paused_and_the_resume_is_its_own_trace():
    h = Harness(parsed={"urgency": "Urgent"})
    sink = _traced(h)
    r = h.start(_patient(h.tools), thread_id="case-8")
    review = next(s for s in sink.spans if s["name"] == "step.review")
    root = next(s for s in sink.spans if s["kind"] == "trace")
    assert review["status"] == "paused" and (root["status"], root["route"], root["case_status"]) == ("ok", "human_review", "awaiting_human_review")
    assert root["review_id"] == r.review_id

    sink.spans.clear()
    h.resume("case-8", APPROVE)
    root = next(s for s in sink.spans if s["kind"] == "trace")
    assert (root["name"], root["action"], root["reviewer_id"]) == ("triage.resume", "approve", "RN-0042")
    assert "step.finalize" in _names(sink) and "step.intake" not in _names(sink)


def test_a_failing_step_is_recorded_as_an_error_and_still_raises():
    sink = MemorySink()
    tracer = Tracer([sink])
    with pytest.raises(KeyError):
        with tracer.trace("t1", "triage"):
            with tracer.span("step.intake", step="intake"):
                raise KeyError("boom")
    assert [(s["name"], s["status"], s["error_type"]) for s in sink.spans] == [
        ("step.intake", "error", "KeyError"), ("triage", "error", "KeyError")]


# ---- privacy -------------------------------------------------------------------------------------------
def test_no_patient_content_reaches_the_trace(tmp_path):
    text = "Hi, I am Zelda Quixote, born 14/02/1961, member MER73516094. My knee hurts, can I see ortho?"
    steps = [_msg(calls=[_call("c1", "find_patient", given_name="Zelda", family_name="Quixote", birth_date="1961-02-14",
                               member_number="MER73516094"),
                         _call("c2", "flag_for_human", reason="Zelda Quixote says her knee hurts badly", urgency="Routine")]),
             _msg("Zelda, I have passed this to staff. Your member number MER73516094 is on file.")]
    h = Harness(steps=steps, parsed={"patient_name": "Zelda Quixote", "insurance_member_number": "MER73516094",
                                     "symptoms": ["knee hurts"]})
    sink = _traced(h)
    path = tmp_path / "traces.jsonl"
    h.deps.agent.tracer.sinks.append(JsonlSink(path))
    ref = _patient(h.tools)
    from careflow.workflows import start_triage

    record = {"channel": "email", "received_at": "2026-08-01T09:00:00Z", "raw_text": text, "patient_ref": ref}
    start_triage(record, h.deps, h.checkpointer)

    written = path.read_text(encoding="utf-8")
    assert len(written.splitlines()) == len(sink.spans) > 8
    for secret in ("Zelda", "Quixote", "MER73516094", "1961", "14/02", "knee", "passed this to staff"):
        assert secret not in written, secret
    assert ref in written  # the pseudonymous record ID is kept so staff can find the trace
    flag = next(s for s in sink.spans if s["name"] == "tool.flag_for_human")
    assert "reason" not in flag and flag["urgency"] == "Routine" and flag["ok"] is True


def test_scrub_and_the_attribute_allow_list():
    assert scrub("member MER73516094 born 1961-02-14 or 14/02/1961") == "member [member] born [date] or [date]"
    assert len(scrub("x" * 5000)) == 200
    attrs = safe_attrs({"patient_ref": "MHP-P-00001", "given_name": "Zelda", "query": "my knee", "reason": "free text",
                        "raw_text": "anything", "plan_code": "MER12345678", "ok": True, "count": 3, "slot_id": None})
    assert attrs == {"patient_ref": "MHP-P-00001", "plan_code": "[member]", "ok": True, "count": 3}
    assert scrub({"patient_ref": "MHP-P-00001", "family_name": "Quixote"}) == {"patient_ref": "MHP-P-00001"}
    assert scrub(object()) == "object"


# ---- switching it on and off ---------------------------------------------------------------------------
def test_tracing_off_records_nothing_and_changes_nothing():
    assert default_tracer(Settings(careflow_tracing="off")) is NULL_TRACER and not NULL_TRACER.enabled
    h = Harness()
    r = h.start(_patient(h.tools))  # the harness default: NULL_TRACER
    assert (r.route, r.status) == ("auto", "completed")
    with NULL_TRACER.trace("t", "x") as span:
        span.set(ok=True)  # harmless


def test_jsonl_is_the_default_and_langfuse_without_keys_falls_back_to_it(tmp_path):
    for mode in ("jsonl", "langfuse"):
        tracer = default_tracer(Settings(careflow_tracing=mode, langfuse_public_key="", langfuse_secret_key=""),
                                tmp_path / f"{mode}.jsonl")
        assert [type(s) for s in tracer.sinks] == [JsonlSink]
        with tracer.trace("t1", "triage", channel="sms"):
            pass
        (line,) = (tmp_path / f"{mode}.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(line)["channel"] == "sms"


def test_a_broken_sink_never_breaks_a_request():
    class Broken:
        def on_start(self, span, parent):
            raise OSError("disk full")

        def on_end(self, span):
            raise OSError("disk full")

        def close(self):
            raise OSError("disk full")

    h = Harness()
    good = _traced(h)
    h.deps.agent.tracer.sinks.insert(0, Broken())
    r = h.start(_patient(h.tools))
    assert (r.route, r.status) == ("auto", "completed") and good.spans
    h.deps.agent.tracer.close()


def test_langfuse_sink_mirrors_the_span_tree():
    calls = []

    class Observation:
        def __init__(self, name, parent):
            self.name, self.parent = name, parent

        def start_observation(self, name, as_type, metadata):
            calls.append(("start", name, as_type, self.name))
            return Observation(name, self.name)

        def update(self, metadata, level):
            calls.append(("update", self.name, level, metadata.get("ok")))

        def end(self):
            calls.append(("end", self.name))

    client = SimpleNamespace(
        start_observation=lambda name, as_type, metadata: calls.append(("start", name, as_type, None)) or Observation(name, None),
        flush=lambda: calls.append(("flush",)),
    )
    tracer = Tracer([LangfuseSink(client)])
    with tracer.trace("t1", "triage"):
        with tracer.span("tool.check_eligibility", kind="tool", ok=True):
            pass
        with pytest.raises(ValueError):
            with tracer.span("model.complete", kind="generation"):
                raise ValueError("x")
    assert calls == [
        ("start", "triage", "span", None),
        ("start", "tool.check_eligibility", "tool", "triage"), ("update", "tool.check_eligibility", "DEFAULT", True),
        ("end", "tool.check_eligibility"),
        ("start", "model.complete", "generation", "triage"), ("update", "model.complete", "ERROR", None),
        ("end", "model.complete"),
        ("update", "triage", "DEFAULT", None), ("end", "triage"), ("flush",),
    ]
