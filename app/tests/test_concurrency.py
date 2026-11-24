"""Concurrency: what must hold when several API worker threads (or processes, through separate SQLite
connections to one file) use the same store, breaker, trace file or embedding cache at once.

Nothing here sleeps or depends on timing. Threads are released together with a Barrier, every clock is
injected, and the assertions hold for every interleaving, so a failure is a real race and not a flaky test.
File databases live in tmp_path and are closed again so Windows can delete the folder."""

import json
import sqlite3
import struct
import threading

import pytest

from src.config import Settings
from src.rag.embeddings import LiteLLMEmbedder, _normalise
from src.reliability import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    ModelUnavailableError,
    ResilientTools,
    make_resilient_model,
)
from src.store import RuntimeStore, SlotUnavailableError
from src.telemetry import JsonlSink, MemorySink, Tracer
from src.tools import CareFlowTools

THREADS = 12
JOIN_TIMEOUT_S = 30
SLOT_TAKEN = "that slot is no longer available"  # book_appointment's documented error for a lost race


def run_together(count, task):
    """Run fn(0) .. fn(n-1) on n threads released at the same moment; return their results in order.
    Fails the test if a thread raised or is still running after the timeout."""
    barrier = threading.Barrier(count)
    results, errors = [None] * count, []

    def work(index):
        try:
            barrier.wait(timeout=JOIN_TIMEOUT_S)
            results[index] = task(index)
        except BaseException as exc:  # reported below, on the test's own thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_TIMEOUT_S)
    assert not [thread.name for thread in threads if thread.is_alive()], "a thread did not finish (deadlock?)"
    assert errors == [], errors
    return results


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


@pytest.fixture()
def stores(tmp_path):
    """open() gives a new connection to the one database file of this test; all are closed afterwards."""
    opened = []

    def open_store():
        opened.append(RuntimeStore(tmp_path / "runtime.db"))
        return opened[-1]

    yield open_store
    for store in opened:
        store.close()


def _pick_established_active(tools, count):
    """n patients with active cover and a completed visit: any open slot can be booked for them."""
    established = tools.established
    refs = sorted(ref for ref, patient in tools.patients.items() if patient["eligibility_status"] == "active" and ref in established)
    assert len(refs) >= count
    return refs[:count]


def _pick_open_slots(tools, count):
    slots = sorted(slot["slot_id"] for slot in tools.slots.values() if slot["status"] == "open")
    assert len(slots) >= count
    return slots[:count]


def _list_bookings_of(store, slot_id):
    return [appointment for appointment in store.list_rows("appointments") if appointment["slot_id"] == slot_id]


# ---- booking: one slot, many callers -------------------------------------------------------------------
def test_one_slot_booked_by_many_threads_on_one_store_goes_to_exactly_one(stores):
    tools = CareFlowTools(store=stores())
    refs, (slot,) = _pick_established_active(tools, THREADS), _pick_open_slots(tools, 1)

    results = run_together(THREADS, lambda index: tools.call("book_appointment", {"patient_ref": refs[index], "slot_id": slot}))

    won = [result for result in results if result["ok"]]
    lost = [result for result in results if not result["ok"]]
    assert len(won) == 1 and len(lost) == THREADS - 1
    assert {result["error"] for result in lost} == {SLOT_TAKEN} and all(result["slot_id"] == slot for result in lost)
    (appointment,) = _list_bookings_of(tools.store, slot)
    assert appointment["patient_ref"] == won[0]["patient_ref"] and appointment["appointment_id"] == won[0]["appointment_id"]
    assert tools.slots[slot]["status"] == "booked"


def test_one_slot_booked_through_separate_connections_goes_to_exactly_one(stores):
    # One connection per caller, as with several API workers or the API and the MCP server process.
    callers = [CareFlowTools(store=stores()) for _ in range(6)]
    refs, (slot,) = _pick_established_active(callers[0], len(callers)), _pick_open_slots(callers[0], 1)

    results = run_together(
        len(callers), lambda index: callers[index].call("book_appointment", {"patient_ref": refs[index], "slot_id": slot}))

    assert sorted(result["ok"] for result in results) == [False] * (len(callers) - 1) + [True]
    assert {result["error"] for result in results if not result["ok"]} == {SLOT_TAKEN}
    reader = stores()  # a connection that took no part
    (appointment,) = _list_bookings_of(reader, slot)
    assert appointment["patient_ref"] == next(result["patient_ref"] for result in results if result["ok"])
    assert reader.find_one("appointment_slots", "slot_id", slot)["status"] == "booked"


def test_the_store_itself_never_books_a_slot_twice_when_the_check_is_skipped(stores):
    # store.book is the last line of defence: callers that all saw the slot open still get one booking.
    store = stores()
    tools = CareFlowTools(store=store)
    refs, (slot,) = _pick_established_active(tools, THREADS), _pick_open_slots(tools, 1)

    def book(index):
        try:
            return store.book(slot, refs[index], "insurance")[0]["appointment_id"]
        except SlotUnavailableError:
            return None

    results = run_together(THREADS, book)
    assert len([result for result in results if result]) == 1 and len(_list_bookings_of(store, slot)) == 1


def test_different_slots_booked_at_once_all_succeed_with_distinct_appointment_ids(stores):
    callers = [CareFlowTools(store=stores()) for _ in range(4)]
    count = 16
    refs, slots = _pick_established_active(callers[0], count), _pick_open_slots(callers[0], count)
    before = len(callers[0].store.list_rows("appointments"))

    results = run_together(
        count, lambda index: callers[index % 4].call("book_appointment", {"patient_ref": refs[index], "slot_id": slots[index]}))

    assert all(result["ok"] for result in results), results
    assert len({result["appointment_id"] for result in results}) == count  # the serial is taken inside the transaction
    rows = callers[0].store.list_rows("appointments")
    assert len(rows) == before + count and len({appointment["appointment_id"] for appointment in rows}) == len(rows)
    assert {(appointment["slot_id"], appointment["patient_ref"]) for appointment in rows if appointment["slot_id"]} == set(zip(slots, refs, strict=True))


def test_an_existing_database_opened_by_many_threads_at_once_is_not_seeded_again(tmp_path):
    first = RuntimeStore(tmp_path / "runtime.db")
    expected = {table: len(first.list_rows(table)) for table in ("patients", "appointments", "appointment_slots", "referrals")}
    first.book(sorted(row["slot_id"] for row in first.list_rows("appointment_slots") if row["status"] == "open")[0],
               "MHP-P-00000", "insurance")
    expected["appointments"] += 1

    opened = run_together(8, lambda _: RuntimeStore(tmp_path / "runtime.db"))
    try:
        for store in opened:  # every connection sees the one seeded copy and the booking made before it opened
            assert {table: len(store.list_rows(table)) for table in expected} == expected
    finally:
        for store in [first, *opened]:
            store.close()


def test_opening_a_new_database_while_another_connection_holds_it_waits_instead_of_failing(tmp_path):
    path = tmp_path / "fresh.db"
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")  # what the first opener holds while it creates the file
    outcome = []

    def open_store():
        try:
            outcome.append(RuntimeStore(path))
        except sqlite3.OperationalError as exc:
            outcome.append(exc)

    opener = threading.Thread(target=open_store, daemon=True)
    opener.start()
    # Not a timing assumption: a correct opener is still waiting for the lock here (its busy timeout is 5 s)
    # and one that has already given up has failed, whenever this returns.
    opener.join(timeout=0.5)
    holder.execute("COMMIT")
    holder.close()
    opener.join(timeout=JOIN_TIMEOUT_S)
    assert not opener.is_alive()
    try:
        assert isinstance(outcome[0], RuntimeStore), outcome
        assert len(outcome[0].list_rows("patients")) > 0
    finally:
        if isinstance(outcome[0], RuntimeStore):
            outcome[0].close()


# ---- staff flags and reviews ---------------------------------------------------------------------------
def test_the_same_flag_raised_at_once_in_one_contact_is_recorded_once(stores):
    tools = CareFlowTools(store=stores())
    (ref,) = _pick_established_active(tools, 1)
    args = {"reason": "Caller asked for a call back", "urgency": "Priority", "patient_ref": ref}

    results = run_together(THREADS, lambda _: tools.call("flag_for_human", args, context={"contact_id": "contact-1"}))

    assert all(result["ok"] for result in results) and len({result["flag_id"] for result in results}) == 1
    assert sorted(result["duplicate"] for result in results) == [False] + [True] * (THREADS - 1)
    assert len(tools.flags) == 1


def test_the_same_flag_raised_through_separate_connections_is_recorded_once(stores):
    callers = [CareFlowTools(store=stores()) for _ in range(4)]
    args = {"reason": "Policy question the documents do not cover"}

    results = run_together(8, lambda index: callers[index % 4].call("flag_for_human", args, context={"contact_id": "contact-2"}))

    assert len({result["flag_id"] for result in results}) == 1 and [result["duplicate"] for result in results].count(False) == 1
    assert len(callers[0].flags) == 1


def test_flag_ids_never_collide_under_concurrent_inserts(stores):
    callers = [CareFlowTools(store=stores()) for _ in range(4)]
    per_thread = 5

    def raise_flags(index):
        return [callers[index % 4].call("flag_for_human", {"reason": f"reason {index}-{sequence}"}, context={"contact_id": f"c{index}"})
                for sequence in range(per_thread)]

    results = [result for batch in run_together(8, raise_flags) for result in batch]
    total = 8 * per_thread
    assert all(result["ok"] and not result["duplicate"] for result in results)
    assert sorted(result["flag_id"] for result in results) == [f"FLAG-{number:04d}" for number in range(1, total + 1)]
    assert len(callers[0].flags) == total


def test_review_ids_never_collide_and_a_thread_has_one_review(stores):
    connections = [stores() for _ in range(4)]

    def request(index):  # two callers per case thread ask for its review at the same moment
        return connections[index % 4].add_review(f"case-{index // 2}", "human_review", ["Urgent band"], "2026-08-01T09:00")

    results = run_together(THREADS, request)
    cases = THREADS // 2
    assert sorted({result["review_id"] for result in results}) == [f"REV-{number:04d}" for number in range(1, cases + 1)]
    for index in range(0, THREADS, 2):
        assert results[index]["review_id"] == results[index + 1]["review_id"]
    pending = connections[0].list_pending_reviews()
    assert len(pending) == cases and len({review["thread_id"] for review in pending}) == cases


def test_a_review_decided_by_several_staff_at_once_records_one_decision(stores):
    store = stores()
    flag, _ = store.add_flag(None, "Triage review needed", "Urgent", "2026-08-01T09:00", contact_id="case-1")
    review = store.add_review("case-1", "human_review", ["Urgent band"], "2026-08-01T09:00", flag_id=flag["flag_id"])
    other = stores()
    actions = ("approve", "override")

    def decide(index):
        connection = store if index % 2 else other
        return connection.decide_review(review["review_id"], f"RN-{index:04d}", actions[index % 2], "2026-08-01T09:05", note=f"n{index}")

    results = run_together(THREADS, decide)
    final = store.get_review(review["review_id"])
    assert final["status"] == "decided" and final["reviewer_id"] in {f"RN-{index:04d}" for index in range(THREADS)}
    winner = int(final["reviewer_id"][3:])
    assert (final["action"], final["note"]) == (actions[winner % 2], f"n{winner}")  # one caller's decision, whole
    assert results[winner] == final
    assert store.get_flag(flag["flag_id"])["status"] == f"reviewed_{final['action']}"
    assert store.list_pending_reviews() == []


def test_one_conversation_recorded_at_once_is_one_contact_row(stores):
    connections = [stores() for _ in range(3)]
    ref = sorted(row["patient_ref"] for row in connections[0].list_rows("patients"))[0]
    before = len(connections[0].list_rows("patient_interactions"))

    def record(index):
        return connections[index % 3].add_interaction(ref, "2026-08-01T09:00", "sms", "billing_question", "resolved",
                                                  f"summary {index}", thread_id="thread-1")

    results = run_together(9, record)
    assert len({result["interaction_id"] for result in results}) == 1
    rows = connections[0].list_rows("patient_interactions")
    assert len(rows) == before + 1 and len({row["interaction_id"] for row in rows}) == len(rows)


# ---- circuit breaker -----------------------------------------------------------------------------------
def test_breaker_counts_every_failure_reported_at_once():
    breaker = CircuitBreaker("ehr", failure_threshold=10_000, cooldown_s=30, now=Clock())
    per_thread = 200
    run_together(THREADS, lambda _: [breaker.record_failure() for _ in range(per_thread)])
    assert breaker.failures == THREADS * per_thread and breaker.state == CLOSED


@pytest.mark.parametrize("failures, state", [(THREADS - 1, CLOSED), (THREADS, OPEN)])
def test_breaker_opens_exactly_at_the_threshold_under_threads(failures, state):
    breaker = CircuitBreaker("ehr", failure_threshold=THREADS, cooldown_s=30, now=Clock())
    run_together(failures, lambda _: breaker.record_failure())
    assert breaker.failures == failures and breaker.state == state and breaker.allow() is (state == CLOSED)


def test_half_open_breaker_lets_exactly_one_of_many_callers_through():
    clock = Clock()
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    breaker.record_failure()
    assert run_together(THREADS, lambda _: breaker.allow()) == [False] * THREADS  # open: nobody

    clock.seconds = 30
    assert breaker.state == HALF_OPEN
    assert sorted(run_together(THREADS, lambda _: breaker.allow())) == [False] * (THREADS - 1) + [True]

    breaker.record_failure()  # the trial failed: open again, and after the next cool-down one trial again
    assert run_together(THREADS, lambda _: breaker.allow()) == [False] * THREADS
    clock.seconds = 60
    assert run_together(THREADS, lambda _: breaker.allow()).count(True) == 1
    breaker.record_success()
    assert breaker.state == CLOSED and run_together(THREADS, lambda _: breaker.allow()) == [True] * THREADS


class BlockingTools:
    """Tools whose call waits, inside the backend, until the test lets it finish."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def call(self, name, arguments=None, context=None):
        self.calls.append(name)
        self.entered.set()
        assert self.release.wait(JOIN_TIMEOUT_S)
        return {"ok": True, "found": False}


def test_while_a_half_open_trial_call_is_in_flight_every_other_caller_fails_fast():
    clock = Clock()
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    inner = BlockingTools()
    tools = ResilientTools(inner, breakers={"ehr": breaker})
    breaker.record_failure()
    clock.seconds = 30

    trial_result = []
    trial = threading.Thread(target=lambda: trial_result.append(tools.call("check_eligibility", {})), daemon=True)
    trial.start()
    assert inner.entered.wait(JOIN_TIMEOUT_S)  # the trial call is now inside the backend

    try:
        others = run_together(THREADS, lambda _: tools.call("check_eligibility", {}))
        assert all(result["unavailable"] and "circuit open" in result["error"] for result in others)
        assert inner.calls == ["check_eligibility"]  # only the trial reached the backend
    finally:
        inner.release.set()
        trial.join(timeout=JOIN_TIMEOUT_S)
    assert not trial.is_alive() and trial_result == [{"ok": True, "found": False}]
    assert breaker.state == CLOSED and tools.call("check_eligibility", {})["ok"]


def test_an_open_breaker_keeps_every_thread_away_from_the_backend_and_the_model():
    inner = BlockingTools()
    inner.release.set()
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=Clock())
    breaker.record_failure()
    tools = ResilientTools(inner, breakers={"ehr": breaker})
    results = run_together(THREADS, lambda _: tools.call("find_patient", {"given_name": "A", "family_name": "B"}))
    assert all(result == results[0] and result["ok"] is False and result["unavailable"] for result in results) and inner.calls == []

    asked = []
    model = make_resilient_model(lambda messages, specs: asked.append(1), CircuitBreaker("model", 1, 30, now=Clock()))
    model.breaker.record_failure()

    def ask(_):
        with pytest.raises(ModelUnavailableError):
            model([], [])

    run_together(THREADS, ask)
    assert asked == []


def test_model_failures_from_many_threads_are_all_counted_against_its_breaker():
    def raise_provider_down(messages, specs):
        raise ConnectionError("provider down")

    model = make_resilient_model(raise_provider_down, CircuitBreaker("model", failure_threshold=10_000, cooldown_s=30, now=Clock()))

    def ask(_):
        for _ in range(50):
            with pytest.raises(ConnectionError):
                model([], [])

    run_together(THREADS, ask)
    assert model.breaker.failures == THREADS * 50


# ---- tracing -------------------------------------------------------------------------------------------
def test_jsonl_sink_written_from_many_threads_holds_only_complete_lines(tmp_path):
    path = tmp_path / "traces" / "traces.jsonl"  # the folder is created on first write, also under a race
    memory = MemorySink()
    tracer = Tracer([JsonlSink(path), memory])
    spans_per_trace = 25

    def request(index):
        with tracer.open_trace(f"thread-{index}", "triage", channel="sms"):
            for number in range(spans_per_trace):
                with tracer.open_span(f"tool.t{number}", kind="tool", tool="x" * 150, count=number, patient_ref=f"MHP-P-{index:05d}"):
                    pass

    run_together(THREADS, request)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == THREADS * (spans_per_trace + 1)
    written = [json.loads(line) for line in lines]  # every line is one whole JSON object
    assert sorted(span["span_id"] for span in written) == sorted(span["span_id"] for span in memory.spans)
    assert len({span["span_id"] for span in written}) == len(written)


def test_spans_opened_on_different_threads_never_nest_under_another_threads_trace():
    sink = MemorySink()
    tracer = Tracer([sink])

    def request(index):
        with tracer.open_trace(f"thread-{index}", "triage"), tracer.open_span("step.intake", step="intake"), \
                tracer.open_span("tool.check_eligibility", kind="tool", count=index):
            pass

    run_together(THREADS, request)

    by_id = {span["span_id"]: span for span in sink.spans}
    assert len(sink.spans) == THREADS * 3
    for index in range(THREADS):
        mine = [span for span in sink.spans if span["trace_id"] == f"thread-{index}"]
        assert sorted(span["name"] for span in mine) == ["step.intake", "tool.check_eligibility", "triage"]
        tool = next(span for span in mine if span["kind"] == "tool")
        step = by_id[tool["parent_id"]]
        root = by_id[step["parent_id"]]
        assert (tool["count"], step["name"], root["name"], root["parent_id"]) == (index, "step.intake", "triage", None)
        assert step["trace_id"] == root["trace_id"] == f"thread-{index}"


# ---- embedding cache -----------------------------------------------------------------------------------
class CountingEmbedder(LiteLLMEmbedder):
    """The cached embedder with the provider request replaced by a deterministic local function."""

    DIM = 4

    def __init__(self, cache_path):
        settings = Settings(_env_file=None, careflow_embedding_model="fake/embedder",
                            careflow_embedding_dimensions=self.DIM)
        super().__init__(settings, cache_path=cache_path)
        self.requested: list[str] = []  # every text sent to the "provider" (list.append is atomic)

    @staticmethod
    def embed_raw(text):
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0, float(ord(text[0]))]

    def _request(self, batch):
        self.requested.extend(batch)
        return [self.embed_raw(text) for text in batch]

    def compute_expected(self, text):
        """What embed() must return: the unit vector as float32, the cache's storage format."""
        return list(struct.unpack(f"{self.DIM}f", struct.pack(f"{self.DIM}f", *_normalise(self.embed_raw(text)))))


def test_embedding_cache_under_concurrent_use_returns_the_right_vector_for_every_text(tmp_path):
    embedder = CountingEmbedder(tmp_path / "embeddings.sqlite")
    texts = [f"policy passage number {number}" for number in range(115)]  # the ten slices below cover all of them
    try:
        def embed(index):  # overlapping slices, each in its own order
            mine = texts[index * 5: index * 5 + 70]
            mine = mine if index % 2 else mine[::-1]
            return mine, embedder.embed(mine)

        for mine, vectors in run_together(10, embed):
            assert vectors == [embedder.compute_expected(text) for text in mine]
        # No text was sent more often than there were callers racing for it, and none was lost.
        assert set(embedder.requested) == set(texts)
        assert max(embedder.requested.count(text) for text in texts) <= 10

        # Once cached, nothing is requested again, by any thread.
        embedder.requested.clear()
        for mine, vectors in run_together(10, embed):
            assert vectors == [embedder.compute_expected(text) for text in mine]
        assert embedder.requested == []
        assert embedder._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == len(texts)
    finally:
        embedder.close()
