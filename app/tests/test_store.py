"""Offline tests for the transactional SQLite runtime store. File databases live in tmp_path; the seed is never written."""

import csv
import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from src.config import DATA_DIR
from src.store import RuntimeStore, SlotUnavailableError
from src.tools import CareFlowTools


@pytest.fixture()
def database_path(tmp_path):
    return tmp_path / "runtime.db"


def _pick_established_active(tools):
    # Established (has a completed visit), so any open slot can be booked regardless of the provider's
    # new-patient setting; that rule is tested in test_tools.py.
    established = tools.established
    return next(patient["patient_ref"] for patient in tools.patients.values() if patient["eligibility_status"] == "active" and patient["patient_ref"] in established)


def _find_first_slot(tools, ref, specialty="ENT"):
    return tools.find_appointment_slots(specialty, patient_ref=ref)["slots"][0]["slot_id"]


def _hash_seed_files():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((DATA_DIR / "mock_api").iterdir())}


def test_seeding_loads_every_seed_table_with_all_rows():
    store = RuntimeStore(":memory:")
    for path in (DATA_DIR / "mock_api").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as seed_file:
            assert len(store.list_rows(path.stem)) == sum(1 for _ in csv.DictReader(seed_file)), path.stem
    assert {"staff_flags", "meta"} <= store.list_tables()


def test_file_database_uses_wal_for_multi_process_access(database_path):
    store = RuntimeStore(database_path)
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_booking_persists_across_a_new_store_and_tools(database_path):
    original = CareFlowTools(store=RuntimeStore(database_path))
    ref = _pick_established_active(original)
    slot = _find_first_slot(original, ref)
    result = original.book_appointment(ref, slot)
    assert result["ok"] and result["appointment_id"].startswith("MHP-A-")
    original.store.close()

    reopened = CareFlowTools(store=RuntimeStore(database_path))  # a "restart": new connection, new objects
    assert reopened.slots[slot]["status"] == "booked" and reopened.booked[slot] == ref
    assert slot not in [offered["slot_id"] for offered in reopened.find_appointment_slots("ENT", patient_ref=ref, limit=20)["slots"]]
    again = reopened.book_appointment(ref, slot)
    assert again["ok"] is False and "no longer available" in again["error"]


def test_booking_writes_slot_appointment_and_payment_together():
    tools = CareFlowTools()
    ref = _pick_established_active(tools)
    slot = _find_first_slot(tools, ref, "Cardiology")
    before = len(tools.store.list_rows("appointments"))
    appointment_id = tools.book_appointment(ref, slot)["appointment_id"]
    rows = tools.store.list_rows("appointments")
    assert len(rows) == before + 1 and tools.slots[slot]["status"] == "booked"
    new = next(row for row in rows if row["appointment_id"] == appointment_id)
    assert new["patient_ref"] == ref and new["status"] == "scheduled" and new["slot_id"] == slot
    assert new["payment"] == "insurance" and isinstance(new["duration_minutes"], int)
    assert rows[0]["payment"] is None and rows[0]["slot_id"] is None  # seeded rows predate these columns


def test_self_pay_booking_and_its_staff_flag_are_saved_together(database_path):
    tools = CareFlowTools(store=RuntimeStore(database_path))
    lapsed = next(patient["patient_ref"] for patient in tools.patients.values() if patient["eligibility_status"] == "lapsed")
    result = tools.book_appointment(lapsed, _find_first_slot(tools, lapsed), self_pay=True)
    fresh = RuntimeStore(database_path)
    row = next(appointment for appointment in fresh.list_rows("appointments") if appointment["appointment_id"] == result["appointment_id"])
    assert row["payment"] == "self_pay"
    assert [flag["flag_id"] for flag in fresh.list_flags()] == [result["flag_id"]] and fresh.list_flags()[0]["patient_ref"] == lapsed


def test_two_processes_cannot_double_book_a_slot(database_path):
    first, second = CareFlowTools(store=RuntimeStore(database_path)), CareFlowTools(store=RuntimeStore(database_path))
    ref = _pick_established_active(first)
    slot = _find_first_slot(first, ref)
    assert first.book_appointment(ref, slot)["ok"]
    late = second.book_appointment(ref, slot)  # second connection, stale view is irrelevant: the write checks
    assert late["ok"] is False and "no longer available" in late["error"]
    assert sum(1 for row in RuntimeStore(database_path).list_rows("appointments") if row["slot_id"] == slot) == 1


def test_concurrent_bookings_of_one_slot_give_exactly_one_success(database_path):
    RuntimeStore(database_path)  # seed once
    ref = _pick_established_active(CareFlowTools(store=RuntimeStore(database_path)))
    slot = _find_first_slot(CareFlowTools(store=RuntimeStore(database_path)), ref)
    stores = [RuntimeStore(database_path) for _ in range(8)]
    results, gate = [], threading.Barrier(len(stores))

    def attempt(store):
        gate.wait()
        try:
            store.book(slot, ref, "insurance")
            results.append("ok")
        except SlotUnavailableError:
            results.append("taken")

    threads = [threading.Thread(target=attempt, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["ok"] + ["taken"] * 7
    assert sum(1 for row in RuntimeStore(database_path).list_rows("appointments") if row["slot_id"] == slot) == 1


def test_a_failure_mid_booking_rolls_everything_back(monkeypatch):
    tools = CareFlowTools()
    ref = _pick_established_active(tools)
    slot = _find_first_slot(tools, ref)
    before = len(tools.store.list_rows("appointments"))

    def fail_with_disk_full(connection):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(tools.store, "_make_next_appointment_id", fail_with_disk_full)
    with pytest.raises(sqlite3.OperationalError):
        tools.book_appointment(ref, slot)
    assert tools.slots[slot]["status"] == "open" and len(tools.store.list_rows("appointments")) == before


def test_flags_persist_and_stay_idempotent_across_instances(database_path):
    original = CareFlowTools(store=RuntimeStore(database_path))
    ref = _pick_established_active(original)
    first = original.flag_for_human("asks about a dosing change", "Urgent", ref)
    reopened = CareFlowTools(store=RuntimeStore(database_path))
    again = reopened.flag_for_human("asks about a dosing change", "Urgent", ref)
    assert again["duplicate"] is True and again["flag_id"] == first["flag_id"]
    other = reopened.flag_for_human("a different reason", "Routine", ref)
    assert other["flag_id"] != first["flag_id"]
    assert [flag["flag_id"] for flag in RuntimeStore(database_path).list_flags()] == [first["flag_id"], other["flag_id"]]


def test_reset_restores_the_seed_and_clears_flags(database_path):
    store = RuntimeStore(database_path)
    tools = CareFlowTools(store=store)
    ref = _pick_established_active(tools)
    slot = _find_first_slot(tools, ref)
    tools.book_appointment(ref, slot)
    tools.flag_for_human("x")
    store.reset()
    fresh = CareFlowTools(store=store)
    assert fresh.slots[slot]["status"] == "open"
    assert store.list_flags() == [] and len(store.list_rows("appointments")) == 900


def test_in_memory_stores_are_isolated_from_each_other():
    first, second = CareFlowTools(), CareFlowTools()
    ref = _pick_established_active(first)
    slot = _find_first_slot(first, ref)
    assert first.book_appointment(ref, slot)["ok"]
    assert second.slots[slot]["status"] == "open" and second.booked == {} and second.flags == []


def test_the_seeded_files_are_never_modified(tmp_path):
    before = _hash_seed_files()
    store = RuntimeStore(tmp_path / "rt.db")
    tools = CareFlowTools(store=store)
    ref = _pick_established_active(tools)
    tools.book_appointment(ref, _find_first_slot(tools, ref))
    tools.flag_for_human("x")
    store.reset()
    store.close()
    assert _hash_seed_files() == before
    assert not any(path.suffix == ".db" for path in Path(DATA_DIR).rglob("*"))


# ---- human-review requests and decisions --------------------------------------------------------------
def test_review_round_trip_closes_its_flag(database_path):
    store = RuntimeStore(database_path)
    flag, _ = store.add_flag(None, "Triage review needed", "Urgent", "2026-08-01T09:00", contact_id="t1")
    review = store.add_review("t1", "human_review", ["Urgent band", "No specialty"], "2026-08-01T09:00", flag_id=flag["flag_id"])
    assert review["review_id"] == "REV-0001" and review["status"] == "pending"
    assert review["reasons"] == "Urgent band; No specialty"
    assert store.add_review("t1", "human_review", ["again"], "2026-08-01T09:05") == review  # one per thread
    assert [pending["review_id"] for pending in store.list_pending_reviews()] == ["REV-0001"]

    decided = store.decide_review("REV-0001", "RN-7", "approve", "2026-08-01T09:30", note="Called.", urgency="Priority")
    assert (decided["status"], decided["reviewer_id"], decided["decided_at"]) == ("decided", "RN-7", "2026-08-01T09:30")
    assert store.list_pending_reviews() == []
    assert (store.list_flags()[0]["status"], store.list_flags()[0]["urgency"]) == ("reviewed_approve", "Priority")
    # A second decision changes nothing: the first reviewer stays on record.
    assert store.decide_review("REV-0001", "RN-9", "reject", "2026-08-01T10:00") == decided
    store.close()
    again = RuntimeStore(database_path)  # survives a restart; re-opening does not recreate or clear the table
    assert again.get_review("REV-0001") == decided
    again.close()


def test_deciding_an_unknown_review_raises():
    with pytest.raises(KeyError):
        RuntimeStore(":memory:").decide_review("REV-9999", "RN-1", "approve", "2026-08-01T09:00")
