"""Offline tests for the transactional SQLite runtime store. File databases live in tmp_path; the seed is never written."""

import csv
import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from careflow.config import DATA_DIR
from careflow.store import RuntimeStore, SlotUnavailable
from careflow.tools import CareFlowTools


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "runtime.db"


def _established_active(t):
    # Established (has a completed visit), so any open slot can be booked regardless of the provider's
    # new-patient setting; that rule is tested in test_tools.py.
    est = t.established
    return next(p["patient_ref"] for p in t.patients.values() if p["eligibility_status"] == "active" and p["patient_ref"] in est)


def _first_slot(t, ref, specialty="ENT"):
    return t.find_appointment_slots(specialty, patient_ref=ref)["slots"][0]["slot_id"]


def _seed_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((DATA_DIR / "mock_api").iterdir())}


def test_seeding_loads_every_seed_table_with_all_rows():
    s = RuntimeStore(":memory:")
    for path in (DATA_DIR / "mock_api").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as fh:
            assert len(s.rows(path.stem)) == sum(1 for _ in csv.DictReader(fh)), path.stem
    assert {"staff_flags", "meta"} <= s.tables()


def test_file_database_uses_wal_for_multi_process_access(db):
    s = RuntimeStore(db)
    assert s.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_booking_persists_across_a_new_store_and_tools(db):
    a = CareFlowTools(store=RuntimeStore(db))
    ref = _established_active(a)
    slot = _first_slot(a, ref)
    res = a.book_appointment(ref, slot)
    assert res["ok"] and res["appointment_id"].startswith("MHP-A-")
    a.store.close()

    b = CareFlowTools(store=RuntimeStore(db))  # a "restart": new connection, new objects
    assert b.slots[slot]["status"] == "booked" and b.booked[slot] == ref
    assert slot not in [s["slot_id"] for s in b.find_appointment_slots("ENT", patient_ref=ref, limit=20)["slots"]]
    again = b.book_appointment(ref, slot)
    assert again["ok"] is False and "no longer available" in again["error"]


def test_booking_writes_slot_appointment_and_payment_together():
    t = CareFlowTools()
    ref = _established_active(t)
    slot = _first_slot(t, ref, "Cardiology")
    before = len(t.store.rows("appointments"))
    appt = t.book_appointment(ref, slot)["appointment_id"]
    rows = t.store.rows("appointments")
    assert len(rows) == before + 1 and t.slots[slot]["status"] == "booked"
    new = next(r for r in rows if r["appointment_id"] == appt)
    assert new["patient_ref"] == ref and new["status"] == "scheduled" and new["slot_id"] == slot
    assert new["payment"] == "insurance" and isinstance(new["duration_minutes"], int)
    assert rows[0]["payment"] is None and rows[0]["slot_id"] is None  # seeded rows predate these columns


def test_self_pay_booking_and_its_staff_flag_are_saved_together(db):
    t = CareFlowTools(store=RuntimeStore(db))
    lapsed = next(p["patient_ref"] for p in t.patients.values() if p["eligibility_status"] == "lapsed")
    res = t.book_appointment(lapsed, _first_slot(t, lapsed), self_pay=True)
    fresh = RuntimeStore(db)
    row = next(r for r in fresh.rows("appointments") if r["appointment_id"] == res["appointment_id"])
    assert row["payment"] == "self_pay"
    assert [f["flag_id"] for f in fresh.flags()] == [res["flag_id"]] and fresh.flags()[0]["patient_ref"] == lapsed


def test_two_processes_cannot_double_book_a_slot(db):
    first, second = CareFlowTools(store=RuntimeStore(db)), CareFlowTools(store=RuntimeStore(db))
    ref = _established_active(first)
    slot = _first_slot(first, ref)
    assert first.book_appointment(ref, slot)["ok"]
    late = second.book_appointment(ref, slot)  # second connection, stale view is irrelevant: the write checks
    assert late["ok"] is False and "no longer available" in late["error"]
    assert sum(1 for r in RuntimeStore(db).rows("appointments") if r["slot_id"] == slot) == 1


def test_concurrent_bookings_of_one_slot_give_exactly_one_success(db):
    RuntimeStore(db)  # seed once
    ref = _established_active(CareFlowTools(store=RuntimeStore(db)))
    slot = _first_slot(CareFlowTools(store=RuntimeStore(db)), ref)
    stores = [RuntimeStore(db) for _ in range(8)]
    results, gate = [], threading.Barrier(len(stores))

    def attempt(s):
        gate.wait()
        try:
            s.book(slot, ref, "insurance")
            results.append("ok")
        except SlotUnavailable:
            results.append("taken")

    threads = [threading.Thread(target=attempt, args=(s,)) for s in stores]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(results) == ["ok"] + ["taken"] * 7
    assert sum(1 for r in RuntimeStore(db).rows("appointments") if r["slot_id"] == slot) == 1


def test_a_failure_mid_booking_rolls_everything_back(monkeypatch):
    t = CareFlowTools()
    ref = _established_active(t)
    slot = _first_slot(t, ref)
    before = len(t.store.rows("appointments"))

    def boom(c):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(t.store, "_next_appointment_id", boom)
    with pytest.raises(sqlite3.OperationalError):
        t.book_appointment(ref, slot)
    assert t.slots[slot]["status"] == "open" and len(t.store.rows("appointments")) == before


def test_flags_persist_and_stay_idempotent_across_instances(db):
    a = CareFlowTools(store=RuntimeStore(db))
    ref = _established_active(a)
    first = a.flag_for_human("asks about a dosing change", "Urgent", ref)
    b = CareFlowTools(store=RuntimeStore(db))
    again = b.flag_for_human("asks about a dosing change", "Urgent", ref)
    assert again["duplicate"] is True and again["flag_id"] == first["flag_id"]
    other = b.flag_for_human("a different reason", "Routine", ref)
    assert other["flag_id"] != first["flag_id"]
    assert [f["flag_id"] for f in RuntimeStore(db).flags()] == [first["flag_id"], other["flag_id"]]


def test_reset_restores_the_seed_and_clears_flags(db):
    s = RuntimeStore(db)
    t = CareFlowTools(store=s)
    ref = _established_active(t)
    slot = _first_slot(t, ref)
    t.book_appointment(ref, slot)
    t.flag_for_human("x")
    s.reset()
    fresh = CareFlowTools(store=s)
    assert fresh.slots[slot]["status"] == "open"
    assert s.flags() == [] and len(s.rows("appointments")) == 900


def test_in_memory_stores_are_isolated_from_each_other():
    a, b = CareFlowTools(), CareFlowTools()
    ref = _established_active(a)
    slot = _first_slot(a, ref)
    assert a.book_appointment(ref, slot)["ok"]
    assert b.slots[slot]["status"] == "open" and b.booked == {} and b.flags == []


def test_the_seeded_files_are_never_modified(tmp_path):
    before = _seed_hashes()
    s = RuntimeStore(tmp_path / "rt.db")
    t = CareFlowTools(store=s)
    ref = _established_active(t)
    t.book_appointment(ref, _first_slot(t, ref))
    t.flag_for_human("x")
    s.reset()
    s.close()
    assert _seed_hashes() == before
    assert not any(p.suffix == ".db" for p in Path(DATA_DIR).rglob("*"))


# ---- M5: human-review requests and decisions ----------------------------------------------------------
def test_review_round_trip_closes_its_flag(db):
    s = RuntimeStore(db)
    flag, _ = s.add_flag(None, "Triage review needed", "Urgent", "2026-08-01T09:00", contact_id="t1")
    review = s.add_review("t1", "human_review", ["Urgent band", "No specialty"], "2026-08-01T09:00", flag_id=flag["flag_id"])
    assert review["review_id"] == "REV-0001" and review["status"] == "pending"
    assert review["reasons"] == "Urgent band; No specialty"
    assert s.add_review("t1", "human_review", ["again"], "2026-08-01T09:05") == review  # one per thread
    assert [r["review_id"] for r in s.pending_reviews()] == ["REV-0001"]

    decided = s.decide_review("REV-0001", "RN-7", "approve", "2026-08-01T09:30", note="Called.", urgency="Priority")
    assert (decided["status"], decided["reviewer_id"], decided["decided_at"]) == ("decided", "RN-7", "2026-08-01T09:30")
    assert s.pending_reviews() == []
    assert (s.flags()[0]["status"], s.flags()[0]["urgency"]) == ("reviewed_approve", "Priority")
    # A second decision changes nothing: the first reviewer stays on record.
    assert s.decide_review("REV-0001", "RN-9", "reject", "2026-08-01T10:00") == decided
    s.close()
    again = RuntimeStore(db)  # survives a restart; re-opening does not recreate or clear the table
    assert again.review("REV-0001") == decided
    again.close()


def test_deciding_an_unknown_review_raises():
    with pytest.raises(KeyError):
        RuntimeStore(":memory:").decide_review("REV-9999", "RN-1", "approve", "2026-08-01T09:00")
