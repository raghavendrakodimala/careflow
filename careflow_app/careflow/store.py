"""Transactional runtime store: SQLite seeded read-only from the toolkit's mock_api/*.csv (architecture ADR-02).

The seeded CSV/JSON files are never written (the toolkit validator checks them against a fresh build).
On first use the store creates its tables from those CSVs; afterwards all state lives in SQLite:

- RuntimeStore(":memory:")  private, per-instance database (tests, one-off runs; nothing touches disk)
- RuntimeStore()            out/runtime.db (gitignored), WAL mode, shared safely by several processes
                            (the M8 API and the M6 MCP server)

A booking is one transaction: the slot moves from open to booked only if it is still open (so two callers
can never both book it), the appointment row is added, and any staff flag is recorded, or none of it is.
Runtime-only columns: appointments.slot_id and appointments.payment (blank on seeded rows), and the
staff_flags table (no seeded equivalent). M5 adds the reviews table (human-review requests and decisions).
"""

from __future__ import annotations

import csv
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import DATA_DIR, RUNTIME_DB

# Column types in the toolkit tables (CSV stores everything as text).
_INT = {
    "annual_deductible_usd", "out_of_pocket_max_usd", "specialist_copay_usd", "telehealth_copay_usd",
    "copay_usd", "list_price_usd", "duration_minutes", "deductible_met_usd", "years_in_practice",
    "approval_window_business_days", "sla_business_days",
}
_FLOAT = {"coinsurance_rate"}
_BOOL = {"accepts_new_patients", "offers_telehealth", "telehealth_enabled", "covered", "preauth_required"}


class SlotUnavailable(Exception):
    """The slot was booked by someone else (or never open)."""


def _typed(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = None  # e.g. approval_window_business_days when no pre-authorisation is needed
        elif key in _INT:
            out[key] = int(value)
        elif key in _FLOAT:
            out[key] = float(value)
        elif key in _BOOL:
            out[key] = value == "True"
        else:
            out[key] = value
    return out


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class RuntimeStore:
    def __init__(self, path: Path | str | None = None, seed_dir: Path | str | None = None):
        self.path = str(path) if path else str(RUNTIME_DB)
        self.seed = Path(seed_dir) if seed_dir else DATA_DIR
        self._lock = threading.RLock()  # one connection per store; serialise its use across threads
        self._open()

    @property
    def in_memory(self) -> bool:
        return self.path == ":memory:"

    # ---- connection and seeding -------------------------------------------------------------
    def _open(self) -> None:
        if not self.in_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=5)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=5000")
        if not self.in_memory:
            self.conn.execute("PRAGMA journal_mode=WAL")  # readers never block the single writer
        self._seed_if_needed()
        self._migrate()

    def _migrate(self) -> None:
        """Additive changes for databases seeded by an earlier milestone (safe to run on every open)."""
        with self._tx() as c:
            # M3: per-patient memory reads go straight to one patient's rows.
            c.execute("CREATE INDEX IF NOT EXISTS ix_interactions_patient ON patient_interactions(patient_ref)")
            c.execute("CREATE INDEX IF NOT EXISTS ix_referrals_patient ON referrals(patient_ref)")
            # M3: one interaction row per conversation (thread); blank on seeded rows.
            if "thread_id" not in self._columns(c, "patient_interactions"):
                c.execute("ALTER TABLE patient_interactions ADD COLUMN thread_id TEXT NOT NULL DEFAULT ''")
            # M3: clinical vs administrative staff flags (M5's human-in-the-loop step routes on this).
            if "category" not in self._columns(c, "staff_flags"):
                c.execute("ALTER TABLE staff_flags ADD COLUMN category TEXT NOT NULL DEFAULT 'administrative'")
            # M3: the contact (conversation) a flag was raised in. De-duplication is per contact, so a
            # retry does not create a second ticket but a new contact always gets its own (OP-CS-014 8.2).
            if "contact_id" not in self._columns(c, "staff_flags"):
                c.execute("ALTER TABLE staff_flags ADD COLUMN contact_id TEXT")
            # M5: one row per human-review request of the triage workflow, and the staff decision on it.
            c.execute(
                "CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL UNIQUE, "
                "flag_id TEXT, patient_ref TEXT, route TEXT NOT NULL, reasons TEXT NOT NULL, requested_at TEXT NOT NULL, "
                "status TEXT NOT NULL, reviewer_id TEXT, action TEXT, note TEXT, urgency TEXT, decided_at TEXT)"
            )

    @staticmethod
    def _columns(c, table: str) -> set[str]:
        return {r[1] for r in c.execute(f"PRAGMA table_info({_q(table)})")}

    @contextmanager
    def _tx(self):
        """BEGIN IMMEDIATE takes the write lock up front, so check-then-write cannot interleave."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def _seed_if_needed(self) -> None:
        with self._tx() as c:
            has_meta = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
            if has_meta and c.execute("SELECT 1 FROM meta WHERE key='seeded'").fetchone():
                return  # another process may have seeded it first; BEGIN IMMEDIATE makes this check safe
            for path in sorted((self.seed / "mock_api").glob("*.csv")):
                with path.open(encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    cols = list(reader.fieldnames or [])
                    rows = [tuple(r[k] for k in cols) for r in reader]
                c.execute(f"CREATE TABLE {_q(path.stem)} ({', '.join(_q(col) + ' TEXT' for col in cols)})")
                marks = ", ".join("?" for _ in cols)
                c.executemany(f"INSERT INTO {_q(path.stem)} VALUES ({marks})", rows)
            c.execute("ALTER TABLE appointments ADD COLUMN slot_id TEXT NOT NULL DEFAULT ''")
            c.execute("ALTER TABLE appointments ADD COLUMN payment TEXT NOT NULL DEFAULT ''")
            c.execute("CREATE UNIQUE INDEX ux_slots ON appointment_slots(slot_id)")
            c.execute("CREATE UNIQUE INDEX ux_appointments ON appointments(appointment_id)")
            c.execute("CREATE INDEX ix_appointments_patient ON appointments(patient_ref)")
            c.execute(
                "CREATE TABLE staff_flags (flag_id TEXT PRIMARY KEY, patient_ref TEXT, reason TEXT NOT NULL, "
                "urgency TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("INSERT INTO meta VALUES ('seeded', '1')")

    def reset(self) -> None:
        """Throw away runtime state and seed again from the CSVs."""
        with self._lock:
            self.conn.close()
            if not self.in_memory:
                for suffix in ("", "-wal", "-shm"):
                    Path(self.path + suffix).unlink(missing_ok=True)
            self._open()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---- reads ----------------------------------------------------------------------------
    def tables(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(self, table: str) -> list[dict]:
        with self._lock:
            return [_typed(dict(r)) for r in self.conn.execute(f"SELECT * FROM {_q(table)}")]

    def one(self, table: str, key: str, value) -> dict | None:
        with self._lock:
            row = self.conn.execute(f"SELECT * FROM {_q(table)} WHERE {_q(key)} = ?", (value,)).fetchone()
        return _typed(dict(row)) if row else None

    def rows_for_patient(self, table: str, patient_ref: str) -> list[dict]:
        """One patient's rows only. Memory reads go through this, so another patient's rows are never loaded."""
        with self._lock:
            cur = self.conn.execute(f"SELECT * FROM {_q(table)} WHERE patient_ref = ?", (patient_ref,))
            return [_typed(dict(r)) for r in cur]

    def flags(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM staff_flags ORDER BY flag_id")]

    # ---- writes ---------------------------------------------------------------------------
    def book(self, slot_id: str, patient_ref: str, payment: str, flag: dict | None = None) -> tuple[dict, dict | None]:
        """Atomically: slot open -> booked, add the appointment, record the staff flag (if any)."""
        with self._tx() as c:
            cur = c.execute("UPDATE appointment_slots SET status='booked' WHERE slot_id=? AND status='open'", (slot_id,))
            if cur.rowcount != 1:
                raise SlotUnavailable(slot_id)
            slot = dict(c.execute("SELECT * FROM appointment_slots WHERE slot_id=?", (slot_id,)).fetchone())
            appointment = {
                "appointment_id": self._next_appointment_id(c),
                "patient_ref": patient_ref,
                "specialty": slot["specialty"],
                "provider_id": slot["provider_id"],
                "scheduled_for": slot["starts_at"],
                "status": "scheduled",
                "duration_minutes": slot["duration_minutes"],
                "slot_id": slot_id,
                "payment": payment,
            }
            cols = ", ".join(_q(k) for k in appointment)
            c.execute(
                f"INSERT INTO appointments ({cols}) VALUES ({', '.join('?' for _ in appointment)})",
                [str(v) for v in appointment.values()],
            )
            flag_row = self._insert_flag(c, **flag)[0] if flag else None
        return _typed(appointment), flag_row

    def _next_appointment_id(self, c: sqlite3.Connection) -> str:
        last = c.execute("SELECT MAX(CAST(substr(appointment_id, 7) AS INTEGER)) FROM appointments").fetchone()[0]
        return f"MHP-A-{(last if last is not None else -1) + 1:06d}"

    def add_flag(
        self,
        patient_ref: str | None,
        reason: str,
        urgency: str,
        created_at: str,
        category: str = "administrative",
        contact_id: str | None = None,
    ) -> tuple[dict, bool]:
        """Record a staff flag. The same patient and reason within the same contact is stored once (a retry
        is not a second ticket); a later contact gets its own flag. Returns (flag, was_duplicate)."""
        with self._tx() as c:
            return self._insert_flag(
                c, patient_ref=patient_ref, reason=reason, urgency=urgency, created_at=created_at, category=category,
                contact_id=contact_id,
            )

    def interaction_for_thread(self, patient_ref: str, thread_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM patient_interactions WHERE patient_ref = ? AND thread_id = ?", (patient_ref, thread_id)
            ).fetchone()
        return _typed(dict(row)) if row else None

    def add_interaction(
        self,
        patient_ref: str,
        occurred_at: str,
        channel: str,
        topic: str,
        outcome: str,
        summary: str,
        referral_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        """Record one contact in patient_interactions (M3 long-term memory), in the seed table's format.

        With a thread_id, a conversation is one contact: the first call adds the row and later calls on the
        same thread update its topic, outcome and summary (occurred_at stays the start of the contact)."""
        with self._tx() as c:
            if thread_id:
                existing = c.execute(
                    "SELECT interaction_id FROM patient_interactions WHERE patient_ref = ? AND thread_id = ?",
                    (patient_ref, thread_id),
                ).fetchone()
                if existing:
                    c.execute(
                        "UPDATE patient_interactions SET topic = ?, outcome = ?, summary = ?, referral_id = ? "
                        "WHERE interaction_id = ?",
                        (topic, outcome, summary, referral_id or "", existing[0]),
                    )
                    row = c.execute("SELECT * FROM patient_interactions WHERE interaction_id = ?", (existing[0],))
                    return _typed(dict(row.fetchone()))
            last = c.execute(
                "SELECT MAX(CAST(substr(interaction_id, 7) AS INTEGER)) FROM patient_interactions"
            ).fetchone()[0]
            row = {
                "interaction_id": f"MHP-I-{(last if last is not None else -1) + 1:05d}",
                "patient_ref": patient_ref,
                "occurred_at": occurred_at,
                "channel": channel,
                "topic": topic,
                "outcome": outcome,
                "summary": summary,
                "referral_id": referral_id or "",
                "thread_id": thread_id or "",
            }
            c.execute(
                f"INSERT INTO patient_interactions ({', '.join(_q(k) for k in row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
        return _typed(row)

    # ---- human review (M5) ------------------------------------------------------------------
    def add_review(self, thread_id: str, route: str, reasons: list[str], requested_at: str,
                   flag_id: str | None = None, patient_ref: str | None = None) -> dict:
        """Record that a workflow run is waiting for staff. One per thread: asking again returns the same row."""
        with self._tx() as c:
            row = c.execute("SELECT * FROM reviews WHERE thread_id = ?", (thread_id,)).fetchone()
            if row:
                return dict(row)
            n = (c.execute("SELECT MAX(CAST(substr(review_id, 5) AS INTEGER)) FROM reviews").fetchone()[0] or 0) + 1
            review = {
                "review_id": f"REV-{n:04d}", "thread_id": thread_id, "flag_id": flag_id, "patient_ref": patient_ref,
                "route": route, "reasons": "; ".join(reasons), "requested_at": requested_at, "status": "pending",
                "reviewer_id": None, "action": None, "note": None, "urgency": None, "decided_at": None,
            }
            c.execute(
                f"INSERT INTO reviews ({', '.join(review)}) VALUES ({', '.join('?' * len(review))})",
                tuple(review.values()),
            )
        return review

    def decide_review(self, review_id: str, reviewer_id: str, action: str, decided_at: str,
                      note: str | None = None, urgency: str | None = None) -> dict:
        """Record the staff decision and close the review's flag, in one transaction. A review that is
        already decided is returned unchanged (a retried resume records nothing twice)."""
        with self._tx() as c:
            row = c.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            if row is None:
                raise KeyError(review_id)
            if row["status"] == "pending":
                c.execute(
                    "UPDATE reviews SET status = 'decided', reviewer_id = ?, action = ?, note = ?, urgency = ?, "
                    "decided_at = ? WHERE review_id = ?",
                    (reviewer_id, action, note, urgency, decided_at, review_id),
                )
                if row["flag_id"]:
                    c.execute(
                        "UPDATE staff_flags SET status = ?, urgency = COALESCE(?, urgency) WHERE flag_id = ?",
                        (f"reviewed_{action}", urgency, row["flag_id"]),
                    )
            return dict(c.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone())

    def review(self, review_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        return dict(row) if row else None

    def pending_reviews(self) -> list[dict]:
        """The staff queue: undecided reviews, oldest first."""
        with self._lock:
            cur = self.conn.execute("SELECT * FROM reviews WHERE status = 'pending' ORDER BY review_id")
            return [dict(r) for r in cur]

    def promote_flag(self, flag_id: str, urgency: str, order: tuple[str, ...], prefix: str | None = None,
                     detail: str | None = None) -> dict:
        """Bring a flag someone else raised up to the workflow's view of the case: urgency is raised to
        at least `urgency` (never lowered; `order` is lowest to highest), and the repeat marker and the
        triage reasons are added to its reason if they are not there yet."""
        with self._tx() as c:
            row = c.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone()
            if row is None:
                raise KeyError(flag_id)
            reason = row["reason"]
            if detail and detail not in reason:
                reason = f"{reason} | {detail}"
            if prefix and not reason.lower().startswith(prefix.lower()):
                reason = f"{prefix}: {reason}"
            top = max(row["urgency"], urgency, key=lambda u: order.index(u) if u in order else -1)
            c.execute("UPDATE staff_flags SET urgency = ?, reason = ? WHERE flag_id = ?", (top, reason, flag_id))
            return dict(c.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone())

    def flag(self, flag_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone()
        return dict(row) if row else None

    def flags_for_patient(self, patient_ref: str) -> list[dict]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM staff_flags WHERE patient_ref = ? ORDER BY flag_id", (patient_ref,))
            return [dict(r) for r in cur]

    @staticmethod
    def _insert_flag(
        c, patient_ref, reason, urgency, created_at, category="administrative", contact_id=None
    ) -> tuple[dict, bool]:
        existing = c.execute(
            "SELECT * FROM staff_flags WHERE patient_ref IS ? AND reason = ? AND contact_id IS ?",
            (patient_ref, reason, contact_id),
        ).fetchone()
        if existing:
            return dict(existing), True
        n = (c.execute("SELECT MAX(CAST(substr(flag_id, 6) AS INTEGER)) FROM staff_flags").fetchone()[0] or 0) + 1
        flag = {
            "flag_id": f"FLAG-{n:04d}",
            "patient_ref": patient_ref,
            "reason": reason,
            "urgency": urgency,
            "status": "awaiting_staff",
            "created_at": created_at,
            "category": category,
            "contact_id": contact_id,
        }
        c.execute(
            f"INSERT INTO staff_flags ({', '.join(flag)}) VALUES ({', '.join('?' * len(flag))})", tuple(flag.values())
        )
        return flag, False
