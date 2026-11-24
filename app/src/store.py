"""Transactional runtime store: SQLite seeded read-only from the toolkit's mock_api/*.csv.

The seeded CSV/JSON files are never written (the toolkit validator checks them against a fresh build).
On first use the store creates its tables from those CSVs; afterwards all state lives in SQLite:

- RuntimeStore(":memory:")  private, per-instance database (tests, one-off runs; nothing touches disk)
- RuntimeStore()            out/runtime.db (gitignored), WAL mode, shared safely by several processes
                            (the API and the MCP server)

A booking is one transaction: the slot moves from open to booked only if it is still open (so two callers
can never both book it), the appointment row is added, and any staff flag is recorded, or none of it is.
Runtime-only columns: appointments.slot_id and appointments.payment (blank on seeded rows), and the
staff_flags and reviews tables (no seeded equivalent; reviews holds human-review requests and decisions).
"""

from __future__ import annotations

import csv
import sqlite3
import threading
import time
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from .config import DATA_DIR, RUNTIME_DB
from .domain.policies import AppointmentStatus

# Column types in the toolkit tables (CSV stores everything as text).
_INT = {
    "annual_deductible_usd", "out_of_pocket_max_usd", "specialist_copay_usd", "telehealth_copay_usd",
    "copay_usd", "list_price_usd", "duration_minutes", "deductible_met_usd", "years_in_practice",
    "approval_window_business_days", "sla_business_days",
}
_FLOAT = {"coinsurance_rate"}
_BOOL = {"accepts_new_patients", "offers_telehealth", "telehealth_enabled", "covered", "preauth_required"}
# How long a connection waits for another process's write lock before sqlite3 raises "database is locked".
# The session checkpointer (agent.py) opens its SQLite file with the same limit.
BUSY_TIMEOUT_S = 5


# The states of the runtime tables. Each member equals the text stored in the column; rows are written with
# the plain string (`.value`) and read back as plain strings.
class SlotStatus(StrEnum):
    """appointment_slots.status."""

    OPEN = "open"
    BOOKED = "booked"


class FlagCategory(StrEnum):
    """staff_flags.category: a Clinical Routing Ticket (OP-CS-014) or an administrative handoff."""

    CLINICAL = "clinical"
    ADMINISTRATIVE = "administrative"


class FlagStatus(StrEnum):
    """staff_flags.status. A reviewed flag's status is "reviewed_" plus the review's action."""

    AWAITING_STAFF = "awaiting_staff"
    REVIEWED_APPROVE = "reviewed_approve"
    REVIEWED_REJECT = "reviewed_reject"


class ReviewStatus(StrEnum):
    """reviews.status."""

    PENDING = "pending"
    DECIDED = "decided"


class SlotUnavailableError(Exception):
    """The slot was booked by someone else (or never open)."""


def _convert_row_types(row: dict) -> dict:
    """A table row with its text values converted to the column's type; blank becomes None."""
    out = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = None  # e.g. approval_window_business_days when no pre-authorisation is needed
        elif key in _INT:
            out[key] = int(value)
        elif key in _FLOAT:
            out[key] = float(value)
        elif key in _BOOL:
            out[key] = value == "True"  # the CSVs spell booleans True / False
        else:
            out[key] = value
    return out


def _quote_identifier(name: str) -> str:
    """`name` quoted as an SQL identifier. Table and column names cannot be bound as parameters."""
    return '"' + name.replace('"', '""') + '"'


class RuntimeStore:
    """One SQLite connection and the clinic's tables. Rows of seeded tables come back typed (see _convert_row_types);
    staff_flags and reviews rows are returned as stored."""

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
        # isolation_level=None: no implicit transactions; every write goes through _begin_transaction.
        self.connection = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=BUSY_TIMEOUT_S
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_S * 1000}")
        if not self.in_memory:
            self._use_wal()  # readers never block the single writer
        self._seed_if_needed()
        self._migrate()

    def _use_wal(self) -> None:
        """Switch the database to write-ahead logging. The switch needs the write lock and, unlike ordinary
        statements, does not wait for it, so it is retried here for as long as the busy timeout."""
        deadline = time.monotonic() + BUSY_TIMEOUT_S
        while True:
            try:
                if self.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    self.connection.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def _migrate(self) -> None:
        """Additive schema changes on top of the seeded tables, so a database created by an older version
        of the code still opens (safe to run on every open)."""
        with self._begin_transaction() as connection:
            # Per-patient memory reads go straight to one patient's rows.
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_interactions_patient ON patient_interactions(patient_ref)")
            connection.execute("CREATE INDEX IF NOT EXISTS ix_referrals_patient ON referrals(patient_ref)")
            # One interaction row per conversation (thread); blank on seeded rows.
            if "thread_id" not in self._list_columns(connection, "patient_interactions"):
                connection.execute("ALTER TABLE patient_interactions ADD COLUMN thread_id TEXT NOT NULL DEFAULT ''")
            # Clinical vs administrative staff flags (the human-review step routes on this).
            if "category" not in self._list_columns(connection, "staff_flags"):
                connection.execute("ALTER TABLE staff_flags ADD COLUMN category TEXT NOT NULL "
                          f"DEFAULT '{FlagCategory.ADMINISTRATIVE.value}'")
            # The contact (conversation) a flag was raised in. De-duplication is per contact, so a
            # retry does not create a second ticket but a new contact always gets its own (OP-CS-014 8.2).
            if "contact_id" not in self._list_columns(connection, "staff_flags"):
                connection.execute("ALTER TABLE staff_flags ADD COLUMN contact_id TEXT")
            # One row per human-review request of the triage workflow, and the staff decision on it.
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL UNIQUE, "
                "flag_id TEXT, patient_ref TEXT, route TEXT NOT NULL, reasons TEXT NOT NULL, requested_at TEXT NOT NULL, "
                "status TEXT NOT NULL, reviewer_id TEXT, action TEXT, note TEXT, urgency TEXT, decided_at TEXT)"
            )

    @staticmethod
    def _list_columns(connection, table: str) -> set[str]:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")}

    @staticmethod
    def _find_last_serial(connection, table: str, id_column: str, prefix: str) -> int | None:
        """The highest number among `table`'s IDs of the form <prefix><digits> (e.g. "FLAG-0007" -> 7), or
        None for an empty table. Call inside _begin_transaction so the next ID cannot be handed out twice."""
        sql = f"SELECT MAX(CAST(substr({id_column}, {len(prefix) + 1}) AS INTEGER)) FROM {table}"
        return connection.execute(sql).fetchone()[0]

    @staticmethod
    def _insert(connection, table: str, row: dict) -> None:
        """Insert `row` (column -> value) into `table`."""
        columns = ", ".join(_quote_identifier(column) for column in row)
        connection.execute(
            f"INSERT INTO {_quote_identifier(table)} ({columns}) VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()))

    @contextmanager
    def _begin_transaction(self):
        """BEGIN IMMEDIATE takes the write lock up front, so check-then-write cannot interleave."""
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")

    def _seed_if_needed(self) -> None:
        """First open only: one all-text table per mock_api CSV, then the runtime-only columns and tables."""
        with self._begin_transaction() as connection:
            has_meta = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
            if has_meta and connection.execute("SELECT 1 FROM meta WHERE key='seeded'").fetchone():
                return  # another process may have seeded it first; BEGIN IMMEDIATE makes this check safe
            for path in sorted((self.seed / "mock_api").glob("*.csv")):
                with path.open(encoding="utf-8", newline="") as csv_file:
                    reader = csv.DictReader(csv_file)
                    columns = list(reader.fieldnames or [])
                    rows = [tuple(record[column] for column in columns) for record in reader]
                connection.execute(
                    f"CREATE TABLE {_quote_identifier(path.stem)} ({', '.join(_quote_identifier(column) + ' TEXT' for column in columns)})")
                marks = ", ".join("?" for _ in columns)
                connection.executemany(f"INSERT INTO {_quote_identifier(path.stem)} VALUES ({marks})", rows)
            connection.execute("ALTER TABLE appointments ADD COLUMN slot_id TEXT NOT NULL DEFAULT ''")
            connection.execute("ALTER TABLE appointments ADD COLUMN payment TEXT NOT NULL DEFAULT ''")
            connection.execute("CREATE UNIQUE INDEX ux_slots ON appointment_slots(slot_id)")
            connection.execute("CREATE UNIQUE INDEX ux_appointments ON appointments(appointment_id)")
            connection.execute("CREATE INDEX ix_appointments_patient ON appointments(patient_ref)")
            connection.execute(
                "CREATE TABLE staff_flags (flag_id TEXT PRIMARY KEY, patient_ref TEXT, reason TEXT NOT NULL, "
                "urgency TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO meta VALUES ('seeded', '1')")

    def reset(self) -> None:
        """Throw away runtime state and seed again from the CSVs."""
        with self._lock:
            self.connection.close()
            if not self.in_memory:
                for suffix in ("", "-wal", "-shm"):
                    Path(self.path + suffix).unlink(missing_ok=True)
            self._open()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    # ---- reads ----------------------------------------------------------------------------
    def list_tables(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def list_rows(self, table: str) -> list[dict]:
        """Every row of a table, typed."""
        with self._lock:
            return [_convert_row_types(dict(row))
                    for row in self.connection.execute(f"SELECT * FROM {_quote_identifier(table)}")]

    def find_one(self, table: str, key: str, value) -> dict | None:
        """The first row where column `key` equals `value` (meant for unique keys), typed, or None."""
        with self._lock:
            row = self.connection.execute(
                f"SELECT * FROM {_quote_identifier(table)} WHERE {_quote_identifier(key)} = ?", (value,)).fetchone()
        return _convert_row_types(dict(row)) if row else None

    def list_rows_for_patient(self, table: str, patient_ref: str) -> list[dict]:
        """One patient's rows only. Memory reads go through this, so another patient's rows are never loaded."""
        with self._lock:
            cursor = self.connection.execute(
                f"SELECT * FROM {_quote_identifier(table)} WHERE patient_ref = ?", (patient_ref,))
            return [_convert_row_types(dict(row)) for row in cursor]

    def list_flags(self) -> list[dict]:
        """Every staff flag, oldest first."""
        with self._lock:
            return [dict(row) for row in self.connection.execute("SELECT * FROM staff_flags ORDER BY flag_id")]

    # ---- writes ---------------------------------------------------------------------------
    def book(self, slot_id: str, patient_ref: str, payment: str, flag: dict | None = None) -> tuple[dict, dict | None]:
        """Atomically: slot open -> booked, add the appointment, record the staff flag (if any).

        `flag` holds _insert_flag's keyword arguments. Returns (appointment row, flag row or None); raises
        SlotUnavailableError, with nothing written, if the slot is not open."""
        with self._begin_transaction() as connection:
            cursor = connection.execute("UPDATE appointment_slots SET status=? WHERE slot_id=? AND status=?",
                            (SlotStatus.BOOKED.value, slot_id, SlotStatus.OPEN.value))
            if cursor.rowcount != 1:
                raise SlotUnavailableError(slot_id)
            slot = dict(connection.execute("SELECT * FROM appointment_slots WHERE slot_id=?", (slot_id,)).fetchone())
            appointment = {
                "appointment_id": self._make_next_appointment_id(connection),
                "patient_ref": patient_ref,
                "specialty": slot["specialty"],
                "provider_id": slot["provider_id"],
                "scheduled_for": slot["starts_at"],
                "status": AppointmentStatus.SCHEDULED.value,
                "duration_minutes": slot["duration_minutes"],
                "slot_id": slot_id,
                "payment": payment,
            }
            # seeded columns are text
            self._insert(connection, "appointments", {key: str(value) for key, value in appointment.items()})
            flag_row = self._insert_flag(connection, **flag)[0] if flag else None
        return _convert_row_types(appointment), flag_row

    def _make_next_appointment_id(self, connection: sqlite3.Connection) -> str:
        # seeded IDs start at 000000
        last = self._find_last_serial(connection, "appointments", "appointment_id", "MHP-A-")
        return f"MHP-A-{(last if last is not None else -1) + 1:06d}"

    def add_flag(
        self,
        patient_ref: str | None,
        reason: str,
        urgency: str,
        created_at: str,
        category: str = FlagCategory.ADMINISTRATIVE.value,
        contact_id: str | None = None,
    ) -> tuple[dict, bool]:
        """Record a staff flag. The same patient and reason within the same contact is stored once (a retry
        is not a second ticket); a later contact gets its own flag. Returns (flag, was_duplicate)."""
        with self._begin_transaction() as connection:
            return self._insert_flag(
                connection, patient_ref=patient_ref, reason=reason, urgency=urgency, created_at=created_at,
                category=category, contact_id=contact_id,
            )

    def find_interaction_for_thread(self, patient_ref: str, thread_id: str) -> dict | None:
        """The contact row already recorded for this conversation, or None."""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM patient_interactions WHERE patient_ref = ? AND thread_id = ?", (patient_ref, thread_id)
            ).fetchone()
        return _convert_row_types(dict(row)) if row else None

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
        """Record one contact in patient_interactions (long-term memory), in the seed table's format.

        With a thread_id, a conversation is one contact: the first call adds the row and later calls on the
        same thread update its topic, outcome and summary (occurred_at stays the start of the contact)."""
        with self._begin_transaction() as connection:
            if thread_id:
                existing = connection.execute(
                    "SELECT interaction_id FROM patient_interactions WHERE patient_ref = ? AND thread_id = ?",
                    (patient_ref, thread_id),
                ).fetchone()
                if existing:
                    connection.execute(
                        "UPDATE patient_interactions SET topic = ?, outcome = ?, summary = ?, referral_id = ? "
                        "WHERE interaction_id = ?",
                        (topic, outcome, summary, referral_id or "", existing[0]),
                    )
                    row = connection.execute(
                        "SELECT * FROM patient_interactions WHERE interaction_id = ?", (existing[0],))
                    return _convert_row_types(dict(row.fetchone()))
            last = self._find_last_serial(connection, "patient_interactions", "interaction_id", "MHP-I-")
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
            self._insert(connection, "patient_interactions", row)
        return _convert_row_types(row)

    # ---- human review -----------------------------------------------------------------------
    def add_review(self, thread_id: str, route: str, reasons: list[str], requested_at: str,
                   flag_id: str | None = None, patient_ref: str | None = None) -> dict:
        """Record that a workflow run is waiting for staff. One per thread: asking again returns the same row."""
        with self._begin_transaction() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE thread_id = ?", (thread_id,)).fetchone()
            if row:
                return dict(row)
            serial = (self._find_last_serial(connection, "reviews", "review_id", "REV-") or 0) + 1
            review = {
                "review_id": f"REV-{serial:04d}", "thread_id": thread_id, "flag_id": flag_id,
                "patient_ref": patient_ref, "route": route, "reasons": "; ".join(reasons), "requested_at": requested_at,
                "status": ReviewStatus.PENDING.value,
                "reviewer_id": None, "action": None, "note": None, "urgency": None, "decided_at": None,
            }
            self._insert(connection, "reviews", review)
        return review

    def decide_review(self, review_id: str, reviewer_id: str, action: str, decided_at: str,
                      note: str | None = None, urgency: str | None = None) -> dict:
        """Record the staff decision and close the review's flag, in one transaction. A review that is
        already decided is returned unchanged (a retried resume records nothing twice). A given `urgency`
        (staff re-banded the case) also replaces the flag's. Raises KeyError for an unknown review_id."""
        with self._begin_transaction() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            if row is None:
                raise KeyError(review_id)
            if row["status"] == ReviewStatus.PENDING:
                connection.execute(
                    "UPDATE reviews SET status = ?, reviewer_id = ?, action = ?, note = ?, urgency = ?, "
                    "decided_at = ? WHERE review_id = ?",
                    (ReviewStatus.DECIDED.value, reviewer_id, action, note, urgency, decided_at, review_id),
                )
                if row["flag_id"]:
                    connection.execute(
                        "UPDATE staff_flags SET status = ?, urgency = COALESCE(?, urgency) WHERE flag_id = ?",
                        (f"reviewed_{action}", urgency, row["flag_id"]),  # a FlagStatus for every ReviewAction
                    )
            return dict(connection.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone())

    def get_review(self, review_id: str) -> dict | None:
        with self._lock:
            row = self.connection.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        return dict(row) if row else None

    def list_pending_reviews(self) -> list[dict]:
        """The staff queue: undecided reviews, oldest first."""
        with self._lock:
            cursor = self.connection.execute("SELECT * FROM reviews WHERE status = ? ORDER BY review_id",
                                    (ReviewStatus.PENDING.value,))
            return [dict(row) for row in cursor]

    def promote_flag(self, flag_id: str, urgency: str, order: tuple[str, ...], prefix: str | None = None,
                     detail: str | None = None) -> dict:
        """Bring a flag someone else raised up to the workflow's view of the case: urgency is raised to
        at least `urgency` (never lowered; `order` is lowest to highest), and the repeat marker and the
        triage reasons are added to its reason if they are not there yet. `prefix` is the repeat marker,
        `detail` the triage reasons. Raises KeyError for an unknown flag_id and ValueError for an urgency
        that is not in `order`."""
        if urgency not in order:
            raise ValueError(f"unknown urgency {urgency!r}; expected one of {list(order)}")
        with self._begin_transaction() as connection:
            row = connection.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone()
            if row is None:
                raise KeyError(flag_id)
            reason = row["reason"]
            if detail and detail not in reason:
                reason = f"{reason} | {detail}"
            if prefix and not reason.lower().startswith(prefix.lower()):
                reason = f"{prefix}: {reason}"
            top = max(row["urgency"], urgency, key=lambda band: order.index(band) if band in order else -1)
            connection.execute(
                "UPDATE staff_flags SET urgency = ?, reason = ? WHERE flag_id = ?", (top, reason, flag_id))
            return dict(connection.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone())

    def get_flag(self, flag_id: str) -> dict | None:
        with self._lock:
            row = self.connection.execute("SELECT * FROM staff_flags WHERE flag_id = ?", (flag_id,)).fetchone()
        return dict(row) if row else None

    def list_flags_for_patient(self, patient_ref: str) -> list[dict]:
        with self._lock:
            cursor = self.connection.execute(
                "SELECT * FROM staff_flags WHERE patient_ref = ? ORDER BY flag_id", (patient_ref,))
            return [dict(row) for row in cursor]

    @staticmethod
    def _insert_flag(
        connection, patient_ref, reason, urgency, created_at, category=FlagCategory.ADMINISTRATIVE.value,
        contact_id=None
    ) -> tuple[dict, bool]:
        """add_flag's work inside the caller's transaction. `IS` rather than `=` so that a missing patient
        or contact (NULL) matches an earlier flag that also had none."""
        existing = connection.execute(
            "SELECT * FROM staff_flags WHERE patient_ref IS ? AND reason = ? AND contact_id IS ?",
            (patient_ref, reason, contact_id),
        ).fetchone()
        if existing:
            return dict(existing), True
        serial = (RuntimeStore._find_last_serial(connection, "staff_flags", "flag_id", "FLAG-") or 0) + 1
        flag = {
            "flag_id": f"FLAG-{serial:04d}",
            "patient_ref": patient_ref,
            "reason": reason,
            "urgency": urgency,
            "status": FlagStatus.AWAITING_STAFF.value,
            "created_at": created_at,
            "category": category,
            "contact_id": contact_id,
        }
        RuntimeStore._insert(connection, "staff_flags", flag)
        return flag, False
