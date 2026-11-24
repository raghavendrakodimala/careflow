"""Injectable time. Tests and evals use the toolkit's frozen "today"; a live service would use SystemClock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

FROZEN_NOW = datetime(2026, 8, 1, 9, 0)  # the toolkit's frozen "today"


class Clock(Protocol):
    """Anything with now(). Times are naive clinic-local datetimes, like the timestamps in the data."""

    def now(self) -> datetime: ...


@dataclass(frozen=True)
class FrozenClock:
    """Always the same instant, so results that depend on "today" are repeatable."""

    at: datetime = FROZEN_NOW

    def now(self) -> datetime:
        return self.at


class SystemClock:
    """The machine's local wall-clock time (naive)."""

    def now(self) -> datetime:
        return datetime.now()
