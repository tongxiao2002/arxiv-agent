"""Utilities for run-once date intervals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


@dataclass(frozen=True)
class RunOnceInterval:
    """Date-based interval used by run-once backfills."""

    start_date: date
    end_date: date
    query_start: datetime
    query_end: datetime

    @classmethod
    def from_dates(
        cls,
        start: date,
        end: date,
        *,
        max_span_days: int = 31,
    ) -> "RunOnceInterval":
        """Create a validated interval from direct date inputs."""
        if start > end:
            raise ValueError("--from must be earlier than or equal to --to")
        if end - start > timedelta(days=max_span_days):
            raise ValueError(f"run-once intervals cannot exceed {max_span_days} days")

        return cls(
            start_date=start,
            end_date=end,
            query_start=datetime.combine(start, time.min),
            query_end=datetime.combine(end, time.min),
        )

    def contains(self, value: datetime | None) -> bool:
        """Return True when a timestamp falls inside the direct arXiv date window."""
        if value is None:
            return False
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return self.query_start <= value <= self.query_end

    def storage_date_for(self, value: datetime) -> date:
        """Return the raw arXiv publication date without local-time conversion."""
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.date()

    def iter_dates(self) -> list[date]:
        """Return all dates touched by the interval."""
        current = self.start_date
        dates = []
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def gmt_bounds(self) -> tuple[str, str]:
        """Return the direct arXiv query bounds."""
        return (
            self.query_start.strftime("%Y%m%d%H%M"),
            self.query_end.strftime("%Y%m%d%H%M"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the interval for workflow results and logging."""
        query_start, query_end = self.gmt_bounds()
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "query_start": self.query_start.isoformat(),
            "query_end": self.query_end.isoformat(),
            "query_start_gmt_minute": query_start,
            "query_end_gmt_minute": query_end,
        }
