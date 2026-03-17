"""Utilities for run-once local datetime intervals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from arxiv_agent.utils.timezone import get_timezone


def _floor_to_minute(value: datetime) -> datetime:
    """Floor an aware datetime to minute precision."""
    return value.replace(second=0, microsecond=0)


def _ceil_to_minute(value: datetime) -> datetime:
    """Ceil an aware datetime to minute precision."""
    floored = _floor_to_minute(value)
    if value == floored:
        return floored
    return floored + timedelta(minutes=1)


@dataclass(frozen=True)
class RunOnceInterval:
    """Closed local datetime interval used by run-once backfills."""

    local_start: datetime
    local_end: datetime
    timezone_name: str
    query_start_utc: datetime
    query_end_utc: datetime

    @classmethod
    def from_local_naive(
        cls,
        start: datetime,
        end: datetime,
        timezone_name: str,
        *,
        max_span_days: int = 31,
    ) -> "RunOnceInterval":
        """Create a validated interval from naive local datetimes."""
        if start.tzinfo is not None or end.tzinfo is not None:
            raise ValueError("--from and --to must be naive local datetimes")
        if start > end:
            raise ValueError("--from must be earlier than or equal to --to")
        if end - start > timedelta(days=max_span_days):
            raise ValueError(f"run-once intervals cannot exceed {max_span_days} days")

        tzinfo = get_timezone(timezone_name)
        local_start = start.replace(tzinfo=tzinfo)
        local_end = end.replace(tzinfo=tzinfo)
        start_utc = local_start.astimezone(timezone.utc)
        end_utc = local_end.astimezone(timezone.utc)
        return cls(
            local_start=local_start,
            local_end=local_end,
            timezone_name=timezone_name,
            query_start_utc=_floor_to_minute(start_utc),
            query_end_utc=_ceil_to_minute(end_utc),
        )

    def contains(self, value: datetime | None) -> bool:
        """Return True when a timestamp falls inside the closed local interval."""
        if value is None:
            return False
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        local_value = value.astimezone(get_timezone(self.timezone_name))
        return self.local_start <= local_value <= self.local_end

    def local_date_for(self, value: datetime) -> date:
        """Convert a paper timestamp into the configured local date."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(get_timezone(self.timezone_name)).date()

    def iter_local_dates(self) -> list[date]:
        """Return all local dates touched by the interval."""
        current = self.local_start.date()
        end_date = self.local_end.date()
        dates = []
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def gmt_bounds(self) -> tuple[str, str]:
        """Return arXiv GMT minute query bounds."""
        return (
            self.query_start_utc.strftime("%Y%m%d%H%M"),
            self.query_end_utc.strftime("%Y%m%d%H%M"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the interval for workflow results and logging."""
        query_start, query_end = self.gmt_bounds()
        return {
            "local_start": self.local_start.isoformat(),
            "local_end": self.local_end.isoformat(),
            "timezone": self.timezone_name,
            "query_start_utc": self.query_start_utc.isoformat(),
            "query_end_utc": self.query_end_utc.isoformat(),
            "query_start_gmt_minute": query_start,
            "query_end_gmt_minute": query_end,
        }
