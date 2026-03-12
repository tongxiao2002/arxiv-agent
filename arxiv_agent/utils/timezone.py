"""Timezone helpers for scheduling and digest formatting."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def is_valid_timezone(timezone_name: str) -> bool:
    """Return True when the timezone name is a valid IANA timezone."""
    try:
        ZoneInfo(timezone_name)
        return True
    except ZoneInfoNotFoundError:
        return False


def get_timezone(timezone_name: str) -> ZoneInfo:
    """Return the ZoneInfo object for an IANA timezone name."""
    return ZoneInfo(timezone_name)


def get_current_date_in_timezone(timezone_name: str) -> date:
    """Return today's date in the configured timezone."""
    return datetime.now(get_timezone(timezone_name)).date()


def format_digest_date(target_date: date, timezone_name: str) -> str:
    """Format a digest date for subjects and email bodies."""
    get_timezone(timezone_name)
    return target_date.strftime("%B %d, %Y")
