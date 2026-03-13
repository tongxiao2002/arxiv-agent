"""Runtime helpers for configurable retries, timeouts, and error reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, TypeVar, Union

from arxiv_agent.utils.retry import RetryError, retry

T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeOptions:
    """Runtime settings threaded into I/O-heavy helpers."""

    max_retries: int = 5
    retry_backoff_factor: float = 2.0
    request_timeout: int = 30

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RuntimeOptions":
        """Build runtime settings from a nested config mapping."""
        if not data:
            return cls()

        return cls(
            max_retries=int(data.get("max_retries", cls.max_retries)),
            retry_backoff_factor=float(
                data.get("retry_backoff_factor", cls.retry_backoff_factor)
            ),
            request_timeout=int(data.get("request_timeout", cls.request_timeout)),
        )


def call_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_retries: int,
    backoff_factor: float,
    retry_on: Union[type[Exception], tuple[type[Exception], ...]] = Exception,
    respect_retry_after: bool = True,
) -> T:
    """Execute an operation with configurable retry settings."""

    def _operation() -> T:
        return operation()

    _operation.__name__ = operation_name
    wrapped = retry(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        jitter=True,
        retry_on=retry_on,
        respect_retry_after=respect_retry_after,
    )(_operation)
    return wrapped()


def describe_retry_error(exc: RetryError, action: str) -> str:
    """Return an operator-facing message that keeps the root cause visible."""
    if exc.last_exception:
        return f"{action}: {exc.last_exception}"
    return f"{action}: {exc}"
