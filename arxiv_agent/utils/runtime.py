"""Runtime helpers for configurable retries, timeouts, and error reporting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, TypeVar, Union

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


async def call_with_retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    max_retries: int,
    backoff_factor: float,
    retry_on: Union[type[Exception], tuple[type[Exception], ...]] = Exception,
    respect_retry_after: bool = True,
) -> T:
    """Execute an async operation with configurable retry settings."""
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await operation()
        except retry_on as exc:
            last_exception = exc

            retry_after = None
            if respect_retry_after and hasattr(exc, "response"):
                try:
                    retry_after_header = exc.response.headers.get("Retry-After")
                    if retry_after_header:
                        retry_after = int(retry_after_header)
                except (AttributeError, ValueError):
                    pass

            if attempt < max_retries - 1:
                delay = (
                    float(retry_after)
                    if retry_after is not None
                    else backoff_factor**attempt
                )
                await asyncio.sleep(delay)
            else:
                raise RetryError(
                    f"Function {operation_name} failed after {max_retries} attempts",
                    last_exception=last_exception,
                ) from last_exception

    raise RetryError(
        f"Function {operation_name} failed after {max_retries} attempts",
        last_exception=last_exception,
    )


def describe_retry_error(exc: RetryError, action: str) -> str:
    """Return an operator-facing message that keeps the root cause visible."""
    if exc.last_exception:
        return f"{action}: {exc.last_exception}"
    return f"{action}: {exc}"
