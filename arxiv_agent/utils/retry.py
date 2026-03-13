"""Exponential backoff retry logic for network operations."""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        self.last_exception = last_exception
        super().__init__(message)

    def __str__(self) -> str:
        """Include the final underlying exception when available."""
        message = super().__str__()
        if self.last_exception is None:
            return message
        return f"{message}. Last error: {self.last_exception}"


def retry(
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Union[Type[Exception], tuple] = Exception,
    respect_retry_after: bool = True,
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts (including initial call)
        backoff_factor: Exponential backoff factor (e.g., 2 = 1, 2, 4, 8 seconds)
        jitter: Whether to add random jitter to avoid thundering herd
        retry_on: Exception type(s) that trigger retry (default: all exceptions)
        respect_retry_after: Whether to respect Retry-After headers (if present)

    Returns:
        Decorated function that retries on specified exceptions
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.debug(
                            f"Retry attempt {attempt}/{max_retries-1} for {func.__name__}"
                        )
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    # Check for Retry-After header if present
                    retry_after = None
                    if respect_retry_after and hasattr(e, "response"):
                        try:
                            retry_after_header = e.response.headers.get("Retry-After")
                            if retry_after_header:
                                retry_after = int(retry_after_header)
                        except (AttributeError, ValueError):
                            pass

                    # Calculate delay with exponential backoff
                    if attempt < max_retries - 1:
                        delay = _calculate_backoff(
                            attempt,
                            backoff_factor,
                            jitter,
                            retry_after=retry_after,
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: "
                            f"{e}. Retrying in {delay:.2f} seconds."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )
                        raise RetryError(
                            f"Function {func.__name__} failed after {max_retries} attempts",
                            last_exception=last_exception,
                        ) from last_exception

            # This should never be reached due to raise in else clause
            raise RetryError(
                f"Function {func.__name__} failed after {max_retries} attempts",
                last_exception=last_exception,
            )

        return wrapper

    return decorator


class retry_context:
    """
    Context manager for exponential backoff retry logic.

    Example:
        with retry_context(max_retries=3) as ctx:
            while ctx.attempts < ctx.max_retries:
                try:
                    result = do_something()
                    break
                except Exception as e:
                    ctx.handle_exception(e)
    """

    def __init__(
        self,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        respect_retry_after: bool = True,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.respect_retry_after = respect_retry_after
        self.attempts = 0
        self.last_exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def handle_exception(self, exception: Exception) -> None:
        """
        Handle an exception that occurred during an operation.

        Args:
            exception: The exception that was raised

        Raises:
            RetryError: If max retries are exhausted
        """
        self.last_exception = exception
        self.attempts += 1

        if self.attempts >= self.max_retries:
            raise RetryError(
                f"Operation failed after {self.max_retries} attempts",
                last_exception=self.last_exception,
            ) from exception

        # Check for Retry-After header
        retry_after = None
        if self.respect_retry_after and hasattr(exception, "response"):
            try:
                retry_after_header = exception.response.headers.get("Retry-After")
                if retry_after_header:
                    retry_after = int(retry_after_header)
            except (AttributeError, ValueError):
                pass

        delay = _calculate_backoff(
            self.attempts - 1,  # -1 because attempts is already incremented
            self.backoff_factor,
            self.jitter,
            retry_after=retry_after,
        )
        logger.warning(
            f"Attempt {self.attempts}/{self.max_retries} failed: "
            f"{exception}. Retrying in {delay:.2f} seconds."
        )
        time.sleep(delay)


def _calculate_backoff(
    attempt: int,
    backoff_factor: float,
    jitter: bool,
    retry_after: Optional[int] = None,
) -> float:
    """
    Calculate backoff delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        backoff_factor: Exponential backoff factor
        jitter: Whether to add random jitter
        retry_after: Retry-After header value (seconds)

    Returns:
        Delay in seconds
    """
    if retry_after is not None:
        return float(retry_after)

    # Exponential backoff: backoff_factor^attempt
    delay = backoff_factor**attempt

    # Add jitter: random value between 0 and delay * 0.1
    if jitter:
        delay += random.uniform(0, delay * 0.1)

    return delay


# Convenience function for easy imports
def with_retry(
    func: Callable,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Union[Type[Exception], tuple] = Exception,
    respect_retry_after: bool = True,
) -> Callable:
    """
    Apply retry decorator to a function.

    Convenience function for applying retry logic without decorator syntax.

    Args:
        func: Function to wrap with retry logic
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        jitter: Whether to add random jitter
        retry_on: Exception type(s) that trigger retry
        respect_retry_after: Whether to respect Retry-After headers

    Returns:
        Function wrapped with retry logic
    """
    return retry(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retry_on=retry_on,
        respect_retry_after=respect_retry_after,
    )(func)
