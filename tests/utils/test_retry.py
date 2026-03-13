"""Tests for retry utility."""

import random
import time
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.utils.retry import (
    RetryError,
    _calculate_backoff,
    retry,
    retry_context,
    with_retry,
)


def test_calculate_backoff_basic():
    """Test basic exponential backoff calculation."""
    # No jitter, attempt 0
    delay = _calculate_backoff(0, backoff_factor=2.0, jitter=False)
    assert delay == 1.0  # 2^0 = 1

    # No jitter, attempt 1
    delay = _calculate_backoff(1, backoff_factor=2.0, jitter=False)
    assert delay == 2.0  # 2^1 = 2

    # No jitter, attempt 2
    delay = _calculate_backoff(2, backoff_factor=2.0, jitter=False)
    assert delay == 4.0  # 2^2 = 4

    # Different factor
    delay = _calculate_backoff(2, backoff_factor=3.0, jitter=False)
    assert delay == 9.0  # 3^2 = 9


def test_calculate_backoff_with_jitter():
    """Test backoff calculation with jitter."""
    with patch.object(random, "uniform") as mock_uniform:
        mock_uniform.return_value = 0.5
        delay = _calculate_backoff(2, backoff_factor=2.0, jitter=True)
        # Base delay = 4.0, jitter adds random(0, 0.4) = 0.5 (mocked)
        assert delay == 4.5
        mock_uniform.assert_called_once_with(0, 0.4)


def test_calculate_backoff_with_retry_after():
    """Test backoff calculation with Retry-After header."""
    # When retry_after is provided, it should be used directly
    delay = _calculate_backoff(
        attempt=2,
        backoff_factor=2.0,
        jitter=True,
        retry_after=30,
    )
    assert delay == 30.0


def test_retry_decorator_success():
    """Test retry decorator when function succeeds on first attempt."""
    call_count = 0

    @retry(max_retries=3)
    def successful_func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = successful_func()
    assert result == "success"
    assert call_count == 1


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_decorator_retries_then_succeeds(mock_sleep):
    """Test retry decorator when function succeeds after retries."""
    call_count = 0

    @retry(max_retries=3, backoff_factor=1.0, jitter=False)
    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    result = flaky_func()
    assert result == "success"
    assert call_count == 3
    assert mock_sleep.call_count == 2  # Sleep after first two failures


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_decorator_exhausts_retries(mock_sleep):
    """Test retry decorator when all retries are exhausted."""
    call_count = 0

    @retry(max_retries=3, backoff_factor=1.0, jitter=False)
    def always_failing_func():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Persistent failure")

    with pytest.raises(RetryError) as exc_info:
        always_failing_func()

    assert call_count == 3
    assert "Function always_failing_func failed after 3 attempts" in str(exc_info.value)
    assert isinstance(exc_info.value.last_exception, RuntimeError)
    assert mock_sleep.call_count == 2  # Sleep after first two failures


def test_retry_decorator_custom_exception_types():
    """Test retry decorator with custom exception types."""
    call_count = 0

    @retry(max_retries=3, retry_on=(ValueError, TypeError))
    def func_with_specific_exceptions():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Retry on this")
        elif call_count == 2:
            raise TypeError("Also retry on this")
        else:
            return "success"

    # Should retry on ValueError and TypeError
    result = func_with_specific_exceptions()
    assert result == "success"
    assert call_count == 3

    # Exception not in retry_on should propagate immediately
    @retry(max_retries=2, retry_on=ValueError)
    def func_with_unretryable_exception():
        raise KeyError("Not retryable")

    with pytest.raises(KeyError):
        func_with_unretryable_exception()


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_decorator_respect_retry_after(mock_sleep):
    """Test retry decorator with Retry-After header."""

    class MockResponse:
        headers = {"Retry-After": "5"}

    class MockException(Exception):
        response = MockResponse()

    call_count = 0

    @retry(max_retries=3, respect_retry_after=True)
    def func_with_retry_after():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockException("Rate limited")
        return "success"

    result = func_with_retry_after()
    assert result == "success"
    assert call_count == 3
    # Should use Retry-After value (5 seconds) instead of exponential backoff
    # First call fails, sleeps 5 seconds, second call fails, sleeps 5 seconds
    mock_sleep.assert_called_with(5.0)
    assert mock_sleep.call_count == 2


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_decorator_no_respect_retry_after(mock_sleep):
    """Test retry decorator ignoring Retry-After header."""

    class MockResponse:
        headers = {"Retry-After": "5"}

    class MockException(Exception):
        response = MockResponse()

    call_count = 0

    @retry(max_retries=3, respect_retry_after=False, jitter=False)
    def func_with_retry_after():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockException("Rate limited")
        return "success"

    result = func_with_retry_after()
    assert result == "success"
    assert call_count == 3
    # Should use exponential backoff (1s, 2s) instead of Retry-After
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
    assert mock_sleep.call_count == 2


def test_retry_context_success():
    """Test retry context manager when operation succeeds."""
    with retry_context(max_retries=3) as ctx:
        attempts = 0
        while attempts < ctx.max_retries:
            try:
                result = "success"
                break
            except Exception as e:
                ctx.handle_exception(e)
            attempts += 1

    assert result == "success"
    assert ctx.attempts == 0


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_context_retries_then_succeeds(mock_sleep):
    """Test retry context manager when operation succeeds after retries."""
    call_count = 0

    with retry_context(max_retries=3, backoff_factor=1.0, jitter=False) as ctx:
        while True:
            try:
                call_count += 1
                if call_count < 3:
                    raise ValueError("Temporary failure")
                result = "success"
                break
            except Exception as e:
                ctx.handle_exception(e)

    assert result == "success"
    assert call_count == 3
    assert ctx.attempts == 2  # Two failures handled
    assert mock_sleep.call_count == 2


@patch("arxiv_agent.utils.retry.time.sleep")
def test_retry_context_exhausts_retries(mock_sleep):
    """Test retry context manager when all retries are exhausted."""
    call_count = 0

    with pytest.raises(RetryError) as exc_info:
        with retry_context(max_retries=3, backoff_factor=1.0, jitter=False) as ctx:
            while True:
                try:
                    call_count += 1
                    raise RuntimeError("Persistent failure")
                except Exception as e:
                    ctx.handle_exception(e)

    assert call_count == 3
    assert "Operation failed after 3 attempts" in str(exc_info.value)
    assert isinstance(exc_info.value.last_exception, RuntimeError)
    assert mock_sleep.call_count == 2


def test_retry_context_retry_after():
    """Test retry context manager with Retry-After header."""

    class MockResponse:
        headers = {"Retry-After": "10"}

    class MockException(Exception):
        response = MockResponse()

    call_count = 0

    with patch("arxiv_agent.utils.retry.time.sleep") as mock_sleep:
        with retry_context(max_retries=3, respect_retry_after=True) as ctx:
            while True:
                try:
                    call_count += 1
                    if call_count < 3:
                        raise MockException("Rate limited")
                    result = "success"
                    break
                except Exception as e:
                    ctx.handle_exception(e)

    assert result == "success"
    assert call_count == 3
    mock_sleep.assert_called_with(10.0)
    assert mock_sleep.call_count == 2


def test_with_retry_convenience_function():
    """Test with_retry convenience function."""
    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Fail once")
        return "success"

    retry_func = with_retry(flaky_func, max_retries=3)
    result = retry_func()

    assert result == "success"
    assert call_count == 2


def test_retry_error_initialization():
    """Test RetryError initialization."""
    original_exception = ValueError("Original error")
    retry_error = RetryError("All retries exhausted", original_exception)

    assert str(retry_error) == "All retries exhausted. Last error: Original error"
    assert retry_error.last_exception == original_exception
