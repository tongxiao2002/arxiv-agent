"""LLM-based abstract summarization for Arxiv-Agent."""

import asyncio
import logging
import threading
from typing import Coroutine, List, Optional, Tuple, TypeVar

from arxiv_agent.config import LLMConfig
from arxiv_agent.llm.provider_utils import get_provider_api_key, get_provider_env_var
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.utils.runtime import (
    RuntimeOptions,
    call_with_retry,
    call_with_retry_async,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class SummarizationError(Exception):
    """Exception for summarization-specific errors."""

    pass


# Prompt templates
SUMMARIZATION_PROMPT_TEMPLATE = """You are a research assistant helping a researcher understand academic papers.
Given a paper's title and abstract, provide a concise 2-3 sentence summary that captures:
1. The main research question or problem addressed
2. The key approach or methodology used
3. The primary findings or contributions

Paper Title: {title}
Paper Abstract: {abstract}

Provide only the summary text, no additional commentary or formatting."""


def _format_summarization_prompt(paper: Paper) -> str:
    """Format summarization prompt with paper details."""
    return SUMMARIZATION_PROMPT_TEMPLATE.format(
        title=paper.title,
        abstract=paper.abstract,
    )


def _run_coroutine(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine from sync code, even if a loop is already active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - passthrough
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]

    return result[0]


def _call_openai_api_summarization(
    prompt: str,
    config: LLMConfig,
    runtime_options: Optional[RuntimeOptions] = None,
) -> str:
    """Make API call to OpenAI for summarization with retry logic."""
    return _run_coroutine(
        _call_openai_api_summarization_async(
            prompt,
            config,
            runtime_options=runtime_options,
        )
    )


async def _call_openai_api_summarization_async(
    prompt: str,
    config: LLMConfig,
    runtime_options: Optional[RuntimeOptions] = None,
    client: Optional[object] = None,
) -> str:
    """Make async API call to OpenAI with optional shared client reuse."""
    runtime = runtime_options or RuntimeOptions()
    managed_client = client
    should_close = False

    if managed_client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise SummarizationError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        api_key = get_provider_api_key("openai")
        if not api_key:
            env_var = get_provider_env_var("openai")
            raise SummarizationError(f"{env_var} environment variable not set")

        managed_client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
        should_close = True

    async def operation() -> str:
        try:
            response = await managed_client.chat.completions.create(
                model=config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=config.summarization_temperature,
                max_tokens=1024,
                timeout=runtime.request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise SummarizationError(f"OpenAI API error: {exc}") from exc

    try:
        return await call_with_retry_async(
            operation,
            operation_name="_call_openai_api_summarization_async",
            max_retries=runtime.max_retries,
            backoff_factor=runtime.retry_backoff_factor,
        )
    finally:
        if should_close:
            await managed_client.close()


def _call_anthropic_api_summarization(
    prompt: str,
    config: LLMConfig,
    runtime_options: Optional[RuntimeOptions] = None,
) -> str:
    """Make API call to Anthropic for summarization with retry logic."""
    try:
        import anthropic
    except ImportError:
        raise SummarizationError(
            "Anthropic library not installed. Install with: pip install anthropic"
        )

    runtime = runtime_options or RuntimeOptions()
    api_key = get_provider_api_key("anthropic")
    if not api_key:
        env_var = get_provider_env_var("anthropic")
        raise SummarizationError(f"{env_var} environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    def operation() -> str:
        try:
            response = client.messages.create(
                model=config.model,
                max_tokens=1024,
                temperature=config.summarization_temperature,
                system="You are a helpful research assistant.",
                messages=[{"role": "user", "content": prompt}],
                timeout=runtime.request_timeout,
            )
            return response.content[0].text
        except Exception as exc:
            raise SummarizationError(f"Anthropic API error: {exc}") from exc

    return call_with_retry(
        operation,
        operation_name="_call_anthropic_api_summarization",
        max_retries=runtime.max_retries,
        backoff_factor=runtime.retry_backoff_factor,
    )


def summarize_abstract(
    paper: Paper,
    config: LLMConfig,
    runtime_options: Optional[RuntimeOptions] = None,
) -> str:
    """
    Generate a concise 2-3 sentence summary of a paper's abstract using LLM.

    Args:
        paper: Paper object to summarize
        config: LLM configuration

    Returns:
        2-3 sentence summary string

    Raises:
        SummarizationError: If summarization fails
    """
    logger.info(f"Summarizing paper: {paper.title[:100]}...")

    if not paper.title or not paper.abstract:
        raise SummarizationError("Paper must have title and abstract")

    # Format prompt
    prompt = _format_summarization_prompt(paper)

    # Call appropriate provider
    provider = config.provider.lower()
    try:
        if provider == "openai":
            summary = _call_openai_api_summarization(prompt, config, runtime_options)
        elif provider == "anthropic":
            summary = _call_anthropic_api_summarization(prompt, config, runtime_options)
        elif provider == "local":
            # Local provider not yet implemented
            raise SummarizationError(
                "Local provider not yet implemented. Use 'openai' or 'anthropic'."
            )
        else:
            raise SummarizationError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        logger.error(f"LLM summarization failed for paper {paper.title}: {e}")
        raise

    # Clean up summary (remove extra whitespace, ensure it's not empty)
    summary = summary.strip()
    if not summary:
        raise SummarizationError("LLM returned empty summary")

    logger.info(f"Generated summary for '{paper.title[:50]}...' ({len(summary)} chars)")
    return summary


async def asummarize_abstract(
    paper: Paper,
    config: LLMConfig,
    runtime_options: Optional[RuntimeOptions] = None,
    client: Optional[object] = None,
) -> str:
    """Asynchronously summarize a paper abstract using LLM."""
    logger.info(f"Summarizing paper: {paper.title[:100]}...")

    if not paper.title or not paper.abstract:
        raise SummarizationError("Paper must have title and abstract")

    prompt = _format_summarization_prompt(paper)
    provider = config.provider.lower()

    try:
        if provider == "openai":
            summary = await _call_openai_api_summarization_async(
                prompt,
                config,
                runtime_options=runtime_options,
                client=client,
            )
        elif provider == "anthropic":
            summary = _call_anthropic_api_summarization(prompt, config, runtime_options)
        elif provider == "local":
            raise SummarizationError(
                "Local provider not yet implemented. Use 'openai' or 'anthropic'."
            )
        else:
            raise SummarizationError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        logger.error(f"LLM summarization failed for paper {paper.title}: {e}")
        raise

    summary = summary.strip()
    if not summary:
        raise SummarizationError("LLM returned empty summary")

    logger.info(f"Generated summary for '{paper.title[:50]}...' ({len(summary)} chars)")
    return summary


def batch_summarize_papers(
    papers: List[Paper],
    config: LLMConfig,
    max_concurrent: int = 5,
) -> List[Tuple[Paper, str]]:
    """
    Summarize multiple papers with optional concurrency control.

    Args:
        papers: List of Paper objects to summarize
        config: LLM configuration
        max_concurrent: Maximum concurrent summarization requests

    Returns:
        List of (paper, summary) tuples
    """
    return _run_coroutine(
        abatch_summarize_papers(
            papers,
            config,
            max_concurrent=max_concurrent,
        )
    )


async def abatch_summarize_papers(
    papers: List[Paper],
    config: LLMConfig,
    max_concurrent: int = 5,
    client: Optional[object] = None,
) -> List[Tuple[Paper, str]]:
    """Summarize multiple papers concurrently."""
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be a positive integer")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _summarize_one(index: int, paper: Paper) -> Optional[Tuple[Paper, str]]:
        async with semaphore:
            logger.info(
                "Summarizing paper %s/%s: %s...",
                index + 1,
                len(papers),
                paper.title[:100],
            )
            try:
                if client is not None and config.provider.lower() == "openai":
                    summary = await asummarize_abstract(
                        paper,
                        config,
                        client=client,
                    )
                else:
                    summary = await asyncio.to_thread(
                        summarize_abstract,
                        paper,
                        config,
                    )
                return (paper, summary)
            except Exception as exc:
                logger.error(
                    "Failed to summarize paper '%s': %s",
                    paper.title,
                    exc,
                )
                return None

    tasks = [
        asyncio.create_task(_summarize_one(index, paper))
        for index, paper in enumerate(papers)
    ]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]
