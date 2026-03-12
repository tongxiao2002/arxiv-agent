"""LLM-based abstract summarization for Arxiv-Agent."""

import logging
from typing import List, Optional, Tuple

from arxiv_agent.config import LLMConfig
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.utils.retry import retry

logger = logging.getLogger(__name__)


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


@retry(max_retries=3, backoff_factor=2.0, jitter=True)
def _call_openai_api_summarization(prompt: str, config: LLMConfig) -> str:
    """Make API call to OpenAI for summarization with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SummarizationError(
            "OpenAI library not installed. Install with: pip install openai"
        )

    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SummarizationError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.summarization_temperature,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise SummarizationError(f"OpenAI API error: {e}", provider="openai")


@retry(max_retries=3, backoff_factor=2.0, jitter=True)
def _call_anthropic_api_summarization(prompt: str, config: LLMConfig) -> str:
    """Make API call to Anthropic for summarization with retry logic."""
    try:
        import anthropic
    except ImportError:
        raise SummarizationError(
            "Anthropic library not installed. Install with: pip install anthropic"
        )

    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SummarizationError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=300,
            temperature=config.summarization_temperature,
            system="You are a helpful research assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text
    except Exception as e:
        raise SummarizationError(f"Anthropic API error: {e}", provider="anthropic")


def summarize_abstract(
    paper: Paper,
    config: LLMConfig,
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
            summary = _call_openai_api_summarization(prompt, config)
        elif provider == "anthropic":
            summary = _call_anthropic_api_summarization(prompt, config)
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

    Note:
        Currently implements sequential processing. Concurrent version
        can be added later when needed.
    """
    # Type imports already at top

    results = []
    for i, paper in enumerate(papers):
        logger.info(f"Summarizing paper {i+1}/{len(papers)}: {paper.title[:100]}...")
        try:
            summary = summarize_abstract(paper, config)
            results.append((paper, summary))
        except Exception as e:
            logger.error(f"Failed to summarize paper '{paper.title}': {e}")
            # Continue with other papers
            continue

    return results