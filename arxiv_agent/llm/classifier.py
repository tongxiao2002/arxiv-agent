"""LLM-based paper classification for Arxiv-Agent."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from arxiv_agent.config import LLMConfig
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.utils.retry import retry

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(
            f"LLM Error ({provider}): {message}" if provider else f"LLM Error: {message}"
        )


class LLMConfigurationError(LLMError):
    """Exception for LLM configuration errors (e.g., missing API keys)."""
    pass

class ClassificationError(LLMError):
    """Exception for classification-specific errors."""

    pass


class ProviderNotSupportedError(LLMError):
    """Exception for unsupported LLM provider."""

    pass


# Prompt templates
CLASSIFICATION_PROMPT_TEMPLATE = """You are a research assistant helping a researcher filter academic papers.
Given a paper's title and abstract, determine if it's relevant to the researcher's interests.

Researcher's topics of interest: {topics}

Paper Title: {title}
Paper Abstract: {abstract}

Please provide a JSON response with the following fields:
- "relevance_score": a float between 0 and 1 indicating how relevant the paper is (1 = highly relevant)
- "is_relevant": boolean indicating if the paper is relevant (relevance_score >= 0.5)
- "matched_topics": list of specific topics from the researcher's interests that match this paper
- "classification_reason": brief explanation of why the paper is or isn't relevant

Response must be valid JSON only, no other text."""


def _format_classification_prompt(paper: Paper, topics: List[str]) -> str:
    """Format classification prompt with paper details and topics."""
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        title=paper.title,
        abstract=paper.abstract,
        topics=", ".join(topics),
    )


def _parse_classification_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM classification response into structured data."""
    import json
    import re

    # Try to extract JSON if response contains other text
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ClassificationError(f"Failed to parse classification response: {e}")

    # Validate required fields
    required_fields = ["relevance_score", "is_relevant", "matched_topics", "classification_reason"]
    for field in required_fields:
        if field not in data:
            raise ClassificationError(f"Missing field in classification response: {field}")

    # Validate types
    if not isinstance(data["relevance_score"], (int, float)):
        raise ClassificationError("relevance_score must be a number")
    if not isinstance(data["is_relevant"], bool):
        raise ClassificationError("is_relevant must be a boolean")
    if not isinstance(data["matched_topics"], list):
        raise ClassificationError("matched_topics must be a list")
    if not isinstance(data["classification_reason"], str):
        raise ClassificationError("classification_reason must be a string")

    # Ensure relevance_score is within bounds
    data["relevance_score"] = float(data["relevance_score"])
    if data["relevance_score"] < 0 or data["relevance_score"] > 1:
        raise ClassificationError("relevance_score must be between 0 and 1")

    return data


@retry(max_retries=3, backoff_factor=2.0, jitter=True)
def _call_openai_api(prompt: str, config: LLMConfig) -> str:
    """Make API call to OpenAI with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigurationError("OpenAI library not installed. Install with: pip install openai")

    api_key = _get_openai_api_key()
    if not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.classification_temperature,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise ClassificationError(f"OpenAI API error: {e}", provider="openai")


@retry(max_retries=3, backoff_factor=2.0, jitter=True)
def _call_anthropic_api(prompt: str, config: LLMConfig) -> str:
    """Make API call to Anthropic with retry logic."""
    try:
        import anthropic
    except ImportError:
        raise LLMConfigurationError(
            "Anthropic library not installed. Install with: pip install anthropic"
        )

    api_key = _get_anthropic_api_key()
    if not api_key:
        raise LLMConfigurationError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        # Anthropic doesn't support JSON response format directly, but we can request JSON
        response = client.messages.create(
            model=config.model,
            max_tokens=500,
            temperature=config.classification_temperature,
            system="You are a helpful research assistant. Respond with valid JSON only.",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text
    except Exception as e:
        raise ClassificationError(f"Anthropic API error: {e}", provider="anthropic")


def _get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    import os
    return os.getenv("OPENAI_API_KEY")


def _get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    import os
    return os.getenv("ANTHROPIC_API_KEY")


def classify_paper(
    paper: Paper,
    config: LLMConfig,
    topics: List[str],
) -> Dict[str, Any]:
    """
    Classify a paper's relevance using LLM.

    Args:
        paper: Paper object to classify
        config: LLM configuration
        topics: List of research topics of interest

    Returns:
        Dictionary with classification results:
            - relevance_score: float between 0 and 1
            - is_relevant: boolean
            - matched_topics: list of matched topics
            - classification_reason: string explanation

    Raises:
        ClassificationError: If classification fails
        ProviderNotSupportedError: If LLM provider is not supported
    """
    logger.info(f"Classifying paper: {paper.title[:100]}...")

    if not paper.title or not paper.abstract:
        raise ClassificationError("Paper must have title and abstract")

    if not topics:
        raise ClassificationError("No topics provided for classification")

    # Format prompt
    prompt = _format_classification_prompt(paper, topics)

    # Call appropriate provider
    provider = config.provider.lower()
    try:
        if provider == "openai":
            response_text = _call_openai_api(prompt, config)
        elif provider == "anthropic":
            response_text = _call_anthropic_api(prompt, config)
        elif provider == "local":
            # Local provider not yet implemented
            raise ProviderNotSupportedError(
                "Local provider not yet implemented. Use 'openai' or 'anthropic'."
            )
        else:
            raise ProviderNotSupportedError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        logger.error(f"LLM classification failed for paper {paper.title}: {e}")
        raise

    # Parse response
    try:
        result = _parse_classification_response(response_text)
        logger.info(
            f"Classification result for '{paper.title[:50]}...': "
            f"relevance={result['relevance_score']:.2f}, "
            f"relevant={result['is_relevant']}"
        )
        return result
    except Exception as e:
        raise ClassificationError(f"Failed to process classification response: {e}")


def batch_classify_papers(
    papers: List[Paper],
    config: LLMConfig,
    topics: List[str],
    max_concurrent: int = 5,
) -> List[Tuple[Paper, Dict[str, Any]]]:
    """
    Classify multiple papers with optional concurrency control.

    Args:
        papers: List of Paper objects to classify
        config: LLM configuration
        topics: List of research topics of interest
        max_concurrent: Maximum concurrent classification requests

    Returns:
        List of (paper, classification_result) tuples

    Note:
        Currently implements sequential processing. Concurrent version
        can be added later when needed.
    """
    results = []
    for i, paper in enumerate(papers):
        logger.info(f"Classifying paper {i+1}/{len(papers)}: {paper.title[:100]}...")
        try:
            result = classify_paper(paper, config, topics)
            results.append((paper, result))
        except Exception as e:
            logger.error(f"Failed to classify paper '{paper.title}': {e}")
            # Continue with other papers
            continue

    return results