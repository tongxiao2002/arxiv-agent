"""LLM module for Arxiv-Agent."""

from .classifier import (
    LLMError,
    ClassificationError,
    ProviderNotSupportedError,
    classify_paper,
    batch_classify_papers,
)
from .summarizer import (
    SummarizationError,
    summarize_abstract,
    batch_summarize_papers,
)

__all__ = [
    # Classifier
    "LLMError",
    "ClassificationError",
    "ProviderNotSupportedError",
    "classify_paper",
    "batch_classify_papers",
    # Summarizer
    "SummarizationError",
    "summarize_abstract",
    "batch_summarize_papers",
]