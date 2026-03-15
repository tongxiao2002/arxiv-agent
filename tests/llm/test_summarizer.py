"""Tests for LLM summarizer."""

from unittest.mock import patch

import pytest

from arxiv_agent.config import LLMConfig
from arxiv_agent.llm.summarizer import (
    SummarizationError,
    batch_summarize_papers,
    summarize_abstract,
)
from arxiv_agent.sources.base_source import Paper

SAMPLE_PAPER = Paper(
    title="Test Paper Title",
    abstract="This is a test abstract about machine learning and natural language processing.",
    authors=["Author One", "Author Two"],
    arxiv_id="2101.12345",
    categories=["cs.CL", "cs.LG"],
    source="arxiv",
)


def test_summarize_abstract_openai_success():
    """Test successful summarization with OpenAI provider."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        summarization_temperature=0.3,
    )

    with patch("arxiv_agent.llm.summarizer._call_openai_api_summarization") as mock_api:
        mock_api.return_value = "A concise test summary."

        result = summarize_abstract(SAMPLE_PAPER, config)

    assert result == "A concise test summary."
    mock_api.assert_called_once()


def test_summarize_abstract_missing_title_abstract():
    """Test summarization with missing title or abstract."""
    config = LLMConfig(provider="openai")

    with pytest.raises(SummarizationError, match="Paper must have title and abstract"):
        summarize_abstract(Paper(title="", abstract="Abstract", authors=[]), config)

    with pytest.raises(SummarizationError, match="Paper must have title and abstract"):
        summarize_abstract(Paper(title="Title", abstract="", authors=[]), config)


def test_batch_summarize_papers():
    """Test concurrent batch summarization preserves results."""
    config = LLMConfig(provider="openai")
    papers = [
        SAMPLE_PAPER,
        Paper(
            title="Another Paper",
            abstract="Abstract about computer vision.",
            authors=["Author"],
        ),
    ]

    with patch("arxiv_agent.llm.summarizer.summarize_abstract") as mock_summarize:
        mock_summarize.side_effect = ["Summary one.", "Summary two."]

        results = batch_summarize_papers(papers, config)

    assert len(results) == 2
    assert results[0] == (papers[0], "Summary one.")
    assert results[1] == (papers[1], "Summary two.")


def test_batch_summarize_papers_with_failures():
    """Test batch summarization skips failed papers."""
    config = LLMConfig(provider="openai")
    papers = [SAMPLE_PAPER, Paper(title="Bad Paper", abstract="", authors=[])]

    with patch("arxiv_agent.llm.summarizer.summarize_abstract") as mock_summarize:
        mock_summarize.side_effect = [
            "Summary one.",
            SummarizationError("Missing abstract"),
        ]

        results = batch_summarize_papers(papers, config)

    assert len(results) == 1
    assert results[0] == (papers[0], "Summary one.")
