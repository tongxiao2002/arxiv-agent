"""Tests for LLM classifier."""

import json
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.config import LLMConfig
from arxiv_agent.llm.classifier import (
    ClassificationError,
    LLMConfigurationError,
    LLMError,
    ProviderNotSupportedError,
    batch_classify_papers,
    classify_paper,
)
from arxiv_agent.sources.base_source import Paper

# Sample test data
SAMPLE_PAPER = Paper(
    title="Test Paper Title",
    abstract="This is a test abstract about machine learning and natural language processing.",
    authors=["Author One", "Author Two"],
    arxiv_id="2101.12345",
    categories=["cs.CL", "cs.LG"],
    source="arxiv",
)

SAMPLE_TOPICS = ["machine learning", "natural language processing", "computer vision"]


def test_classify_paper_openai_success():
    """Test successful classification with OpenAI provider."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        classification_temperature=0.1,
    )

    mock_response = {
        "relevance_score": 0.85,
        "is_relevant": True,
        "matched_topics": ["machine learning", "natural language processing"],
        "classification_reason": "The paper discusses ML and NLP techniques.",
    }

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.return_value = json.dumps(mock_response)

        result = classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)

        assert result["relevance_score"] == 0.85
        assert result["is_relevant"] is True
        assert set(result["matched_topics"]) == {
            "machine learning",
            "natural language processing",
        }
        assert "classification_reason" in result
        mock_api.assert_called_once()


def test_classify_paper_anthropic_success():
    """Test successful classification with Anthropic provider."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        classification_temperature=0.1,
    )

    mock_response = {
        "relevance_score": 0.6,
        "is_relevant": True,
        "matched_topics": ["machine learning"],
        "classification_reason": "The paper focuses on ML algorithms.",
    }

    with patch("arxiv_agent.llm.classifier._call_anthropic_api") as mock_api:
        mock_api.return_value = json.dumps(mock_response)

        result = classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)

        assert result["relevance_score"] == 0.6
        assert result["is_relevant"] is True
        assert result["matched_topics"] == ["machine learning"]
        mock_api.assert_called_once()


def test_classify_paper_provider_not_supported():
    """Test classification with unsupported provider."""
    config = LLMConfig(provider="local", model="local-model")

    with pytest.raises(
        ProviderNotSupportedError, match="Local provider not yet implemented"
    ):
        classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_invalid_provider():
    """Test classification with invalid provider string."""
    config = LLMConfig(provider="invalid", model="gpt-4")

    with pytest.raises(ProviderNotSupportedError, match="Unsupported LLM provider"):
        classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_missing_title_abstract():
    """Test classification with missing title or abstract."""
    config = LLMConfig(provider="openai")

    paper_no_title = Paper(title="", abstract="Abstract", authors=[])
    paper_no_abstract = Paper(title="Title", abstract="", authors=[])

    with pytest.raises(ClassificationError, match="Paper must have title and abstract"):
        classify_paper(paper_no_title, config, SAMPLE_TOPICS)

    with pytest.raises(ClassificationError, match="Paper must have title and abstract"):
        classify_paper(paper_no_abstract, config, SAMPLE_TOPICS)


def test_classify_paper_no_topics():
    """Test classification with empty topics list."""
    config = LLMConfig(provider="openai")

    with pytest.raises(
        ClassificationError, match="No topics provided for classification"
    ):
        classify_paper(SAMPLE_PAPER, config, [])


def test_classify_paper_api_error():
    """Test classification when API call fails."""
    config = LLMConfig(provider="openai")

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.side_effect = Exception("API timeout")

        with pytest.raises(Exception, match="API timeout"):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_invalid_json_response():
    """Test classification with invalid JSON response."""
    config = LLMConfig(provider="openai")

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.return_value = "Not a JSON response"

        with pytest.raises(
            ClassificationError, match="Failed to parse classification response"
        ):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_missing_fields_in_response():
    """Test classification with missing required fields in response."""
    config = LLMConfig(provider="openai")
    incomplete_response = {
        "relevance_score": 0.5,
        # Missing is_relevant, matched_topics, classification_reason
    }

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.return_value = json.dumps(incomplete_response)

        with pytest.raises(
            ClassificationError, match="Missing field in classification response"
        ):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_invalid_relevance_score():
    """Test classification with invalid relevance score."""
    config = LLMConfig(provider="openai")
    invalid_response = {
        "relevance_score": 1.5,  # Out of range
        "is_relevant": True,
        "matched_topics": ["ml"],
        "classification_reason": "test",
    }

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.return_value = json.dumps(invalid_response)

        with pytest.raises(
            ClassificationError, match="relevance_score must be between 0 and 1"
        ):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_classify_paper_json_extraction():
    """Test classification when response contains extra text around JSON."""
    config = LLMConfig(provider="openai")
    response_text = """Here's the classification result:
    {
        "relevance_score": 0.7,
        "is_relevant": true,
        "matched_topics": ["machine learning"],
        "classification_reason": "Relevant to ML."
    }
    Thanks!"""

    with patch("arxiv_agent.llm.classifier._call_openai_api") as mock_api:
        mock_api.return_value = response_text

        result = classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)
        assert result["relevance_score"] == 0.7
        assert result["is_relevant"] is True


def test_batch_classify_papers():
    """Test batch classification of multiple papers."""
    config = LLMConfig(provider="openai")
    papers = [
        SAMPLE_PAPER,
        Paper(
            title="Another Paper",
            abstract="Abstract about computer vision.",
            authors=["Author"],
        ),
    ]

    mock_responses = [
        {
            "relevance_score": 0.8,
            "is_relevant": True,
            "matched_topics": ["machine learning"],
            "classification_reason": "ML paper",
        },
        {
            "relevance_score": 0.3,
            "is_relevant": False,
            "matched_topics": [],
            "classification_reason": "Not relevant",
        },
    ]

    with patch("arxiv_agent.llm.classifier.classify_paper") as mock_classify:
        mock_classify.side_effect = lambda paper, cfg, topics: mock_responses.pop(0)

        results = batch_classify_papers(papers, config, SAMPLE_TOPICS)

        assert len(results) == 2
        assert results[0][0] == papers[0]
        assert results[0][1]["relevance_score"] == 0.8
        assert results[1][0] == papers[1]
        assert results[1][1]["relevance_score"] == 0.3


def test_batch_classify_papers_with_failures():
    """Test batch classification with some papers failing."""
    config = LLMConfig(provider="openai")
    papers = [SAMPLE_PAPER, Paper(title="Bad Paper", abstract="", authors=[])]

    with patch("arxiv_agent.llm.classifier.classify_paper") as mock_classify:
        mock_classify.side_effect = [
            {
                "relevance_score": 0.8,
                "is_relevant": True,
                "matched_topics": [],
                "classification_reason": "",
            },
            ClassificationError("Missing abstract"),
        ]

        results = batch_classify_papers(papers, config, SAMPLE_TOPICS)

        # Only first paper should be in results
        assert len(results) == 1
        assert results[0][0] == papers[0]


def test_openai_api_key_missing():
    """Test classification when OpenAI API key is missing."""
    config = LLMConfig(provider="openai")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            LLMConfigurationError,
            match="OPENAI_API_KEY environment variable not set",
        ):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_anthropic_api_key_missing():
    """Test classification when Anthropic API key is missing."""
    config = LLMConfig(provider="anthropic")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            LLMConfigurationError,
            match="ANTHROPIC_API_KEY environment variable not set",
        ):
            classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_openai_library_not_installed():
    """Test classification when OpenAI library is not installed."""
    config = LLMConfig(provider="openai")

    with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-key"}):
        with patch(
            "arxiv_agent.llm.classifier._call_openai_api", side_effect=ImportError
        ):
            with pytest.raises(ImportError):
                classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)


def test_anthropic_library_not_installed():
    """Test classification when Anthropic library is not installed."""
    config = LLMConfig(provider="anthropic")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy-key"}):
        with patch(
            "arxiv_agent.llm.classifier._call_anthropic_api",
            side_effect=ImportError,
        ):
            with pytest.raises(ImportError):
                classify_paper(SAMPLE_PAPER, config, SAMPLE_TOPICS)
