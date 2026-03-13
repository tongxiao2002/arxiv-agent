"""Tests for the ClassifierAgent."""

from datetime import date
from unittest.mock import Mock, patch

from arxiv_agent.agents.classifier_agent import ClassifierAgent
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper


def make_config():
    """Return a minimal classifier config."""
    return {
        "agent": {"timezone": "Asia/Shanghai"},
        "topics": ["agents"],
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "storage": {"data_dir": "./papers"},
        "advanced": {
            "max_retries": 2,
            "retry_backoff_factor": 1.0,
            "request_timeout": 15,
        },
    }


def make_plain_paper() -> Paper:
    """Create an unenhanced paper."""
    return Paper(
        title="Plain Paper",
        abstract="Abstract about agents.",
        authors=["Author"],
        arxiv_id="2501.12345",
        source="arxiv",
    )


def make_enhanced_paper() -> EnhancedPaper:
    """Create an already enhanced paper."""
    return EnhancedPaper(
        title="Enhanced Paper",
        abstract="Abstract about agents.",
        authors=["Author"],
        arxiv_id="2501.99999",
        source="arxiv",
        is_relevant=True,
        summary="Existing summary",
        matched_topics=["agents"],
        relevance_score=0.9,
        classification_reason="Already processed.",
    )


def test_classifier_agent_skips_already_enhanced_papers():
    """Test reruns do not reprocess papers already stored as enhanced."""
    storage = Mock()
    storage.load_papers.return_value = [make_enhanced_paper(), make_plain_paper()]
    storage.save_papers.return_value = True

    agent = ClassifierAgent(make_config())
    agent.storage = storage

    with patch(
        "arxiv_agent.agents.classifier_agent.classify_paper",
        return_value={
            "relevance_score": 0.7,
            "is_relevant": True,
            "matched_topics": ["agents"],
            "classification_reason": "Good fit.",
        },
    ) as mock_classify:
        with patch(
            "arxiv_agent.agents.classifier_agent.summarize_abstract",
            return_value="New summary.",
        ) as mock_summarize:
            result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is True
    assert result["papers_processed"] == 1
    assert result["papers_skipped"] == 1
    assert result["enhanced_papers"] == 1
    mock_classify.assert_called_once()
    mock_summarize.assert_called_once()

    saved_date, saved_papers = storage.save_papers.call_args.args
    assert saved_date == date(2026, 3, 12)
    assert len(saved_papers) == 2
    assert all(isinstance(paper, EnhancedPaper) for paper in saved_papers)


def test_classifier_agent_preserves_retry_root_cause_in_errors():
    """Test operator-facing errors keep the underlying retry cause visible."""
    storage = Mock()
    storage.load_papers.return_value = [make_plain_paper()]

    agent = ClassifierAgent(make_config())
    agent.storage = storage

    with patch(
        "arxiv_agent.agents.classifier_agent.classify_paper",
        side_effect=Exception("LLM timeout"),
    ):
        result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is False
    assert "LLM timeout" in result["errors"][0]
