"""Tests for the ClassifierAgent."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from arxiv_agent.agents.classifier_agent import ClassifierAgent
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper
from arxiv_agent.storage.json_storage import JsonStorage


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

    shared_client = Mock()
    shared_client.close = AsyncMock()

    with patch(
        "arxiv_agent.agents.classifier_agent.ClassifierAgent._create_openai_client",
        new=AsyncMock(return_value=shared_client),
    ):
        with patch(
            "arxiv_agent.agents.classifier_agent.aclassify_paper",
            new=AsyncMock(
                return_value={
                    "relevance_score": 0.7,
                    "is_relevant": True,
                    "matched_topics": ["agents"],
                    "classification_reason": "Good fit.",
                }
            ),
        ) as mock_classify:
            with patch(
                "arxiv_agent.agents.classifier_agent.asummarize_abstract",
                new=AsyncMock(return_value="New summary."),
            ) as mock_summarize:
                result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is True
    assert result["papers_processed"] == 1
    assert result["papers_skipped"] == 1
    assert result["enhanced_papers"] == 1
    mock_classify.assert_awaited_once()
    mock_summarize.assert_awaited_once()
    shared_client.close.assert_awaited_once()

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

    shared_client = Mock()
    shared_client.close = AsyncMock()

    with patch(
        "arxiv_agent.agents.classifier_agent.ClassifierAgent._create_openai_client",
        new=AsyncMock(return_value=shared_client),
    ):
        with patch(
            "arxiv_agent.agents.classifier_agent.aclassify_paper",
            new=AsyncMock(side_effect=Exception("LLM timeout")),
        ):
            with patch(
                "arxiv_agent.agents.classifier_agent.asummarize_abstract",
                new=AsyncMock(return_value="New summary."),
            ):
                result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is False
    assert "LLM timeout" in result["errors"][0]
    shared_client.close.assert_awaited_once()


def test_classifier_agent_only_processes_new_raw_papers_after_duplicate_safe_merge(
    temp_dir,
):
    """Test reruns only classify newly added raw papers after storage merges."""
    target_date = date(2026, 3, 12)
    storage = JsonStorage(str(temp_dir))
    storage.save_papers(
        target_date,
        [
            EnhancedPaper(
                title="Already Enhanced",
                abstract="Processed already.",
                authors=["Author"],
                arxiv_id="2501.99999",
                paper_id="2501.99999",
                publication_date=datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
                source="arxiv",
                is_relevant=True,
                summary="Existing summary",
                matched_topics=["agents"],
                relevance_score=0.95,
                classification_reason="Already processed.",
            ),
            Paper(
                title="New Raw",
                abstract="Fresh abstract about agents.",
                authors=["Author"],
                arxiv_id="2501.12345",
                paper_id="2501.12345",
                publication_date=datetime(2026, 3, 12, 2, 0, tzinfo=timezone.utc),
                source="arxiv",
            ),
        ],
    )

    config = make_config()
    config["storage"]["data_dir"] = str(temp_dir)
    agent = ClassifierAgent(config)

    shared_client = Mock()
    shared_client.close = AsyncMock()

    with patch(
        "arxiv_agent.agents.classifier_agent.ClassifierAgent._create_openai_client",
        new=AsyncMock(return_value=shared_client),
    ):
        with patch(
            "arxiv_agent.agents.classifier_agent.aclassify_paper",
            new=AsyncMock(
                return_value={
                    "relevance_score": 0.8,
                    "is_relevant": True,
                    "matched_topics": ["agents"],
                    "classification_reason": "New match.",
                }
            ),
        ) as mock_classify:
            with patch(
                "arxiv_agent.agents.classifier_agent.asummarize_abstract",
                new=AsyncMock(return_value="Fresh summary."),
            ) as mock_summarize:
                result = agent.run(target_date=target_date)

    assert result["success"] is True
    assert result["papers_processed"] == 1
    assert result["papers_skipped"] == 1
    assert result["enhanced_papers"] == 1
    mock_classify.assert_awaited_once()
    mock_summarize.assert_awaited_once()
    shared_client.close.assert_awaited_once()

    saved_papers = storage.load_papers(target_date)
    assert len(saved_papers) == 2
    assert all(isinstance(paper, EnhancedPaper) for paper in saved_papers)
