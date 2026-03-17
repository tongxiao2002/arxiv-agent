"""Integration test for the Phase 4 notification pipeline."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from arxiv_agent.cli import run_once_command
from arxiv_agent.config import Config
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.intervals import RunOnceInterval


def test_notification_pipeline_end_to_end(temp_dir, monkeypatch):
    """Test scrape -> classify -> email -> archive flow with mocked externals."""
    papers_dir = temp_dir / "papers"
    archive_dir = temp_dir / "archive"
    config = Config.from_dict(
        {
            "agent": {"timezone": "Asia/Shanghai"},
            "sources": {
                "primary": "arxiv",
                "arxiv": {"categories": ["cs"], "max_papers": 10},
            },
            "topics": ["agents"],
            "storage": {
                "data_dir": str(papers_dir),
                "archive_dir": str(archive_dir),
                "retention_days": 30,
            },
            "email": {
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
                "smtp_security": "starttls",
                "smtp_username": "mailer",
                "from_email": "agent@example.com",
                "to_emails": ["user@example.com"],
                "subject_template": "Digest - {date}",
            },
        }
    )

    target_date = date(2026, 3, 12)
    old_date = date(2026, 1, 15)
    JsonStorage(str(papers_dir)).save_papers(
        old_date,
        [Paper(title="Old Paper", abstract="Old abstract", authors=["Author"])],
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-secret")

    sample_paper = Paper(
        title="Fresh Relevant Paper",
        abstract="Abstract about agents.",
        authors=["Researcher"],
        arxiv_id="2603.12345",
        categories=["cs.AI"],
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2603.12345.pdf",
        webpage_url="https://arxiv.org/abs/2603.12345",
    )

    class MockDate:
        @classmethod
        def today(cls):
            return target_date

        @classmethod
        def fromisoformat(cls, value):
            return date.fromisoformat(value)

    smtp_client = Mock()
    shared_client = Mock()
    shared_client.close = AsyncMock()

    with patch(
        "arxiv_agent.cli.get_current_date_in_timezone", return_value=target_date
    ):
        with patch(
            "arxiv_agent.sources.arxiv_source.ArxivSource.fetch_papers",
            return_value=[sample_paper],
        ):
            with patch(
                "arxiv_agent.agents.classifier_agent.ClassifierAgent._create_openai_client",
                new=AsyncMock(return_value=shared_client),
            ):
                with patch(
                    "arxiv_agent.agents.classifier_agent.aclassify_paper",
                    new=AsyncMock(
                        return_value={
                            "relevance_score": 0.95,
                            "is_relevant": True,
                            "matched_topics": ["agents"],
                            "classification_reason": "Strong fit.",
                        }
                    ),
                ):
                    with patch(
                        "arxiv_agent.agents.classifier_agent.asummarize_abstract",
                        new=AsyncMock(return_value="Short summary."),
                    ):
                        with patch(
                            "arxiv_agent.email.sender.smtplib.SMTP",
                            return_value=smtp_client,
                        ):
                            with patch("arxiv_agent.storage.archiver.date", MockDate):
                                result = run_once_command(config, dry_run=False)

    assert result["success"] is True
    assert smtp_client.send_message.call_count == 1
    assert (papers_dir / "papers_2026-03-12.json").exists()
    assert (archive_dir / "papers_2026-01.tar.gz").exists()
    shared_client.close.assert_awaited_once()


def test_interval_run_once_spans_multiple_days_and_skips_email(temp_dir, monkeypatch):
    """Test interval run-once writes per-day files and skips email when requested."""
    papers_dir = temp_dir / "papers"
    archive_dir = temp_dir / "archive"
    config = Config.from_dict(
        {
            "agent": {"timezone": "Asia/Shanghai"},
            "sources": {
                "primary": "arxiv",
                "arxiv": {"categories": ["cs"], "max_papers": 10},
            },
            "topics": ["agents"],
            "storage": {
                "data_dir": str(papers_dir),
                "archive_dir": str(archive_dir),
                "retention_days": 30,
            },
            "email": {
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
                "smtp_security": "starttls",
                "smtp_username": "mailer",
                "from_email": "agent@example.com",
                "to_emails": ["user@example.com"],
                "subject_template": "Digest - {date}",
            },
        }
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    interval = RunOnceInterval.from_local_naive(
        datetime(2026, 3, 10, 8, 30),
        datetime(2026, 3, 11, 9, 0),
        "Asia/Shanghai",
    )
    sample_papers = [
        Paper(
            title="Day One Relevant Paper",
            abstract="Abstract about agents.",
            authors=["Researcher"],
            arxiv_id="2603.20001",
            paper_id="2603.20001",
            publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
            categories=["cs.AI"],
            source="arxiv",
            pdf_url="https://arxiv.org/pdf/2603.20001.pdf",
            webpage_url="https://arxiv.org/abs/2603.20001",
        ),
        Paper(
            title="Day Two Relevant Paper",
            abstract="Abstract about agents.",
            authors=["Researcher"],
            arxiv_id="2603.20002",
            paper_id="2603.20002",
            publication_date=datetime(2026, 3, 10, 17, 30, tzinfo=timezone.utc),
            categories=["cs.AI"],
            source="arxiv",
            pdf_url="https://arxiv.org/pdf/2603.20002.pdf",
            webpage_url="https://arxiv.org/abs/2603.20002",
        ),
    ]

    shared_client = Mock()
    shared_client.close = AsyncMock()
    smtp_client = Mock()

    with patch(
        "arxiv_agent.sources.arxiv_source.ArxivSource.fetch_papers_for_interval",
        return_value=sample_papers,
    ):
        with patch(
            "arxiv_agent.agents.classifier_agent.ClassifierAgent._create_openai_client",
            new=AsyncMock(return_value=shared_client),
        ):
            with patch(
                "arxiv_agent.agents.classifier_agent.aclassify_paper",
                new=AsyncMock(
                    return_value={
                        "relevance_score": 0.95,
                        "is_relevant": True,
                        "matched_topics": ["agents"],
                        "classification_reason": "Strong fit.",
                    }
                ),
            ):
                with patch(
                    "arxiv_agent.agents.classifier_agent.asummarize_abstract",
                    new=AsyncMock(return_value="Short summary."),
                ):
                    with patch(
                        "arxiv_agent.email.sender.smtplib.SMTP",
                        return_value=smtp_client,
                    ):
                        result = run_once_command(
                            config,
                            dry_run=False,
                            run_interval=interval,
                            no_email=True,
                        )

    assert result["success"] is True
    assert result["mode"] == "interval"
    assert result["email_skipped"] is True
    assert result["affected_days"] == ["2026-03-10", "2026-03-11"]
    assert (papers_dir / "papers_2026-03-10.json").exists()
    assert (papers_dir / "papers_2026-03-11.json").exists()
    smtp_client.send_message.assert_not_called()
    assert shared_client.close.await_count == 2
