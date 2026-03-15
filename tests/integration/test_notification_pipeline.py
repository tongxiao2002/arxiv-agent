"""Integration test for the Phase 4 notification pipeline."""

from datetime import date
from unittest.mock import AsyncMock, Mock, patch

from arxiv_agent.cli import run_once_command
from arxiv_agent.config import Config
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.storage.json_storage import JsonStorage


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
