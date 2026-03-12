"""Tests for the EmailerAgent."""

from datetime import date
from unittest.mock import Mock

from arxiv_agent.agents.emailer_agent import EmailerAgent
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper


def make_config():
    """Return a minimal emailer agent config."""
    return {
        "agent": {"timezone": "Asia/Shanghai"},
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_security": "starttls",
            "smtp_username": "mailer",
            "from_email": "agent@example.com",
            "to_emails": ["user@example.com"],
            "subject_template": "Digest - {date}",
        },
        "storage": {"data_dir": "./papers"},
    }


def make_relevant_paper() -> EnhancedPaper:
    """Create a relevant enhanced paper."""
    return EnhancedPaper(
        title="Relevant Paper",
        abstract="Abstract",
        authors=["Author"],
        source="arxiv",
        webpage_url="https://example.com/paper",
        pdf_url="https://example.com/paper.pdf",
        is_relevant=True,
        summary="Summary",
        matched_topics=["agents"],
        relevance_score=0.9,
    )


def make_irrelevant_paper() -> EnhancedPaper:
    """Create an irrelevant enhanced paper."""
    paper = make_relevant_paper()
    paper.is_relevant = False
    paper.matched_topics = []
    return paper


def test_emailer_agent_sends_digest():
    """Test digest sending when relevant papers exist."""
    storage = Mock()
    sender = Mock()
    storage.papers_exist_for_date.return_value = True
    storage.load_papers.return_value = [make_relevant_paper()]
    sender.send_email.return_value = {"success": True}

    agent = EmailerAgent(make_config(), storage=storage, sender=sender)
    result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is True
    assert result["email_type"] == "digest"
    assert result["relevant_papers"] == 1
    sender.send_email.assert_called_once()


def test_emailer_agent_sends_no_papers_notification():
    """Test no-papers email when the daily file exists but nothing is relevant."""
    storage = Mock()
    sender = Mock()
    storage.papers_exist_for_date.return_value = True
    storage.load_papers.return_value = [make_irrelevant_paper()]
    sender.send_email.return_value = {"success": True}

    agent = EmailerAgent(make_config(), storage=storage, sender=sender)
    result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is True
    assert result["email_type"] == "no_papers"
    sender.send_email.assert_called_once()


def test_emailer_agent_fails_for_missing_daily_file():
    """Test missing storage data does not send a no-papers email."""
    storage = Mock()
    sender = Mock()
    storage.papers_exist_for_date.return_value = False

    agent = EmailerAgent(make_config(), storage=storage, sender=sender)
    result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is False
    assert result["sent"] is False
    sender.send_email.assert_not_called()


def test_emailer_agent_rejects_unenhanced_papers():
    """Test plain papers are treated as not ready for emailing."""
    storage = Mock()
    sender = Mock()
    storage.papers_exist_for_date.return_value = True
    storage.load_papers.return_value = [
        Paper(title="Plain Paper", abstract="Abstract", authors=["Author"])
    ]

    agent = EmailerAgent(make_config(), storage=storage, sender=sender)
    result = agent.run(target_date=date(2026, 3, 12))

    assert result["success"] is False
    assert "not fully enhanced" in result["message"]
    sender.send_email.assert_not_called()


def test_emailer_agent_dry_run_uses_sender_dry_run_flag():
    """Test dry-run mode delegates to sender without real delivery."""
    storage = Mock()
    sender = Mock()
    storage.papers_exist_for_date.return_value = True
    storage.load_papers.return_value = [make_relevant_paper()]
    sender.send_email.return_value = {"success": True, "dry_run": True}

    agent = EmailerAgent(make_config(), storage=storage, sender=sender)
    result = agent.run(target_date=date(2026, 3, 12), dry_run=True)

    assert result["success"] is True
    assert result["dry_run"] is True
    _, kwargs = sender.send_email.call_args
    assert kwargs["dry_run"] is True
