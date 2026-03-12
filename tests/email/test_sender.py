"""Tests for SMTP email sending."""

import smtplib
from unittest.mock import Mock, patch

from arxiv_agent.config import EmailConfig
from arxiv_agent.email.sender import SmtpEmailSender


def make_email_config(**overrides):
    """Create a reusable SMTP config for sender tests."""
    values = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_security": "starttls",
        "smtp_username": "mailer",
        "from_email": "agent@example.com",
        "to_emails": ["user@example.com", "team@example.com"],
        "subject_template": "Digest - {date}",
    }
    values.update(overrides)
    return EmailConfig(**values)


def test_send_email_starttls_with_auth():
    """Test authenticated STARTTLS delivery."""
    config = make_email_config()
    smtp_client = Mock()

    with patch("arxiv_agent.email.sender.smtplib.SMTP", return_value=smtp_client) as mock_smtp:
        sender = SmtpEmailSender(config, smtp_password="smtp-secret")
        result = sender.send_email(
            subject="Digest",
            text_body="Text body",
            html_body="<p>HTML body</p>",
        )

    assert result["success"] is True
    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=30)
    assert smtp_client.ehlo.call_count == 2
    smtp_client.starttls.assert_called_once()
    smtp_client.login.assert_called_once_with("mailer", "smtp-secret")
    smtp_client.send_message.assert_called_once()
    _, kwargs = smtp_client.send_message.call_args
    assert kwargs["to_addrs"] == ["user@example.com", "team@example.com"]


def test_send_email_ssl_transport():
    """Test SSL delivery uses SMTP_SSL."""
    config = make_email_config(smtp_port=465, smtp_security="ssl")
    smtp_client = Mock()

    with patch(
        "arxiv_agent.email.sender.smtplib.SMTP_SSL",
        return_value=smtp_client,
    ) as mock_smtp_ssl:
        sender = SmtpEmailSender(config, smtp_password="smtp-secret")
        sender.send_email(
            subject="Digest",
            text_body="Text body",
            html_body="<p>HTML body</p>",
        )

    mock_smtp_ssl.assert_called_once_with("smtp.example.com", 465, timeout=30)
    smtp_client.starttls.assert_not_called()


def test_send_email_without_authentication():
    """Test unauthenticated SMTP flow skips login."""
    config = make_email_config(smtp_username="", smtp_security="none")
    smtp_client = Mock()

    with patch("arxiv_agent.email.sender.smtplib.SMTP", return_value=smtp_client):
        sender = SmtpEmailSender(config)
        sender.send_email(
            subject="Digest",
            text_body="Text body",
            html_body="<p>HTML body</p>",
        )

    smtp_client.login.assert_not_called()


def test_send_email_retries_transient_failure():
    """Test transient SMTP failures trigger retry."""
    config = make_email_config()
    smtp_client = Mock()
    smtp_client.send_message.side_effect = [
        smtplib.SMTPServerDisconnected("temporary disconnect"),
        None,
    ]

    with patch("arxiv_agent.email.sender.smtplib.SMTP", return_value=smtp_client):
        with patch("arxiv_agent.utils.retry.time.sleep", return_value=None):
            sender = SmtpEmailSender(config, smtp_password="smtp-secret")
            sender.send_email(
                subject="Digest",
                text_body="Text body",
                html_body="<p>HTML body</p>",
            )

    assert smtp_client.send_message.call_count == 2


def test_send_email_dry_run_does_not_connect():
    """Test dry-run path avoids SMTP completely."""
    config = make_email_config()

    with patch("arxiv_agent.email.sender.smtplib.SMTP") as mock_smtp:
        with patch("arxiv_agent.email.sender.smtplib.SMTP_SSL") as mock_smtp_ssl:
            sender = SmtpEmailSender(config, smtp_password="smtp-secret")
            result = sender.send_email(
                subject="Digest",
                text_body="Text body",
                html_body="<p>HTML body</p>",
                dry_run=True,
            )

    assert result["dry_run"] is True
    mock_smtp.assert_not_called()
    mock_smtp_ssl.assert_not_called()
