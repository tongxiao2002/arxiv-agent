"""Email helpers for Arxiv-Agent."""

from .sender import EmailDeliveryError, SmtpEmailSender
from .templates import (
    render_digest_html,
    render_digest_subject,
    render_digest_text,
    render_no_papers_html,
    render_no_papers_subject,
    render_no_papers_text,
)

__all__ = [
    "EmailDeliveryError",
    "SmtpEmailSender",
    "render_digest_subject",
    "render_digest_text",
    "render_digest_html",
    "render_no_papers_subject",
    "render_no_papers_text",
    "render_no_papers_html",
]
