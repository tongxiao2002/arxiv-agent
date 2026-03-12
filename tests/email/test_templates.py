"""Tests for digest template rendering."""

from datetime import date

from arxiv_agent.email.templates import (
    render_digest_html,
    render_digest_subject,
    render_digest_text,
    render_no_papers_html,
    render_no_papers_subject,
    render_no_papers_text,
)
from arxiv_agent.sources.enhanced_paper import EnhancedPaper


def make_paper() -> EnhancedPaper:
    """Create a deterministic enhanced paper for template tests."""
    return EnhancedPaper(
        title="A Relevant Paper",
        abstract="Abstract",
        authors=["Author One", "Author Two"],
        arxiv_id="1234.5678",
        categories=["cs.LG"],
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
        webpage_url="https://arxiv.org/abs/1234.5678",
        relevance_score=0.9,
        is_relevant=True,
        summary="A concise summary.",
        matched_topics=["machine learning", "agents"],
        classification_reason="Strong topic overlap.",
    )


def test_render_digest_subject():
    """Test digest subject rendering."""
    subject = render_digest_subject(
        date(2026, 3, 12),
        "Digest for {date}",
        "Asia/Shanghai",
    )
    assert subject == "Digest for March 12, 2026"


def test_render_digest_text():
    """Test plain-text digest rendering."""
    body = render_digest_text(date(2026, 3, 12), "Asia/Shanghai", [make_paper()])
    assert "A Relevant Paper" in body
    assert "Author One, Author Two" in body
    assert "machine learning, agents" in body
    assert "A concise summary." in body
    assert "https://arxiv.org/abs/1234.5678" in body


def test_render_digest_html():
    """Test HTML digest rendering."""
    body = render_digest_html(date(2026, 3, 12), "Asia/Shanghai", [make_paper()])
    assert "<h1>Daily papers digest for March 12, 2026</h1>" in body
    assert "<strong>Matched topics:</strong> machine learning, agents" in body
    assert 'href="https://arxiv.org/abs/1234.5678"' in body


def test_render_no_papers_templates():
    """Test no-papers subject and body rendering."""
    subject = render_no_papers_subject(date(2026, 3, 12), "Asia/Shanghai")
    text_body = render_no_papers_text(date(2026, 3, 12), "Asia/Shanghai")
    html_body = render_no_papers_html(date(2026, 3, 12), "Asia/Shanghai")

    assert subject == "No papers today - March 12, 2026"
    assert "No relevant papers were found for March 12, 2026." in text_body
    assert "<h1>No relevant papers for March 12, 2026</h1>" in html_body
