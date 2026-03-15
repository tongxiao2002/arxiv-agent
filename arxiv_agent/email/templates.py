"""Email template helpers for digest rendering."""

from __future__ import annotations

from html import escape
from typing import Iterable, List

from arxiv_agent.sources.enhanced_paper import EnhancedPaper
from arxiv_agent.utils.timezone import format_digest_date


def render_digest_subject(
    target_date,
    subject_template: str,
    timezone_name: str,
) -> str:
    """Render the digest subject from a configured template."""
    return subject_template.format(date=format_digest_date(target_date, timezone_name))


def render_no_papers_subject(target_date, timezone_name: str) -> str:
    """Render the subject for a no-papers notification."""
    return f"No papers today - {format_digest_date(target_date, timezone_name)}"


def render_digest_text(
    target_date,
    timezone_name: str,
    papers: Iterable[EnhancedPaper],
) -> str:
    """Render the plain-text digest body."""
    rendered_date = format_digest_date(target_date, timezone_name)
    paper_sections = [
        _render_paper_text(index, paper) for index, paper in enumerate(papers, start=1)
    ]
    joined_sections = "\n\n".join(paper_sections)
    return f"Daily papers digest for {rendered_date}\n\n" f"{joined_sections}\n"


def render_digest_html(
    target_date,
    timezone_name: str,
    papers: Iterable[EnhancedPaper],
) -> str:
    """Render the HTML digest body."""
    rendered_date = escape(format_digest_date(target_date, timezone_name))
    paper_sections = "".join(
        _render_paper_html(index, paper) for index, paper in enumerate(papers, start=1)
    )
    return (
        "<html><body>"
        f"<h1>Daily papers digest for {rendered_date}</h1>"
        f"{paper_sections}"
        "</body></html>"
    )


def render_no_papers_text(target_date, timezone_name: str) -> str:
    """Render the plain-text no-papers body."""
    rendered_date = format_digest_date(target_date, timezone_name)
    return (
        f"No relevant papers were found for {rendered_date}.\n\n"
        "Today's scan completed successfully, but none of the classified papers matched your topics."
    )


def render_no_papers_html(target_date, timezone_name: str) -> str:
    """Render the HTML no-papers body."""
    rendered_date = escape(format_digest_date(target_date, timezone_name))
    return (
        "<html><body>"
        f"<h1>No relevant papers for {rendered_date}</h1>"
        "<p>Today's scan completed successfully, but none of the classified papers matched your topics.</p>"
        "</body></html>"
    )


def _render_paper_text(index: int, paper: EnhancedPaper) -> str:
    """Render a single paper in plain text."""
    authors = ", ".join(paper.authors) if paper.authors else "Unknown authors"
    topics = ", ".join(paper.matched_topics) if paper.matched_topics else "None"
    summary = paper.summary or "No summary available."
    links = _paper_links_text(paper)
    return (
        f"{index}. {paper.title}\n"
        f"Authors: {authors}\n"
        f"Matched topics: {topics}\n"
        f"Summary: {summary}\n"
        f"{links}"
    )


def _render_paper_html(index: int, paper: EnhancedPaper) -> str:
    """Render a single paper in HTML."""
    authors = escape(", ".join(paper.authors) if paper.authors else "Unknown authors")
    topics = escape(", ".join(paper.matched_topics) if paper.matched_topics else "None")
    summary = escape(paper.summary or "No summary available.")
    title = escape(paper.title)
    links = _paper_links_html(paper)
    return (
        "<section>"
        f"<h2>{index}. {title}</h2>"
        f"<p><strong>Authors:</strong> {authors}</p>"
        f"<p><strong>Matched topics:</strong> {topics}</p>"
        f"<p><strong>Summary:</strong> {summary}</p>"
        f"{links}"
        "</section>"
    )


def _paper_links_text(paper: EnhancedPaper) -> str:
    """Render links for a paper in text format."""
    links: List[str] = []
    if paper.webpage_url:
        links.append(f"Paper: {paper.webpage_url}")
    if paper.pdf_url:
        links.append(f"PDF: {paper.pdf_url}")
    return "\n".join(links) if links else "Links: unavailable"


def _paper_links_html(paper: EnhancedPaper) -> str:
    """Render links for a paper in HTML format."""
    links: List[str] = []
    if paper.webpage_url:
        links.append(
            f'<a href="{escape(paper.webpage_url, quote=True)}">Paper page</a>'
        )
    if paper.pdf_url:
        links.append(f'<a href="{escape(paper.pdf_url, quote=True)}">PDF</a>')
    if not links:
        return "<p><strong>Links:</strong> unavailable</p>"
    return "<p><strong>Links:</strong> " + " | ".join(links) + "</p>"
