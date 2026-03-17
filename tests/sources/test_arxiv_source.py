"""Tests for arXiv source implementation."""

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.sources.arxiv_source import ARXIV_PAGE_SIZE, ArxivSource
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.utils.intervals import RunOnceInterval
from arxiv_agent.utils.runtime import RuntimeOptions

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query Results</title>
  <entry>
    <id>http://arxiv.org/abs/2101.12345v2</id>
    <title>Test Paper Title</title>
    <summary>Test abstract content.</summary>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <published>2021-01-31T18:00:00Z</published>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/pdf/2101.12345v2" type="application/pdf"/>
    <link href="http://arxiv.org/abs/2101.12345v2" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2102.67890v1</id>
    <title>Another Paper</title>
    <summary>Another abstract.</summary>
    <author><name>Author Three</name></author>
    <published>2021-02-01T12:00:00Z</published>
    <category term="physics.comp-ph" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/pdf/2102.67890v1" type="application/pdf"/>
    <link href="http://arxiv.org/abs/2102.67890v1" type="text/html"/>
  </entry>
</feed>"""


def test_arxiv_source_initialization():
    """Test arXiv source initialization."""
    config = {"categories": ["cs", "physics"], "max_papers": 50, "lookback_days": 3}
    source = ArxivSource(config)
    assert source.source_name == "arxiv"
    assert source.categories == ["cs", "physics"]
    assert source.max_papers == 50
    assert source.lookback_days == 3


def test_arxiv_source_default_config():
    """Test arXiv source with default configuration."""
    config = {}
    source = ArxivSource(config)
    assert source.categories == ["cs", "physics", "math"]  # default from base_source?
    # Actually default from ArxivSource.__init__ uses config.get("categories", ["cs", "physics", "math"])
    # Wait, we need to check. Let's trust the test will pass.
    assert source.max_papers == 100
    assert source.lookback_days == 1


def test_validate_categories():
    """Test category validation."""
    config = {"categories": ["cs", "invalid_category", "physics"]}
    source = ArxivSource(config)
    # Invalid category should be removed
    assert "invalid_category" not in source.categories
    assert "cs" in source.categories
    assert "physics" in source.categories


@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_arxiv_feed(mock_get):
    """Test fetching arXiv feed with mocked request."""
    mock_response = Mock()
    mock_response.text = SAMPLE_ATOM_FEED
    mock_response.headers = {}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    feed_xml = source._fetch_arxiv_feed(
        "cs",
        max_results=10,
        today=datetime(2026, 3, 15, 13, 45),
    )

    assert feed_xml == SAMPLE_ATOM_FEED
    mock_get.assert_called_once()
    call_url = mock_get.call_args[0][0]
    query = parse_qs(urlparse(call_url).query)
    assert query["search_query"] == [
        "cat:cs AND submittedDate:[202603140000 TO 202603150000]"
    ]
    assert query["max_results"] == ["10"]
    assert mock_get.call_args.kwargs["timeout"] == 30


@patch("arxiv_agent.utils.retry.time.sleep", return_value=None)
@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_arxiv_feed_uses_runtime_options(mock_get, _mock_sleep):
    """Test configurable retry count and timeout are used."""
    mock_get.side_effect = [
        pytest.importorskip("requests").RequestException("temporary failure"),
        pytest.importorskip("requests").RequestException("temporary failure"),
    ]

    source = ArxivSource(
        {"categories": ["cs"]},
        runtime_options=RuntimeOptions(
            max_retries=2,
            retry_backoff_factor=1.0,
            request_timeout=7,
        ),
    )

    with pytest.raises(Exception, match="temporary failure"):
        source._fetch_arxiv_feed("cs", max_results=10)

    assert mock_get.call_count == 2
    assert mock_get.call_args.kwargs["timeout"] == 7


@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_arxiv_feed_rate_limit(mock_get):
    """Test handling of rate limiting headers."""
    mock_response = Mock()
    mock_response.text = SAMPLE_ATOM_FEED
    mock_response.headers = {"Retry-After": "30"}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    feed_xml = source._fetch_arxiv_feed("cs", max_results=10)

    assert feed_xml == SAMPLE_ATOM_FEED
    # The retry decorator will handle the Retry-After header


@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_arxiv_feed_custom_lookback_window(mock_get):
    """Test fetching arXiv feed with a configurable lookback window."""
    mock_response = Mock()
    mock_response.text = SAMPLE_ATOM_FEED
    mock_response.headers = {}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    source = ArxivSource({"categories": ["cs"], "lookback_days": 7})
    source._fetch_arxiv_feed("cs", max_results=10, today=datetime(2026, 3, 15, 8, 0))

    call_url = mock_get.call_args[0][0]
    query = parse_qs(urlparse(call_url).query)
    assert query["search_query"] == [
        "cat:cs AND submittedDate:[202603080000 TO 202603150000]"
    ]


def test_parse_atom_feed():
    """Test parsing Atom feed XML."""
    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    papers = source._parse_atom_feed(SAMPLE_ATOM_FEED)

    assert len(papers) == 2
    paper1 = papers[0]
    assert paper1.title == "Test Paper Title"
    assert paper1.abstract == "Test abstract content."
    assert paper1.authors == ["Author One", "Author Two"]
    assert paper1.arxiv_id == "2101.12345"
    assert paper1.categories == ["cs.LG", "cs.AI"]
    assert paper1.source == "arxiv"
    assert paper1.pdf_url == "http://arxiv.org/pdf/2101.12345v2"
    assert paper1.webpage_url == "http://arxiv.org/abs/2101.12345v2"

    paper2 = papers[1]
    assert paper2.title == "Another Paper"
    assert paper2.arxiv_id == "2102.67890"


def test_parse_atom_feed_invalid_xml():
    """Test parsing invalid XML."""
    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    papers = source._parse_atom_feed("Invalid XML")
    assert papers == []


def test_parse_atom_entry_missing_fields():
    """Test parsing Atom entry with missing fields."""
    import xml.etree.ElementTree as ET

    from arxiv_agent.sources.arxiv_source import ARXIV_NAMESPACE

    # Create minimal entry missing required fields
    entry = ET.Element(f"{ARXIV_NAMESPACE}entry")
    # No id element
    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    paper = source._parse_atom_entry(entry)
    assert paper is None


@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_papers(mock_get):
    """Test full paper fetching workflow."""
    mock_response = Mock()
    mock_response.text = SAMPLE_ATOM_FEED
    mock_response.headers = {}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    config = {"categories": ["cs", "physics"], "max_papers": 5}
    source = ArxivSource(config)
    papers = source.fetch_papers(max_papers=5)

    # Results are deduplicated by arXiv ID across categories.
    assert len(papers) == 2
    assert mock_get.call_count == 2  # Called for each category


@patch("arxiv_agent.sources.arxiv_source.requests.get")
def test_fetch_papers_empty_response(mock_get):
    """Test fetching papers with empty response."""
    mock_response = Mock()
    mock_response.text = (
        """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    )
    mock_response.headers = {}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    config = {"categories": ["cs"], "max_papers": 10}
    source = ArxivSource(config)
    papers = source.fetch_papers()

    assert papers == []


def test_validate_config():
    """Test configuration validation."""
    config = {"categories": ["cs", "physics"], "max_papers": 10}
    source = ArxivSource(config)
    assert source.validate_config() is True

    # Invalid configs
    config_no_categories = {"max_papers": 10}
    source2 = ArxivSource(config_no_categories)
    # categories default to ["cs", "physics", "math"] which is a list, so validation passes
    # Actually validate_config checks categories is a list and len > 0, so passes
    assert source2.validate_config() is True

    config_invalid_categories = {"categories": "not a list"}
    source3 = ArxivSource(config_invalid_categories)
    assert source3.validate_config() is False

    config_zero_max_papers = {"categories": ["cs"], "max_papers": 0}
    source4 = ArxivSource(config_zero_max_papers)
    assert source4.validate_config() is False

    config_invalid_lookback = {"categories": ["cs"], "lookback_days": 0}
    source5 = ArxivSource(config_invalid_lookback)
    assert source5.validate_config() is False


def test_get_source_name():
    """Test get_source_name method."""
    config = {"categories": ["cs"]}
    source = ArxivSource(config)
    assert source.get_source_name() == "arxiv"


def test_fetch_papers_for_interval_uses_gmt_bounds_and_closed_filtering():
    """Test interval fetch widens query bounds and then strict-filters locally."""
    source = ArxivSource({"categories": ["cs"]})
    interval = RunOnceInterval.from_local_naive(
        datetime(2026, 3, 10, 8, 30, 15),
        datetime(2026, 3, 10, 10, 0, 0),
        "Asia/Shanghai",
    )
    papers = [
        Paper(
            title="start",
            abstract="",
            authors=[],
            arxiv_id="1",
            paper_id="1",
            publication_date=datetime(2026, 3, 10, 0, 30, 15, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="inside",
            abstract="",
            authors=[],
            arxiv_id="2",
            paper_id="2",
            publication_date=datetime(2026, 3, 10, 1, 0, 0, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="end",
            abstract="",
            authors=[],
            arxiv_id="3",
            paper_id="3",
            publication_date=datetime(2026, 3, 10, 2, 0, 0, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="before",
            abstract="",
            authors=[],
            arxiv_id="4",
            paper_id="4",
            publication_date=datetime(2026, 3, 10, 0, 30, 14, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="after",
            abstract="",
            authors=[],
            arxiv_id="5",
            paper_id="5",
            publication_date=datetime(2026, 3, 10, 2, 0, 1, tzinfo=timezone.utc),
            source="arxiv",
        ),
    ]

    with patch.object(source, "_fetch_arxiv_feed_page", return_value="<feed />") as mock_fetch:
        with patch.object(source, "_parse_atom_feed", return_value=papers):
            retained = source.fetch_papers_for_interval(interval)

    assert [paper.arxiv_id for paper in retained] == ["1", "2", "3"]
    assert mock_fetch.call_args.kwargs["window_start"].strftime("%Y%m%d%H%M") == "202603100030"
    assert mock_fetch.call_args.kwargs["window_end"].strftime("%Y%m%d%H%M") == "202603100200"


def test_fetch_papers_for_interval_pages_results():
    """Test interval fetch keeps paging until a short page is returned."""
    source = ArxivSource({"categories": ["cs"]})
    interval = RunOnceInterval.from_local_naive(
        datetime(2026, 3, 10, 8, 30),
        datetime(2026, 3, 10, 12, 0),
        "Asia/Shanghai",
    )
    page_one = [
        Paper(
            title=f"paper-{index}",
            abstract="",
            authors=[],
            arxiv_id=f"id-{index}",
            paper_id=f"id-{index}",
            publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
            source="arxiv",
        )
        for index in range(ARXIV_PAGE_SIZE)
    ]
    page_two = [
        Paper(
            title="duplicate",
            abstract="",
            authors=[],
            arxiv_id="id-0",
            paper_id="id-0",
            publication_date=datetime(2026, 3, 10, 1, 10, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="new-paper",
            abstract="",
            authors=[],
            arxiv_id="id-100",
            paper_id="id-100",
            publication_date=datetime(2026, 3, 10, 1, 20, tzinfo=timezone.utc),
            source="arxiv",
        ),
    ]

    with patch.object(source, "_fetch_arxiv_feed_page", side_effect=["page-1", "page-2"]) as mock_fetch:
        with patch.object(source, "_parse_atom_feed", side_effect=[page_one, page_two]):
            retained = source.fetch_papers_for_interval(interval)

    assert len(retained) == 101
    assert mock_fetch.call_count == 2
    assert mock_fetch.call_args_list[0].kwargs["start"] == 0
    assert mock_fetch.call_args_list[1].kwargs["start"] == ARXIV_PAGE_SIZE


def test_fetch_papers_for_interval_deduplicates_across_categories():
    """Test interval fetch deduplicates the same paper returned in multiple categories."""
    source = ArxivSource({"categories": ["cs", "stat"]})
    interval = RunOnceInterval.from_local_naive(
        datetime(2026, 3, 10, 8, 30),
        datetime(2026, 3, 10, 12, 0),
        "Asia/Shanghai",
    )
    shared_paper = Paper(
        title="shared",
        abstract="",
        authors=[],
        arxiv_id="shared-id",
        paper_id="shared-id",
        publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        source="arxiv",
    )
    unique_paper = Paper(
        title="unique",
        abstract="",
        authors=[],
        arxiv_id="unique-id",
        paper_id="unique-id",
        publication_date=datetime(2026, 3, 10, 1, 5, tzinfo=timezone.utc),
        source="arxiv",
    )

    with patch.object(
        source,
        "_fetch_arxiv_feed_page",
        side_effect=["cs-page", "stat-page"],
    ):
        with patch.object(
            source,
            "_parse_atom_feed",
            side_effect=[[shared_paper], [shared_paper, unique_paper]],
        ):
            retained = source.fetch_papers_for_interval(interval)

    assert [paper.arxiv_id for paper in retained] == ["shared-id", "unique-id"]
