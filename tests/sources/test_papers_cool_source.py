"""Tests for Papers.cool source implementation."""

import pytest
from unittest.mock import Mock, patch

from arxiv_agent.sources.papers_cool_source import PapersCoolSource


def test_papers_cool_source_initialization():
    """Test Papers.cool source initialization."""
    config = {"categories": ["cs.ai", "cs.lg"], "max_papers": 30}
    source = PapersCoolSource(config)
    assert source.source_name == "papers_cool"
    assert source.categories == ["cs.ai", "cs.lg"]
    assert source.max_papers == 30


def test_papers_cool_source_default_config():
    """Test Papers.cool source with default configuration."""
    config = {}
    source = PapersCoolSource(config)
    assert source.categories == ["cs.ai", "cs.lg"]
    assert source.max_papers == 50


def test_validate_categories():
    """Test category validation."""
    config = {"categories": ["cs.ai", "invalid_category", "cs.lg"]}
    source = PapersCoolSource(config)
    # Invalid category should be removed
    assert "invalid_category" not in source.categories
    assert "cs.ai" in source.categories
    assert "cs.lg" in source.categories


@patch("arxiv_agent.sources.papers_cool_source.requests.get")
def test_fetch_category_page(mock_get):
    """Test fetching category page with mocked request."""
    mock_response = Mock()
    mock_response.text = "<html>Test HTML</html>"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    config = {"categories": ["cs.ai"]}
    source = PapersCoolSource(config)
    html = source._fetch_category_page("cs.ai")

    assert html == "<html>Test HTML</html>"
    mock_get.assert_called_once()
    # Check that URL contains the category
    call_url = mock_get.call_args[0][0]
    assert "papers.cool" in call_url
    assert "cs.ai" in call_url


@patch("arxiv_agent.sources.papers_cool_source.requests.get")
def test_fetch_category_page_failure(mock_get):
    """Test handling of request failure after retries."""
    mock_get.side_effect = Exception("Network error")

    config = {"categories": ["cs.ai"]}
    source = PapersCoolSource(config)
    with pytest.raises(Exception) as exc_info:
        source._fetch_category_page("cs.ai")
    # Should raise RetryError after all retries exhausted
    assert "failed after" in str(exc_info.value)


@patch("arxiv_agent.sources.papers_cool_source.BeautifulSoup")
def test_parse_html_page(mock_bs):
    """Test HTML parsing."""
    mock_soup = Mock()
    mock_bs.return_value = mock_soup
    # Mock that no paper elements are found (placeholder implementation)
    mock_soup.find_all.return_value = []

    config = {"categories": ["cs.ai"]}
    source = PapersCoolSource(config)
    papers = source._parse_html_page("<html></html>", "cs.ai")

    # Placeholder implementation returns empty list
    assert papers == []


def test_parse_html_page_invalid_html():
    """Test parsing invalid HTML."""
    config = {"categories": ["cs.ai"]}
    source = PapersCoolSource(config)
    papers = source._parse_html_page("Invalid HTML", "cs.ai")

    # Should return empty list
    assert papers == []


@patch("arxiv_agent.sources.papers_cool_source.requests.get")
@patch("arxiv_agent.sources.papers_cool_source.BeautifulSoup")
def test_fetch_papers(mock_bs, mock_get):
    """Test full paper fetching workflow."""
    mock_response = Mock()
    mock_response.text = "<html>Test</html>"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    mock_soup = Mock()
    mock_soup.find_all.return_value = []
    mock_bs.return_value = mock_soup

    config = {"categories": ["cs.ai", "cs.lg"], "max_papers": 10}
    source = PapersCoolSource(config)
    papers = source.fetch_papers(max_papers=10)

    # Placeholder returns empty list
    assert papers == []
    # Should have made requests for each category
    assert mock_get.call_count == 2


def test_validate_config():
    """Test configuration validation."""
    config = {"categories": ["cs.ai", "cs.lg"], "max_papers": 20}
    source = PapersCoolSource(config)
    assert source.validate_config() is True

    # Invalid configs
    config_no_categories = {"max_papers": 20}
    source2 = PapersCoolSource(config_no_categories)
    # categories default to ["cs.ai", "cs.lg"] which is a list
    assert source2.validate_config() is True

    config_invalid_categories = {"categories": "not a list"}
    source3 = PapersCoolSource(config_invalid_categories)
    assert source3.validate_config() is False

    config_zero_max_papers = {"categories": ["cs.ai"], "max_papers": 0}
    source4 = PapersCoolSource(config_zero_max_papers)
    assert source4.validate_config() is False


def test_get_source_name():
    """Test get_source_name method."""
    config = {"categories": ["cs.ai"]}
    source = PapersCoolSource(config)
    assert source.get_source_name() == "papers_cool"