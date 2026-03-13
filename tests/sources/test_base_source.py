"""Tests for base source class."""

from unittest.mock import Mock

import pytest

from arxiv_agent.sources.base_source import BaseSource, Paper, SourceError


class ConcreteSource(BaseSource):
    """Concrete implementation for testing BaseSource."""

    def fetch_papers(self, max_papers=None):
        return [Paper(title="Test", abstract="Abstract", authors=["Author"])]

    def get_source_name(self):
        return self.source_name


def test_base_source_initialization():
    """Test BaseSource initialization."""
    config = {"categories": ["cs"]}
    source = ConcreteSource(config, source_name="test")
    assert source.config == config
    assert source.source_name == "test"
    assert source.logger.name == "source.test"


def test_base_source_abstract_methods():
    """Test that BaseSource cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseSource({}, "test")


def test_concrete_source_methods():
    """Test concrete source implementation."""
    config = {"categories": ["cs"]}
    source = ConcreteSource(config, source_name="test")
    papers = source.fetch_papers()
    assert len(papers) == 1
    assert papers[0].title == "Test"
    assert source.get_source_name() == "test"


def test_validate_config():
    """Test configuration validation."""
    config = {"categories": ["cs"]}
    source = ConcreteSource(config, source_name="test")
    assert source.validate_config() is True

    # Missing required key
    config_bad = {}
    source_bad = ConcreteSource(config_bad, source_name="test")
    assert source_bad.validate_config() is False


def test_paper_dataclass():
    """Test Paper dataclass serialization."""
    paper = Paper(
        title="Test Paper",
        abstract="Test abstract",
        authors=["Author 1", "Author 2"],
        arxiv_id="1234.56789",
        paper_id="1234.56789",
        categories=["cs"],
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/1234.56789.pdf",
        webpage_url="https://arxiv.org/abs/1234.56789",
    )

    # Check fields
    assert paper.title == "Test Paper"
    assert paper.arxiv_id == "1234.56789"

    # Convert to dict and back
    paper_dict = paper.to_dict()
    assert paper_dict["title"] == "Test Paper"
    assert paper_dict["arxiv_id"] == "1234.56789"

    paper_from_dict = Paper.from_dict(paper_dict)
    assert paper_from_dict.title == paper.title
    assert paper_from_dict.arxiv_id == paper.arxiv_id


def test_source_error():
    """Test SourceError exception."""
    error = SourceError("Test error", source_name="test_source")
    assert error.source_name == "test_source"
    assert "test_source" in str(error)

    error_no_source = SourceError("Test error")
    assert error_no_source.source_name is None
    assert "test_source" not in str(error_no_source)


def test_source_configuration_error():
    """Test SourceConfigurationError."""
    from arxiv_agent.sources.base_source import SourceConfigurationError

    error = SourceConfigurationError("Config error", source_name="test")
    assert isinstance(error, SourceError)


def test_source_network_error():
    """Test SourceNetworkError."""
    from arxiv_agent.sources.base_source import SourceNetworkError

    error = SourceNetworkError("Network error", source_name="test")
    assert isinstance(error, SourceError)
