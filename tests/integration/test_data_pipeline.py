"""Integration tests for data pipeline."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.agents.scraper_agent import ScraperAgent
from arxiv_agent.sources.arxiv_source import ArxivSource
from arxiv_agent.storage.json_storage import JsonStorage


def test_data_pipeline_integration(temp_dir):
    """Test full data pipeline integration with mocked arXiv API."""
    # Create directories
    data_dir = temp_dir / "papers"
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 10},
        },
        "storage": {"data_dir": str(data_dir)},
    }

    # Mock arXiv API response
    sample_atom_feed = """<?xml version="1.0" encoding="UTF-8"?>
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

    with patch("arxiv_agent.sources.arxiv_source.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.text = sample_atom_feed
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch(
            "arxiv_agent.agents.scraper_agent.get_current_date_in_timezone",
            return_value=date(2023, 1, 15),
        ):
            agent = ScraperAgent(config)
            result = agent.run()

            assert result["success"] is True
            assert result["papers_fetched"] == 2
            assert result["source"] == "arxiv"
            assert result["storage_date"] == "2023-01-15"
            assert "categories" in result

            storage_file = data_dir / "papers_2023-01-15.json"
            assert storage_file.exists()

            with open(storage_file, "r") as f:
                stored_data = json.load(f)

            assert stored_data["date"] == "2023-01-15"
            assert stored_data["count"] == 2
            assert len(stored_data["papers"]) == 2

            paper1 = stored_data["papers"][0]
            assert paper1["title"] == "Test Paper Title"
            assert paper1["abstract"] == "Test abstract content."
            assert paper1["authors"] == ["Author One", "Author Two"]
            assert paper1["arxiv_id"] == "2101.12345"

            paper2 = stored_data["papers"][1]
            assert paper2["title"] == "Another Paper"
            assert paper2["arxiv_id"] == "2102.67890"

            loaded_papers = agent.get_stored_papers(date(2023, 1, 15))
            assert len(loaded_papers) == 2
            assert loaded_papers[0].title == "Test Paper Title"
            assert loaded_papers[1].arxiv_id == "2102.67890"

            dates = agent.get_available_dates()
            assert dates == [date(2023, 1, 15)]


def test_data_pipeline_with_papers_cool(temp_dir):
    """Test data pipeline with Papers.cool source (mocked)."""
    data_dir = temp_dir / "papers"
    config = {
        "sources": {
            "primary": "papers_cool",
            "papers_cool": {"categories": ["cs.ai"], "max_papers": 5},
        },
        "storage": {"data_dir": str(data_dir)},
    }

    # Mock Papers.cool source to return sample papers
    with patch("arxiv_agent.sources.papers_cool_source.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.text = "<html>Mock HTML</html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock BeautifulSoup to return no papers (simplify test)
        with patch("arxiv_agent.sources.papers_cool_source.BeautifulSoup") as mock_bs:
            mock_soup = Mock()
            mock_soup.find_all.return_value = []
            mock_bs.return_value = mock_soup

            with patch(
                "arxiv_agent.agents.scraper_agent.get_current_date_in_timezone",
                return_value=date(2023, 1, 16),
            ):
                agent = ScraperAgent(config)
                result = agent.run()

                assert result["success"] is True
                assert result["papers_fetched"] == 0
                assert result["message"] == "No papers fetched"

                storage_file = data_dir / "papers_2023-01-16.json"
                assert not storage_file.exists()


def test_data_pipeline_error_handling(temp_dir):
    """Test data pipeline error handling when source fails."""
    data_dir = temp_dir / "papers"
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 10},
        },
        "storage": {"data_dir": str(data_dir)},
    }

    # Mock arXiv API to raise exception
    with patch("arxiv_agent.sources.arxiv_source.requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")

        with patch(
            "arxiv_agent.agents.scraper_agent.get_current_date_in_timezone",
            return_value=date(2023, 1, 15),
        ):
            agent = ScraperAgent(config)
            result = agent.run()
            assert result["success"] is True
            assert result["papers_fetched"] == 0
            assert result["message"] == "No papers fetched"
            storage_file = data_dir / "papers_2023-01-15.json"
            assert not storage_file.exists()


def test_data_pipeline_storage_failure(temp_dir):
    """Test data pipeline when storage fails."""
    data_dir = temp_dir / "papers"
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 10},
        },
        "storage": {"data_dir": str(data_dir)},
    }

    sample_atom_feed = """<?xml version="1.0" encoding="UTF-8"?>
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
</feed>"""

    with patch("arxiv_agent.sources.arxiv_source.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.text = sample_atom_feed
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch(
            "arxiv_agent.agents.scraper_agent.get_current_date_in_timezone",
            return_value=date(2023, 1, 15),
        ):
            with patch(
                "arxiv_agent.storage.json_storage.JsonStorage.save_papers"
            ) as mock_save:
                mock_save.return_value = False

                agent = ScraperAgent(config)
                from arxiv_agent.utils.retry import RetryError

                with pytest.raises(
                    RetryError, match="Function run failed after 3 attempts"
                ):
                    agent.run()


def test_data_pipeline_config_validation():
    """Test configuration validation in pipeline."""
    # Missing storage section
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        }
    }

    agent = ScraperAgent(config)
    assert agent.validate() is False

    # Missing sources section
    config2 = {"storage": {"data_dir": "./papers"}}

    agent2 = ScraperAgent(config2)
    assert agent2.validate() is False

    # Invalid primary source
    config3 = {
        "sources": {
            "primary": "invalid",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    # Should raise ValueError during initialization
    with pytest.raises(ValueError, match="Unknown primary source"):
        ScraperAgent(config3)
