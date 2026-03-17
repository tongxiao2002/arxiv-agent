"""Tests for scraper agent."""

from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.agents.scraper_agent import ScraperAgent
from arxiv_agent.sources.arxiv_source import ArxivSource
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.papers_cool_source import PapersCoolSource
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.intervals import RunOnceInterval
from arxiv_agent.utils.retry import RetryError


def test_scraper_agent_initialization():
    """Test scraper agent initialization with arXiv source."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs", "physics"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    assert agent.name == "scraper"
    assert isinstance(agent.source, ArxivSource)
    assert isinstance(agent.storage, JsonStorage)
    assert agent.source.categories == ["cs", "physics"]
    assert agent.source.max_papers == 50


def test_scraper_agent_initialization_papers_cool():
    """Test scraper agent initialization with Papers.cool source."""
    config = {
        "sources": {
            "primary": "papers_cool",
            "papers_cool": {"categories": ["cs.ai", "cs.lg"], "max_papers": 30},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    assert agent.name == "scraper"
    assert isinstance(agent.source, PapersCoolSource)
    assert isinstance(agent.storage, JsonStorage)
    assert agent.source.categories == ["cs.ai", "cs.lg"]
    assert agent.source.max_papers == 30


def test_scraper_agent_initialization_invalid_source():
    """Test scraper agent initialization with invalid source."""
    config = {
        "sources": {
            "primary": "invalid_source",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    with pytest.raises(ValueError, match="Unknown primary source"):
        ScraperAgent(config)


def test_scraper_agent_validate_valid_config():
    """Test configuration validation with valid config."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    assert agent.validate() is True


def test_scraper_agent_validate_missing_sections():
    """Test configuration validation with missing sections."""
    config = {"sources": {"primary": "arxiv", "arxiv": {"categories": ["cs"]}}}
    agent = ScraperAgent(config)
    assert agent.validate() is False  # Missing storage section

    config2 = {"storage": {"data_dir": "./papers"}}
    agent2 = ScraperAgent(config2)
    assert agent2.validate() is False  # Missing sources section


def test_scraper_agent_validate_invalid_primary_source():
    """Test configuration validation with invalid primary source."""
    config = {
        "sources": {
            "primary": "invalid",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    with pytest.raises(ValueError, match="Unknown primary source"):
        ScraperAgent(config)


def test_scraper_agent_validate_missing_categories():
    """Test configuration validation with missing categories."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {},  # Missing categories
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    assert agent.validate() is False


def test_scraper_agent_validate_missing_data_dir():
    """Test configuration validation with missing data_dir."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {},  # Missing data_dir
    }

    agent = ScraperAgent(config)
    assert agent.validate() is False


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "fetch_papers")
@patch.object(JsonStorage, "merge_papers")
def test_scraper_agent_run_success_arxiv(mock_merge, mock_fetch, mock_now):
    """Test successful scraper agent run with arXiv source."""
    mock_now.return_value = date(2023, 1, 15)

    # Mock papers
    mock_papers = [
        Mock(
            title="Paper 1",
            abstract="Abstract 1",
            authors=["Author 1"],
            arxiv_id="2101.12345",
        ),
        Mock(
            title="Paper 2",
            abstract="Abstract 2",
            authors=["Author 2"],
            arxiv_id="2102.67890",
        ),
    ]
    mock_fetch.return_value = mock_papers
    mock_merge.return_value = True

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    result = agent.run()

    assert result["success"] is True
    assert result["papers_fetched"] == 2
    assert result["source"] == "arxiv"
    assert result["storage_date"] == "2023-01-15"
    assert "categories" in result
    mock_fetch.assert_called_once_with(
        max_papers=50,
        today=datetime(2023, 1, 16, 0, 0),
    )
    mock_merge.assert_called_once_with(date(2023, 1, 15), mock_papers)


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(PapersCoolSource, "fetch_papers")
@patch.object(JsonStorage, "save_papers")
def test_scraper_agent_run_success_papers_cool(mock_save, mock_fetch, mock_now):
    """Test successful scraper agent run with Papers.cool source."""
    mock_now.return_value = date(2023, 1, 16)

    mock_papers = [
        Mock(title="Paper A", abstract="Abstract A", authors=["Author A"]),
    ]
    mock_fetch.return_value = mock_papers
    mock_save.return_value = True

    config = {
        "sources": {
            "primary": "papers_cool",
            "papers_cool": {"categories": ["cs.ai"], "max_papers": 30},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    result = agent.run()

    assert result["success"] is True
    assert result["papers_fetched"] == 1
    assert result["source"] == "papers_cool"
    mock_fetch.assert_called_once_with(max_papers=30)
    mock_save.assert_called_once_with(date(2023, 1, 16), mock_papers)


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "fetch_papers")
def test_scraper_agent_run_no_papers(mock_fetch, mock_now):
    """Test scraper agent run when no papers are fetched."""
    mock_now.return_value = date(2023, 1, 15)
    mock_fetch.return_value = []

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    result = agent.run()

    assert result["success"] is True
    assert result["papers_fetched"] == 0
    assert result["message"] == "No papers fetched"
    mock_fetch.assert_called_once_with(
        max_papers=50,
        today=datetime(2023, 1, 16, 0, 0),
    )


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "fetch_papers")
@patch.object(JsonStorage, "merge_papers")
def test_scraper_agent_run_save_failure(mock_merge, mock_fetch, mock_now):
    """Test scraper agent run when arXiv paper persistence fails."""
    mock_now.return_value = date(2023, 1, 15)
    mock_papers = [Mock(title="Paper", abstract="Abstract", authors=["Author"])]
    mock_fetch.return_value = mock_papers
    mock_merge.return_value = False

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    with pytest.raises(RetryError, match="Function run failed after 3 attempts"):
        agent.run()


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "validate_config")
def test_scraper_agent_run_source_validation_fails(mock_validate, mock_now):
    """Test scraper agent run when source validation fails."""
    mock_now.return_value = date(2023, 1, 15)
    mock_validate.return_value = False

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    with pytest.raises(RetryError, match="Function run failed after 3 attempts"):
        agent.run()


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "fetch_papers")
def test_scraper_agent_run_source_not_initialized(mock_fetch, mock_now):
    """Test scraper agent run when source is not initialized."""
    mock_now.return_value = date(2023, 1, 15)
    mock_fetch.return_value = []

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    # Manually set source to None to simulate initialization failure
    agent.source = None
    with pytest.raises(RetryError, match="Function run failed after 3 attempts"):
        agent.run()


@patch.object(JsonStorage, "merge_papers")
@patch.object(ArxivSource, "fetch_papers_for_interval")
def test_scraper_agent_run_interval_groups_and_merges(mock_fetch, mock_merge):
    """Test interval runs split papers by local day and merge into daily files."""
    mock_fetch.return_value = [
        Paper(
            title="Day One",
            abstract="",
            authors=["Author"],
            arxiv_id="2603.00001",
            paper_id="2603.00001",
            publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
            source="arxiv",
        ),
        Paper(
            title="Day Two",
            abstract="",
            authors=["Author"],
            arxiv_id="2603.00002",
            paper_id="2603.00002",
            publication_date=datetime(2026, 3, 10, 17, 30, tzinfo=timezone.utc),
            source="arxiv",
        ),
    ]
    mock_merge.return_value = True

    config = {
        "agent": {"timezone": "Asia/Shanghai"},
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": "./papers"},
    }
    interval = RunOnceInterval.from_local_naive(
        datetime(2026, 3, 10, 8, 30),
        datetime(2026, 3, 11, 9, 0),
        "Asia/Shanghai",
    )

    agent = ScraperAgent(config)
    result = agent.run(run_interval=interval)

    assert result["success"] is True
    assert result["affected_days"] == ["2026-03-10", "2026-03-11"]
    assert result["stored_by_day"] == {"2026-03-10": 1, "2026-03-11": 1}
    mock_fetch.assert_called_once_with(interval, max_papers=50)
    assert mock_merge.call_count == 2
    assert mock_merge.call_args_list[0].args[0] == date(2026, 3, 10)
    assert mock_merge.call_args_list[1].args[0] == date(2026, 3, 11)


@patch("arxiv_agent.agents.scraper_agent.get_current_date_in_timezone")
@patch.object(ArxivSource, "fetch_papers")
def test_scraper_agent_daily_rerun_merges_without_overwriting_existing_raw(
    mock_fetch,
    mock_now,
    temp_dir,
):
    """Test daily arXiv reruns preserve existing raw records and append new papers."""
    target_date = date(2026, 3, 10)
    mock_now.return_value = target_date
    original_paper = Paper(
        title="Original",
        abstract="Keep me",
        authors=["Author"],
        arxiv_id="2603.00001",
        paper_id="2603.00001",
        publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        source="arxiv",
    )
    duplicate_raw = Paper(
        title="Original",
        abstract="Incoming duplicate should lose",
        authors=["Author"],
        arxiv_id="2603.00001",
        paper_id="2603.00001",
        publication_date=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        source="arxiv",
    )
    new_paper = Paper(
        title="New",
        abstract="Brand new paper",
        authors=["Author"],
        arxiv_id="2603.00002",
        paper_id="2603.00002",
        publication_date=datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc),
        source="arxiv",
    )
    mock_fetch.side_effect = [[original_paper], [duplicate_raw, new_paper]]

    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 50},
        },
        "storage": {"data_dir": str(temp_dir)},
    }

    agent = ScraperAgent(config)
    first_result = agent.run()
    second_result = agent.run()

    assert first_result["success"] is True
    assert second_result["success"] is True

    stored_papers = agent.get_stored_papers(target_date)
    stored_by_id = {paper.arxiv_id: paper for paper in stored_papers}
    assert set(stored_by_id) == {"2603.00001", "2603.00002"}
    assert stored_by_id["2603.00001"].abstract == "Keep me"
    assert stored_by_id["2603.00002"].abstract == "Brand new paper"


def test_scraper_agent_get_stored_papers():
    """Test retrieving stored papers."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    target_date = date(2023, 1, 15)

    # Mock storage.load_papers
    with patch.object(agent.storage, "load_papers") as mock_load:
        mock_load.return_value = ["paper1", "paper2"]
        papers = agent.get_stored_papers(target_date)

        assert papers == ["paper1", "paper2"]
        mock_load.assert_called_once_with(target_date)


def test_scraper_agent_get_stored_papers_storage_not_initialized():
    """Test retrieving stored papers when storage is not initialized."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    agent.storage = None

    with pytest.raises(ValueError, match="Storage not initialized"):
        agent.get_stored_papers(date(2023, 1, 15))


def test_scraper_agent_get_available_dates():
    """Test retrieving available dates."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)

    with patch.object(agent.storage, "list_dates") as mock_list:
        mock_list.return_value = [date(2023, 1, 15), date(2023, 1, 16)]
        dates = agent.get_available_dates()

        assert dates == [date(2023, 1, 15), date(2023, 1, 16)]
        mock_list.assert_called_once()


def test_scraper_agent_get_available_dates_storage_not_initialized():
    """Test retrieving available dates when storage is not initialized."""
    config = {
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"]},
        },
        "storage": {"data_dir": "./papers"},
    }

    agent = ScraperAgent(config)
    agent.storage = None

    with pytest.raises(ValueError, match="Storage not initialized"):
        agent.get_available_dates()
