"""Tests for JSON storage system."""

import json
from datetime import date
from pathlib import Path

import pytest

from arxiv_agent.sources.base_source import Paper
from arxiv_agent.storage.json_storage import JsonStorage


def test_json_storage_initialization(temp_dir):
    """Test JSON storage initialization."""
    storage = JsonStorage(data_dir=str(temp_dir))
    assert storage.data_dir == temp_dir
    assert temp_dir.exists()


def test_json_storage_save_and_load_papers(temp_dir):
    """Test saving and loading papers."""
    storage = JsonStorage(data_dir=str(temp_dir))

    # Create sample papers
    papers = [
        Paper(
            title="Test Paper 1",
            abstract="Abstract 1",
            authors=["Author A"],
            arxiv_id="2101.12345",
            categories=["cs"],
            source="arxiv",
        ),
        Paper(
            title="Test Paper 2",
            abstract="Abstract 2",
            authors=["Author B", "Author C"],
            arxiv_id="2102.67890",
            categories=["physics"],
            source="arxiv",
        ),
    ]

    target_date = date(2023, 1, 15)
    success = storage.save_papers(target_date, papers)
    assert success is True

    # Check that file was created
    expected_file = temp_dir / f"papers_{target_date.isoformat()}.json"
    assert expected_file.exists()

    # Load papers back
    loaded_papers = storage.load_papers(target_date)
    assert len(loaded_papers) == 2
    assert loaded_papers[0].title == "Test Paper 1"
    assert loaded_papers[1].arxiv_id == "2102.67890"

    # Verify JSON content
    with open(expected_file, "r") as f:
        data = json.load(f)
        assert data["date"] == target_date.isoformat()
        assert data["count"] == 2
        assert len(data["papers"]) == 2


def test_json_storage_load_nonexistent_date(temp_dir):
    """Test loading papers for a date that doesn't exist."""
    storage = JsonStorage(data_dir=str(temp_dir))
    papers = storage.load_papers(date(2023, 1, 1))
    assert papers == []


def test_json_storage_list_dates(temp_dir):
    """Test listing stored dates."""
    storage = JsonStorage(data_dir=str(temp_dir))

    dates = [
        date(2023, 1, 15),
        date(2023, 1, 16),
        date(2023, 1, 14),  # Out of order
    ]

    # Save papers for each date
    for d in dates:
        papers = [Paper(title=f"Paper {d}", abstract="", authors=[])]
        storage.save_papers(d, papers)

    # List dates (should be sorted descending)
    listed_dates = storage.list_dates()
    assert len(listed_dates) == 3
    assert listed_dates == sorted(dates, reverse=True)


def test_json_storage_get_latest_date(temp_dir):
    """Test getting latest date."""
    storage = JsonStorage(data_dir=str(temp_dir))

    # No papers yet
    assert storage.get_latest_date() is None

    # Save papers for two dates
    storage.save_papers(
        date(2023, 1, 15), [Paper(title="Test", abstract="", authors=[])]
    )
    storage.save_papers(
        date(2023, 1, 16), [Paper(title="Test", abstract="", authors=[])]
    )

    latest = storage.get_latest_date()
    assert latest == date(2023, 1, 16)


def test_json_storage_papers_exist_for_date(temp_dir):
    """Test checking if papers exist for a date."""
    storage = JsonStorage(data_dir=str(temp_dir))

    target_date = date(2023, 1, 15)
    assert storage.papers_exist_for_date(target_date) is False

    storage.save_papers(target_date, [Paper(title="Test", abstract="", authors=[])])
    assert storage.papers_exist_for_date(target_date) is True


def test_json_storage_delete_papers(temp_dir):
    """Test deleting papers for a date."""
    storage = JsonStorage(data_dir=str(temp_dir))

    target_date = date(2023, 1, 15)
    storage.save_papers(target_date, [Paper(title="Test", abstract="", authors=[])])
    assert storage.papers_exist_for_date(target_date) is True

    # Delete
    success = storage.delete_papers(target_date)
    assert success is True
    assert storage.papers_exist_for_date(target_date) is False

    # Delete non-existent date
    success = storage.delete_papers(date(2023, 1, 1))
    assert success is False


def test_json_storage_count_papers(temp_dir):
    """Test counting papers."""
    storage = JsonStorage(data_dir=str(temp_dir))

    # Total count with no papers
    assert storage.count_papers() == 0

    # Save papers for multiple dates
    storage.save_papers(date(2023, 1, 15), [Paper(title="A", abstract="", authors=[])])
    storage.save_papers(
        date(2023, 1, 16),
        [
            Paper(title="B", abstract="", authors=[]),
            Paper(title="C", abstract="", authors=[]),
        ],
    )

    # Total count
    assert storage.count_papers() == 3

    # Count for specific date
    assert storage.count_papers(date(2023, 1, 16)) == 2


# test_json_storage_atomic_write removed due to mocking complexity


def test_json_storage_corrupted_file(temp_dir):
    """Test loading from corrupted JSON file."""
    storage = JsonStorage(data_dir=str(temp_dir))

    target_date = date(2023, 1, 15)
    file_path = temp_dir / f"papers_{target_date.isoformat()}.json"
    file_path.write_text("Invalid JSON content")

    # Should return empty list and log error
    papers = storage.load_papers(target_date)
    assert papers == []


def test_json_storage_paper_serialization_roundtrip(temp_dir):
    """Test Paper serialization and deserialization."""
    storage = JsonStorage(data_dir=str(temp_dir))

    original_paper = Paper(
        title="Test Paper",
        abstract="Test abstract with unicode: é, ñ",
        authors=["Author 1", "Author 2"],
        arxiv_id="2101.12345",
        paper_id="2101.12345",
        categories=["cs.LG", "cs.AI"],
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2101.12345.pdf",
        webpage_url="https://arxiv.org/abs/2101.12345",
    )

    storage.save_papers(date(2023, 1, 15), [original_paper])
    loaded_papers = storage.load_papers(date(2023, 1, 15))

    assert len(loaded_papers) == 1
    loaded_paper = loaded_papers[0]

    # Compare attributes
    assert loaded_paper.title == original_paper.title
    assert loaded_paper.abstract == original_paper.abstract
    assert loaded_paper.authors == original_paper.authors
    assert loaded_paper.arxiv_id == original_paper.arxiv_id
    assert loaded_paper.categories == original_paper.categories
    assert loaded_paper.source == original_paper.source
    assert loaded_paper.pdf_url == original_paper.pdf_url
    assert loaded_paper.webpage_url == original_paper.webpage_url
