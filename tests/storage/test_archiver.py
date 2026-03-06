"""Tests for archiver system."""

import json
import tarfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_agent.storage.archiver import Archiver


def create_daily_file(data_dir: Path, file_date: date, paper_count: int = 1):
    """Create a dummy daily JSON file for testing."""
    filename = f"papers_{file_date.isoformat()}.json"
    file_path = data_dir / filename

    data = {
        "date": file_date.isoformat(),
        "count": paper_count,
        "papers": [{"title": f"Paper {i}", "abstract": "", "authors": []} for i in range(paper_count)],
        "saved_at": "2023-01-01T00:00:00"
    }

    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


def test_archiver_initialization(temp_dir):
    """Test archiver initialization."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    assert archiver.data_dir == data_dir
    assert archiver.archive_dir == archive_dir
    assert archive_dir.exists()


def test_archive_month(temp_dir):
    """Test archiving a specific month."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"
    data_dir.mkdir()

    # Create daily files for January 2023
    dates = [
        date(2023, 1, 15),
        date(2023, 1, 16),
        date(2023, 1, 17),
    ]

    for d in dates:
        create_daily_file(data_dir, d)

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    success = archiver.archive_month(2023, 1)

    assert success is True

    # Check archive was created
    archive_path = archive_dir / "papers_2023-01.tar.gz"
    assert archive_path.exists()

    # Check daily files were removed
    for d in dates:
        file_path = data_dir / f"papers_{d.isoformat()}.json"
        assert not file_path.exists()

    # Check archive contents
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        assert len(members) == 3
        filenames = [m.name for m in members]
        assert all(f.startswith("papers_2023-01-") for f in filenames)


def test_archive_month_no_files(temp_dir):
    """Test archiving a month with no files."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    success = archiver.archive_month(2023, 1)

    assert success is False
    assert not (archive_dir / "papers_2023-01.tar.gz").exists()


def test_archive_old_data(temp_dir):
    """Test archiving data older than retention period."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"
    data_dir.mkdir()

    # Create files with various dates
    old_date = date.today() - timedelta(days=40)  # Older than 30 days
    recent_date = date.today() - timedelta(days=10)  # Within retention

    create_daily_file(data_dir, old_date)
    create_daily_file(data_dir, recent_date)

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    archives_created = archiver.archive_old_data(retention_days=30)

    # Only the old month should be archived
    if old_date.month == recent_date.month:
        # Same month, both would be archived (since we archive entire month)
        expected_count = 1
    else:
        expected_count = 1  # Only old month

    assert len(archives_created) == expected_count

    # Recent file should still exist
    recent_file = data_dir / f"papers_{recent_date.isoformat()}.json"
    # Might be archived if same month, check condition
    if old_date.month != recent_date.month:
        assert recent_file.exists()


def test_list_archives(temp_dir):
    """Test listing archive files."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))

    # Create dummy archive files
    archive_files = [
        archive_dir / "papers_2023-01.tar.gz",
        archive_dir / "papers_2023-02.tar.gz",
        archive_dir / "papers_2022-12.tar.gz",
    ]

    for af in archive_files:
        af.parent.mkdir(parents=True, exist_ok=True)
        af.write_bytes(b"dummy")

    archives = archiver.list_archives()
    assert len(archives) == 3

    # Should be sorted newest first
    assert archives[0].name == "papers_2023-02.tar.gz"
    assert archives[1].name == "papers_2023-01.tar.gz"
    assert archives[2].name == "papers_2022-12.tar.gz"


def test_extract_archive(temp_dir):
    """Test extracting an archive."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"
    extract_dir = temp_dir / "extract"

    # Create a simple archive
    archive_path = archive_dir / "papers_2023-01.tar.gz"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Create archive with a test file
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add a dummy file
        dummy_file = temp_dir / "dummy.txt"
        dummy_file.write_text("test content")
        tar.add(dummy_file, arcname="dummy.txt")

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    success = archiver.extract_archive(archive_path, extract_dir)

    assert success is True
    assert (extract_dir / "dummy.txt").exists()
    assert (extract_dir / "dummy.txt").read_text() == "test content"


def test_get_archive_info(temp_dir):
    """Test getting archive information."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    # Create a simple archive
    archive_path = archive_dir / "papers_2023-01.tar.gz"
    archive_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        dummy_file = temp_dir / "dummy.txt"
        dummy_file.write_text("test")
        tar.add(dummy_file, arcname="dummy.txt")

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    info = archiver.get_archive_info(archive_path)

    assert info is not None
    assert info["path"] == str(archive_path)
    assert info["file_count"] == 1
    assert "dummy.txt" in info["file_names"]
    assert info["size_bytes"] > 0
    assert "modified" in info


def test_get_archive_info_nonexistent(temp_dir):
    """Test getting info for non-existent archive."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))
    info = archiver.get_archive_info(archive_dir / "nonexistent.tar.gz")

    assert info is None


def test_cleanup_old_archives(temp_dir):
    """Test deleting old archives."""
    data_dir = temp_dir / "data"
    archive_dir = temp_dir / "archive"

    # Create archive files with different dates
    archive_files = [
        (archive_dir / "papers_2020-01.tar.gz", date(2020, 1, 1)),
        (archive_dir / "papers_2023-01.tar.gz", date(2023, 1, 1)),
        (archive_dir / "papers_2024-01.tar.gz", date(2024, 1, 1)),
    ]

    for archive_path, archive_date in archive_files:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"dummy")
        # Set modification time (not needed for this test)

    archiver = Archiver(data_dir=str(data_dir), archive_dir=str(archive_dir))

    # Delete archives older than 2 years (from 2024-01-01)
    # Mock date.today to be 2024-01-01
    # Create a mock date class that has today and fromisoformat methods
    class MockDate:
        @classmethod
        def today(cls):
            return date(2024, 1, 1)

        @classmethod
        def fromisoformat(cls, date_str):
            return date.fromisoformat(date_str)

    with patch('arxiv_agent.storage.archiver.date', MockDate):
        deleted_count = archiver.cleanup_old_archives(max_archive_age_days=365*2)  # 2 years
        # Archives from 2020 should be deleted (1), others kept
        assert deleted_count == 1
        assert not (archive_dir / "papers_2020-01.tar.gz").exists()
        assert (archive_dir / "papers_2023-01.tar.gz").exists()
        assert (archive_dir / "papers_2024-01.tar.gz").exists()