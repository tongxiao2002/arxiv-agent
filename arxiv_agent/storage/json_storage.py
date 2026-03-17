"""JSON file storage system for paper metadata."""

import json
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper

logger = logging.getLogger(__name__)


class JsonStorage:
    """JSON file storage for paper metadata."""

    def __init__(self, data_dir: str = "./papers"):
        """
        Initialize JSON storage.

        Args:
            data_dir: Directory for storing JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized JSON storage at {self.data_dir}")

    def _get_file_path(self, target_date: date) -> Path:
        """
        Get file path for a given date.

        Args:
            target_date: Date for which to get file path

        Returns:
            Path to JSON file
        """
        filename = f"papers_{target_date.isoformat()}.json"
        return self.data_dir / filename

    def _paper_to_dict(self, paper: Paper) -> dict:
        """
        Convert Paper or EnhancedPaper to dictionary with enhanced flag.

        Args:
            paper: Paper or EnhancedPaper object

        Returns:
            Dictionary representation with enhanced flag
        """
        paper_dict = paper.to_dict()

        # Add enhanced flag if not already present
        if isinstance(paper, EnhancedPaper):
            paper_dict["enhanced"] = True
        elif "enhanced" not in paper_dict:
            paper_dict["enhanced"] = False

        return paper_dict

    def _dict_to_paper(self, paper_dict: dict) -> Paper:
        """
        Convert dictionary to Paper or EnhancedPaper based on enhanced flag.

        Args:
            paper_dict: Dictionary representation of a paper

        Returns:
            Paper or EnhancedPaper object
        """
        # Check enhanced flag
        if paper_dict.get("enhanced") is True:
            return EnhancedPaper.from_dict(paper_dict)
        else:
            return Paper.from_dict(paper_dict)

    def save_papers(self, target_date: date, papers: List[Paper]) -> bool:
        """
        Save papers to JSON file for a given date.

        Args:
            target_date: Date to associate with papers
            papers: List of Paper objects to save

        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_file_path(target_date)
        temp_path = None

        try:
            # Prepare data for JSON serialization
            data = {
                "date": target_date.isoformat(),
                "count": len(papers),
                "papers": [self._paper_to_dict(paper) for paper in papers],
                "saved_at": datetime.now().isoformat(),
            }

            # Write to temporary file first (atomic write)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.data_dir,
                prefix=".tmp_",
                suffix=".json",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(data, temp_file, indent=2, ensure_ascii=False)

            # Move temporary file to final location
            temp_path.replace(file_path)
            logger.info(f"Saved {len(papers)} papers to {file_path}")
            return True

        except (IOError, OSError, TypeError) as e:
            logger.error(f"Failed to save papers to {file_path}: {e}")
            # Clean up temporary file if it exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False

    def merge_papers(self, target_date: date, papers: List[Paper]) -> bool:
        """
        Merge papers into an existing daily file without dropping unrelated entries.

        Incoming enhanced papers replace raw records for the same identifier, while
        existing enhanced papers are preserved if a later merge only provides raw data.
        """
        existing_papers = self.load_papers(target_date)
        merged_papers = self._merge_paper_lists(existing_papers, papers)
        return self.save_papers(target_date, merged_papers)

    def load_papers(self, target_date: date) -> List[Paper]:
        """
        Load papers from JSON file for a given date.

        Args:
            target_date: Date to load papers for

        Returns:
            List of Paper objects
        """
        file_path = self._get_file_path(target_date)

        if not file_path.exists():
            logger.warning(f"No papers found for date {target_date}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert dicts back to Paper objects
            papers = []
            for paper_dict in data.get("papers", []):
                try:
                    paper = self._dict_to_paper(paper_dict)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper from dict: {e}")
                    continue

            logger.info(f"Loaded {len(papers)} papers from {file_path}")
            return papers

        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load papers from {file_path}: {e}")
            return []

    def list_dates(self) -> List[date]:
        """
        List all dates for which papers are stored.

        Returns:
            List of dates in descending order (most recent first)
        """
        dates = []
        for file_path in self.data_dir.glob("papers_*.json"):
            try:
                # Extract date from filename: papers_YYYY-MM-DD.json
                date_str = file_path.stem.split("_")[1]
                file_date = date.fromisoformat(date_str)
                dates.append(file_date)
            except (IndexError, ValueError):
                logger.warning(f"Invalid filename format: {file_path.name}")
                continue

        # Sort descending (most recent first)
        dates.sort(reverse=True)
        return dates

    def get_latest_date(self) -> Optional[date]:
        """
        Get the most recent date for which papers are stored.

        Returns:
            Most recent date, or None if no papers are stored
        """
        dates = self.list_dates()
        return dates[0] if dates else None

    def papers_exist_for_date(self, target_date: date) -> bool:
        """
        Check if papers exist for a given date.

        Args:
            target_date: Date to check

        Returns:
            True if papers exist, False otherwise
        """
        file_path = self._get_file_path(target_date)
        return file_path.exists()

    def delete_papers(self, target_date: date) -> bool:
        """
        Delete papers for a given date.

        Args:
            target_date: Date to delete papers for

        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_file_path(target_date)

        if not file_path.exists():
            logger.warning(f"No papers to delete for date {target_date}")
            return False

        try:
            file_path.unlink()
            logger.info(f"Deleted papers for date {target_date}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete papers for date {target_date}: {e}")
            return False

    def count_papers(self, target_date: Optional[date] = None) -> int:
        """
        Count total papers stored.

        Args:
            target_date: If provided, count papers for specific date only

        Returns:
            Number of papers
        """
        if target_date:
            papers = self.load_papers(target_date)
            return len(papers)

        total = 0
        for file_date in self.list_dates():
            papers = self.load_papers(file_date)
            total += len(papers)

        return total

    def _merge_paper_lists(
        self,
        existing_papers: List[Paper],
        incoming_papers: List[Paper],
    ) -> List[Paper]:
        """Merge paper lists by stable identifier while preserving daily file shape."""
        merged: dict[str, Paper] = {}

        for paper in existing_papers:
            merged[self._paper_identity(paper)] = paper

        for paper in incoming_papers:
            identity = self._paper_identity(paper)
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = paper
                continue
            merged[identity] = self._select_preferred_paper(existing, paper)

        return list(merged.values())

    def _paper_identity(self, paper: Paper) -> str:
        """Build a stable identifier for merge/upsert behavior."""
        return (
            paper.arxiv_id
            or paper.paper_id
            or paper.webpage_url
            or paper.pdf_url
            or paper.title
        )

    def _select_preferred_paper(self, existing: Paper, incoming: Paper) -> Paper:
        """Preserve stored papers unless the incoming record upgrades raw to enhanced."""
        if isinstance(existing, EnhancedPaper) and not isinstance(
            incoming, EnhancedPaper
        ):
            return existing
        if not isinstance(existing, EnhancedPaper) and isinstance(
            incoming, EnhancedPaper
        ):
            return incoming
        return existing
