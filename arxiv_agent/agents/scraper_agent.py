"""Scraper agent for fetching papers from configured sources."""

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List

from arxiv_agent.agents.base import BaseAgent
from arxiv_agent.config import AdvancedConfig
from arxiv_agent.sources.arxiv_source import ArxivSource
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.papers_cool_source import PapersCoolSource
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.intervals import RunOnceInterval
from arxiv_agent.utils.retry import retry
from arxiv_agent.utils.runtime import RuntimeOptions
from arxiv_agent.utils.timezone import get_current_date_in_timezone

logger = logging.getLogger(__name__)


class ScraperAgent(BaseAgent):
    """Scraper agent that fetches papers from configured sources."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scraper agent.

        Args:
            config: Agent configuration with source settings
        """
        super().__init__(name="scraper", config=config)
        self.source = None
        self.storage = None
        self.timezone = config.get("agent", {}).get("timezone", "Asia/Shanghai")
        self.advanced_config = AdvancedConfig(**config.get("advanced", {}))
        self.runtime_options = RuntimeOptions.from_mapping(config.get("advanced", {}))
        self._setup_sources()
        self._setup_storage()

    def _setup_sources(self) -> None:
        """Set up paper sources based on configuration."""
        sources_config = self.config.get("sources", {})
        primary_source = sources_config.get("primary", "arxiv")

        if primary_source == "arxiv":
            arxiv_config = sources_config.get("arxiv", {})
            self.source = ArxivSource(
                arxiv_config,
                runtime_options=self.runtime_options,
            )
        elif primary_source == "papers_cool":
            papers_cool_config = sources_config.get("papers_cool", {})
            self.source = PapersCoolSource(
                papers_cool_config,
                runtime_options=self.runtime_options,
            )
        else:
            raise ValueError(f"Unknown primary source: {primary_source}")

        logger.info("Configured scraper agent with source: %s", primary_source)

    def _setup_storage(self) -> None:
        """Set up JSON storage."""
        storage_config = self.config.get("storage", {})
        data_dir = storage_config.get("data_dir", "./papers")
        self.storage = JsonStorage(data_dir=data_dir)
        logger.info("Configured scraper agent storage at %s", data_dir)

    @retry(max_retries=3, backoff_factor=2.0, jitter=True)
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Run scraper agent to fetch and store papers.

        Returns:
            Dictionary with execution results
        """
        logger.info("Starting scraper agent run")

        if not self.source:
            raise ValueError("Source not initialized")

        run_interval = kwargs.get("run_interval")
        if run_interval is not None and not isinstance(run_interval, RunOnceInterval):
            raise ValueError("run_interval must be a RunOnceInterval")

        target_date = (
            None
            if run_interval is not None
            else self._parse_target_date(*args, **kwargs)
        )

        # Validate source configuration
        if not self.source.validate_config():
            raise ValueError("Source configuration validation failed")

        # Determine maximum papers to fetch
        max_papers = None
        if self.source.source_name == "arxiv":
            max_papers = self.source.max_papers
        elif self.source.source_name == "papers_cool":
            max_papers = self.source.max_papers

        # Fetch papers
        logger.info("Fetching papers from %s", self.source.source_name)
        if run_interval is not None:
            if self.source.source_name != "arxiv":
                raise ValueError(
                    "Interval run-once is only supported for the arXiv source"
                )
            papers = self.source.fetch_papers_for_interval(
                run_interval,
                max_papers=max_papers,
            )
        elif self.source.source_name == "arxiv":
            assert target_date is not None
            fetch_today = datetime.combine(target_date + timedelta(days=1), time.min)
            papers = self.source.fetch_papers(max_papers=max_papers, today=fetch_today)
        else:
            papers = self.source.fetch_papers(max_papers=max_papers)

        if not papers:
            logger.warning("No papers fetched from %s", self.source.source_name)
            result = {
                "success": True,
                "papers_fetched": 0,
                "message": "No papers fetched",
            }
            if target_date is not None:
                result["target_date"] = target_date.isoformat()
            if run_interval is not None:
                result["interval"] = run_interval.to_dict()
                result["affected_days"] = []
                result["stored_by_day"] = {}
            return result

        # Store papers
        if run_interval is not None:
            stored_by_day = self._store_interval_papers(run_interval, papers)
            if stored_by_day is None:
                raise RuntimeError("Failed to merge interval papers into storage")

            logger.info(
                "Successfully fetched and stored %s interval papers across %s day(s)",
                len(papers),
                len(stored_by_day),
            )
            return {
                "success": True,
                "papers_fetched": len(papers),
                "source": self.source.source_name,
                "interval": run_interval.to_dict(),
                "affected_days": sorted(stored_by_day.keys()),
                "stored_by_day": stored_by_day,
                "categories": (
                    self.source.categories if hasattr(self.source, "categories") else []
                ),
            }

        assert target_date is not None
        if self.source.source_name == "arxiv":
            success = self.storage.merge_papers(target_date, papers)
        else:
            success = self.storage.save_papers(target_date, papers)

        if not success:
            raise RuntimeError("Failed to persist papers to storage")

        logger.info(
            "Successfully fetched and stored %s papers from %s",
            len(papers),
            self.source.source_name,
        )

        return {
            "success": True,
            "papers_fetched": len(papers),
            "source": self.source.source_name,
            "storage_date": target_date.isoformat(),
            "categories": (
                self.source.categories if hasattr(self.source, "categories") else []
            ),
        }

    def _parse_target_date(self, *args: Any, **kwargs: Any) -> date:
        """Parse a target date for storage."""
        if args and isinstance(args[0], str):
            return date.fromisoformat(args[0])
        if args and hasattr(args[0], "isoformat"):
            return args[0]

        target_date = kwargs.get("target_date")
        if isinstance(target_date, str):
            return date.fromisoformat(target_date)
        if target_date is not None and hasattr(target_date, "isoformat"):
            return target_date

        return get_current_date_in_timezone(self.timezone)

    def validate(self) -> bool:
        """
        Validate scraper agent configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Validating scraper agent configuration")

        # Check required configuration sections
        required_sections = ["sources", "storage"]
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing configuration section: {section}")
                return False

        # Validate source configuration
        sources_config = self.config.get("sources", {})
        primary_source = sources_config.get("primary")

        if primary_source not in ["arxiv", "papers_cool"]:
            logger.error(f"Invalid primary source: {primary_source}")
            return False

        # Validate source-specific configuration
        if primary_source == "arxiv":
            arxiv_config = sources_config.get("arxiv", {})
            if "categories" not in arxiv_config:
                logger.error("arXiv configuration missing 'categories'")
                return False
        elif primary_source == "papers_cool":
            papers_cool_config = sources_config.get("papers_cool", {})
            if "categories" not in papers_cool_config:
                logger.error("Papers.cool configuration missing 'categories'")
                return False

        # Validate storage configuration
        storage_config = self.config.get("storage", {})
        if "data_dir" not in storage_config:
            logger.error("Storage configuration missing 'data_dir'")
            return False

        logger.info("Scraper agent configuration validated successfully")
        return True

    def _setup(self) -> None:
        """Internal setup method."""
        # Additional setup if needed
        pass

    def _teardown(self) -> None:
        """Internal teardown method."""
        # Cleanup if needed
        pass

    def get_stored_papers(self, target_date: date) -> List[Paper]:
        """
        Get papers stored for a specific date.

        Args:
            target_date: Date to retrieve papers for

        Returns:
            List of Paper objects
        """
        if not self.storage:
            raise ValueError("Storage not initialized")
        return self.storage.load_papers(target_date)

    def get_available_dates(self) -> List[date]:
        """
        Get dates for which papers are stored.

        Returns:
            List of dates
        """
        if not self.storage:
            raise ValueError("Storage not initialized")
        return self.storage.list_dates()

    def _store_interval_papers(
        self,
        run_interval: RunOnceInterval,
        papers: List[Paper],
    ) -> Dict[str, int] | None:
        """Group interval papers by their raw publication date and merge storage."""
        grouped: Dict[str, List[Paper]] = {}
        for paper in papers:
            if paper.publication_date is None:
                logger.warning(
                    "Skipping paper without publication date during interval storage: %s",
                    paper.title,
                )
                continue
            storage_day = run_interval.storage_date_for(paper.publication_date)
            grouped.setdefault(storage_day.isoformat(), []).append(paper)

        stored_by_day: Dict[str, int] = {}
        for date_text, day_papers in grouped.items():
            storage_date = date.fromisoformat(date_text)
            if not self.storage.merge_papers(storage_date, day_papers):
                return None
            stored_by_day[date_text] = len(day_papers)

        return stored_by_day
