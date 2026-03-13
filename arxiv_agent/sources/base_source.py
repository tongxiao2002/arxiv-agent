"""Abstract base class for paper sources."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Paper metadata representation."""

    # Required fields
    title: str
    abstract: str
    authors: List[str]
    arxiv_id: Optional[str] = None  # arXiv ID if available
    paper_id: str = ""  # Source-specific paper ID
    publication_date: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    source: str = ""  # Source name (e.g., "arxiv", "papers_cool")
    pdf_url: Optional[str] = None
    webpage_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary for JSON serialization."""
        result = {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "arxiv_id": self.arxiv_id,
            "paper_id": self.paper_id,
            "categories": self.categories,
            "source": self.source,
            "pdf_url": self.pdf_url,
            "webpage_url": self.webpage_url,
        }
        if self.publication_date:
            result["publication_date"] = self.publication_date.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """Create Paper from dictionary."""
        publication_date = None
        if "publication_date" in data and data["publication_date"]:
            publication_date = datetime.fromisoformat(data["publication_date"])
        return cls(
            title=data["title"],
            abstract=data["abstract"],
            authors=data["authors"],
            arxiv_id=data.get("arxiv_id"),
            paper_id=data.get("paper_id", ""),
            publication_date=publication_date,
            categories=data.get("categories", []),
            source=data.get("source", ""),
            pdf_url=data.get("pdf_url"),
            webpage_url=data.get("webpage_url"),
        )


class BaseSource(ABC):
    """Abstract base class for paper sources."""

    def __init__(self, config: Dict[str, Any], source_name: str):
        """
        Initialize source with configuration.

        Args:
            config: Source-specific configuration dictionary
            source_name: Name of this source (e.g., "arxiv", "papers_cool")
        """
        self.config = config
        self.source_name = source_name
        self.logger = logging.getLogger(f"source.{source_name}")

    @abstractmethod
    def fetch_papers(self, max_papers: Optional[int] = None) -> List[Paper]:
        """
        Fetch papers from this source.

        Args:
            max_papers: Maximum number of papers to fetch (None for no limit)

        Returns:
            List of Paper objects
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Get the name of this source.

        Returns:
            Source name (e.g., "arxiv", "papers_cool")
        """
        return self.source_name

    def validate_config(self) -> bool:
        """
        Validate source configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = ["categories"]
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required config key: {key}")
                return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.source_name})"


class SourceError(Exception):
    """Base exception for source-related errors."""

    def __init__(self, message: str, source_name: Optional[str] = None):
        self.source_name = source_name
        self.message = message
        super().__init__(f"Source {source_name}: {message}" if source_name else message)


class SourceConfigurationError(SourceError):
    """Exception for source configuration errors."""

    pass


class SourceNetworkError(SourceError):
    """Exception for source network errors."""

    pass
