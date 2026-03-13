"""Enhanced Paper dataclass with LLM classification results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_source import Paper


@dataclass
class EnhancedPaper(Paper):
    """Paper with LLM classification results."""

    # LLM classification fields
    relevance_score: Optional[float] = None
    is_relevant: Optional[bool] = None
    summary: Optional[str] = None
    matched_topics: List[str] = field(default_factory=list)
    classification_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert enhanced paper to dictionary for JSON serialization."""
        base_dict = super().to_dict()

        # Add LLM classification fields if they exist
        if self.relevance_score is not None:
            base_dict["relevance_score"] = self.relevance_score
        if self.is_relevant is not None:
            base_dict["is_relevant"] = self.is_relevant
        if self.summary is not None:
            base_dict["summary"] = self.summary
        if self.matched_topics:
            base_dict["matched_topics"] = self.matched_topics
        if self.classification_reason is not None:
            base_dict["classification_reason"] = self.classification_reason

        # Add enhanced flag to indicate this paper has LLM classification
        base_dict["enhanced"] = True

        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedPaper":
        """Create EnhancedPaper from dictionary."""
        # Extract LLM classification fields (with defaults)
        relevance_score = data.get("relevance_score")
        is_relevant = data.get("is_relevant")
        summary = data.get("summary")
        matched_topics = data.get("matched_topics", [])
        classification_reason = data.get("classification_reason")

        # Create base Paper object using parent's from_dict
        # We need to handle publication_date conversion separately
        publication_date = None
        if "publication_date" in data and data["publication_date"]:
            publication_date = datetime.fromisoformat(data["publication_date"])

        # Create EnhancedPaper instance
        paper = cls(
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
            relevance_score=relevance_score,
            is_relevant=is_relevant,
            summary=summary,
            matched_topics=matched_topics,
            classification_reason=classification_reason,
        )

        return paper

    @classmethod
    def from_paper(cls, paper: Paper, **kwargs) -> "EnhancedPaper":
        """Create EnhancedPaper from existing Paper object."""
        # Convert Paper to dictionary, then add LLM fields
        data = paper.to_dict()
        data.update(kwargs)
        return cls.from_dict(data)

    def is_classified(self) -> bool:
        """Check if this paper has been classified by LLM."""
        return (
            self.relevance_score is not None
            or self.is_relevant is not None
            or self.summary is not None
            or self.matched_topics
            or self.classification_reason is not None
        )
