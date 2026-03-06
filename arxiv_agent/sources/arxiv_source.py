"""arXiv.org API source implementation."""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from arxiv_agent.sources.base_source import BaseSource, Paper
from arxiv_agent.utils.retry import retry

logger = logging.getLogger(__name__)

# arXiv API constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"
ARXIV_CATEGORIES = {
    "cs": "Computer Science",
    "physics": "Physics",
    "math": "Mathematics",
    "stat": "Statistics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "eess": "Electrical Engineering and Systems Science",
}


class ArxivSource(BaseSource):
    """arXiv.org paper source implementation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize arXiv source.

        Args:
            config: arXiv source configuration (categories, max_papers, etc.)
        """
        super().__init__(config, source_name="arxiv")
        self.categories = config.get("categories", ["cs", "physics", "math"])
        self.max_papers = config.get("max_papers", 100)
        self._validate_categories()

    def _validate_categories(self) -> None:
        """Validate arXiv categories against known categories."""
        valid_categories = set(ARXIV_CATEGORIES.keys())
        invalid = [cat for cat in self.categories if cat not in valid_categories]
        if invalid:
            logger.warning(
                f"Invalid arXiv categories: {invalid}. "
                f"Valid categories: {list(valid_categories)}"
            )
            # Remove invalid categories
            self.categories = [cat for cat in self.categories if cat not in invalid]

    @retry(max_retries=3, backoff_factor=2.0, jitter=True)
    def _fetch_arxiv_feed(self, category: str, max_results: int) -> str:
        """
        Fetch arXiv Atom feed for a category.

        Args:
            category: arXiv category (e.g., "cs")
            max_results: Maximum number of results to fetch

        Returns:
            Atom feed XML as string
        """
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
            "start": 0,
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"

        logger.info(f"Fetching arXiv feed for category '{category}'")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Check rate limiting
        if "Retry-After" in response.headers:
            retry_after = int(response.headers["Retry-After"])
            logger.warning(f"arXiv API rate limited. Retry after {retry_after} seconds")
            # The retry decorator will handle this via the exception

        return response.text

    def _parse_atom_feed(self, xml_content: str) -> List[Paper]:
        """
        Parse arXiv Atom feed XML into Paper objects.

        Args:
            xml_content: Atom feed XML string

        Returns:
            List of Paper objects
        """
        papers = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv Atom feed: {e}")
            return papers

        # Find all entry elements
        for entry in root.findall(f"{ARXIV_NAMESPACE}entry"):
            try:
                paper = self._parse_atom_entry(entry)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {e}")
                continue

        return papers

    def _parse_atom_entry(self, entry: ET.Element) -> Optional[Paper]:
        """
        Parse a single Atom entry element into a Paper object.

        Args:
            entry: Atom entry element

        Returns:
            Paper object or None if parsing fails
        """
        try:
            # Extract arXiv ID
            arxiv_id_elem = entry.find(f"{ARXIV_NAMESPACE}id")
            if arxiv_id_elem is None or arxiv_id_elem.text is None:
                return None

            # arXiv ID format: http://arxiv.org/abs/2101.12345v2
            arxiv_url = arxiv_id_elem.text
            arxiv_id_match = re.search(r"arxiv\.org/abs/(\d+\.\d+)(v\d+)?", arxiv_url)
            if not arxiv_id_match:
                return None
            arxiv_id = arxiv_id_match.group(1)

            # Title
            title_elem = entry.find(f"{ARXIV_NAMESPACE}title")
            title = title_elem.text.strip() if title_elem is not None else ""

            # Abstract
            summary_elem = entry.find(f"{ARXIV_NAMESPACE}summary")
            abstract = summary_elem.text.strip() if summary_elem is not None else ""

            # Authors
            authors = []
            for author_elem in entry.findall(f"{ARXIV_NAMESPACE}author"):
                name_elem = author_elem.find(f"{ARXIV_NAMESPACE}name")
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Publication date
            published_elem = entry.find(f"{ARXIV_NAMESPACE}published")
            publication_date = None
            if published_elem is not None and published_elem.text:
                try:
                    publication_date = datetime.fromisoformat(
                        published_elem.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # Categories
            categories = []
            for category_elem in entry.findall(f"{ARXIV_NAMESPACE}category"):
                term = category_elem.get("term")
                if term:
                    categories.append(term)

            # Links
            pdf_url = None
            webpage_url = None
            for link_elem in entry.findall(f"{ARXIV_NAMESPACE}link"):
                link_type = link_elem.get("type")
                link_href = link_elem.get("href")
                if link_type == "application/pdf":
                    pdf_url = link_href
                elif link_type == "text/html":
                    webpage_url = link_href

            return Paper(
                title=title,
                abstract=abstract,
                authors=authors,
                arxiv_id=arxiv_id,
                paper_id=arxiv_id,
                publication_date=publication_date,
                categories=categories,
                source="arxiv",
                pdf_url=pdf_url,
                webpage_url=webpage_url,
            )
        except Exception as e:
            logger.debug(f"Error parsing arXiv entry: {e}")
            return None

    def fetch_papers(self, max_papers: Optional[int] = None) -> List[Paper]:
        """
        Fetch papers from arXiv.

        Args:
            max_papers: Maximum number of papers to fetch (None for no limit)

        Returns:
            List of Paper objects
        """
        if max_papers is None:
            max_papers = self.max_papers

        all_papers = []
        papers_per_category = max(1, max_papers // len(self.categories))

        for category in self.categories:
            try:
                logger.info(f"Fetching arXiv papers for category '{category}'")
                feed_xml = self._fetch_arxiv_feed(category, papers_per_category)
                papers = self._parse_atom_feed(feed_xml)
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers in category '{category}'")

                # Stop if we've reached the limit
                if len(all_papers) >= max_papers:
                    all_papers = all_papers[:max_papers]
                    break

            except Exception as e:
                logger.error(f"Failed to fetch papers for category '{category}': {e}")
                continue

        logger.info(f"Total arXiv papers fetched: {len(all_papers)}")
        return all_papers

    def get_source_name(self) -> str:
        """Get source name."""
        return self.source_name

    def validate_config(self) -> bool:
        """Validate arXiv source configuration."""
        if not isinstance(self.categories, list):
            logger.error("arXiv categories must be a list")
            return False

        if len(self.categories) == 0:
            logger.error("At least one arXiv category must be specified")
            return False

        if self.max_papers <= 0:
            logger.error("max_papers must be positive")
            return False

        return True