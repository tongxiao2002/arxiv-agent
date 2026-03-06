"""Papers.cool source implementation (web scraping)."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from arxiv_agent.sources.base_source import BaseSource, Paper
from arxiv_agent.utils.retry import retry

logger = logging.getLogger(__name__)

# Papers.cool constants
PAPERS_COOL_BASE_URL = "https://papers.cool"
PAPERS_COOL_CATEGORIES = {
    "cs.ai": "Artificial Intelligence",
    "cs.lg": "Machine Learning",
    "cs.cv": "Computer Vision",
    "cs.cl": "Computational Linguistics",
    "cs.ne": "Neural and Evolutionary Computing",
    "cs.ro": "Robotics",
}


class PapersCoolSource(BaseSource):
    """Papers.cool paper source implementation (web scraping)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Papers.cool source.

        Args:
            config: Papers.cool source configuration (categories, max_papers, etc.)
        """
        super().__init__(config, source_name="papers_cool")
        self.categories = config.get("categories", ["cs.ai", "cs.lg"])
        self.max_papers = config.get("max_papers", 50)
        self._validate_categories()

    def _validate_categories(self) -> None:
        """Validate Papers.cool categories against known categories."""
        valid_categories = set(PAPERS_COOL_CATEGORIES.keys())
        invalid = [cat for cat in self.categories if cat not in valid_categories]
        if invalid:
            logger.warning(
                f"Invalid Papers.cool categories: {invalid}. "
                f"Valid categories: {list(valid_categories)}"
            )
            # Remove invalid categories
            self.categories = [cat for cat in self.categories if cat not in invalid]

    @retry(max_retries=3, backoff_factor=2.0, jitter=True)
    def _fetch_category_page(self, category: str, page: int = 1) -> Optional[str]:
        """
        Fetch Papers.cool category page HTML.

        Args:
            category: Papers.cool category (e.g., "cs.ai")
            page: Page number

        Returns:
            HTML content as string, or None if request fails
        """
        try:
            # Construct URL for category (this is a placeholder - actual URL may differ)
            url = f"{PAPERS_COOL_BASE_URL}/arxiv/{category}"
            if page > 1:
                url = f"{url}?page={page}"

            logger.info(f"Fetching Papers.cool page for category '{category}'")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Respect robots.txt and rate limiting
            delay = 2  # Be respectful with delays between requests
            logger.debug(f"Waiting {delay} seconds before next request")
            import time

            time.sleep(delay)

            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Papers.cool page: {e}")
            return None

    def _parse_html_page(self, html_content: str, category: str) -> List[Paper]:
        """
        Parse Papers.cool HTML page into Paper objects.

        Args:
            html_content: HTML content string
            category: Source category for tagging papers

        Returns:
            List of Paper objects
        """
        papers = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception as e:
            logger.error(f"Failed to parse Papers.cool HTML: {e}")
            return papers

        # This is a placeholder implementation
        # Actual implementation would need to inspect Papers.cool HTML structure
        # and extract paper titles, abstracts, authors, etc.

        # For now, we'll return an empty list and log a warning
        logger.warning(
            "Papers.cool HTML parsing not fully implemented. "
            "Returning empty paper list."
        )

        # Example placeholder - would need actual parsing logic
        # paper_elements = soup.find_all("div", class_="paper-item")
        # for elem in paper_elements:
        #     paper = self._parse_paper_element(elem, category)
        #     if paper:
        #         papers.append(paper)

        return papers

    def _parse_paper_element(
        self, paper_elem: BeautifulSoup, category: str
    ) -> Optional[Paper]:
        """
        Parse a single paper element into a Paper object.

        Args:
            paper_elem: BeautifulSoup element containing paper information
            category: Source category for tagging papers

        Returns:
            Paper object or None if parsing fails
        """
        # Placeholder implementation
        # Actual implementation would extract:
        # - Title from <h2 class="paper-title"> or similar
        # - Abstract from <div class="abstract"> or similar
        # - Authors from <span class="authors"> or similar
        # - arXiv ID from link or text
        # - Publication date if available
        # - PDF and webpage URLs

        try:
            # Example extraction (adjust based on actual HTML structure)
            title_elem = paper_elem.find("h2", class_="paper-title")
            abstract_elem = paper_elem.find("div", class_="abstract")
            authors_elem = paper_elem.find("span", class_="authors")
            link_elem = paper_elem.find("a", href=re.compile(r"arxiv\.org/abs/\d+\.\d+"))

            if not title_elem or not abstract_elem:
                return None

            title = title_elem.get_text(strip=True)
            abstract = abstract_elem.get_text(strip=True)
            authors = []
            if authors_elem:
                authors = [
                    author.strip()
                    for author in authors_elem.get_text(strip=True).split(",")
                ]

            arxiv_id = None
            pdf_url = None
            webpage_url = None
            if link_elem:
                link_href = link_elem.get("href", "")
                arxiv_id_match = re.search(r"arxiv\.org/abs/(\d+\.\d+)(v\d+)?", link_href)
                if arxiv_id_match:
                    arxiv_id = arxiv_id_match.group(1)
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    webpage_url = f"https://arxiv.org/abs/{arxiv_id}"

            return Paper(
                title=title,
                abstract=abstract,
                authors=authors,
                arxiv_id=arxiv_id,
                paper_id=arxiv_id or "",
                publication_date=datetime.now(),
                categories=[category],
                source="papers_cool",
                pdf_url=pdf_url,
                webpage_url=webpage_url,
            )
        except Exception as e:
            logger.debug(f"Error parsing Papers.cool paper element: {e}")
            return None

    def fetch_papers(self, max_papers: Optional[int] = None) -> List[Paper]:
        """
        Fetch papers from Papers.cool.

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
                logger.info(f"Fetching Papers.cool papers for category '{category}'")
                html_content = self._fetch_category_page(category)
                if html_content is None:
                    continue

                papers = self._parse_html_page(html_content, category)
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers in category '{category}'")

                # Stop if we've reached the limit
                if len(all_papers) >= max_papers:
                    all_papers = all_papers[:max_papers]
                    break

            except Exception as e:
                logger.error(f"Failed to fetch papers for category '{category}': {e}")
                continue

        logger.info(f"Total Papers.cool papers fetched: {len(all_papers)}")
        return all_papers

    def get_source_name(self) -> str:
        """Get source name."""
        return self.source_name

    def validate_config(self) -> bool:
        """Validate Papers.cool source configuration."""
        if not isinstance(self.categories, list):
            logger.error("Papers.cool categories must be a list")
            return False

        if len(self.categories) == 0:
            logger.error("At least one Papers.cool category must be specified")
            return False

        if self.max_papers <= 0:
            logger.error("max_papers must be positive")
            return False

        return True