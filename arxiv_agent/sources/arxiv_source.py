"""arXiv.org API source implementation."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from arxiv_agent.sources.base_source import BaseSource, Paper
from arxiv_agent.utils.intervals import RunOnceInterval
from arxiv_agent.utils.runtime import RuntimeOptions, call_with_retry

logger = logging.getLogger(__name__)

# arXiv API constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"
ARXIV_PAGE_SIZE = 100
ARXIV_REQUEST_PACING_SECONDS = 3


class ArxivSource(BaseSource):
    """arXiv.org paper source implementation."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        runtime_options: Optional[RuntimeOptions] = None,
    ):
        """
        Initialize arXiv source.

        Args:
            config: arXiv source configuration (categories, max_papers, etc.)
        """
        super().__init__(config, source_name="arxiv")
        self.categories = self._normalize_categories(
            config.get("categories", ["cs.LG", "cs.CV"])
        )
        self.max_papers = config.get("max_papers", -1)
        self.lookback_days = config.get("lookback_days", 1)
        self.runtime_options = runtime_options or RuntimeOptions()

    def _normalize_categories(self, categories: Any) -> Any:
        """Normalize categories and drop invalid identifiers."""
        if not isinstance(categories, list):
            return categories

        normalized = []
        for category in categories:
            if not isinstance(category, str):
                continue
            value = category.strip()
            if re.fullmatch(r"[A-Za-z]+(?:\.[A-Za-z\-]+)?", value):
                normalized.append(value)
        return normalized

    def _get_submission_window(
        self,
        today: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        """Return the [start, end) submission window for arXiv queries."""
        window_end = (today or datetime.now()).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        window_start = window_end - timedelta(days=self.lookback_days)
        return window_start, window_end

    def _fetch_arxiv_feed(
        self,
        category: str,
        max_results: int,
        today: Optional[datetime] = None,
    ) -> str:
        """
        Fetch arXiv Atom feed for a category.

        Args:
            category: arXiv category (e.g., "cs")
            max_results: Maximum number of results to fetch
            today: Datetime representing the exclusive end of the query window

        Returns:
            Atom feed XML as string
        """
        window_start, window_end = self._get_submission_window(today)
        return self._fetch_arxiv_feed_page(
            category,
            start=0,
            max_results=max_results,
            window_start=window_start,
            window_end=window_end,
        )

    def _resolve_per_category_limit(self, max_papers: Optional[int]) -> Optional[int]:
        """Return a per-category cap, using None to represent unlimited paging."""
        limit = self.max_papers if max_papers is None else max_papers
        return None if limit == -1 else limit

    def _sleep_between_page_requests(self) -> None:
        """Pause between sequential arXiv page requests per API guidance."""
        logger.info(
            "Sleeping %s seconds before the next arXiv page request",
            ARXIV_REQUEST_PACING_SECONDS,
        )
        time.sleep(ARXIV_REQUEST_PACING_SECONDS)

    def _fetch_category_papers(
        self,
        category: str,
        *,
        window_start: datetime,
        window_end: datetime,
        max_papers: Optional[int],
    ) -> List[Paper]:
        """Fetch and page all papers for one category inside an explicit query window."""
        category_papers: List[Paper] = []
        page_start = 0
        per_category_limit = self._resolve_per_category_limit(max_papers)

        while True:
            remaining = None
            if per_category_limit is not None:
                remaining = per_category_limit - len(category_papers)
                if remaining <= 0:
                    break

            page_size = (
                min(ARXIV_PAGE_SIZE, remaining)
                if remaining is not None
                else ARXIV_PAGE_SIZE
            )

            if page_start > 0:
                self._sleep_between_page_requests()

            feed_xml = self._fetch_arxiv_feed_page(
                category,
                start=page_start,
                max_results=page_size,
                window_start=window_start,
                window_end=window_end,
            )
            page_papers = self._parse_atom_feed(feed_xml)
            if not page_papers:
                break

            category_papers.extend(page_papers)
            if len(page_papers) < page_size:
                break

            page_start += page_size

        logger.info(
            "Fetched %s papers for category '%s' before final deduplication",
            len(category_papers),
            category,
        )
        return category_papers

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicates across categories and pages using a stable identity."""
        unique_papers: List[Paper] = []
        seen_ids = set()
        for paper in papers:
            identity = paper.arxiv_id or paper.paper_id or paper.title
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            unique_papers.append(paper)
        return unique_papers

    def _fetch_arxiv_feed_page(
        self,
        category: str,
        *,
        start: int,
        max_results: int,
        window_start: datetime,
        window_end: datetime,
    ) -> str:
        """Fetch a single arXiv Atom feed page for an explicit GMT window."""
        params = {
            "search_query": (
                "cat:"
                f"{category}+AND+submittedDate:[{window_start.strftime('%Y%m%d%H%M')}"
                f"+TO+{window_end.strftime('%Y%m%d%H%M')}]"
            ),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": max_results,
        }

        params_str = "&".join([f"{key}={val}" for key, val in params.items()])
        url = f"{ARXIV_API_URL}?{params_str}"
        logger.info(
            "Fetching arXiv feed for category '%s' between %s and %s (start=%s, max_results=%s)",
            category,
            window_start.isoformat(),
            window_end.isoformat(),
            start,
            max_results,
        )

        def operation() -> str:
            response = requests.get(url, timeout=self.runtime_options.request_timeout)
            response.raise_for_status()

            if "Retry-After" in response.headers:
                retry_after = int(response.headers["Retry-After"])
                logger.warning(
                    "arXiv API rate limited. Retry after %s seconds",
                    retry_after,
                )

            return response.text

        return call_with_retry(
            operation,
            operation_name="_fetch_arxiv_feed",
            max_retries=self.runtime_options.max_retries,
            backoff_factor=self.runtime_options.retry_backoff_factor,
            retry_on=requests.RequestException,
        )

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

    def fetch_papers(
        self,
        max_papers: Optional[int] = None,
        today: Optional[datetime] = None,
    ) -> List[Paper]:
        """
        Fetch papers from arXiv.

        Args:
            max_papers: Maximum number of papers to fetch (None for no limit)
            today: Datetime representing the exclusive end of the query window

        Returns:
            List of Paper objects
        """
        effective_max_papers = self.max_papers if max_papers is None else max_papers
        window_start, window_end = self._get_submission_window(today)
        all_papers: List[Paper] = []

        for category in self.categories:
            try:
                logger.info(f"Fetching arXiv papers for category '{category}'")
                papers = self._fetch_category_papers(
                    category,
                    window_start=window_start,
                    window_end=window_end,
                    max_papers=effective_max_papers,
                )
                all_papers.extend(papers)

            except Exception as e:
                logger.error(f"Failed to fetch papers for category '{category}': {e}")
                continue

        papers = self._deduplicate_papers(all_papers)

        logger.info(f"Total arXiv papers fetched: {len(papers)}")
        return papers

    def fetch_papers_for_interval(
        self,
        interval: RunOnceInterval,
        *,
        max_papers: Optional[int] = None,
    ) -> List[Paper]:
        """
        Fetch papers for an explicit local datetime interval.

        The arXiv API query is widened to minute precision in GMT and then strict
        filtering is applied against the original closed local interval.
        """
        gmt_start, gmt_end = interval.gmt_bounds()
        logger.info(
            "Fetching arXiv interval: local=[%s, %s] timezone=%s gmt=[%s, %s]",
            interval.local_start.isoformat(),
            interval.local_end.isoformat(),
            interval.timezone_name,
            gmt_start,
            gmt_end,
        )

        effective_max_papers = self.max_papers if max_papers is None else max_papers
        all_papers: List[Paper] = []

        for category in self.categories:
            try:
                category_papers = self._fetch_category_papers(
                    category,
                    window_start=interval.query_start_utc,
                    window_end=interval.query_end_utc,
                    max_papers=effective_max_papers,
                )
                all_papers.extend(category_papers)
            except Exception as exc:
                logger.error(
                    "Failed to fetch interval papers for category '%s': %s",
                    category,
                    exc,
                )
                continue

        unique_papers = self._deduplicate_papers(all_papers)

        retained_papers = [
            paper
            for paper in unique_papers
            if interval.contains(paper.publication_date)
        ]
        logger.info(
            "Interval fetch retained %s/%s papers after strict local filtering",
            len(retained_papers),
            len(unique_papers),
        )
        return retained_papers

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

        if not isinstance(self.max_papers, int):
            logger.error("max_papers must be an integer")
            return False

        if self.max_papers == 0 or self.max_papers < -1:
            logger.error("max_papers must be -1 or a positive integer")
            return False

        if not isinstance(self.lookback_days, int) or self.lookback_days <= 0:
            logger.error("lookback_days must be a positive integer")
            return False

        return True
