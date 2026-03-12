"""Classifier agent for LLM-based paper classification and summarization."""

import logging
from datetime import date
from typing import Any, Dict, List

from arxiv_agent.agents.base import BaseAgent
from arxiv_agent.config import LLMConfig
from arxiv_agent.llm import classify_paper, summarize_abstract
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.retry import retry

logger = logging.getLogger(__name__)


class ClassifierAgent(BaseAgent):
    """Classifier agent that processes papers using LLM classification and summarization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize classifier agent.

        Args:
            config: Agent configuration with llm, topics, and storage settings
        """
        super().__init__(name="classifier", config=config)
        self.storage = None
        self.llm_config = None
        self.topics = []
        self.timezone = config.get("agent", {}).get("timezone", "Asia/Shanghai")
        self._setup_llm()
        self._setup_storage()
        self._setup_topics()

    def _setup_llm(self) -> None:
        """Set up LLM configuration."""
        llm_config_dict = self.config.get("llm", {})
        self.llm_config = LLMConfig(**llm_config_dict)
        logger.info(f"Configured LLM provider: {self.llm_config.provider}, model: {self.llm_config.model}")

    def _setup_storage(self) -> None:
        """Set up JSON storage."""
        storage_config = self.config.get("storage", {})
        data_dir = storage_config.get("data_dir", "./papers")
        self.storage = JsonStorage(data_dir=data_dir)
        logger.info(f"Configured classifier agent storage at {data_dir}")

    def _setup_topics(self) -> None:
        """Set up research topics."""
        self.topics = self.config.get("topics", [])
        if not self.topics:
            logger.warning("No topics configured for classification")
        else:
            logger.info(f"Configured {len(self.topics)} topics for classification")

    @retry(max_retries=3, backoff_factor=2.0, jitter=True)
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Run classifier agent to classify and summarize papers.

        Args:
            *args: Optional date argument (date object or date string)
            **kwargs: May contain 'target_date' key

        Returns:
            Dictionary with execution results:
                - success: boolean
                - papers_processed: number of papers processed
                - papers_classified: number successfully classified
                - papers_summarized: number successfully summarized
                - enhanced_papers: number of enhanced papers saved
                - errors: list of error messages
        """
        logger.info("Starting classifier agent run")

        # Determine target date (default: today)
        target_date = self._parse_target_date(*args, **kwargs)
        logger.info(f"Processing papers for date: {target_date}")

        # Load papers for target date
        papers = self.storage.load_papers(target_date)
        if not papers:
            logger.warning(f"No papers found for date {target_date}")
            return {
                "success": True,
                "papers_processed": 0,
                "papers_classified": 0,
                "papers_summarized": 0,
                "enhanced_papers": 0,
                "errors": [],
                "message": "No papers to process",
            }

        logger.info(f"Loaded {len(papers)} papers for classification")

        # Filter out papers that are already enhanced (optional)
        papers_to_process = self._filter_non_enhanced_papers(papers)
        if len(papers_to_process) != len(papers):
            logger.info(f"Skipping {len(papers) - len(papers_to_process)} already enhanced papers")

        if not papers_to_process:
            logger.info("All papers already enhanced")
            return {
                "success": True,
                "papers_processed": 0,
                "papers_classified": 0,
                "papers_summarized": 0,
                "enhanced_papers": 0,
                "errors": [],
                "message": "All papers already enhanced",
            }

        # Process papers
        results = self._process_papers(papers_to_process, target_date)

        logger.info(
            f"Classifier agent completed: "
            f"{results['papers_classified']}/{len(papers_to_process)} classified, "
            f"{results['papers_summarized']}/{len(papers_to_process)} summarized"
        )

        return results

    def _parse_target_date(self, *args: Any, **kwargs: Any) -> date:
        """Parse target date from args or kwargs."""
        if args and isinstance(args[0], str):
            return date.fromisoformat(args[0])
        if args and hasattr(args[0], "isoformat"):
            return args[0]
        if "target_date" in kwargs:
            target_date = kwargs["target_date"]
            if isinstance(target_date, str):
                return date.fromisoformat(target_date)
            if target_date is not None and hasattr(target_date, "isoformat"):
                return target_date

        return date.today()

    def _filter_non_enhanced_papers(self, papers: List[Paper]) -> List[Paper]:
        """Filter out papers that are already enhanced."""
        # For now, we process all papers. Enhanced detection will be added
        # when storage supports enhanced flag.
        return papers

    def _process_papers(
        self,
        papers: List[Paper],
        target_date: date,
    ) -> Dict[str, Any]:
        """Process papers through classification and summarization."""
        enhanced_papers = []
        errors = []
        classified_count = 0
        summarized_count = 0

        for i, paper in enumerate(papers):
            paper_id = paper.arxiv_id or paper.paper_id or f"paper-{i}"
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.title[:100]}...")

            try:
                # Classify paper
                classification_result = classify_paper(paper, self.llm_config, self.topics)
                classified_count += 1

                # Summarize abstract
                summary = summarize_abstract(paper, self.llm_config)
                summarized_count += 1

                # Create enhanced paper
                enhanced_paper = EnhancedPaper.from_paper(
                    paper,
                    relevance_score=classification_result["relevance_score"],
                    is_relevant=classification_result["is_relevant"],
                    matched_topics=classification_result["matched_topics"],
                    classification_reason=classification_result["classification_reason"],
                    summary=summary,
                )
                enhanced_papers.append(enhanced_paper)

                logger.debug(
                    f"Paper '{paper.title[:50]}...' classified: "
                    f"relevant={classification_result['is_relevant']}, "
                    f"score={classification_result['relevance_score']:.2f}"
                )

            except Exception as e:
                error_msg = f"Failed to process paper {paper_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with next paper
                continue

        # Save enhanced papers
        enhanced_saved = 0
        if enhanced_papers:
            success = self.storage.save_papers(target_date, enhanced_papers)
            if success:
                enhanced_saved = len(enhanced_papers)
                logger.info(f"Saved {enhanced_saved} enhanced papers")
            else:
                errors.append("Failed to save enhanced papers to storage")

        return {
            "success": len(errors) == 0,
            "papers_processed": len(papers),
            "papers_classified": classified_count,
            "papers_summarized": summarized_count,
            "enhanced_papers": enhanced_saved,
            "errors": errors,
            "target_date": target_date.isoformat(),
        }

    def validate(self) -> bool:
        """
        Validate classifier agent configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Validating classifier agent configuration")

        # Check required configuration sections
        required_sections = ["llm", "topics", "storage"]
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing configuration section: {section}")
                return False

        # Validate LLM configuration
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        if provider not in ["openai", "anthropic", "local"]:
            logger.error(f"Invalid LLM provider: {provider}")
            return False

        # Check for required API keys (only if provider is not local)
        if provider in ["openai", "anthropic"]:
            import os
            if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY environment variable not set")
                return False
            if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
                logger.error("ANTHROPIC_API_KEY environment variable not set")
                return False

        # Validate topics
        topics = self.config.get("topics", [])
        if not topics:
            logger.error("No topics configured for classification")
            return False
        if not isinstance(topics, list):
            logger.error("Topics must be a list")
            return False

        # Validate storage configuration
        storage_config = self.config.get("storage", {})
        if "data_dir" not in storage_config:
            logger.error("Storage configuration missing 'data_dir'")
            return False

        logger.info("Classifier agent configuration validated successfully")
        return True

    def _setup(self) -> None:
        """Internal setup method."""
        # Additional setup if needed
        pass

    def _teardown(self) -> None:
        """Internal teardown method."""
        # Cleanup if needed
        pass
