"""Classifier agent for LLM-based paper classification and summarization."""

import asyncio
import logging
import threading
from datetime import date
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar

from arxiv_agent.agents.base import BaseAgent
from arxiv_agent.config import AdvancedConfig, LLMConfig
from arxiv_agent.llm.classifier import aclassify_paper
from arxiv_agent.llm.provider_utils import get_provider_api_key, get_provider_env_var
from arxiv_agent.llm.summarizer import asummarize_abstract
from arxiv_agent.sources.base_source import Paper
from arxiv_agent.sources.enhanced_paper import EnhancedPaper
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.retry import RetryError, retry
from arxiv_agent.utils.runtime import RuntimeOptions, describe_retry_error
from arxiv_agent.utils.timezone import get_current_date_in_timezone

logger = logging.getLogger(__name__)
T = TypeVar("T")


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
        self.advanced_config = AdvancedConfig(**config.get("advanced", {}))
        self.runtime_options = RuntimeOptions.from_mapping(config.get("advanced", {}))
        self._setup_llm()
        self._setup_storage()
        self._setup_topics()

    def _setup_llm(self) -> None:
        """Set up LLM configuration."""
        llm_config_dict = self.config.get("llm", {})
        self.llm_config = LLMConfig(**llm_config_dict)
        logger.info(
            "Configured LLM provider: %s, model: %s",
            self.llm_config.provider,
            self.llm_config.model,
        )

    def _setup_storage(self) -> None:
        """Set up JSON storage."""
        storage_config = self.config.get("storage", {})
        data_dir = storage_config.get("data_dir", "./papers")
        self.storage = JsonStorage(data_dir=data_dir)
        logger.info("Configured classifier agent storage at %s", data_dir)

    def _setup_topics(self) -> None:
        """Set up research topics."""
        self.topics = self.config.get("topics", [])
        if not self.topics:
            logger.warning("No topics configured for classification")
        else:
            logger.info("Configured %s topics for classification", len(self.topics))

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
        logger.info("Processing papers for date: %s", target_date)

        # Load papers for target date
        papers = self.storage.load_papers(target_date)
        if not papers:
            logger.warning("No papers found for date %s", target_date)
            return {
                "success": True,
                "papers_processed": 0,
                "papers_classified": 0,
                "papers_summarized": 0,
                "enhanced_papers": 0,
                "errors": [],
                "message": "No papers to process",
            }

        logger.info("Loaded %s papers for classification", len(papers))

        already_enhanced, papers_to_process = self._split_enhanced_papers(papers)
        if already_enhanced:
            logger.info("Skipping %s already enhanced papers", len(already_enhanced))

        if not papers_to_process:
            logger.info("All papers already enhanced")
            return {
                "success": True,
                "papers_processed": 0,
                "papers_classified": 0,
                "papers_summarized": 0,
                "enhanced_papers": 0,
                "papers_skipped": len(already_enhanced),
                "errors": [],
                "message": "All papers already enhanced",
            }

        # Process papers
        results = self._process_papers(
            papers_to_process,
            target_date,
            already_enhanced=already_enhanced,
        )

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

        return get_current_date_in_timezone(self.timezone)

    def _split_enhanced_papers(
        self, papers: List[Paper]
    ) -> Tuple[List[EnhancedPaper], List[Paper]]:
        """Split stored papers into already-enhanced and pending items."""
        enhanced = [paper for paper in papers if isinstance(paper, EnhancedPaper)]
        pending = [paper for paper in papers if not isinstance(paper, EnhancedPaper)]
        return enhanced, pending

    def _run_coroutine(self, coro: Coroutine[object, object, T]) -> T:
        """Run async processing from the synchronous agent entrypoint."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: list[T] = []
        error: list[BaseException] = []

        def _runner() -> None:
            try:
                result.append(asyncio.run(coro))
            except BaseException as exc:  # pragma: no cover - passthrough
                error.append(exc)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if error:
            raise error[0]

        return result[0]

    async def _create_openai_client(self) -> Any:
        """Create a shared AsyncOpenAI client for one agent run."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ValueError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        api_key = get_provider_api_key("openai")
        if not api_key:
            env_var = get_provider_env_var("openai")
            raise ValueError(f"{env_var} environment variable not set")

        return AsyncOpenAI(api_key=api_key, base_url=self.llm_config.base_url)

    def _process_papers(
        self,
        papers: List[Paper],
        target_date: date,
        *,
        already_enhanced: List[EnhancedPaper],
    ) -> Dict[str, Any]:
        """Process papers through classification and summarization."""
        return self._run_coroutine(
            self._process_papers_async(
                papers,
                target_date,
                already_enhanced=already_enhanced,
            )
        )

    async def _process_papers_async(
        self,
        papers: List[Paper],
        target_date: date,
        *,
        already_enhanced: List[EnhancedPaper],
    ) -> Dict[str, Any]:
        """Process papers through classification and summarization concurrently."""
        enhanced_papers = []
        errors = []
        classified_count = 0
        summarized_count = 0
        max_concurrent = min(32, max(1, len(papers)))
        shared_client: Optional[Any] = None

        if self.llm_config.provider.lower() == "openai":
            shared_client = await self._create_openai_client()

        async def _process_one(
            index: int, paper: Paper
        ) -> Tuple[Optional[EnhancedPaper], int, int, List[str]]:
            paper_id = paper.arxiv_id or paper.paper_id or f"paper-{index}"
            logger.info(
                "Processing paper %s/%s: %s...",
                index + 1,
                len(papers),
                paper.title[:100],
            )

            classification_result, summary_result = await asyncio.gather(
                aclassify_paper(
                    paper,
                    self.llm_config,
                    self.topics,
                    self.runtime_options,
                    client=shared_client,
                ),
                asummarize_abstract(
                    paper,
                    self.llm_config,
                    self.runtime_options,
                    client=shared_client,
                ),
                return_exceptions=True,
            )

            paper_errors = []
            paper_classified = 0
            paper_summarized = 0

            if isinstance(classification_result, Exception):
                paper_errors.append(
                    self._format_processing_error(paper_id, classification_result)
                )
            else:
                paper_classified = 1

            if isinstance(summary_result, Exception):
                paper_errors.append(
                    self._format_processing_error(paper_id, summary_result)
                )
            else:
                paper_summarized = 1

            if paper_errors:
                return None, paper_classified, paper_summarized, paper_errors

            enhanced_paper = EnhancedPaper.from_paper(
                paper,
                relevance_score=classification_result["relevance_score"],
                is_relevant=classification_result["is_relevant"],
                matched_topics=classification_result["matched_topics"],
                classification_reason=classification_result["classification_reason"],
                summary=summary_result,
            )
            logger.debug(
                "Paper '%s...' classified: relevant=%s, score=%.2f",
                paper.title[:50],
                classification_result["is_relevant"],
                classification_result["relevance_score"],
            )
            return enhanced_paper, paper_classified, paper_summarized, []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_with_limit(
            index: int, paper: Paper
        ) -> Tuple[Optional[EnhancedPaper], int, int, List[str]]:
            async with semaphore:
                return await _process_one(index, paper)

        try:
            results = await asyncio.gather(
                *[
                    asyncio.create_task(_process_with_limit(index, paper))
                    for index, paper in enumerate(papers)
                ]
            )
        finally:
            if shared_client is not None:
                await shared_client.close()

        for enhanced_paper, paper_classified, paper_summarized, paper_errors in results:
            classified_count += paper_classified
            summarized_count += paper_summarized
            errors.extend(paper_errors)
            if enhanced_paper is not None:
                enhanced_papers.append(enhanced_paper)

        enhanced_saved = 0
        papers_to_save = [*already_enhanced, *enhanced_papers]
        if papers_to_save:
            success = self.storage.save_papers(target_date, papers_to_save)
            if success:
                enhanced_saved = len(enhanced_papers)
                logger.info(
                    "Saved %s newly enhanced papers (%s total retained)",
                    enhanced_saved,
                    len(papers_to_save),
                )
            else:
                errors.append("Failed to save enhanced papers to storage")

        return {
            "success": len(errors) == 0,
            "papers_processed": len(papers),
            "papers_classified": classified_count,
            "papers_summarized": summarized_count,
            "enhanced_papers": enhanced_saved,
            "papers_skipped": len(already_enhanced),
            "errors": errors,
            "target_date": target_date.isoformat(),
        }

    def _format_processing_error(self, paper_id: str, exc: Exception) -> str:
        """Format processing errors while preserving retry root causes."""
        if isinstance(exc, RetryError):
            return describe_retry_error(
                exc,
                f"Failed to process paper {paper_id}",
            )
        return f"Failed to process paper {paper_id}: {exc}"

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
