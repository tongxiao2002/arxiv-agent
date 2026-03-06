"""Arxiv-Agent: Automated paper discovery system."""

__version__ = "0.1.0"
__author__ = "Tongxiao <tongxiao@example.com>"
__license__ = "MIT"

# Public API exports
# Import key classes for easy access
from arxiv_agent.config import Config

# Sources
from arxiv_agent.sources.base_source import Paper, BaseSource, SourceError
from arxiv_agent.sources.arxiv_source import ArxivSource
from arxiv_agent.sources.papers_cool_source import PapersCoolSource

# Storage
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.storage.archiver import Archiver

# Agents
from arxiv_agent.agents.scraper_agent import ScraperAgent

# Utilities
from arxiv_agent.utils.retry import retry, retry_context, RetryError

__all__ = [
    "Config",
    "Paper",
    "BaseSource",
    "SourceError",
    "ArxivSource",
    "PapersCoolSource",
    "JsonStorage",
    "Archiver",
    "ScraperAgent",
    "retry",
    "retry_context",
    "RetryError",
]
