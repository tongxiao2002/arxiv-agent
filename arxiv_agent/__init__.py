"""Arxiv-Agent: Automated paper discovery system."""

__version__ = "0.1.0"
__author__ = "Tongxiao <tongxiao@example.com>"
__license__ = "MIT"

# Agents
from arxiv_agent.agents.scraper_agent import ScraperAgent

# Public API exports
# Import key classes for easy access
from arxiv_agent.config import Config
from arxiv_agent.sources.arxiv_source import ArxivSource

# Sources
from arxiv_agent.sources.base_source import BaseSource, Paper, SourceError
from arxiv_agent.sources.papers_cool_source import PapersCoolSource
from arxiv_agent.storage.archiver import Archiver

# Storage
from arxiv_agent.storage.json_storage import JsonStorage

# Utilities
from arxiv_agent.utils.retry import RetryError, retry, retry_context

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
