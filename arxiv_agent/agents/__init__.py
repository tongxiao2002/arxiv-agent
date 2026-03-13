"""Agents module for Arxiv-Agent."""

from .base import (
    DEEPAGENTS_AVAILABLE,
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    BaseAgent,
    DeepAgentMixin,
    ExampleAgent,
)
from .classifier_agent import ClassifierAgent
from .emailer_agent import EmailerAgent
from .scraper_agent import ScraperAgent
from .supervisor import SupervisorAgent

__all__ = [
    # Base classes and exceptions
    "BaseAgent",
    "AgentError",
    "AgentConfigurationError",
    "AgentExecutionError",
    "DeepAgentMixin",
    "DEEPAGENTS_AVAILABLE",
    # Concrete agents
    "ExampleAgent",
    "ScraperAgent",
    "ClassifierAgent",
    "EmailerAgent",
    "SupervisorAgent",
]
