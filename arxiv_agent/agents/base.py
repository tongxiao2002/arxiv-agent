"""Base agent class for Arxiv-Agent."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base agent class for Arxiv-Agent framework."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize agent.

        Args:
            name: Agent name
            config: Agent configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"agent.{name}")
        self._initialized = False

    def initialize(self) -> None:
        """Initialize agent resources."""
        if self._initialized:
            self.logger.warning(f"Agent {self.name} already initialized")
            return

        self.logger.info(f"Initializing agent {self.name}")
        self._setup()
        self._initialized = True
        self.logger.info(f"Agent {self.name} initialized successfully")

    def cleanup(self) -> None:
        """Clean up agent resources."""
        if not self._initialized:
            self.logger.warning(f"Agent {self.name} not initialized")
            return

        self.logger.info(f"Cleaning up agent {self.name}")
        self._teardown()
        self._initialized = False
        self.logger.info(f"Agent {self.name} cleaned up")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute agent's main logic.

        Returns:
            Agent execution result
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate agent configuration and state.

        Returns:
            True if validation passes, False otherwise
        """
        pass

    def _setup(self) -> None:
        """Internal setup method for resource initialization."""
        # Override in subclasses if needed
        pass

    def _teardown(self) -> None:
        """Internal teardown method for resource cleanup."""
        # Override in subclasses if needed
        pass

    def __enter__(self) -> "BaseAgent":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, agent_name: Optional[str] = None):
        self.agent_name = agent_name
        self.message = message
        super().__init__(f"Agent {agent_name}: {message}" if agent_name else message)


class AgentConfigurationError(AgentError):
    """Exception for agent configuration errors."""

    pass


class AgentExecutionError(AgentError):
    """Exception for agent execution errors."""

    pass


# DeepAgents integration (optional)
try:
    from deepagents import Agent as DeepAgent  # type: ignore

    class DeepAgentMixin(DeepAgent):
        """Mixin for DeepAgents integration."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Get name and config from kwargs or args
            super().__init__(*args, **kwargs)

        def deepagents_run(self, *args: Any, **kwargs: Any) -> Any:
            """Delegate to DeepAgents run method."""
            return super().run(*args, **kwargs)

    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False
    DeepAgentMixin = object  # type: ignore


# Example concrete agent for testing
class ExampleAgent(BaseAgent):
    """Example agent for demonstration."""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Example run method."""
        self.logger.info("Example agent running")
        return "Example result"

    def validate(self) -> bool:
        """Example validation."""
        self.logger.info("Example agent validation")
        return True
