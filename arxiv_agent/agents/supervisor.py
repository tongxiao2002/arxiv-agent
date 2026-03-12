"""Supervisor agent for coordinating specialized agents."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from arxiv_agent.agents.base import AgentError, AgentExecutionError, BaseAgent

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent."""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANED_UP = "cleaned_up"


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    agent: BaseAgent
    status: AgentStatus = AgentStatus.CREATED
    last_run_result: Optional[Any] = None
    error: Optional[str] = None


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates specialized agents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize supervisor agent.

        Args:
            config: Supervisor configuration
        """
        super().__init__(name="supervisor", config=config)
        self.agents: Dict[str, AgentInfo] = {}
        self.execution_order: List[str] = []  # Order in which agents should run

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the supervisor.

        Args:
            agent: Agent to register
        """
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already registered, replacing")

        self.agents[agent.name] = AgentInfo(agent=agent, status=AgentStatus.CREATED)
        logger.info(f"Registered agent: {agent.name}")

    def set_execution_order(self, order: List[str]) -> None:
        """
        Set the execution order for agents.

        Args:
            order: List of agent names in execution order
        """
        # Validate that all agents are registered
        for name in order:
            if name not in self.agents:
                raise AgentError(f"Agent {name} not registered", agent_name=name)

        self.execution_order = order
        logger.info(f"Set execution order: {order}")

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute all registered agents in specified order.

        Returns:
            Dictionary mapping agent names to their results
        """
        results = {}
        self._reset_run_state()

        if not self.execution_order:
            logger.warning(
                "No execution order set, running all agents in registration order"
            )
            execution_order = list(self.agents.keys())
        else:
            execution_order = list(self.execution_order)

        logger.info(f"Starting agent execution with order: {execution_order}")

        for agent_name in execution_order:
            agent_info = self.agents[agent_name]
            agent = agent_info.agent

            try:
                # Initialize agent if not already initialized
                if agent_info.status == AgentStatus.CREATED:
                    logger.info(f"Initializing agent {agent_name}")
                    agent.initialize()
                    agent_info.status = AgentStatus.INITIALIZED

                # Run agent
                logger.info(f"Running agent {agent_name}")
                agent_info.status = AgentStatus.RUNNING
                result = agent.run(*args, **kwargs)
                agent_info.last_run_result = result
                agent_info.status = AgentStatus.COMPLETED
                results[agent_name] = result

                logger.info(f"Agent {agent_name} completed successfully")

            except Exception as e:
                error_msg = f"Agent {agent_name} failed: {e}"
                logger.error(error_msg, exc_info=True)
                agent_info.status = AgentStatus.FAILED
                agent_info.error = str(e)
                results[agent_name] = {"error": str(e), "success": False}

                # Decide whether to continue or stop
                if self.config.get("stop_on_agent_failure", True):
                    logger.error("Stopping execution due to agent failure")
                    break

        logger.info("Agent execution completed")
        return results

    def _reset_run_state(self) -> None:
        """Reset per-run supervisor state so multiple executions stay isolated."""
        for agent_info in self.agents.values():
            agent_info.last_run_result = None
            agent_info.error = None
            agent_info.status = (
                AgentStatus.INITIALIZED
                if agent_info.agent._initialized
                else AgentStatus.CREATED
            )

    def validate(self) -> bool:
        """
        Validate all registered agents.

        Returns:
            True if all agents validate successfully
        """
        logger.info("Validating all registered agents")

        all_valid = True
        for agent_name, agent_info in self.agents.items():
            try:
                if agent_info.agent.validate():
                    logger.info(f"Agent {agent_name} validation passed")
                else:
                    logger.error(f"Agent {agent_name} validation failed")
                    all_valid = False
            except Exception as e:
                logger.error(f"Agent {agent_name} validation error: {e}")
                all_valid = False

        return all_valid

    def cleanup_all(self) -> None:
        """Clean up all registered agents."""
        logger.info("Cleaning up all agents")
        for agent_name, agent_info in self.agents.items():
            if agent_info.status in [
                AgentStatus.INITIALIZED,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
            ]:
                try:
                    agent_info.agent.cleanup()
                    agent_info.status = AgentStatus.CLEANED_UP
                    logger.info(f"Cleaned up agent {agent_name}")
                except Exception as e:
                    logger.error(f"Failed to clean up agent {agent_name}: {e}")

    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """
        Get status of a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent status or None if agent not found
        """
        if agent_name in self.agents:
            return self.agents[agent_name].status
        return None

    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """
        Get status of all registered agents.

        Returns:
            Dictionary mapping agent names to their statuses
        """
        return {name: info.status for name, info in self.agents.items()}

    def _setup(self) -> None:
        """Setup supervisor resources."""
        # Initialize all registered agents
        for agent_name, agent_info in self.agents.items():
            if agent_info.status == AgentStatus.CREATED:
                try:
                    agent_info.agent.initialize()
                    agent_info.status = AgentStatus.INITIALIZED
                    logger.info(
                        f"Initialized agent {agent_name} during supervisor setup"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize agent {agent_name} during setup: {e}"
                    )
                    raise

    def _teardown(self) -> None:
        """Teardown supervisor resources."""
        self.cleanup_all()

    def __enter__(self) -> "SupervisorAgent":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup_all()


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "stop_on_agent_failure": True,
        "agents": {
            "scraper": {"type": "scraper"},
            "classifier": {"type": "classifier"},
            "emailer": {"type": "emailer"},
        },
    }

    supervisor = SupervisorAgent(config)

    # Create example agents (would be actual agents in real usage)
    from arxiv_agent.agents.base import ExampleAgent

    scraper = ExampleAgent(name="scraper", config={})
    classifier = ExampleAgent(name="classifier", config={})
    emailer = ExampleAgent(name="emailer", config={})

    supervisor.register_agent(scraper)
    supervisor.register_agent(classifier)
    supervisor.register_agent(emailer)

    supervisor.set_execution_order(["scraper", "classifier", "emailer"])

    # Run validation
    if supervisor.validate():
        print("All agents validated successfully")

        # Run agents
        results = supervisor.run()
        print(f"Execution results: {results}")

        # Clean up
        supervisor.cleanup_all()
    else:
        print("Agent validation failed")
