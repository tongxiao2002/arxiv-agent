"""Configuration management for Arxiv-Agent."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Agent configuration."""

    name: str = "arxiv-agent"
    timezone: str = "Asia/Shanghai"


@dataclass
class ArxivSourceConfig:
    """arXiv source configuration."""

    categories: List[str] = field(default_factory=lambda: ["cs", "physics", "math"])
    max_papers: int = 100


@dataclass
class PapersCoolSourceConfig:
    """Papers.cool source configuration."""

    categories: List[str] = field(default_factory=lambda: ["cs.ai", "cs.lg"])
    max_papers: int = 50


@dataclass
class SourceConfig:
    """Source configuration."""

    primary: str = "arxiv"  # "arxiv" or "papers_cool"
    arxiv: ArxivSourceConfig = field(default_factory=ArxivSourceConfig)
    papers_cool: PapersCoolSourceConfig = field(default_factory=PapersCoolSourceConfig)


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai"  # "openai", "anthropic", "local"
    model: str = "gpt-4-turbo-preview"
    classification_temperature: float = 0.1
    summarization_temperature: float = 0.3


@dataclass
class EmailConfig:
    """Email service configuration."""

    service: str = "sendgrid"  # "sendgrid" or "mailgun"
    from_email: str = "arxiv-agent@example.com"
    to_emails: List[str] = field(default_factory=lambda: ["user@example.com"])
    subject_template: str = "Daily Papers Digest - {date}"


@dataclass
class StorageConfig:
    """Storage configuration."""

    data_dir: str = "./papers"
    archive_dir: str = "./archive"
    log_dir: str = "./logs"
    retention_days: int = 30


@dataclass
class ScheduleConfig:
    """Schedule configuration."""

    scan_time: str = "00:00"
    email_time: str = "09:00"


@dataclass
class AdvancedConfig:
    """Advanced configuration."""

    max_retries: int = 5
    retry_backoff_factor: float = 2.0
    request_timeout: int = 30
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    topics: List[str] = field(
        default_factory=lambda: ["machine learning", "deep learning"]
    )
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {yaml_path}")
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # For simplicity, we'll merge defaults with provided data
        # In a full implementation, we would recursively merge nested dicts
        config = cls()

        # Merge top-level sections
        for section_name, section_data in data.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                if isinstance(section_data, dict):
                    # Update dataclass fields
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                        else:
                            logger.warning(f"Unknown config key {section_name}.{key}")
                else:
                    # For topics list
                    if section_name == "topics":
                        config.topics = section_data
                    else:
                        logger.warning(
                            f"Unexpected type for {section_name}: {type(section_data)}"
                        )
            else:
                logger.warning(f"Unknown configuration section: {section_name}")

        return config

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Validate timezone
        try:
            import pytz

            pytz.timezone(self.agent.timezone)
        except (ImportError, pytz.exceptions.UnknownTimeZoneError):
            errors.append(f"Invalid timezone: {self.agent.timezone}")

        # Validate LLM provider
        if self.llm.provider not in ["openai", "anthropic", "local"]:
            errors.append(f"Invalid LLM provider: {self.llm.provider}")

        # Validate email service
        if self.email.service not in ["sendgrid", "mailgun"]:
            errors.append(f"Invalid email service: {self.email.service}")

        # Validate source primary
        if self.sources.primary not in ["arxiv", "papers_cool"]:
            errors.append(f"Invalid primary source: {self.sources.primary}")

        # Validate email addresses
        import re

        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        for email in self.email.to_emails:
            if not re.match(email_regex, email):
                errors.append(f"Invalid email address: {email}")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        return True

    def load_env(self, env_path: Optional[Path] = None) -> None:
        """Load environment variables from .env file."""
        if env_path is None:
            env_path = Path(".env")

        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f"Environment file not found: {env_path}")

        # Map environment variables to configuration
        # This can be extended based on needs
        tz = os.getenv("TZ")
        if tz:
            self.agent.timezone = tz

        # LLM API keys are handled by respective providers, not stored in config

    def __repr__(self) -> str:
        return (
            f"Config(agent={self.agent}, sources={self.sources}, topics={self.topics})"
        )


# Default configuration instance
default_config = Config()
