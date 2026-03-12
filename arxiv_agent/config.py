"""Configuration management for Arxiv-Agent."""

import logging
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from arxiv_agent.utils.timezone import is_valid_timezone

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
TIME_REGEX = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
VALID_LLM_PROVIDERS = {"openai", "anthropic", "local"}
VALID_EMAIL_SECURITY_MODES = {"starttls", "ssl", "none"}
VALID_PRIMARY_SOURCES = {"arxiv", "papers_cool"}


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

    primary: str = "arxiv"
    arxiv: ArxivSourceConfig = field(default_factory=ArxivSourceConfig)
    papers_cool: PapersCoolSourceConfig = field(default_factory=PapersCoolSourceConfig)


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    classification_temperature: float = 0.1
    summarization_temperature: float = 0.3


@dataclass
class EmailConfig:
    """SMTP email configuration."""

    smtp_host: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_security: str = "starttls"
    smtp_username: str = "arxiv-agent@example.com"
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
        logger.info("Loading configuration from %s", yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        for section_name, section_data in data.items():
            if not hasattr(config, section_name):
                logger.warning("Unknown configuration section: %s", section_name)
                continue

            if section_name == "topics":
                config.topics = list(section_data)
                continue

            section = getattr(config, section_name)
            if is_dataclass(section) and isinstance(section_data, dict):
                _merge_dataclass(section, section_data, prefix=section_name)
                continue

            logger.warning(
                "Unexpected type for %s: %s",
                section_name,
                type(section_data).__name__,
            )

        return config

    def validate(self) -> bool:
        """Validate configuration shape and static values."""
        errors = self.get_validation_errors()
        for error in errors:
            logger.error("Configuration error: %s", error)
        return not errors

    def get_validation_errors(self) -> List[str]:
        """Return static configuration validation errors."""
        errors: List[str] = []

        if not is_valid_timezone(self.agent.timezone):
            errors.append(f"Invalid timezone: {self.agent.timezone}")

        if self.llm.provider not in VALID_LLM_PROVIDERS:
            errors.append(f"Invalid LLM provider: {self.llm.provider}")

        if self.sources.primary not in VALID_PRIMARY_SOURCES:
            errors.append(f"Invalid primary source: {self.sources.primary}")

        if not self.topics or not all(isinstance(topic, str) and topic.strip() for topic in self.topics):
            errors.append("Topics must contain at least one non-empty topic")

        if not TIME_REGEX.match(self.schedule.scan_time):
            errors.append(f"Invalid schedule.scan_time: {self.schedule.scan_time}")

        if not TIME_REGEX.match(self.schedule.email_time):
            errors.append(f"Invalid schedule.email_time: {self.schedule.email_time}")

        if self.email.smtp_security not in VALID_EMAIL_SECURITY_MODES:
            errors.append(f"Invalid email smtp_security: {self.email.smtp_security}")

        if not self.email.smtp_host.strip():
            errors.append("Email smtp_host is required")

        if not isinstance(self.email.smtp_port, int) or self.email.smtp_port <= 0:
            errors.append("Email smtp_port must be a positive integer")

        if not EMAIL_REGEX.match(self.email.from_email):
            errors.append(f"Invalid email address: {self.email.from_email}")

        if not self.email.to_emails:
            errors.append("At least one recipient email is required")
        else:
            for email in self.email.to_emails:
                if not EMAIL_REGEX.match(email):
                    errors.append(f"Invalid email address: {email}")

        if not self.email.subject_template or "{date}" not in self.email.subject_template:
            errors.append("Email subject_template must contain the {date} placeholder")

        return errors

    def validate_runtime_requirements(
        self,
        *,
        require_llm: bool = False,
        require_email: bool = False,
    ) -> bool:
        """Validate environment-dependent runtime requirements."""
        errors = self.get_runtime_validation_errors(
            require_llm=require_llm,
            require_email=require_email,
        )
        for error in errors:
            logger.error("Runtime configuration error: %s", error)
        return not errors

    def get_runtime_validation_errors(
        self,
        *,
        require_llm: bool = False,
        require_email: bool = False,
    ) -> List[str]:
        """Return runtime validation errors that depend on environment secrets."""
        errors: List[str] = []

        if require_llm and self.llm.provider in {"openai", "anthropic"}:
            env_var = (
                "OPENAI_API_KEY"
                if self.llm.provider == "openai"
                else "ANTHROPIC_API_KEY"
            )
            if not os.getenv(env_var):
                errors.append(
                    f"{env_var} environment variable not set for LLM provider {self.llm.provider}"
                )

        if require_email and self.email.smtp_username and not os.getenv("SMTP_PASSWORD"):
            errors.append(
                "SMTP_PASSWORD environment variable not set for authenticated SMTP delivery"
            )

        return errors

    def load_env(self, env_path: Optional[Path] = None) -> None:
        """Load environment variables from .env file."""
        env_file = env_path or Path(".env")

        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            logger.info("Loaded environment variables from %s", env_file)
        else:
            logger.warning("Environment file not found: %s", env_file)

        tz = os.getenv("TZ")
        if tz:
            self.agent.timezone = tz

        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            self.advanced.log_level = log_level

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Config(agent={self.agent}, sources={self.sources}, topics={self.topics})"
        )


def _merge_dataclass(instance: Any, values: Dict[str, Any], prefix: str) -> None:
    """Merge a nested dictionary into a dataclass instance."""
    for key, value in values.items():
        if not hasattr(instance, key):
            logger.warning("Unknown config key %s.%s", prefix, key)
            continue

        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value, prefix=f"{prefix}.{key}")
            continue

        setattr(instance, key, value)


default_config = Config()
