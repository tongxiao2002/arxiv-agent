"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from arxiv_agent.config import AgentConfig, Config, EmailConfig, LLMConfig, SourceConfig


def test_config_defaults():
    """Test that default configuration is created correctly."""
    config = Config()
    assert config.agent.name == "arxiv-agent"
    assert config.agent.timezone == "Asia/Shanghai"
    assert config.sources.primary == "arxiv"
    assert "machine learning" in config.topics
    assert config.schedule.scan_time == "00:00"
    assert config.llm.provider == "openai"
    assert config.email.smtp_security == "starttls"
    assert config.storage.data_dir == "./papers"


def test_config_from_dict():
    """Test creating Config from dictionary."""
    data = {
        "agent": {"name": "test-agent", "timezone": "UTC"},
        "sources": {
            "primary": "papers_cool",
            "arxiv": {"categories": ["physics"]},
            "papers_cool": {"categories": ["cs.ai"]},
        },
        "topics": ["quantum computing", "AI"],
        "schedule": {"scan_time": "01:00", "email_time": "10:00"},
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-latest",
            "classification_temperature": 0.2,
            "summarization_temperature": 0.4,
        },
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 465,
            "smtp_security": "ssl",
            "smtp_username": "mailer",
            "from_email": "test@example.com",
            "to_emails": ["user1@example.com", "user2@example.com"],
            "subject_template": "Test - {date}",
        },
        "storage": {
            "data_dir": "/custom/papers",
            "archive_dir": "/custom/archive",
            "log_dir": "/custom/logs",
            "retention_days": 90,
        },
    }

    config = Config.from_dict(data)
    assert config.agent.name == "test-agent"
    assert config.agent.timezone == "UTC"
    assert config.sources.primary == "papers_cool"
    assert config.sources.arxiv.categories == ["physics"]
    assert config.topics == ["quantum computing", "AI"]
    assert config.schedule.scan_time == "01:00"
    assert config.llm.provider == "anthropic"
    assert config.email.smtp_security == "ssl"
    assert config.email.to_emails == ["user1@example.com", "user2@example.com"]
    assert config.storage.data_dir == "/custom/papers"
    assert config.storage.retention_days == 90


def test_config_from_yaml_file(temp_dir, sample_config_dict):
    """Test loading configuration from YAML file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as handle:
        yaml.dump(sample_config_dict, handle)

    config = Config.from_yaml(config_file)
    assert config.agent.name == "test-agent"
    assert config.sources.primary == "arxiv"


def test_config_from_yaml_file_not_found():
    """Test error when YAML file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(Path("/nonexistent/config.yaml"))


def test_config_validation_valid(monkeypatch, config):
    """Test schema validation does not depend on runtime secrets."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert config.validate() is True


def test_config_validation_invalid_timezone():
    """Test validation with invalid timezone."""
    config = Config()
    config.agent.timezone = "Invalid/Timezone"
    assert config.validate() is False


def test_config_validation_invalid_llm_provider():
    """Test validation with invalid LLM provider."""
    config = Config()
    config.llm.provider = "invalid_provider"
    assert config.validate() is False


def test_config_validation_invalid_smtp_security():
    """Test validation with invalid SMTP security mode."""
    config = Config()
    config.email.smtp_security = "tls"
    assert config.validate() is False


def test_config_validation_invalid_primary_source():
    """Test validation with invalid primary source."""
    config = Config()
    config.sources.primary = "invalid_source"
    assert config.validate() is False


def test_config_validation_invalid_email_address():
    """Test validation with invalid email address."""
    config = Config()
    config.email.to_emails = ["not-an-email"]
    assert config.validate() is False


def test_config_validation_missing_recipient():
    """Test validation with no recipient emails."""
    config = Config()
    config.email.to_emails = []
    assert config.validate() is False


def test_config_runtime_validation_llm_missing_key(monkeypatch, config):
    """Test runtime readiness for LLM credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert config.validate_runtime_requirements(require_llm=True) is False


def test_config_runtime_validation_rejects_unsupported_source(config):
    """Test runtime validation rejects unsupported primary sources."""
    config.sources.primary = "papers_cool"
    errors = config.get_runtime_validation_errors()
    assert any("papers_cool" in error for error in errors)


def test_config_runtime_validation_rejects_local_provider(config):
    """Test runtime validation rejects unsupported local provider."""
    config.llm.provider = "local"
    errors = config.get_runtime_validation_errors()
    assert any("local" in error for error in errors)


def test_config_runtime_validation_email_missing_password(monkeypatch, config):
    """Test runtime readiness for SMTP credentials."""
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert config.validate_runtime_requirements(require_email=True) is False


def test_config_runtime_validation_email_not_required(monkeypatch, config):
    """Test runtime validation can skip SMTP checks."""
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert config.validate_runtime_requirements(require_email=False) is True


def test_config_load_env(env_file, config):
    """Test loading environment variables."""
    config.load_env(env_file)
    assert config.agent.timezone == "UTC"
    assert config.advanced.log_level == "DEBUG"


def test_config_load_env_advanced_overrides(temp_dir, config):
    """Test advanced runtime overrides can be loaded from .env."""
    env_file = temp_dir / ".env"
    env_file.write_text(
        "\n".join(
            [
                'MAX_RETRIES="7"',
                'RETRY_BACKOFF_FACTOR="3.5"',
                'REQUEST_TIMEOUT="45"',
            ]
        )
    )

    config.load_env(env_file)
    assert config.advanced.max_retries == 7
    assert config.advanced.retry_backoff_factor == 3.5
    assert config.advanced.request_timeout == 45


def test_config_load_env_file_not_found(config, caplog):
    """Test loading environment variables when file doesn't exist."""
    config.load_env(Path("/nonexistent/.env"))
    assert "not found" in caplog.text


def test_config_dataclasses():
    """Test individual configuration dataclasses."""
    agent = AgentConfig(name="test", timezone="UTC")
    assert agent.name == "test"

    source = SourceConfig(primary="arxiv")
    assert source.primary == "arxiv"

    llm = LLMConfig(provider="openai", model="gpt-4")
    assert llm.provider == "openai"

    email = EmailConfig(smtp_host="smtp.example.com", from_email="test@example.com")
    assert email.smtp_host == "smtp.example.com"


def test_config_unknown_section(caplog):
    """Test that unknown configuration sections are logged."""
    Config.from_dict({"unknown_section": {"key": "value"}, "agent": {"name": "test"}})
    assert "Unknown configuration section" in caplog.text


def test_config_unknown_key(caplog):
    """Test that unknown keys within sections are logged."""
    Config.from_dict({"agent": {"name": "test", "unknown_key": "value"}})
    assert "Unknown config key" in caplog.text


def test_config_repr():
    """Test string representation of Config."""
    config = Config()
    repr_str = repr(config)
    assert "Config" in repr_str
    assert "agent" in repr_str


def test_config_topics_override():
    """Test that topics list can be overridden."""
    data = {"topics": ["single_topic"]}
    config = Config.from_dict(data)
    assert config.topics == ["single_topic"]


def test_config_partial_override():
    """Test partial configuration override."""
    data = {
        "agent": {"timezone": "Europe/London"},
        "llm": {"model": "gpt-3.5-turbo"},
    }
    config = Config.from_dict(data)
    assert config.agent.timezone == "Europe/London"
    assert config.agent.name == "arxiv-agent"
    assert config.llm.model == "gpt-3.5-turbo"
    assert config.llm.provider == "openai"


def test_config_validation_invalid_advanced_settings():
    """Test static validation rejects invalid advanced values."""
    config = Config()
    config.advanced.max_retries = 0
    config.advanced.retry_backoff_factor = 0.5
    config.advanced.request_timeout = 0
    config.advanced.log_level = "LOUD"

    errors = config.get_validation_errors()
    assert any("advanced.max_retries" in error for error in errors)
    assert any("advanced.retry_backoff_factor" in error for error in errors)
    assert any("advanced.request_timeout" in error for error in errors)
    assert any("advanced.log_level" in error for error in errors)
