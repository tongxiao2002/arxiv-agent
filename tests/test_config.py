"""Tests for configuration management."""

import os
import tempfile
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
    assert config.email.service == "sendgrid"
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
            "model": "claude-3-opus",
            "classification_temperature": 0.2,
            "summarization_temperature": 0.4,
        },
        "email": {
            "service": "mailgun",
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
    assert config.sources.arxiv["categories"] == ["physics"]
    assert config.topics == ["quantum computing", "AI"]
    assert config.schedule.scan_time == "01:00"
    assert config.llm.provider == "anthropic"
    assert config.email.service == "mailgun"
    assert config.email.to_emails == ["user1@example.com", "user2@example.com"]
    assert config.storage.data_dir == "/custom/papers"
    assert config.storage.retention_days == 90


def test_config_from_yaml_file(temp_dir, sample_config_dict):
    """Test loading configuration from YAML file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)

    config = Config.from_yaml(config_file)
    assert config.agent.name == "test-agent"
    assert config.sources.primary == "arxiv"


def test_config_from_yaml_file_not_found():
    """Test error when YAML file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(Path("/nonexistent/config.yaml"))


def test_config_validation_valid(config):
    """Test validation with valid configuration."""
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


def test_config_validation_invalid_email_service():
    """Test validation with invalid email service."""
    config = Config()
    config.email.service = "invalid_service"
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


def test_config_load_env(env_file, config):
    """Test loading environment variables."""
    config.load_env(env_file)
    # Check that timezone was updated from .env
    assert config.agent.timezone == "UTC"


def test_config_load_env_file_not_found(config, caplog):
    """Test loading environment variables when file doesn't exist."""
    config.load_env(Path("/nonexistent/.env"))
    # Should log warning but not raise exception
    assert "not found" in caplog.text


def test_config_dataclasses():
    """Test individual configuration dataclasses."""
    agent = AgentConfig(name="test", timezone="UTC")
    assert agent.name == "test"

    source = SourceConfig(primary="arxiv")
    assert source.primary == "arxiv"

    llm = LLMConfig(provider="openai", model="gpt-4")
    assert llm.provider == "openai"

    email = EmailConfig(service="sendgrid", from_email="test@example.com")
    assert email.service == "sendgrid"


def test_config_unknown_section(caplog):
    """Test that unknown configuration sections are logged."""
    data = {"unknown_section": {"key": "value"}, "agent": {"name": "test"}}
    config = Config.from_dict(data)
    assert "Unknown configuration section" in caplog.text


def test_config_unknown_key(caplog):
    """Test that unknown keys within sections are logged."""
    data = {"agent": {"name": "test", "unknown_key": "value"}}
    config = Config.from_dict(data)
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
    data = {"agent": {"timezone": "Europe/London"}, "llm": {"model": "gpt-3.5-turbo"}}
    config = Config.from_dict(data)
    assert config.agent.timezone == "Europe/London"
    assert config.agent.name == "arxiv-agent"  # default preserved
    assert config.llm.model == "gpt-3.5-turbo"
    assert config.llm.provider == "openai"  # default preserved
