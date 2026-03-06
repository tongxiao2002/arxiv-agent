"""Shared pytest fixtures for Arxiv-Agent tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import yaml

from arxiv_agent.config import Config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp(prefix="arxiv_agent_test_")
    tmp_path = Path(tmpdir)
    yield tmp_path
    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Return a sample configuration dictionary."""
    return {
        "agent": {"name": "test-agent", "timezone": "UTC"},
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs", "physics"], "max_papers": 10},
            "papers_cool": {"categories": ["cs.ai"], "max_papers": 5},
        },
        "topics": ["machine learning", "deep learning"],
        "schedule": {"scan_time": "00:00", "email_time": "09:00"},
        "llm": {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "classification_temperature": 0.1,
            "summarization_temperature": 0.3,
        },
        "email": {
            "service": "sendgrid",
            "from_email": "test@example.com",
            "to_emails": ["user@example.com"],
            "subject_template": "Test Digest - {date}",
        },
        "storage": {
            "data_dir": "./papers",
            "archive_dir": "./archive",
            "log_dir": "./logs",
            "retention_days": 30,
        },
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """Create a temporary YAML configuration file."""
    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_file


@pytest.fixture
def config(sample_config_dict: Dict[str, Any]) -> Config:
    """Create a Config instance from sample configuration."""
    return Config.from_dict(sample_config_dict)


@pytest.fixture
def env_file(temp_dir: Path) -> Path:
    """Create a temporary .env file with test variables."""
    env_file = temp_dir / ".env"
    env_content = """# Test environment variables
OPENAI_API_KEY="sk-test123"
SENDGRID_API_KEY="SG.test"
TZ="UTC"
LOG_LEVEL="DEBUG"
"""
    env_file.write_text(env_content)
    return env_file


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


@pytest.fixture
def mock_requests(mocker):
    """Mock requests module to prevent external HTTP calls."""
    return mocker.patch("requests.get")


@pytest.fixture
def mock_scheduler(mocker):
    """Mock APScheduler to prevent actual scheduling."""
    return mocker.patch("apscheduler.schedulers.background.BackgroundScheduler")


# Async fixtures if needed
try:
    import pytest_asyncio

    @pytest_asyncio.fixture
    async def async_fixture():
        """Example async fixture."""
        yield "async_value"

except ImportError:
    pass
