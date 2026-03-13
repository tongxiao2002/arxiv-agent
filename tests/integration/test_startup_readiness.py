"""Integration tests for CLI startup readiness and fail-fast behavior."""

from pathlib import Path
from unittest.mock import patch

import yaml

from arxiv_agent.cli import main


def _write_config(path: Path, data: dict) -> Path:
    """Write a config dict to disk."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base_config(tmp_path: Path) -> dict:
    """Return a supported config for startup tests."""
    return {
        "agent": {"timezone": "UTC"},
        "sources": {
            "primary": "arxiv",
            "arxiv": {"categories": ["cs"], "max_papers": 5},
        },
        "topics": ["agents"],
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_security": "starttls",
            "smtp_username": "mailer",
            "from_email": "agent@example.com",
            "to_emails": ["user@example.com"],
            "subject_template": "Digest - {date}",
        },
        "storage": {
            "data_dir": str(tmp_path / "papers"),
            "archive_dir": str(tmp_path / "archive"),
            "log_dir": str(tmp_path / "runtime-logs"),
            "retention_days": 30,
        },
        "advanced": {
            "log_level": "DEBUG",
            "max_retries": 2,
            "retry_backoff_factor": 1.0,
            "request_timeout": 11,
        },
    }


def test_cli_startup_uses_configured_log_dir(tmp_path, monkeypatch):
    """Test startup logging honors configured log directory and level."""
    config_file = _write_config(tmp_path / "config.yaml", _base_config(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("arxiv_agent.cli.run_once_command", return_value={"success": True}):
        exit_code = main(["run-once", "--dry-run", "--config", str(config_file)])

    assert exit_code == 0
    log_dir = tmp_path / "runtime-logs"
    assert log_dir.exists()
    assert (log_dir / "arxiv-agent.log").exists()


def test_cli_startup_fails_fast_for_unsupported_source(tmp_path):
    """Test unsupported sources fail before workflow execution."""
    config = _base_config(tmp_path)
    config["sources"]["primary"] = "papers_cool"
    config["sources"]["papers_cool"] = {"categories": ["cs.ai"], "max_papers": 5}
    config_file = _write_config(tmp_path / "config.yaml", config)

    with patch("arxiv_agent.cli.run_once_command") as mock_run_once:
        exit_code = main(["run-once", "--dry-run", "--config", str(config_file)])

    assert exit_code == 1
    mock_run_once.assert_not_called()


def test_cli_startup_fails_fast_for_unsupported_local_provider(tmp_path):
    """Test unsupported local provider fails before workflow execution."""
    config = _base_config(tmp_path)
    config["llm"]["provider"] = "local"
    config["llm"]["model"] = "local-model"
    config_file = _write_config(tmp_path / "config.yaml", config)

    with patch("arxiv_agent.cli.run_once_command") as mock_run_once:
        exit_code = main(["run-once", "--dry-run", "--config", str(config_file)])

    assert exit_code == 1
    mock_run_once.assert_not_called()


def test_cli_startup_requires_llm_secret(tmp_path):
    """Test startup blocks execution when the required LLM key is missing."""
    config_file = _write_config(tmp_path / "config.yaml", _base_config(tmp_path))

    with patch("arxiv_agent.cli.run_once_command") as mock_run_once:
        exit_code = main(["run-once", "--dry-run", "--config", str(config_file)])

    assert exit_code == 1
    mock_run_once.assert_not_called()
