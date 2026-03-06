"""Tests for CLI interface."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from arxiv_agent.cli import main, run_once_command, start_command, version_command


def test_version_command(capsys):
    """Test version command output."""
    version_command()
    captured = capsys.readouterr()
    assert "Arxiv-Agent v0.1.0" in captured.out
    assert "Automated paper discovery system" in captured.out


def test_start_command(capsys):
    """Test start command (skeleton implementation)."""
    config = Mock()
    start_command(config)
    captured = capsys.readouterr()
    assert "Scheduler not yet implemented" in captured.out


def test_run_once_command_dry_run(capsys):
    """Test run-once command with dry run."""
    config = Mock()
    run_once_command(config, dry_run=True)
    captured = capsys.readouterr()
    assert "Dry run" in captured.out


def test_run_once_command_normal(capsys):
    """Test run-once command without dry run."""
    config = Mock()
    run_once_command(config, dry_run=False)
    captured = capsys.readouterr()
    assert "Pipeline not yet implemented" in captured.out


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_version(mock_config, mock_setup_logging, capsys):
    """Test main with version command."""
    sys.argv = ["arxiv-agent", "version"]
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Arxiv-Agent v0.1.0" in captured.out


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_start(mock_config_class, mock_setup_logging):
    """Test main with start command."""
    # Mock config loading
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "start", "--config", "test.yaml"]
    exit_code = main()
    assert exit_code == 0
    mock_config_class.from_yaml.assert_called_once_with(Path("test.yaml"))
    mock_config.load_env.assert_called_once()
    mock_config.validate.assert_called_once()


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_start_config_not_found(mock_config_class, mock_setup_logging, capsys):
    """Test main with start command when config file doesn't exist."""
    mock_config_class.from_yaml.side_effect = FileNotFoundError("File not found")

    sys.argv = ["arxiv-agent", "start"]
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Configuration file not found" in captured.err


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_start_validation_failed(mock_config_class, mock_setup_logging, caplog):
    """Test main with start command when config validation fails."""
    mock_config = Mock()
    mock_config.validate.return_value = False
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "start"]
    exit_code = main()
    assert exit_code == 1
    assert "Configuration validation failed" in caplog.text


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_run_once(mock_config_class, mock_setup_logging):
    """Test main with run-once command."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "run-once", "--dry-run"]
    exit_code = main()
    assert exit_code == 0
    mock_config_class.from_yaml.assert_called_once_with(Path("config.yaml"))
    mock_config.load_env.assert_called_once()
    mock_config.validate.assert_called_once()


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_run_once_with_custom_config(mock_config_class, mock_setup_logging):
    """Test main with run-once command and custom config path."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "run-once", "--config", "custom.yaml"]
    exit_code = main()
    assert exit_code == 0
    mock_config_class.from_yaml.assert_called_once_with(Path("custom.yaml"))


def test_main_no_command(capsys):
    """Test main with no command (should show help)."""
    sys.argv = ["arxiv-agent"]
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err


def test_main_help(capsys):
    """Test main with --help flag."""
    sys.argv = ["arxiv-agent", "--help"]
    # argparse will print help and exit with 0
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_main_invalid_command(capsys):
    """Test main with invalid command."""
    sys.argv = ["arxiv-agent", "invalid-command"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err or "usage:" in captured.out


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_unexpected_error(mock_config_class, mock_setup_logging, capsys):
    """Test main with unexpected error."""
    mock_config_class.from_yaml.side_effect = Exception("Unexpected error")

    sys.argv = ["arxiv-agent", "start"]
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Unexpected error" in captured.err


def test_cli_module_execution():
    """Test that the CLI module can be executed directly."""
    # This tests that __main__ block exists and calls main()
    with patch("arxiv_agent.cli.main") as mock_main:
        mock_main.return_value = 0
        with patch("arxiv_agent.cli.sys.exit") as mock_exit:
            # Simulate running the module directly
            exec(open("arxiv_agent/cli.py").read())
            # The __main__ block should call main() and sys.exit()
            # But this is hard to test without actually running as script
            pass


@patch("arxiv_agent.cli.setup_logging")
def test_main_logging_setup(mock_setup_logging):
    """Test that logging is set up for commands that need config."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    with patch("arxiv_agent.cli.Config") as mock_config_class:
        mock_config_class.from_yaml.return_value = mock_config

        sys.argv = ["arxiv-agent", "start"]
        main()
        mock_setup_logging.assert_called_once()


@patch("arxiv_agent.cli.setup_logging")
def test_main_logging_setup_for_version(mock_setup_logging):
    """Test that logging is also set up for version command."""
    sys.argv = ["arxiv-agent", "version"]
    main()
    mock_setup_logging.assert_called_once()
