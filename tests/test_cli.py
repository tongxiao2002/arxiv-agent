"""Tests for CLI interface."""

import sys
from argparse import Namespace
from datetime import date
from unittest.mock import Mock, patch

import pytest

from arxiv_agent.cli import (
    _parse_run_once_interval,
    main,
    run_once_command,
    start_command,
    version_command,
)
from arxiv_agent.config import Config
from arxiv_agent.utils.intervals import RunOnceInterval


def make_config() -> Config:
    """Create a minimal real config for CLI unit tests."""
    return Config.from_dict(
        {
            "agent": {"timezone": "Asia/Shanghai"},
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
            "storage": {"log_dir": "./custom-logs"},
            "advanced": {"log_level": "DEBUG"},
        }
    )


def test_version_command(capsys):
    """Test version command output."""
    version_command()
    captured = capsys.readouterr()
    assert "Arxiv-Agent v0.1.0" in captured.out
    assert "Automated paper discovery system" in captured.out


def test_start_command_wires_scheduler():
    """Test start command schedules both workflows and blocks in foreground."""
    config = make_config()
    scheduler = Mock()

    with patch("arxiv_agent.cli.Scheduler", return_value=scheduler):
        with patch(
            "arxiv_agent.cli._run_scan_workflow", return_value={"success": True}
        ) as mock_scan:
            with patch(
                "arxiv_agent.cli._run_email_workflow", return_value={"success": True}
            ) as mock_email:
                start_command(config)
                kwargs = scheduler.configure_daily_jobs.call_args.kwargs
                kwargs["scan_job"]()
                kwargs["email_job"]()

    scheduler.start.assert_called_once()
    scheduler.configure_daily_jobs.assert_called_once()
    scheduler.run_forever.assert_called_once()

    mock_scan.assert_called_once()
    mock_email.assert_called_once()


def test_run_once_command_runs_scan_then_email(capsys):
    """Test run-once executes scan then email workflow."""
    config = make_config()
    observed = []

    def scan_side_effect(*args, **kwargs):
        observed.append("scan")
        return {"success": True}

    def email_side_effect(*args, **kwargs):
        observed.append("email")
        return {"success": True}

    with patch("arxiv_agent.cli._run_scan_workflow", side_effect=scan_side_effect):
        with patch(
            "arxiv_agent.cli._run_email_workflow", side_effect=email_side_effect
        ):
            result = run_once_command(config, dry_run=False)

    assert result["success"] is True
    assert observed == ["scan", "email"]
    captured = capsys.readouterr()
    assert "Run-once live completed" in captured.out


def test_run_once_command_dry_run(capsys):
    """Test run-once dry-run passes the dry-run flag to email workflow."""
    config = make_config()

    with patch("arxiv_agent.cli._run_scan_workflow", return_value={"success": True}):
        with patch(
            "arxiv_agent.cli._run_email_workflow", return_value={"success": True}
        ) as mock_email:
            result = run_once_command(config, dry_run=True)

    assert result["success"] is True
    _, kwargs = mock_email.call_args
    assert kwargs["dry_run"] is True
    captured = capsys.readouterr()
    assert "Run-once dry-run completed" in captured.out


def test_run_once_command_no_email_skips_email_workflow():
    """Test run-once can skip email entirely while still scanning."""
    config = make_config()

    with patch("arxiv_agent.cli._run_scan_workflow", return_value={"success": True}):
        with patch("arxiv_agent.cli._run_email_workflow") as mock_email:
            result = run_once_command(config, dry_run=False, no_email=True)

    assert result["success"] is True
    assert result["email_skipped"] is True
    mock_email.assert_not_called()


def test_run_once_command_interval_mode_runs_interval_workflows(capsys):
    """Test interval mode uses per-interval scan and email orchestration."""
    config = make_config()
    interval = RunOnceInterval.from_dates(date(2026, 3, 10), date(2026, 3, 11))

    with patch(
        "arxiv_agent.cli._run_interval_scan_workflow",
        return_value={"success": True, "affected_days": ["2026-03-10", "2026-03-11"]},
    ) as mock_scan:
        with patch(
            "arxiv_agent.cli._run_interval_email_workflow",
            return_value={"success": True, "skipped": False, "by_day": {}},
        ) as mock_email:
            result = run_once_command(
                config,
                dry_run=False,
                run_interval=interval,
                no_email=False,
            )

    assert result["success"] is True
    assert result["mode"] == "interval"
    mock_scan.assert_called_once_with(config, interval)
    mock_email.assert_called_once()
    captured = capsys.readouterr()
    assert "completed for interval" in captured.out


def test_parse_run_once_interval_requires_both_values():
    """Test interval mode rejects partial date input."""
    config = make_config()
    args = Namespace(from_date="2026-03-10", to_date=None)

    with pytest.raises(ValueError, match="both --from and --to together"):
        _parse_run_once_interval(config, args)


def test_parse_run_once_interval_rejects_invalid_date():
    """Test interval mode rejects invalid ISO dates."""
    config = make_config()
    args = Namespace(from_date="2026/03/10", to_date="2026-03-11")

    with pytest.raises(ValueError, match="valid ISO date"):
        _parse_run_once_interval(config, args)


def test_parse_run_once_interval_rejects_long_intervals():
    """Test interval mode enforces the 31-day cap."""
    config = make_config()
    args = Namespace(
        from_date="2026-03-01",
        to_date="2026-04-02",
    )

    with pytest.raises(ValueError, match="cannot exceed 31 days"):
        _parse_run_once_interval(config, args)


@patch("arxiv_agent.cli.setup_logging")
def test_main_version(mock_setup_logging, capsys):
    """Test main with version command."""
    sys.argv = ["arxiv-agent", "version"]
    exit_code = main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Arxiv-Agent v0.1.0" in captured.out
    mock_setup_logging.assert_called_once_with(file=False)


@patch("arxiv_agent.cli.start_command")
@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_start(mock_config_class, mock_setup_logging, mock_start_command):
    """Test main with start command."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config.validate_runtime_requirements.return_value = True
    mock_config.storage.log_dir = "./logs"
    mock_config.advanced.log_level = "WARNING"
    mock_config.agent.timezone = "UTC"
    mock_config.sources.primary = "arxiv"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.storage.data_dir = "./papers"
    mock_config.storage.archive_dir = "./archive"
    mock_config.schedule.scan_time = "00:00"
    mock_config.schedule.email_time = "09:00"
    mock_config.advanced.max_retries = 5
    mock_config.advanced.request_timeout = 30
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "start", "--config", "test.yaml"]
    exit_code = main()

    assert exit_code == 0
    mock_config_class.from_yaml.assert_called_once()
    mock_setup_logging.assert_called_once_with(
        log_dir="./logs",
        log_level="WARNING",
    )
    mock_config.load_env.assert_called_once()
    mock_config.validate.assert_called_once()
    mock_config.validate_runtime_requirements.assert_called_once_with(
        require_llm=True,
        require_email=True,
    )
    mock_start_command.assert_called_once_with(mock_config)


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


@patch("arxiv_agent.cli.run_once_command")
@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_run_once(mock_config_class, mock_setup_logging, mock_run_once_command):
    """Test main with run-once command."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config.validate_runtime_requirements.return_value = True
    mock_config.storage.log_dir = "./logs"
    mock_config.advanced.log_level = "INFO"
    mock_config.agent.timezone = "UTC"
    mock_config.sources.primary = "arxiv"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.storage.data_dir = "./papers"
    mock_config.storage.archive_dir = "./archive"
    mock_config.schedule.scan_time = "00:00"
    mock_config.schedule.email_time = "09:00"
    mock_config.advanced.max_retries = 5
    mock_config.advanced.request_timeout = 30
    mock_config_class.from_yaml.return_value = mock_config

    exit_code = main(["run-once", "--dry-run"])

    assert exit_code == 0
    mock_setup_logging.assert_called_once_with(
        log_dir="./logs",
        log_level="INFO",
    )
    mock_config.validate_runtime_requirements.assert_called_once_with(
        require_llm=True,
        require_email=False,
    )
    mock_run_once_command.assert_called_once_with(
        mock_config,
        dry_run=True,
        run_interval=None,
        no_email=False,
    )


@patch("arxiv_agent.cli.run_once_command")
@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_run_once_interval_no_email(
    mock_config_class,
    mock_setup_logging,
    mock_run_once_command,
):
    """Test main parses interval run-once flags and disables email requirements."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config.validate_runtime_requirements.return_value = True
    mock_config.storage.log_dir = "./logs"
    mock_config.advanced.log_level = "INFO"
    mock_config.agent.timezone = "Asia/Shanghai"
    mock_config.sources.primary = "arxiv"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.storage.data_dir = "./papers"
    mock_config.storage.archive_dir = "./archive"
    mock_config.schedule.scan_time = "00:00"
    mock_config.schedule.email_time = "09:00"
    mock_config.advanced.max_retries = 5
    mock_config.advanced.request_timeout = 30
    mock_config_class.from_yaml.return_value = mock_config

    exit_code = main(
        [
            "run-once",
            "--from",
            "2026-03-10",
            "--to",
            "2026-03-11",
            "--no-email",
        ]
    )

    assert exit_code == 0
    mock_config.validate_runtime_requirements.assert_called_once_with(
        require_llm=True,
        require_email=False,
    )
    _, kwargs = mock_run_once_command.call_args
    assert kwargs["no_email"] is True
    assert isinstance(kwargs["run_interval"], RunOnceInterval)


@patch("arxiv_agent.cli.setup_logging")
@patch("arxiv_agent.cli.Config")
def test_main_run_once_runtime_validation_failed(
    mock_config_class,
    mock_setup_logging,
    caplog,
):
    """Test main with run-once command when runtime validation fails."""
    mock_config = Mock()
    mock_config.validate.return_value = True
    mock_config.validate_runtime_requirements.return_value = False
    mock_config_class.from_yaml.return_value = mock_config

    sys.argv = ["arxiv-agent", "run-once"]
    exit_code = main()

    assert exit_code == 1
    assert "Runtime configuration validation failed" in caplog.text


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
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_start_rejects_interval_flags(capsys):
    """Test start does not accept run-once interval flags."""
    with pytest.raises(SystemExit) as exc_info:
        main(["start", "--from", "2026-03-10", "--to", "2026-03-11"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err


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
