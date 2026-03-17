"""Command-line interface for Arxiv-Agent."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from arxiv_agent.agents import (
    ClassifierAgent,
    EmailerAgent,
    ScraperAgent,
    SupervisorAgent,
)
from arxiv_agent.config import Config
from arxiv_agent.scheduler import Scheduler
from arxiv_agent.storage.archiver import Archiver
from arxiv_agent.utils.intervals import RunOnceInterval
from arxiv_agent.utils.logging import setup_logging
from arxiv_agent.utils.retry import RetryError
from arxiv_agent.utils.runtime import describe_retry_error
from arxiv_agent.utils.timezone import get_current_date_in_timezone

logger = logging.getLogger(__name__)


def version_command() -> None:
    """Print version information."""
    from arxiv_agent import __version__

    print(f"Arxiv-Agent v{__version__}")
    print("Automated paper discovery system")


def start_command(config: Config) -> Scheduler:
    """Start the scheduled agent in the foreground."""
    logger.info("Starting scheduled agent...")
    scheduler = Scheduler(config.agent.timezone)
    scheduler.start()

    def run_scan_job() -> Dict[str, Any]:
        target_date = _get_target_date(config)
        logger.info("Starting scheduled scan workflow for %s", target_date)
        result = _run_scan_workflow(config, target_date)
        if not result["success"]:
            logger.error("Scheduled scan workflow failed: %s", result)
        return result

    def run_email_job() -> Dict[str, Any]:
        target_date = _get_target_date(config)
        logger.info("Starting scheduled email workflow for %s", target_date)
        result = _run_email_workflow(config, target_date, dry_run=False)
        if not result["success"]:
            logger.error("Scheduled email workflow failed: %s", result)
        return result

    scheduler.configure_daily_jobs(
        scan_job=run_scan_job,
        email_job=run_email_job,
        scan_time=config.schedule.scan_time,
        email_time=config.schedule.email_time,
    )
    print("Scheduler started. Press Ctrl+C to stop.")
    scheduler.run_forever()
    return scheduler


def run_once_command(
    config: Config,
    dry_run: bool = False,
    *,
    run_interval: Optional[RunOnceInterval] = None,
    no_email: bool = False,
) -> Dict[str, Any]:
    """Run a single scan and email cycle."""
    if run_interval is not None:
        logger.info(
            "Running one-time interval execution from %s to %s (dry_run=%s, no_email=%s)",
            run_interval.start_date.isoformat(),
            run_interval.end_date.isoformat(),
            dry_run,
            no_email,
        )
        scan_result = _run_interval_scan_workflow(config, run_interval)
        if not scan_result["success"]:
            raise RuntimeError(f"Interval scan workflow failed: {scan_result}")

        if no_email:
            email_result = _build_skipped_email_result(
                "Email skipped by --no-email",
                dry_run=dry_run,
            )
        elif not scan_result["affected_days"]:
            email_result = _build_skipped_email_result(
                "No affected days found for interval",
                dry_run=dry_run,
            )
        else:
            email_result = _run_interval_email_workflow(
                config,
                [date.fromisoformat(day) for day in scan_result["affected_days"]],
                dry_run=dry_run,
            )
            if not email_result["success"]:
                raise RuntimeError(f"Interval email workflow failed: {email_result}")

        mode = "dry-run" if dry_run else "live"
        print(
            "Run-once "
            f"{mode} completed for interval "
            f"{run_interval.start_date.isoformat()} -> {run_interval.end_date.isoformat()}"
        )
        return {
            "success": True,
            "mode": "interval",
            "interval": run_interval.to_dict(),
            "affected_days": scan_result["affected_days"],
            "email_skipped": bool(email_result.get("skipped")),
            "scan": scan_result,
            "email": email_result,
        }

    target_date = _get_target_date(config)
    logger.info(
        "Running one-time execution for %s (dry_run=%s, no_email=%s)",
        target_date,
        dry_run,
        no_email,
    )

    scan_result = _run_scan_workflow(config, target_date)
    if not scan_result["success"]:
        raise RuntimeError(f"Scan workflow failed: {scan_result}")

    if no_email:
        email_result = _build_skipped_email_result(
            "Email skipped by --no-email",
            dry_run=dry_run,
        )
    else:
        email_result = _run_email_workflow(config, target_date, dry_run=dry_run)
        if not email_result["success"]:
            raise RuntimeError(f"Email workflow failed: {email_result}")

    mode = "dry-run" if dry_run else "live"
    print(f"Run-once {mode} completed for {target_date.isoformat()}")
    return {
        "success": True,
        "mode": "single_day",
        "target_date": target_date.isoformat(),
        "email_skipped": bool(email_result.get("skipped")),
        "scan": scan_result,
        "email": email_result,
    }


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Arxiv-Agent: Automated paper discovery system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arxiv-agent start                # Start scheduled agent
  arxiv-agent run-once             # Run one-time scan and email
  arxiv-agent run-once --dry-run   # Dry run without actual operations
  arxiv-agent run-once --from 2026-03-15 --to 2026-03-16
                                   # One-off arXiv date interval with no timezone shift
  arxiv-agent version              # Show version information
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    start_parser = subparsers.add_parser("start", help="Start scheduled agent")
    start_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )

    run_once_parser = subparsers.add_parser(
        "run-once", help="Run one-time scan and email"
    )
    run_once_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )
    run_once_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (no actual operations)"
    )
    run_once_parser.add_argument(
        "--from",
        dest="from_date",
        help="Start date for the direct arXiv interval (YYYY-MM-DD)",
    )
    run_once_parser.add_argument(
        "--to",
        dest="to_date",
        help="End date for the direct arXiv interval (YYYY-MM-DD)",
    )
    run_once_parser.add_argument(
        "--no-email",
        action="store_true",
        help="Skip the email workflow after scraping and classification",
    )

    subparsers.add_parser("version", help="Show version information")
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command in ["start", "run-once"]:
            config = Config.from_yaml(parsed_args.config)
            config.load_env(parsed_args.config.parent / ".env")
            setup_logging(
                log_dir=config.storage.log_dir,
                log_level=config.advanced.log_level,
            )
            _log_runtime_configuration(config, parsed_args.command)

            if not config.validate():
                logger.error("Configuration validation failed")
                return 1

            require_email = parsed_args.command == "start" or not getattr(
                parsed_args,
                "dry_run",
                False,
            )
            if getattr(parsed_args, "no_email", False):
                require_email = False
            if not config.validate_runtime_requirements(
                require_llm=True,
                require_email=require_email,
            ):
                logger.error("Runtime configuration validation failed")
                return 1
        else:
            setup_logging(file=False)
            config = None

        if parsed_args.command == "start":
            assert config is not None
            start_command(config)
        elif parsed_args.command == "run-once":
            assert config is not None
            run_interval = _parse_run_once_interval(config, parsed_args)
            run_once_command(
                config,
                dry_run=parsed_args.dry_run,
                run_interval=run_interval,
                no_email=parsed_args.no_email,
            )
        elif parsed_args.command == "version":
            version_command()
        else:
            parser.print_help()
            return 1

        return 0

    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        print(f"Error: Configuration file not found: {exc}", file=sys.stderr)
        print("Please create config.yaml from config.yaml.example", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        if isinstance(exc, RetryError):
            print(
                f"Error: {describe_retry_error(exc, 'Operation failed')}",
                file=sys.stderr,
            )
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


def _log_runtime_configuration(config: Config, command: str) -> None:
    """Log the effective runtime configuration without leaking secrets."""
    logger.info(
        (
            "Runtime configuration ready: command=%s timezone=%s source=%s "
            "llm_provider=%s model=%s log_level=%s log_dir=%s data_dir=%s "
            "archive_dir=%s scan_time=%s email_time=%s retries=%s timeout=%ss"
        ),
        command,
        config.agent.timezone,
        config.sources.primary,
        config.llm.provider,
        config.llm.model,
        config.advanced.log_level.upper(),
        config.storage.log_dir,
        config.storage.data_dir,
        config.storage.archive_dir,
        config.schedule.scan_time,
        config.schedule.email_time,
        config.advanced.max_retries,
        config.advanced.request_timeout,
    )


def _run_scan_workflow(config: Config, target_date: date) -> Dict[str, Any]:
    """Run the scrape and classify workflow and then archive old data."""
    scraper_result = _run_scraper_workflow(config, target_date=target_date)
    if not scraper_result["success"]:
        return {
            "success": False,
            "target_date": target_date.isoformat(),
            "scraper": scraper_result,
            "classifier": {"success": False, "message": "Skipped after scraper failure"},
            "archives_created": [],
        }

    classifier_result = _run_classifier_workflow(config, target_date=target_date)
    workflow_success = bool(
        scraper_result.get("success") and classifier_result.get("success")
    )

    archives_created = []
    if workflow_success:
        archives_created = _archive_old_data(config)

    return {
        "success": workflow_success,
        "target_date": target_date.isoformat(),
        "scraper": scraper_result,
        "classifier": classifier_result,
        "archives_created": [str(path) for path in archives_created],
    }


def _run_interval_scan_workflow(
    config: Config,
    run_interval: RunOnceInterval,
) -> Dict[str, Any]:
    """Run the scrape/classify workflow for a custom datetime interval."""
    scraper_result = _run_scraper_workflow(config, run_interval=run_interval)
    if not scraper_result["success"]:
        return {
            "success": False,
            "interval": run_interval.to_dict(),
            "affected_days": [],
            "scraper": scraper_result,
            "classifiers": {},
            "archives_created": [],
        }

    affected_days = list(scraper_result.get("affected_days", []))
    classifier_results: Dict[str, Any] = {}
    workflow_success = bool(scraper_result.get("success"))

    for day_text in affected_days:
        day = date.fromisoformat(day_text)
        classifier_result = _run_classifier_workflow(config, target_date=day)
        classifier_results[day_text] = classifier_result
        workflow_success = workflow_success and bool(classifier_result.get("success"))

    archives_created = []
    if workflow_success:
        archives_created = _archive_old_data(config)

    return {
        "success": workflow_success,
        "interval": run_interval.to_dict(),
        "affected_days": affected_days,
        "scraper": scraper_result,
        "classifiers": classifier_results,
        "archives_created": [str(path) for path in archives_created],
    }


def _run_email_workflow(
    config: Config,
    target_date: date,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run the email workflow for the target date."""
    supervisor = SupervisorAgent({"stop_on_agent_failure": True})
    supervisor.register_agent(EmailerAgent(config.to_dict()))
    supervisor.set_execution_order(["emailer"])

    try:
        if not supervisor.validate():
            return {"success": False, "message": "Email workflow validation failed"}

        results = supervisor.run(target_date=target_date, dry_run=dry_run)
    finally:
        supervisor.cleanup_all()

    emailer_result = results.get("emailer", {})
    return {
        **emailer_result,
        "success": bool(emailer_result.get("success")),
        "target_date": target_date.isoformat(),
        "results": results,
    }


def _run_interval_email_workflow(
    config: Config,
    affected_days: list[date],
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run the existing daily email workflow once per affected day."""
    per_day: Dict[str, Any] = {}
    overall_success = True

    for day in sorted(affected_days):
        result = _run_email_workflow(config, day, dry_run=dry_run)
        per_day[day.isoformat()] = result
        overall_success = overall_success and bool(result.get("success"))

    return {
        "success": overall_success,
        "dry_run": dry_run,
        "skipped": False,
        "by_day": per_day,
    }


def _run_scraper_workflow(
    config: Config,
    *,
    target_date: Optional[date] = None,
    run_interval: Optional[RunOnceInterval] = None,
) -> Dict[str, Any]:
    """Run only the scraper agent."""
    supervisor = SupervisorAgent({"stop_on_agent_failure": True})
    supervisor.register_agent(ScraperAgent(config.to_dict()))
    supervisor.set_execution_order(["scraper"])

    try:
        if not supervisor.validate():
            return {"success": False, "message": "Scraper workflow validation failed"}

        if run_interval is not None:
            results = supervisor.run(run_interval=run_interval)
        else:
            assert target_date is not None
            results = supervisor.run(target_date=target_date)
    finally:
        supervisor.cleanup_all()

    scraper_result = results.get("scraper", {})
    response: Dict[str, Any] = {
        **scraper_result,
        "success": bool(scraper_result.get("success")),
        "results": results,
    }
    if target_date is not None:
        response["target_date"] = target_date.isoformat()
    if run_interval is not None:
        response["interval"] = run_interval.to_dict()
        response["affected_days"] = scraper_result.get("affected_days", [])
        response["stored_by_day"] = scraper_result.get("stored_by_day", {})
    return response


def _run_classifier_workflow(config: Config, *, target_date: date) -> Dict[str, Any]:
    """Run only the classifier agent for one storage day."""
    supervisor = SupervisorAgent({"stop_on_agent_failure": True})
    supervisor.register_agent(ClassifierAgent(config.to_dict()))
    supervisor.set_execution_order(["classifier"])

    try:
        if not supervisor.validate():
            return {"success": False, "message": "Classifier workflow validation failed"}

        results = supervisor.run(target_date=target_date)
    finally:
        supervisor.cleanup_all()

    classifier_result = results.get("classifier", {})
    return {
        **classifier_result,
        "success": bool(classifier_result.get("success")),
        "target_date": target_date.isoformat(),
        "results": results,
    }


def _archive_old_data(config: Config) -> list[Path]:
    """Archive expired daily files using the configured retention policy."""
    archiver = Archiver(
        data_dir=config.storage.data_dir,
        archive_dir=config.storage.archive_dir,
    )
    return archiver.archive_old_data(config.storage.retention_days)


def _build_skipped_email_result(message: str, *, dry_run: bool) -> Dict[str, Any]:
    """Return a structured skipped-email result for run-once workflows."""
    return {
        "success": True,
        "skipped": True,
        "dry_run": dry_run,
        "by_day": {},
        "message": message,
    }


def _parse_run_once_interval(
    config: Config,
    parsed_args: argparse.Namespace,
) -> Optional[RunOnceInterval]:
    """Parse and validate the optional run-once interval arguments."""
    from_value = getattr(parsed_args, "from_date", None)
    to_value = getattr(parsed_args, "to_date", None)

    if bool(from_value) != bool(to_value):
        raise ValueError("run-once requires both --from and --to together")
    if not from_value:
        return None

    start = _parse_date_value("--from", from_value)
    end = _parse_date_value("--to", to_value)
    return RunOnceInterval.from_dates(
        start,
        end,
    )


def _parse_date_value(flag_name: str, value: str) -> date:
    """Parse a date-only value for run-once interval mode."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{flag_name} must be a valid ISO date, for example 2026-03-15"
        ) from exc
    return parsed


def _get_target_date(config: Config) -> date:
    """Return the yesterday's date in the configured timezone."""
    return get_current_date_in_timezone(config.agent.timezone) - timedelta(days=1)


if __name__ == "__main__":
    sys.exit(main())
