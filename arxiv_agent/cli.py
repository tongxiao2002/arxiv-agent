"""Command-line interface for Arxiv-Agent."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from arxiv_agent.agents import ClassifierAgent, EmailerAgent, ScraperAgent, SupervisorAgent
from arxiv_agent.config import Config
from arxiv_agent.scheduler import Scheduler
from arxiv_agent.storage.archiver import Archiver
from arxiv_agent.utils.logging import setup_logging
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


def run_once_command(config: Config, dry_run: bool = False) -> Dict[str, Any]:
    """Run a single scan and email cycle."""
    target_date = _get_target_date(config)
    logger.info("Running one-time execution for %s (dry_run=%s)", target_date, dry_run)

    scan_result = _run_scan_workflow(config, target_date)
    if not scan_result["success"]:
        raise RuntimeError(f"Scan workflow failed: {scan_result}")

    email_result = _run_email_workflow(config, target_date, dry_run=dry_run)
    if not email_result["success"]:
        raise RuntimeError(f"Email workflow failed: {email_result}")

    mode = "dry-run" if dry_run else "live"
    print(f"Run-once {mode} completed for {target_date.isoformat()}")
    return {
        "success": True,
        "target_date": target_date.isoformat(),
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

    subparsers.add_parser("version", help="Show version information")
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        setup_logging()

        if parsed_args.command in ["start", "run-once"]:
            logger.info("Loading configuration from %s", parsed_args.config)
            config = Config.from_yaml(parsed_args.config)
            config.load_env()

            if not config.validate():
                logger.error("Configuration validation failed")
                return 1

            require_email = parsed_args.command == "start" or not getattr(
                parsed_args,
                "dry_run",
                False,
            )
            if not config.validate_runtime_requirements(
                require_llm=True,
                require_email=require_email,
            ):
                logger.error("Runtime configuration validation failed")
                return 1
        else:
            config = None

        if parsed_args.command == "start":
            assert config is not None
            start_command(config)
        elif parsed_args.command == "run-once":
            assert config is not None
            run_once_command(config, dry_run=parsed_args.dry_run)
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
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run_scan_workflow(config: Config, target_date: date) -> Dict[str, Any]:
    """Run the scrape and classify workflow and then archive old data."""
    supervisor = SupervisorAgent({"stop_on_agent_failure": True})
    agent_config = config.to_dict()
    supervisor.register_agent(ScraperAgent(agent_config))
    supervisor.register_agent(ClassifierAgent(agent_config))
    supervisor.set_execution_order(["scraper", "classifier"])

    try:
        if not supervisor.validate():
            return {"success": False, "message": "Scan workflow validation failed"}

        results = supervisor.run(target_date=target_date)
    finally:
        supervisor.cleanup_all()

    scraper_result = results.get("scraper", {})
    classifier_result = results.get("classifier", {})
    workflow_success = bool(
        scraper_result.get("success") and classifier_result.get("success")
    )

    archives_created = []
    if workflow_success:
        archiver = Archiver(
            data_dir=config.storage.data_dir,
            archive_dir=config.storage.archive_dir,
        )
        archives_created = archiver.archive_old_data(config.storage.retention_days)

    return {
        "success": workflow_success,
        "target_date": target_date.isoformat(),
        "results": results,
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
        "success": bool(emailer_result.get("success")),
        "target_date": target_date.isoformat(),
        "results": results,
    }


def _get_target_date(config: Config) -> date:
    """Return the current date in the configured timezone."""
    return get_current_date_in_timezone(config.agent.timezone)


if __name__ == "__main__":
    sys.exit(main())
