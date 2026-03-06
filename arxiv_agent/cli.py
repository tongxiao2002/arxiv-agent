"""Command-line interface for Arxiv-Agent."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from arxiv_agent.config import Config
from arxiv_agent.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def version_command() -> None:
    """Print version information."""
    from arxiv_agent import __version__

    print(f"Arxiv-Agent v{__version__}")
    print("Automated paper discovery system")


def start_command(config: Config) -> None:
    """Start the scheduled agent."""
    logger.info("Starting scheduled agent...")
    # TODO: Implement scheduler integration
    print("Scheduler not yet implemented in Phase 1")
    logger.info("Scheduled agent would run with configuration: %s", config)


def run_once_command(config: Config, dry_run: bool = False) -> None:
    """Run a single scan and email cycle."""
    logger.info("Running one-time execution...")
    if dry_run:
        logger.info("Dry run mode - no actual operations will be performed")
        print("Dry run: Would fetch, classify, and send emails")
    else:
        # TODO: Implement actual pipeline
        print("Pipeline not yet implemented in Phase 1")
    logger.info("One-time execution complete")


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

    # Start command
    start_parser = subparsers.add_parser("start", help="Start scheduled agent")
    start_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )

    # Run-once command
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

    # Version command
    subparsers.add_parser("version", help="Show version information")

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Handle no command
    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        # Setup logging early
        setup_logging()

        # Load configuration for commands that need it
        if parsed_args.command in ["start", "run-once"]:
            logger.info(f"Loading configuration from {parsed_args.config}")
            config = Config.from_yaml(parsed_args.config)
            config.load_env()

            if not config.validate():
                logger.error("Configuration validation failed")
                return 1
        else:
            config = None

        # Execute command
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

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        print("Please create config.yaml from config.yaml.example", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
