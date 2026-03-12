"""Scheduler integration for Arxiv-Agent."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional

try:
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore
    MemoryJobStore = None  # type: ignore
    ThreadPoolExecutor = None  # type: ignore

from arxiv_agent.utils.timezone import get_timezone

logger = logging.getLogger(__name__)


class Scheduler:
    """Scheduler for managing timed jobs."""

    def __init__(self, timezone: str = "Asia/Shanghai"):
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "APScheduler is not installed. Install with 'pip install apscheduler'"
            )

        self.timezone_name = timezone
        self.timezone = get_timezone(timezone)
        self.scheduler: Optional[BackgroundScheduler] = None
        self.jobstores = {"default": MemoryJobStore()}
        self.executors = {"default": ThreadPoolExecutor(20)}
        self.job_defaults = {"coalesce": False, "max_instances": 1}

    def start(self) -> None:
        """Start the scheduler."""
        if self.scheduler and self.scheduler.running:
            logger.warning("Scheduler is already running")
            return

        logger.info("Starting scheduler with timezone: %s", self.timezone_name)
        self.scheduler = BackgroundScheduler(
            jobstores=self.jobstores,
            executors=self.executors,
            job_defaults=self.job_defaults,
            timezone=self.timezone,
        )
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            logger.info("Stopping scheduler...")
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")
        else:
            logger.warning("Scheduler is not running")

    def add_job(self, func: Callable[..., Any], trigger: Any, **trigger_args: Any) -> Optional[str]:
        """Add a job to the scheduler."""
        if not self.scheduler:
            logger.error("Scheduler not started. Call start() first.")
            return None

        job = self.scheduler.add_job(func, trigger=trigger, **trigger_args)
        logger.info("Added job %s: %s", job.id, func.__name__)
        return job.id

    def schedule_scan_job(self, func: Callable[..., Any], scan_time: str) -> Optional[str]:
        """Register the daily scan job."""
        return self._schedule_daily_job(
            func=func,
            job_id="daily_scan",
            name="Daily scan workflow",
            scheduled_time=scan_time,
        )

    def schedule_email_job(self, func: Callable[..., Any], email_time: str) -> Optional[str]:
        """Register the daily email job."""
        return self._schedule_daily_job(
            func=func,
            job_id="daily_email",
            name="Daily email workflow",
            scheduled_time=email_time,
        )

    def configure_daily_jobs(
        self,
        *,
        scan_job: Callable[..., Any],
        email_job: Callable[..., Any],
        scan_time: str,
        email_time: str,
    ) -> List[str]:
        """Register the scan and email jobs for application runtime."""
        job_ids: List[str] = []

        scan_job_id = self.schedule_scan_job(scan_job, scan_time)
        if scan_job_id:
            job_ids.append(scan_job_id)

        email_job_id = self.schedule_email_job(email_job, email_time)
        if email_job_id:
            job_ids.append(email_job_id)

        return job_ids

    def run_forever(self, poll_interval: float = 1.0) -> None:
        """Block the foreground process while the scheduler runs."""
        try:
            while True:
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
            self.stop()

    def list_jobs(self) -> list:
        """List all scheduled jobs."""
        if not self.scheduler:
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time,
                    "trigger": str(job.trigger),
                }
            )
        return jobs

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler."""
        if not self.scheduler:
            logger.error("Scheduler not started")
            return False

        try:
            self.scheduler.remove_job(job_id)
            logger.info("Removed job %s", job_id)
            return True
        except Exception as exc:
            logger.error("Failed to remove job %s: %s", job_id, exc)
            return False

    def _schedule_daily_job(
        self,
        *,
        func: Callable[..., Any],
        job_id: str,
        name: str,
        scheduled_time: str,
    ) -> Optional[str]:
        """Create a daily cron job using the configured timezone."""
        if not self.scheduler:
            logger.error("Scheduler not started. Call start() first.")
            return None

        hour, minute = _parse_time(scheduled_time)
        trigger = CronTrigger(hour=hour, minute=minute, timezone=self.timezone)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            name=name,
            replace_existing=True,
        )
        logger.info(
            "Scheduled %s at %02d:%02d %s",
            job_id,
            hour,
            minute,
            self.timezone_name,
        )
        return job.id

    def _job_executed(self, event: Any) -> None:
        """Handle job executed event."""
        logger.info(
            "Job %s executed successfully at %s",
            event.job_id,
            event.scheduled_run_time,
        )

    def _job_error(self, event: Any) -> None:
        """Handle job error event."""
        logger.error("Job %s failed with exception: %s", event.job_id, event.exception)
        logger.error("Traceback: %s", event.traceback)

    def __enter__(self) -> "Scheduler":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


def _parse_time(time_value: str) -> tuple[int, int]:
    """Parse a HH:MM time string into hour/minute values."""
    hour_text, minute_text = time_value.split(":", maxsplit=1)
    return int(hour_text), int(minute_text)
