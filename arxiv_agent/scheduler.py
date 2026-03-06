"""Scheduler integration for Arxiv-Agent."""

import logging
from datetime import datetime
from typing import Any, Callable, Optional

# Try to import APScheduler, but make it optional for now
try:
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.background import BackgroundScheduler

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None  # type: ignore
    MemoryJobStore = None  # type: ignore
    ThreadPoolExecutor = None  # type: ignore

logger = logging.getLogger(__name__)


class Scheduler:
    """Scheduler for managing timed jobs."""

    def __init__(self, timezone: str = "Asia/Shanghai"):
        """
        Initialize scheduler.

        Args:
            timezone: Timezone for scheduling (default: Asia/Shanghai)
        """
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "APScheduler is not installed. Install with 'pip install apscheduler'"
            )

        self.timezone = timezone
        self.scheduler: Optional[BackgroundScheduler] = None

        # Configure job stores and executors
        self.jobstores = {"default": MemoryJobStore()}
        self.executors = {"default": ThreadPoolExecutor(20)}
        self.job_defaults = {"coalesce": False, "max_instances": 1}

    def start(self) -> None:
        """Start the scheduler."""
        if self.scheduler and self.scheduler.running:
            logger.warning("Scheduler is already running")
            return

        logger.info(f"Starting scheduler with timezone: {self.timezone}")
        self.scheduler = BackgroundScheduler(
            jobstores=self.jobstores,
            executors=self.executors,
            job_defaults=self.job_defaults,
            timezone=self.timezone,
        )

        # Add event listeners for monitoring
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

    def add_job(
        self, func: Callable, trigger: str, **trigger_args: Any
    ) -> Optional[str]:
        """
        Add a job to the scheduler.

        Args:
            func: Function to execute
            trigger: Trigger type ('cron', 'interval', 'date')
            **trigger_args: Trigger-specific arguments

        Returns:
            Job ID or None if scheduler not started
        """
        if not self.scheduler:
            logger.error("Scheduler not started. Call start() first.")
            return None

        job = self.scheduler.add_job(func, trigger, **trigger_args)
        logger.info(f"Added job {job.id}: {func.__name__} with trigger {trigger}")
        return job.id  # type: ignore

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was removed, False otherwise
        """
        if not self.scheduler:
            logger.error("Scheduler not started")
            return False

        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def list_jobs(self) -> list:
        """
        List all scheduled jobs.

        Returns:
            List of job information dictionaries
        """
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

    def _job_executed(self, event: Any) -> None:
        """Handle job executed event."""
        logger.info(
            f"Job {event.job_id} executed successfully at {event.scheduled_run_time}"
        )

    def _job_error(self, event: Any) -> None:
        """Handle job error event."""
        logger.error(f"Job {event.job_id} failed with exception: {event.exception}")
        logger.error(f"Traceback: {event.traceback}")

    def __enter__(self) -> "Scheduler":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


# Example usage
if __name__ == "__main__":
    import time

    def sample_job() -> None:
        print(f"Sample job executed at {datetime.now()}")

    scheduler = Scheduler()
    scheduler.start()

    # Add a job that runs every 10 seconds
    scheduler.add_job(sample_job, "interval", seconds=10)

    try:
        print("Scheduler running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
