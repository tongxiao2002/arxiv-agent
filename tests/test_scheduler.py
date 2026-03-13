"""Tests for scheduler integration."""

from unittest.mock import Mock, patch

from arxiv_agent.scheduler import Scheduler


def test_scheduler_registers_daily_jobs():
    """Test scan and email jobs are scheduled with cron triggers."""
    mock_scheduler = Mock()
    mock_scheduler.running = True
    mock_scheduler.add_job.side_effect = [
        Mock(id="daily_scan"),
        Mock(id="daily_email"),
    ]

    with patch(
        "arxiv_agent.scheduler.BackgroundScheduler", return_value=mock_scheduler
    ):
        scheduler = Scheduler("Asia/Shanghai")
        scheduler.start()
        job_ids = scheduler.configure_daily_jobs(
            scan_job=lambda: None,
            email_job=lambda: None,
            scan_time="00:00",
            email_time="09:00",
        )

    assert job_ids == ["daily_scan", "daily_email"]
    assert mock_scheduler.add_job.call_count == 2

    scan_call = mock_scheduler.add_job.call_args_list[0]
    email_call = mock_scheduler.add_job.call_args_list[1]

    assert "hour='0'" in str(scan_call.kwargs["trigger"])
    assert "minute='0'" in str(scan_call.kwargs["trigger"])
    assert scan_call.kwargs["trigger"].timezone.key == "Asia/Shanghai"

    assert "hour='9'" in str(email_call.kwargs["trigger"])
    assert "minute='0'" in str(email_call.kwargs["trigger"])
    assert email_call.kwargs["trigger"].timezone.key == "Asia/Shanghai"


def test_scheduler_job_callbacks_are_wired():
    """Test scheduled callbacks are the functions supplied by the application."""
    mock_scheduler = Mock()
    mock_scheduler.running = True
    mock_scheduler.add_job.side_effect = [
        Mock(id="daily_scan"),
        Mock(id="daily_email"),
    ]
    observed = []

    def scan_job():
        observed.append("scan")

    def email_job():
        observed.append("email")

    with patch(
        "arxiv_agent.scheduler.BackgroundScheduler", return_value=mock_scheduler
    ):
        scheduler = Scheduler("Asia/Shanghai")
        scheduler.start()
        scheduler.configure_daily_jobs(
            scan_job=scan_job,
            email_job=email_job,
            scan_time="00:00",
            email_time="09:00",
        )

    mock_scheduler.add_job.call_args_list[0].args[0]()
    mock_scheduler.add_job.call_args_list[1].args[0]()
    assert observed == ["scan", "email"]


def test_scheduler_run_forever_stops_on_interrupt():
    """Test foreground scheduler loop shuts down cleanly on interrupt."""
    scheduler = Scheduler("Asia/Shanghai")
    scheduler.scheduler = Mock(running=True)

    with patch.object(scheduler, "stop") as mock_stop:
        with patch("arxiv_agent.scheduler.time.sleep", side_effect=KeyboardInterrupt):
            scheduler.run_forever()

    mock_stop.assert_called_once()
