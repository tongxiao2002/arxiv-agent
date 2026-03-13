"""Emailer agent for sending daily paper digests."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, Optional

from arxiv_agent.agents.base import BaseAgent
from arxiv_agent.config import AdvancedConfig, EmailConfig
from arxiv_agent.email import (
    SmtpEmailSender,
    render_digest_html,
    render_digest_subject,
    render_digest_text,
    render_no_papers_html,
    render_no_papers_subject,
    render_no_papers_text,
)
from arxiv_agent.sources.enhanced_paper import EnhancedPaper
from arxiv_agent.storage.json_storage import JsonStorage
from arxiv_agent.utils.timezone import get_current_date_in_timezone

logger = logging.getLogger(__name__)


class EmailerAgent(BaseAgent):
    """Agent that renders and sends the daily email digest."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        sender: Optional[SmtpEmailSender] = None,
        storage: Optional[JsonStorage] = None,
    ) -> None:
        super().__init__(name="emailer", config=config)
        self.timezone = config.get("agent", {}).get("timezone", "Asia/Shanghai")
        self.email_config = EmailConfig(**config.get("email", {}))
        self.advanced_config = AdvancedConfig(**config.get("advanced", {}))
        self.storage = storage or JsonStorage(
            data_dir=config.get("storage", {}).get("data_dir", "./papers")
        )
        self.sender = sender or SmtpEmailSender(
            self.email_config,
            smtp_password=os.getenv("SMTP_PASSWORD"),
            timeout=self.advanced_config.request_timeout,
            max_retries=self.advanced_config.max_retries,
            retry_backoff_factor=self.advanced_config.retry_backoff_factor,
        )

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Render and send a digest for the target date."""
        target_date = self._parse_target_date(*args, **kwargs)
        dry_run = kwargs.get("dry_run", False)
        self.logger.info(
            "Starting emailer agent run for %s (dry_run=%s)",
            target_date,
            dry_run,
        )

        if not self.storage.papers_exist_for_date(target_date):
            message = f"No stored papers found for {target_date}"
            self.logger.error(message)
            return {
                "success": False,
                "sent": False,
                "target_date": target_date.isoformat(),
                "message": message,
            }

        papers = self.storage.load_papers(target_date)
        if any(not isinstance(paper, EnhancedPaper) for paper in papers):
            message = f"Stored papers for {target_date} are not fully enhanced and cannot be emailed"
            self.logger.error(message)
            return {
                "success": False,
                "sent": False,
                "target_date": target_date.isoformat(),
                "message": message,
            }

        relevant_papers = [paper for paper in papers if paper.is_relevant is True]
        if relevant_papers:
            subject = render_digest_subject(
                target_date,
                self.email_config.subject_template,
                self.timezone,
            )
            text_body = render_digest_text(target_date, self.timezone, relevant_papers)
            html_body = render_digest_html(target_date, self.timezone, relevant_papers)
            email_type = "digest"
        else:
            subject = render_no_papers_subject(target_date, self.timezone)
            text_body = render_no_papers_text(target_date, self.timezone)
            html_body = render_no_papers_html(target_date, self.timezone)
            email_type = "no_papers"

        delivery_result = self.sender.send_email(
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            dry_run=dry_run,
        )
        return {
            "success": True,
            "sent": True,
            "dry_run": dry_run,
            "target_date": target_date.isoformat(),
            "papers_loaded": len(papers),
            "relevant_papers": len(relevant_papers),
            "email_type": email_type,
            "delivery": delivery_result,
        }

    def validate(self) -> bool:
        """Validate agent configuration."""
        if "email" not in self.config:
            self.logger.error("Missing configuration section: email")
            return False

        if "storage" not in self.config:
            self.logger.error("Missing configuration section: storage")
            return False

        if not self.email_config.to_emails:
            self.logger.error("Email configuration missing recipients")
            return False

        return True

    def _parse_target_date(self, *args: Any, **kwargs: Any) -> date:
        """Parse target date from args or kwargs."""
        if args and isinstance(args[0], str):
            return date.fromisoformat(args[0])
        if args and hasattr(args[0], "isoformat"):
            return args[0]

        target_date = kwargs.get("target_date")
        if isinstance(target_date, str):
            return date.fromisoformat(target_date)
        if target_date is not None and hasattr(target_date, "isoformat"):
            return target_date

        return get_current_date_in_timezone(self.timezone)
