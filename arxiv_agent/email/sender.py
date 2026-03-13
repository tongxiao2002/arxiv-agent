"""SMTP email transport for Arxiv-Agent."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Dict, Optional

from arxiv_agent.config import EmailConfig
from arxiv_agent.utils.runtime import call_with_retry

logger = logging.getLogger(__name__)


class EmailDeliveryError(Exception):
    """Raised when SMTP delivery cannot be completed."""


class SmtpEmailSender:
    """SMTP sender that delivers multipart emails."""

    def __init__(
        self,
        config: EmailConfig,
        *,
        smtp_password: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self.config = config
        self.smtp_password = smtp_password
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor

    def send_email(
        self,
        *,
        subject: str,
        text_body: str,
        html_body: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Build and optionally deliver a multipart email."""
        message = self._build_message(
            subject=subject,
            text_body=text_body,
            html_body=html_body,
        )

        if dry_run:
            logger.info(
                "Dry-run email render complete for %s recipients with subject %s",
                len(self.config.to_emails),
                subject,
            )
            logger.debug("Dry-run text body:\n%s", text_body)
            logger.debug("Dry-run html body:\n%s", html_body)
            return {
                "success": True,
                "dry_run": True,
                "subject": subject,
                "recipient_count": len(self.config.to_emails),
            }

        self._deliver_message(message)
        logger.info(
            "Delivered email to %s recipients with subject %s",
            len(self.config.to_emails),
            subject,
        )
        return {
            "success": True,
            "dry_run": False,
            "subject": subject,
            "recipient_count": len(self.config.to_emails),
        }

    def _build_message(
        self,
        *,
        subject: str,
        text_body: str,
        html_body: str,
    ) -> EmailMessage:
        """Build a multipart email message."""
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.from_email
        message["To"] = ", ".join(self.config.to_emails)
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")
        return message

    def _deliver_message(self, message: EmailMessage) -> None:
        """Connect to the SMTP server and send a message."""

        def operation() -> None:
            client = self._create_client()
            try:
                if self.config.smtp_security == "starttls":
                    client.ehlo()
                    client.starttls(context=ssl.create_default_context())
                    client.ehlo()

                if self.config.smtp_username:
                    if not self.smtp_password:
                        raise EmailDeliveryError(
                            "SMTP_PASSWORD is required when smtp_username is configured"
                        )
                    client.login(self.config.smtp_username, self.smtp_password)

                client.send_message(message, to_addrs=self.config.to_emails)
            except EmailDeliveryError:
                raise
            except (smtplib.SMTPException, OSError) as exc:
                logger.warning("SMTP send failed: %s", exc)
                raise
            finally:
                try:
                    client.quit()
                except (smtplib.SMTPException, OSError):
                    logger.debug(
                        "SMTP client quit failed during cleanup",
                        exc_info=True,
                    )

        call_with_retry(
            operation,
            operation_name="_deliver_message",
            max_retries=self.max_retries,
            backoff_factor=self.retry_backoff_factor,
            retry_on=(smtplib.SMTPException, OSError),
            respect_retry_after=False,
        )

    def _create_client(self) -> smtplib.SMTP:
        """Create an SMTP client based on configured transport security."""
        if self.config.smtp_security == "ssl":
            return smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=self.timeout,
            )

        return smtplib.SMTP(
            self.config.smtp_host,
            self.config.smtp_port,
            timeout=self.timeout,
        )
